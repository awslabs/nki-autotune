"""Codegen helpers for the per-op ``SkipPredicate`` annotations.

``propagate_compute_skip`` (run at build time) tags each affected
``NKIOp`` with a ``SkipPredicate``. At codegen time the main op
renderer (``codegen.nki_ops``) takes the lines each op produced
at each depth and, if the op is annotated, wraps them in a
three-state classifier:

* ``skip_all``: `pass`.
* ``compute_only``: unmodified op emission.
* ``mask_and_compute``: op emission plus, when
  ``inject_mask == True``, an in-place ``nisa.affine_select`` on
  the op's output tile — restoring the ``-inf`` fill that the
  deleted standalone ``NKIAffineSelect`` used to do.

The classifier uses a static Python ``if`` on the loop-index
variables. ``@nki.jit`` unrolls outer integer-range loops so the
branches resolve at trace time — no hardware conditional.
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir.compute_skip_spec import SkipPredicate
from nkigym.kernel_ir.ir import KernelIR
from nkigym.ops.base import NKIOp


def snapshot_before_lengths(before_plan: DepthPlan, gi: int) -> dict[int, int]:
    """Capture per-depth line counts for ``before_plan[gi]`` so we can diff after emission."""
    return {depth: len(lines) for depth, lines in before_plan.get(gi, {}).items()}


def record_op_delta(
    before_plan: DepthPlan, gi: int, baseline: dict[int, int], op: NKIOp, per_op_lines: dict[int, dict[int, list[str]]]
) -> None:
    """Record lines appended to ``before_plan[gi]`` since ``baseline`` under ``id(op)``."""
    current = before_plan.get(gi, {})
    bucket = per_op_lines.setdefault(id(op), {})
    for depth, lines in current.items():
        start = baseline.get(depth, 0)
        if len(lines) > start:
            bucket.setdefault(depth, []).extend(lines[start:])


def wrap_annotated_ops(ir: KernelIR, before_plan: DepthPlan, per_op_lines: dict[int, dict[int, list[str]]]) -> None:
    """Rebuild the innermost-body line list with each annotated op's contribution in a 3-branch ``if``.

    One pass per group: walks the inner-body lines in original
    emission order and, for every span of lines contributed by
    an annotated op, replaces the span with the classifier
    block. Unannotated ops' lines pass through unchanged. Lines
    at shallower depths (scratch decls, reduction init memsets)
    are never gated — they are already outside the classifier's
    per-tile scope.
    """
    context = ir.context
    for gi, group in enumerate(ir.graph.groups):
        n = len(group.dim_order)
        inner_depth = 2 * n
        original = before_plan.get(gi, {}).get(inner_depth, [])
        if not original:
            continue
        rebuilt = _rebuild_inner_body(ir, gi, original, group.ops, context.op_skip_spec, per_op_lines, inner_depth)
        before_plan.setdefault(gi, {})[inner_depth] = rebuilt


def _rebuild_inner_body(
    ir: KernelIR,
    gi: int,
    original: list[str],
    group_ops: list[NKIOp],
    op_skip_spec: dict[NKIOp, SkipPredicate],
    per_op_lines: dict[int, dict[int, list[str]]],
    inner_depth: int,
) -> list[str]:
    """Build a new inner-body line list with annotated ops' contributions wrapped in classifiers.

    An op is wrapped iff it has a ``SkipPredicate`` AND at least
    one of its tensors touches BOTH the predicate's partition
    and free dims. Ops that don't touch the free dim (e.g. a
    Q-side transpose whose output is only ``(partition, K)``)
    can't be meaningfully gated by a per-``(P, F)`` classifier —
    the classifier expression references ``i_block_free``, which
    is out of scope at their emission slot.
    """
    op_contrib: list[tuple[NKIOp, list[str]]] = []
    for op in group_ops:
        lines = per_op_lines.get(id(op), {}).get(inner_depth, [])
        if lines:
            op_contrib.append((op, lines))
    rebuilt: list[str] = []
    cursor = 0
    for op, lines in op_contrib:
        span_len = len(lines)
        if original[cursor : cursor + span_len] != lines:
            raise RuntimeError(
                f"compute_skip.wrap_annotated_ops: op-span mismatch at group {gi} op {type(op).__name__}. "
                "The per-op line delta is out of sync with the before_plan stream."
            )
        predicate = op_skip_spec.get(op)
        if predicate is not None and _op_touches_both_dims(ir, op, predicate):
            mask_extra = _mask_injection_lines(ir, op, predicate) if predicate.inject_mask else []
            wrapped = _emit_branches(predicates=_build_predicates(ir, predicate), body=lines, mask_extra=mask_extra)
            rebuilt.extend(wrapped)
        else:
            rebuilt.extend(lines)
        cursor += span_len
    rebuilt.extend(original[cursor:])
    return rebuilt


def _op_touches_both_dims(ir: KernelIR, op: NKIOp, predicate: SkipPredicate) -> bool:
    """True iff the op touches both ``partition_dim_id`` and ``free_dim_id`` in any of its tensors."""
    context = ir.context
    names = list(context.op_inputs.get(op, {}).values()) + list(context.op_outputs.get(op, []))
    touches: set[str] = set()
    for name in names:
        tinfo = context.logical_tensors.get(name)
        if tinfo is not None:
            touches.update(tinfo.dim_ids)
    return predicate.partition_dim_id in touches and predicate.free_dim_id in touches


def _emit_branches(predicates: dict[str, str], body: list[str], mask_extra: list[str]) -> list[str]:
    """Emit the three-branch ``if`` block for one op.

    ``skip_all`` ⇒ ``pass``.
    ``compute_only`` ⇒ ``body``.
    ``mask_and_compute`` ⇒ ``body + mask_extra`` (mask_extra is the
    in-place affine_select injection when ``inject_mask`` is set,
    otherwise empty so the two branches are identical).
    """
    result: list[str] = []
    result.append(f"if {predicates['skip_all']}:")
    result.append("    pass")
    result.append(f"elif {predicates['compute_only']}:")
    result.extend(f"    {line}" for line in body)
    result.append("else:")
    result.extend(f"    {line}" for line in body)
    result.extend(f"    {line}" for line in mask_extra)
    return result


def _mask_injection_lines(ir: KernelIR, op: NKIOp, predicate: SkipPredicate) -> list[str]:
    """Emit ``nisa.affine_select`` against ``op``'s output tile to inject the ``-inf`` mask.

    Only fires on ``mask_and_compute`` tiles of the op marked
    ``inject_mask=True``. The ISA call reads and writes the same
    tile (in-place), matching what the standalone ``NKIAffineSelect``
    would have produced on the same data.
    """
    context = ir.context
    outputs = context.op_outputs.get(op, [])
    lines: list[str] = []
    for oname in outputs:
        tinfo = context.logical_tensors.get(oname)
        if tinfo is None:
            continue
        buf = sbuf_buffer(ir, oname)
        group_idx = _group_of(ir, op)
        placements = ir.graph.groups[group_idx].tensor_placements
        dim_order = ir.graph.groups[group_idx].dim_order
        p_access = _body_axis_access(oname, tinfo.dim_ids[0], placements, dim_order)
        if len(tinfo.dim_ids) == 2:
            f_access = _body_axis_access(oname, tinfo.dim_ids[1], placements, dim_order)
        else:
            f_access = AxisAccess(block="0", ltile="0")
        tile_expr = buf.get_tile(p_access, f_access)
        pattern_expr = f"[[{predicate.free_step}, {predicate.free_tile_size}]]"
        offset_expr = _tile_offset_expr(ir, predicate)
        cmp_nl = f"nl.{predicate.cmp_op}"
        lines.append(
            f"nisa.affine_select({tile_expr}, {pattern_expr}, {predicate.channel_multiplier}, "
            f"{tile_expr}, {predicate.on_false_value}, offset={offset_expr}, cmp_op={cmp_nl})"
        )
    return lines


def _tile_offset_expr(ir: KernelIR, predicate: SkipPredicate) -> str:
    """Build the per-tile ``offset`` expression mirroring what ``NKIAffineSelect.format_isa_call`` did.

    The user's global ``offset`` plus ``channel_multiplier *
    p_tile_start`` plus ``free_step * f_tile_start`` so the affine
    expression reproduces its global value at the tile's
    boundary.
    """
    p_start = _tile_start_with_strides(ir, predicate.partition_dim_id)
    f_start = _tile_start_with_strides(ir, predicate.free_dim_id)
    terms: list[str] = [str(predicate.offset)]
    if predicate.channel_multiplier != 0:
        terms.append(f"({predicate.channel_multiplier}) * ({p_start})")
    if predicate.free_step != 0:
        terms.append(f"({predicate.free_step}) * ({f_start})")
    return " + ".join(terms)


def _group_of(ir: KernelIR, op: NKIOp) -> int:
    """Return the group index containing ``op``."""
    for gi, group in enumerate(ir.graph.groups):
        if op in group.ops:
            return gi
    raise ValueError(f"op {op!r} not found in any fusion group")


def _body_axis_access(
    tensor_name: str, dim_id: str, placements: dict[tuple[str, str, str], str], dim_order: list[str]
) -> AxisAccess:
    """Build ``AxisAccess`` for one dim at the group's innermost body."""
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    block = "0"
    ltile = "0"
    if dim_id in dim_order:
        if tier == "full":
            block = f"i_block_{dim_id}"
        if tier in ("per_block", "full"):
            ltile = f"i_ltile_{dim_id}"
    return AxisAccess(block=block, ltile=ltile)


def _build_predicates(ir: KernelIR, spec: SkipPredicate) -> dict[str, str]:
    """Construct Python predicate expressions for ``skip_all`` / ``compute_only``.

    Linear-affine form ``offset + p * channel_multiplier + f *
    free_step`` has extrema at tile-box corners; the corner used
    depends on the coefficient signs and whether we are
    minimizing or maximizing over the tile.
    """
    p_start = _tile_start_with_strides(ir, spec.partition_dim_id)
    f_start = _tile_start_with_strides(ir, spec.free_dim_id)
    max_aff = _extremum_expr(spec, p_start, f_start, spec.partition_tile_size, spec.free_tile_size, maximize=True)
    min_aff = _extremum_expr(spec, p_start, f_start, spec.partition_tile_size, spec.free_tile_size, maximize=False)
    return _predicates_for_cmp(spec.cmp_op, max_aff, min_aff)


def _tile_start_with_strides(ir: KernelIR, dim_id: str) -> str:
    """Element start of one block/ltile slice for ``dim_id`` at the innermost body."""
    di = ir.context.dimensions[dim_id]
    tpb = ir.context.ltiles_per_block.get(dim_id, 1)
    logical = di.logical_tile_size
    block_stride = logical * tpb
    terms: list[str] = []
    terms.append(f"i_block_{dim_id} * {block_stride}" if block_stride > 1 else f"i_block_{dim_id}")
    terms.append(f"i_ltile_{dim_id} * {logical}" if logical > 1 else f"i_ltile_{dim_id}")
    return " + ".join(terms)


def _predicates_for_cmp(cmp_op: str, max_aff: str, min_aff: str) -> dict[str, str]:
    """Return ``{skip_all, compute_only}`` predicate strings for a given cmp_op."""
    if cmp_op == "greater_equal":
        result = {"skip_all": f"({max_aff}) < 0", "compute_only": f"({min_aff}) >= 0"}
    elif cmp_op == "equal":
        result = {"skip_all": f"(({max_aff}) < 0) or (({min_aff}) > 0)", "compute_only": "False"}
    else:
        raise NotImplementedError(f"compute-skip predicate for cmp_op={cmp_op!r} not supported")
    return result


def _extremum_expr(spec: SkipPredicate, p_start: str, f_start: str, p_size: int, f_size: int, maximize: bool) -> str:
    """Pick tile-box corner at which the linear affine form reaches its extremum."""
    p_term = _axis_corner(spec.channel_multiplier, p_start, p_size, maximize)
    f_term = _axis_corner(spec.free_step, f_start, f_size, maximize)
    terms: list[str] = [str(spec.offset)]
    if spec.channel_multiplier != 0:
        terms.append(f"({spec.channel_multiplier}) * ({p_term})")
    if spec.free_step != 0:
        terms.append(f"({spec.free_step}) * ({f_term})")
    return " + ".join(terms)


def _axis_corner(coef: int, start_expr: str, size: int, maximize: bool) -> str:
    """Box corner for one axis based on coefficient sign and direction."""
    max_corner = maximize
    if coef < 0:
        max_corner = not maximize
    return f"({start_expr}) + {size - 1}" if max_corner and size > 1 else f"({start_expr})"
