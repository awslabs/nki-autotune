"""Render path for ``ComputeSkipSpec``-annotated ``FusionGroup`` instances.

When a fusion group is tagged with a ``ComputeSkipSpec`` by the
``ComputeSkipPattern`` rewrite, the ops whose emission lives at
the group's innermost body (depth ``2 * n``) are wrapped with a
three-state classifier that routes each ``(partition_tile,
free_tile)`` pair:

* ``skip_all`` — no op runs.
* ``compute_only`` — every gated op runs EXCEPT the affine-
  select (whose mask is an identity on fully-compute tiles).
* ``mask_and_compute`` — every gated op runs, including the
  affine-select.

The classifier is a static Python ``if`` on the loop-index
variables. The NKI ``@nki.jit`` tracer unrolls the outer Python
loops with integer ranges, so each branch survives only for the
tiles where its predicate is true — no hardware conditional
ISA calls are emitted.

Ops in the group that emit at shallower depths (because they
don't touch every dim in the group's ``dim_order``) are NOT
wrapped — they already run at a broader scope where per-tile
skipping does not apply.
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir.graph.compute_skip_spec import ComputeSkipSpec
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


def wrap_skip_groups(
    ir: KernelIR, before_plan: DepthPlan, after_plan: DepthPlan, per_op_lines: dict[int, dict[int, list[str]]]
) -> None:
    """Rewrite innermost-body lines of each skip-annotated group to use a three-state ``if``.

    ``per_op_lines[id(op)][depth]`` carries the lines produced by
    each absorbed op at each depth the op wrote to. The pass
    splits the innermost depth (``2 * n``) of each skip-annotated
    group's ``before_plan`` into three classifier branches.

    Any lines at shallower depths — produced by ops that emit
    outside the innermost body — are left untouched.
    """
    _ = after_plan
    for gi, group in enumerate(ir.graph.groups):
        if group.skip_spec is None:
            continue
        n = len(group.dim_order)
        depth = 2 * n
        _wrap_one_group(ir, gi, depth, group.skip_spec, before_plan, per_op_lines)


def _wrap_one_group(
    ir: KernelIR,
    gi: int,
    depth: int,
    spec: ComputeSkipSpec,
    before_plan: DepthPlan,
    per_op_lines: dict[int, dict[int, list[str]]],
) -> None:
    """Rewrite depth-``2*n`` lines of one skip-annotated group."""
    existing = list(before_plan.get(gi, {}).get(depth, []))
    gated_lines_by_op = _innermost_lines_by_op(ir, gi, depth, per_op_lines)
    tracked: set[str] = set()
    for lines in gated_lines_by_op.values():
        tracked.update(lines)
    passthrough = [line for line in existing if line not in tracked]
    affine_id = id(spec.affine_select_op)
    compute_only_replacement = {affine_id: _affine_select_to_copy(gated_lines_by_op.get(affine_id, []))}
    compute_only = _concat_in_order(ir, gi, gated_lines_by_op, substitute=compute_only_replacement)
    mask_and_compute = _concat_in_order(ir, gi, gated_lines_by_op, substitute={})
    predicates = _build_predicates(ir, spec)
    skip_all_lines = _emit_skip_all_memsets(ir, gi, spec)
    new_lines = _emit_branches(passthrough, predicates, skip_all_lines, compute_only, mask_and_compute)
    before_plan.setdefault(gi, {})[depth] = new_lines


def _emit_skip_all_memsets(ir: KernelIR, gi: int, spec: ComputeSkipSpec) -> list[str]:
    """Emit ``nisa.memset(buffer_tile, on_false_value)`` for each per-``(P, F)`` boundary tensor.

    Only tensors carrying BOTH the skip's partition and free dims
    are memset per-iteration — those are the tiles the skip
    classifier gates. Reduced outputs (e.g. ``neg_max`` which
    drops the free dim) are NOT per-tile tensors and MUST NOT be
    memset per-(P, F) iteration, since doing so would overwrite
    accumulated running-reduction state.

    The memset preserves downstream correctness when the
    per-tile tensor is consumed outside the skip group:
    external readers see the mask-out sentinel (e.g. ``-inf``
    for the softmax path) on skipped tiles instead of stale
    memory.
    """
    context = ir.context
    group = ir.graph.groups[gi]
    placements = group.tensor_placements
    dim_order = group.dim_order
    lines: list[str] = []
    for tname in spec.boundary_tensors:
        tinfo = context.logical_tensors.get(tname)
        if tinfo is None:
            continue
        if spec.partition_dim_id not in tinfo.dim_ids or spec.free_dim_id not in tinfo.dim_ids:
            continue
        buf = sbuf_buffer(ir, tname)
        p_access = _body_axis_access(tname, tinfo.dim_ids[0], placements, dim_order)
        if len(tinfo.dim_ids) == 2:
            f_access = _body_axis_access(tname, tinfo.dim_ids[1], placements, dim_order)
        else:
            f_access = AxisAccess(block="0", ltile="0")
        tile_expr = buf.get_tile(p_access, f_access)
        lines.append(f"nisa.memset({tile_expr}, {spec.on_false_value})")
    return lines


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


def _affine_select_to_copy(affine_lines: list[str]) -> list[str]:
    """Rewrite ``nisa.affine_select(dst, pattern, cm, src, on_false, ...)`` lines to ``nisa.tensor_copy(dst, src)``.

    On ``compute_only`` tiles the mask is an identity so the
    affine-select is equivalent to a pure copy of ``src`` into
    ``dst``. Dropping the mask comparison saves work while still
    materializing the consumer's expected buffer.
    """
    result: list[str] = []
    for line in affine_lines:
        if "nisa.affine_select(" in line:
            result.append(_rewrite_affine_select_line(line))
        else:
            result.append(line)
    return result


def _rewrite_affine_select_line(line: str) -> str:
    """Turn one ``nisa.affine_select`` source line into a ``nisa.tensor_copy`` on the same dst/src."""
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    prefix = "nisa.affine_select("
    assert stripped.startswith(prefix), f"unexpected affine_select shape: {line!r}"
    inside = stripped[len(prefix) :]
    dst_expr, remainder = _split_top_level_arg(inside)
    src_expr = _extract_on_true_tile(remainder)
    return f"{indent}nisa.tensor_copy({dst_expr}, {src_expr})"


def _split_top_level_arg(text: str) -> tuple[str, str]:
    """Return ``(first_arg, rest)`` for a call's argument list string (balanced brackets)."""
    depth = 0
    split_idx = -1
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            split_idx = i
            break
    if split_idx >= 0:
        first, rest = text[:split_idx].strip(), text[split_idx + 1 :].strip()
    else:
        first, rest = text.strip(), ""
    return first, rest


def _extract_on_true_tile(remainder: str) -> str:
    """Extract the ``on_true_tile`` positional argument — 3rd remaining positional in affine_select."""
    current = remainder
    for _ in range(2):
        _arg, current = _split_top_level_arg(current)
    arg, _rest = _split_top_level_arg(current)
    return arg


def _innermost_lines_by_op(
    ir: KernelIR, gi: int, depth: int, per_op_lines: dict[int, dict[int, list[str]]]
) -> dict[int, list[str]]:
    """Return ``{id(op): [lines at innermost depth]}`` for ops in group ``gi``."""
    result: dict[int, list[str]] = {}
    for op in ir.graph.groups[gi].ops:
        lines = per_op_lines.get(id(op), {}).get(depth, [])
        if lines:
            result[id(op)] = list(lines)
    return result


def _concat_in_order(
    ir: KernelIR, gi: int, lines_by_op: dict[int, list[str]], substitute: dict[int, list[str]]
) -> list[str]:
    """Concatenate per-op lines in the group's op order, replacing any op's lines found in ``substitute``."""
    result: list[str] = []
    for op in ir.graph.groups[gi].ops:
        op_id = id(op)
        if op_id in substitute:
            result.extend(substitute[op_id])
        else:
            result.extend(lines_by_op.get(op_id, []))
    return result


def _emit_branches(
    passthrough: list[str],
    predicates: dict[str, str],
    skip_all: list[str],
    compute_only: list[str],
    mask_and_compute: list[str],
) -> list[str]:
    """Emit the three-branch ``if`` block."""
    result: list[str] = list(passthrough)
    result.append(f"if {predicates['skip_all']}:")
    if skip_all:
        result.extend(f"    {line}" for line in skip_all)
    else:
        result.append("    pass")
    result.append(f"elif {predicates['compute_only']}:")
    if compute_only:
        result.extend(f"    {line}" for line in compute_only)
    else:
        result.append("    pass")
    result.append("else:")
    if mask_and_compute:
        result.extend(f"    {line}" for line in mask_and_compute)
    else:
        result.append("    pass")
    return result


def _build_predicates(ir: KernelIR, spec: ComputeSkipSpec) -> dict[str, str]:
    """Construct Python predicate expressions for ``skip_all`` / ``compute_only``.

    The affine expression evaluated per element is

        affine(p, f) = offset + p * channel_multiplier + f * free_step

    where ``p`` and ``f`` are the GLOBAL element indices along
    the partition and free axes. Because the form is linear, its
    extrema over the tile box ``[p_start, p_start + P) × [f_start,
    f_start + F)`` fall at the corners.

    For ``cmp_op == "greater_equal"`` the passing condition is
    ``affine >= 0`` so ``skip_all`` is ``max(affine) < 0`` and
    ``compute_only`` is ``min(affine) >= 0``. The corner chosen
    depends on the signs of ``channel_multiplier`` and
    ``free_step``.
    """
    p_start = _tile_start_expr(ir, spec.partition_dim_id)
    f_start = _tile_start_expr(ir, spec.free_dim_id)
    p_size = spec.partition_tile_size
    f_size = spec.free_tile_size
    max_aff = _extremum_expr(spec, p_start, f_start, p_size, f_size, maximize=True)
    min_aff = _extremum_expr(spec, p_start, f_start, p_size, f_size, maximize=False)
    return _predicates_for_cmp(spec.cmp_op, max_aff, min_aff)


def _predicates_for_cmp(cmp_op: str, max_aff: str, min_aff: str) -> dict[str, str]:
    """Return the ``{skip_all, compute_only}`` predicate strings for ``cmp_op``."""
    if cmp_op == "greater_equal":
        result = {"skip_all": f"({max_aff}) < 0", "compute_only": f"({min_aff}) >= 0"}
    elif cmp_op == "equal":
        result = {"skip_all": f"(({max_aff}) < 0) or (({min_aff}) > 0)", "compute_only": "False"}
    else:
        raise NotImplementedError(f"compute-skip predicate for cmp_op={cmp_op!r} not supported")
    return result


def _tile_start_expr(ir: KernelIR, dim_id: str) -> str:
    """Element start of one block/ltile slice for ``dim_id`` at the innermost body."""
    di = ir.context.dimensions[dim_id]
    tpb = ir.context.ltiles_per_block.get(dim_id, 1)
    logical = di.logical_tile_size
    block_stride = logical * tpb
    terms: list[str] = []
    terms.append(f"i_block_{dim_id} * {block_stride}" if block_stride > 1 else f"i_block_{dim_id}")
    terms.append(f"i_ltile_{dim_id} * {logical}" if logical > 1 else f"i_ltile_{dim_id}")
    return " + ".join(terms)


def _extremum_expr(spec: ComputeSkipSpec, p_start: str, f_start: str, p_size: int, f_size: int, maximize: bool) -> str:
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
