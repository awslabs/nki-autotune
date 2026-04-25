"""Lightweight validity checks over a ``KernelIR``'s knob assignment.

These are cheap, mechanical guards meant to filter random samples
before rendering — no rendering, no exceptions. Two entry points:

* :func:`is_valid` — boolean fast-path for sampler reject loops.
* :func:`validity_report` — structured list of every failed check
  with a fix hint. Use this in tuning loops to decide which knob to
  flip instead of discarding a near-miss IR.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import BufferScope, KernelIR, Op
from nkigym.kernel_ir.types import DimRole


@dataclass(frozen=True)
class ValidityFailure:
    """One failed validity check, plus a concrete fix hint.

    Attributes:
        check: Identifier of the failed check — one of
            ``emission_depth_ceiling``, ``rotation_axis_closed``,
            ``transpose_scope_mismatch``, ``accumulator_coverage``.
        buffer: Buffer name the failure pertains to, or ``None`` when
            the failure is whole-IR.
        detail: Human-readable description of what went wrong
            (actual vs. expected numbers).
        fix_hint: One-line actionable suggestion naming the knob to
            adjust and the direction of adjustment.
    """

    check: str
    buffer: str | None
    detail: str
    fix_hint: str


def is_valid(ir: KernelIR) -> bool:
    """Return ``True`` iff every validity check passes."""
    return not validity_report(ir)


def validity_report(ir: KernelIR) -> list[ValidityFailure]:
    """Return every failed validity check on ``ir``.

    Empty list = valid. Each :class:`ValidityFailure` names the
    check, the buffer, the numerical detail, and a one-line fix
    hint. Useful when a tuning loop wants to nudge a single knob
    rather than discard the whole IR.
    """
    failures: list[ValidityFailure] = []
    failures.extend(_check_emission_depth(ir))
    failures.extend(_check_rotation_axes(ir))
    failures.extend(_check_transpose_scopes(ir))
    failures.extend(_check_accumulator_coverage(ir))
    return failures


def _accumulator_covers_closed_dims(ir: KernelIR) -> bool:
    """Matmul accumulator must span FULL extent on every output dim at
    position ≥ ``first_acc_position`` in ``dim_order``.

    Store fires once all ACC loops close — at that depth, any dim
    positioned at or below the first ACC dim is out of scope, so the
    store emits the full HBM span on those axes and expects the SBUF
    source to match that span.
    """
    return not _check_accumulator_coverage(ir)


def _check_accumulator_coverage(ir: KernelIR) -> list[ValidityFailure]:
    """List every output axis the matmul accumulator fails to cover."""
    acc_op = _find_matmul_op(ir)
    if acc_op is None or not acc_op.outputs:
        return []
    acc_buf_name = acc_op.outputs[0]
    if acc_buf_name not in ir.buffer_scopes:
        return []
    first_acc_pos = _store_depth(ir)
    buf = ir.physical_buffers[acc_buf_name]
    scope = ir.buffer_scopes[acc_buf_name]
    outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
    failures: list[ValidityFailure] = []
    for axis in (buf.p_axis, buf.f_axis):
        if axis is None or axis not in ir.dim_order:
            continue
        if ir.dim_order.index(axis) < first_acc_pos:
            continue
        num_tiles = _num_tiles_for_scope(ir, axis, scope, is_outer=(axis == outer_axis))
        full = ir.dimensions[axis].dim_size // ir.dimensions[axis].physical_tile_size
        if num_tiles < full:
            failures.append(
                ValidityFailure(
                    check="accumulator_coverage",
                    buffer=acc_buf_name,
                    detail=(
                        f"axis {axis!r} covered by {num_tiles} tiles but needs {full} "
                        f"(position {ir.dim_order.index(axis)} ≥ first_acc_position "
                        f"{first_acc_pos} → its loop is already closed at store)"
                    ),
                    fix_hint=(
                        f"widen {acc_buf_name} scope to OUTER (or MIDDLE if "
                        f"{axis!r} is outermost), or move {axis!r} before any "
                        f"ACCUMULATION dim in dim_order so its loop stays open at store"
                    ),
                )
            )
    return failures


def _find_matmul_op(ir: KernelIR) -> Op | None:
    """First ``NKIMatmul`` op, or ``None``."""
    for op in ir.ops:
        if op.kind == "NKIMatmul":
            return op
    return None


def _transpose_scopes_match(ir: KernelIR) -> bool:
    """Transpose src and dst must share the same ``buffer_scope``.

    ``transpose_block`` / ``dma_transpose_block`` require
    ``src.num_p_tiles == dst.num_f_tiles`` and vice-versa. Since the
    two buffers' axes are swapped, that equality holds iff both sides
    use the same scope-sizing rule on each dim — equivalent to
    matching scope labels.
    """
    return not _check_transpose_scopes(ir)


def _check_transpose_scopes(ir: KernelIR) -> list[ValidityFailure]:
    """List every transpose whose src/dst scopes disagree."""
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in ("NKITranspose", "NKIDMATranspose"):
            continue
        src = op.inputs.get("data")
        dst = op.outputs[0] if op.outputs else None
        if src is None or dst is None:
            continue
        if src not in ir.buffer_scopes or dst not in ir.buffer_scopes:
            continue
        src_scope = ir.buffer_scopes[src]
        dst_scope = ir.buffer_scopes[dst]
        if src_scope is not dst_scope:
            failures.append(
                ValidityFailure(
                    check="transpose_scope_mismatch",
                    buffer=dst,
                    detail=(
                        f"transpose {src!r}→{dst!r} has src scope {src_scope.name} "
                        f"but dst scope {dst_scope.name}; gadget requires matching "
                        f"num_p_tiles/num_f_tiles across the swapped axes"
                    ),
                    fix_hint=(
                        f"set buffer_scopes[{dst!r}] = buffer_scopes[{src!r}] "
                        f"(or vice-versa) — transpose pairs share one scope"
                    ),
                )
            )
    return failures


def _emission_depth_is_valid(ir: KernelIR) -> bool:
    """Each buffer's ``emission_depth`` must be ≤ its first use depth.

    Otherwise the consumer-side reference (``cur_<buf> = <buf>[...]``)
    emits before the allocation, triggering ``UnboundLocalError`` at
    sim time.
    """
    return not _check_emission_depth(ir)


def _check_emission_depth(ir: KernelIR) -> list[ValidityFailure]:
    """List every buffer whose ``emission_depth`` exceeds its first use."""
    failures: list[ValidityFailure] = []
    for name, depth in ir.emission_depth.items():
        fu = first_use_depth(ir, name)
        if depth > fu:
            failures.append(
                ValidityFailure(
                    check="emission_depth_ceiling",
                    buffer=name,
                    detail=(
                        f"emission_depth={depth} exceeds first_use_depth={fu} "
                        f"(allocation would land below the first reference)"
                    ),
                    fix_hint=(
                        f"lower emission_depth[{name!r}] to ≤ {fu}, or widen the "
                        f"buffer's scope / change dim_order to push first_use_depth deeper"
                    ),
                )
            )
    return failures


def first_use_depth(ir: KernelIR, buf_name: str) -> int:
    """Shallowest depth at which ``buf_name`` may be referenced by render.

    * Matmul accumulator: the memset prologue fires at
      ``1 + min(dim_order.index(d))`` over the accumulator's non-ACC
      axes when any rotation is chosen on a non-ACC axis, otherwise
      at ``_store_depth(ir)``. We return the shallower bound —
      ``emission_depth`` and ``num_buffers`` must be valid under
      every rotation choice the sampler may still make.
    * Every other buffer: its producer op's depth.
    """
    acc_op = _find_matmul_op(ir)
    if acc_op is not None and buf_name in acc_op.outputs:
        return _accumulator_first_use_depth(ir, buf_name)
    producer = producer_op(ir, buf_name)
    if producer is None:
        return len(ir.dim_order)
    return op_depth(ir, producer)


def _accumulator_first_use_depth(ir: KernelIR, buf_name: str) -> int:
    """Accumulator prologue depth — matches ``_accumulator_prologue_depth`` in render.

    Rule (same as loads): ``1 + max(dim_order.index(d))`` over the
    accumulator's block-indexed axes — the axes whose ``i_block_<d>``
    appears in the slot expression. OUTER-scoped accumulators have no
    block-indexed axes → depth 0.
    """
    block_axes = _block_indexed_axes(ir, buf_name)
    if not block_axes:
        return 0
    positions = [ir.dim_order.index(d) for d in block_axes if d in ir.dim_order]
    if not positions:
        return 0
    return 1 + max(positions)


def _rotation_axes_in_scope(ir: KernelIR) -> bool:
    """Rotation on an axis requires ``i_block_<axis>`` to be open at the buffer's first use.

    ``num_buffers.num_p_buffers`` / ``num_f_buffers`` emit
    ``[i_block_<axis> % N]`` at the ``cur_<buf>`` binding. If the
    axis's loop hasn't been entered yet, that reference triggers
    ``UnboundLocalError``.
    """
    return not _check_rotation_axes(ir)


def _check_rotation_axes(ir: KernelIR) -> list[ValidityFailure]:
    """List every rotation whose index axis is closed at first use."""
    failures: list[ValidityFailure] = []
    for name, nb in ir.num_buffers.items():
        depth = first_use_depth(ir, name)
        buf = ir.physical_buffers[name]
        if nb.num_p_buffers is not None and not axis_open(ir, buf.p_axis, depth):
            failures.append(
                ValidityFailure(
                    check="rotation_axis_closed",
                    buffer=name,
                    detail=(
                        f"num_p_buffers={nb.num_p_buffers} on p_axis={buf.p_axis!r} "
                        f"but that axis is not open at first_use_depth={depth} "
                        f"(position {_axis_position(ir, buf.p_axis)} in dim_order)"
                    ),
                    fix_hint=(
                        f"set num_buffers[{name!r}].num_p_buffers=None, or move "
                        f"{buf.p_axis!r} above position {depth} in dim_order so its "
                        f"loop is open when {name!r} is first referenced"
                    ),
                )
            )
        if nb.num_f_buffers is not None and buf.f_axis is not None and not axis_open(ir, buf.f_axis, depth):
            failures.append(
                ValidityFailure(
                    check="rotation_axis_closed",
                    buffer=name,
                    detail=(
                        f"num_f_buffers={nb.num_f_buffers} on f_axis={buf.f_axis!r} "
                        f"but that axis is not open at first_use_depth={depth} "
                        f"(position {_axis_position(ir, buf.f_axis)} in dim_order)"
                    ),
                    fix_hint=(
                        f"set num_buffers[{name!r}].num_f_buffers=None, or move "
                        f"{buf.f_axis!r} above position {depth} in dim_order so its "
                        f"loop is open when {name!r} is first referenced"
                    ),
                )
            )
    return failures


def _axis_position(ir: KernelIR, axis: str | None) -> int | str:
    """Position of ``axis`` in ``dim_order``, or ``'absent'`` when missing."""
    if axis is None or axis not in ir.dim_order:
        return "absent"
    return ir.dim_order.index(axis)


def axis_open(ir: KernelIR, axis: str, depth: int) -> bool:
    """``i_block_<axis>`` is open iff the axis sits above ``depth`` in ``dim_order``."""
    if axis not in ir.dim_order:
        return False
    return ir.dim_order.index(axis) < depth


def producer_op(ir: KernelIR, buf_name: str) -> Op | None:
    """The single op that writes ``buf_name`` as an output, or ``None``."""
    for op in ir.ops:
        if buf_name in op.outputs:
            return op
    return None


def op_depth(ir: KernelIR, op: Op) -> int:
    """Loop-nest depth at which ``op`` is emitted in ``render_ir``."""
    if op.kind in _LOAD_KINDS or (op.kind == "NKIDMATranspose" and _is_hbm_sourced(ir, op)):
        return _load_depth(ir, op)
    if op.kind == "NKIStore":
        return _store_depth(ir)
    return len(ir.dim_order)


_LOAD_KINDS = frozenset({"NKILoad"})


def _is_hbm_sourced(ir: KernelIR, op: Op) -> bool:
    """``NKIDMATranspose`` whose ``data`` is a kernel parameter (HBM)."""
    return op.inputs.get("data") in ir.param_names


def _load_depth(ir: KernelIR, op: Op) -> int:
    """Load fires inside every block-loop whose index appears in its destination slice."""
    dst = op.outputs[0]
    block_axes = _block_indexed_axes(ir, dst)
    if not block_axes:
        return 0
    positions = [ir.dim_order.index(a) for a in block_axes if a in ir.dim_order]
    if not positions:
        return 0
    return max(positions) + 1


def _store_depth(ir: KernelIR) -> int:
    """Store fires at the first ACCUMULATION position in ``dim_order``."""
    for i, d in enumerate(ir.dim_order):
        if ir.dimensions[d].role is DimRole.ACCUMULATION:
            return i
    return len(ir.dim_order)


def _block_indexed_axes(ir: KernelIR, buf_name: str) -> list[str]:
    """Axes whose per-block width is < full dim — i.e. the buffer is sliced along them."""
    buf = ir.physical_buffers[buf_name]
    scope = ir.buffer_scopes.get(buf_name, BufferScope.INNER)
    outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
    axes: list[str] = []
    for axis in (buf.p_axis, buf.f_axis):
        if axis is None:
            continue
        num_tiles = _num_tiles_for_scope(ir, axis, scope, is_outer=(axis == outer_axis))
        if num_tiles * ir.dimensions[axis].physical_tile_size < ir.dimensions[axis].dim_size:
            axes.append(axis)
    return axes


def _num_tiles_for_scope(ir: KernelIR, axis: str, scope: BufferScope, is_outer: bool) -> int:
    """Tile count along ``axis`` under ``scope`` — matches ``_scope_extents`` in render."""
    info = ir.dimensions[axis]
    ltiles = ir.ltiles_per_block[axis]
    full = info.dim_size // info.physical_tile_size
    if scope is BufferScope.INNER:
        return ltiles
    if scope is BufferScope.OUTER:
        return full
    return ltiles if is_outer else full


def _outer_axis_in_order(ir: KernelIR, p_axis: str, f_axis: str | None) -> str:
    """Whichever of ``(p_axis, f_axis)`` appears first in ``dim_order``."""
    if f_axis is None:
        return p_axis
    return p_axis if ir.dim_order.index(p_axis) < ir.dim_order.index(f_axis) else f_axis
