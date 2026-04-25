"""Lightweight validity checks over a ``KernelIR``'s knob assignment.

These are cheap, mechanical guards meant to filter random samples
before rendering — each function returns ``True`` iff the IR satisfies
the structural invariant. No rendering, no exceptions.
"""

from nkigym.kernel_ir.ir import BufferScope, KernelIR, Op
from nkigym.kernel_ir.types import DimRole


def is_valid(ir: KernelIR) -> bool:
    """Return ``True`` iff every validity check passes."""
    return _emission_depth_is_valid(ir) and _rotation_axes_in_scope(ir)


def _emission_depth_is_valid(ir: KernelIR) -> bool:
    """Each buffer's ``emission_depth`` must be ≤ its producer op's depth.

    Otherwise the consumer-side reference (``cur_<buf> = <buf>[...]``)
    emits before the allocation, triggering ``UnboundLocalError`` at
    sim time.
    """
    for name, depth in ir.emission_depth.items():
        producer = _producer_op(ir, name)
        if producer is None:
            continue
        if depth > _op_depth(ir, producer):
            return False
    return True


def _rotation_axes_in_scope(ir: KernelIR) -> bool:
    """Rotation on an axis requires ``i_block_<axis>`` to be open at the producer op.

    ``num_buffers.num_p_buffers`` / ``num_f_buffers`` emit
    ``[i_block_<axis> % N]`` at the producer's depth. If the axis's
    loop hasn't been entered yet (its ``dim_order`` position ≥ op
    depth), that reference triggers ``UnboundLocalError``.
    """
    for name, nb in ir.num_buffers.items():
        producer = _producer_op(ir, name)
        if producer is None:
            continue
        op_depth = _op_depth(ir, producer)
        buf = ir.physical_buffers[name]
        if nb.num_p_buffers is not None and not _axis_open(ir, buf.p_axis, op_depth):
            return False
        if nb.num_f_buffers is not None and buf.f_axis is not None and not _axis_open(ir, buf.f_axis, op_depth):
            return False
    return True


def _axis_open(ir: KernelIR, axis: str, depth: int) -> bool:
    """``i_block_<axis>`` is open iff the axis sits above ``depth`` in ``dim_order``."""
    if axis not in ir.dim_order:
        return False
    return ir.dim_order.index(axis) < depth


def _producer_op(ir: KernelIR, buf_name: str) -> Op | None:
    """The single op that writes ``buf_name`` as an output, or ``None``."""
    for op in ir.ops:
        if buf_name in op.outputs:
            return op
    return None


def _op_depth(ir: KernelIR, op: Op) -> int:
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
