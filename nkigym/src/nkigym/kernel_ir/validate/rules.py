"""KernelIR legality checks on the ``(ir, ir)`` shape."""

from nkigym.kernel_ir.fusion_group import BufferPlacement
from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.placement_semantics import block_loop_open, buffer_dim_positions, effective_placement
from nkigym.kernel_ir.validate.emission import Placement, op_emission_placement
from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul


def validate(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Rejection checks disabled — accept every sampled IR.

    Temporary: the sampler now trusts codegen to raise on
    truly-illegal shapes rather than filtering via the multi-check
    rejection loop. Left as ``(ir, op_to_group, staged) -> True``
    so the sampler's ``if validate(...)`` guard always passes.
    """
    _ = ir, op_to_group, staged
    return True


def _build_tensor_to_groups(ir: KernelIR) -> dict[str, set[int]]:
    """Map tensor names → set of group indices whose ops touch them."""
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            for name in _op_tensors(ir, op):
                if ir.has_tensor(name):
                    result.setdefault(name, set()).add(gi)
    return result


def _op_tensors(ir: KernelIR, op: NKIOp) -> list[str]:
    """Return inputs + outputs tensor names for an op."""
    return [*ir.op_inputs.get(op, {}).values(), *ir.op_outputs.get(op, [])]


def _check_emission_feasibility(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Every op must have a legal emission slot."""
    memo: dict[int, Placement] = {}
    return all(
        _placement_ok(ir, op, gi, op_to_group, staged, memo) for gi, group in enumerate(ir.groups) for op in group.ops
    )


def _placement_ok(
    ir: KernelIR, op: NKIOp, gi: int, op_to_group: dict[int, int], staged: set[str], memo: dict[int, Placement]
) -> bool:
    """True iff ``op_emission_placement`` returns a legal slot."""
    ok = True
    try:
        op_emission_placement(ir, op, gi, op_to_group, staged, memo)
    except ValueError:
        ok = False
    return ok


def _check_blocking_innermost(ir: KernelIR) -> bool:
    """Blocking dims must be innermost (after non-blocking) per op."""
    return all(_op_blocking_innermost_ok(ir, gi, op) for gi, group in enumerate(ir.groups) for op in group.ops)


def _op_blocking_innermost_ok(ir: KernelIR, group_idx: int, op: NKIOp) -> bool:
    """Every effective-blocking dim inner to every non-blocking dim this op touches.

    Dims whose ltile iteration is absorbed by a gadget (e.g.
    ``matmul_block`` iterates K / M / N internally) are NOT
    treated as blocking for this check — the surrounding loop
    nest never opens their ltile loop, so their position in
    ``dim_order`` doesn't constrain PSUM accumulation order the
    way a renderer-emitted blocking loop would.
    """
    ir = ir
    blocking = ir.op_blocking_dims.get(op, set()) - _gadget_absorbed_blocking_dims(ir, group_idx)
    dim_order = ir.groups[group_idx].dim_order
    op_dims: set[str] = set()
    for name in _op_tensors(ir, op):
        if ir.has_tensor(name):
            op_dims.update(ir.tensor_info(name).dim_ids)
    op_dims &= set(dim_order)
    blocking_positions = [dim_order.index(d) for d in blocking if d in op_dims]
    non_blocking_positions = [dim_order.index(d) for d in op_dims if d not in blocking]
    return not (blocking_positions and non_blocking_positions and min(blocking_positions) < max(non_blocking_positions))


def _gadget_absorbed_blocking_dims(ir: KernelIR, group_idx: int) -> set[str]:
    """All dims absorbed by a gadget in ``group_idx`` (K, M, N for matmul_block).

    Mirrors ``codegen.matmul_block_detect.gadget_absorbed_dims``
    — named ``_blocking`` for historical reasons; includes
    non-blocking M/N too since all three are absorbed by the
    gadget's internal iteration. Kept local here to avoid a
    ``validate`` → ``codegen`` → ``kernel_ir`` import cycle.
    """
    result: set[str] = set()
    for op in ir.groups[group_idx].ops:
        if _is_matmul_block_candidate(ir, op, group_idx):
            for abstract in ("K", "M", "N"):
                dim_id = ir.op_axis_map.get(op, {}).get(abstract)
                if dim_id is not None:
                    result.add(dim_id)
    return result


def _is_matmul_block_candidate(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """Local copy of the codegen-level matmul_block candidate predicate.

    matmul_block fires iff:

    * ``op`` is an ``NKIMatmul`` with a blocked K dim
      (``num_blocks[K] > 1``).
    * Both K-input buffer placements leave the K block loop OPEN
      at alloc — i.e. the input is reloaded per outer-K block.
    """
    ok = type(op) is NKIMatmul
    if ok:
        axis_map = ir.op_axis_map.get(op, {})
        k_dim = axis_map.get("K")
        if k_dim is None:
            ok = False
        else:
            di = ir.dimensions[k_dim]
            tpb = ir.ltiles_per_block.get(k_dim, 1)
            num_blocks = di.dim_size // (tpb * di.logical_tile_size)
            ok = num_blocks > 1 and _k_inputs_reload_per_block(ir, op, group_idx, k_dim)
    return ok


def _k_inputs_reload_per_block(ir: KernelIR, op: NKIOp, group_idx: int, k_dim: str) -> bool:
    """True iff both K-input buffers have their K block loop OPEN at alloc.

    A buffer's K block loop is open at alloc iff its placement
    is ``MIDDLE`` (when K is the outer buffer dim) or ``INNER``
    (when K is the inner buffer dim).
    """
    group = ir.groups[group_idx]
    inputs = ir.op_inputs.get(op, {})
    result = True
    if k_dim not in group.dim_order:
        result = False
    else:
        k_pos = group.dim_order.index(k_dim)
        for role in ("stationary", "moving"):
            tensor = inputs.get(role)
            if tensor is None or not ir.has_tensor(tensor):
                result = False
                break
            placement = effective_placement(ir, group_idx, tensor)
            positions = buffer_dim_positions(ir.tensor_info(tensor).dim_ids, group.dim_order)
            if not block_loop_open(placement, k_pos, positions):
                result = False
                break
    return result


def _check_placement_feasibility(ir: KernelIR) -> bool:
    """Each tensor must have at least one feasible emission depth."""
    return all(_group_feasibility_ok(ir, gi) for gi in range(len(ir.groups)))


def _group_feasibility_ok(ir: KernelIR, group_idx: int) -> bool:
    """Check placement feasibility for every tensor touched by one group."""
    ir = ir
    group = ir.groups[group_idx]
    dim_order = group.dim_order
    pos = {d: i for i, d in enumerate(dim_order)}
    tensors: set[str] = set()
    for op in group.ops:
        tensors.update(name for name in _op_tensors(ir, op) if ir.has_tensor(name))
    return all(_tensor_feasibility_ok(ir, group_idx, name, pos, len(dim_order)) for name in tensors)


def _tensor_feasibility_ok(ir: KernelIR, group_idx: int, tensor_name: str, pos: dict[str, int], n: int) -> bool:
    """Under the per-buffer placement scheme every placement is feasible by construction.

    The old per-dim tier intersection could be empty; the new
    3-way placement always has a valid alloc slot.
    """
    _ = ir, group_idx, tensor_name, pos, n
    return True


def _check_cross_group_placements(ir: KernelIR, tensor_to_groups: dict[str, set[int]]) -> bool:
    """Cross-group tensors with an explicit placement entry must be ``OUTER``.

    ``buffer_placements`` only stores sampled choices for FG input
    buffers (Load destinations). When such an input is shared
    across 2+ groups, every touching group's entry must be
    ``OUTER`` so the buffer's data survives the group boundary.
    Non-input tensors derive their placement (``effective_placement``
    already forces ``OUTER`` for cross-FG cases).
    """
    ok = True
    for tname, groups in tensor_to_groups.items():
        if len(groups) < 2:
            continue
        for gi in groups:
            entry = ir.groups[gi].buffer_placements.get(("sbuf", tname))
            if entry is not None and entry is not BufferPlacement.OUTER:
                ok = False
                break
        if not ok:
            break
    return ok
