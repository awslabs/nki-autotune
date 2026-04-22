"""KernelIR legality checks on the ``(context, graph)`` shape."""

from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.validate.emission import Placement, block_depth, body_depth, ltile_depth, op_emission_placement
from nkigym.ops.base import NKIOp


def validate(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Return True iff every legality rule passes for ``ir``."""
    tensor_to_groups = _build_tensor_to_groups(ir)
    return (
        _check_cross_group_placements(ir, tensor_to_groups)
        and _check_blocking_innermost(ir)
        and _check_placement_feasibility(ir)
        and _check_emission_feasibility(ir, op_to_group, staged)
    )


def _build_tensor_to_groups(ir: KernelIR) -> dict[str, set[int]]:
    """Map tensor names → set of group indices whose ops touch them."""
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(ir.graph.groups):
        for op in group.ops:
            for name in _op_tensors(ir, op):
                if name in ir.context.logical_tensors:
                    result.setdefault(name, set()).add(gi)
    return result


def _op_tensors(ir: KernelIR, op: NKIOp) -> list[str]:
    """Return inputs + outputs tensor names for an op."""
    return [*ir.context.op_inputs.get(op, {}).values(), *ir.context.op_outputs.get(op, [])]


def _check_emission_feasibility(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Every op must have a legal emission slot."""
    memo: dict[int, Placement] = {}
    return all(
        _placement_ok(ir, op, gi, op_to_group, staged, memo)
        for gi, group in enumerate(ir.graph.groups)
        for op in group.ops
    )


def _placement_ok(
    ir: KernelIR, op: NKIOp, gi: int, op_to_group: dict[int, int], staged: set[str], memo: dict[int, Placement]
) -> bool:
    """True iff ``op_emission_placement`` returns a legal slot."""
    ok = True
    try:
        op_emission_placement(ir.context, ir.graph, op, gi, op_to_group, staged, memo)
    except ValueError:
        ok = False
    return ok


def _check_blocking_innermost(ir: KernelIR) -> bool:
    """Blocking dims must be innermost (after non-blocking) per op."""
    return all(_op_blocking_innermost_ok(ir, gi, op) for gi, group in enumerate(ir.graph.groups) for op in group.ops)


def _op_blocking_innermost_ok(ir: KernelIR, group_idx: int, op: NKIOp) -> bool:
    """Every blocking dim inner to every non-blocking dim this op touches."""
    context = ir.context
    blocking = context.op_blocking_dims.get(op, set())
    dim_order = ir.graph.groups[group_idx].dim_order
    op_dims: set[str] = set()
    for name in _op_tensors(ir, op):
        tinfo = context.logical_tensors.get(name)
        if tinfo is not None:
            op_dims.update(tinfo.dim_ids)
    op_dims &= set(dim_order)
    blocking_positions = [dim_order.index(d) for d in blocking if d in op_dims]
    non_blocking_positions = [dim_order.index(d) for d in op_dims if d not in blocking]
    return not (blocking_positions and non_blocking_positions and min(blocking_positions) < max(non_blocking_positions))


def _check_placement_feasibility(ir: KernelIR) -> bool:
    """Each tensor must have at least one feasible emission depth."""
    return all(_group_feasibility_ok(ir, gi) for gi in range(len(ir.graph.groups)))


def _group_feasibility_ok(ir: KernelIR, group_idx: int) -> bool:
    """Check placement feasibility for every tensor touched by one group."""
    context = ir.context
    group = ir.graph.groups[group_idx]
    dim_order = group.dim_order
    pos = {d: i for i, d in enumerate(dim_order)}
    tensors: set[str] = set()
    for op in group.ops:
        tensors.update(name for name in _op_tensors(ir, op) if name in context.logical_tensors)
    return all(_tensor_feasibility_ok(ir, group_idx, name, pos, len(dim_order)) for name in tensors)


def _tensor_feasibility_ok(ir: KernelIR, group_idx: int, tensor_name: str, pos: dict[str, int], n: int) -> bool:
    """Intersection of per-dim depth ranges must be non-empty."""
    lo = 0
    hi = body_depth(n)
    placements = ir.graph.groups[group_idx].tensor_placements
    for d in ir.context.logical_tensors[tensor_name].dim_ids:
        if d not in pos:
            continue
        key = ("sbuf", tensor_name, d)
        if key not in placements:
            continue
        tier = placements[key]
        d_lo, d_hi = tier_depth_range(tier, pos[d], n)
        lo = max(lo, d_lo)
        hi = min(hi, d_hi)
    return lo <= hi


def tier_depth_range(tier: str, pos: int, n: int) -> tuple[int, int]:
    """Allowed emission-depth range for a dim at ``pos`` under a given tier."""
    if tier == "per_tile":
        rng = (ltile_depth(pos) + 1, body_depth(n))
    elif tier == "per_block":
        rng = (block_depth(pos) + 1, ltile_depth(pos))
    elif tier == "full":
        rng = (0, block_depth(pos))
    else:
        raise ValueError(f"Unknown tier {tier!r}")
    return rng


def _check_cross_group_placements(ir: KernelIR, tensor_to_groups: dict[str, set[int]]) -> bool:
    """Cross-group tensors must be ``full`` in every touching group on shared-scope dims."""
    context = ir.context
    return all(
        ir.graph.groups[gi].tensor_placements.get(("sbuf", tname, d)) == "full"
        for tname, groups in tensor_to_groups.items()
        if len(groups) >= 2
        for gi in groups
        for d in set(ir.graph.groups[gi].dim_order) & set(context.logical_tensors[tname].dim_ids)
    )
