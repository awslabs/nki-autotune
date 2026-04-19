"""KernelIR legality checks.

A single ``validate(ir)`` function answers: is this ``KernelIR``
a coherent kernel? Each rule is a small, self-contained check.
Rejection sampling in ``ir.sample_valid_ir`` uses this to filter
independently-sampled IR candidates.
"""

from typing import Any

from nkigym.kernel_ir.dim_analysis import op_blocking_dims


def validate(ir: Any, tensor_to_groups: dict[str, set[int]]) -> bool:
    """Return True iff every legality rule passes for *ir*.

    ``tensor_placements`` is an SBUF-only concept (PSUM has no
    meaningful tier — it's always a single narrow-scoped tile
    allocated just before use), so every placement rule here
    implicitly subjects the SBUF buffer.
    """
    return (
        _check_cross_group_placements(ir, tensor_to_groups)
        and _check_blocking_innermost(ir)
        and _check_placement_feasibility(ir)
    )


def _check_blocking_innermost(ir: Any) -> bool:
    """Blocking dims must come after all non-blocking dims in each group's dim_order.

    A PSUM-producing op accumulates over its blocking axes; the
    output spans the non-blocking (output) dims. With PSUM sized
    for one output tile at a time, the kernel must iterate all
    blocking dims to completion *within* a single output tile
    before advancing to the next — i.e. blocking dims must be
    inner. Any dim_order that nests a non-blocking dim inside a
    blocking dim causes the non-blocking iterations to share a
    single PSUM, summing across output positions.
    """
    return all(
        _op_blocking_innermost_ok(ir, gi, op_idx)
        for gi, group in enumerate(ir.fusion_groups)
        for op_idx in group.op_indices
    )


def _op_blocking_innermost_ok(ir: Any, group_idx: int, op_idx: int) -> bool:
    """Check that every blocking dim is inner to every non-blocking dim for one op."""
    da = ir.dim_analysis
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, da.per_op_axis_maps[op_idx])
    dim_order = ir.fusion_groups[group_idx].dim_order
    blocking_positions = [dim_order.index(d) for d in blocking if d in dim_order]
    non_blocking_positions = [idx for idx, d in enumerate(dim_order) if d not in blocking]
    return not (blocking_positions and non_blocking_positions and min(blocking_positions) < max(non_blocking_positions))


def _check_placement_feasibility(ir: Any) -> bool:
    """Each tensor must have at least one feasible emission depth.

    For every dim ``d`` the tensor carries, the tier placed on
    ``(tensor, d)`` constrains the depths at which a load/stage/
    store gadget for that tensor can fire:

    * ``per_tile``  → depth in ``[N + i + 1, 2N]`` (inside the
      block and ltile loops at position ``i``)
    * ``per_block`` → depth in ``[i + 1, N + i]`` (inside the
      block loop, outside the ltile loop)
    * ``full``      → depth in ``[0, i]`` (outside the block loop)

    The tensor is feasible iff the intersection of its per-dim
    ranges is non-empty. Strictly-worse placements where an
    inner dim has a broader tier than an outer dim are rejected
    because their intervals don't intersect.
    """
    return all(_group_feasibility_ok(ir, gi) for gi in range(len(ir.fusion_groups)))


def _group_feasibility_ok(ir: Any, group_idx: int) -> bool:
    """Check placement feasibility for every tensor touched by one group."""
    da = ir.dim_analysis
    graph = ir.op_graph
    dim_order = ir.fusion_groups[group_idx].dim_order
    pos = {d: i for i, d in enumerate(dim_order)}
    tensors: set[str] = set()
    for op_idx in ir.fusion_groups[group_idx].op_indices:
        tensors.update(name for name in graph.op_tensor_names(op_idx) if name in da.tensors)
    return all(_tensor_feasibility_ok(ir, group_idx, name, pos, len(dim_order)) for name in tensors)


def _tensor_feasibility_ok(ir: Any, group_idx: int, tensor_name: str, pos: dict[str, int], n: int) -> bool:
    """Intersection of per-dim depth ranges must be non-empty for this group's tiers."""
    lo = 0
    hi = 2 * n
    placements = ir.fusion_groups[group_idx].tensor_placements
    for d in ir.dim_analysis.tensors[tensor_name].dim_ids:
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
        rng = (n + pos + 1, 2 * n)
    elif tier == "per_block":
        rng = (pos + 1, n + pos)
    elif tier == "full":
        rng = (0, pos)
    else:
        raise ValueError(f"Unknown tier {tier!r}")
    return rng


def _check_cross_group_placements(ir: Any, tensor_to_groups: dict[str, set[int]]) -> bool:
    """Cross-group tensors must be ``full`` in every touching group on shared-scope dims.

    A tensor ``t`` that is touched by two or more fusion groups
    survives between their sequential loop nests. For every dim
    ``d`` the tensor carries, each touching group that has ``d``
    in its ``dim_order`` must pin ``tensor_placements[(g, t, d)]``
    to ``"full"``. Any lesser tier lets one group's outer
    iterations overwrite slots another group hasn't read yet, or
    leaves a kernel input under-loaded when a later consumer
    group reads it.
    """
    da = ir.dim_analysis
    return all(
        ir.fusion_groups[gi].tensor_placements.get(("sbuf", tensor_name, d)) == "full"
        for tensor_name, groups in tensor_to_groups.items()
        if len(groups) >= 2
        for gi in groups
        for d in set(ir.fusion_groups[gi].dim_order) & set(da.tensors[tensor_name].dim_ids)
    )
