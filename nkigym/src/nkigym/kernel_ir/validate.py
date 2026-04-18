"""KernelIR legality checks.

A single ``validate(ir)`` function answers: is this ``KernelIR``
a coherent kernel? Each rule is a small, self-contained check.
Rejection sampling in ``ir.sample_valid_ir`` uses this to filter
independently-sampled IR candidates.
"""

from typing import Any

from nkigym.kernel_ir.dim_analysis import op_blocking_dims


def validate(ir: Any, op_to_group: dict[int, int]) -> bool:
    """Return True iff every legality rule passes for *ir*."""
    return (
        _check_cross_group_placements(ir, op_to_group)
        and _check_blocking_innermost(ir)
        and _check_placement_feasibility(ir)
        and _check_psum_output_reachable(ir)
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
        _op_blocking_innermost_ok(ir, gi, op_idx) for gi, group in enumerate(ir.fusion_groups) for op_idx in group
    )


def _op_blocking_innermost_ok(ir: Any, group_idx: int, op_idx: int) -> bool:
    """Check that every blocking dim is inner to every non-blocking dim for one op."""
    da = ir.dim_analysis
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, da.per_op_axis_maps[op_idx])
    dim_order = ir.group_dim_orders[group_idx]
    blocking_positions = [dim_order.index(d) for d in blocking if d in dim_order]
    non_blocking_positions = [idx for idx, d in enumerate(dim_order) if d not in blocking]
    return not (blocking_positions and non_blocking_positions and min(blocking_positions) < max(non_blocking_positions))


def _check_psum_output_reachable(ir: Any) -> bool:
    """Every PSUM-produced tensor must be stageable at the depth where its producer finishes accumulating.

    A PSUM op accumulates over its blocking dims; the output is
    only defined after the outermost blocking dim's loop closes
    — i.e. at depth ``producer_finished = min(pos(b) for b in
    blocking)`` in the producer group's dim_order. The tier
    intervals on the output's dims must contain that depth, else
    the stage/store fires inside the accumulation loop and reads
    a partial sum.
    """
    return all(_group_psum_outputs_ok(ir, gi) for gi in range(len(ir.fusion_groups)))


def _group_psum_outputs_ok(ir: Any, group_idx: int) -> bool:
    """Check every PSUM-producing op in one group has reachable outputs at its finished depth."""
    dim_order = ir.group_dim_orders[group_idx]
    pos = {d: i for i, d in enumerate(dim_order)}
    n = len(dim_order)
    psum_ops = [i for i in ir.fusion_groups[group_idx] if ir.op_graph.op_classes[i].ISA_LOC == "psum"]
    return all(_op_psum_outputs_ok(ir, op_idx, pos, n) for op_idx in psum_ops)


def _op_psum_outputs_ok(ir: Any, op_idx: int, pos: dict[str, int], n: int) -> bool:
    """Check one PSUM op's outputs are reachable at the producer-finished depth."""
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, ir.dim_analysis.per_op_axis_maps[op_idx]) & pos.keys()
    finished = min(pos[d] for d in blocking) if blocking else 2 * n
    outputs = ir.op_graph.op_tensors[op_idx][1]
    return all(_depth_in_tensor_interval(ir, oname, pos, n, finished) for oname in outputs)


def _depth_in_tensor_interval(ir: Any, tensor_name: str, pos: dict[str, int], n: int, depth: int) -> bool:
    """True iff ``depth`` lies in every per-dim tier interval for *tensor_name*."""
    return all(
        _dim_interval_contains(ir, tensor_name, d, pos[d], n, depth)
        for d in ir.dim_analysis.tensors[tensor_name].dim_ids
        if d in pos
    )


def _dim_interval_contains(ir: Any, tensor_name: str, dim_id: str, position: int, n: int, depth: int) -> bool:
    """True iff the tier interval for one dim contains ``depth``."""
    tier = ir.tensor_placements[(tensor_name, dim_id)]
    d_lo, d_hi = tier_depth_range(tier, position, n)
    return d_lo <= depth <= d_hi


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
    dim_order = ir.group_dim_orders[group_idx]
    pos = {d: i for i, d in enumerate(dim_order)}
    tensors: set[str] = set()
    for op_idx in ir.fusion_groups[group_idx]:
        tensors.update(name for name in graph.op_tensor_names(op_idx) if name in da.tensors)
    return all(_tensor_feasibility_ok(ir, name, pos, len(dim_order)) for name in tensors)


def _tensor_feasibility_ok(ir: Any, tensor_name: str, pos: dict[str, int], n: int) -> bool:
    """Intersection of per-dim depth ranges must be non-empty."""
    lo = 0
    hi = 2 * n
    for d in ir.dim_analysis.tensors[tensor_name].dim_ids:
        if d not in pos:
            continue
        tier = ir.tensor_placements[(tensor_name, d)]
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


def _check_cross_group_placements(ir: Any, op_to_group: dict[int, int]) -> bool:
    """Cross-group tensors must be ``full`` on shared-scope dims.

    A tensor ``t`` produced by op ``p`` and consumed by op ``c``
    in a different fusion group survives between two group loop
    nests that run sequentially. For every dim ``d`` that (a)
    ``t`` carries, (b) appears in the producer group's
    ``dim_order``, and (c) appears in the consumer group's
    ``dim_order``, the SBUF buffer must hold every slot the
    consumer will read — i.e. ``tensor_placements[(t, d)] ==
    "full"``. Any lesser tier lets the producer's outer
    iterations overwrite slots the consumer hasn't read yet.
    """
    graph = ir.op_graph
    return all(_consumer_placements_ok(ir, op_to_group, consumer_idx) for consumer_idx in range(len(graph.op_tensors)))


def _consumer_placements_ok(ir: Any, op_to_group: dict[int, int], consumer_idx: int) -> bool:
    """Check every cross-group edge feeding one consumer op."""
    inputs, _ = ir.op_graph.op_tensors[consumer_idx]
    g_c = op_to_group[consumer_idx]
    consumer_dims = set(ir.group_dim_orders[g_c])
    return all(_edge_ok(ir, op_to_group, name, g_c, consumer_dims) for name in inputs.values())


def _edge_ok(ir: Any, op_to_group: dict[int, int], tensor_name: str, g_c: int, consumer_dims: set[str]) -> bool:
    """Check a single producer→consumer edge for a cross-group placement rule."""
    da = ir.dim_analysis
    producer = ir.op_graph.producer_op(tensor_name) if tensor_name in da.tensors else None
    shared: set[str] = set()
    if producer is not None and op_to_group[producer] != g_c:
        producer_dims = set(ir.group_dim_orders[op_to_group[producer]])
        shared = producer_dims & consumer_dims & set(da.tensors[tensor_name].dim_ids)
    return all(ir.tensor_placements.get((tensor_name, d)) == "full" for d in shared)
