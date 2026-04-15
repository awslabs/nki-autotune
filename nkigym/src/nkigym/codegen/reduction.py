"""Reduction loop generation: per-group sibling blocks inside DP loops."""

import heapq

from nkigym.codegen.kernel_ir import KernelIR
from nkigym.dim_analysis.dim_analysis import DimAnalysis
from nkigym.graph_analysis.op_graph import OpGraph


def render_reduction_loops(ir: KernelIR, dp_indent: int) -> str:
    """Emit the reduction loop region inside the innermost DP loop.

    Each fusion group emits its own reduction loop nest as a
    sibling block, ordered by topological sort of the group-level
    DAG derived from ``op_graph``.

    Args:
        ir: Complete kernel IR.
        dp_indent: Indentation level of the innermost DP loop body.

    Returns:
        Indented NKI source lines for the reduction region,
        with a ``...`` placeholder per group body.
    """
    da = ir.dim_analysis
    graph = ir.op_graph

    group_order = _toposort_groups(ir)
    dp_dims = {d for d, di in da.dims.items() if di.is_data_parallel}

    lines: list[str] = []
    for group_idx in group_order:
        group = ir.fusion_groups[group_idx]
        red_dims = _group_reduction_dims(group, graph, da, dp_dims)
        group_lines = _render_group(ir, group_idx, group, red_dims, dp_indent)
        lines.extend(group_lines)

    return "\n".join(lines)


def _toposort_groups(ir: KernelIR) -> list[int]:
    """Topologically sort fusion groups by the group-level DAG.

    For each op-level edge, lift to group level. Ties broken
    by minimum op_idx in each group.
    """
    num_groups = len(ir.fusion_groups)
    op_to_group: dict[int, int] = {}
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group:
            op_to_group[op_idx] = gi

    group_edges: set[tuple[int, int]] = set()
    in_degree: dict[int, int] = {gi: 0 for gi in range(num_groups)}

    for producer, consumer, _tensor, _role in ir.op_graph.edges:
        gp = op_to_group[producer]
        gc = op_to_group[consumer]
        if gp != gc and (gp, gc) not in group_edges:
            group_edges.add((gp, gc))
            in_degree[gc] += 1

    adjacency: dict[int, list[int]] = {gi: [] for gi in range(num_groups)}
    for gp, gc in group_edges:
        adjacency[gp].append(gc)

    heap: list[tuple[int, int]] = []
    for gi in range(num_groups):
        if in_degree[gi] == 0:
            heapq.heappush(heap, (min(ir.fusion_groups[gi]), gi))
    order: list[int] = []

    while heap:
        _priority, gi = heapq.heappop(heap)
        order.append(gi)
        for neighbor in adjacency[gi]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, (min(ir.fusion_groups[neighbor]), neighbor))

    if len(order) != num_groups:
        raise ValueError("Cycle detected in group-level DAG")

    return order


def _group_reduction_dims(group: list[int], graph: OpGraph, da: DimAnalysis, dp_dims: set[str]) -> list[str]:
    """Collect reduction dims for a fusion group.

    Union all dim_ids from all tensors touched by ops in the
    group, subtract DP dims, return sorted.
    """
    all_dims: set[str] = set()
    for op_idx in group:
        inputs, outputs = graph.op_tensors[op_idx]
        for tensor_name in list(inputs.values()) + outputs:
            if tensor_name in da.tensors:
                all_dims.update(da.tensors[tensor_name].dim_ids)
    return sorted(all_dims - dp_dims)


def _get_tpb(ir: KernelIR, group: list[int], dim_id: str) -> int:
    """Get tiles_per_block for a dimension using the first op in group."""
    tpb = 1
    for op_idx in group:
        key = (op_idx, dim_id)
        if key in ir.tiles_per_block:
            tpb = ir.tiles_per_block[key]
            break
    return tpb


def _render_group(ir: KernelIR, group_idx: int, group: list[int], red_dims: list[str], base_indent: int) -> list[str]:
    """Render one fusion group's reduction loop nest.

    Args:
        ir: Complete kernel IR.
        group_idx: Index into fusion_groups.
        group: List of op indices in this group.
        red_dims: Sorted reduction dim IDs for this group.
        base_indent: Indentation level to start at.

    Returns:
        List of source lines.
    """
    da = ir.dim_analysis
    lines: list[str] = []
    pad = "    " * base_indent

    op_names = ", ".join(ir.op_graph.nodes[i] for i in group)
    red_str = ", ".join(red_dims) if red_dims else "(none)"
    lines.append(f"{pad}# Group {group_idx}: {op_names} [reduction: {red_str}]")

    loop_order = ir.loop_order[group_idx]
    ordered_red = [d for d in loop_order if d in set(red_dims)]

    indent = base_indent

    for dim_id in ordered_red:
        di = da.dims[dim_id]
        tpb = _get_tpb(ir, group, dim_id)
        num_blocks = di.dim_size // (tpb * di.tile_size)
        p = "    " * indent
        lines.append(f"{p}for i_block_{dim_id} in range({num_blocks}):")
        indent += 1

    for dim_id in ordered_red:
        tpb = _get_tpb(ir, group, dim_id)
        p = "    " * indent
        lines.append(f"{p}for i_tile_{dim_id} in range({tpb}):")
        indent += 1

    for dim_id in ordered_red:
        di = da.dims[dim_id]
        num_ig = di.tile_size // di.min_tile_size
        p = "    " * indent
        lines.append(f"{p}for i_ig_{dim_id} in range({num_ig}):")
        indent += 1

    p = "    " * indent
    lines.append(f"{p}...")
    lines.append("")

    return lines
