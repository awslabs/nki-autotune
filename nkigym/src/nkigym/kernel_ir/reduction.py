"""Reduction loop generation: per-group sibling blocks inside DP loops.

Current scope: skeleton only — block and logical-tile loops per
fusion group, with a ``pass`` body. DMA loads, ISA calls, memset,
and PSUM staging are not yet emitted.
"""

import heapq

from nkigym.kernel_ir.dim_analysis import DimAnalysis
from nkigym.kernel_ir.ir import KernelIR, get_tpb
from nkigym.kernel_ir.op_graph import OpGraph


def render_reduction_loops(ir: KernelIR, body_indent: int) -> str:
    """Emit the reduction region inside the innermost DP loop.

    Each fusion group emits its own reduction loop nest as a
    sibling block, ordered by topological sort of the group-level
    DAG derived from ``op_graph``. Each group's nest has two
    phases: all block loops outermost, then all logical-tile
    loops. Physical-tile iteration is per op (hidden in gadgets)
    and never appears at the kernel level.

    Args:
        ir: Complete kernel IR.
        body_indent: Indentation level of the innermost DP loop
            body (i.e. where each group's comment header and
            first ``for`` line are written).

    Returns:
        Indented NKI source lines for the reduction region, with
        a ``pass`` placeholder body per group.
    """
    da = ir.dim_analysis
    graph = ir.op_graph

    dp_dims = {d for d, di in da.dims.items() if di.is_data_parallel}
    num_dp = sum(1 for entry in ir.loop_order if isinstance(entry, str))

    group_order = _toposort_groups(ir)
    lines: list[str] = []
    for group_idx in group_order:
        group = ir.fusion_groups[group_idx]
        red_dims = _group_reduction_dims(group, graph, da, dp_dims)
        lines.extend(_render_group(ir, group_idx, group, red_dims, body_indent, num_dp))

    return "\n".join(lines)


def _toposort_groups(ir: KernelIR) -> list[int]:
    """Topologically sort fusion groups by the group-level DAG.

    For each op-level edge, lift to group level. Ties broken by
    minimum ``op_idx`` in each group.
    """
    num_groups = len(ir.fusion_groups)
    op_to_group: dict[int, int] = {}
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group:
            op_to_group[op_idx] = gi

    group_edges: set[tuple[int, int]] = set()
    in_degree: dict[int, int] = dict.fromkeys(range(num_groups), 0)

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


def _render_group(
    ir: KernelIR, group_idx: int, group: list[int], red_dims: list[str], base_indent: int, num_dp: int
) -> list[str]:
    """Render one fusion group's reduction loop nest (skeleton only).

    Emits a comment header, then block and logical-tile loops
    over the group's reduction dims, then a ``pass`` body.
    Groups with no reduction dims emit just the comment + ``pass``.

    Args:
        ir: Complete kernel IR.
        group_idx: Index into ``ir.fusion_groups``.
        group: Op indices in this group.
        red_dims: This group's reduction dim IDs (unordered).
        base_indent: Indentation level for the group's top line.
        num_dp: Count of top-level DP dim entries in ``ir.loop_order``.

    Returns:
        Source lines for the group.
    """
    da = ir.dim_analysis
    lines: list[str] = []
    pad = "    " * base_indent

    op_names = ", ".join(ir.op_graph.op_classes[i].NAME for i in group)
    red_str = ", ".join(red_dims) if red_dims else "(none)"
    lines.append(f"{pad}# Group {group_idx}: {op_names} [reduction: {red_str}]")

    group_entry = ir.loop_order[num_dp + group_idx]
    if not isinstance(group_entry, list):
        raise ValueError(f"loop_order entry for group {group_idx} must be a list, got {group_entry!r}")
    red_set = set(red_dims)
    ordered_red = [d for d in group_entry if d in red_set]

    indent = base_indent

    for dim_id in ordered_red:
        di = da.dims[dim_id]
        tpb = get_tpb(ir, dim_id)
        num_blocks = di.dim_size // (tpb * di.logical_tile_size)
        p = "    " * indent
        lines.append(f"{p}for i_block_{dim_id} in range({num_blocks}):")
        indent += 1

    for dim_id in ordered_red:
        tpb = get_tpb(ir, dim_id)
        p = "    " * indent
        lines.append(f"{p}for i_ltile_{dim_id} in range({tpb}):")
        indent += 1

    body_pad = "    " * indent
    lines.append(f"{body_pad}pass")
    lines.append("")

    return lines
