"""KernelIR: structured representation for lowering to NKI source."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.dim_analysis.dim_analysis import DimAnalysis, analyze_dims
from nkigym.graph_analysis.op_graph import OpGraph, build_op_graph


@dataclass
class KernelIR:
    """Complete representation for lowering to NKI source.

    Composes two independent analysis results with rendering
    parameters that control loop structure, buffer sizes, and
    DMA placement.

    Attributes:
        dim_analysis: Dimension IDs, tile sizes, tensor metadata.
        op_graph: Producer-consumer DAG.
        fusion_groups: Which ops share a loop nest.
        tiles_per_block: Per (op_idx, dim_id) tiling factor.
        buffer_degrees: Per (group_idx, tensor_name, dim_id) degree.
        loop_order: Per-group dimension ordering within each phase.
        load_placements: Per (tensor_name, dim_id) DMA tier.
    """

    dim_analysis: DimAnalysis
    op_graph: OpGraph
    fusion_groups: list[list[int]]
    tiles_per_block: dict[tuple[int, str], int]
    buffer_degrees: dict[tuple[int, str, str], int]
    loop_order: list[list[str]]
    load_placements: dict[tuple[str, str], str]


def _init_buffer_degrees(fusion_groups: list[list[int]], da: DimAnalysis) -> dict[tuple[int, str, str], int]:
    """Set all buffer degrees to 1 (single-buffered)."""
    degrees: dict[tuple[int, str, str], int] = {}
    for group_idx, _group in enumerate(fusion_groups):
        for tensor_name, tinfo in da.tensors.items():
            for dim_id in tinfo.dim_ids:
                degrees.setdefault((group_idx, tensor_name, dim_id), 1)
    return degrees


def _init_loop_order(fusion_groups: list[list[int]], da: DimAnalysis) -> list[list[str]]:
    """Dimension ID order per group (all dims, sorted)."""
    order: list[list[str]] = []
    for _group in fusion_groups:
        order.append(sorted(da.dims))
    return order


def build_ir(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Construct the initial KernelIR from a math function.

    Runs dimension analysis and graph analysis, then sets all
    rendering parameters to their default (naive) values.

    Args:
        func: Math function using NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}``.

    Returns:
        KernelIR with default rendering parameters.
    """
    da = analyze_dims(func, input_specs)
    graph = build_op_graph(func)

    num_ops = len(graph.nodes)

    fusion_groups = [[i] for i in range(num_ops)]

    tiles_per_block: dict[tuple[int, str], int] = {}
    for op_idx in range(num_ops):
        for dim_id in da.dims:
            tiles_per_block[(op_idx, dim_id)] = 1

    buffer_degrees = _init_buffer_degrees(fusion_groups, da)
    loop_order = _init_loop_order(fusion_groups, da)

    return KernelIR(
        dim_analysis=da,
        op_graph=graph,
        fusion_groups=fusion_groups,
        tiles_per_block=tiles_per_block,
        buffer_degrees=buffer_degrees,
        loop_order=loop_order,
        load_placements={},
    )
