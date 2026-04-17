"""KernelIR: structured representation for lowering to NKI source."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.kernel_ir.dim_analysis import DimAnalysis, analyze_dims
from nkigym.kernel_ir.op_graph import OpGraph, build_op_graph


@dataclass
class KernelIR:
    """Complete representation for lowering to NKI source.

    Composes two independent analysis results with rendering
    parameters that control loop structure, buffer sizes, and
    DMA placement.

    Attributes:
        dim_analysis: Dimension IDs, tile sizes, tensor metadata.
        op_graph: Producer-consumer DAG.
        fusion_groups: Which ops share a loop nest. Initially one
            singleton group per op; loop fusion merges groups.
        ltiles_per_block: Per-dimension tiling factor.
        buffer_degrees: Per (tensor_name, dim_id) multi-buffering
            degree. ``dim_id`` must be one of the tensor's
            ``dim_ids``.
        group_dim_orders: Per fusion group, the complete ordered
            list of dim IDs this group loops over (outer-to-inner).
            Positional on ``fusion_groups``. A group's dim list
            covers every dim touched by any op in the group. An
            empty list marks a group with no loops (every op has
            trip count 1 on every dim it touches). Stored rather
            than derived so loop-reordering transforms can permute
            the order without mutating ``op_graph``.
        tensor_placements: Per (tensor_name, dim_id) DMA tier.
    """

    dim_analysis: DimAnalysis
    op_graph: OpGraph
    fusion_groups: list[list[int]]
    ltiles_per_block: dict[str, int]
    buffer_degrees: dict[tuple[str, str], int]
    group_dim_orders: list[list[str]]
    tensor_placements: dict[tuple[str, str], str]

    def __repr__(self) -> str:
        """Show KernelIR with each field on its own line."""
        lines = [
            "KernelIR(",
            f"  dim_analysis={self.dim_analysis!r}",
            f"  op_graph={self.op_graph!r}",
            f"  fusion_groups={self.fusion_groups!r}",
            "  ltiles_per_block=",
            self._fmt_ltiles_per_block(),
            "  buffer_degrees=",
            self._fmt_buffer_degrees(),
            f"  group_dim_orders={self.group_dim_orders!r}",
            "  tensor_placements=",
            self._fmt_tensor_placements(),
            ")",
        ]
        return "\n".join(lines)

    def _fmt_ltiles_per_block(self) -> str:
        """Format ltiles_per_block as a dim → count table."""
        rows = [[dim_id, str(self.ltiles_per_block[dim_id])] for dim_id in sorted(self.ltiles_per_block)]
        return _fmt_table(["dim", "ltiles_per_block"], rows)

    def _fmt_buffer_degrees(self) -> str:
        """Format buffer_degrees as a tensor-by-dim table."""
        rows = [[tensor, dim_id, str(degree)] for (tensor, dim_id), degree in sorted(self.buffer_degrees.items())]
        return _fmt_table(["tensor", "dim", "degree"], rows)

    def _fmt_tensor_placements(self) -> str:
        """Format tensor_placements as a tensor-by-dim table."""
        rows = [[tensor, dim_id, tier] for (tensor, dim_id), tier in sorted(self.tensor_placements.items())]
        return _fmt_table(["tensor", "dim", "placement"], rows)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a list of rows as an aligned text table."""
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "  | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = ["  | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) for r in rows]
    pad = "    "
    return "\n".join([f"{pad}{header_line}", f"{pad}{sep_line}"] + [f"{pad}{line}" for line in data_lines])


def get_tpb(ir: KernelIR, dim_id: str) -> int:
    """Return ltiles_per_block for a dimension.

    ``ltiles_per_block`` is per-dimension — the same block
    structure applies to every op and tensor touching that
    dim. Missing keys raise ``ValueError``.

    Args:
        ir: Complete kernel IR.
        dim_id: Dimension to look up.

    Returns:
        ltiles_per_block value for the dimension.

    Raises:
        ValueError: If the dim is not in ``ir.ltiles_per_block``.
    """
    if dim_id not in ir.ltiles_per_block:
        raise ValueError(f"No ltiles_per_block for dim {dim_id!r}")
    return ir.ltiles_per_block[dim_id]


def _init_buffer_degrees(da: DimAnalysis) -> dict[tuple[str, str], int]:
    """Set all buffer degrees to 1 (single-buffered).

    One entry per ``(tensor_name, dim_id)`` where ``dim_id`` is in
    the tensor's own ``dim_ids``.
    """
    degrees: dict[tuple[str, str], int] = {}
    for tensor_name, tinfo in da.tensors.items():
        for dim_id in tinfo.dim_ids:
            degrees[(tensor_name, dim_id)] = 1
    return degrees


def _init_tensor_placements(da: DimAnalysis) -> dict[tuple[str, str], str]:
    """Set all tensor placements to per_tile (innermost, one tile)."""
    placements: dict[tuple[str, str], str] = {}
    for tensor_name, tinfo in da.tensors.items():
        for dim_id in tinfo.dim_ids:
            placements[(tensor_name, dim_id)] = "per_tile"
    return placements


def _group_dims(group: list[int], da: DimAnalysis, op_graph: OpGraph) -> list[str]:
    """Collect every dim any op in the group touches.

    Union of ``dim_ids`` across all input and output tensors of
    every op in ``group``. Returned sorted for a deterministic
    initial order; loop-reordering exposes permutations later.
    """
    dims: set[str] = set()
    for op_idx in group:
        for tensor_name in op_graph.op_tensor_names(op_idx):
            if tensor_name in da.tensors:
                dims.update(da.tensors[tensor_name].dim_ids)
    return sorted(dims)


def _init_group_dim_orders(fusion_groups: list[list[int]], da: DimAnalysis, op_graph: OpGraph) -> list[list[str]]:
    """Build the default per-group dim_order list.

    One entry per fusion group, positional on ``fusion_groups``.
    Each entry is every dim the group's ops touch, sorted by
    dim_id. No DP-vs-reduction distinction — every dim the ops
    touch is a loop.
    """
    return [_group_dims(group, da, op_graph) for group in fusion_groups]


def build_ir(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Construct the initial KernelIR from a math function.

    Runs dimension analysis and graph analysis, then sets all
    rendering parameters to their default (naive) values. The
    default is maximally unfused: one singleton fusion group per
    op, each owning a complete loop nest over every dim its op
    touches.

    Args:
        func: Math function using NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}``.

    Returns:
        KernelIR with default rendering parameters.
    """
    da = analyze_dims(func, input_specs)
    graph = build_op_graph(func)

    num_ops = len(graph.op_classes)

    fusion_groups = [[i] for i in range(num_ops)]

    ltiles_per_block: dict[str, int] = {dim_id: 1 for dim_id in da.dims}

    buffer_degrees = _init_buffer_degrees(da)
    group_dim_orders = _init_group_dim_orders(fusion_groups, da, graph)

    return KernelIR(
        dim_analysis=da,
        op_graph=graph,
        fusion_groups=fusion_groups,
        ltiles_per_block=ltiles_per_block,
        buffer_degrees=buffer_degrees,
        group_dim_orders=group_dim_orders,
        tensor_placements=_init_tensor_placements(da),
    )
