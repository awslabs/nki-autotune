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
        fusion_groups: Which ops share a loop nest.
        ltiles_per_block: Per-dimension tiling factor.
        buffer_degrees: Per (group_idx, tensor_name, dim_id) degree.
        loop_order: Single flat list describing the whole kernel
            loop nest. Top-level string entries are DP dimension
            IDs in outer-to-inner order; each nested ``list[str]``
            is one fusion group's reduction dim IDs in
            outer-to-inner order, positioned by fusion-group
            index. Groups with no reduction dims use an empty
            nested list. Example: ``["d0", "d4", ["d1"], ["d1", "d2"]]``
            means two DP loops ``d0, d4`` enclosing two sibling
            reduction groups, the first over ``d1`` alone and the
            second over ``d1`` then ``d2``.
        tensor_placements: Per (tensor_name, dim_id) DMA tier.
    """

    dim_analysis: DimAnalysis
    op_graph: OpGraph
    fusion_groups: list[list[int]]
    ltiles_per_block: dict[str, int]
    buffer_degrees: dict[tuple[int, str, str], int]
    loop_order: list[str | list[str]]
    tensor_placements: dict[tuple[str, str], str]

    def __repr__(self) -> str:
        """Show KernelIR with each field on its own line."""
        lines = [
            "KernelIR(",
            f"  dim_analysis={self.dim_analysis!r}",
            f"  op_graph={self.op_graph!r}",
            f"  fusion_groups={self.fusion_groups!r}",
            f"  ltiles_per_block=",
            self._fmt_ltiles_per_block(),
            "  buffer_degrees=",
            self._fmt_buffer_degrees(),
            f"  loop_order={self.loop_order!r}",
            "  tensor_placements=",
            self._fmt_tensor_placements(),
            ")",
        ]
        return "\n".join(lines)

    def _fmt_ltiles_per_block(self) -> str:
        """Format ltiles_per_block as a dim → count table."""
        rows: list[list[str]] = [
            [dim_id, str(self.ltiles_per_block[dim_id])] for dim_id in sorted(self.ltiles_per_block)
        ]

        headers = ["dim", "ltiles_per_block"]
        col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

        header_line = "  | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        data_lines = ["  | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) for r in rows]

        pad = "    "
        return "\n".join([f"{pad}{header_line}", f"{pad}{sep_line}"] + [f"{pad}{line}" for line in data_lines])

    def _fmt_buffer_degrees(self) -> str:
        """Format buffer_degrees as a group-by-tensor-by-dim table."""
        group_indices = sorted({g for g, _, _ in self.buffer_degrees})
        tensor_dim_pairs = sorted({(t, d) for _, t, d in self.buffer_degrees})
        tensor_names = sorted({t for _, t, _ in self.buffer_degrees})

        rows: list[list[str]] = []
        for g in group_indices:
            for t in tensor_names:
                dims_for_t = [d for tn, d in tensor_dim_pairs if tn == t]
                if not dims_for_t:
                    continue
                for d in dims_for_t:
                    val = self.buffer_degrees.get((g, t, d), "")
                    rows.append([str(g), t, d, str(val)])

        headers = ["group", "tensor", "dim", "degree"]
        col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

        header_line = "  | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        data_lines = ["  | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) for r in rows]

        pad = "    "
        return "\n".join([f"{pad}{header_line}", f"{pad}{sep_line}"] + [f"{pad}{line}" for line in data_lines])

    def _fmt_tensor_placements(self) -> str:
        """Format tensor_placements as a tensor-by-dim table."""
        rows: list[list[str]] = []
        for (tensor, dim_id), placement in sorted(self.tensor_placements.items()):
            rows.append([tensor, dim_id, placement])

        headers = ["tensor", "dim", "placement"]
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


def _init_buffer_degrees(fusion_groups: list[list[int]], da: DimAnalysis) -> dict[tuple[int, str, str], int]:
    """Set all buffer degrees to 1 (single-buffered)."""
    degrees: dict[tuple[int, str, str], int] = {}
    for group_idx, _group in enumerate(fusion_groups):
        for tensor_name, tinfo in da.tensors.items():
            for dim_id in tinfo.dim_ids:
                degrees.setdefault((group_idx, tensor_name, dim_id), 1)
    return degrees


def _init_tensor_placements(da: DimAnalysis) -> dict[tuple[str, str], str]:
    """Set all tensor placements to per_tile (innermost, one tile)."""
    placements: dict[tuple[str, str], str] = {}
    for tensor_name, tinfo in da.tensors.items():
        for dim_id in tinfo.dim_ids:
            placements[(tensor_name, dim_id)] = "per_tile"
    return placements


def _init_loop_order(fusion_groups: list[list[int]], da: DimAnalysis, op_graph: OpGraph) -> list[str | list[str]]:
    """Build the default nested loop_order.

    DP dims (sorted) go at the top level; one sorted reduction
    sublist per fusion group follows, positional on
    ``fusion_groups``. A group with no reduction dims contributes
    an empty sublist.
    """
    dp_dims = [d for d in sorted(da.dims) if da.dims[d].is_data_parallel]
    order: list[str | list[str]] = list(dp_dims)
    dp_set = set(dp_dims)
    for group in fusion_groups:
        group_dims: set[str] = set()
        for op_idx in group:
            inputs, outputs = op_graph.op_tensors[op_idx]
            for tensor_name in list(inputs.values()) + outputs:
                if tensor_name in da.tensors:
                    group_dims.update(da.tensors[tensor_name].dim_ids)
        order.append(sorted(group_dims - dp_set))
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

    num_ops = len(graph.op_classes)

    fusion_groups = [[i] for i in range(num_ops)]

    ltiles_per_block: dict[str, int] = {dim_id: 1 for dim_id in da.dims}

    buffer_degrees = _init_buffer_degrees(fusion_groups, da)
    loop_order = _init_loop_order(fusion_groups, da, graph)

    return KernelIR(
        dim_analysis=da,
        op_graph=graph,
        fusion_groups=fusion_groups,
        ltiles_per_block=ltiles_per_block,
        buffer_degrees=buffer_degrees,
        loop_order=loop_order,
        tensor_placements=_init_tensor_placements(da),
    )
