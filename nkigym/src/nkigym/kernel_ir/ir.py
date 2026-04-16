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
        ltiles_per_block: Per (op_idx, dim_id) tiling factor.
        buffer_degrees: Per (group_idx, tensor_name, dim_id) degree.
        loop_order: Per-group dimension ordering within each phase.
        tensor_placements: Per (tensor_name, dim_id) DMA tier.
    """

    dim_analysis: DimAnalysis
    op_graph: OpGraph
    fusion_groups: list[list[int]]
    ltiles_per_block: dict[tuple[int, str], int]
    buffer_degrees: dict[tuple[int, str, str], int]
    loop_order: list[list[str]]
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
        """Format ltiles_per_block as an op-by-dim table."""
        dim_ids = sorted({d for _, d in self.ltiles_per_block})
        op_indices = sorted({o for o, _ in self.ltiles_per_block})
        op_names = [self.op_graph.op_classes[i].NAME for i in op_indices]

        op_col_w = max(len("op"), *(len(n) for n in op_names))
        dim_col_ws = [
            max(len(d), *(len(str(self.ltiles_per_block.get((o, d), ""))) for o in op_indices)) for d in dim_ids
        ]

        header = "op".ljust(op_col_w) + "  | " + "  | ".join(d.ljust(dim_col_ws[i]) for i, d in enumerate(dim_ids))
        sep = "-" * op_col_w + "-+-" + "-+-".join("-" * w for w in dim_col_ws)
        rows: list[str] = []
        for o, name in zip(op_indices, op_names):
            vals = "  | ".join(
                str(self.ltiles_per_block.get((o, d), "")).ljust(dim_col_ws[i]) for i, d in enumerate(dim_ids)
            )
            rows.append(f"{name.ljust(op_col_w)}  | {vals}")

        pad = "    "
        return "\n".join([f"{pad}{header}", f"{pad}{sep}"] + [f"{pad}{r}" for r in rows])

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


def get_tpb(ir: KernelIR, dim_id: str, op_indices: list[int]) -> int:
    """Get ltiles_per_block for a dimension from the first matching op.

    Scans ``op_indices`` in order and returns the first
    ``ltiles_per_block[(op_idx, dim_id)]`` found.

    Args:
        ir: Complete kernel IR.
        dim_id: Dimension to look up.
        op_indices: Op indices to scan.

    Returns:
        ltiles_per_block value.

    Raises:
        ValueError: If no matching key is found.
    """
    for op_idx in op_indices:
        key = (op_idx, dim_id)
        if key in ir.ltiles_per_block:
            return ir.ltiles_per_block[key]
    raise ValueError(f"No ltiles_per_block for dim {dim_id!r} in ops {op_indices}")


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

    num_ops = len(graph.op_classes)

    fusion_groups = [[i] for i in range(num_ops)]

    ltiles_per_block: dict[tuple[int, str], int] = {}
    for op_idx in range(num_ops):
        for dim_id in da.op_tile_sizes[op_idx]:
            ltiles_per_block[(op_idx, dim_id)] = 1

    buffer_degrees = _init_buffer_degrees(fusion_groups, da)
    loop_order = _init_loop_order(fusion_groups, da)

    return KernelIR(
        dim_analysis=da,
        op_graph=graph,
        fusion_groups=fusion_groups,
        ltiles_per_block=ltiles_per_block,
        buffer_degrees=buffer_degrees,
        loop_order=loop_order,
        tensor_placements=_init_tensor_placements(da),
    )
