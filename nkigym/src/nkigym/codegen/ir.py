"""KernelIR data model per design doc section 5.1.

Immutable state (dims, tensors, op_graph) is produced once by
``build_ir``.  Transform state (fusion_groups, tiles_per_block,
loop_order, etc.) starts at defaults and is modified by transforms.
"""

from dataclasses import dataclass, field


@dataclass
class DimInfo:
    """Per-dimension global info, computed once by build_ir."""

    dim_size: int
    unified_tile_size: int
    min_tile_size: int

    def __repr__(self) -> str:
        """Compact single-line summary."""
        return f"DimInfo(size={self.dim_size}," f" tile={self.unified_tile_size}," f" min={self.min_tile_size})"


@dataclass
class TensorInfo:
    """Per-tensor info, computed once by build_ir.

    Attributes:
        isa_loc: Where the tensor lives or is produced.
            ``"hbm"`` -- kernel input, lives in HBM.
            ``"psum"`` -- ISA writes output to PSUM (nc_transpose, nc_matmul).
            ``"sbuf"`` -- ISA writes directly to SBUF (tensor_scalar, etc.).
    """

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    isa_loc: str

    def __repr__(self) -> str:
        """Compact single-line summary."""
        dims = ", ".join(self.dim_ids)
        shape = ", ".join(str(s) for s in self.shape)
        return f"TensorInfo(({dims}), ({shape}), {self.dtype}, {self.isa_loc})"


@dataclass
class OpDimInfo:
    """Per-op per-dimension tiling info, derived from hardware limits."""

    op_tile_size: int
    num_ig: int
    tiles_per_ig: int

    def __repr__(self) -> str:
        """Compact single-line summary."""
        return f"OpDimInfo(tile={self.op_tile_size}," f" ig={self.num_ig}," f" per_ig={self.tiles_per_ig})"


@dataclass
class OpInfo:
    """Node in the computation DAG."""

    op_type: str
    op_cls: type
    operands: dict[str, str]
    output: str
    dim_map: dict[str, str]
    per_dim: dict[str, OpDimInfo]
    predecessors: list[int]
    blocking_axes: frozenset[str]

    def __repr__(self) -> str:
        """Multi-line summary with operands, dim map, and tiling."""
        operands = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        dim_map = ", ".join(f"{k}->{v}" for k, v in self.dim_map.items())
        per_dim = "\n".join(f"    {k}: {v}" for k, v in self.per_dim.items())
        blocking = ", ".join(sorted(self.blocking_axes)) or "none"
        return (
            f"OpInfo({self.op_type} -> {self.output}\n"
            f"  operands: {operands}\n"
            f"  dim_map: {dim_map}\n"
            f"  per_dim:\n{per_dim}\n"
            f"  blocking: {{{blocking}}}, preds: {self.predecessors})"
        )


@dataclass
class OpGraph:
    """Computation DAG -- ops in topological order with explicit edges."""

    nodes: list[OpInfo]
    tensor_producers: dict[str, int]

    def __repr__(self) -> str:
        """Summary listing each op node."""
        node_lines = "\n".join(f"  [{i}] {n.op_type} -> {n.output}" for i, n in enumerate(self.nodes))
        return f"OpGraph({len(self.nodes)} ops\n{node_lines})"


@dataclass
class KernelIR:
    """Complete representation for lowering to NKI source."""

    func_name: str
    param_names: list[str]
    return_name: str
    dims: dict[str, DimInfo]
    tensors: dict[str, TensorInfo]
    op_graph: OpGraph
    input_specs: dict[str, tuple[tuple[int, ...], str]]

    """Transform state (mutable -- modified by transforms)."""
    fusion_groups: list[list[int]]
    tiles_per_block: dict[tuple[int, str], int]
    buffer_degrees: dict[tuple[int, str, str], int] = field(default_factory=dict)
    loop_order: list[list[str]] = field(default_factory=list)
    load_positions: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Multi-line summary of dims, tensors, ops, and transform state."""
        lines = [f"KernelIR({self.func_name})"]
        lines.append(f"  params: {self.param_names} -> {self.return_name}")
        lines.append("  dims:")
        for d, info in self.dims.items():
            lines.append(f"    {d}: {info}")
        lines.append("  tensors:")
        for name, t in self.tensors.items():
            lines.append(f"    {name}: {t}")
        lines.append("  ops:")
        for i, node in enumerate(self.op_graph.nodes):
            operands = ", ".join(f"{k}={v}" for k, v in node.operands.items())
            lines.append(f"    [{i}] {node.op_type}({operands}) -> {node.output}")
            for dim_id, odi in node.per_dim.items():
                lines.append(f"        {dim_id}: {odi}")
        lines.append(f"  fusion_groups: {self.fusion_groups}")
        lines.append(f"  loop_order: {self.loop_order}")
        tpb_parts = ", ".join(f"op{k[0]}:{k[1]}={v}" for k, v in self.tiles_per_block.items())
        lines.append(f"  tiles_per_block: {{{tpb_parts}}}")
        lp_parts = ", ".join(f"{k}={v}" for k, v in self.load_positions.items())
        lines.append(f"  load_positions: {{{lp_parts}}}")
        return "\n".join(lines)
