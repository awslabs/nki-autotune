"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo


def render_buffers(ir: KernelIR, indent: int) -> str:
    """Emit buffer allocations for every tensor in the IR.

    Args:
        ir: Complete kernel IR.
        indent: Indentation level.

    Returns:
        Indented NKI source lines.
    """
    da = ir.dim_analysis
    needs_sbuf_staging = find_psum_tensors_needing_sbuf(ir)
    tensor_to_psum_dtype = _build_psum_dtype_map(ir)
    lines: list[str] = []
    pad = "    " * indent

    for name, tinfo in da.tensors.items():
        shape = _buffer_shape(ir, name, tinfo)
        shape_str = ", ".join(str(s) for s in shape)

        if ir.op_graph.producer_isa_loc(name) == "psum":
            psum_dtype = tensor_to_psum_dtype.get(name, tinfo.dtype)
            lines.append(f"{pad}psum_{name} = nl.ndarray(({shape_str}), dtype=nl.{psum_dtype}, buffer=nl.psum)")
            if name in needs_sbuf_staging:
                lines.append(f"{pad}sbuf_{name} = nl.ndarray(({shape_str}), dtype=nl.{tinfo.dtype}, buffer=nl.sbuf)")
        else:
            lines.append(f"{pad}sbuf_{name} = nl.ndarray(({shape_str}), dtype=nl.{tinfo.dtype}, buffer=nl.sbuf)")

    return "\n".join(lines)


def find_psum_tensors_needing_sbuf(ir: KernelIR) -> set[str]:
    """Find PSUM tensors that need an SBUF staging buffer.

    A PSUM tensor needs staging when:
    1. A consumer op requires SBUF for the operand that reads
       this tensor (``INPUT_LOCS[role] == "sbuf"``).
    2. The tensor is the return tensor (needs dma_copy to HBM,
       which reads from SBUF).
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    result: set[str] = set()

    for consumer_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        input_locs = graph.op_classes[consumer_idx].INPUT_LOCS
        for role, tensor_name in inputs.items():
            if tensor_name not in da.tensors:
                continue
            if graph.producer_isa_loc(tensor_name) != "psum":
                continue
            if input_locs.get(role) == "sbuf":
                result.add(tensor_name)

    if graph.producer_isa_loc(da.return_name) == "psum":
        result.add(da.return_name)

    return result


def _build_psum_dtype_map(ir: KernelIR) -> dict[str, str]:
    """Map tensor names to PSUM dtype overrides.

    Only tensors produced by ops with ``PSUM_DTYPE`` set (e.g.
    nc_matmul → float32) get an override.
    """
    result: dict[str, str] = {}
    for op_idx, op_cls in enumerate(ir.op_graph.op_classes):
        if op_cls.PSUM_DTYPE is None:
            continue
        _, outputs = ir.op_graph.op_tensors[op_idx]
        for tensor_name in outputs:
            result[tensor_name] = op_cls.PSUM_DTYPE
    return result


def _buffer_shape(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> tuple[int, ...]:
    """Compute the buffer shape for a tensor.

    2D tensor → 4D: (phys_P, num_tiles_P, num_tiles_F, phys_F).
    1D tensor → 2D: (phys_P, num_tiles_P).
    """
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids

    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")

    shape: tuple[int, ...] = ()
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids[0], dim_ids[1]
        tile_p = da.dims[d_p].physical_tile_size
        tile_f = da.dims[d_f].physical_tile_size
        num_tiles_p = _compute_num_tiles(ir, tensor_name, d_p)
        num_tiles_f = _compute_num_tiles(ir, tensor_name, d_f)
        shape = (tile_p, num_tiles_p, num_tiles_f, tile_f)
    elif len(dim_ids) == 1:
        d_p = dim_ids[0]
        tile_p = da.dims[d_p].physical_tile_size
        num_tiles_p = _compute_num_tiles(ir, tensor_name, d_p)
        shape = (tile_p, num_tiles_p)

    return shape


def _compute_num_tiles(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Derive num_tiles for one dimension from KernelIR fields.

    num_tiles = num_ptiles * tpb_factor * blocks_factor * buffer_degree

    | tier       | tpb_factor | blocks_factor              |
    |------------|------------|----------------------------|
    | per_tile   | 1          | 1                          |
    | per_block  | tpb        | 1                          |
    | full       | tpb        | num_blocks                 |
    """
    da = ir.dim_analysis
    di = da.dims[dim_id]

    max_op_tile = _max_op_tile_for_tensor(ir, tensor_name, dim_id)
    num_ptiles = max_op_tile // di.physical_tile_size

    tier = ir.tensor_placements[(tensor_name, dim_id)]
    tpb = get_tpb(ir, dim_id)
    degree = ir.buffer_degrees[(tensor_name, dim_id)]

    if tier == "per_tile":
        tpb_factor = 1
        blocks_factor = 1
    elif tier == "per_block":
        tpb_factor = tpb
        blocks_factor = 1
    elif tier == "full":
        tpb_factor = tpb
        blocks_factor = di.dim_size // (tpb * di.logical_tile_size)
    else:
        raise ValueError(f"Unknown tensor_placement tier: {tier!r}")

    return num_ptiles * tpb_factor * blocks_factor * degree


def _max_op_tile_for_tensor(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Find the largest op tile size on a dimension across all ops touching a tensor."""
    da = ir.dim_analysis
    ops = _ops_for_tensor(ir, tensor_name)
    max_tile = da.dims[dim_id].logical_tile_size
    for op_idx in ops:
        op_tile = da.op_tile_sizes[op_idx].get(dim_id)
        if op_tile is not None:
            max_tile = max(max_tile, op_tile)
    return max_tile


def producer_op_tiles(ir: KernelIR, tensor_name: str) -> dict[str, int]:
    """Get the producing op's tile sizes for a tensor.

    For HBM inputs (no producer), returns the first consumer
    op's tile sizes. For on-chip tensors, returns the producing
    op's tile sizes.
    """
    graph = ir.op_graph
    match_idx = graph.producer_op(tensor_name)
    if match_idx is None:
        touching = graph.ops_touching(tensor_name)
        if not touching:
            raise ValueError(f"No op produces or consumes tensor {tensor_name!r}")
        match_idx = touching[0]
    return ir.dim_analysis.op_tile_sizes[match_idx]


def _ops_for_tensor(ir: KernelIR, tensor_name: str) -> list[int]:
    """Find all op indices that touch a tensor (as input or output)."""
    result = ir.op_graph.ops_touching(tensor_name)
    if not result:
        raise ValueError(f"No ops touch tensor {tensor_name!r}")
    return result
