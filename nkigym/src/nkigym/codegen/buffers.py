"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo


def render_buffers(ir: KernelIR, indent: int, needs_sbuf_staging: set[str]) -> str:
    """Emit buffer allocations for all tensors needing on-chip buffers.

    HBM inputs get an SBUF staging buffer for DMA loads. On-chip
    tensors (``"sbuf"`` or ``"psum"``) get their primary buffer,
    plus PSUM tensors get an SBUF staging buffer when a consumer
    requires it. All buffers use the uniform ``(tile_size,
    num_tiles)`` layout per dimension.

    Args:
        ir: Complete kernel IR.
        indent: Indentation level for the allocations.
        needs_sbuf_staging: PSUM tensors needing SBUF staging.

    Returns:
        Indented NKI source lines for buffer allocations.
    """
    da = ir.dim_analysis

    tensor_to_psum_dtype = _build_psum_dtype_map(ir)

    lines: list[str] = []
    pad = "    " * indent

    for name, tinfo in da.tensors.items():
        shape = _buffer_shape(ir, name, tinfo)
        shape_str = ", ".join(str(s) for s in shape)

        if tinfo.isa_loc == "hbm":
            lines.append(f"{pad}sbuf_{name} = nl.ndarray(({shape_str}), dtype=nl.{tinfo.dtype}, buffer=nl.sbuf)")
        elif tinfo.isa_loc == "psum":
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
            tinfo = da.tensors.get(tensor_name)
            if tinfo is None or tinfo.isa_loc != "psum":
                continue
            if input_locs.get(role) == "sbuf":
                result.add(tensor_name)

    if da.return_name in da.tensors and da.tensors[da.return_name].isa_loc == "psum":
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

    Always uses the global unified tile size per dimension.
    Interleave is NOT folded into num_tiles — it is handled
    by reshape at the op level (section 6).

    2D tensor → 4D: (di_tile_P, num_tiles_P, num_tiles_F, di_tile_F).
    1D tensor → 2D: (di_tile_P, num_tiles_P).
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

    num_tiles = ig * tpb_factor * num_blocks_factor * buffer_degree

    ig = max(op_tile) / di_tile_size — enough slots for the
    largest op's tile on this dimension. This is the interleave
    factor: ops with larger tiles than di_tile consume multiple
    buffer slots and reshape.

    | tier       | tpb_factor | num_blocks_factor          |
    |------------|------------|----------------------------|
    | per_tile   | 1          | 1                          |
    | per_block  | tpb        | 1                          |
    | full       | tpb        | num_blocks                 |
    """
    da = ir.dim_analysis
    di = da.dims[dim_id]

    max_op_tile = _max_op_tile_for_tensor(ir, tensor_name, dim_id)
    ig = max_op_tile // di.physical_tile_size

    tier = ir.load_placements[(tensor_name, dim_id)]
    ops_touching = _ops_for_tensor(ir, tensor_name)
    tpb = get_tpb(ir, dim_id, ops_touching)
    degree = ir.buffer_degrees[_buffer_degree_key(ir, tensor_name, dim_id)]

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
        raise ValueError(f"Unknown load_placement tier: {tier!r}")

    return ig * tpb_factor * blocks_factor * degree


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
    da = ir.dim_analysis
    graph = ir.op_graph

    match_idx: int | None = None
    for op_idx, (_inputs, outputs) in enumerate(graph.op_tensors):
        if tensor_name in outputs:
            match_idx = op_idx
            break

    if match_idx is None:
        for op_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
            if tensor_name in inputs.values():
                match_idx = op_idx
                break

    if match_idx is None:
        raise ValueError(f"No op produces or consumes tensor {tensor_name!r}")

    return da.op_tile_sizes[match_idx]


def _ops_for_tensor(ir: KernelIR, tensor_name: str) -> list[int]:
    """Find all op indices that touch a tensor (as input or output)."""
    result: list[int] = []
    for op_idx, (inputs, outputs) in enumerate(ir.op_graph.op_tensors):
        if tensor_name in inputs.values() or tensor_name in outputs:
            result.append(op_idx)
    if not result:
        raise ValueError(f"No ops touch tensor {tensor_name!r}")
    return result


def _buffer_degree_key(ir: KernelIR, tensor_name: str, dim_id: str) -> tuple[int, str, str]:
    """Find the buffer_degrees key for a tensor+dim (first group that has it)."""
    for group_idx in range(len(ir.fusion_groups)):
        key = (group_idx, tensor_name, dim_id)
        if key in ir.buffer_degrees:
            return key
    raise ValueError(f"No buffer_degree for tensor {tensor_name!r}, dim {dim_id!r}")
