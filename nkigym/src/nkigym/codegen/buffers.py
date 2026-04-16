"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.codegen.kernel_ir import KernelIR, get_tpb
from nkigym.dim_analysis.dim_analysis import TensorInfo


def render_buffers(ir: KernelIR, indent: int) -> str:
    """Emit buffer allocations for all tensors needing on-chip buffers.

    HBM inputs get an SBUF staging buffer for DMA loads. On-chip
    tensors (``"sbuf"`` or ``"psum"``) get their primary buffer,
    plus PSUM tensors get an SBUF staging buffer when a consumer
    requires it. All buffers use the uniform ``(tile_size,
    num_tiles)`` layout per dimension.

    Args:
        ir: Complete kernel IR.
        indent: Indentation level for the allocations.

    Returns:
        Indented NKI source lines for buffer allocations.
    """
    da = ir.dim_analysis

    tensor_to_psum_dtype = _build_psum_dtype_map(ir)
    needs_sbuf_staging = _find_psum_tensors_needing_sbuf(ir)

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


def _find_psum_tensors_needing_sbuf(ir: KernelIR) -> set[str]:
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

    2D tensor → 4D: (tile_size_P, num_tiles_P, num_tiles_F, tile_size_F).
    1D tensor → 2D: (tile_size_P, num_tiles_P).

    num_tiles is derived from load_placements, tiles_per_block,
    and buffer_degrees — no hardcoded defaults.
    """
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids

    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")

    shape: tuple[int, ...] = ()
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids[0], dim_ids[1]
        tile_p = da.dims[d_p].tile_size
        tile_f = da.dims[d_f].tile_size
        num_tiles_p = _compute_num_tiles(ir, tensor_name, d_p)
        num_tiles_f = _compute_num_tiles(ir, tensor_name, d_f)
        shape = (tile_p, num_tiles_p, num_tiles_f, tile_f)
    elif len(dim_ids) == 1:
        d_p = dim_ids[0]
        tile_p = da.dims[d_p].tile_size
        num_tiles_p = _compute_num_tiles(ir, tensor_name, d_p)
        shape = (tile_p, num_tiles_p)

    return shape


def _compute_num_tiles(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Derive num_tiles for one dimension from KernelIR fields.

    Determined by two independent choices in KernelIR:
    - ``load_placements`` → allocation location → how many
      loop iterations the buffer spans.
    - ``buffer_degrees`` → multi-buffering degree.

    | tier       | tiles covered                            |
    |------------|------------------------------------------|
    | per_tile   | 1                                        |
    | per_block  | tiles_per_block * interleave             |
    | full       | num_blocks * tiles_per_block * interleave|

    num_tiles = tiles_covered * buffer_degree
    """
    da = ir.dim_analysis
    di = da.dims[dim_id]

    tier = ir.load_placements[(tensor_name, dim_id)]
    ops_touching = _ops_for_tensor(ir, tensor_name)
    tpb = get_tpb(ir, dim_id, ops_touching)
    interleave = di.tile_size // di.min_tile_size
    degree = ir.buffer_degrees[_buffer_degree_key(ir, tensor_name, dim_id)]

    if tier == "per_tile":
        tiles_covered = 1
    elif tier == "per_block":
        tiles_covered = tpb * interleave
    elif tier == "full":
        num_blocks = di.dim_size // (tpb * di.tile_size)
        tiles_covered = num_blocks * tpb * interleave
    else:
        raise ValueError(f"Unknown load_placement tier: {tier!r}")

    return tiles_covered * degree


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
