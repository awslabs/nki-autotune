"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.codegen.kernel_ir import KernelIR
from nkigym.dim_analysis.dim_analysis import TensorInfo


def render_buffers(ir: KernelIR, indent: int) -> str:
    """Emit buffer allocations for all on-chip tensors.

    Each tensor with ``isa_loc`` of ``"sbuf"`` or ``"psum"`` gets
    a buffer with uniform ``(tile_size, num_tiles)`` layout per
    dimension. In the default lowering, ``num_tiles = 1`` for all
    dimensions.

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
        if tinfo.isa_loc == "hbm":
            continue

        shape = _buffer_shape(ir, name, tinfo)
        shape_str = ", ".join(str(s) for s in shape)

        if tinfo.isa_loc == "psum":
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
       this tensor (``op_input_locs[role] == "sbuf"``).
    2. The tensor is the return tensor (needs dma_copy to HBM,
       which reads from SBUF).
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    result: set[str] = set()

    for consumer_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        input_locs = graph.op_input_locs[consumer_idx]
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
    for op_idx, psum_dtype in enumerate(ir.op_graph.op_psum_dtypes):
        if psum_dtype is None:
            continue
        _, outputs = ir.op_graph.op_tensors[op_idx]
        for tensor_name in outputs:
            result[tensor_name] = psum_dtype
    return result


def _buffer_shape(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> tuple[int, ...]:
    """Compute the buffer shape for a tensor.

    2D tensor → 4D: (tile_size_P, num_tiles_P, num_tiles_F, tile_size_F).
    1D tensor → 2D: (tile_size_P, num_tiles_P).

    num_tiles per dimension = num_blocks * tiles_per_block * interleave * buffer_degree.
    In the default lowering, all factors are 1 so num_tiles = 1.
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
        num_tiles_p = _find_buffer_degree(ir, tensor_name, d_p)
        num_tiles_f = _find_buffer_degree(ir, tensor_name, d_f)
        shape = (tile_p, num_tiles_p, num_tiles_f, tile_f)
    elif len(dim_ids) == 1:
        d_p = dim_ids[0]
        tile_p = da.dims[d_p].tile_size
        num_tiles_p = _find_buffer_degree(ir, tensor_name, d_p)
        shape = (tile_p, num_tiles_p)

    return shape


def _find_buffer_degree(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Find buffer_degree for a tensor+dim (first group that has the key)."""
    degree = 1
    for group_idx in range(len(ir.fusion_groups)):
        key = (group_idx, tensor_name, dim_id)
        if key in ir.buffer_degrees:
            degree = ir.buffer_degrees[key]
            break
    return degree
