"""DMA codegen: load and store instructions for HBM↔SBUF transfers."""

from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import DimAnalysis, TensorInfo
from nkigym.kernel_ir.op_graph import OpGraph


def render_loads_for_group(ir: KernelIR, group: list[int], indent: int) -> list[str]:
    """Emit load_tensor_block calls for HBM inputs of a fusion group.

    Uses the load_tensor_block gadget which iterates all
    num_tiles slots in the buffer (including physical tile slots).
    The HBM offset is computed from the loop variables for
    the tensor's dimensions.

    Args:
        ir: Complete kernel IR.
        group: List of op indices in this group.
        indent: Current indentation level.

    Returns:
        List of source lines for load calls.
    """
    da = ir.dim_analysis
    graph = ir.op_graph

    hbm_inputs = _group_hbm_inputs(group, graph, da)

    lines: list[str] = []
    pad = "    " * indent

    for tensor_name in hbm_inputs:
        tinfo = da.tensors[tensor_name]
        par_ofs, free_ofs = _tensor_offsets(ir, tinfo, group)
        lines.append(f"{pad}load_tensor_block(sbuf_{tensor_name}, {tensor_name}, {par_ofs}, {free_ofs})")

    return lines


def render_store(ir: KernelIR, indent: int) -> str:
    """Emit store_tensor_block for the return tensor.

    Placed at the bottom of the innermost DP loop, after all
    reduction groups finish.

    Args:
        ir: Complete kernel IR.
        indent: Indentation level (innermost DP loop body).

    Returns:
        Source line for the store call.
    """
    da = ir.dim_analysis
    ret = da.return_name
    tinfo = da.tensors[ret]
    all_ops = list(range(len(ir.op_graph.op_classes)))

    par_ofs, free_ofs = _tensor_offsets(ir, tinfo, all_ops)

    pad = "    " * indent
    src_name = f"sbuf_{ret}"
    return f"{pad}store_tensor_block({ret}, {src_name}, {par_ofs}, {free_ofs})"


def _tensor_offsets(ir: KernelIR, tinfo: TensorInfo, ops: list[int]) -> tuple[str, str]:
    """Build HBM offset expressions for a tensor's partition and free axes.

    Args:
        ir: Complete kernel IR.
        tinfo: Tensor metadata with dim_ids.
        ops: Op indices to scan for tiles_per_block.

    Returns:
        Tuple of (par_offset_expr, free_offset_expr). Free offset
        is ``"0"`` for 1D tensors.
    """
    da = ir.dim_analysis
    par_tpb = get_tpb(ir, tinfo.dim_ids[0], ops)
    par_ofs = _offset_expr(tinfo.dim_ids[0], da, par_tpb)
    if len(tinfo.dim_ids) == 2:
        free_tpb = get_tpb(ir, tinfo.dim_ids[1], ops)
        free_ofs = _offset_expr(tinfo.dim_ids[1], da, free_tpb)
    else:
        free_ofs = "0"
    return par_ofs, free_ofs


def _group_hbm_inputs(group: list[int], graph: OpGraph, da: DimAnalysis) -> list[str]:
    """Find HBM tensors consumed by ops in a group (deduplicated, ordered)."""
    seen: set[str] = set()
    result: list[str] = []
    for op_idx in group:
        inputs, _ = graph.op_tensors[op_idx]
        for tensor_name in inputs.values():
            if tensor_name in da.tensors and da.tensors[tensor_name].isa_loc == "hbm":
                if tensor_name not in seen:
                    seen.add(tensor_name)
                    result.append(tensor_name)
    return result


def _offset_expr(dim_id: str, da: DimAnalysis, tpb: int) -> str:
    """Build the HBM offset expression for one dimension.

    Combines block, tile, and physical tile loop variables:
    ``i_block * (tpb * logical_tile_size) + i_tile * logical_tile_size + i_ptile * physical_tile_size``
    """
    di = da.dims[dim_id]
    logical = di.logical_tile_size
    physical = di.physical_tile_size
    block_stride = tpb * logical
    return f"i_block_{dim_id} * {block_stride}" f" + i_tile_{dim_id} * {logical}" f" + i_ptile_{dim_id} * {physical}"
