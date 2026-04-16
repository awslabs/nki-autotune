"""NKI op rendering: ISA calls, memset, and PSUM staging."""

from nkigym.codegen.buffers import _find_psum_tensors_needing_sbuf
from nkigym.codegen.kernel_ir import KernelIR
from nkigym.dim_analysis.dim_analysis import TensorInfo


def render_ops_for_group(
    ir: KernelIR, group: list[int], red_dims: list[str], inner_indent: int, base_indent: int
) -> tuple[list[str], list[str], list[str]]:
    """Emit ISA calls, memset, and PSUM staging for a fusion group.

    For each op in the group, emits:
    - memset before blocking loops (for ops with BLOCKING_AXES)
    - ISA call at the innermost loop level
    - stage_tensor_block after blocking loops (for PSUM outputs needing SBUF)

    Args:
        ir: Complete kernel IR.
        group: List of op indices in this group.
        red_dims: Reduction dim IDs for this group.
        inner_indent: Indentation at the innermost loop body.
        base_indent: Indentation at the group's top level (before reduction loops).

    Returns:
        Tuple of (pre_lines, inner_lines, post_lines):
        - pre_lines: before reduction loops (memset)
        - inner_lines: at innermost loop level (ISA calls + non-blocking staging)
        - post_lines: after reduction loops (blocking staging)
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    needs_staging = _find_psum_tensors_needing_sbuf(ir)

    pre_lines: list[str] = []
    inner_lines: list[str] = []
    post_lines: list[str] = []

    for op_idx in group:
        op_cls = graph.op_classes[op_idx]
        outputs = graph.op_tensors[op_idx][1]
        axis_map = da.per_op_axis_maps[op_idx]
        blocking_dims = {axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map}
        has_blocking = bool(blocking_dims & set(red_dims))

        for oname in outputs:
            tinfo = da.tensors[oname]
            if tinfo.isa_loc == "psum" and has_blocking:
                pre_lines.extend(_render_memset(ir, oname, base_indent))

        inner_lines.extend(_render_isa_call(ir, op_idx, needs_staging, inner_indent))

        for oname in outputs:
            if oname in needs_staging:
                if has_blocking:
                    post_lines.extend(_render_staging(oname, base_indent))
                else:
                    inner_lines.extend(_render_staging(oname, inner_indent))

    return pre_lines, inner_lines, post_lines


def _render_memset(ir: KernelIR, tensor_name: str, indent: int) -> list[str]:
    """Emit nisa.memset to zero a PSUM buffer before its blocking loop."""
    tinfo = ir.dim_analysis.tensors[tensor_name]
    pad = "    " * indent
    idx = _buf_index_expr(ir, tinfo)
    return [f"{pad}nisa.memset(psum_{tensor_name}{idx}, 0.0)"]


def _render_isa_call(ir: KernelIR, op_idx: int, needs_staging: set[str], indent: int) -> list[str]:
    """Emit the nisa.* ISA call for one op."""
    op_cls = ir.op_graph.op_classes[op_idx]
    pad = "    " * indent

    dst = _dst_expr(ir, op_idx)
    operands = _operand_exprs(ir, op_idx, needs_staging)

    call = op_cls.format_isa_call(dst, operands)
    return [f"{pad}{call}"]


def _render_staging(tensor_name: str, indent: int) -> list[str]:
    """Emit stage_tensor_block for a PSUM tensor."""
    pad = "    " * indent
    return [f"{pad}stage_tensor_block(sbuf_{tensor_name}, psum_{tensor_name})"]


def _dst_expr(ir: KernelIR, op_idx: int) -> str:
    """Build the destination expression for an op's primary output."""
    outputs = ir.op_graph.op_tensors[op_idx][1]
    oname = outputs[0]
    tinfo = ir.dim_analysis.tensors[oname]
    loc = tinfo.isa_loc
    idx = _tile_index_expr(ir, op_idx, tinfo)
    return f"{loc}_{oname}{idx}"


def _operand_exprs(ir: KernelIR, op_idx: int, needs_staging: set[str]) -> dict[str, str]:
    """Build operand expressions for an op's inputs."""
    da = ir.dim_analysis
    op_cls = ir.op_graph.op_classes[op_idx]
    inputs = ir.op_graph.op_tensors[op_idx][0]

    result: dict[str, str] = {}
    for role, tensor_name in inputs.items():
        if role not in op_cls.OPERAND_AXES:
            continue
        tinfo = da.tensors.get(tensor_name)
        if tinfo is None:
            continue

        if tinfo.isa_loc == "hbm":
            buf_name = f"sbuf_{tensor_name}"
        elif tinfo.isa_loc == "psum":
            input_loc = op_cls.INPUT_LOCS.get(role, "sbuf")
            if input_loc == "sbuf" and tensor_name in needs_staging:
                buf_name = f"sbuf_{tensor_name}"
            else:
                buf_name = f"psum_{tensor_name}"
        else:
            buf_name = f"sbuf_{tensor_name}"

        idx = _tile_index_expr(ir, op_idx, tinfo)
        result[role] = f"{buf_name}{idx}"

    return result


def _buf_index_expr(ir: KernelIR, tinfo: TensorInfo) -> str:
    """Build buffer index expression using unified tile sizes.

    Used for buffer-level operations (memset, stage_tensor_block)
    that address the full buffer, not per-op tile slices.
    """
    da = ir.dim_analysis
    ndims = len(tinfo.dim_ids)
    idx = ""
    if ndims == 2:
        tp = da.dims[tinfo.dim_ids[0]].tile_size
        tf = da.dims[tinfo.dim_ids[1]].tile_size
        idx = f"[0:{tp}, 0, 0, 0:{tf}]"
    elif ndims == 1:
        tp = da.dims[tinfo.dim_ids[0]].tile_size
        idx = f"[0:{tp}, 0]"
    return idx


def _tile_index_expr(ir: KernelIR, op_idx: int, tinfo: TensorInfo) -> str:
    """Build the tile index expression using the op's own tile sizes.

    Each op uses its own tile limit per dimension (from
    ``op_tile_sizes``), not the unified dimension tile size.
    Degree-1 buffers have structurally-zero tile indices:
    4D: ``[0:tp, 0, 0, 0:tf]``
    2D: ``[0:tp, 0]``
    """
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    ndims = len(tinfo.dim_ids)
    idx = ""
    if ndims == 2:
        tp = op_tiles.get(tinfo.dim_ids[0], da.dims[tinfo.dim_ids[0]].tile_size)
        tf = op_tiles.get(tinfo.dim_ids[1], da.dims[tinfo.dim_ids[1]].tile_size)
        idx = f"[0:{tp}, 0, 0, 0:{tf}]"
    elif ndims == 1:
        tp = op_tiles.get(tinfo.dim_ids[0], da.dims[tinfo.dim_ids[0]].tile_size)
        idx = f"[0:{tp}, 0]"
    return idx
