"""NKI op rendering: ISA calls, memset, and PSUM staging."""

from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.dim_analysis import TensorInfo, op_blocking_dims


def render_ops_for_group(
    ir: KernelIR, group: list[int], red_dims: list[str], inner_indent: int, base_indent: int, needs_staging: set[str]
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
        needs_staging: PSUM tensors needing SBUF staging.

    Returns:
        Tuple of (pre_lines, inner_lines, post_lines):
        - pre_lines: before reduction loops (memset)
        - inner_lines: at innermost loop level (ISA calls + non-blocking staging)
        - post_lines: after reduction loops (blocking staging)
    """
    da = ir.dim_analysis
    graph = ir.op_graph

    pre_lines: list[str] = []
    inner_lines: list[str] = []
    post_lines: list[str] = []

    red_set = set(red_dims)
    for op_idx in group:
        op_cls = graph.op_classes[op_idx]
        outputs = graph.op_tensors[op_idx][1]
        has_blocking = bool(op_blocking_dims(op_cls, da.per_op_axis_maps[op_idx]) & red_set)

        if op_cls.ISA_LOC == "psum" and has_blocking:
            for oname in outputs:
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

    dst_map = _dst_exprs(ir, op_idx)
    dst = list(dst_map.values())[0]
    operands = _operand_exprs(ir, op_idx, needs_staging)

    scalar_kwargs = dict(ir.op_graph.op_all_kwargs[op_idx])
    for ax_name, expr in dst_map.items():
        scalar_kwargs[f"__dst_{ax_name}"] = expr
    call = op_cls.format_isa_call(dst, operands, scalar_kwargs)
    return [f"{pad}{call}"]


def _render_staging(tensor_name: str, indent: int) -> list[str]:
    """Emit stage_tensor_block for a PSUM tensor."""
    pad = "    " * indent
    return [f"{pad}stage_tensor_block(sbuf_{tensor_name}, psum_{tensor_name})"]


def _dst_exprs(ir: KernelIR, op_idx: int) -> dict[str, str]:
    """Build destination expressions for all of an op's outputs.

    Returns ``{output_axis_name: buffer_expr}`` keyed by the
    names in ``OUTPUT_AXES``.
    """
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    outputs = graph.op_tensors[op_idx][1]
    result: dict[str, str] = {}
    for ax_name, oname in zip(op_cls.OUTPUT_AXES, outputs):
        tinfo = ir.dim_analysis.tensors[oname]
        idx = _dst_index_expr(ir, op_idx, tinfo)
        result[ax_name] = f"{op_cls.ISA_LOC}_{oname}{idx}"
    return result


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

        if ir.op_graph.producer_isa_loc(tensor_name) == "psum":
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
        tp = da.dims[tinfo.dim_ids[0]].physical_tile_size
        tf = da.dims[tinfo.dim_ids[1]].physical_tile_size
        idx = f"[0:{tp}, 0, 0, 0:{tf}]"
    elif ndims == 1:
        tp = da.dims[tinfo.dim_ids[0]].physical_tile_size
        idx = f"[0:{tp}, 0]"
    return idx


def _dst_index_expr(ir: KernelIR, op_idx: int, tinfo: TensorInfo) -> str:
    """Build destination index expression — no reshape.

    Destinations use the physical tile size and physical tile
    slice directly in the 4D layout. NKI simulator does not
    support reshape on write destinations.
    """
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    ndims = len(tinfo.dim_ids)
    idx = ""
    if ndims == 2:
        d_p, d_f = tinfo.dim_ids[0], tinfo.dim_ids[1]
        di_tp = da.dims[d_p].physical_tile_size
        di_tf = da.dims[d_f].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        op_tf = op_tiles.get(d_f, di_tf)
        ptile_p = _dim_ptile_slice(di_tp, op_tp)
        ptile_f = _dim_ptile_slice(di_tf, op_tf)
        idx = f"[0:{di_tp}, {ptile_p}, {ptile_f}, 0:{di_tf}]"
    elif ndims == 1:
        d_p = tinfo.dim_ids[0]
        di_tp = da.dims[d_p].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        ptile_p = _dim_ptile_slice(di_tp, op_tp)
        idx = f"[0:{di_tp}, {ptile_p}]"
    return idx


def _tile_index_expr(ir: KernelIR, op_idx: int, tinfo: TensorInfo) -> str:
    """Build the tile index expression for an op accessing a buffer.

    The buffer uses physical tile sizes with
    num_ptiles_per_ltile folded into num_tiles.
    Each op slices the buffer according to how many physical
    tiles it needs, then reshapes to its own tile size.

    - op_tile == physical_tile: one slot ``[0:di_t, 0, ...]``
    - op_tile > physical_tile: multi-slot ``[0:di_t, 0:n, ...]``
    - op_tile < physical_tile: sub-tile (via ptile loop offset)

    Always appends ``.reshape((op_tp, op_tf))`` for consistency.
    """
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    ndims = len(tinfo.dim_ids)
    idx = ""
    if ndims == 2:
        d_p, d_f = tinfo.dim_ids[0], tinfo.dim_ids[1]
        di_tp = da.dims[d_p].physical_tile_size
        di_tf = da.dims[d_f].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        op_tf = op_tiles.get(d_f, di_tf)
        ptile_p = _dim_ptile_slice(di_tp, op_tp)
        ptile_f = _dim_ptile_slice(di_tf, op_tf)
        idx = f"[0:{di_tp}, {ptile_p}, {ptile_f}, 0:{di_tf}].reshape(({op_tp}, {op_tf}))"
    elif ndims == 1:
        d_p = tinfo.dim_ids[0]
        di_tp = da.dims[d_p].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        ptile_p = _dim_ptile_slice(di_tp, op_tp)
        idx = f"[0:{di_tp}, {ptile_p}].reshape(({op_tp},))"
    return idx


def _dim_ptile_slice(physical_tile: int, op_tile: int) -> str:
    """Build the num_tiles index for one dimension.

    - op_tile == physical_tile: single slot → ``0``
    - op_tile > physical_tile: multi-slot → ``0:{op_tile // physical_tile}``
    - op_tile < physical_tile: sub-tile → ``0`` (ptile loop handles offset)
    """
    num_ptiles = op_tile // physical_tile
    return f"0:{num_ptiles}" if num_ptiles > 1 else "0"
