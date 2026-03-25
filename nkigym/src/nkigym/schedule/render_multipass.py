"""Multi-pass NKI kernel renderer for reduction schedules.

Handles kernels with two or more reduction passes over the same dimension,
such as rmsnorm+matmul (pass 0: activation+reduce, pass 1: nc_matmul).
Parallel dims nest via recursion; reduction passes iterate sequentially.
"""

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.codegen.passes import _PassAssignment
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.transpose import NKITranspose
from nkigym.schedule.render_mp_helpers import (
    _active_dim_levels,
    _all_barriers_done,
    _barrier_for_pass,
    _barrier_level,
    _barrier_output_dims,
    _emit_dma_loops,
    _mp_dma_hbm_tile_sl,
    _mp_dma_sbuf_tile_sl,
    _mp_hbm_load_sl,
    _mp_load_shape,
    _mp_one_tile_sl,
    _mp_output_store_sl,
    _mp_store_hbm_tile_sl,
    _mp_store_sbuf_tile_sl,
    _mp_tile_loop_lines,
    _MPCtx,
    _next_name,
    _nt_from_shape,
    _pass_acc_shape,
    _pass_needed_params,
    _should_emit_store,
    _split_tile_dims,
    resolve_var_buf,
)
from nkigym.schedule.render_slices import _multidim_shape, _num_blocks
from nkigym.schedule.types import Schedule, _ds_map, _total_tiles, _var_dim_ids


def render_multi_pass(
    analysis: _Analysis,
    schedule: Schedule,
    op_calls: list[_OpCall],
    params: tuple[str, ...],
    func_name: str,
    pa: _PassAssignment,
) -> str:
    """Render complete NKI kernel for a multi-pass reduction schedule."""
    tile_vars, loop_start = _mp_tile_var_map(analysis, schedule)
    ctx = _MPCtx(
        analysis=analysis,
        schedule=schedule,
        op_calls=op_calls,
        params=params,
        pa=pa,
        counters={"sbuf": 0, "psum": 0},
        var_buf={},
        buf_dims={},
        buf_nt={},
        tile_vars=tile_vars,
        loop_counter=[loop_start],
    )
    lines = _mp_preamble(func_name, params, analysis)
    lines.extend(_render_levels(0, len(schedule.loop_order), 1, ctx))
    lines.append("    return hbm_tensor_0")
    lines.append("")
    return "\n".join(lines)


def _mp_tile_var_map(analysis: _Analysis, schedule: Schedule) -> tuple[dict[str, str], int]:
    """Assign ``i_N`` loop variable names for multi-pass tile loops.

    Assigns a variable to every dim with total_tiles > 1.

    Returns:
        Tuple of (dim_id to var name mapping, next available counter).
    """
    counter = len(schedule.loop_order)
    tile_vars: dict[str, str] = {}
    for dim_id, _pass in schedule.loop_order:
        if dim_id not in tile_vars and _total_tiles(dim_id, analysis) > 1:
            tile_vars[dim_id] = f"i_{counter}"
            counter += 1
    return tile_vars, counter


def _mp_preamble(func_name: str, params: tuple[str, ...], analysis: _Analysis) -> list[str]:
    """Render imports, @nki.jit header, and output alloc."""
    shape = analysis.var_shapes[analysis.return_var]
    return [
        "import numpy as np",
        "import nki",
        "import nki.language as nl",
        "import nki.isa as nisa",
        "",
        "",
        "@nki.jit",
        f"def {func_name}({', '.join(params)}):",
        f"    hbm_tensor_0 = nl.ndarray({shape}, dtype={params[0]}.dtype, buffer=nl.shared_hbm)",
    ]


def _render_levels(start: int, end: int, indent: int, ctx: _MPCtx) -> list[str]:
    """Dispatch parallel dims (recursive) and reduction passes (sequential)."""
    lines: list[str] = []
    level = start
    reduction_set = set(ctx.analysis.reduction_dims)
    done = False
    while level < end and not done:
        dim_id = ctx.schedule.loop_order[level][0]
        if dim_id not in reduction_set:
            if _all_barriers_done(level, ctx):
                lines.extend(_post_barrier_store(level, end, indent, ctx))
            else:
                lines.extend(_parallel_level(level, end, indent, ctx))
            done = True
        else:
            pass_idx = ctx.schedule.loop_order[level][1]
            lines.extend(_reduction_pass(level, indent, ctx))
            lines.extend(_inter_pass_ops(dim_id, pass_idx, indent, ctx))
            level += 1
    if not done and _should_emit_store(start, end, ctx):
        lines.extend(_store_block(indent, ctx))
    return lines


def _post_barrier_store(start_level: int, end: int, indent: int, ctx: _MPCtx) -> list[str]:
    """Emit degenerate parallel loops + store for post-barrier dims."""
    lines: list[str] = []
    extra = 0
    for level in range(start_level, end):
        pad = "    " * (indent + extra)
        lines.append(f"{pad}for i_{level} in nl.affine_range(1):")
        extra += 1
    lines.extend(_store_block(indent + extra, ctx))
    return lines


def _parallel_level(level: int, end: int, indent: int, ctx: _MPCtx) -> list[str]:
    """Emit parallel for loop and recurse into deeper levels."""
    pad = "    " * indent
    nb = _num_blocks(level, ctx.schedule, ctx.analysis)
    lines = [f"{pad}for i_{level} in nl.affine_range({nb}):"]
    lines.extend(_render_levels(level + 1, end, indent + 1, ctx))
    return lines


def _reduction_pass(level: int, indent: int, ctx: _MPCtx) -> list[str]:
    """Emit one reduction pass: accumulator init + loop header + body."""
    pad = "    " * indent
    dim_id, pass_idx = ctx.schedule.loop_order[level]
    dim_levels = _active_dim_levels(level, ctx)
    barrier_op = _barrier_for_pass(dim_id, pass_idx, ctx)
    acc_shape = _pass_acc_shape(barrier_op, dim_levels, level, ctx)
    acc_name = _next_name(ctx, "psum")
    output_dims = _barrier_output_dims(barrier_op, ctx)
    ctx.buf_dims[acc_name] = output_dims
    ctx.buf_nt[acc_name] = _nt_from_shape(output_dims, acc_shape)
    ctx.var_buf[barrier_op.output_var] = acc_name
    nb = _num_blocks(level, ctx.schedule, ctx.analysis)
    lines = [
        f"{pad}{acc_name} = nl.ndarray({acc_shape}, dtype=nl.float32, buffer=nl.psum)",
        f"{pad}for i_{level} in nl.affine_range({nb}):",
    ]
    lines.extend(_pass_body(level, indent + 1, dim_levels, acc_name, ctx))
    return lines


def _pass_body(level: int, indent: int, dim_levels: dict[str, int], acc_name: str, ctx: _MPCtx) -> list[str]:
    """Inside reduction loop: loads, scratch, tile loops, pre-compute, barrier."""
    dim_id, pass_idx = ctx.schedule.loop_order[level]
    lines, load_bufs = _pass_loads(dim_id, pass_idx, indent, dim_levels, level, ctx)
    barrier_op = _barrier_for_pass(dim_id, pass_idx, ctx)
    scratch_name = ""
    if barrier_op.stmt_type is NKIActivationReduce:
        scratch_name, scratch_lines = _alloc_scratch(barrier_op, indent, dim_levels, level, load_bufs, ctx)
        lines.extend(scratch_lines)
    pre_dims, barrier_only = _split_tile_dims(dim_id, pass_idx, ctx)
    outer_lines, outer_extra = _mp_tile_loop_lines(
        indent, dim_levels, level, ctx, pre_dims | barrier_only if not pre_dims else pre_dims
    )
    lines.extend(outer_lines)
    pre_indent = indent + outer_extra
    for op_idx in ctx.pa.pre_compute.get((dim_id, pass_idx), []):
        lines.extend(_render_pre_op(ctx.op_calls[op_idx], pre_indent, dim_levels, level, load_bufs, ctx))
    inner_lines, inner_extra = (
        _mp_tile_loop_lines(pre_indent, dim_levels, level, ctx, barrier_only) if pre_dims else ([], 0)
    )
    lines.extend(inner_lines)
    lines.extend(
        _barrier_compute(
            barrier_op, pre_indent + inner_extra, dim_levels, level, acc_name, scratch_name, load_bufs, ctx
        )
    )
    return lines


def _pass_loads(
    dim_id: str, pass_idx: int, indent: int, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx
) -> tuple[list[str], dict[str, str]]:
    """Load all needed params for a pass, return lines and load_bufs mapping."""
    needed = _pass_needed_params(dim_id, pass_idx, ctx)
    lines: list[str] = []
    load_bufs: dict[str, str] = {}
    for pidx in needed:
        buf_name, load_lines = _mp_load_one(pidx, indent, dim_levels, pass_level, ctx)
        load_bufs[ctx.params[pidx]] = buf_name
        lines.extend(load_lines)
    return lines, load_bufs


def _mp_load_one(
    pidx: int, indent: int, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx
) -> tuple[str, list[str]]:
    """Load one param: alloc SBUF + DMA copy."""
    pad = "    " * indent
    param = ctx.params[pidx]
    sbuf_name = _next_name(ctx, "sbuf")
    shape = _mp_load_shape(param, dim_levels, pass_level, ctx)
    dims = _var_dim_ids(ctx.analysis, param)
    ctx.buf_dims[sbuf_name] = dims
    ctx.buf_nt[sbuf_name] = _nt_from_shape(dims, shape)
    lines = [f"{pad}{sbuf_name} = nl.ndarray({shape}, dtype={param}.dtype, buffer=nl.sbuf)"]
    num_free = (len(shape) - 2) // 2
    nt_par = shape[1]
    nt_free = tuple(shape[2 + k] for k in range(num_free))
    if nt_par > 1 or any(n > 1 for n in nt_free):
        lines.extend(_mp_load_dma(pidx, sbuf_name, indent, dim_levels, pass_level, nt_par, nt_free, ctx))
    else:
        full_sl = ", ".join(f"0:{s}" for s in shape)
        hbm_sl = _mp_hbm_load_sl(param, dim_levels, pass_level, ctx)
        lines.append(f"{pad}nisa.dma_copy(dst={sbuf_name}[{full_sl}], src={param}[{hbm_sl}])")
    return sbuf_name, lines


def _mp_load_dma(
    pidx: int,
    sbuf_name: str,
    indent: int,
    dim_levels: dict[str, int],
    pass_level: int,
    nt_par: int,
    nt_free: tuple[int, ...],
    ctx: _MPCtx,
) -> list[str]:
    """Emit per-tile DMA loops for a param load."""
    loop_lines, par_lv, free_lvs, extra = _emit_dma_loops(ctx.loop_counter, indent, nt_par, nt_free)
    inner_pad = "    " * (indent + extra)
    sbuf_sl = _mp_dma_sbuf_tile_sl(pidx, ctx, par_lv, free_lvs)
    hbm_sl = _mp_dma_hbm_tile_sl(pidx, dim_levels, pass_level, ctx, par_lv, free_lvs)
    param = ctx.params[pidx]
    lines = list(loop_lines)
    lines.append(f"{inner_pad}nisa.dma_copy(dst={sbuf_name}[{sbuf_sl}], src={param}[{hbm_sl}])")
    return lines


def _alloc_scratch(
    barrier_op: _OpCall,
    indent: int,
    dim_levels: dict[str, int],
    pass_level: int,
    load_bufs: dict[str, str],
    ctx: _MPCtx,
) -> tuple[str, list[str]]:
    """Allocate scratch SBUF for activation_reduce activated-data output."""
    pad = "    " * indent
    data_var = barrier_op.input_vars[0]
    scratch_shape = _mp_load_shape(data_var, dim_levels, pass_level, ctx)
    scratch_name = _next_name(ctx, "sbuf")
    dims = _var_dim_ids(ctx.analysis, data_var)
    ctx.buf_dims[scratch_name] = dims
    ctx.buf_nt[scratch_name] = _nt_from_shape(dims, scratch_shape)
    return scratch_name, [f"{pad}{scratch_name} = nl.ndarray({scratch_shape}, dtype=nl.float32, buffer=nl.sbuf)"]


def _render_pre_op(
    op: _OpCall, indent: int, dim_levels: dict[str, int], pass_level: int, load_bufs: dict[str, str], ctx: _MPCtx
) -> list[str]:
    """Render one pre-compute op inside the tile loop."""
    lines: list[str] = []
    if op.stmt_type is NKITranspose:
        lines = _render_transpose(op, indent, dim_levels, pass_level, load_bufs, ctx)
    else:
        lines = _render_standard_pre(op, indent, dim_levels, pass_level, load_bufs, ctx)
    return lines


def _render_standard_pre(
    op: _OpCall, indent: int, dim_levels: dict[str, int], pass_level: int, load_bufs: dict[str, str], ctx: _MPCtx
) -> list[str]:
    """Render a standard pre-compute op (tensor_scalar broadcast, etc.)."""
    pad = "    " * indent
    operand_exprs = _pre_op_operands(op, dim_levels, pass_level, load_bufs, ctx)
    data_var = op.input_vars[0]
    data_buf, data_sl = _resolve_operand_sl(data_var, dim_levels, pass_level, load_bufs, ctx)
    dst_expr = f"{data_buf}[{data_sl}]"
    line = op.stmt_type.render_post_compute(dst_expr, operand_exprs, op.config_kwargs)
    ctx.var_buf[op.output_var] = data_buf
    return [f"{pad}{line}"]


def _render_transpose(
    op: _OpCall, indent: int, dim_levels: dict[str, int], pass_level: int, load_bufs: dict[str, str], ctx: _MPCtx
) -> list[str]:
    """Render 3-line transpose: alloc PSUM, nc_transpose, nl.copy back."""
    pad = "    " * indent
    data_var = op.input_vars[0]
    data_buf, data_sl = _resolve_operand_sl(data_var, dim_levels, pass_level, load_bufs, ctx)
    out_dims = _var_dim_ids(ctx.analysis, op.output_var)
    tp_shape = _single_tile_shape(out_dims, ctx)
    tp_sl = ", ".join(f"0:{s}" for s in tp_shape)
    psum_name = _next_name(ctx, "psum")
    ctx.var_buf[op.output_var] = data_buf
    return [
        f"{pad}{psum_name} = nl.ndarray({tp_shape}, dtype={ctx.params[0]}.dtype, buffer=nl.psum)",
        f"{pad}nisa.nc_transpose(dst={psum_name}[{tp_sl}], data={data_buf}[{data_sl}])",
        f"{pad}nisa.tensor_copy(dst={data_buf}[{data_sl}], src={psum_name}[{tp_sl}])",
    ]


def _single_tile_shape(dims: tuple[str, ...], ctx: _MPCtx) -> tuple[int, ...]:
    """Single-tile multi-dim shape (all num_tiles=1)."""
    ds = _ds_map(ctx.schedule)
    ts_map = {d: ds[d].tile_size for d in dims}
    nt_map = {d: 1 for d in dims}
    return _multidim_shape(dims, ts_map, nt_map)


def _pre_op_operands(
    op: _OpCall, dim_levels: dict[str, int], pass_level: int, load_bufs: dict[str, str], ctx: _MPCtx
) -> dict[str, str]:
    """Build operand expressions for a pre-compute op."""
    operand_axes: dict[str, tuple[str, ...]] = getattr(op.stmt_type, "OPERAND_AXES", {})
    exprs: dict[str, str] = {}
    for (op_name, _axes), var_name in zip(operand_axes.items(), op.input_vars):
        buf, sl = _resolve_operand_sl(var_name, dim_levels, pass_level, load_bufs, ctx)
        exprs[op_name] = f"{buf}[{sl}]"
    return exprs


def _resolve_operand_sl(
    var_name: str, dim_levels: dict[str, int], pass_level: int, load_bufs: dict[str, str], ctx: _MPCtx
) -> tuple[str, str]:
    """Resolve a variable to its buffer name and per-tile slice."""
    buf = resolve_var_buf(var_name, load_bufs, ctx)
    dims = ctx.buf_dims.get(buf, _var_dim_ids(ctx.analysis, var_name))
    nt = ctx.buf_nt.get(buf, {})
    sl = _mp_one_tile_sl(dims, dim_levels, pass_level + 1, ctx, nt)
    return buf, sl


def _barrier_compute(
    barrier_op: _OpCall,
    indent: int,
    dim_levels: dict[str, int],
    pass_level: int,
    acc_name: str,
    scratch_name: str,
    load_bufs: dict[str, str],
    ctx: _MPCtx,
) -> list[str]:
    """Render barrier op compute line."""
    pad = "    " * indent
    acc_dims = ctx.buf_dims[acc_name]
    acc_nt = ctx.buf_nt.get(acc_name, {})
    acc_sl = _mp_one_tile_sl(acc_dims, dim_levels, pass_level + 1, ctx, acc_nt)
    dst_expr = f"{acc_name}[{acc_sl}]"
    operand_exprs = _barrier_operands(barrier_op, dim_levels, pass_level, scratch_name, load_bufs, ctx)
    line = barrier_op.stmt_type.render_compute(dst_expr, operand_exprs, barrier_op.config_kwargs)
    return [f"{pad}{line}"]


def _barrier_operands(
    barrier_op: _OpCall,
    dim_levels: dict[str, int],
    pass_level: int,
    scratch_name: str,
    load_bufs: dict[str, str],
    ctx: _MPCtx,
) -> dict[str, str]:
    """Build operand expressions for a barrier op."""
    operand_axes: dict[str, tuple[str, ...]] = getattr(barrier_op.stmt_type, "OPERAND_AXES", {})
    exprs: dict[str, str] = {}
    first_sl = ""
    for (op_name, _axes), var_name in zip(operand_axes.items(), barrier_op.input_vars):
        buf, sl = _resolve_operand_sl(var_name, dim_levels, pass_level, load_bufs, ctx)
        exprs[op_name] = f"{buf}[{sl}]"
        if not first_sl:
            first_sl = sl
    if scratch_name:
        exprs["_scratch_dst"] = f"{scratch_name}[{first_sl}]"
    return exprs


def _inter_pass_ops(dim_id: str, pass_idx: int, indent: int, ctx: _MPCtx) -> list[str]:
    """Stage PSUM to SBUF and render inter-pass 1D ops if any exist."""
    inter_ops = ctx.pa.inter_pass.get((dim_id, pass_idx), [])
    lines: list[str] = []
    if inter_ops:
        lines = _inter_pass_stage(dim_id, pass_idx, indent, inter_ops, ctx)
    return lines


def _inter_pass_stage(dim_id: str, pass_idx: int, indent: int, inter_ops: list[int], ctx: _MPCtx) -> list[str]:
    """Stage PSUM to SBUF and render each inter-pass op in-place."""
    pad = "    " * indent
    barrier_op = _barrier_for_pass(dim_id, pass_idx, ctx)
    psum_name = ctx.var_buf[barrier_op.output_var]
    acc_dims = ctx.buf_dims[psum_name]
    blvl = _barrier_level(dim_id, pass_idx, ctx.schedule)
    dim_levels = _active_dim_levels(blvl, ctx)
    acc_shape = _pass_acc_shape(barrier_op, dim_levels, blvl, ctx)
    full_sl = ", ".join(f"0:{s}" for s in acc_shape)
    stage_name = _next_name(ctx, "sbuf")
    ctx.buf_dims[stage_name] = acc_dims
    ctx.buf_nt[stage_name] = _nt_from_shape(acc_dims, acc_shape)
    lines = [
        f"{pad}{stage_name} = nl.ndarray({acc_shape}, dtype=nl.float32, buffer=nl.sbuf)",
        f"{pad}nisa.tensor_copy(dst={stage_name}[{full_sl}], src={psum_name}[{full_sl}])",
    ]
    for op_idx in inter_ops:
        lines.extend(_render_inter_op(ctx.op_calls[op_idx], stage_name, pad, full_sl, ctx))
    return lines


def _render_inter_op(op: _OpCall, stage_name: str, pad: str, full_sl: str, ctx: _MPCtx) -> list[str]:
    """Render one inter-pass op in-place on the staging buffer."""
    dst_expr = f"{stage_name}[{full_sl}]"
    operand_exprs = {"data": f"{stage_name}[{full_sl}]"}
    line = op.stmt_type.render_post_compute(dst_expr, operand_exprs, op.config_kwargs)
    ctx.var_buf[op.output_var] = stage_name
    return [f"{pad}{line}"]


def _store_block(indent: int, ctx: _MPCtx) -> list[str]:
    """Emit PSUM to SBUF staging and DMA store to HBM."""
    pad = "    " * indent
    last_b = ctx.pa.barrier_ops[-1]
    blvl = _barrier_level(last_b[1], last_b[2], ctx.schedule)
    dim_levels = _active_dim_levels(blvl, ctx)
    barrier_op = ctx.op_calls[last_b[0]]
    acc_shape = _pass_acc_shape(barrier_op, dim_levels, blvl, ctx)
    psum_name = ctx.var_buf[barrier_op.output_var]
    full_sl = ", ".join(f"0:{s}" for s in acc_shape)
    stage_name = _next_name(ctx, "sbuf")
    output_dims = _barrier_output_dims(barrier_op, ctx)
    ctx.buf_nt[stage_name] = _nt_from_shape(output_dims, acc_shape)
    lines = [
        f"{pad}{stage_name} = nl.ndarray({acc_shape}, dtype={ctx.params[0]}.dtype, buffer=nl.sbuf)",
        f"{pad}nisa.tensor_copy(dst={stage_name}[{full_sl}], src={psum_name}[{full_sl}])",
    ]
    lines.extend(_store_dma(stage_name, acc_shape, indent, ctx))
    return lines


def _store_dma(stage_name: str, acc_shape: tuple[int, ...], indent: int, ctx: _MPCtx) -> list[str]:
    """Emit per-tile DMA store loops from SBUF to HBM."""
    pad = "    " * indent
    num_free = (len(acc_shape) - 2) // 2
    nt_par = acc_shape[1]
    nt_free = tuple(acc_shape[2 + k] for k in range(num_free))
    lines: list[str] = []
    if nt_par > 1 or any(n > 1 for n in nt_free):
        loop_lines, par_lv, free_lvs, extra = _emit_dma_loops(ctx.loop_counter, indent, nt_par, nt_free)
        lines.extend(loop_lines)
        inner_pad = "    " * (indent + extra)
        hbm_sl = _mp_store_hbm_tile_sl(ctx, par_lv, free_lvs)
        sbuf_sl = _mp_store_sbuf_tile_sl(ctx, par_lv, free_lvs)
        lines.append(f"{inner_pad}nisa.dma_copy(dst=hbm_tensor_0[{hbm_sl}], src={stage_name}[{sbuf_sl}])")
    else:
        out_sl = _mp_output_store_sl(ctx)
        full_sl = ", ".join(f"0:{s}" for s in acc_shape)
        lines.append(f"{pad}nisa.dma_copy(dst=hbm_tensor_0[{out_sl}], src={stage_name}[{full_sl}])")
    return lines
