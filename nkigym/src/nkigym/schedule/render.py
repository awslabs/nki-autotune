"""Render NKI source code from schedule + workload.

Produces a complete NKI kernel with ``nl.affine_range`` loops
controlled by the schedule descriptor.  Infrastructure op positions
(init_acc, staging, store, post-compute) are derived from the
schedule rather than from explicit placement entries.

Design doc reference: nkigym_ir_guide.md sections 2 and 5.
"""

from nkigym.codegen.analysis import _Analysis, _OpCall, has_reduction
from nkigym.codegen.passes import _PassAssignment
from nkigym.schedule.render_mp_helpers import _emit_dma_loops
from nkigym.schedule.render_multipass import render_multi_pass
from nkigym.schedule.render_names import _assign_names, _Names
from nkigym.schedule.render_slices import (
    _acc_compute_slices,
    _dma_load_hbm_tile_slice,
    _dma_load_sbuf_tile_slice,
    _dma_store_hbm_tile_slice,
    _dma_store_sbuf_tile_slice,
    _full_tile_slices,
    _hbm_load_slices,
    _num_blocks,
    _output_store_slices,
    _output_tile_shape,
    _post_compute_sbuf_slices,
    _sbuf_compute_slices,
    _sbuf_load_shape,
)
from nkigym.schedule.types import Schedule, _ds_map, _first_reduction_position, _load_loop_level


def render_schedule(
    analysis: _Analysis,
    schedule: Schedule,
    op_calls: list[_OpCall],
    params: tuple[str, ...],
    func_name: str,
    pa: _PassAssignment,
) -> str:
    """Render complete NKI kernel source from schedule + workload."""
    is_multi = any(n > 1 for n in pa.passes_per_dim.values())
    result = ""
    if is_multi:
        result = render_multi_pass(analysis, schedule, op_calls, params, func_name, pa)
    else:
        names = _assign_names(analysis, schedule, op_calls, params)
        lines = _preamble(func_name, params, analysis, names)
        lines.extend(_level_lines(0, analysis, schedule, op_calls, params, names))
        lines.append(f"    return {names.hbm}")
        lines.append("")
        result = "\n".join(lines)
    return result


def _preamble(func_name: str, params: tuple[str, ...], analysis: _Analysis, names: _Names) -> list[str]:
    """Render imports, @nki.jit header, and output alloc."""
    lines = ["import nki", "import nki.language as nl", "import nki.isa as nisa", "", ""]
    params_str = ", ".join(params)
    lines.append("@nki.jit")
    lines.append(f"def {func_name}({params_str}):")
    output_shape = analysis.var_shapes[analysis.return_var]
    lines.append(f"    {names.hbm} = nl.ndarray({output_shape}, dtype={params[0]}.dtype, buffer=nl.shared_hbm)")
    return lines


def _level_lines(
    level: int, analysis: _Analysis, schedule: Schedule, op_calls: list[_OpCall], params: tuple[str, ...], names: _Names
) -> list[str]:
    """Recursively render lines for one loop level and deeper."""
    lines: list[str] = []
    indent = 1 + level
    pad = "    " * indent
    num_items = len(schedule.loop_order)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    has_red = any(has_reduction(op) for op in op_calls)
    acc_level = red_pos if has_red else -1
    store_level = red_pos if has_red else num_items
    lines.extend(_pre_ops(level, indent, analysis, schedule, params, acc_level, names))
    if level < num_items:
        nb = _num_blocks(level, schedule, analysis)
        lines.append(f"{pad}for i_{level} in nl.affine_range({nb}):")
        lines.extend(_level_lines(level + 1, analysis, schedule, op_calls, params, names))
    else:
        lines.extend(_compute_lines(indent, analysis, schedule, op_calls, params, names))
    lines.extend(_post_ops(level, indent, analysis, schedule, op_calls, params, store_level, names))
    return lines


def _pre_ops(
    level: int,
    indent: int,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    acc_level: int,
    names: _Names,
) -> list[str]:
    """Emit pre-loop operations at this level (init_acc, loads)."""
    lines: list[str] = []
    if level == acc_level:
        lines.extend(_init_acc_lines(indent, analysis, schedule, names))
    for i, p in enumerate(params):
        load_level = _load_loop_level(i, schedule, analysis, params)
        if load_level == level:
            lines.extend(_load_lines(i, indent, analysis, schedule, params, names))
    return lines


def _post_ops(
    level: int,
    indent: int,
    analysis: _Analysis,
    schedule: Schedule,
    op_calls: list[_OpCall],
    params: tuple[str, ...],
    store_level: int,
    names: _Names,
) -> list[str]:
    """Emit post-loop operations (staging, post-compute ops, store)."""
    lines: list[str] = []
    if level == store_level:
        if names.staged:
            lines.extend(_staging_lines(indent, analysis, schedule, params, names))
        post_ops = _post_compute_op_calls(op_calls)
        var_to_buf = _build_var_to_buf(op_calls, names)
        for idx, pc_op in enumerate(post_ops):
            lines.extend(_post_compute_lines(idx, pc_op, var_to_buf, indent, analysis, schedule, params, names))
        lines.extend(_store_lines(indent, analysis, schedule, var_to_buf, names))
    return lines


def _tile_loop_lines(indent: int, schedule: Schedule, names: _Names) -> tuple[list[str], int]:
    """Emit tile-within-block loops for any dim with tiles_per_block > 1."""
    ds = _ds_map(schedule)
    lines: list[str] = []
    extra = 0
    for _level, (dim_id, _pass) in enumerate(schedule.loop_order):
        if ds[dim_id].tiles_per_block > 1:
            inner_pad = "    " * (indent + extra)
            var = names.tile_vars[dim_id]
            lines.append(f"{inner_pad}for {var} in nl.affine_range({ds[dim_id].tiles_per_block}):")
            extra += 1
    return lines, extra


def _compute_lines(
    indent: int,
    analysis: _Analysis,
    schedule: Schedule,
    op_calls: list[_OpCall],
    params: tuple[str, ...],
    names: _Names,
) -> list[str]:
    """Emit compute ops at the innermost level."""
    lines, extra = _tile_loop_lines(indent, schedule, names)
    final_pad = "    " * (indent + extra)
    acc_sl = _acc_compute_slices(analysis, schedule, names.tile_vars)
    for op in op_calls:
        if has_reduction(op):
            lines.append(f"{final_pad}{_render_one_compute(op, acc_sl, analysis, schedule, params, names)}")
    return lines


def _render_one_compute(
    op: _OpCall, acc_sl: str, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...], names: _Names
) -> str:
    """Render a single reduction op's compute line."""
    operand_axes: dict[str, tuple[str, ...]] = getattr(op.stmt_type, "OPERAND_AXES", {})
    operand_names = list(operand_axes.keys())
    param_index = {p: i for i, p in enumerate(params)}
    operand_exprs: dict[str, str] = {}
    for operand_name, var_name in zip(operand_names, op.input_vars):
        pidx = param_index[var_name]
        sl = _sbuf_compute_slices(pidx, analysis, schedule, params, names.tile_vars)
        operand_exprs[operand_name] = f"{names.load_sbufs[pidx]}[{sl}]"
    return op.stmt_type.render_compute(f"{names.psum}[{acc_sl}]", operand_exprs, op.config_kwargs)


def _load_lines(
    param_idx: int, indent: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...], names: _Names
) -> list[str]:
    """Emit SBUF allocation and per-tile DMA load for one parameter."""
    pad = "    " * indent
    param = params[param_idx]
    sbuf_name = names.load_sbufs[param_idx]
    sbuf_shape = _sbuf_load_shape(param_idx, analysis, schedule, params)
    num_free = (len(sbuf_shape) - 2) // 2
    nt_par = sbuf_shape[1]
    nt_free = tuple(sbuf_shape[2 + k] for k in range(num_free))
    lines = [f"{pad}{sbuf_name} = nl.ndarray({sbuf_shape}, dtype={param}.dtype, buffer=nl.sbuf)"]
    if nt_par > 1 or any(n > 1 for n in nt_free):
        loop_lines, par_lv, free_lvs, extra = _emit_dma_loops(names.counter, indent, nt_par, nt_free)
        lines.extend(loop_lines)
        inner_pad = "    " * (indent + extra)
        sbuf_sl = _dma_load_sbuf_tile_slice(param_idx, analysis, schedule, params, par_lv, free_lvs)
        hbm_sl = _dma_load_hbm_tile_slice(param_idx, analysis, schedule, params, par_lv, free_lvs)
        lines.append(f"{inner_pad}nisa.dma_copy(dst={sbuf_name}[{sbuf_sl}], src={param}[{hbm_sl}])")
    else:
        full_sl = ", ".join(f"0:{s}" for s in sbuf_shape)
        hbm_sl = _hbm_load_slices(param_idx, analysis, schedule, params)
        lines.append(f"{pad}nisa.dma_copy(dst={sbuf_name}[{full_sl}], src={param}[{hbm_sl}])")
    return lines


def _init_acc_lines(indent: int, analysis: _Analysis, schedule: Schedule, names: _Names) -> list[str]:
    """Emit accumulator allocation."""
    pad = "    " * indent
    tile = _output_tile_shape(analysis, schedule)
    return [f"{pad}{names.psum} = nl.ndarray({tile}, dtype=nl.float32, buffer=nl.psum)"]


def _staging_lines(
    indent: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...], names: _Names
) -> list[str]:
    """Emit PSUM to SBUF tensor copy for post-reduction compute."""
    pad = "    " * indent
    tile = _output_tile_shape(analysis, schedule)
    sl = _full_tile_slices(analysis, schedule)
    return [
        f"{pad}{names.staged} = nl.ndarray({tile}, dtype={params[0]}.dtype, buffer=nl.sbuf)",
        f"{pad}nisa.tensor_copy(dst={names.staged}[{sl}], src={names.psum}[{sl}])",
    ]


def _post_compute_op_calls(op_calls: list[_OpCall]) -> list[_OpCall]:
    """Collect post-compute ops (ops without reduction dims)."""
    return [op for op in op_calls if not has_reduction(op)]


def _build_var_to_buf(op_calls: list[_OpCall], names: _Names) -> dict[str, str]:
    """Map each op's output variable to its SBUF buffer name."""
    var_to_buf: dict[str, str] = {}
    post_idx = 0
    for op in op_calls:
        if has_reduction(op):
            var_to_buf[op.output_var] = names.staged
        else:
            var_to_buf[op.output_var] = names.post_sbufs[post_idx]
            post_idx += 1
    return var_to_buf


def _post_compute_lines(
    idx: int,
    op: _OpCall,
    var_to_buf: dict[str, str],
    indent: int,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    names: _Names,
) -> list[str]:
    """Emit lines for one post-compute op."""
    pad = "    " * indent
    tile = _output_tile_shape(analysis, schedule)
    sl = _full_tile_slices(analysis, schedule)
    buf_name = names.post_sbufs[idx]
    dst_expr = f"{buf_name}[{sl}]"
    operand_exprs = _build_operand_exprs(op, var_to_buf, sl, analysis, schedule, params, names)
    compute_line = op.stmt_type.render_post_compute(dst_expr, operand_exprs, op.config_kwargs)
    return [f"{pad}{buf_name} = nl.ndarray({tile}, dtype={params[0]}.dtype, buffer=nl.sbuf)", f"{pad}{compute_line}"]


def _build_operand_exprs(
    op: _OpCall,
    var_to_buf: dict[str, str],
    full_sl: str,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    names: _Names,
) -> dict[str, str]:
    """Build operand name to source expression mapping for a post-compute op."""
    operand_axes: dict[str, tuple[str, ...]] = getattr(op.stmt_type, "OPERAND_AXES", {})
    operand_names = list(operand_axes.keys())
    param_index = {p: i for i, p in enumerate(params)}
    exprs: dict[str, str] = {}
    for operand_name, var_name in zip(operand_names, op.input_vars):
        exprs[operand_name] = _resolve_operand(
            var_name, var_to_buf, param_index, full_sl, analysis, schedule, params, names
        )
    return exprs


def _resolve_operand(
    var_name: str,
    var_to_buf: dict[str, str],
    param_index: dict[str, int],
    full_sl: str,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    names: _Names,
) -> str:
    """Resolve one post-compute operand to its SBUF source expression."""
    if var_name in param_index:
        pidx = param_index[var_name]
        sl = _post_compute_sbuf_slices(pidx, analysis, schedule, params)
        result = f"{names.load_sbufs[pidx]}[{sl}]"
    elif var_name in var_to_buf:
        result = f"{var_to_buf[var_name]}[{full_sl}]"
    else:
        raise ValueError(f"Cannot resolve source for variable {var_name!r}")
    return result


def _store_lines(
    indent: int, analysis: _Analysis, schedule: Schedule, var_to_buf: dict[str, str], names: _Names
) -> list[str]:
    """Emit per-tile DMA store from SBUF to HBM output."""
    pad = "    " * indent
    tile = _output_tile_shape(analysis, schedule)
    num_free = (len(tile) - 2) // 2
    nt_par = tile[1]
    nt_free = tuple(tile[2 + k] for k in range(num_free))
    src = var_to_buf[analysis.return_var]
    lines: list[str] = []
    if nt_par > 1 or any(n > 1 for n in nt_free):
        loop_lines, par_lv, free_lvs, extra = _emit_dma_loops(names.counter, indent, nt_par, nt_free)
        lines.extend(loop_lines)
        inner_pad = "    " * (indent + extra)
        out_sl = _dma_store_hbm_tile_slice(analysis, schedule, par_lv, free_lvs)
        sbuf_sl = _dma_store_sbuf_tile_slice(analysis, schedule, par_lv, free_lvs)
        lines.append(f"{inner_pad}nisa.dma_copy(dst={names.hbm}[{out_sl}], src={src}[{sbuf_sl}])")
    else:
        out_sl = _output_store_slices(analysis, schedule)
        full_sl = _full_tile_slices(analysis, schedule)
        lines.append(f"{pad}nisa.dma_copy(dst={names.hbm}[{out_sl}], src={src}[{full_sl}])")
    return lines
