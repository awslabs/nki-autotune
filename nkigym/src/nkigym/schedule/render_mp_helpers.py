"""Helper types and functions for multi-pass NKI kernel rendering.

Provides the rendering context, buffer naming, dimension-level mapping,
and slice computation for multi-pass schedules.
"""

from dataclasses import dataclass

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.codegen.passes import _PassAssignment
from nkigym.schedule.render_slices import _hbm_par_tile_expr, _multidim_shape, _nt_slice_one
from nkigym.schedule.types import Schedule, _dim_position, _ds_map, _total_tiles, _var_dim_ids


@dataclass
class _MPCtx:
    """Multi-pass rendering context with mutable state."""

    analysis: _Analysis
    schedule: Schedule
    op_calls: list[_OpCall]
    params: tuple[str, ...]
    pa: _PassAssignment
    counters: dict[str, int]
    var_buf: dict[str, str]
    buf_dims: dict[str, tuple[str, ...]]
    buf_nt: dict[str, dict[str, int]]
    tile_vars: dict[str, str]
    loop_counter: list[int]


def _next_name(ctx: _MPCtx, space: str) -> str:
    """Allocate next tensor name for a memory space (sbuf or psum)."""
    n = ctx.counters[space]
    ctx.counters[space] = n + 1
    return f"{space}_tensor_{n}"


def _nt_from_shape(dims: tuple[str, ...], shape: tuple[int, ...]) -> dict[str, int]:
    """Extract per-dim num_tiles from multi-dim buffer shape."""
    nt_map: dict[str, int] = {dims[0]: shape[1]}
    for k, d in enumerate(dims[1:]):
        nt_map[d] = shape[2 + k]
    return nt_map


def _active_dim_levels(pass_level: int, ctx: _MPCtx) -> dict[str, int]:
    """Map each dim to its active loop level for a given pass context.

    Parallel dims use their position in loop_order.
    The reduction dim uses pass_level (current pass position).
    """
    result: dict[str, int] = {}
    reduction_set = set(ctx.analysis.reduction_dims)
    for i, (dim_id, _p) in enumerate(ctx.schedule.loop_order):
        if dim_id in reduction_set:
            if i == pass_level:
                result[dim_id] = i
        elif dim_id not in result:
            result[dim_id] = i
    return result


def _barrier_for_pass(dim_id: str, pass_idx: int, ctx: _MPCtx) -> _OpCall:
    """Find the barrier op call for a given (dim, pass)."""
    result = ctx.op_calls[0]
    for idx, d, p in ctx.pa.barrier_ops:
        if d == dim_id and p == pass_idx:
            result = ctx.op_calls[idx]
    return result


def _barrier_level(dim_id: str, pass_idx: int, schedule: Schedule) -> int:
    """Find the loop_order position for a barrier's (dim, pass)."""
    result = -1
    for i, (d, p) in enumerate(schedule.loop_order):
        if d == dim_id and p == pass_idx:
            result = i
    return result


def _should_emit_store(start: int, end: int, ctx: _MPCtx) -> bool:
    """Check if store should be emitted after processing [start, end).

    True when the last barrier's level falls in [start, end), meaning
    the barrier was processed at this recursion level.
    """
    last_b = ctx.pa.barrier_ops[-1]
    lvl = _barrier_level(last_b[1], last_b[2], ctx.schedule)
    return start <= lvl < end


def _all_barriers_done(level: int, ctx: _MPCtx) -> bool:
    """Check if all barrier ops have levels before the given level."""
    last_b = ctx.pa.barrier_ops[-1]
    lvl = _barrier_level(last_b[1], last_b[2], ctx.schedule)
    return lvl < level


def _pass_needed_params(dim_id: str, pass_idx: int, ctx: _MPCtx) -> list[int]:
    """Determine which input param indices need loading for a pass."""
    param_set = set(ctx.params)
    op_indices = list(ctx.pa.pre_compute.get((dim_id, pass_idx), []))
    for idx, d, p in ctx.pa.barrier_ops:
        if d == dim_id and p == pass_idx:
            op_indices.append(idx)
    needed: set[str] = set()
    for idx in op_indices:
        for v in ctx.op_calls[idx].input_vars:
            if v in param_set:
                needed.add(v)
    param_to_idx = {p: i for i, p in enumerate(ctx.params)}
    return sorted(param_to_idx[p] for p in needed)


def _build_axis_to_dim(op: _OpCall, ctx: _MPCtx) -> dict[str, str]:
    """Map axis labels to dim IDs using all operand variables."""
    operand_axes: dict[str, tuple[str, ...]] = getattr(op.stmt_type, "OPERAND_AXES", {})
    axis_to_dim: dict[str, str] = {}
    for op_idx, (operand_name, axes) in enumerate(operand_axes.items()):
        var_name = op.input_vars[op_idx]
        var_dims = ctx.analysis.var_dims[var_name]
        for ax_idx, ax_label in enumerate(axes):
            dim_val = var_dims[ax_idx]
            if ax_label not in axis_to_dim and dim_val is not None:
                axis_to_dim[ax_label] = dim_val
    return axis_to_dim


def _barrier_output_dims(barrier_op: _OpCall, ctx: _MPCtx) -> tuple[str, ...]:
    """Get the output dim IDs for a barrier op."""
    output_axes: tuple[str, ...] = getattr(barrier_op.stmt_type, "OUTPUT_AXES", ())
    axis_to_dim = _build_axis_to_dim(barrier_op, ctx)
    return tuple(axis_to_dim[ax] for ax in output_axes)


def _pass_acc_shape(barrier_op: _OpCall, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx) -> tuple[int, ...]:
    """Compute multi-dim accumulator shape for a pass barrier."""
    output_dims = _barrier_output_dims(barrier_op, ctx)
    ds = _ds_map(ctx.schedule)
    ts_map = {d: ds[d].tile_size for d in output_dims}
    nt_map: dict[str, int] = {}
    for d in output_dims:
        entered = dim_levels.get(d, pass_level) < pass_level
        nt_map[d] = ds[d].tiles_per_block if entered else _total_tiles(d, ctx.analysis)
    return _multidim_shape(output_dims, ts_map, nt_map)


def _mp_one_tile_sl(
    dims: tuple[str, ...], dim_levels: dict[str, int], threshold: int, ctx: _MPCtx, buf_nt: dict[str, int]
) -> str:
    """Multi-dim slices selecting one tile per dim at compute time.

    Entered dims (level < threshold) use standard tile/block vars.
    Non-entered dims (level >= threshold) use t_{dim} tile loop vars.
    """
    ds = _ds_map(ctx.schedule)
    par = dims[0]
    free = dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}"]
    parts.append(_mp_nt_slice(par, dim_levels, threshold, ctx, buf_nt))
    for d in free:
        parts.append(_mp_nt_slice(d, dim_levels, threshold, ctx, buf_nt))
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _mp_nt_slice(dim_id: str, dim_levels: dict[str, int], threshold: int, ctx: _MPCtx, buf_nt: dict[str, int]) -> str:
    """Generate num_tiles slice for one dim in multi-pass context.

    Entered dims (level < threshold) use standard tile/block vars.
    Non-entered dims use t_{dim} tile loop var or 0:1 if single tile.
    When buf_nt shows the buffer has total_tiles (> tiles_per_block),
    entered dims use block-indexed slices instead.
    """
    ds = _ds_map(ctx.schedule)
    j = dim_levels[dim_id]
    entered = j < threshold
    tpb = ds[dim_id].tiles_per_block
    buf_local = buf_nt.get(dim_id, tpb) <= tpb
    tile_var = ctx.tile_vars.get(dim_id, "")
    expr = _nt_slice_one(j, tile_var, tpb, entered and buf_local) if entered else ""
    if not expr:
        total = _total_tiles(dim_id, ctx.analysis)
        expr = "0:1" if total == 1 else f"{tile_var}:{tile_var} + 1"
    return expr


def _mp_load_shape(param: str, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx) -> tuple[int, ...]:
    """SBUF load buffer shape for a param in a pass context."""
    dims = _var_dim_ids(ctx.analysis, param)
    ds = _ds_map(ctx.schedule)
    load_threshold = pass_level + 1
    ts_map = {d: ds[d].tile_size for d in dims}
    nt_map: dict[str, int] = {}
    for d in dims:
        entered = dim_levels[d] < load_threshold
        nt_map[d] = ds[d].tiles_per_block if entered else _total_tiles(d, ctx.analysis)
    return _multidim_shape(dims, ts_map, nt_map)


def _mp_hbm_load_sl(param: str, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx) -> str:
    """HBM block-level load slices (simple case, no per-tile loops)."""
    dims = _var_dim_ids(ctx.analysis, param)
    shape = ctx.analysis.var_shapes[param]
    ds = _ds_map(ctx.schedule)
    load_threshold = pass_level + 1
    parts: list[str] = []
    for dim_id, full_size in zip(dims, shape):
        j = dim_levels[dim_id]
        bs = ds[dim_id].tile_size * ds[dim_id].tiles_per_block
        expr = f"i_{j} * {bs}:i_{j} * {bs} + {bs}" if j < load_threshold else f"0:{full_size}"
        parts.append(expr)
    return ", ".join(parts)


def _mp_tile_loop_lines(
    indent: int, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx, relevant_dims: set[str]
) -> tuple[list[str], int]:
    """Emit tile loops for dims relevant to the current pass.

    Entered/pass-level dims iterate tiles_per_block.
    Non-entered dims iterate total_tiles.
    Only dims in relevant_dims are included.
    """
    ds = _ds_map(ctx.schedule)
    tile_dims: list[tuple[str, int, int]] = []
    for d, lvl in dim_levels.items():
        if d not in relevant_dims:
            continue
        count = ds[d].tiles_per_block if lvl <= pass_level else _total_tiles(d, ctx.analysis)
        if count > 1:
            tile_dims.append((d, count, lvl))
    tile_dims.sort(key=lambda x: x[2])
    lines: list[str] = []
    extra = 0
    for dim_id, count, _lvl in tile_dims:
        pad = "    " * (indent + extra)
        var = ctx.tile_vars[dim_id]
        lines.append(f"{pad}for {var} in nl.affine_range({count}):")
        extra += 1
    return lines, extra


def _pass_relevant_dims(dim_id: str, pass_idx: int, ctx: _MPCtx) -> set[str]:
    """Collect dims used by the barrier and pre-compute ops of a pass."""
    relevant: set[str] = set()
    barrier = _barrier_for_pass(dim_id, pass_idx, ctx)
    for var in barrier.input_vars:
        for d in _var_dim_ids(ctx.analysis, var):
            relevant.add(d)
    for d in _barrier_output_dims(barrier, ctx):
        relevant.add(d)
    for op_idx in ctx.pa.pre_compute.get((dim_id, pass_idx), []):
        op = ctx.op_calls[op_idx]
        for var in op.input_vars:
            for d in _var_dim_ids(ctx.analysis, var):
                relevant.add(d)
    return relevant


def _pre_compute_dims(dim_id: str, pass_idx: int, ctx: _MPCtx) -> set[str]:
    """Collect dims used by pre-compute ops only."""
    dims: set[str] = set()
    for op_idx in ctx.pa.pre_compute.get((dim_id, pass_idx), []):
        op = ctx.op_calls[op_idx]
        for var in op.input_vars:
            for d in _var_dim_ids(ctx.analysis, var):
                dims.add(d)
        for d in _var_dim_ids(ctx.analysis, op.output_var):
            dims.add(d)
    return dims


def _split_tile_dims(dim_id: str, pass_idx: int, ctx: _MPCtx) -> tuple[set[str], set[str]]:
    """Split pass dims into pre-compute and barrier-only sets.

    Returns (pre_compute_dims, barrier_only_dims).  When no pre-compute
    ops exist, pre_compute_dims is empty and all go to barrier_only.
    """
    all_dims = _pass_relevant_dims(dim_id, pass_idx, ctx)
    pre_dims = _pre_compute_dims(dim_id, pass_idx, ctx)
    barrier_only = all_dims - pre_dims
    return pre_dims, barrier_only


def resolve_var_buf(var_name: str, load_bufs: dict[str, str], ctx: _MPCtx) -> str:
    """Resolve a variable name to its buffer: load_bufs for params, var_buf otherwise."""
    result = load_bufs.get(var_name, "")
    if not result:
        result = ctx.var_buf.get(var_name, "")
    return result


def _mp_dma_sbuf_tile_sl(param_idx: int, ctx: _MPCtx, par_lv: str, free_lvs: tuple[str, ...]) -> str:
    """SBUF per-tile slice selecting one tile per dim for DMA load."""
    param = ctx.params[param_idx]
    dims = _var_dim_ids(ctx.analysis, param)
    ds = _ds_map(ctx.schedule)
    par = dims[0]
    free = dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}", f"{par_lv}:{par_lv} + 1"]
    for k, d in enumerate(free):
        parts.append(f"{free_lvs[k]}:{free_lvs[k]} + 1")
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _mp_dma_hbm_tile_sl(
    param_idx: int, dim_levels: dict[str, int], pass_level: int, ctx: _MPCtx, par_lv: str, free_lvs: tuple[str, ...]
) -> str:
    """HBM per-tile slice for DMA load using pass-aware dim levels."""
    param = ctx.params[param_idx]
    dims = _var_dim_ids(ctx.analysis, param)
    ds = _ds_map(ctx.schedule)
    parts: list[str] = []
    for dim_idx, dim_id in enumerate(dims):
        j = dim_levels[dim_id]
        ts = ds[dim_id].tile_size
        bs = ts * ds[dim_id].tiles_per_block
        lv = par_lv if dim_idx == 0 else free_lvs[dim_idx - 1]
        parts.append(_hbm_par_tile_expr(j, pass_level + 1, bs, ts, lv))
    return ", ".join(parts)


def _mp_store_hbm_tile_sl(ctx: _MPCtx, par_lv: str, free_lvs: tuple[str, ...]) -> str:
    """HBM per-tile slice for DMA store."""
    return_dims = _var_dim_ids(ctx.analysis, ctx.analysis.return_var)
    ds = _ds_map(ctx.schedule)
    parts: list[str] = []
    for dim_idx, d in enumerate(return_dims):
        j = _dim_position(d, ctx.schedule.loop_order)
        ts = ds[d].tile_size
        bs = ts * ds[d].tiles_per_block
        lv = par_lv if dim_idx == 0 else free_lvs[dim_idx - 1]
        start = f"i_{j} * {bs} + {lv} * {ts}"
        parts.append(f"{start}:{start} + {ts}")
    return ", ".join(parts)


def _mp_store_sbuf_tile_sl(ctx: _MPCtx, par_lv: str, free_lvs: tuple[str, ...]) -> str:
    """SBUF per-tile slice for DMA store."""
    return_dims = _var_dim_ids(ctx.analysis, ctx.analysis.return_var)
    ds = _ds_map(ctx.schedule)
    par = return_dims[0]
    free = return_dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}", f"{par_lv}:{par_lv} + 1"]
    for k, d in enumerate(free):
        parts.append(f"{free_lvs[k]}:{free_lvs[k]} + 1")
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _mp_output_store_sl(ctx: _MPCtx) -> str:
    """HBM block-level output store slices (simple case)."""
    return_dims = _var_dim_ids(ctx.analysis, ctx.analysis.return_var)
    ds = _ds_map(ctx.schedule)
    parts: list[str] = []
    for d in return_dims:
        j = _dim_position(d, ctx.schedule.loop_order)
        bs = ds[d].tile_size * ds[d].tiles_per_block
        parts.append(f"i_{j} * {bs}:i_{j} * {bs} + {bs}")
    return ", ".join(parts)


def _emit_dma_loops(
    counter: list[int], indent: int, nt_par: int, nt_free: tuple[int, ...]
) -> tuple[list[str], str, tuple[str, ...], int]:
    """Emit nested affine_range loops for per-tile DMA.

    Args:
        counter: Mutable ``[next_idx]`` — allocates ``i_N`` vars and increments.
        indent: Base indentation level.
        nt_par: Number of tiles in the partition dimension.
        nt_free: Number of tiles in each free dimension.

    Returns:
        Tuple of (loop_lines, par_loop_var, free_loop_vars, extra_indent).
    """
    loop_lines: list[str] = []
    extra = 0
    par_lv = "0"
    if nt_par > 1:
        par_lv = f"i_{counter[0]}"
        counter[0] += 1
        pad = "    " * (indent + extra)
        loop_lines.append(f"{pad}for {par_lv} in nl.affine_range({nt_par}):")
        extra += 1
    free_lvs: list[str] = []
    for _k, nt in enumerate(nt_free):
        if nt > 1:
            lv = f"i_{counter[0]}"
            counter[0] += 1
            pad = "    " * (indent + extra)
            loop_lines.append(f"{pad}for {lv} in nl.affine_range({nt}):")
            free_lvs.append(lv)
            extra += 1
        else:
            free_lvs.append("0")
    return loop_lines, par_lv, tuple(free_lvs), extra
