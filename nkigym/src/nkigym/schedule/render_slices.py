"""Slice and shape computation helpers for schedule rendering.

Multi-dimensional buffer shapes follow the IR guide convention:
``(tile_size_par, num_tiles_par, num_tiles_free_0, ..., tile_size_free_0, ...)``.
Partition tile_size first, then partition num_tiles, then all free dims'
num_tiles grouped together, then all free dims' tile_sizes.
"""

from nkigym.codegen.analysis import _Analysis
from nkigym.schedule.types import (
    DimSchedule,
    Schedule,
    _dim_position,
    _ds_map,
    _first_reduction_position,
    _load_loop_level,
    _total_tiles,
    _var_dim_ids,
)


def _multidim_shape(dims: tuple[str, ...], ts_map: dict[str, int], nt_map: dict[str, int]) -> tuple[int, ...]:
    """Build multi-dim shape from per-dim tile sizes and tile counts.

    Convention: ``(ts_par, nt_par, nt_free0, ..., ts_free0, ...)``.

    Args:
        dims: Ordered dimension IDs (first = partition).
        ts_map: Dim ID to tile size.
        nt_map: Dim ID to num_tiles for this buffer.

    Returns:
        Multi-dimensional shape tuple.
    """
    par = dims[0]
    free = dims[1:]
    shape: list[int] = [ts_map[par], nt_map[par]]
    for d in free:
        shape.append(nt_map[d])
    for d in free:
        shape.append(ts_map[d])
    return tuple(shape)


def _output_tile_shape(analysis: _Analysis, schedule: Schedule) -> tuple[int, ...]:
    """Compute multi-dim accumulator tile shape.

    Dims outside reduction hold tiles_per_block tiles.
    Dims inside reduction hold total_tiles tiles.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.

    Returns:
        Multi-dimensional shape tuple.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    ds = _ds_map(schedule)
    ts_map = {d: ds[d].tile_size for d in return_dims}
    nt_map: dict[str, int] = {}
    for d in return_dims:
        outside = _dim_position(d, schedule.loop_order) < red_pos
        nt_map[d] = ds[d].tiles_per_block if outside else _total_tiles(d, analysis)
    return _multidim_shape(return_dims, ts_map, nt_map)


def _full_tile_slices(analysis: _Analysis, schedule: Schedule) -> str:
    """Generate full-range slices for the multi-dim accumulator shape.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.

    Returns:
        Comma-separated full-range slice expressions.
    """
    tile = _output_tile_shape(analysis, schedule)
    return ", ".join(f"0:{s}" for s in tile)


def _sbuf_load_shape(
    param_idx: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...]
) -> tuple[int, ...]:
    """Compute multi-dim SBUF shape for a load.

    Entered dims (position < load_level) hold tiles_per_block tiles.
    Outside dims hold total_tiles tiles.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.

    Returns:
        Multi-dimensional shape tuple.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    ts_map = {d: ds[d].tile_size for d in dims}
    nt_map: dict[str, int] = {}
    for d in dims:
        entered = _dim_position(d, schedule.loop_order) < load_level
        nt_map[d] = ds[d].tiles_per_block if entered else _total_tiles(d, analysis)
    return _multidim_shape(dims, ts_map, nt_map)


def _nt_slice_one(level: int, tile_var: str, tpb: int, entered: bool) -> str:
    """Generate num_tiles slice for selecting one tile at compute time.

    Args:
        level: Loop level for this dimension (for ``i_`` block var).
        tile_var: Loop variable name for tile-within-block iteration.
        tpb: Tiles per block.
        entered: Whether this dim was entered before the current scope.

    Returns:
        Slice expression for the num_tiles dimension.
    """
    if entered and tpb == 1:
        expr = "0:1"
    elif entered:
        expr = f"{tile_var}:{tile_var} + 1"
    elif tpb > 1:
        expr = f"i_{level} * {tpb} + {tile_var}:i_{level} * {tpb} + {tile_var} + 1"
    else:
        expr = f"i_{level}:i_{level} + 1"
    return expr


def _multidim_one_tile_slices(
    dims: tuple[str, ...],
    ds_map: dict[str, DimSchedule],
    lo: tuple[tuple[str, int], ...],
    threshold: int,
    tile_vars: dict[str, str],
) -> str:
    """Build multi-dim slices selecting one tile per dim.

    Convention: ``ts_par, nt_par, nt_free..., ts_free...``.
    A dim with ``position < threshold`` is considered entered.

    Args:
        dims: Ordered dimension IDs (first = partition).
        ds_map: DimSchedule map from ``_ds_map``.
        lo: Loop order.
        threshold: Loop level threshold for entered check.
        tile_vars: Mapping from dim_id to loop variable name for tile loops.

    Returns:
        Comma-separated slice expressions.
    """
    par = dims[0]
    free = dims[1:]
    parts: list[str] = [f"0:{ds_map[par].tile_size}"]
    j = _dim_position(par, lo)
    parts.append(_nt_slice_one(j, tile_vars.get(par, ""), ds_map[par].tiles_per_block, j < threshold))
    for d in free:
        j = _dim_position(d, lo)
        parts.append(_nt_slice_one(j, tile_vars.get(d, ""), ds_map[d].tiles_per_block, j < threshold))
    for d in free:
        parts.append(f"0:{ds_map[d].tile_size}")
    return ", ".join(parts)


def _acc_compute_slices(analysis: _Analysis, schedule: Schedule, tile_vars: dict[str, str]) -> str:
    """Generate multi-dim accumulator slices for the compute destination.

    Dims outside reduction are entered; dims inside are not.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        tile_vars: Mapping from dim_id to loop variable name for tile loops.

    Returns:
        Comma-separated slice expressions.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    ds = _ds_map(schedule)
    return _multidim_one_tile_slices(return_dims, ds, schedule.loop_order, red_pos, tile_vars)


def _sbuf_compute_slices(
    param_idx: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...], tile_vars: dict[str, str]
) -> str:
    """Generate multi-dim SBUF sub-tile slices for operands at compute time.

    Dims entered before the load level are inside; others are outside.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.
        tile_vars: Mapping from dim_id to loop variable name for tile loops.

    Returns:
        Comma-separated slice expressions.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    return _multidim_one_tile_slices(dims, ds, schedule.loop_order, load_level, tile_vars)


def _hbm_load_slices(param_idx: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...]) -> str:
    """Generate HBM source slice expressions for a DMA load.

    HBM tensors remain flat (2D). Entered dims use block-offset slices;
    outside dims use full-range slices.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.

    Returns:
        Comma-separated slice expressions.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    shape = analysis.var_shapes[param]
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    parts: list[str] = []
    for dim_id, full_size in zip(dims, shape):
        j = _dim_position(dim_id, schedule.loop_order)
        bs = ds[dim_id].tile_size * ds[dim_id].tiles_per_block
        expr = f"i_{j} * {bs}:i_{j} * {bs} + {bs}" if j < load_level else f"0:{full_size}"
        parts.append(expr)
    return ", ".join(parts)


def _output_store_slices(analysis: _Analysis, schedule: Schedule) -> str:
    """Generate HBM output slice expressions for the DMA store.

    HBM tensors remain flat. Dims outside reduction use block offsets.
    Dims inside reduction use the full range.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.

    Returns:
        Comma-separated slice expressions.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    ds = _ds_map(schedule)
    parts: list[str] = []
    for d in return_dims:
        j = _dim_position(d, schedule.loop_order)
        if j < red_pos:
            bs = ds[d].tile_size * ds[d].tiles_per_block
            parts.append(f"i_{j} * {bs}:i_{j} * {bs} + {bs}")
        else:
            full = _total_tiles(d, analysis) * analysis.dim_tile_sizes[d]
            parts.append(f"0:{full}")
    return ", ".join(parts)


def _nt_slice_block(level: int, tpb: int, load_level: int, red_pos: int, total: int) -> str:
    """Generate num_tiles slice for a block of tiles at post-compute time.

    Entered dims: full buffer range ``0:tpb``.
    In-scope dims: block selection via ``i_level``.
    Outside-scope dims: full range ``0:total``.

    Args:
        level: Loop level of this dimension.
        tpb: Tiles per block.
        load_level: Loop level where the param was loaded.
        red_pos: First reduction position.
        total: Total tiles for this dimension.

    Returns:
        Slice expression for the num_tiles dimension.
    """
    entered = level < load_level
    in_scope = level < red_pos
    if entered:
        expr = f"0:{tpb}"
    elif in_scope:
        expr = f"i_{level} * {tpb}:i_{level} * {tpb} + {tpb}"
    else:
        expr = f"0:{total}"
    return expr


def _post_compute_sbuf_slices(param_idx: int, analysis: _Analysis, schedule: Schedule, params: tuple[str, ...]) -> str:
    """Generate multi-dim SBUF slices for a post-compute operand.

    Block-granularity addressing at the post-compute level.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.

    Returns:
        Comma-separated slice expressions.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    par = dims[0]
    free = dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}"]
    j_par = _dim_position(par, schedule.loop_order)
    parts.append(_nt_slice_block(j_par, ds[par].tiles_per_block, load_level, red_pos, _total_tiles(par, analysis)))
    for d in free:
        j = _dim_position(d, schedule.loop_order)
        parts.append(_nt_slice_block(j, ds[d].tiles_per_block, load_level, red_pos, _total_tiles(d, analysis)))
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _hbm_par_tile_expr(level: int, threshold: int, block_size: int, tile_size: int, loop_var: str) -> str:
    """Generate HBM partition-dim slice for one tile within a block.

    Args:
        level: Loop level of this dimension.
        threshold: Level threshold (load_level or red_pos).
        block_size: Block size (tile_size * tiles_per_block).
        tile_size: Single tile size.
        loop_var: Loop variable name for tile iteration.

    Returns:
        Slice expression string.
    """
    if level < threshold:
        start = f"i_{level} * {block_size} + {loop_var} * {tile_size}"
        expr = f"{start}:{start} + {tile_size}"
    else:
        start = f"{loop_var} * {tile_size}"
        expr = f"{start}:{start} + {tile_size}"
    return expr


def _dma_load_sbuf_tile_slice(
    param_idx: int,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    par_lv: str,
    free_lvs: tuple[str, ...],
) -> str:
    """SBUF dest slice selecting one tile per dim for per-tile DMA load.

    Each num_tiles dim selects one position via its loop variable.
    Tile_size dims use full range.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.
        par_lv: Loop variable for partition num_tiles dim.
        free_lvs: Loop variables for each free num_tiles dim.

    Returns:
        Comma-separated slice expressions.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    ds = _ds_map(schedule)
    par = dims[0]
    free = dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}", f"{par_lv}:{par_lv} + 1"]
    for k, d in enumerate(free):
        parts.append(f"{free_lvs[k]}:{free_lvs[k]} + 1")
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _dma_load_hbm_tile_slice(
    param_idx: int,
    analysis: _Analysis,
    schedule: Schedule,
    params: tuple[str, ...],
    par_lv: str,
    free_lvs: tuple[str, ...],
) -> str:
    """HBM source slice for one tile per dim during per-tile DMA load.

    Each dim uses per-tile offset via its loop variable.

    Args:
        param_idx: Index into params / op_placements.
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.
        par_lv: Loop variable for partition dim.
        free_lvs: Loop variables for each free dim.

    Returns:
        Comma-separated slice expressions.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param)
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    parts: list[str] = []
    for dim_idx, dim_id in enumerate(dims):
        j = _dim_position(dim_id, schedule.loop_order)
        ts = ds[dim_id].tile_size
        bs = ts * ds[dim_id].tiles_per_block
        lv = par_lv if dim_idx == 0 else free_lvs[dim_idx - 1]
        parts.append(_hbm_par_tile_expr(j, load_level, bs, ts, lv))
    return ", ".join(parts)


def _dma_store_sbuf_tile_slice(analysis: _Analysis, schedule: Schedule, par_lv: str, free_lvs: tuple[str, ...]) -> str:
    """SBUF source slice selecting one tile per dim for per-tile DMA store.

    Each num_tiles dim selects one position via its loop variable.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        par_lv: Loop variable for partition num_tiles dim.
        free_lvs: Loop variables for each free num_tiles dim.

    Returns:
        Comma-separated slice expressions.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    ds = _ds_map(schedule)
    par = return_dims[0]
    free = return_dims[1:]
    parts: list[str] = [f"0:{ds[par].tile_size}", f"{par_lv}:{par_lv} + 1"]
    for k, d in enumerate(free):
        parts.append(f"{free_lvs[k]}:{free_lvs[k]} + 1")
    for d in free:
        parts.append(f"0:{ds[d].tile_size}")
    return ", ".join(parts)


def _dma_store_hbm_tile_slice(analysis: _Analysis, schedule: Schedule, par_lv: str, free_lvs: tuple[str, ...]) -> str:
    """HBM dest slice for one tile per dim during per-tile DMA store.

    Each dim uses per-tile offset via its loop variable.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        par_lv: Loop variable for partition dim.
        free_lvs: Loop variables for each free dim.

    Returns:
        Comma-separated slice expressions.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    ds = _ds_map(schedule)
    parts: list[str] = []
    for dim_idx, d in enumerate(return_dims):
        j = _dim_position(d, schedule.loop_order)
        ts = ds[d].tile_size
        bs = ts * ds[d].tiles_per_block
        lv = par_lv if dim_idx == 0 else free_lvs[dim_idx - 1]
        parts.append(_hbm_par_tile_expr(j, red_pos, bs, ts, lv))
    return ", ".join(parts)


def _num_blocks(level: int, schedule: Schedule, analysis: _Analysis) -> int:
    """Compute the number of block iterations for a loop level.

    Args:
        level: Loop level index.
        schedule: Schedule descriptor.
        analysis: Dimension analysis result.

    Returns:
        Number of block iterations.
    """
    dim_id = schedule.loop_order[level][0]
    ds = _ds_map(schedule)
    tpb = ds[dim_id].tiles_per_block
    return _total_tiles(dim_id, analysis) // tpb
