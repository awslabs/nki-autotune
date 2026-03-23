"""Schedule descriptor types for kernel generation.

A ``Schedule`` captures the execution strategy for a kernel
independently of the algorithm (user workload).  All types are
immutable NamedTuples so they are hashable and dedup-friendly.

Design doc reference: nkigym_ir_guide.md sections 2-4.
"""

from typing import NamedTuple

from nkigym.codegen.analysis import _Analysis

_SBUF_PARTITION_LIMIT = 128
_MAX_BUFFER_ELEMENTS = 1048576


class DimSchedule(NamedTuple):
    """Per-dimension blocking parameters.

    Attributes:
        dim_id: Global dimension ID from analysis (e.g. ``"d0"``).
        tile_size: Hardware tile size (from TILE_LIMITS or default 128).
        tiles_per_block: Number of tiles grouped into one block iteration.
    """

    dim_id: str
    tile_size: int
    tiles_per_block: int


class Schedule(NamedTuple):
    """Complete schedule descriptor for kernel generation.

    Attributes:
        loop_order: Item sequence of ``(dim_id, pass_index)`` tuples.
            Parallel dims use ``pass_index=0``.  Reduction passes use
            ``pass_index=0, 1, 2, ...``.
        dim_schedules: Per-dimension blocking (one per unique dim).
        op_placements: Per-load-op placement level (one per input param).
            Level 0 = all dims outside (load all tiles).
            Level N = all dims active (natural, smallest buffer).
    """

    loop_order: tuple[tuple[str, int], ...]
    dim_schedules: tuple[DimSchedule, ...]
    op_placements: tuple[int, ...]


def _ds_map(schedule: Schedule) -> dict[str, DimSchedule]:
    """Build dim_id to DimSchedule lookup.

    Args:
        schedule: Schedule descriptor.

    Returns:
        Mapping from dim_id to DimSchedule.
    """
    return {ds.dim_id: ds for ds in schedule.dim_schedules}


def _total_tiles(dim_id: str, analysis: _Analysis) -> int:
    """Return total tile count for a dimension.

    Args:
        dim_id: Dimension ID.
        analysis: Dimension analysis result.

    Returns:
        Number of tiles for this dimension.
    """
    counts = {**analysis.tile_counts, **analysis.reduction_tile_counts}
    return counts[dim_id]


def _dim_position(dim_id: str, loop_order: tuple[tuple[str, int], ...]) -> int:
    """Find position of first occurrence of dim_id in loop_order.

    Args:
        dim_id: Dimension ID to find.
        loop_order: Item sequence.

    Returns:
        Index of the first item with this dim_id.
    """
    for i, (d, _pass) in enumerate(loop_order):
        if d == dim_id:
            return i
    raise ValueError(f"Dim {dim_id!r} not in loop_order")


def _var_dim_ids(analysis: _Analysis, var: str) -> tuple[str, ...]:
    """Extract non-None dimension IDs for a variable.

    Args:
        analysis: Dimension analysis result.
        var: Variable name.

    Returns:
        Tuple of dimension ID strings (None entries filtered out).
    """
    return tuple(d for d in analysis.var_dims[var] if d is not None)


def _dependent_dims_ordered(param: str, analysis: _Analysis, loop_order: tuple[tuple[str, int], ...]) -> list[str]:
    """Get dependent dims of a param, ordered by first appearance in loop_order.

    Args:
        param: Parameter name.
        analysis: Dimension analysis result.
        loop_order: Item sequence.

    Returns:
        List of dim IDs sorted by their position in loop_order.
    """
    dims = _var_dim_ids(analysis, param) if param in analysis.var_dims else ()
    positions = {d: _dim_position(d, loop_order) for d in dims}
    return sorted(dims, key=lambda d: positions[d])


def _load_loop_level(param_idx: int, schedule: Schedule, analysis: _Analysis, params: tuple[str, ...]) -> int:
    """Convert semantic placement level to actual loop level.

    Level 0 means all dims outside (load before all loops).
    Level K means first K dependent dims active.
    Level N (natural) means all dependent dims active.

    Args:
        param_idx: Index into params / op_placements.
        schedule: Schedule descriptor.
        analysis: Dimension analysis result.
        params: Input parameter names.

    Returns:
        Actual loop level (0 to len(loop_order)).
    """
    param = params[param_idx]
    level = schedule.op_placements[param_idx]
    ordered = _dependent_dims_ordered(param, analysis, schedule.loop_order)
    active = ordered[:level]
    result = max((_dim_position(d, schedule.loop_order) for d in active), default=-1) + 1
    return result


def _first_reduction_position(loop_order: tuple[tuple[str, int], ...], analysis: _Analysis) -> int:
    """Find the position of the first reduction item in loop_order.

    Args:
        loop_order: Item sequence.
        analysis: Dimension analysis result.

    Returns:
        Position of the first reduction item, or len(loop_order) if none.
    """
    reduction_set = set(analysis.reduction_dims)
    matches = [i for i, (d, _pass) in enumerate(loop_order) if d in reduction_set]
    return matches[0] if matches else len(loop_order)


def _valid_loop_order(analysis: _Analysis, schedule: Schedule) -> bool:
    """Check loop_order contains all dims with valid pass structure.

    Each parallel dim appears exactly once with pass_index=0.
    Each reduction dim appears at least once.  Same-dim passes
    maintain ascending pass_index order.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule to validate.

    Returns:
        True if loop_order structure is valid.
    """
    all_dims = set(analysis.parallel_dims) | set(analysis.reduction_dims)
    seen_dims = {d for d, _ in schedule.loop_order}
    par_set = set(analysis.parallel_dims)
    last_pass: dict[str, int] = {}
    pass_order_ok = True
    for d, p in schedule.loop_order:
        if d in par_set:
            pass_order_ok = pass_order_ok and (p == 0)
        else:
            pass_order_ok = pass_order_ok and (d not in last_pass or p > last_pass[d])
            last_pass[d] = p
    return seen_dims == all_dims and pass_order_ok


def _valid_blocking(analysis: _Analysis, schedule: Schedule) -> bool:
    """Check blocking factors divide total tile counts.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule to validate.

    Returns:
        True if all blocking factors are valid divisors.
    """
    return all(_total_tiles(ds.dim_id, analysis) % ds.tiles_per_block == 0 for ds in schedule.dim_schedules)


def _sbuf_partition(param: str, analysis: _Analysis) -> int:
    """Compute SBUF partition dimension size for a load.

    With multi-dim buffer shapes, the partition dim is always the
    tile_size of the first dimension (never exceeds 128 by construction).

    Args:
        param: Parameter name.
        analysis: Dimension analysis result.

    Returns:
        Partition dimension size in elements.
    """
    dims = _var_dim_ids(analysis, param) if param in analysis.var_dims else ()
    first_dim = dims[0] if dims else ""
    return analysis.dim_tile_sizes[first_dim] if first_dim else analysis.var_shapes.get(param, (0,))[0]


def _valid_sbuf_sizes(analysis: _Analysis, params: tuple[str, ...]) -> bool:
    """Check no SBUF load creates a partition dim exceeding 128.

    Args:
        analysis: Dimension analysis result.
        params: Input parameter names.

    Returns:
        True if all SBUF partition dims are within limits.
    """
    return all(_sbuf_partition(p, analysis) <= _SBUF_PARTITION_LIMIT for p in params)


def _valid_acc_partition(analysis: _Analysis) -> bool:
    """Check that the accumulator partition dim does not exceed 128.

    With multi-dim buffer shapes, the partition dim is always the
    tile_size of the first return dimension.

    Args:
        analysis: Dimension analysis result.

    Returns:
        True if the accumulator partition dim is within limits.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    first_dim = return_dims[0] if return_dims else ""
    ts = analysis.dim_tile_sizes.get(first_dim, 0)
    return ts <= _SBUF_PARTITION_LIMIT


def _load_buffer_elements(param_idx: int, schedule: Schedule, analysis: _Analysis, params: tuple[str, ...]) -> int:
    """Compute total element count for one SBUF load buffer.

    Each dim contributes ``tile_size * num_tiles`` where num_tiles is
    ``tiles_per_block`` (entered) or ``total_tiles`` (outside).

    Args:
        param_idx: Index into params / op_placements.
        schedule: Schedule descriptor.
        analysis: Dimension analysis result.
        params: Input parameter names.

    Returns:
        Total number of elements in the buffer.
    """
    param = params[param_idx]
    dims = _var_dim_ids(analysis, param) if param in analysis.var_dims else ()
    load_level = _load_loop_level(param_idx, schedule, analysis, params)
    ds = _ds_map(schedule)
    total = 1
    for d in dims:
        nt = ds[d].tiles_per_block if _dim_position(d, schedule.loop_order) < load_level else _total_tiles(d, analysis)
        total *= ds[d].tile_size * nt
    return total


def _acc_buffer_elements(analysis: _Analysis, schedule: Schedule) -> int:
    """Compute total element count for the PSUM accumulator buffer.

    Dims outside reduction hold ``tiles_per_block`` tiles.
    Dims inside reduction hold ``total_tiles`` tiles.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.

    Returns:
        Total number of elements in the accumulator.
    """
    return_dims = _var_dim_ids(analysis, analysis.return_var)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    ds = _ds_map(schedule)
    total = 1
    for d in return_dims:
        nt = ds[d].tiles_per_block if _dim_position(d, schedule.loop_order) < red_pos else _total_tiles(d, analysis)
        total *= ds[d].tile_size * nt
    return total


def _valid_buffer_capacity(analysis: _Analysis, schedule: Schedule, params: tuple[str, ...]) -> bool:
    """Check that no single on-chip buffer exceeds capacity.

    Rejects schedules where any SBUF load or the PSUM accumulator
    exceeds ``_MAX_BUFFER_ELEMENTS`` (1M elements = 4MB at float32).

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule descriptor.
        params: Input parameter names.

    Returns:
        True if all buffers are within capacity limits.
    """
    loads_ok = all(
        _load_buffer_elements(i, schedule, analysis, params) <= _MAX_BUFFER_ELEMENTS for i in range(len(params))
    )
    acc_ok = _acc_buffer_elements(analysis, schedule) <= _MAX_BUFFER_ELEMENTS
    return loads_ok and acc_ok


def validate(analysis: _Analysis, schedule: Schedule, params: tuple[str, ...]) -> bool:
    """Check whether a schedule is valid for the given analysis.

    Args:
        analysis: Dimension analysis result.
        schedule: Schedule to validate.
        params: Input parameter names.

    Returns:
        True if the schedule passes all validation checks.
    """
    return (
        _valid_loop_order(analysis, schedule)
        and _valid_blocking(analysis, schedule)
        and _valid_sbuf_sizes(analysis, params)
        and _valid_acc_partition(analysis)
        and _valid_buffer_capacity(analysis, schedule, params)
    )
