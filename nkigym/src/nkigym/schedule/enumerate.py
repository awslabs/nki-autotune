"""Combinatorial schedule enumeration.

Generates the full schedule space as the cross-product of three axes:
loop orders x op placements x blocking.  Each schedule is an
independent point in this grid.

Design doc reference: nkigym_ir_guide.md section 3.
"""

import itertools

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.schedule.types import DimSchedule, Schedule, _total_tiles, _var_dim_ids, validate


def _build_items(analysis: _Analysis, op_calls: list[_OpCall]) -> list[tuple[str, int]]:
    """Build the set of loop items from analysis.

    Each parallel dim produces one item ``(dim_id, 0)``.
    Each reduction dim produces one item per pass ``(dim_id, pass_idx)``.
    Currently supports single-pass reduction (one pass per reduction dim).

    Args:
        analysis: Dimension analysis result.
        op_calls: Parsed operation calls.

    Returns:
        List of ``(dim_id, pass_index)`` items.
    """
    items: list[tuple[str, int]] = []
    for d in analysis.parallel_dims:
        items.append((d, 0))
    for d in analysis.reduction_dims:
        items.append((d, 0))
    return items


def _valid_pass_order(perm: tuple[tuple[str, int], ...]) -> bool:
    """Check that same-dim passes maintain ascending pass_index order.

    Args:
        perm: A permutation of items.

    Returns:
        True if pass ordering constraint is satisfied.
    """
    last_pass: dict[str, int] = {}
    ok = True
    for d, p in perm:
        if d in last_pass:
            ok = ok and (p > last_pass[d])
        last_pass[d] = p
    return ok


def enumerate_loop_orders(analysis: _Analysis, op_calls: list[_OpCall]) -> list[tuple[tuple[str, int], ...]]:
    """Generate all valid loop orderings.

    Valid orderings are permutations of items where same-dim passes
    maintain their relative order.

    Args:
        analysis: Dimension analysis result.
        op_calls: Parsed operation calls.

    Returns:
        List of valid loop_order tuples.
    """
    items = _build_items(analysis, op_calls)
    return [p for p in itertools.permutations(items) if _valid_pass_order(p)]


def _num_dependent_dims(param: str, analysis: _Analysis) -> int:
    """Count dimensions that a parameter depends on.

    Args:
        param: Parameter name.
        analysis: Dimension analysis result.

    Returns:
        Number of non-None dimensions.
    """
    return len(_var_dim_ids(analysis, param)) if param in analysis.var_dims else 0


def enumerate_op_placements(analysis: _Analysis, params: tuple[str, ...]) -> list[tuple[int, ...]]:
    """Generate all op placement combinations.

    Each load op gets a level from 0 to num_dependent_dims.

    Args:
        analysis: Dimension analysis result.
        params: Input parameter names.

    Returns:
        List of placement tuples (one int per param).
    """
    ranges = [range(_num_dependent_dims(p, analysis) + 1) for p in params]
    return [tuple(combo) for combo in itertools.product(*ranges)]


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n in ascending order.

    Uses O(sqrt(n)) trial division with paired collection.

    Args:
        n: Positive integer.

    Returns:
        Sorted list of all positive divisors.
    """
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    large.reverse()
    return small + large


def enumerate_blocking(analysis: _Analysis) -> list[tuple[DimSchedule, ...]]:
    """Generate all valid blocking combinations.

    For each dimension, enumerate divisors of total tile count
    as valid tiles_per_block values.

    Args:
        analysis: Dimension analysis result.

    Returns:
        List of dim_schedules tuples.
    """
    all_dims = analysis.parallel_dims + analysis.reduction_dims
    per_dim_options: list[list[DimSchedule]] = []
    for dim_id in all_dims:
        ts = analysis.dim_tile_sizes[dim_id]
        total = _total_tiles(dim_id, analysis)
        per_dim_options.append([DimSchedule(dim_id, ts, tpb) for tpb in _divisors(total)])
    return [tuple(combo) for combo in itertools.product(*per_dim_options)]


def enumerate_all(analysis: _Analysis, op_calls: list[_OpCall], params: tuple[str, ...]) -> list[Schedule]:
    """Generate all valid schedules via cross-product enumeration.

    Enumerates loop orders x op placements x blocking, validates
    each combination, and deduplicates via Schedule hash.

    Args:
        analysis: Dimension analysis result.
        op_calls: Parsed operation calls.
        params: Input parameter names.

    Returns:
        List of unique valid Schedule descriptors.
    """
    orders = enumerate_loop_orders(analysis, op_calls)
    placements = enumerate_op_placements(analysis, params)
    blockings = enumerate_blocking(analysis)
    seen: set[Schedule] = set()
    results: list[Schedule] = []
    for lo, pl, bl in itertools.product(orders, placements, blockings):
        sched = Schedule(loop_order=lo, dim_schedules=bl, op_placements=pl)
        if sched not in seen and validate(analysis, sched, params):
            seen.add(sched)
            results.append(sched)
    return results


def default_schedule(analysis: _Analysis, op_calls: list[_OpCall], params: tuple[str, ...]) -> Schedule:
    """Generate the naive default schedule.

    Parallel dims outermost, reduction dims innermost.
    ``tiles_per_block=1`` everywhere.  All loads at natural level.

    Args:
        analysis: Dimension analysis result.
        op_calls: Parsed operation calls.
        params: Input parameter names.

    Returns:
        Default Schedule descriptor.
    """
    items: list[tuple[str, int]] = [(d, 0) for d in analysis.parallel_dims]
    for d in analysis.reduction_dims:
        items.append((d, 0))
    loop_order = tuple(items)
    all_dims = analysis.parallel_dims + analysis.reduction_dims
    dim_schedules = tuple(DimSchedule(d, analysis.dim_tile_sizes[d], 1) for d in all_dims)
    op_placements = tuple(_num_dependent_dims(p, analysis) for p in params)
    return Schedule(loop_order=loop_order, dim_schedules=dim_schedules, op_placements=op_placements)
