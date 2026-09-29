"""Build canonical operand regions and explicit ISA tensor views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import reduce
from math import gcd, prod

from nkigym.ir.arith.expr import Add, Const, Expr, FloorDiv, Mod, Mul, Var
from nkigym.ir.dimension_analysis import _AnalysisResult, _OpRecord
from nkigym.ir.tree import AccessPattern, Buffer, BufferRegion, partition_extent

TileSize = Callable[[_OpRecord, str, _AnalysisResult], int]
TripCount = Callable[[_OpRecord, str, _AnalysisResult], int]


class CanonicalTileError(ValueError):
    """A dimension cannot be divided into legal canonical tiles."""

    def __init__(self, message: str, dimension: str, extent: int, minimum: int) -> None:
        """Record the failed concrete dimension and minimum legal tile."""
        super().__init__(message)
        self.dimension = dimension
        self.extent = extent
        self.minimum = minimum


def canonical_tile_size(rec: _OpRecord, abstract: str, analysis: _AnalysisResult) -> int:
    """Return the widest legal divisor for one canonical operation axis."""
    extent = analysis.dim_sizes[rec.axis_map[abstract]]
    upper = extent if (maximum := rec.op_cls.MAX_TILE_SIZE.get(abstract)) is None else min(extent, maximum)
    if abstract in _partition_axes(rec, analysis):
        upper = min(upper, analysis.partition_tile_limits.get(rec.axis_map[abstract], upper))
    minimum = min(rec.op_cls.MIN_TILE_SIZE.get(abstract, 1), extent, upper)
    step = 128 if rec.op_cls.NAME == "nc_matmul" and abstract == "N" and upper >= 128 else 1
    tile = next(filter(lambda value: extent % value == 0, range(upper - upper % step, minimum - 1, -step)), None)
    if tile is None:
        message = (
            f"{rec.op_cls.__name__}.{abstract} extent {extent} has no canonical tile between {minimum} and {upper}"
        )
        raise CanonicalTileError(message, rec.axis_map[abstract], extent, minimum)
    return tile


def _partition_axes(rec: _OpRecord, analysis: _AnalysisResult) -> set[str]:
    """Return axes that directly address an on-chip partition dimension."""
    return {
        groups[0][0]
        for slot, name in rec.operand_names.items()
        if analysis.tensors[name].location != "shared_hbm"
        and (groups := _groups(rec, slot, analysis))
        and len(groups[0]) == 1
    }


def seed_partition_tile_limits(analysis: _AnalysisResult) -> None:
    """Keep canonical partition widths compatible across connected operations."""
    widths: dict[str, list[int]] = {}
    for rec in analysis.ops:
        for axis in _partition_axes(rec, analysis):
            widths.setdefault(rec.axis_map[axis], []).append(canonical_tile_size(rec, axis, analysis))
    analysis.partition_tile_limits = {dimension: reduce(gcd, tiles) for dimension, tiles in widths.items()}


def canonical_trip_count(rec: _OpRecord, abstract: str, analysis: _AnalysisResult) -> int:
    """Return the loop trip count outside one canonical operation tile."""
    extent = analysis.dim_sizes[rec.axis_map[abstract]]
    return extent // canonical_tile_size(rec, abstract, analysis)


def _sum(terms: list[Expr]) -> Expr:
    """Return an expression sum with a canonical zero identity."""
    return reduce(lambda left, right: Add(left=left, right=right), terms) if terms else Const(value=0)


def _groups(rec: _OpRecord, slot: str, analysis: _AnalysisResult) -> tuple[tuple[str, ...], ...]:
    """Return physical axis groups present in one configured operation."""
    groups = tuple(group for group in rec.op_cls.operand_axis_groups(slot) if set(group) <= rec.axis_map.keys())
    return groups[: len(analysis.tensors[rec.operand_names[slot]].shape)]


def _axis_offset(coefficient: int, divisor: int, loop_var: str, label: str) -> Expr:
    """Normalize one active loop offset and require an exact tile boundary."""
    if coefficient % divisor:
        raise ValueError(f"{label} offset {coefficient} is not divisible by {divisor}")
    variable = Var(name=loop_var)
    return variable if coefficient == divisor else Mul(left=variable, right=Const(value=coefficient // divisor))


def _physical_range(
    rec: _OpRecord,
    slot: str,
    group: tuple[str, ...],
    dimension: int,
    loop_vars: dict[str, str],
    analysis: _AnalysisResult,
    tile_size: TileSize,
    trip_count: TripCount,
) -> tuple[Expr, Expr]:
    """Return one conservative physical range for an axis group."""
    tensor = analysis.tensors[rec.operand_names[slot]]
    extents = tuple(analysis.dim_sizes[rec.axis_map[axis]] for axis in group)
    tiles = tuple(tile_size(rec, axis, analysis) for axis in group)
    strides = tuple(prod(extents[index + 1 :]) for index in range(len(group)))
    span = 1 + sum((tile - 1) * stride for tile, stride in zip(tiles, strides, strict=True))
    contiguous = span == prod(tiles)
    if not contiguous and rec.op_cls.operand_view_axis_groups(slot) is None:
        raise ValueError(f"{rec.op_cls.__name__}.{slot} selects a non-contiguous flattened region")
    divisor = 1
    if dimension == 0 and tensor.location != "shared_hbm":
        divisor = gcd(partition_extent(tensor.shape[0]), span)
        if divisor < 1 or tensor.shape[0] % divisor:
            raise ValueError(f"{rec.op_cls.__name__}.{slot} partition span {span} is not tile-aligned")
    terms: list[Expr] = []
    for axis, stride, tile in zip(group, strides, tiles, strict=True):
        if trip_count(rec, axis, analysis) > 1:
            terms.append(_axis_offset(tile * stride, divisor, loop_vars[axis], f"{rec.op_cls.__name__}.{axis}"))
    return _sum(terms), Const(value=span)


def build_operand_region(
    rec: _OpRecord,
    slot: str,
    loop_vars: dict[str, str],
    analysis: _AnalysisResult,
    tile_size: TileSize,
    trip_count: TripCount,
) -> BufferRegion:
    """Build one logical dependency region from flattened axis groups."""
    tensor = rec.operand_names[slot]
    groups = _groups(rec, slot, analysis)
    if len(groups) != len(analysis.tensors[tensor].shape):
        raise ValueError(f"{rec.op_cls.__name__}.{slot} axis groups do not match tensor shape")
    ranges = tuple(
        _physical_range(rec, slot, group, index, loop_vars, analysis, tile_size, trip_count)
        for index, group in enumerate(groups)
    )
    return BufferRegion(tensor=tensor, ranges=ranges)


def _row_major_strides(axes: tuple[str, ...], extents: dict[str, int], base: int) -> dict[str, int]:
    """Return row-major strides for one abstract axis sequence."""
    return {axis: base * prod(extents[other] for other in axes[index + 1 :]) for index, axis in enumerate(axes)}


def _storage_strides(
    rec: _OpRecord, slot: str, analysis: _AnalysisResult, tile_size: TileSize
) -> tuple[dict[str, int], dict[str, int]]:
    """Return view strides and divisors for loops crossing partition tiles."""
    tensor = analysis.tensors[rec.operand_names[slot]]
    groups = _groups(rec, slot, analysis)
    extents = {axis: analysis.dim_sizes[rec.axis_map[axis]] for group in groups for axis in group}
    if tensor.location == "shared_hbm":
        axes = tuple(axis for group in groups for axis in group)
        return _row_major_strides(axes, extents, 1), {}
    if len(groups) not in {1, 2}:
        raise ValueError(f"{rec.op_cls.__name__}.{slot} on-chip views require rank one or two")
    leading, free = tensor.shape[0], tensor.shape[1] if len(tensor.shape) == 2 else 1
    first = groups[0]
    group_extents = tuple(extents[axis] for axis in first)
    group_tiles = tuple(tile_size(rec, axis, analysis) for axis in first)
    logical_strides = tuple(prod(group_extents[index + 1 :]) for index in range(len(first)))
    span = 1 + sum((tile - 1) * stride for tile, stride in zip(group_tiles, logical_strides, strict=True))
    partition = gcd(partition_extent(leading), span)
    suffix_product, cut = 1, len(first)
    while cut > 0 and suffix_product < partition:
        cut -= 1
        suffix_product *= extents[first[cut]]
    if suffix_product % partition:
        raise ValueError(f"{rec.op_cls.__name__}.{slot} cannot expose its partition axes")
    prefix, suffix = first[:cut], first[cut:]
    strides = _row_major_strides(prefix, extents, free * (suffix_product // partition))
    strides.update(_row_major_strides(suffix, extents, leading // partition * free))
    if len(groups) == 2:
        strides.update(_row_major_strides(groups[1], extents, 1))
    return strides, {first[cut]: leading} if suffix_product != partition else {}


def _view_dimension(
    rec: _OpRecord,
    slot: str,
    group: tuple[str, ...],
    strides: dict[str, int],
    analysis: _AnalysisResult,
    tile_size: TileSize,
) -> tuple[Expr, Expr]:
    """Return one explicit tensor-view stride and extent."""
    active = tuple(axis for axis in group if analysis.dim_sizes[rec.axis_map[axis]] > 1)
    present = tuple(axis for axis in active if axis in strides)
    if present and len(present) != len(active):
        raise ValueError(f"{rec.op_cls.__name__}.{slot} view mixes stored and broadcast axes")
    extent = prod(tile_size(rec, axis, analysis) for axis in group)
    if not present:
        return Const(value=int(extent == 1)), Const(value=extent)
    for left, right in zip(active, active[1:]):
        if strides[left] != strides[right] * analysis.dim_sizes[rec.axis_map[right]]:
            raise ValueError(f"{rec.op_cls.__name__}.{slot} view axes {group} are not contiguous")
    return Const(value=strides[present[-1]]), Const(value=extent)


def build_access_patterns(
    rec: _OpRecord, loop_vars: dict[str, str], analysis: _AnalysisResult, tile_size: TileSize, trip_count: TripCount
) -> dict[str, AccessPattern]:
    """Build configured ISA tensor views for one operation record."""
    patterns: dict[str, AccessPattern] = {}
    for slot in rec.op_cls.OPERAND_AXES:
        view = rec.op_cls.operand_view_axis_groups(slot)
        if view is None or slot not in rec.operand_names:
            continue
        strides, divisors = _storage_strides(rec, slot, analysis, tile_size)
        offset_terms = []
        for axis, stride in strides.items():
            if trip_count(rec, axis, analysis) > 1:
                coefficient = tile_size(rec, axis, analysis) * stride
                label = f"{rec.op_cls.__name__}.{axis}"
                offset_terms.append(_axis_offset(coefficient, divisors.get(axis, 1), loop_vars[axis], label))
        patterns[slot] = AccessPattern(
            pattern=tuple(_view_dimension(rec, slot, group, strides, analysis, tile_size) for group in view),
            offset=_sum(offset_terms),
        )
    return patterns


def access_pattern_allocation_view(access_pattern: AccessPattern, buf: Buffer) -> tuple[Expr, AccessPattern]:
    """Map one logical on-chip tensor view to its physical allocation list entry."""
    if buf.location == "shared_hbm":
        return Const(value=0), access_pattern
    free = buf.per_tile_physical_shape()[2]
    logical_span = buf.tiles_per_list() * free
    full_stride = buf.logical_tile_count() * free
    first_stride, extent = access_pattern.pattern[0]
    valid = first_stride == Const(value=full_stride) or extent == Const(value=1)
    if not valid:
        raise AssertionError(f"{buf.name}: access pattern must expose the on-chip partition axis first")
    pattern = ((Const(value=logical_span * buf.versions), extent), *access_pattern.pattern[1:])
    if buf.list_len == 1:
        return Const(value=0), replace(access_pattern, pattern=pattern)
    divisor = Const(value=logical_span)
    return (
        FloorDiv(left=access_pattern.offset, right=divisor),
        replace(access_pattern, pattern=pattern, offset=Mod(left=access_pattern.offset, right=divisor)),
    )
