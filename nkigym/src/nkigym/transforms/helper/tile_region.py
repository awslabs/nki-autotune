"""Retile buffer regions after Split and Fuse rewrites."""

from __future__ import annotations

from collections.abc import Callable

from nkigym.ir.arith.expr import Const, Expr
from nkigym.ir.tree import BufferRegion


def retile_region(
    region: BufferRegion,
    axis_groups: tuple[tuple[str, ...], ...],
    abstract_axis: str | None,
    rewrite: Callable[[Expr, int], tuple[Expr, int]],
) -> BufferRegion:
    """Apply a width rewrite to the region range for one abstract axis."""
    if abstract_axis is None:
        return region
    index = next((i for i, group in enumerate(axis_groups) if abstract_axis in group), None)
    if index is None or index >= len(region.ranges):
        return region
    lo, width = region.ranges[index]
    assert isinstance(width, Const), f"region width must be Const; got {width!r}"
    new_lo, new_width = rewrite(lo, width.value)
    new_ranges = list(region.ranges)
    new_ranges[index] = (new_lo, Const(value=new_width))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))
