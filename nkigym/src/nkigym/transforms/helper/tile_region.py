"""Retile buffer regions after Split and Fuse rewrites."""

from __future__ import annotations

from collections.abc import Callable

from nkigym.ir.arith.expr import Const, Expr
from nkigym.ir.tree import BufferRegion


def retile_region(
    region: BufferRegion,
    axes: tuple[str, ...],
    abstract_axis: str | None,
    rewrite: Callable[[Expr, int], tuple[Expr, int]],
) -> BufferRegion:
    """Apply a width rewrite to the region range for one abstract axis."""
    if abstract_axis is None or abstract_axis not in axes:
        return region
    idx = axes.index(abstract_axis)
    if idx >= len(region.ranges):
        return region
    lo, width = region.ranges[idx]
    assert isinstance(width, Const), f"region width must be Const; got {width!r}"
    new_lo, new_width = rewrite(lo, width.value)
    new_ranges = list(region.ranges)
    new_ranges[idx] = (new_lo, Const(value=new_width))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))
