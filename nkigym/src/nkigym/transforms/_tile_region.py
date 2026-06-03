"""Tile-width re-write for tensorize Split / Fuse.

A buffer-region axis is ``(lo, width)`` where ``lo`` is an affine
combination of loop vars in ELEMENT space and a loop iterating a tile of
``width`` elements strides by ``width``. Tensorize Split/Fuse only need to
set a region axis's tile ``width``; the element-space ``lo`` offset is
recomputed by :func:`nkigym.transforms._normalize.normalize_block` from the
surrounding loop structure. :func:`retile_region` applies a caller-supplied
width rewrite to the axis matching the operand's abstract axis name.
"""

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
    """Apply ``rewrite(lo, width) -> (new_lo, new_width)`` to ``region``'s range on
    ``abstract_axis``, returning the region unchanged if ``axes`` does not contain it.

    ``axes`` is the THIS operand's ``OPERAND_AXES`` tuple (the caller passes the slot's
    axes). The range index for ``abstract_axis`` is its position in ``axes`` — correct
    per-operand (matmul ``stationary`` (K,M) lacks N, so an N-retile no-ops on it).
    """
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
