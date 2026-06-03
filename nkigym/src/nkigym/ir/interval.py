"""Affine integer-interval disjointness for buffer-region overlap analysis.

An :class:`AffineInterval` is a half-open integer range
``[base, base + width)`` where ``base`` is an affine combination of
loop-var symbols (``coeffs`` in :func:`nkigym.ir.arith.expr.to_affine` form)
and ``width`` is a constant. Two intervals on the same axis are
*provably disjoint* iff the integer range of ``a.base - b.base`` over
the loop-var box (each var in ``[0, extent)``) cannot fall in the open
overlap window ``(-b.width, a.width)``.

Soundness: when the difference range straddles the window (e.g. two
independent loop vars), we conservatively report *not disjoint* — never
a false "disjoint", so dependency edges are never dropped incorrectly.
"""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir.arith.expr import Const, to_affine
from nkigym.ir.tree import PARTITION_DIM, Buffer, BufferRegion


@dataclass(frozen=True, kw_only=True)
class AffineInterval:
    """Half-open integer interval ``[base, base + width)``.

    Attributes:
        coeffs: affine coefficients of ``base`` (``{var: coeff, ..., None: const}``).
        width: constant tile width.
    """

    coeffs: dict[str | None, int]
    width: int


def intervals_disjoint(a: AffineInterval, b: AffineInterval, loop_extents: dict[str, int]) -> bool:
    """Return True iff ``a`` and ``b`` are disjoint for EVERY assignment of the loop vars.

    ``loop_extents`` maps each loop-var name to its trip count (the var
    ranges over ``[0, extent)``). Overlap requires
    ``-b.width < (a.base - b.base) < a.width``. We compute the integer
    range of ``a.base - b.base`` over the box and check it cannot intersect
    that open window.
    """
    diff = _sub(a.coeffs, b.coeffs)
    lo, hi = _affine_range(diff, loop_extents)
    """Open overlap window: (-b.width, a.width). Overlap iff lo < a.width and hi > -b.width."""
    overlaps = lo < a.width and hi > -b.width
    return not overlaps


def _sub(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise ``a - b``."""
    out = dict(a)
    for var, coeff in b.items():
        out[var] = out.get(var, 0) - coeff
    return out


def _affine_range(coeffs: dict[str | None, int], loop_extents: dict[str, int]) -> tuple[int, int]:
    """Integer range ``[lo, hi]`` of an affine expression over the loop-var box.

    Each var ``v`` ranges over ``[0, extent_v - 1]``. A var with no known
    extent is treated as unbounded → returns a window-spanning range so the
    caller conservatively reports overlap.
    """
    lo = coeffs.get(None, 0)
    hi = coeffs.get(None, 0)
    for var, coeff in coeffs.items():
        if var is None:
            continue
        if var not in loop_extents:
            """Unknown extent — unbounded; force overlap by returning a maximal range."""
            return (-(1 << 62), (1 << 62))
        span = coeff * (loop_extents[var] - 1)
        if span >= 0:
            hi += span
        else:
            lo += span
    return (lo, hi)


def regions_disjoint(
    a: BufferRegion, b: BufferRegion, buf_a: Buffer, buf_b: Buffer, loop_extents: dict[str, int]
) -> bool:
    """Return True iff two same-tensor regions are provably disjoint.

    Disjoint iff ANY axis is provably disjoint (box intersection empty).
    Each axis range ``(lo, width_const)`` becomes an :class:`AffineInterval`;
    the SBUF/PSUM partition axis (axis 0) is normalised from bare tile-index
    to element space.
    """
    if len(a.ranges) != len(b.ranges):
        return False
    disjoint_on_some_axis = False
    for axis_index, (range_a, range_b) in enumerate(zip(a.ranges, b.ranges)):
        iv_a = _interval_for_axis(range_a, axis_index, buf_a)
        iv_b = _interval_for_axis(range_b, axis_index, buf_b)
        if intervals_disjoint(iv_a, iv_b, loop_extents):
            disjoint_on_some_axis = True
            break
    return disjoint_on_some_axis


def _interval_for_axis(axis_range: tuple, axis_index: int, buf: Buffer) -> AffineInterval:
    """Build an element-space :class:`AffineInterval` from one ``(lo, width_const)`` range.

    Axis 0 of SBUF/PSUM buffers carries a bare tile-index ``lo`` and
    width 128; convert to element space (multiply base coeffs by 128,
    width = 128) so every axis shares a coordinate system.
    """
    lo_expr, width_expr = axis_range
    if not isinstance(width_expr, Const):
        raise ValueError(f"region width must be Const; got {width_expr!r}")
    base = to_affine(lo_expr)
    width = width_expr.value
    is_partition = axis_index == 0 and buf.location in ("sbuf", "psum") and width == PARTITION_DIM
    if is_partition:
        """bare tile index -> element space: base *= 128, width stays 128."""
        base = {var: coeff * PARTITION_DIM for var, coeff in base.items()}
    return AffineInterval(coeffs=base, width=width)


__all__ = ["AffineInterval", "intervals_disjoint", "regions_disjoint"]
