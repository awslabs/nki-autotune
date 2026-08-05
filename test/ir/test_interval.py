"""Tests for affine interval and buffer-region disjointness."""

from __future__ import annotations

from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.interval import AffineInterval, intervals_disjoint, regions_disjoint
from nkigym.ir.tree import Buffer, BufferRegion


def _interval(coefficients: dict[str | None, int], width: int) -> AffineInterval:
    """Build one affine interval."""
    return AffineInterval(coeffs=coefficients, width=width)


def test_affine_interval_disjointness_cases() -> None:
    """Interval analysis handles overlap, separation, independent variables, and reversal."""
    extents = {"i_m": 16}
    assert not intervals_disjoint(_interval({"i_m": 128}, 128), _interval({"i_m": 128}, 128), extents)
    assert intervals_disjoint(_interval({"i_m": 128, None: 0}, 128), _interval({"i_m": 128, None: 128}, 128), extents)
    assert intervals_disjoint(_interval({"i_m": 128, None: 0}, 128), _interval({"i_m": 128, None: 256}, 128), extents)
    assert not intervals_disjoint(_interval({"i_m": 128}, 128), _interval({"i_n": 128}, 128), {"i_m": 16, "i_n": 16})
    assert not intervals_disjoint(_interval({"i_m": -128}, 128), _interval({None: 0}, 128), extents)
    assert intervals_disjoint(_interval({None: 0}, 128), _interval({None: 128}, 128), {})
    assert not intervals_disjoint(_interval({None: 0}, 128), _interval({None: 64}, 128), {})


def test_buffer_region_disjointness_cases() -> None:
    """Regions are disjoint when any normalized axis is disjoint."""
    hbm = Buffer(name="t", shape=(2048, 2048), dtype="float32", location="shared_hbm")
    first = BufferRegion(
        tensor="t",
        ranges=(
            (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
            (Const(value=0), Const(value=512)),
        ),
    )
    second = BufferRegion(
        tensor="t",
        ranges=(
            (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
            (Const(value=512), Const(value=512)),
        ),
    )
    assert regions_disjoint(first, second, hbm, hbm, {"i": 16})
    assert not regions_disjoint(first, first, hbm, hbm, {"i": 16})

    psum = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    partition_first = BufferRegion(
        tensor="p", ranges=((Var(name="i_m"), Const(value=128)), (Const(value=0), Const(value=512)))
    )
    partition_second = BufferRegion(
        tensor="p", ranges=((Var(name="i_m"), Const(value=128)), (Const(value=512), Const(value=512)))
    )
    assert regions_disjoint(partition_first, partition_second, psum, psum, {"i_m": 16})

    first_batch = BufferRegion(
        tensor="p", ranges=((Const(value=0), Const(value=512)), (Const(value=0), Const(value=512)))
    )
    second_batch = BufferRegion(
        tensor="p", ranges=((Const(value=4), Const(value=512)), (Const(value=0), Const(value=512)))
    )
    assert regions_disjoint(first_batch, second_batch, psum, psum, {})
