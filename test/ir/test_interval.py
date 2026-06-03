"""Tests for nkigym.ir.interval affine interval disjointness."""

from __future__ import annotations

from nkigym.ir.interval import AffineInterval, intervals_disjoint


def _iv(coeffs, width):
    return AffineInterval(coeffs=coeffs, width=width)


def test_same_var_same_tile_overlaps():
    """i_m*128 vs i_m*128, width 128 each -> difference 0 in (-128,128) -> overlap."""
    a = _iv({"i_m": 128}, 128)
    b = _iv({"i_m": 128}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16})


def test_same_var_adjacent_tiles_disjoint():
    """i_m*128 vs i_m*128+128 -> difference -128, not in (-128,128) -> disjoint."""
    a = _iv({"i_m": 128, None: 0}, 128)
    b = _iv({"i_m": 128, None: 128}, 128)
    assert intervals_disjoint(a, b, {"i_m": 16})


def test_same_var_gap_disjoint():
    """offset 256 apart, width 128 -> disjoint."""
    a = _iv({"i_m": 128, None: 0}, 128)
    b = _iv({"i_m": 128, None: 256}, 128)
    assert intervals_disjoint(a, b, {"i_m": 16})


def test_different_vars_overlap_sound_fallback():
    """i_m*128 vs i_n*128 (independent vars) -> difference ranges across 0 -> overlap."""
    a = _iv({"i_m": 128}, 128)
    b = _iv({"i_n": 128}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16, "i_n": 16})


def test_negative_coefficient_range():
    """Reversed-iteration base (negative coeff) still bounded correctly.
    a.base = -i_m*128 (ranges [-1920, 0]); b.base = 0. diff = -i_m*128 in [-1920,0].
    Overlap window (-128, 128) intersects [-1920,0] at [-127,0] -> overlap."""
    a = _iv({"i_m": -128}, 128)
    b = _iv({None: 0}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16})


def test_constant_only_intervals():
    """Two constant intervals far apart -> disjoint; touching -> disjoint (half-open); overlapping -> overlap."""
    assert intervals_disjoint(_iv({None: 0}, 128), _iv({None: 128}, 128), {})
    assert not intervals_disjoint(_iv({None: 0}, 128), _iv({None: 64}, 128), {})


def test_regions_disjoint_multi_axis_one_disjoint():
    """Disjoint on axis 1 (different constant tiles), same on axis 0 -> regions disjoint."""
    from nkigym.ir.arith.expr import Const, Mul, Var
    from nkigym.ir.interval import regions_disjoint
    from nkigym.ir.tree import Buffer, BufferRegion

    buf = Buffer(name="t", shape=(2048, 2048), dtype="float32", location="shared_hbm")
    a = BufferRegion(
        tensor="t",
        ranges=(
            (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
            (Const(value=0), Const(value=512)),
        ),
    )
    b = BufferRegion(
        tensor="t",
        ranges=(
            (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
            (Const(value=512), Const(value=512)),
        ),
    )
    assert regions_disjoint(a, b, buf, buf, {"i": 16})


def test_regions_overlap_all_axes():
    """Same indices on every axis -> overlap."""
    from nkigym.ir.arith.expr import Const, Mul, Var
    from nkigym.ir.interval import regions_disjoint
    from nkigym.ir.tree import Buffer, BufferRegion

    buf = Buffer(name="t", shape=(2048, 2048), dtype="float32", location="shared_hbm")
    r = BufferRegion(
        tensor="t",
        ranges=(
            (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
            (Const(value=0), Const(value=512)),
        ),
    )
    assert not regions_disjoint(r, r, buf, buf, {"i": 16})


def test_regions_partition_axis_normalised():
    """SBUF partition axis 0 uses bare tile index; different tiles must be disjoint.
    a: psum[i_m, ...]  b: psum[i_m', ...] would overlap (independent vars), but
    a: psum tile index Var i_m vs constant-offset is the canonical case. Here test
    that bare-index axis-0 with the SAME var but the renderer's bare-Var encoding
    is normalised: tile i_m and tile i_m are same -> overlap; tile i_m vs i_m via
    different vars -> overlap (sound). The normalisation must not crash and must
    treat width-128 partition tiles in element space."""
    from nkigym.ir.arith.expr import Const, Var
    from nkigym.ir.interval import regions_disjoint
    from nkigym.ir.tree import Buffer, BufferRegion

    buf = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    """Axis 0 is bare-Var partition index (i_m), axis 1 is element offset."""
    a = BufferRegion(tensor="p", ranges=((Var(name="i_m"), Const(value=128)), (Const(value=0), Const(value=512))))
    b = BufferRegion(tensor="p", ranges=((Var(name="i_m"), Const(value=128)), (Const(value=512), Const(value=512))))
    """Same partition tile, disjoint free-axis tiles -> disjoint overall."""
    assert regions_disjoint(a, b, buf, buf, {"i_m": 16})
