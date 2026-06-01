"""Unit tests for the tile-width re-write helper."""

from __future__ import annotations

from collections.abc import Callable

from nkigym.ir.expr import Const, Expr, Mul, Var, format_expr
from nkigym.ir.tree import BufferRegion
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.transforms._tile_region import retile_region


def _set_width(new_width: int) -> Callable[[Expr, int], tuple[Expr, int]]:
    """Width-only rewrite matching the closures split.py / fuse.py pass to retile_region:
    keep the offset (normalize recomputes it), set the new tile width."""

    def rewrite(lo: Expr, _width: int) -> tuple[Expr, int]:
        return lo, new_width

    return rewrite


def test_retile_region_sets_width_on_the_matching_axis_only():
    """retile_region finds the F-axis range on a load dst (P,F) region and sets its width;
    the P axis range is returned unchanged."""
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=2048)), Const(value=2048)),
        ),
    )
    out = retile_region(region, NKILoad.OPERAND_AXES["dst"], "F", _set_width(128))
    assert out.ranges[1][1].value == 128
    """Offset preserved (normalize recomputes it); P axis untouched."""
    assert format_expr(out.ranges[1][0]) == "i_d1_0 * 2048"
    assert format_expr(out.ranges[0][0]) == "i_d0_0" and out.ranges[0][1].value == 128


def test_retile_region_noops_on_operand_lacking_axis():
    """An operand whose axes lack the target axis is returned unchanged (matmul
    stationary (K,M) on an N-retile must NOT be touched — the bug that corrupted M)."""
    stat = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    out = retile_region(stat, NKIMatmul.OPERAND_AXES["stationary"], "N", _set_width(512))
    assert out.ranges[1][1].value == 128, "stationary M width must stay 128 (no N axis)"
    assert format_expr(out.ranges[1][0]) == "i_d1_0 * 128"


def test_retile_region_noops_when_axis_is_none():
    """A ``None`` abstract axis (operand-axis lookup miss) returns the region unchanged."""
    region = BufferRegion(
        tensor="sbuf_lhs_T", ranges=((Mul(left=Var(name="i_d1_0"), right=Const(value=2048)), Const(value=2048)),)
    )
    out = retile_region(region, NKILoad.OPERAND_AXES["dst"], None, _set_width(128))
    assert out.ranges[0][1].value == 2048
    assert format_expr(out.ranges[0][0]) == "i_d1_0 * 2048"
