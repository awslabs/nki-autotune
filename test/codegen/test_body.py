"""Tests for nkigym.codegen.body BufferRegion rendering."""

from __future__ import annotations

from nkigym.codegen.body import render_buffer_region
from nkigym.ir.expr import Const, Mul, Var
from nkigym.ir.tree import Buffer, BufferRegion


def test_render_hbm_2d_region():
    """An HBM 2D region renders as flat ``[lo:hi, lo:hi]``."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    region = BufferRegion(
        tensor="hbm_out",
        ranges=(
            (Mul(left=Var(name="i_d0_0"), right=Const(value=128)), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "hbm_out[i_d0_0 * 128:i_d0_0 * 128 + 128, i_d1_0 * 512:i_d1_0 * 512 + 512]"


def test_render_sbuf_3d_region_partition_axis_split():
    """An SBUF 3D region splits the partition axis: [0:128, P_coord, F_lo:F_hi]."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"


def test_render_psum_3d_region_partition_axis_split():
    """A PSUM region (also 3D) splits the partition axis the same way."""
    buf = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[0:128, i_d0_0, i_d1_0 * 512:i_d1_0 * 512 + 512]"


def test_render_constant_zero_origin_for_full_extent_axis():
    """When the lo expression is a bare zero Const, the rendered slice starts at 0 explicitly."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    region = BufferRegion(
        tensor="hbm_out", ranges=((Const(value=0), Const(value=2048)), (Const(value=0), Const(value=2048)))
    )
    out = render_buffer_region(region, buf)
    assert out == "hbm_out[0:0 + 2048, 0:0 + 2048]"
