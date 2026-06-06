"""Tests for nkigym.codegen.body BufferRegion rendering."""

from __future__ import annotations

from test.transforms._pipeline_fixtures import m_loop_and_children, parent_block_of, tuned_ir

from nkigym.codegen import render
from nkigym.codegen.body import render_buffer_region
from nkigym.ir.arith.expr import Const, Mod, Mul, Var
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


def test_render_region_rotation_applied():
    """A versions>1 psum buffer rotates the tile-axis index by loop_var % versions."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048))),
    )
    out = render_buffer_region(region, buf, rotation=Mod(left=Var(name="i_d1_0"), right=Const(value=2)))
    assert out == "psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048]"


def test_render_region_no_rotation_when_versions_one():
    """versions=1 (rotation=None) renders byte-identically to today."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    region = BufferRegion(
        tensor="psum_prod",
        ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048))),
    )
    assert render_buffer_region(region, buf, rotation=None) == "psum_prod[0:128, 0, 0:0 + 2048]"


def test_emit_pipeline_annotation_rotates_monolithic_loop():
    """A loop whose parent block carries a software_pipeline annotation emits a
    monolithic loop with every versions>1 access rotated by loop_var % versions.

    ``versions`` is set directly here via ``object.__setattr__`` (Buffer is
    frozen) to isolate the renderer; Task 4 sets it through the transform.
    """
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    parent = parent_block_of(ir, m_loop)
    ir.tree.data(parent).annotations["software_pipeline"] = {
        "loop_nid": m_loop,
        "stages": (0, 0, 1),
        "order": (0, 1, 2),
    }
    src = render(ir)
    assert "psum_prod = nl.ndarray((128, 2, 2048)" in src
    assert "psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048]" in src
    assert "for i_d1_0 in range(16):" in src
