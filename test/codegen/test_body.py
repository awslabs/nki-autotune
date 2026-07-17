"""Tests for nkigym.codegen.body BufferRegion rendering."""

from __future__ import annotations

from dataclasses import replace
from test.transforms._pipeline_fixtures import m_loop_and_children, parent_block_of, tuned_ir

from nkigym.codegen import render
from nkigym.codegen.body import _emit_alloc, _hoisted_scope, render_buffer_region
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.load import NKILoad


def test_root_owned_buffer_does_not_tighten_through_nested_block():
    """A structural-only move must not silently descend a root-owned allocation."""
    tree = KernelTree()
    buf = Buffer(name="sbuf_x", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    root = tree.data(tree.root)
    assert isinstance(root, BlockNode)
    tree.graph.nodes[tree.root]["data"] = replace(root, alloc_buffers=(buf,))

    stage = tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=()), parent=tree.root)
    outer = tree.add_node(ForNode(loop_var="i_d2_0", extent=4), parent=stage)
    inner = tree.add_node(ForNode(loop_var="i_d0_0", extent=16), parent=outer)
    region = BufferRegion(
        tensor="sbuf_x", ranges=((Var(name="i_d0_0"), Const(value=128)), (Const(value=0), Const(value=2048)))
    )
    leaf = tree.add_node(ISANode(op_cls=NKILoad, operand_bindings={"dst": region}), parent=inner)

    assert _hoisted_scope(tree, "sbuf_x", [leaf]) == tree.root


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
    assert out == "sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"


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
    assert out == "psum_prod[0][0:128, i_d0_0, i_d1_0 * 512:i_d1_0 * 512 + 512]"


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
        tensor="psum_prod", ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048)))
    )
    out = render_buffer_region(region, buf, rotation=Mod(left=Var(name="i_d1_0"), right=Const(value=2)))
    assert out == "psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]"


def test_render_region_no_rotation_when_versions_one():
    """versions=1 (rotation=None) renders byte-identically to today."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    region = BufferRegion(
        tensor="psum_prod", ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048)))
    )
    assert render_buffer_region(region, buf, rotation=None) == "psum_prod[0][0:128, 0, 0:0 + 2048]"


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
    ir.tree.block(parent).annotations["software_pipeline"] = {
        "loop_nid": m_loop,
        "stages": (0, 0, 1),
        "order": (0, 1, 2),
    }
    src = render(ir)
    assert "psum_prod = [nl.ndarray((128, 2, 2048)" in src
    assert "psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]" in src
    assert "for i_d1_0 in range(16):" in src


def test_emit_alloc_list_of_tiles():
    """list_len>1 emits a Python list comprehension of per-tile ndarrays."""
    buf = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=16)
    out = _emit_alloc(buf)
    assert out == "sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]"


def test_emit_alloc_packed_unchanged():
    """list_len==1 emits the single packed ndarray, byte-identical to today."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    out = _emit_alloc(buf)
    assert out == "sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"


def test_render_buffer_region_list_of_tiles():
    """list_len>1 peels the tile index into a leading list subscript, middle index 0."""
    buf = Buffer(name="psum_prod", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d1_0"), Const(value=128)),
            (Mul(left=Var(name="i_d2_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512:i_d2_0 * 512 + 512]"


def test_render_general_a_gt_1_bare_tile_index():
    """a>1 (list_len=8 on T=16 => a=2) with a bare Var tile index renders
    (t) // a leading, (t) % a middle."""
    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=8)
    assert buf.per_tile_physical_shape() == (128, 2, 512)
    region = BufferRegion(tensor="s", ranges=((Var(name="t"), Const(value=128)), (Const(value=0), Const(value=512))))
    out = render_buffer_region(region, buf)
    assert out == "s[(t) // 2][0:128, (t) % 2, 0:0 + 512]"


def test_render_general_a_gt_1_compound_tile_index_parenthesized():
    """a>1 with a COMPOUND tile index (i_d1_0 * 4 + i_d1_1, the post-Split M axis) must
    parenthesize before // and % — else Python precedence gives (i*4)+(j//a), which is
    wrong (and OOB). Guards the operator-precedence bug the ladder cannot catch (it uses
    only a==1/b==1)."""
    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=8)
    assert buf.per_tile_physical_shape() == (128, 2, 512)
    tile = Add(left=Mul(left=Var(name="i_d1_0"), right=Const(value=4)), right=Var(name="i_d1_1"))
    region = BufferRegion(tensor="s", ranges=((tile, Const(value=128)), (Const(value=0), Const(value=512))))
    out = render_buffer_region(region, buf)
    assert out == "s[(i_d1_0 * 4 + i_d1_1) // 2][0:128, (i_d1_0 * 4 + i_d1_1) % 2, 0:0 + 512]"


def test_emit_alloc_b1_is_list_of_one():
    """A list_len==1 sbuf buffer emits a list-of-1, not a bare ndarray."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    out = _emit_alloc(buf)
    assert out == "sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"


def test_render_b1_multi_tile_is_list0_whole_index():
    """list_len==1 (T=16) renders buf[0][0:128, i_d0_0, F] — list index 0, whole tile index."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"


def test_render_full_split_is_list_index_middle_zero():
    """list_len==T (a==1) renders buf[t][0:128, 0, F] — the k6 full-split form."""
    buf = Buffer(name="psum_prod", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d1_0"), Const(value=128)),
            (Mul(left=Var(name="i_d2_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512:i_d2_0 * 512 + 512]"


def test_emit_alloc_hbm_stays_bare():
    """shared_hbm keeps its bare 2D ndarray (no tile axis, never listed)."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    out = _emit_alloc(buf)
    assert out == "hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)"
