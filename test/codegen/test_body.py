"""Tests for body allocation and BufferRegion rendering."""

from __future__ import annotations

from dataclasses import replace
from test.transforms._pipeline_fixtures import m_loop_and_children, parent_block_of, tuned_ir

from nkigym.codegen import render
from nkigym.codegen.body import _emit_alloc, _hoisted_scope, render_access_pattern, render_buffer_region
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.load import NKILoad


def test_root_owned_buffer_does_not_tighten_through_nested_block() -> None:
    """A structural-only move must not silently descend a root-owned allocation."""
    tree = KernelTree()
    buffer = Buffer(name="sbuf_x", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    root = tree.data(tree.root)
    assert isinstance(root, BlockNode)
    tree.graph.nodes[tree.root]["data"] = replace(root, alloc_buffers=(buffer,))
    stage = tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=()), parent=tree.root)
    outer = tree.add_node(ForNode(loop_var="i_d2_0", extent=4), parent=stage)
    inner = tree.add_node(ForNode(loop_var="i_d0_0", extent=16), parent=outer)
    region = BufferRegion(
        tensor="sbuf_x", ranges=((Var(name="i_d0_0"), Const(value=128)), (Const(value=0), Const(value=2048)))
    )
    leaf = tree.add_node(ISANode(op_cls=NKILoad, operand_bindings={"dst": region}), parent=inner)
    assert _hoisted_scope(tree, "sbuf_x", [leaf]) == tree.root


def test_basic_buffer_region_forms() -> None:
    """HBM, SBUF, PSUM, and full-extent regions render in their physical layouts."""
    hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    hbm_region = BufferRegion(
        tensor="hbm_out",
        ranges=(
            (Mul(left=Var(name="i_d0_0"), right=Const(value=128)), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    assert (
        render_buffer_region(hbm_region, hbm)
        == "hbm_out[i_d0_0 * 128:i_d0_0 * 128 + 128, i_d1_0 * 512:i_d1_0 * 512 + 512]"
    )
    full = BufferRegion(
        tensor="hbm_out", ranges=((Const(value=0), Const(value=2048)), (Const(value=0), Const(value=2048)))
    )
    assert render_buffer_region(full, hbm) == "hbm_out[0:0 + 2048, 0:0 + 2048]"

    sbuf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    sbuf_region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    assert render_buffer_region(sbuf_region, sbuf) == "sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"
    psum = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    psum_region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    assert render_buffer_region(psum_region, psum) == "psum_prod[0][0:128, i_d0_0, i_d1_0 * 512:i_d1_0 * 512 + 512]"


def test_version_rotation_and_pipeline_rendering() -> None:
    """Versioned regions rotate and pipeline annotations propagate the loop variable."""
    region = BufferRegion(
        tensor="psum_prod", ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048)))
    )
    versioned = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    rotation = Mod(left=Var(name="i_d1_0"), right=Const(value=2))
    assert render_buffer_region(region, versioned, rotation=rotation) == "psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]"
    packed = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    assert render_buffer_region(region, packed, rotation=None) == "psum_prod[0][0:128, 0, 0:0 + 2048]"

    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    parent = parent_block_of(ir, m_loop)
    ir.tree.block(parent).annotations["software_pipeline"] = {
        "loop_nid": m_loop,
        "stages": (0, 0, 1),
        "order": (0, 1, 2),
        "versioned_buffers": ("psum_prod",),
    }
    source = render(ir)
    assert "psum_prod = [nl.ndarray((128, 2, 2048)" in source
    assert "psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]" in source
    assert "for i_d1_0 in range(16):" in source


def test_versioned_access_pattern_adds_flattened_pipeline_rotation() -> None:
    """A physical access pattern selects the pipeline version in its flat offset."""
    buffer = Buffer(name="sbuf_probability_t", shape=(8192, 128), dtype="bfloat16", location="sbuf", versions=2)
    pattern = AccessPattern(
        pattern=(
            (Const(value=16384), Const(value=128)),
            (Const(value=1), Const(value=1)),
            (Const(value=128), Const(value=4)),
            (Const(value=1), Const(value=128)),
        ),
        offset=Mul(left=Var(name="group"), right=Const(value=512)),
    )
    rotation = Mul(left=Mod(left=Var(name="step"), right=Const(value=2)), right=Const(value=64))
    assert render_access_pattern("sbuf_probability_t", pattern, buffer, rotation) == (
        "sbuf_probability_t[0].ap("
        "pattern=[[16384, 128], [1, 1], [128, 4], [1, 128]], "
        "offset=group * 512 + step % 2 * 64 * 128)"
    )


def test_nested_pipeline_does_not_replace_outer_buffer_rotation() -> None:
    """A nested pipeline rotates only buffers that it versioned."""
    ir = tuned_ir()
    outer_loop, children = m_loop_and_children(ir)
    inner_loop = children[1]
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    ir.tree.block(ir.tree.root).annotations["software_pipeline"] = {
        "loop_nid": outer_loop,
        "stages": (0, 0, 1),
        "order": (0, 1, 2),
        "versioned_buffers": ("psum_prod",),
    }
    parent = parent_block_of(ir, outer_loop)
    ir.tree.block(parent).annotations["software_pipeline"] = {
        "loop_nid": inner_loop,
        "stages": (0,),
        "order": (0,),
        "versioned_buffers": (),
    }
    source = render(ir)
    assert "psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]" in source
    assert "psum_prod[0][0:128, i_d2_0 % 2, 0:0 + 2048]" not in source


def test_allocation_forms() -> None:
    """On-chip buffers use lists while shared HBM remains a bare allocation."""
    listed = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=16)
    assert (
        _emit_alloc(listed)
        == "sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]"
    )
    packed = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert (
        _emit_alloc(packed)
        == "sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"
    )
    composed = Buffer(
        name="sbuf_versioned", shape=(2048, 512), dtype="bfloat16", location="sbuf", versions=2, list_len=8
    )
    assert (
        _emit_alloc(composed)
        == "sbuf_versioned = [nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]"
    )
    hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    assert _emit_alloc(hbm) == "hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)"
    hbm_vector = replace(hbm, shape=(128,))
    assert _emit_alloc(hbm_vector) == "hbm_out = nl.ndarray((128,), dtype=nl.bfloat16, buffer=nl.shared_hbm)"


def test_list_tile_region_forms() -> None:
    """Packed and fully split lists place logical tile indices correctly."""
    packed = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    packed_region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    assert (
        render_buffer_region(packed_region, packed) == "sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"
    )
    listed = Buffer(name="psum_prod", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    listed_region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d1_0"), Const(value=128)),
            (Mul(left=Var(name="i_d2_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    assert render_buffer_region(listed_region, listed) == "psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512:i_d2_0 * 512 + 512]"


def test_general_list_factorization_parenthesizes_tile_indices() -> None:
    """Lists derive their list and local indices from logical tiles before rotation."""
    buffer = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=8)
    assert buffer.per_tile_physical_shape() == (128, 2, 512)
    bare = BufferRegion(tensor="s", ranges=((Var(name="t"), Const(value=128)), (Const(value=0), Const(value=512))))
    assert render_buffer_region(bare, buffer) == "s[(t) // 2][0:128, (t) % 2, 0:0 + 512]"
    tile = Add(left=Mul(left=Var(name="i_d1_0"), right=Const(value=4)), right=Var(name="i_d1_1"))
    compound = BufferRegion(tensor="s", ranges=((tile, Const(value=128)), (Const(value=0), Const(value=512))))
    assert (
        render_buffer_region(compound, buffer)
        == "s[(i_d1_0 * 4 + i_d1_1) // 2][0:128, (i_d1_0 * 4 + i_d1_1) % 2, 0:0 + 512]"
    )
    versioned = replace(buffer, versions=2)
    rotation = Mul(left=Mod(left=Var(name="step"), right=Const(value=2)), right=Const(value=2))
    assert versioned.per_tile_physical_shape() == (128, 4, 512)
    assert render_buffer_region(bare, versioned, rotation) == "s[(t) // 2][0:128, (t) % 2 + step % 2 * 2, 0:0 + 512]"
    fully_split = replace(versioned, list_len=16)
    unit_rotation = Mod(left=Var(name="step"), right=Const(value=2))
    assert render_buffer_region(bare, fully_split, unit_rotation) == "s[t][0:128, step % 2, 0:0 + 512]"
