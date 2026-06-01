"""Tests for codegen.compact — buffer shape bounding-box + index rebase."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen.compact import compact_shapes


def test_rebased_region_canonical_unchanged():
    """On canonical IR, rebased_region is identity (no buffer declared under loops)."""
    from nkigym.codegen.compact import rebased_region
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                buf = ir.buffer(region.tensor)
                assert rebased_region(region, buf, ir.tree).ranges == region.ranges


def test_compact_shapes_canonical_is_noop():
    """On canonical IR (buffers at/near root, no anchor loops above declaration),
    compact_shapes leaves every logical Buffer.shape unchanged."""
    ir = build_canonical_ir()
    before = {b.name: b.shape for b in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    after = {b.name: b.shape for b in ir.all_buffers().values()}
    assert before == after


def test_compact_shapes_idempotent():
    """compact_shapes applied twice equals once."""
    ir = build_canonical_ir()
    compact_shapes(ir.tree)
    once = {b.name: b.shape for b in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    twice = {b.name: b.shape for b in ir.all_buffers().values()}
    assert once == twice


def test_compact_shapes_uses_per_leaf_extents_not_global():
    """A loop_var reused with different extents across subtrees must not inflate a
    buffer whose touching region lives in the small-extent subtree (regression: a
    flat tree-global extent map applied the matmul's i_d1_0=16 to the load's
    i_d1_0=1, ballooning sbuf_lhs_T axis-1 from 2048 to 32768)."""
    ir = build_canonical_ir()
    compact_shapes(ir.tree)
    bufs = {b.name: b.shape for b in ir.all_buffers().values()}
    assert bufs["sbuf_lhs_T"] == (2048, 2048), bufs["sbuf_lhs_T"]


def test_emit_alloc_follows_compacted_shape():
    """After compact_shapes writes a smaller logical shape, _emit_alloc emits it
    (no emitter change — physical_shape expands the compacted logical shape)."""
    from dataclasses import replace

    from nkigym.codegen.body import _emit_alloc
    from nkigym.ir.tree import Buffer

    full = Buffer(name="sbuf_x", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    compacted = replace(full, shape=(128, 128))
    assert _emit_alloc(full) == "sbuf_x = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)"
    assert _emit_alloc(compacted) == "sbuf_x = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)"
