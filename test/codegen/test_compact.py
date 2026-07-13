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
    assert (
        _emit_alloc(full)
        == "sbuf_x = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"
    )
    assert (
        _emit_alloc(compacted)
        == "sbuf_x = [nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"
    )


def test_compact_shapes_shrinks_list_len_when_tile_axis_collapses():
    """When a list buffer's leading (tile-count) axis compacts below its list_len,
    compact_shapes shrinks list_len to match — it does not leave list_len > T and
    trip per_tile_physical_shape.

    Models the RFactor psum: a list-16 (2048,512) buffer whose live footprint drops to
    one 128-row tile. After compaction the buffer must be list-1 (128,512), consistent
    under per_tile_physical_shape."""
    from dataclasses import replace

    from nkigym.codegen.compact import _compact_one
    from nkigym.ir.tree import Buffer

    listed = Buffer(name="psum_x", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    assert listed.physical_shape() == (128, 16, 512)
    shrunk = replace(listed, shape=(128, 512))
    """physical tile-count T is now 1; list_len 16 would not divide it."""
    fixed = _compact_one_list_len(shrunk)
    assert fixed.list_len == 1
    assert fixed.per_tile_physical_shape() == (128, 1, 512)


def _compact_one_list_len(buf):
    """Local helper mirroring the shrink _compact_one now performs on list_len, so the
    unit test pins the clamp rule directly on a Buffer (no tree needed)."""
    from nkigym.codegen.compact import _clamp_list_len_to_tiles

    return _clamp_list_len_to_tiles(buf)
