"""Tests for buffer shape compaction and index rebasing."""

from __future__ import annotations

from dataclasses import replace
from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen.body import _emit_alloc
from nkigym.codegen.compact import _clamp_list_len_to_tiles, compact_shapes, rebased_region
from nkigym.ir.tree import Buffer, ISANode


def test_canonical_compaction_is_stable_and_uses_local_extents() -> None:
    """Canonical regions remain unchanged and repeated compaction is idempotent."""
    ir = build_canonical_ir()
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                buffer = ir.buffer(region.tensor)
                assert rebased_region(region, buffer, ir.tree).ranges == region.ranges

    before = {buffer.name: buffer.shape for buffer in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    once = {buffer.name: buffer.shape for buffer in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    twice = {buffer.name: buffer.shape for buffer in ir.all_buffers().values()}
    assert before == once
    assert once == twice
    assert once["sbuf_lhs_T"] == (2048, 2048)


def test_allocation_follows_compacted_shape() -> None:
    """Allocation rendering expands the compacted logical shape."""
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


def test_compaction_clamps_list_length_to_remaining_tiles() -> None:
    """A compacted list buffer cannot retain more lists than logical tiles."""
    listed = Buffer(name="psum_x", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    assert listed.physical_shape() == (128, 16, 512)
    shrunk = replace(listed, shape=(128, 512))
    fixed = _clamp_list_len_to_tiles(shrunk)
    assert fixed.list_len == 1
    assert fixed.per_tile_physical_shape() == (128, 1, 512)
