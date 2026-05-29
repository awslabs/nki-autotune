"""Tests for nkigym.ir.buffer_placement.place_buffers."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.tree import BlockNode


def _block_for_op(ir, op_name):
    from nkigym.ir.tree import ISANode as _ISA

    for nid in ir.tree.blocks():
        leaves = list(ir.tree.leaves(nid))
        if len(leaves) == 1:
            leaf_data = ir.tree.data(leaves[0])
            if isinstance(leaf_data, _ISA) and leaf_data.op_cls.__name__ == op_name:
                return nid
    raise AssertionError(f"no block for {op_name}")


def test_place_buffers_canonical_matches_build():
    """place_buffers on the canonical tree reproduces the build-time placement:
    multi-toucher buffers at root, store-only hbm_out at the store block."""
    ir = build_canonical_ir()
    root = ir.tree.root
    store_nid = _block_for_op(ir, "NKIStore")

    """Capture placement, clear it, re-run, assert identical."""
    before = {
        nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
        for nid in ir.tree.blocks()
        if isinstance(ir.tree.data(nid), BlockNode)
    }
    place_buffers(ir.tree)
    after = {
        nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
        for nid in ir.tree.blocks()
        if isinstance(ir.tree.data(nid), BlockNode)
    }
    assert before == after, f"place_buffers not idempotent: {before} != {after}"

    """hbm_out lives on the store block; the four others at root."""
    assert "hbm_out" in after[store_nid]
    root_names = set(after[root])
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod"} <= root_names
    assert "hbm_out" not in root_names


def test_place_buffers_is_idempotent_when_called_twice():
    """Calling place_buffers twice yields the same placement as once."""
    ir = build_canonical_ir()
    place_buffers(ir.tree)
    once = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers) for nid in ir.tree.blocks()}
    place_buffers(ir.tree)
    twice = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers) for nid in ir.tree.blocks()}
    assert once == twice
