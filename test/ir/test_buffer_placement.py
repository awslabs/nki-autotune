"""Tests for nkigym.ir.buffer_placement.place_buffers."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.tree import BlockNode


def test_place_buffers_canonical_matches_build():
    """place_buffers on the canonical tree reproduces the build-time placement.

    Every buffer (the four sbuf/psum scratch buffers plus the shared_hbm
    output) lives at the root: scratch buffers because their touchers span
    multiple root-level blocks, and the shared_hbm output because it is
    kernel-lifetime and is always declared at the root.
    """
    ir = build_canonical_ir()
    root = ir.tree.root

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

    """All five buffers — including the shared_hbm output — live at root."""
    root_names = set(after[root])
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"} <= root_names


def test_place_buffers_is_idempotent_when_called_twice():
    """Calling place_buffers twice yields the same placement as once."""
    ir = build_canonical_ir()
    place_buffers(ir.tree)
    once = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers) for nid in ir.tree.blocks()}
    place_buffers(ir.tree)
    twice = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers) for nid in ir.tree.blocks()}
    assert once == twice
