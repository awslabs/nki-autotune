"""Tests for nkigym.transforms.Reorder under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode
from nkigym.transforms import Reorder, ReorderOption, TransformLegalityError


def _first_two_adjacent_fors(ir):
    """Return (outer_nid, inner_nid) for the first parent-child ForNode pair."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if isinstance(kid_data, ForNode):
            return nid, kids[0]
    raise AssertionError("no adjacent ForNode pair")


def test_reorder_swaps_payloads():
    """Apply swaps the two ForNode payloads while keeping nids stable."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    outer_data = ir.tree.data(outer)
    inner_data = ir.tree.data(inner)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    assert new_ir.tree.data(outer) == inner_data
    assert new_ir.tree.data(inner) == outer_data


def test_reorder_self_inverse():
    """Apply twice returns the original payload."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    opt = ReorderOption(outer_nid=outer, inner_nid=inner)
    new_ir = Reorder().apply(Reorder().apply(ir, opt), opt)
    assert new_ir.tree.data(outer) == ir.tree.data(outer)
    assert new_ir.tree.data(inner) == ir.tree.data(inner)


def test_reorder_rejects_sequential_role():
    """Reorder rejects a swap on a dim whose enclosing block declares SEQUENTIAL role."""
    from test.transforms._seq_fixture import build_seq_ir

    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
