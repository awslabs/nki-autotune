"""Tests for nkigym.transforms.Split under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.expr import to_affine
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption, TransformLegalityError


def _matmul_block(ir):
    """Return (block_nid, block) for the matmul leaf-block."""
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        """Find blocks whose direct children (not deep descendants) include an ISANode with NKIMatmul."""
        for child_nid in ir.tree.children(nid):
            if isinstance(ir.tree.data(child_nid), ForNode):
                """Walk down from ForNode to find ISANode."""
                for desc in ir.tree.descendants(child_nid):
                    if isinstance(ir.tree.data(desc), ISANode) and ir.tree.data(desc).op_cls is NKIMatmul:
                        return nid, block
        """Direct ISANode child."""
        for child_nid in ir.tree.children(nid):
            if isinstance(ir.tree.data(child_nid), ISANode) and ir.tree.data(child_nid).op_cls is NKIMatmul:
                return nid, block
    raise AssertionError("matmul block not found")


def _first_for_under(ir, block_nid):
    """Return the first ForNode descended from block_nid."""
    for nid in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(nid), ForNode):
            return nid
    raise AssertionError("no ForNode under block")


def test_split_outer_trip_replaces_for_with_chain():
    """Splitting a ForNode trip 16 by factors=(4, 4) gives a 4 -> 4 chain."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))

    """Old IR untouched."""
    assert ir.tree.data(target).extent == target_extent

    """New IR: parent's child slot now contains a fresh ForNode of extent 4 with one ForNode child."""
    parent = ir.tree.parent(target)
    new_kid = new_ir.tree.children(parent)[0]
    new_kid_data = new_ir.tree.data(new_kid)
    assert isinstance(new_kid_data, ForNode)
    assert new_kid_data.extent == 4
    inner = new_ir.tree.children(new_kid)[0]
    assert isinstance(new_ir.tree.data(inner), ForNode)
    assert new_ir.tree.data(inner).extent == target_extent // 4


def test_split_outer_trip_rewrites_iter_value_for_bound_axis():
    """The enclosing block's iter_value for the split iter_var becomes a sum of new loop_vars * strides."""
    ir = build_canonical_ir()
    matmul_block_nid, matmul_block = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_for = ir.tree.data(target)
    target_loop_var = target_for.loop_var
    target_extent = target_for.extent

    """Identify which iter_var was bound by the original loop_var."""
    bound_axis_index = None
    for i, value in enumerate(matmul_block.iter_values):
        from nkigym.ir.expr import Var

        if isinstance(value, Var) and value.name == target_loop_var:
            bound_axis_index = i
            break
    assert bound_axis_index is not None, "could not locate the iter_value bound by the target ForNode"

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    new_block = new_ir.tree.data(matmul_block_nid)
    new_value = new_block.iter_values[bound_axis_index]
    coeffs = to_affine(new_value)
    """The new value is a 2-term affine combination summing two loop_vars."""
    var_terms = {k: v for k, v in coeffs.items() if k is not None}
    assert len(var_terms) == 2
    """Coefficients match outer * inner_extent + inner."""
    assert sorted(var_terms.values()) == [1, target_extent // 4]


def test_split_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent
    snapshot_num_nodes = ir.tree.num_nodes
    Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    assert ir.tree.num_nodes == snapshot_num_nodes


def test_split_rejects_factor_product_mismatch():
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(3, 5)))
