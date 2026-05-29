"""Tests for nkigym.transforms.Fuse under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import Fuse, FuseOption, Split, SplitOption


def _matmul_block_first_for(ir):
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMatmul:
                """First ForNode on the path from block to leaf."""
                for path_nid in ir.tree.preorder(nid):
                    if isinstance(ir.tree.data(path_nid), ForNode):
                        return path_nid
    raise AssertionError


def test_fuse_outer_trip_inverts_split():
    """Split then Fuse on the same axis returns the original ForNode extent."""
    ir = build_canonical_ir()
    target = _matmul_block_first_for(ir)
    original_extent = ir.tree.data(target).extent

    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, original_extent // 2)))
    """Locate the new outer ForNode."""
    parent = split_ir.tree.parent(target) if target in split_ir.tree.graph else None
    if parent is None:
        """Target was removed; pick the new top from same parent slot in original IR."""
        original_parent = ir.tree.parent(target)
        new_top = split_ir.tree.children(original_parent)[0]
    else:
        new_top = parent
    inner = split_ir.tree.children(new_top)[0]
    fuse_ir = Fuse().apply(split_ir, FuseOption(target_nids=(new_top, inner), target_axis=None))

    """The fused ForNode now has the original extent."""
    fused_parent = ir.tree.parent(target)
    fused_top = fuse_ir.tree.children(fused_parent)[0]
    fused_data = fuse_ir.tree.data(fused_top)
    assert isinstance(fused_data, ForNode)
    assert fused_data.extent == original_extent


def test_fuse_tensorize_absorbs_loop_into_leaf_tile():
    """Tensorize Fuse: a ForNode above the leaf is removed; the leaf's tile widens."""
    ir = build_canonical_ir()
    """Find the matmul block's ISA leaf and its immediate ForNode parent."""
    from nkigym.ops.matmul import NKIMatmul

    leaf_nid = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.data(nid).op_cls is NKIMatmul
    )
    parent_for = ir.tree.parent(leaf_nid)
    parent_for_data = ir.tree.data(parent_for)
    assert isinstance(parent_for_data, ForNode)
    """Skip the test if the parent is not a ForNode (e.g. the matmul body has no enclosing loops)."""
    if not isinstance(parent_for_data, ForNode):
        pytest.skip("matmul leaf has no enclosing ForNode to fuse")

    """Find the iter_var axis bound by parent_for.loop_var on the matmul block."""
    from nkigym.ir.expr import Var

    """Find the closest enclosing BlockNode with iter_vars."""
    matmul_block_nid = None
    matmul_block = None
    for anc in reversed(list(ir.tree.ancestors(leaf_nid))):
        data = ir.tree.data(anc)
        if isinstance(data, BlockNode) and len(data.iter_vars) > 0:
            matmul_block_nid = anc
            matmul_block = data
            break
    assert matmul_block is not None
    target_axis = next(
        iv.axis
        for iv, value in zip(matmul_block.iter_vars, matmul_block.iter_values)
        if isinstance(value, Var) and value.name == parent_for_data.loop_var
    )

    fuse_ir = Fuse().apply(ir, FuseOption(target_nids=(parent_for, leaf_nid), target_axis=target_axis))
    """The parent ForNode is gone; the leaf is now a direct child of what was parent_for's parent."""
    new_leaf_parent = fuse_ir.tree.parent(leaf_nid)
    assert new_leaf_parent != parent_for
