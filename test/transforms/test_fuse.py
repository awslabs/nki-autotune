"""Tests for nkigym.transforms.Fuse under BlockNode IR."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Mul, Var
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import CodeMotion, CodeMotionOption, Fuse, FuseOption, Split, SplitOption, TransformLegalityError


def _matmul_block_first_for(ir: KernelIR) -> int:
    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.isa(d).op_cls is NKIMatmul:
                """First ForNode on the path from block to leaf."""
                for path_nid in ir.tree.preorder(nid):
                    if isinstance(ir.tree.data(path_nid), ForNode):
                        return path_nid
    raise AssertionError


def test_fuse_outer_trip_renders_and_passes_numerics(tmp_path) -> None:
    """A split followed by an outer-trip fuse preserves rendered behavior."""
    ir = build_canonical_ir()
    target = _matmul_block_first_for(ir)
    extent = ir.tree.loop(target).extent
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    target_parent = ir.tree.parent(target)
    assert target_parent is not None
    outer = split.tree.children(target_parent)[0]
    inner = split.tree.children(outer)[0]
    fused = Fuse().apply(split, FuseOption(target_nids=(outer, inner), target_axis=None))
    fused_top = fused.tree.children(target_parent)[0]
    assert fused.tree.loop(fused_top).extent == extent
    assert_matmul_ir_simulates(fused, tmp_path, "fuse_outer_trip")


def test_fuse_outer_trip_normalizes_nested_blocks():
    """Fusing enclosing loops updates bindings inside nested blocks."""
    ir = build_canonical_ir()
    matmul = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == "NKIMatmul"
    )
    matmul_k = next(
        ancestor
        for ancestor in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(ancestor), ForNode) and ir.tree.loop(ancestor).loop_var == "i_d0_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=matmul_k, factors=(2, 2, 2, 2), target_axis=None))
    load = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        and ir.tree.isa(nid).op_cls.__name__ == "NKILoad"
        and ir.tree.isa(nid).operand_bindings["src"].tensor == "lhs_T"
    )
    load_k = next(
        ancestor
        for ancestor in ir.tree.ancestors(load)
        if isinstance(ir.tree.data(ancestor), ForNode) and ir.tree.loop(ancestor).loop_var == "i_d0_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=load_k, factors=(2, 2, 2, 2), target_axis=None))
    matmul = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == "NKIMatmul"
    )
    matmul_k_loops = [
        ancestor
        for ancestor in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(ancestor), ForNode) and ir.tree.loop(ancestor).loop_var.startswith("i_d0_")
    ]
    load = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        and ir.tree.isa(nid).op_cls.__name__ == "NKILoad"
        and ir.tree.isa(nid).operand_bindings["src"].tensor == "lhs_T"
    )
    load_block = next(
        ancestor for ancestor in reversed(ir.tree.ancestors(load)) if isinstance(ir.tree.data(ancestor), BlockNode)
    )
    ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=load_block, target_loop_nid=matmul_k_loops[-1], index=0))
    fused = Fuse().apply(ir, FuseOption(target_nids=tuple(matmul_k_loops[:2]), target_axis=None))
    assert "i_d0_3" not in render(fused)


def test_fuse_outer_trip_rejects_partial_nested_loop_dependence():
    """A nested wide copy that uses only the outer source loop cannot be fused."""
    tree = KernelTree()
    outer_var = Var(name="i_d2_0")
    inner_var = Var(name="i_d2_1")
    owner = BlockNode(
        iter_vars=(IterVar(axis="d2", dom=(0, 16), role=AxisRole.PARALLEL),),
        iter_values=(Add(left=Mul(left=outer_var, right=Const(value=2)), right=inner_var),),
        reads=(),
        writes=(),
        axis_map={"F": "d2"},
    )
    owner_nid = tree.add_node(owner, parent=tree.root)
    outer = tree.add_node(ForNode(loop_var=outer_var.name, extent=8), parent=owner_nid)
    inner = tree.add_node(ForNode(loop_var=inner_var.name, extent=2), parent=outer)
    copy_region = BufferRegion(
        tensor="src",
        ranges=((Const(value=0), Const(value=128)), (Mul(left=outer_var, right=Const(value=256)), Const(value=256))),
    )
    nested = BlockNode(
        iter_vars=(IterVar(axis="d2", dom=(0, 8), role=AxisRole.PARALLEL),),
        iter_values=(outer_var,),
        reads=(copy_region,),
        writes=(BufferRegion(tensor="dst", ranges=copy_region.ranges),),
        axis_map={"F": "d2"},
    )
    nested_nid = tree.add_node(nested, parent=inner)
    tree.add_node(
        ISANode(
            op_cls=NKITensorCopy,
            operand_bindings={"src": copy_region, "dst": BufferRegion(tensor="dst", ranges=copy_region.ranges)},
        ),
        parent=nested_nid,
    )
    ir = KernelIR(
        func_name="partial_nested_loop_dependence",
        param_names=[],
        return_name="dst",
        tree=tree,
        dependency=Dependency(tree),
    )
    option = FuseOption(target_nids=(outer, inner), target_axis=None)

    assert option not in Fuse().analyze(ir)
    with pytest.raises(TransformLegalityError, match="depends on only"):
        Fuse().apply(ir, option)


def test_fuse_tensorize_rejects_matmul_n_above_max_tile():
    """Tensorize-Fuse cannot widen matmul N past its hardware tile limit."""
    ir = build_canonical_ir()
    leaf_nid = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    parent_for = ir.tree.parent(leaf_nid)  # i_d2_0, the innermost matmul loop
    assert parent_for is not None
    mb = next(
        ir.tree.block(a)
        for a in reversed(ir.tree.ancestors(leaf_nid))
        if isinstance(ir.tree.data(a), BlockNode) and ir.tree.block(a).iter_vars
    )
    target_axis = next(
        iv.axis
        for iv, v in zip(mb.iter_vars, mb.iter_values)
        if isinstance(v, Var) and v.name == ir.tree.loop(parent_for).loop_var
    )
    option = FuseOption(target_nids=(parent_for, leaf_nid), target_axis=target_axis)
    with pytest.raises(TransformLegalityError, match="MAX_TILE_SIZE"):
        Fuse().apply(ir, option)


def test_split_then_fuse_tensorize_round_trips(tmp_path):
    """Tensorize-Split the load d1 (2048->16x128) then tensorize-Fuse it back == original;
    both intermediate and final render + sim correctly."""
    ir = build_canonical_ir()
    load_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKILoad"
    )
    split_ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    """The load now has a new inner ForNode above the leaf; fuse it back."""
    new_leaf = next(
        n
        for n in split_ir.tree.preorder()
        if isinstance(split_ir.tree.data(n), ISANode) and split_ir.tree.isa(n).op_cls.__name__ == "NKILoad"
    )
    inner_for = split_ir.tree.parent(new_leaf)
    assert inner_for is not None
    assert isinstance(split_ir.tree.data(inner_for), ForNode)
    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(inner_for, new_leaf), target_axis="d1"))
    assert render(fused_ir) == render(ir)
    assert_matmul_ir_simulates(fused_ir, tmp_path, "split_then_fuse_tensorize")
