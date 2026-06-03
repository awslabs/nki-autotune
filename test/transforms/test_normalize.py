"""Unit tests for normalize_block: drop trip-1, dense rename, rewrite bindings."""

from __future__ import annotations

from nkigym.ir.arith.expr import Const, Var
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.transforms._normalize import normalize_block


def _two_loop_d1_block():
    """Hand-build a block whose d1 axis has TWO loops named non-densely (i_d1_0_0, i_d1_0_1)
    over a tile-128 load — the post-split-bug shape — plus a trip-1 loop to be dropped."""
    tree = KernelTree()
    block = BlockNode(
        iter_vars=(IterVar(axis="d1", dom=(0, 2048), role=AxisRole.PARALLEL),),
        iter_values=(Var(name="i_d1_0_0"),),
        reads=(),
        writes=(BufferRegion(tensor="sbuf", ranges=((Var(name="i_d1_0_0"), Const(value=2048)),)),),
        axis_map={"F": "d1"},
    )
    bnid = tree.add_node(block, parent=tree.root)
    outer = tree.add_node(ForNode(loop_var="i_d1_0_0", extent=2), parent=bnid)
    inner = tree.add_node(ForNode(loop_var="i_d1_0_1", extent=8), parent=outer)
    leaf = tree.add_node(
        ISANode(
            op_cls=NKILoad,
            operand_bindings={"dst": BufferRegion(tensor="sbuf", ranges=((Var(name="i_d1_0_1"), Const(value=128)),))},
        ),
        parent=inner,
    )
    return tree, bnid, outer, inner, leaf


def test_normalize_renames_dense():
    """Two d1 loops named i_d1_0_0/i_d1_0_1 -> dense i_d1_0/i_d1_1."""
    tree, bnid, outer, inner, leaf = _two_loop_d1_block()
    normalize_block(tree, bnid)
    assert tree.data(outer).loop_var == "i_d1_0"
    assert tree.data(inner).loop_var == "i_d1_1"


def test_normalize_drops_trip1():
    """A trip-1 ForNode is removed; its child re-links to its parent."""
    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), axis_map={})
    bnid = tree.add_node(block, parent=tree.root)
    real = tree.add_node(ForNode(loop_var="i_d0_0", extent=16), parent=bnid)
    triv = tree.add_node(ForNode(loop_var="i_d1_0", extent=1), parent=real)
    leaf = tree.add_node(ISANode(op_cls=NKILoad, operand_bindings={}), parent=triv)
    normalize_block(tree, bnid)
    """trip-1 gone; leaf now child of the real loop."""
    from nkigym.ir.tree import ForNode as FN

    remaining = [tree.data(n).loop_var for n in tree.preorder(bnid) if isinstance(tree.data(n), FN)]
    assert remaining == ["i_d0_0"]
    assert tree.parent(leaf) == real
