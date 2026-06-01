"""Unit tests for the shared _compute_at_impl move mechanics."""

from __future__ import annotations

import copy
from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _compute_at_impl


def _block_for_op(ir, op_name):
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(op_name)


def _loops_under(ir, block_nid):
    return [ir.tree.data(d).loop_var for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ForNode)]


def test_sink_load_lhsT_under_matmul_d1_collapses_both_loops():
    """ComputeAt-direction: sink load_lhsT (loops d0,d1) under matmul's d1 loop.
    The load's d0,d1 are fully covered by the matmul's enclosing d0,d1 -> both
    collapse; the load block becomes loopless, spliced under matmul's d1 loop."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")  # first NKILoad = lhs_T
    matmul = _block_for_op(ir, "NKIMatmul")
    matmul_loops = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode)]
    d1_loop = matmul_loops[1]  # i_d1_0 (matmul nest is d0, d1, d2)
    new_ir = copy.deepcopy(ir)
    _compute_at_impl(new_ir, block_nid=load, target_loop_nid=d1_loop, index=-2, is_reverse=False)
    assert _loops_under(new_ir, load) == []
    assert load in new_ir.tree.descendants(d1_loop)
