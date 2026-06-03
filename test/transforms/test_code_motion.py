"""Tests for nkigym.transforms._code_motion._move (structural move)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ISANode
from nkigym.transforms._code_motion import _move


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _innermost_for(ir, block_nid: int) -> int:
    leaf = next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))
    return ir.tree.ancestors(leaf)[-1]


def test_move_lifts_tensor_copy_under_matmul_inner_loop():
    """Lifting tensor_copy under the matmul's innermost loop nests it there."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _innermost_for(ir, mm)
    _move(ir, block_nid=tc, target_loop_nid=target, index=-1, is_reverse=True)
    assert tc in ir.tree.descendants(target)
