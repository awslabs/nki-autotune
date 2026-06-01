"""Tests for nkigym.transforms.ReverseComputeAt (consumer-lift)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import ReverseComputeAt, ReverseComputeAtOption, TransformLegalityError


def _block_for_op(ir, op_name: str) -> int:
    """Return the leaf-block nid whose single ISA leaf is ``op_name``."""
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _first_for_in_block(ir, block_nid: int) -> int:
    """Return the outermost ForNode under ``block_nid``."""
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ForNode):
            return d
    raise AssertionError(f"no ForNode under block {block_nid}")


def test_reverse_compute_at_rejects_non_fornode_target():
    """target_loop_nid must be a ForNode; passing a BlockNode nid raises."""
    ir = build_canonical_ir()
    consumer = _block_for_op(ir, "NKITensorCopy")
    matmul_block = _block_for_op(ir, "NKIMatmul")
    opt = ReverseComputeAtOption(block_nid=consumer, target_loop_nid=matmul_block)
    with pytest.raises(TransformLegalityError, match="ForNode"):
        ReverseComputeAt().apply(ir, opt)


def test_reverse_compute_at_rejects_target_inside_moved_block():
    """target_loop_nid must not be a loop of the block being moved (self-nesting)."""
    ir = build_canonical_ir()
    consumer = _block_for_op(ir, "NKITensorCopy")
    own_loop = _first_for_in_block(ir, consumer)
    opt = ReverseComputeAtOption(block_nid=consumer, target_loop_nid=own_loop)
    with pytest.raises(TransformLegalityError, match="ancestor|own|descendant"):
        ReverseComputeAt().apply(ir, opt)


def test_reverse_compute_at_rejects_producer_in_later_root_sibling():
    """Lifting the matmul under load_lhsT's loop is illegal: the matmul's other
    producer (load_rhs) sits in a LATER root-sibling than the target, so not every
    producer is satisfied (condition 5b)."""
    ir = build_canonical_ir()
    matmul = _block_for_op(ir, "NKIMatmul")
    load_lhsT = _block_for_op(ir, "NKILoad")  # first NKILoad block (lhs_T) in pre-order
    target = _first_for_in_block(ir, load_lhsT)
    opt = ReverseComputeAtOption(block_nid=matmul, target_loop_nid=target)
    with pytest.raises(TransformLegalityError, match="producer"):
        ReverseComputeAt().apply(ir, opt)


def test_reverse_compute_at_legal_lift_tensor_copy_under_matmul_m_loop():
    """Lifting tensor_copy under the matmul's M-loop passes legality: the matmul
    (a producer) ENCLOSES the target loop, and every other producer (loads, memset)
    is an earlier root-sibling."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    matmul = _block_for_op(ir, "NKIMatmul")
    m_loop = _first_for_in_block(ir, matmul)  # outermost ForNode under the matmul block
    opt = ReverseComputeAtOption(block_nid=tc, target_loop_nid=m_loop)
    """_check_legality must NOT raise (apply mechanics tested separately)."""
    ReverseComputeAt()._check_legality(ir, opt)
