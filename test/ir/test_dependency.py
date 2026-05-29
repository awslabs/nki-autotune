"""Tests for the block-keyed dependency graph."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import BlockNode, ISANode
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


def _block_for_op(ir, op_cls):
    """Return the leaf-block nid whose body emits ``op_cls``. The init memset block does NOT count
    as the matmul block."""
    for nid in ir.tree.blocks():
        block_data = ir.tree.data(nid)
        assert isinstance(block_data, BlockNode)
        """Only examine leaf blocks (skip the synthetic root with empty iter_vars)."""
        if not block_data.iter_vars:
            continue
        """Search descendants for the op's ISA leaf."""
        for d in ir.tree.descendants(nid):
            """Skip nodes inside child blocks."""
            if d != nid and isinstance(ir.tree.data(d), BlockNode):
                continue
            d_data = ir.tree.data(d)
            if isinstance(d_data, ISANode) and d_data.op_cls is op_cls:
                return nid
    raise AssertionError(f"no leaf block for {op_cls.__name__}")


def test_dependency_orders_canonical_matmul_chain():
    """For canonical matmul: load_lhs / load_rhs precede matmul, which precedes tensor_copy, which precedes store."""
    ir = build_canonical_ir()
    matmul_nid = _block_for_op(ir, NKIMatmul)
    tc_nid = _block_for_op(ir, NKITensorCopy)
    store_nid = _block_for_op(ir, NKIStore)
    """The dependency graph contains an edge from matmul to tensor_copy."""
    assert ir.dependency.must_precede(matmul_nid, tc_nid)
    assert ir.dependency.must_precede(tc_nid, store_nid)
    assert ir.dependency.must_precede(matmul_nid, store_nid)


def test_dependency_does_not_order_independent_loads():
    """Loads of distinct tensors are independent; neither precedes the other."""
    ir = build_canonical_ir()
    load_nids = []
    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        """Skip synthetic root."""
        if not block.iter_vars:
            continue
        for d in ir.tree.descendants(nid):
            """Skip nested blocks."""
            if d != nid and isinstance(ir.tree.data(d), BlockNode):
                continue
            d_data = ir.tree.data(d)
            if isinstance(d_data, ISANode) and d_data.op_cls is NKILoad:
                load_nids.append(nid)
                break
    assert len(load_nids) == 2
    a, b = load_nids
    assert not ir.dependency.must_precede(a, b)


def test_memset_precedes_matmul_in_dependency():
    """The canonical memset block must be ordered BEFORE the matmul block (was inverted under bundled init)."""
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_nid = _block_for_op(ir, NKIMemset)
    matmul_nid = _block_for_op(ir, NKIMatmul)
    assert ir.dependency.must_precede(memset_nid, matmul_nid), "memset must precede matmul"
    assert not ir.dependency.must_precede(matmul_nid, memset_nid), "matmul must NOT precede memset"
