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


def test_canonical_synthesizes_memset_for_matmul():
    """A matmul (RMW dst) gets a synthesized memset sibling block zeroing its PSUM region."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ISANode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_leaves = {
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.data(nid).op_cls is NKIMemset
    }
    assert len(memset_leaves) == 1, "exactly one synthesized memset for the matmul"
    memset = ir.tree.data(next(iter(memset_leaves)))
    assert memset.operand_bindings["dst"].tensor == "psum_prod"
    assert memset.kwargs == {"value": 0.0}


def test_disjoint_tile_writes_have_no_edge():
    """Two hand-built blocks writing disjoint tiles of one buffer get NO dependency edge."""
    from dataclasses import replace

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
    from nkigym.ops.base import AxisRole
    from nkigym.ops.memset import NKIMemset

    tree = KernelTree()
    """Add a Buffer to the root block so the dependency graph can find it."""
    buf = Buffer(name="buf", shape=(256,), dtype="float32", location="shared_hbm")
    root_blk = tree.data(tree.root)
    tree.graph.nodes[tree.root]["data"] = replace(root_blk, alloc_buffers=(buf,))

    """Two sibling blocks under root, each writing a distinct CONSTANT tile of 'buf'."""

    def add_writer(offset):
        blk = BlockNode(
            iter_vars=(IterVar(axis="d0", dom=(0, 256), role=AxisRole.PARALLEL),),
            iter_values=(Var(name="i"),),
            reads=(),
            writes=(BufferRegion(tensor="buf", ranges=((Const(value=offset), Const(value=128)),)),),
        )
        nid = tree.add_node(blk, parent=tree.root)
        f = tree.add_node(ForNode(loop_var="i", extent=1), parent=nid)
        tree.add_node(
            ISANode(
                op_cls=NKIMemset,
                operand_bindings={"dst": BufferRegion(tensor="buf", ranges=((Const(value=offset), Const(value=128)),))},
                kwargs={"value": 0.0},
            ),
            parent=f,
        )
        return nid

    a = add_writer(0)
    b = add_writer(128)
    dep = Dependency(tree)
    assert not dep.must_precede(a, b)
    assert not dep.must_precede(b, a)


def test_overlapping_tile_writes_have_edge():
    """Two blocks writing the SAME tile get a WAW edge."""
    from dataclasses import replace

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
    from nkigym.ops.base import AxisRole
    from nkigym.ops.memset import NKIMemset

    tree = KernelTree()
    """Add a Buffer to the root block so the dependency graph can find it."""
    buf = Buffer(name="buf", shape=(256,), dtype="float32", location="shared_hbm")
    root_blk = tree.data(tree.root)
    tree.graph.nodes[tree.root]["data"] = replace(root_blk, alloc_buffers=(buf,))

    def add_writer():
        blk = BlockNode(
            iter_vars=(IterVar(axis="d0", dom=(0, 256), role=AxisRole.PARALLEL),),
            iter_values=(Var(name="i"),),
            reads=(),
            writes=(BufferRegion(tensor="buf", ranges=((Const(value=0), Const(value=128)),)),),
        )
        nid = tree.add_node(blk, parent=tree.root)
        f = tree.add_node(ForNode(loop_var="i", extent=1), parent=nid)
        tree.add_node(
            ISANode(
                op_cls=NKIMemset,
                operand_bindings={"dst": BufferRegion(tensor="buf", ranges=((Const(value=0), Const(value=128)),))},
                kwargs={"value": 0.0},
            ),
            parent=f,
        )
        return nid

    a = add_writer()
    b = add_writer()
    dep = Dependency(tree)
    assert dep.must_precede(a, b)
