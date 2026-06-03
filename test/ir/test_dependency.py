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


def test_carry_loops_of_matmul_leaf():
    """The matmul leaf's K loop (d0, ACCUMULATION) carries psum_prod; M/N (PARALLEL) carry nothing."""
    from nkigym.ir.dependency import _carry_loops_of_leaf
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    carries = _carry_loops_of_leaf(ir.tree, matmul_leaf)
    carried_buffers = set(carries.values())
    assert carried_buffers == {"psum_prod"}, carried_buffers
    assert len(carries) == 1
    (kloop_nid,) = carries
    from nkigym.ir.tree import ForNode

    assert isinstance(ir.tree.data(kloop_nid), ForNode)
    assert ir.tree.data(kloop_nid).loop_var == "i_d0_0"


def test_carry_loops_empty_for_pure_parallel_leaf():
    """A load leaf (all-PARALLEL axes) has no carry loops."""
    from nkigym.ir.dependency import _carry_loops_of_leaf
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    load_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKILoad
    )
    assert _carry_loops_of_leaf(ir.tree, load_leaf) == {}


def test_dependency_graph_keyed_on_leaf_nids():
    """Graph nodes are ISA-leaf nids or carry-loop ForNode nids, never block nids."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    for node in ir.dependency.graph.nodes:
        data = ir.tree.data(node)
        assert isinstance(data, (ISANode, ForNode)), f"node {node} is neither an ISA leaf nor a carry loop"


def test_must_precede_accepts_block_or_leaf_nids():
    """must_precede works whether given block nids (legacy) or leaf nids (resolved either way)."""
    ir = build_canonical_ir()
    matmul_blk = _block_for_op(ir, NKIMatmul)
    store_blk = _block_for_op(ir, NKIStore)
    from nkigym.ir.tree import ISANode

    def leaf_of(blk):
        return next(
            d
            for d in ir.tree.preorder(blk)
            if isinstance(ir.tree.data(d), ISANode)
            and next(a for a in reversed(ir.tree.ancestors(d)) if isinstance(ir.tree.data(a), BlockNode)) == blk
        )

    assert ir.dependency.must_precede(matmul_blk, store_blk)
    assert ir.dependency.must_precede(leaf_of(matmul_blk), leaf_of(store_blk))


def test_carry_edges_memset_dominates_kloop_and_kloop_dominates_drain():
    """Canonical: memset_leaf -> K_loop and K_loop -> tensor_copy_leaf carry edges exist."""
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    dep = ir.dependency

    def leaf(op_cls):
        return next(
            n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is op_cls
        )

    memset_leaf = leaf(NKIMemset)
    matmul_leaf = leaf(NKIMatmul)
    tc_leaf = leaf(NKITensorCopy)
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )

    assert dep.graph.has_edge(memset_leaf, kloop), "memset must dominate the K loop"
    assert dep.graph.has_edge(kloop, tc_leaf), "K loop must dominate the drain (tensor_copy)"


def test_no_carry_edge_for_input_loads():
    """The lhs_T load (writes sbuf_lhs_T, indexed by K) gets NO edge to the K loop."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    dep = ir.dependency
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    load_leaves = [
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKILoad
    ]
    for ll in load_leaves:
        assert not dep.graph.has_edge(ll, kloop), f"load {ll} must NOT be forced to dominate K"


def test_first_backward_edge_flags_memset_sunk_under_kloop():
    """After sinking the memset (writer of psum_prod, carried over K) under the
    K loop, the memset->K-loop carry edge points backward."""
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, NKIMemset)
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=memset_blk, target_loop_nid=kloop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    memset_leaf = next(
        n
        for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKIMemset
    )
    assert dep.first_backward_edge(memset_leaf) is not None


def test_first_backward_edge_flags_consumer_before_producer():
    """Sinking the tensor_copy (consumer of psum_prod) under the MEMSET's loop puts it
    before the matmul that produces psum_prod -> backward flow edge matmul->tensor_copy."""
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    tc_blk = _block_for_op(ir, NKITensorCopy)
    memset_blk = _block_for_op(ir, NKIMemset)
    memset_loop = next(d for d in ir.tree.preorder(memset_blk) if isinstance(ir.tree.data(d), ForNode))
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=tc_blk, target_loop_nid=memset_loop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    tc_leaf = next(
        n
        for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKITensorCopy
    )
    assert dep.first_backward_edge(tc_leaf) is not None


def test_first_backward_edge_allows_load_under_kloop():
    """Sinking the lhs_T load (writes sbuf_lhs_T, NOT carried over K) under K is legal -> None."""
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    load_blk = _block_for_op(ir, NKILoad)
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=load_blk, target_loop_nid=kloop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    load_leaf = next(
        n
        for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKILoad
    )
    assert dep.first_backward_edge(load_leaf) is None
