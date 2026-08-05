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


def _check_dependency_orders_canonical_matmul_chain():
    """For canonical matmul: load_lhs / load_rhs precede matmul, which precedes tensor_copy, which precedes store."""
    ir = build_canonical_ir()
    matmul_nid = _block_for_op(ir, NKIMatmul)
    tc_nid = _block_for_op(ir, NKITensorCopy)
    store_nid = _block_for_op(ir, NKIStore)
    """The dependency graph contains an edge from matmul to tensor_copy."""
    assert ir.dependency.must_precede(matmul_nid, tc_nid)
    assert ir.dependency.must_precede(tc_nid, store_nid)
    assert ir.dependency.must_precede(matmul_nid, store_nid)


def _check_dependency_does_not_order_independent_loads():
    """Loads of distinct tensors are independent; neither precedes the other."""
    ir = build_canonical_ir()
    load_nids = []
    for nid in ir.tree.blocks():
        block = ir.tree.block(nid)
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


def _check_memset_precedes_matmul_in_dependency():
    """The canonical memset block must be ordered BEFORE the matmul block (was inverted under bundled init)."""
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_nid = _block_for_op(ir, NKIMemset)
    matmul_nid = _block_for_op(ir, NKIMatmul)
    assert ir.dependency.must_precede(memset_nid, matmul_nid), "memset must precede matmul"
    assert not ir.dependency.must_precede(matmul_nid, memset_nid), "matmul must NOT precede memset"


def _check_canonical_synthesizes_memset_for_matmul():
    """A matmul (RMW dst) gets a synthesized memset sibling block zeroing its PSUM region."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ISANode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_leaves = {
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls is NKIMemset
    }
    assert len(memset_leaves) == 1, "exactly one synthesized memset for the matmul"
    memset = ir.tree.isa(next(iter(memset_leaves)))
    assert memset.operand_bindings["dst"].tensor == "psum_prod"
    assert memset.kwargs == {"value": 0.0}


def _check_disjoint_tile_writes_have_no_edge():
    """Two hand-built blocks writing disjoint tiles of one buffer get NO dependency edge."""
    from dataclasses import replace

    from nkigym.ir.arith.expr import Const, Var
    from nkigym.ir.dependency import Dependency
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


def _check_overlapping_tile_writes_have_edge():
    """Two blocks writing the SAME tile get a WAW edge."""
    from dataclasses import replace

    from nkigym.ir.arith.expr import Const, Var
    from nkigym.ir.dependency import Dependency
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


def _check_matmul_carries_psum_over_kloop():
    """The matmul's K loop carries psum_prod (rmw access invariant across K)."""
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "psum_prod") is True


def _check_load_does_not_carry_over_kloop():
    """A load block (pure output, no rmw) does not carry its output over K."""
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    load_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKILoad
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(load_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "sbuf_lhs_T") is False


def _check_dependency_graph_keyed_on_leaf_nids():
    """Graph nodes are ISA-leaf nids or carry-loop ForNode nids, never block nids."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    for node in ir.dependency.graph.nodes:
        data = ir.tree.data(node)
        assert isinstance(data, (ISANode, ForNode)), f"node {node} is neither an ISA leaf nor a carry loop"


def _check_must_precede_accepts_block_or_leaf_nids():
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


def _check_first_backward_edge_flags_memset_sunk_under_kloop():
    """Sinking the psum memset INTO the matmul's K loop is a backward edge, via the
    production insertion query on the pre-move tree (span-promotion; frozen
    directions). Rebuilding Dependency on the moved tree would flip RAW->WAR and
    hide it — that path is invalid now that CARRY edges are gone."""
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, NKIMemset)
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(memset_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, 0) is not None


def _check_span_promotion_rejects_memset_sunk_into_kloop():
    """Sinking the psum memset INTO the matmul's K loop is a backward edge:
    psum_prod is carried across K, so span-promotion widens both endpoints to
    K-span and the memset can no longer sit before the matmul. Checked on the
    pre-move tree via the production insertion query (frozen directions).

    (Verdict relies on the insertion-path promotion completed in the next task.)"""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, NKIMemset)
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(memset_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, 0) is not None


def _check_first_backward_edge_flags_consumer_before_producer():
    """Sinking the tensor_copy (consumer of psum_prod) under the MEMSET's loop puts it
    before the matmul that produces psum_prod -> backward flow edge matmul->tensor_copy.

    (Verdict relies on the insertion-path promotion completed in the next task.)"""
    from nkigym.ir.tree import ForNode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    tc_blk = _block_for_op(ir, NKITensorCopy)
    memset_blk = _block_for_op(ir, NKIMemset)
    memset_loop = next(d for d in ir.tree.preorder(memset_blk) if isinstance(ir.tree.data(d), ForNode))
    moved_leaf = ir.dependency._resolve(tc_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, memset_loop, 0) is not None


def _check_first_backward_edge_frozen_directions_catch_parallel_producer_flip():
    """The direction bug, at the dependency layer: sinking the rhs load (PARALLEL
    producer of sbuf_rhs, no carry edge) past the matmul that reads it.

    Rebuilding Dependency on the moved tree re-derives the RAW load->matmul
    hazard as a forward WAR matmul->load (the load now executes after the
    matmul), so the rebuilt graph reports NO backward edge -> the trap. The fix
    freezes directions from the ORIGINAL graph and evaluates spans on the moved
    tree, keeping the RAW load->matmul orientation, so the post-move backward
    span IS detected.
    """
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms.code_motion import _move

    ir = build_canonical_ir()
    rhs_load = next(
        nid
        for nid in ir.tree.blocks()
        if nid != ir.tree.root
        and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        and (leaf := next(d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)))
        and ir.tree.isa(leaf).op_cls is NKILoad
        and ir.tree.isa(leaf).operand_bindings["src"].tensor == "rhs"
    )
    tc_blk = _block_for_op(ir, NKITensorCopy)
    tc_loop = next(d for d in ir.tree.preorder(tc_blk) if isinstance(ir.tree.data(d), ForNode))
    moved_leaf = ir.dependency._resolve(rhs_load)

    moved = copy.deepcopy(ir)
    _move(moved, block_nid=rhs_load, target_loop_nid=tc_loop, index=0)

    """The trap: rebuilding on the moved tree hides the violation (edge flipped forward)."""
    rebuilt = Dependency(moved.tree)
    assert rebuilt.first_backward_edge(moved_leaf) is None

    """The fix: original directions + moved-tree spans expose the backward RAW edge."""
    offending = ir.dependency.first_backward_edge(moved_leaf, tree=moved.tree)
    assert offending is not None


def _check_first_backward_edge_allows_load_under_kloop():
    """Sinking the lhs_T load (writes sbuf_lhs_T, NOT carried over K) under K is legal -> None."""
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms.code_motion import _move

    ir = build_canonical_ir()
    load_blk = _block_for_op(ir, NKILoad)
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=load_blk, target_loop_nid=kloop, index=0)
    dep = Dependency(moved.tree)
    load_leaf = next(
        n
        for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.isa(n).op_cls is NKILoad
    )
    assert dep.first_backward_edge(load_leaf) is None


def _check_hazard_edges_record_conflicting_tensor():
    """Each RAW/WAW/WAR edge carries the tensor it is about, so span-promotion
    can key on the shared buffer per edge (a leaf may have one carried and one
    non-carried edge at once)."""
    from test.transforms._fixtures import build_canonical_ir

    ir = build_canonical_ir()
    dep = ir.dependency
    base_edges = [
        (a, b, attrs) for a, b, attrs in dep.graph.edges(data=True) if attrs.get("kind") in {"RAW", "WAW", "WAR"}
    ]
    assert base_edges, "expected at least one RAW/WAW/WAR edge in the canonical IR"
    for _a, _b, attrs in base_edges:
        assert isinstance(attrs["tensor"], str) and attrs["tensor"]


def _check_tensor_carried_across_psum_over_kloop():
    """psum (matmul rmw, offset invariant across K) is carried across the K loop;
    a pure-read operand (sbuf_lhs_T) is not."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    kloop = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "psum_prod") is True
    assert _tensor_carried_across(ir.tree, kloop, "sbuf_lhs_T") is False


def test_access_invariant_across_matches_offset_var():
    """The matmul's psum access is invariant across K (K-var absent from the psum
    offset) but NOT across a loop whose var indexes the psum offset."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.dependency import _access_invariant_across
    from nkigym.ir.tree import ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls is NKIMatmul
    )
    assert _access_invariant_across(ir.tree, matmul_leaf, "i_d0_0", "psum_prod") is True
    assert _access_invariant_across(ir.tree, matmul_leaf, "i_d1_0", "psum_prod") is False


def test_fold_accumulator_carried_across_ko_not_broken_by_own_writeback():
    """Post-RFactor, sbuf_prod is accumulated across ko by the fold whose dst
    aliases its data1 rmw — that write-back must NOT be misread as a re-init, so
    sbuf_prod is carried across ko (the drain/store may not sink into ko). psum,
    re-zeroed each ko by a SEPARATE memset leaf, is correctly NOT carried across ko."""
    from test.transforms._fixtures import INPUT_SPECS, f_matmul
    from test.transforms._helpers import matmul_loop

    from nkigym.ir import build_initial_ir
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.transforms import Reorder, ReorderOption, RFactor, RFactorOption, Split, SplitOption

    loop = matmul_loop
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    ir = Split().apply(ir, SplitOption(target_nid=loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None))
    ir = Split().apply(ir, SplitOption(target_nid=loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d1_1"), inner_nid=loop(ir, "i_d2_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d1_0"), inner_nid=loop(ir, "i_d2_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d0_1"), inner_nid=loop(ir, "i_d2_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d0_0"), inner_nid=loop(ir, "i_d2_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d0_1"), inner_nid=loop(ir, "i_d1_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=loop(ir, "i_d0_1"), inner_nid=loop(ir, "i_d1_1")))
    ir = RFactor().apply(ir, RFactorOption(target_loop_nid=loop(ir, "i_d0_0"), factor_axis=0))
    ko = loop(ir, "i_d0_0")
    assert _tensor_carried_across(ir.tree, ko, "sbuf_prod") is True
    assert _tensor_carried_across(ir.tree, ko, "psum_prod") is False


def test_canonical_dependency_and_memset_contract() -> None:
    """Canonical dataflow orders dependent operations while keeping loads independent."""
    _check_dependency_orders_canonical_matmul_chain()
    _check_dependency_does_not_order_independent_loads()
    _check_memset_precedes_matmul_in_dependency()
    _check_canonical_synthesizes_memset_for_matmul()


def test_write_hazards_distinguish_disjoint_and_overlapping_tiles() -> None:
    """Dependency construction omits disjoint writes and orders overlapping writes."""
    _check_disjoint_tile_writes_have_no_edge()
    _check_overlapping_tile_writes_have_edge()


def test_loop_carried_tensor_contract() -> None:
    """RMW tensors carry across invariant loops while pure outputs and reads do not."""
    _check_matmul_carries_psum_over_kloop()
    _check_load_does_not_carry_over_kloop()
    _check_tensor_carried_across_psum_over_kloop()


def test_dependency_graph_node_and_edge_metadata() -> None:
    """The dependency graph resolves block or leaf IDs and labels hazard tensors."""
    _check_dependency_graph_keyed_on_leaf_nids()
    _check_must_precede_accepts_block_or_leaf_nids()
    _check_hazard_edges_record_conflicting_tensor()


def test_backward_edge_queries_cover_illegal_and_legal_moves() -> None:
    """Insertion checks catch reset and producer-order violations while allowing legal loads."""
    _check_first_backward_edge_flags_memset_sunk_under_kloop()
    _check_span_promotion_rejects_memset_sunk_into_kloop()
    _check_first_backward_edge_flags_consumer_before_producer()
    _check_first_backward_edge_frozen_directions_catch_parallel_producer_flip()
    _check_first_backward_edge_allows_load_under_kloop()
