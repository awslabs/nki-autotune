"""Shared structural move for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop by a **verbatim residual splice**,
NOT region recomputation. Precondition (``_check_same_loop_prefix``): the target's
enclosing loop nest (outermost down to ``target_loop_nid``), as an ordered
``(loop_var, extent)`` sequence, must be an EXACT PREFIX of the moved block's own
loop nest. The move then drops the shared-prefix loops the target already provides
and re-parents the moved block's residual loops (the inner loops the target does
not iterate) plus its leaf under the target — regions untouched. Because the prefix
loop VARS are shared by name between the moved block and the target nest, the moved
block's regions referencing them resolve against the target's loops with no rewrite.

This intentionally avoids TVM-style per-dim domain solving: a partial split of a
shared dim, or a different loop order, is REJECTED loudly (``Split``/``Reorder``
first), so every legal move is a clean structural merge. Direction (``is_reverse``)
does not change the structural steps; the caller's dependency-direction check differs.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms._domain_solve import _dim_from_loopvar, _enclosing_block, enclosing_dim_loops
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import TransformLegalityError


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` by a verbatim residual splice.

    Caller has checked legality (``_check_same_loop_prefix``, so the target's
    enclosing nest is an exact prefix of the moved block's nest) and deep-copied.
    ``index`` follows TVM convention: ``-1`` append, ``-2`` prepend, ``>=0`` slot.
    ``is_reverse`` is structurally inert.

    The moved block's loops are its ENCLOSING loops (above the block, shared with
    the target) plus its BLOCK-LOCAL loops (inside the block). The target prefix
    covers the enclosing loops and the outer part of the local loops; the leftover
    local loops are the residual. We drop the prefix-covered local ForNodes and
    re-parent ``block_nid`` (now carrying only its residual loops + leaf) under the
    target. Regions are NOT recomputed: the dropped loops' vars are reintroduced
    identically by the target's own nest, so every region offset still resolves.
    ``normalize_block`` on the fork reconciles trip-1 / names there.
    """
    tree = ir.tree
    moved_seq = _moved_loop_seq(tree, block_nid)
    moved_vars = {lv for lv, _e in moved_seq}
    dep_prefix_len = sum(1 for lv, _e in _target_loop_seq(tree, target_loop_nid) if lv in moved_vars)
    enclosing_count = sum(1 for a in tree.ancestors(block_nid) if isinstance(tree.data(a), ForNode))
    local_prefix_drop = dep_prefix_len - enclosing_count
    _strip_local_prefix_loops(tree, block_nid, local_prefix_drop)
    _splice_under_target(tree, block_nid, target_loop_nid, index)
    fork = _enclosing_block(tree, target_loop_nid)
    normalize_block(tree, fork)
    _assert_single_parent(tree)


def _strip_local_prefix_loops(tree: KernelTree, block_nid: int, count: int) -> None:
    """Remove the outermost ``count`` block-local ForNodes of ``block_nid``.

    The block's body is a single chain ``block -> For -> ... -> For -> leaf``; the
    outermost ``count`` of those ForNodes are the prefix the target already provides
    (their loop vars reappear in the target's enclosing nest, so the leaf's regions
    that reference them still resolve). Each is spliced out by reconnecting its sole
    child to its parent, leaving the residual loops + leaf attached to the block.
    ``count == 0`` is a no-op (the whole moved nest is residual).
    """
    for _ in range(count):
        child = tree.children(block_nid)
        assert len(child) == 1, f"block {block_nid} body is not a single loop chain: children {child}"
        loop = child[0]
        assert isinstance(tree.data(loop), ForNode), f"expected ForNode to strip; got {type(tree.data(loop)).__name__}"
        grandchildren = tree.children(loop)
        tree.graph.remove_node(loop)
        for gc in grandchildren:
            tree.graph.add_edge(block_nid, gc)


def _target_loop_seq(tree: KernelTree, target_loop_nid: int) -> list[tuple[str, int]]:
    """Ordered ``(loop_var, extent)`` of every ForNode from outermost down to the target.

    The target's enclosing nest in TRUE tree order (outermost→innermost, all dims
    interleaved), built from the ancestor chain — not the per-dim-grouped map, which
    loses interleave order. This is the prefix the moved block must match.
    """
    chain = [*tree.ancestors(target_loop_nid), target_loop_nid]
    return [(tree.data(n).loop_var, tree.data(n).extent) for n in chain if isinstance(tree.data(n), ForNode)]


def _moved_loop_seq(tree: KernelTree, block_nid: int) -> list[tuple[str, int]]:
    """Ordered ``(loop_var, extent)`` of the moved block's full loop nest.

    Outermost→innermost: the ForNodes enclosing the block (above it) followed by the
    block-local loop chain (inside it, down to the single ISA leaf). True tree order.
    """
    leaf = next(d for d in tree.preorder(block_nid) if isinstance(tree.data(d), ISANode))
    chain = [*tree.ancestors(leaf), leaf]
    return [(tree.data(n).loop_var, tree.data(n).extent) for n in chain if isinstance(tree.data(n), ForNode)]


def _assert_single_parent(tree: KernelTree) -> None:
    """Raise loudly if any node has more than one parent after a move.

    The tree is a forest: every node except the root has exactly one parent.
    A splice that detaches a node from its old parent without removing the old
    edge leaves the node double-parented (a real corruption, not a legal-move
    distinction); failing here keeps that bug loud instead of surfacing as a
    downstream ``Dependency``/render crash on a malformed tree.
    """
    multi = [n for n in tree.graph.nodes if len(list(tree.graph.predecessors(n))) > 1]
    if multi:
        detail = {n: list(tree.graph.predecessors(n)) for n in multi}
        raise ValueError(f"_move left nodes with multiple parents: {detail}")


def _check_same_loop_prefix(ir: KernelIR, block_nid: int, target_loop_nid: int) -> list[tuple[str, int]]:
    """Raise TransformLegalityError unless the target's enclosing loops, restricted
    to the moved block's DEPENDENT dims, are an exact ``(loop_var, extent)`` prefix
    of the moved block's loop nest.

    Pure, read-only. Returns ``target_seq`` (the full target nest) for the dependency
    check. A moved block is tiled only by the dims it indexes — exactly the loop vars
    in ``moved_seq`` (its own nest). A target loop whose var the block does NOT bind
    is a *non-dependent* (DUPLICATION) loop: splicing the block under it replicates
    the block across that loop's iterations. That is correct for a pure producer (a
    reload re-writes the same buffer — e.g. an N-invariant ``lhs_T`` load reloaded per
    N-block, matching the hand kernel) and is REJECTED for an accumulation block by
    ``_check_no_reduction_replicated`` (re-running a reduction per non-tiled iteration
    corrupts). So only the dependent loops must line up:

    > legal iff ``[t for t in target_seq if t.var in moved_vars]`` is an exact prefix
    > of ``moved_seq``.

    A dependent-dim mismatch (different extent / dim / order) rejects loudly
    (``Split`` / ``Reorder`` first). This replaces the old per-dim
    ``solve_iter_domains`` coverage, whose product-matching collapsed a moved block's
    own inner loop (a memset's private ``i_d2_1``) onto a same-extent enclosing loop
    (the matmul's ``i_d2_0``), re-zeroing only one tile. The reduction guards run on
    the full target var set (non-dependent dims skip the axis-covered guard via
    ``role_of`` ``KeyError``; the replication guard uses them to reject accumulation
    duplication).
    """
    target_seq = _target_loop_seq(ir.tree, target_loop_nid)
    moved_seq = _moved_loop_seq(ir.tree, block_nid)
    moved_vars = {lv for lv, _e in moved_seq}
    dep_target = [(lv, e) for lv, e in target_seq if lv in moved_vars]
    if dep_target != moved_seq[: len(dep_target)]:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) requires the target's enclosing "
            f"loops on the moved block's DEPENDENT dims to be an exact (loop_var, extent) prefix of "
            f"the moved block's loops; dependent-target={dep_target} is not a prefix of "
            f"moved={moved_seq} (Split / Reorder the mismatched loop first)"
        )
    covered_vars = {lv for lv, _e in target_seq}
    _check_no_reduction_axis_covered(ir, block_nid, target_loop_nid, covered_vars)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, covered_vars)
    return target_seq


def _check_no_reduction_replicated(ir: KernelIR, block_nid: int, target_loop_nid: int, covered_vars: set[str]) -> None:
    """Reject sinking a reduction block under a target loop var the block does NOT
    bind (would replicate the accumulation, not re-init it).

    A block with an ACCUMULATION axis accumulates into a carried buffer
    (matmul → ``psum_prod``) whose init (memset) sits outside the block. The
    same-prefix rule already requires every target loop var to appear in the moved
    block's nest, so a clean prefix match cannot replicate. This guard remains as a
    defensive backstop: if any covered (prefix) loop var is absent from the block's
    bound loop vars, splicing under it would blindly replicate the whole K
    accumulation into the SAME accumulator region per iteration (sim: garbled, not
    NaN). PARALLEL producers replicated this way are a benign recompute; only an
    ACCUMULATION block corrupts.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    if not any(iv.role == AxisRole.ACCUMULATION for iv in block.iter_vars):
        return
    bound = {lv for lv, _e in _moved_loop_seq(ir.tree, block_nid)}
    replicated = sorted(covered_vars - bound)
    if replicated:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) replicates a reduction over "
            f"loop(s) {replicated} the block does not bind; the accumulation would re-run per "
            f"iteration into an un-reinitialised accumulator"
        )


def _own_carry_loop_nids(ir: KernelIR, block_nid: int) -> set[int]:
    """Loop NIDs that the moved block's accumulator-init DOMINATES.

    The block's RMW operand (its carried accumulator) is initialised by a
    sibling memset whose write CARRYs into the reduction loop — recorded as a
    ``CARRY`` edge ``init_writer -> loop_nid`` in ``ir.dependency``. Covering a
    reduction axis by one of these loops is the SAFE enclosing-reduction case
    (init dominates); covering by any other loop is foreign (init does not
    dominate -> NaN). Returns the loop NIDs into which the block's own
    accumulator's writers carry.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    leaf = next(d for d in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(d), ISANode))
    isa = ir.tree.data(leaf)
    acc_tensors = {
        isa.operand_bindings[slot].tensor for slot in isa.op_cls.RMW_OPERANDS if slot in isa.operand_bindings
    }
    out: set[int] = set()
    for tensor in acc_tensors:
        for writer in ir.dependency.touches_by_tensor.get(tensor, ()):
            for _w, loop_nid, attrs in ir.dependency.graph.out_edges(writer, data=True):
                if attrs.get("kind") == "CARRY":
                    out.add(loop_nid)
    return out


def _check_no_reduction_axis_covered(
    ir: KernelIR, block_nid: int, target_loop_nid: int, covered_vars: set[str]
) -> None:
    """Reject a move that covers the moved block's ACCUMULATION (reduction) axis
    with a loop the block's own init does NOT dominate (foreign covering loop).

    A reduction axis (matmul K, two-stage fold ko) must iterate as a contiguous
    nest bracketed by its init (memset) before and drain (tensor_copy) after.
    ``covered_vars`` are the prefix (target-provided) loop vars the move folds onto
    the target's nest. Covering a reduction axis by such a loop is SAFE when that
    loop is one the block's own init dominates (a CARRY edge from the block's
    accumulator-init writer into that loop NID exists); covering by any other loop
    is foreign (init does not dominate -> NaN). NID-comparison, NOT loop_var, is
    load-bearing: after RFactor a foreign K-loop and the fold's own ko-loop are BOTH
    named ``i_d0_0`` — but they are different ForNodes; only the target's actual
    prefix ForNodes count, and they are matched against the block's own-carry NIDs.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    own_carry = _own_carry_loop_nids(ir, block_nid)
    target_nid_by_var = {
        ir.tree.data(nid).loop_var: nid
        for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
        if isinstance(ir.tree.data(nid), ForNode)
    }
    result: None = None
    for lv in covered_vars:
        nid = target_nid_by_var.get(lv)
        if nid is None:
            continue
        dim = _dim_from_loopvar(lv)
        try:
            role = role_of(block, dim)
        except KeyError:
            continue
        if role != AxisRole.ACCUMULATION:
            continue
        if nid in own_carry:
            continue
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) would cover reduction axis "
            f"{dim!r} (ACCUMULATION) with loop {lv!r} the block's own init does not dominate; "
            f"a foreign covering loop breaks init-domination"
        )
    return result


def _check_move_preserves_dependencies(
    ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool
) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Pure topological query — no deep copy, no ``_move``. Asks
    ``Dependency.first_backward_edge_for_insertion`` on the **original**
    program's dependency graph: edge *directions* are frozen at construction,
    and the moved leaf's post-splice preorder position is computed analytically
    from ``(target_loop_nid, index)``. One span-based, edge-kind-agnostic rule
    covers reduction-init domination and consumer-before-producer ordering
    alike. ``is_reverse`` does not change the check — both faces forbid the same
    backward edges; only their structural splice differs.

    Directions MUST come from ``ir.dependency`` (the pre-move graph). Rebuilding
    ``Dependency`` on a moved tree would be wrong: ``_build`` re-derives every
    flow edge from execution order, so a PARALLEL producer sunk past its
    consumer flips from RAW ``producer->consumer`` to WAR ``consumer->producer``
    and the violation disappears (matmul reads uninitialised data -> NaN).
    Freezing directions keeps the RAW orientation, so the post-splice backward
    span is detected.

    Legality (same loop prefix) is a separate concern checked by
    ``_check_same_loop_prefix`` before this; here we assume a prefix-legal candidate
    and only test ordering. The matched prefix loop vars are the ones the move folds
    onto the target's nest, so their COVER edges (a full-extent consumer made
    per-tile by the move) are dissolved and skipped in the backward-edge test.
    """
    target_seq = _check_same_loop_prefix(ir, block_nid, target_loop_nid)
    covered_vars = {lv for lv, _e in target_seq}
    target_nid_by_var = {
        ir.tree.data(nid).loop_var: nid
        for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
        if isinstance(ir.tree.data(nid), ForNode)
    }
    skip_cover_loops = frozenset(target_nid_by_var[lv] for lv in covered_vars if lv in target_nid_by_var)
    moved_leaf = ir.dependency._resolve(block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(
        moved_leaf, target_loop_nid, index, skip_cover_loops=skip_cover_loops
    )
    result: None = None
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
    return result


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach ``block_nid`` from its parent and insert under the target loop at ``index``."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    children = tree.children(target_loop_nid)
    if index == -1:
        pos = len(children)
    elif index == -2:
        pos = 0
    elif index >= 0:
        pos = index
    else:
        raise ValueError(f"_splice_under_target: unsupported index {index} (use -1 append, -2 prepend, or >=0)")
    new_order = children[:pos] + [block_nid] + children[pos:]
    for child in children:
        tree.graph.remove_edge(target_loop_nid, child)
    for child in new_order:
        tree.graph.add_edge(target_loop_nid, child)


__all__ = ["_move", "_check_move_preserves_dependencies"]
