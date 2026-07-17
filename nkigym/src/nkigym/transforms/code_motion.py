"""Shared structural move for CodeMotion (the merged compute-at / reverse-compute-at).

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
first), so every legal move is a clean structural merge.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.transforms._domain_solve import _enclosing_block
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` by a verbatim residual splice.

    Caller has checked legality (``_check_same_loop_prefix``, so the target's
    enclosing nest is an exact prefix of the moved block's nest) and deep-copied.
    ``index`` follows TVM convention: ``-1`` append, ``-2`` prepend, ``>=0`` slot.

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
    (the matmul's ``i_d2_0``), re-zeroing only one tile. The replication guard uses
    the full target var set to reject accumulation duplication.
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


def _check_move_preserves_dependencies(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Pure topological query — no deep copy, no ``_move``. Asks
    ``Dependency.first_backward_edge_for_insertion`` on the **original**
    program's dependency graph: edge *directions* are frozen at construction,
    and the moved leaf's post-splice preorder position is computed analytically
    from ``(target_loop_nid, index)``. Span-promotion delivers reduction-init
    domination and coverage — one span-based, edge-kind-agnostic rule covers
    both reduction-init domination and consumer-before-producer ordering.

    Directions MUST come from ``ir.dependency`` (the pre-move graph). Rebuilding
    ``Dependency`` on a moved tree would be wrong: ``_build`` re-derives every
    flow edge from execution order, so a PARALLEL producer sunk past its
    consumer flips from RAW ``producer->consumer`` to WAR ``consumer->producer``
    and the violation disappears (matmul reads uninitialised data -> NaN).
    Freezing directions keeps the RAW orientation, so the post-splice backward
    span is detected.
    """
    _check_same_loop_prefix(ir, block_nid, target_loop_nid)
    moved_leaf = ir.dependency._resolve(block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)
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


@dataclass(frozen=True)
class CodeMotionOption(TransformOption):
    """Relocate ``block_nid`` under ``target_loop_nid`` at child slot ``index``.

    One option type for both directions of motion: sinking a producer under a
    consumer's loop and lifting a consumer under a producer's loop are the same
    structural splice, distinguished only by the dependency graph — not a flag.
    """

    block_nid: int
    target_loop_nid: int
    index: int


class CodeMotion(Transform):
    """Relocate one block under a target loop (the merged former ComputeAt/ReverseComputeAt).

    Legality is dependency-ordering (span-promotion) + the structural same-prefix
    merge + the reduction-replication guard. There is NO output-block guard: the
    block writing the return tensor is relocatable when ordering permits (e.g. the
    k11->k12 store-sink under the matmul's N loop).
    """

    def apply(self, ir: KernelIR, option: CodeMotionOption) -> KernelIR:
        """Re-check legality, deep-copy, move, rebuild deps, return.

        Structural-only: the block relocation + Dependency rebuild. Buffer
        placement/shape/frame is now an explicit BufferCompaction step, not an
        anonymous tail (see the 2026-07-14 BufferCompaction design).
        """
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[CodeMotionOption]:
        """Enumerate (block, target loop, index) triples passing legality."""
        options: list[CodeMotionOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                for index in self._legal_indices(ir, block_nid, target_nid):
                    opt = CodeMotionOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    try:
                        self._check_legality(ir, opt)
                    except TransformLegalityError:
                        continue
                    options.append(opt)
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children.

        Bounded below by the last child holding a producer of the moved block and
        above by the first child holding a consumer — symmetric in both, which is
        why one enumeration serves producer-sink and consumer-lift alike.
        """
        children = ir.tree.children(target_nid)
        producers = ir.dependency.producers(block_nid)
        consumers = ir.dependency.consumers(block_nid)
        lp = -1
        fc = len(children)
        for i, child in enumerate(children):
            sub = ir.tree.descendants(child) | {child}
            if sub & producers:
                lp = i
            if sub & consumers and i < fc:
                fc = i
        return list(range(lp + 1, fc + 1))

    def _check_legality(self, ir: KernelIR, option: CodeMotionOption) -> None:
        """Structural checks (target/block in graph, target a ForNode, target not a
        descendant of the block) then span-promotion ordering. No output guard."""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"CodeMotion requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot move under its own loop)"
            )
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index)


__all__ = ["_move", "_check_move_preserves_dependencies", "CodeMotion", "CodeMotionOption"]
