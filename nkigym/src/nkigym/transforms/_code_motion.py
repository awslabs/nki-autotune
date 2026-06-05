"""Shared structural move for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop via region-regen: solve each
moved-block dim into target-covered + residual (``_domain_solve``),
regenerate residual loops + rebind, splice under the target at ``index``,
then ``normalize_block`` reconciles names / trip-1 / region offsets on both
the moved block (its regenerated residual nest) and the fork block.
Direction (``is_reverse``) does not change the structural steps; the
caller's legality check differs.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, KernelTree, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms._domain_solve import (
    _enclosing_block,
    dim_loops_of_block,
    enclosing_dim_loops,
    regen_and_rebind,
    solve_iter_domains,
)
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import TransformLegalityError


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` in place (region-regen).

    Caller has checked legality and deep-copied. ``index`` follows TVM
    convention: ``-1`` append, ``-2`` prepend, ``>=0`` explicit slot among
    the target loop's children. ``is_reverse`` is structurally inert.

    ``normalize_block`` runs twice: once on ``block_nid`` to rebuild the
    moved block's iter_values + region ``lo`` offsets from its regenerated
    residual loops (``regen_and_rebind`` leaves these as a skeleton), and
    once on the fork block to reconcile names / trip-1 there. The two scopes
    are disjoint (``_block_local_descendants`` does not cross BlockNode
    boundaries), so the order between them is immaterial.
    """
    tree = ir.tree
    moved = dim_loops_of_block(tree, block_nid)
    target = enclosing_dim_loops(tree, target_loop_nid)
    solved = solve_iter_domains(moved, target)
    regen_and_rebind(tree, block_nid, solved)
    _splice_under_target(tree, block_nid, target_loop_nid, index)
    normalize_block(tree, block_nid)
    fork = _enclosing_block(tree, target_loop_nid)
    normalize_block(tree, fork)
    _assert_single_parent(tree)


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


def _check_move_realizable(ir: KernelIR, block_nid: int, target_loop_nid: int) -> None:
    """Raise TransformLegalityError if the target's coverage cannot cleanly
    re-domain the moved block — a pure, read-only check (no tree mutation).

    This is the realizability prefix of ``_move``: ``solve_iter_domains`` over
    the moved block's ``dim_loops_of_block`` and the target's
    ``enclosing_dim_loops`` raises ``DomainSolveError`` when a target dim's
    coverage does not divide the moved dim's extent (partial coverage that no
    residual loop can express). ``analyze`` relies on this rejection to filter
    such candidates; surfacing it as ``TransformLegalityError`` keeps that
    contract without the deep-copy ``_move`` simulation. Other structural
    invariants (single body leaf, single parent after splice) are guaranteed by
    construction for a structurally-valid candidate and are asserted inside the
    real ``_move`` when ``apply`` runs.
    """
    moved = dim_loops_of_block(ir.tree, block_nid)
    target = enclosing_dim_loops(ir.tree, target_loop_nid)
    try:
        solved = solve_iter_domains(moved, target)
    except (ValueError, KeyError) as e:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) is not structurally realizable: {e}"
        ) from e
    _check_no_reduction_axis_covered(ir, block_nid, target_loop_nid, solved)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, moved)


def _check_no_reduction_replicated(ir: KernelIR, block_nid: int, target_loop_nid: int, moved: dict) -> None:
    """Reject sinking a reduction block under a target loop iterating a dim the
    block does NOT tile (would replicate the accumulation, not re-init it).

    A block with an ACCUMULATION axis accumulates into a carried buffer
    (matmul → ``psum_prod``) whose init (memset) sits outside the block. If the
    target loop iterates a dim absent from the moved block's ``dim_loops`` — i.e.
    a dim the block writes at FULL extent (no per-tile index) — the block is
    blindly replicated across that loop's iterations, each re-running the whole
    K accumulation into the SAME PSUM region without an intervening re-init →
    the result is summed ``trip`` times (sim: partial/garbled output, not NaN).
    A PARALLEL producer (load/store/tensor_copy) replicated over such a loop is a
    benign recompute; only an ACCUMULATION block corrupts. No legal ladder move
    of a reduction block iterates a dim it lacks (verified), so this never
    over-rejects.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    if not any(iv.role == AxisRole.ACCUMULATION for iv in block.iter_vars):
        return
    target = enclosing_dim_loops(ir.tree, target_loop_nid)
    replicated = sorted(set(target) - set(moved))
    if replicated:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) replicates a reduction over "
            f"dim(s) {replicated} the block does not tile (writes them at full extent); the "
            f"accumulation would re-run per iteration into an un-reinitialised accumulator"
        )


def _check_no_reduction_axis_covered(ir: KernelIR, block_nid: int, target_loop_nid: int, solved: dict) -> None:
    """Reject a move that covers the moved block's ACCUMULATION (reduction) axis.

    A reduction axis (matmul K) must iterate as a contiguous nest the block
    OWNS, bracketed by its init (memset) before and drain (tensor_copy) after.
    ``solve_iter_domains`` marks a dim *covered* when the target's enclosing
    loops drive it (``target_loops`` non-empty, residual collapsed). Covering a
    reduction axis means the accumulation is absorbed into an enclosing loop —
    in practice one owned by a DIFFERENT block (e.g. a producer's prefetch K
    loop that merely shares the ``i_d0_0`` name) — so the block's init no longer
    dominates the reduction and it accumulates into an unzeroed / wrong-iteration
    PSUM (sim NaN). The reduction axis must stay a residual the moved block keeps
    locally; no legal ladder move ever covers one (verified). Inputs of the
    reduction (loads, whose K axis is PARALLEL on their own block) are unaffected
    — only the block that declares the axis ACCUMULATION is guarded.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    for dim, domain in solved.items():
        if not domain.target_loops:
            continue
        try:
            role = role_of(block, dim)
        except KeyError:
            continue
        if role == AxisRole.ACCUMULATION:
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) would cover reduction axis "
                f"{dim!r} (ACCUMULATION) with enclosing loops {domain.target_loops}; a reduction "
                f"must stay a private nest the block owns (init must dominate it)"
            )


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

    Realizability (target coverage divides the moved extent) is a separate
    concern checked by ``_check_move_realizable`` before this; here we assume a
    realizable candidate and only test ordering.
    """
    _check_move_realizable(ir, block_nid, target_loop_nid)
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


__all__ = ["_move", "_check_move_preserves_dependencies"]
