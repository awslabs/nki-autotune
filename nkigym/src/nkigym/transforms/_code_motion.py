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

import copy

import networkx as nx

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import KernelTree
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


def _check_move_preserves_dependencies(
    ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool
) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Simulates the move on a deep copy, rebuilds the Dependency graph on the
    moved tree, and asks ``first_backward_edge`` for the moved leaf. One
    span-based, edge-kind-agnostic rule — covers reduction-init domination
    and consumer-before-producer ordering alike.

    The structural ``_move`` can fail on a candidate that is impossible to
    splice (e.g. the target loop is stripped before splicing, or the target's
    coverage does not divide a moved dim). Those are "this move cannot be
    realized" signals, not crashes; they are converted to
    ``TransformLegalityError`` so callers (notably ``analyze``) filter the
    candidate instead of propagating a structural exception.
    """
    sim = copy.deepcopy(ir)
    try:
        _move(sim, block_nid=block_nid, target_loop_nid=target_loop_nid, index=index, is_reverse=is_reverse)
    except (nx.NetworkXError, ValueError, KeyError, AssertionError) as e:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) is not structurally realizable: {e}"
        ) from e
    dep = Dependency(sim.tree)
    moved_leaf = dep._resolve(block_nid)
    offending = dep.first_backward_edge(moved_leaf)
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
