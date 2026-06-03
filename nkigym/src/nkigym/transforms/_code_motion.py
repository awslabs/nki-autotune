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


__all__ = ["_move"]
