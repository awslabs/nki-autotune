"""Shared low-level tree-mutation helpers for transforms.

The networkx ``DiGraph`` does not preserve sibling order when nodes are
added or removed in arbitrary order — ``add_edge`` always appends to the
predecessor's successor list. Transforms that replace one or more
contiguous children of a parent must therefore rewrite the parent's full
out-edge list to keep sibling order stable.
"""

from __future__ import annotations

from nkigym.ir.tree import KernelTree


def _replace_in_parent_children(
    tree: KernelTree, parent_nid: int, old_children: list[int], new_children: list[int]
) -> None:
    """Replace ``old_children`` with ``new_children`` at the same position in ``parent_nid``'s child list.

    ``old_children`` must be a contiguous slice of ``tree.children(parent_nid)``
    in order. The function wipes ``parent_nid``'s out-edges and re-adds them
    so the new children occupy the slot the old children occupied; all other
    siblings keep their relative order.

    The nodes themselves are not removed from the graph — caller is
    responsible for any subsequent ``remove_node`` cleanup of orphaned old
    children.
    """
    siblings_before = tree.children(parent_nid)
    start = siblings_before.index(old_children[0])
    assert siblings_before[start : start + len(old_children)] == list(old_children), (
        f"_replace_in_parent_children: old_children {old_children} is not a contiguous "
        f"slice of parent_nid={parent_nid} children {siblings_before}"
    )
    new_order = siblings_before[:start] + list(new_children) + siblings_before[start + len(old_children) :]
    for child in siblings_before:
        tree.graph.remove_edge(parent_nid, child)
    for child in new_order:
        tree.graph.add_edge(parent_nid, child)


__all__ = ["_replace_in_parent_children"]
