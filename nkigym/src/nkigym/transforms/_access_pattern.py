"""Queries for ISA operands carrying explicit physical access patterns."""

from __future__ import annotations

from nkigym.ir.tree import ISANode, KernelTree


def subtree_has_access_patterns(tree: KernelTree, nid: int) -> bool:
    """Return whether ``nid`` or one of its descendants uses an access pattern."""
    nodes = (nid, *tree.descendants(nid))
    return any(
        isinstance((node := tree.data(candidate)), ISANode) and bool(node.access_patterns) for candidate in nodes
    )


def tensor_has_access_pattern(tree: KernelTree, tensor: str) -> bool:
    """Return whether any explicit access pattern references ``tensor``."""
    return any(
        isinstance((node := tree.data(nid)), ISANode)
        and any(
            slot in node.access_patterns and region.tensor == tensor for slot, region in node.operand_bindings.items()
        )
        for nid in tree.preorder()
    )


__all__ = ["subtree_has_access_patterns", "tensor_has_access_pattern"]
