"""Queries for ISA operands carrying explicit physical access patterns."""

from __future__ import annotations

from weakref import WeakKeyDictionary

from nkigym.ir.tree import ISANode, KernelTree

_PATTERN_NODES: WeakKeyDictionary[KernelTree, frozenset[int]] = WeakKeyDictionary()
_PATTERN_TENSORS: WeakKeyDictionary[KernelTree, frozenset[str]] = WeakKeyDictionary()
_SUBTREE_RESULTS: WeakKeyDictionary[KernelTree, dict[int, bool]] = WeakKeyDictionary()


def _pattern_nodes(tree: KernelTree) -> frozenset[int]:
    """Return cached ISA nodes carrying explicit access patterns."""
    nodes = _PATTERN_NODES.get(tree)
    if nodes is None:
        nodes = frozenset(
            nid
            for nid in tree.preorder()
            if isinstance((node := tree.data(nid)), ISANode) and bool(node.access_patterns)
        )
        _PATTERN_NODES[tree] = nodes
    return nodes


def _pattern_tensors(tree: KernelTree) -> frozenset[str]:
    """Return cached tensors referenced by explicit access patterns."""
    tensors = _PATTERN_TENSORS.get(tree)
    if tensors is None:
        tensors = frozenset(
            region.tensor
            for nid in _pattern_nodes(tree)
            for slot, region in tree.isa(nid).operand_bindings.items()
            if slot in tree.isa(nid).access_patterns
        )
        _PATTERN_TENSORS[tree] = tensors
    return tensors


def subtree_has_access_patterns(tree: KernelTree, nid: int) -> bool:
    """Return whether ``nid`` or one of its descendants uses an access pattern."""
    pattern_nodes = _pattern_nodes(tree)
    if not pattern_nodes:
        return False
    result = _SUBTREE_RESULTS.setdefault(tree, {}).get(nid)
    if result is None:
        result = bool(({nid} | tree.descendants(nid)) & pattern_nodes)
        _SUBTREE_RESULTS[tree][nid] = result
    return result


def tensor_has_access_pattern(tree: KernelTree, tensor: str) -> bool:
    """Return whether any explicit access pattern references ``tensor``."""
    return tensor in _pattern_tensors(tree)


__all__ = ["subtree_has_access_patterns", "tensor_has_access_pattern"]
