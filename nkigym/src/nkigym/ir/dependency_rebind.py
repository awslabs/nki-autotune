"""Fast dependency-sidecar rebinding for rewrites with invariant hazards."""

from __future__ import annotations

import copy

import networkx as nx

from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import KernelTree


def rebind_unchanged_dependency(dependency: Dependency, tree: KernelTree) -> Dependency:
    """Attach an unchanged dependency sidecar to an equivalent cloned tree."""
    result = copy.copy(dependency)
    result._tree = tree
    return result


def _clone_dependency_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Clone mutable node attributes while sharing immutable dependency edges."""
    result = nx.DiGraph()
    source = vars(graph)
    target = vars(result)
    target["graph"] = graph.graph.copy()
    target["_node"] = {nid: attrs.copy() for nid, attrs in source["_node"].items()}
    target["_succ"] = source["_succ"]
    target["_pred"] = source["_pred"]
    target["_adj"] = target["_succ"]
    return result


def rebind_exact_retile(dependency: Dependency, tree: KernelTree, block_nid: int) -> Dependency:
    """Rebind after an exact Split that preserves leaves, tensors, and hazards."""
    result = copy.copy(dependency)
    result._tree = tree
    result.graph = _clone_dependency_graph(dependency.graph)
    first_leaf = next(iter(dependency.graph.nodes), None)
    buffers = {} if first_leaf is None else dependency.info(first_leaf).buffers
    leaf_nid = dependency._leaf_of_block[block_nid]
    prior = dependency.info(leaf_nid)
    updated = result._summarise(block_nid, tree.block(block_nid), tree, buffers)
    if (updated.reads, updated.writes) != (prior.reads, prior.writes):
        raise AssertionError(f"exact retile changed tensors for block {block_nid}")
    result.graph.nodes[leaf_nid]["info"] = updated
    leaves = tuple(nid for nid in tree.preorder() if nid in dependency.graph)
    if leaves != tuple(dependency.blocks):
        raise AssertionError("exact retile changed ISA leaf execution order")
    result._topology_valid = False
    return result


__all__ = ["rebind_exact_retile", "rebind_unchanged_dependency"]
