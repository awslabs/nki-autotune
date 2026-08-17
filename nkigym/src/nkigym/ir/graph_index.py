"""Compact reachability indexes for immutable dependency DAGs."""

from __future__ import annotations

import networkx as nx


def ordered_tree_topology(
    graph: nx.DiGraph, root: int
) -> tuple[dict[int, int], dict[int, tuple[int, ...]], dict[int, frozenset[int]]]:
    """Return preorder positions, root-first ancestors, and descendants in one traversal."""
    preorder: list[int] = []
    ancestors: dict[int, tuple[int, ...]] = {root: ()}
    pending = [root]
    while pending:
        nid = pending.pop()
        preorder.append(nid)
        children = tuple(graph.successors(nid))
        child_ancestors = (*ancestors[nid], nid)
        for child in children:
            ancestors[child] = child_ancestors
        pending.extend(reversed(children))
    mutable_descendants: dict[int, set[int]] = {nid: set() for nid in preorder}
    for nid in reversed(preorder):
        for child in graph.successors(nid):
            mutable_descendants[nid].add(child)
            mutable_descendants[nid].update(mutable_descendants[child])
    order = {nid: index for index, nid in enumerate(preorder)}
    descendants = {nid: frozenset(values) for nid, values in mutable_descendants.items()}
    return order, ancestors, descendants


class DAGReachability:
    """Precompute forward reachability as integer bitsets."""

    def __init__(self, graph: nx.DiGraph) -> None:
        """Build one reachability index for ``graph``."""
        nodes = tuple(nx.topological_sort(graph))
        self._graph = graph
        self._nodes = nodes
        self._indices = {nid: index for index, nid in enumerate(nodes)}
        self._successors: dict[int, int] = {}
        self._sets: dict[tuple[int, bool], frozenset[int]] = {}
        for nid in reversed(nodes):
            bits = 0
            for successor in graph.successors(nid):
                bits |= self._successors[successor] | (1 << self._indices[successor])
            self._successors[nid] = bits

    def __reduce__(self) -> tuple[type[DAGReachability], tuple[nx.DiGraph]]:
        """Serialize the DAG and rebuild derived bitsets in the receiving process."""
        return type(self), (self._graph,)

    def precedes(self, producer: int, consumer: int) -> bool:
        """Return whether ``consumer`` is transitively downstream of ``producer``."""
        return bool(self._successors[producer] & (1 << self._indices[consumer]))

    def nodes(self, nid: int, backward: bool) -> frozenset[int]:
        """Return cached transitive predecessors or successors of ``nid``."""
        key = (nid, backward)
        result = self._sets.get(key)
        if result is None:
            result = (
                frozenset(nx.ancestors(self._graph, nid))
                if backward
                else frozenset(self._nodes[index] for index in _set_bit_indices(self._successors[nid]))
            )
            self._sets[key] = result
        return result


def _set_bit_indices(bits: int) -> list[int]:
    """Return ascending indices of set bits."""
    result: list[int] = []
    while bits:
        lowest = bits & -bits
        result.append(lowest.bit_length() - 1)
        bits ^= lowest
    return result
