"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph;
source-based deduplication prunes equivalent states reached via
different orderings.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nkigym.transforms.base import Transform
from nkigym.utils.source import get_source


def _collect_opportunities(func: Callable, transforms: list[Transform]) -> list[tuple[Transform, Any]]:
    """Collect all available transform opportunities for a function.

    Calls ``analyze()`` on each transform and flattens the results into
    ``(transform, option)`` pairs.

    Args:
        func: A tiled function to analyze.
        transforms: List of transforms to query for opportunities.

    Returns:
        Flat list of ``(transform, option)`` pairs across all transforms.
    """
    opportunities: list[tuple[Transform, Any]] = []
    for transform in transforms:
        for option in transform.analyze(func):
            opportunities.append((transform, option))
    return opportunities


@dataclass
class _Node:
    """A node in the transform state graph.

    Args:
        src: Normalized source code for this state.
        func: The callable representing this state.
        opportunities: All ``(transform, option)`` pairs available here.
        unexplored: Indices into ``opportunities`` not yet expanded.
        depth: Depth at which this node was first discovered.
    """

    src: str
    func: Callable
    opportunities: list[tuple[Transform, Any]]
    unexplored: list[int]
    depth: int


@dataclass
class _TransformGraph:
    """Lazily-expanded transform state graph.

    Nodes are deduplicated by source.  The ``frontier`` tracks nodes with
    unexplored opportunity indices.  Every node (not just terminal leaves)
    is a valid kernel candidate.

    Args:
        nodes: Map from source string to ``_Node``.
        edges: Map from ``(parent_src, opp_idx)`` to child source string.
        frontier: Source strings of nodes with unexplored opportunities.
        _frontier_index: Map from frontier source string to its index in
            ``frontier``, enabling O(1) add/remove.
        transforms: Transforms defining the search space.
    """

    nodes: dict[str, _Node] = field(default_factory=dict)
    edges: dict[tuple[str, int], str] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)
    _frontier_index: dict[str, int] = field(default_factory=dict)
    transforms: list[Transform] = field(default_factory=list)

    def __init__(self, func: Callable, transforms: list[Transform]) -> None:
        """Initialize the graph with a root node.

        Args:
            func: Root tiled function.
            transforms: Transforms whose opportunities define the graph.
        """
        self.nodes = {}
        self.edges = {}
        self.frontier = []
        self._frontier_index = {}
        self.transforms = transforms

        root_src = get_source(func)
        self._add_node(func, root_src, depth=0)

    def _frontier_add(self, src: str) -> None:
        """Add a source string to the frontier in O(1).

        Args:
            src: Source string to add.
        """
        self._frontier_index[src] = len(self.frontier)
        self.frontier.append(src)

    def _frontier_remove(self, src: str) -> None:
        """Remove a source string from the frontier in O(1) via swap-and-pop.

        Args:
            src: Source string to remove.
        """
        idx = self._frontier_index.pop(src)
        last = self.frontier[-1]
        if idx < len(self.frontier) - 1:
            self.frontier[idx] = last
            self._frontier_index[last] = idx
        self.frontier.pop()

    def _add_node(self, func: Callable, src: str, depth: int) -> None:
        """Register a new node in the graph.

        Args:
            func: Callable for this state.
            src: Source string (used as key).
            depth: Discovery depth.
        """
        if src in self.nodes:
            return

        opportunities = _collect_opportunities(func, self.transforms)

        if not opportunities:
            self.nodes[src] = _Node(src=src, func=func, opportunities=[], unexplored=[], depth=depth)
            return

        unexplored = list(range(len(opportunities)))
        self.nodes[src] = _Node(src=src, func=func, opportunities=opportunities, unexplored=unexplored, depth=depth)
        self._frontier_add(src)

    def expand_one(self, rng: random.Random) -> None:
        """Expand one unexplored opportunity from a random frontier node.

        Picks a random frontier node, picks a random unexplored opportunity
        index, applies the transform, and records the edge.  If the frontier
        node has no remaining unexplored opportunities after expansion it is
        removed from the frontier.

        Args:
            rng: Random number generator for selection.
        """
        parent_src = rng.choice(self.frontier)
        node = self.nodes[parent_src]

        idx_pos = rng.randrange(len(node.unexplored))
        opp_idx = node.unexplored.pop(idx_pos)

        transform, option = node.opportunities[opp_idx]
        child_func = transform.transform(node.func, option)
        child_src = get_source(child_func)

        self.edges[(parent_src, opp_idx)] = child_src

        if not node.unexplored:
            self._frontier_remove(parent_src)

        self._add_node(child_func, child_src, depth=node.depth + 1)


def search(
    func: Callable, transforms: list[Transform], num_targets: float = math.inf, seed: int | None = None
) -> list[Callable]:
    """Search the transform state graph for unique kernel variants.

    Expands random frontier nodes until ``num_targets`` unique nodes are
    discovered or the frontier is exhausted.  Use ``math.inf`` (the
    default) for exhaustive search.

    Every node in the graph is a valid kernel — not just terminal leaves.
    Nodes are unique by construction since the graph deduplicates by
    source.  Termination is guaranteed even for exhaustive search because
    the set of unique source states is finite.

    Args:
        func: Root tiled function to start from.
        transforms: Transforms whose opportunities define the search graph.
        num_targets: Number of unique nodes to discover.  ``math.inf``
            expands the full graph (exhaustive).
        seed: Random seed for reproducibility.  ``None`` means unseeded RNG.

    Returns:
        List of unique callables, deduplicated by source.
    """
    graph = _TransformGraph(func, transforms)
    rng = random.Random(seed) if seed is not None else random.Random()
    while len(graph.nodes) < num_targets and graph.frontier:
        graph.expand_one(rng)
    return [node.func for node in graph.nodes.values()] or [func]
