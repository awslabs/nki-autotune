"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph;
source-based deduplication prunes equivalent states reached via
different orderings.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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

    def _add_node(self, func: Callable, src: str, depth: int) -> bool:
        """Register a new node in the graph.

        Args:
            func: Callable for this state.
            src: Source string (used as key).
            depth: Discovery depth.

        Returns:
            True if a new node was created, False if already present.
        """
        if src in self.nodes:
            return False

        opportunities = _collect_opportunities(func, self.transforms)

        if not opportunities:
            self.nodes[src] = _Node(src=src, func=func, opportunities=[], unexplored=[], depth=depth)
            return True

        unexplored = list(range(len(opportunities)))
        self.nodes[src] = _Node(src=src, func=func, opportunities=opportunities, unexplored=unexplored, depth=depth)
        self._frontier_add(src)
        return True

    def expand_one(self, rng: random.Random) -> _Node | None:
        """Expand one unexplored opportunity from a random frontier node.

        Picks a random frontier node, picks a random unexplored opportunity
        index, applies the transform, and records the edge.  If the frontier
        node has no remaining unexplored opportunities after expansion it is
        removed from the frontier.

        Args:
            rng: Random number generator for selection.

        Returns:
            The newly created node, or None if the child was a duplicate.
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

        is_new = self._add_node(child_func, child_src, depth=node.depth + 1)
        if is_new:
            return self.nodes[child_src]
        return None


def _process_variant(
    node: _Node,
    func_name: str,
    depth_counts: dict[int, int],
    cache_dir: Path | None,
    verify: bool,
    kernel_kwargs: dict[str, Any] | None,
    expected: np.ndarray | None,
) -> None:
    """Save and/or verify a single qualifying variant.

    Args:
        node: The graph node representing this variant.
        func_name: Base name for cache files (from the root function).
        depth_counts: Mutable map from depth to number of variants already
            saved at that depth.  Updated in place.
        cache_dir: Directory to write source files, or None to skip.
        verify: Whether to verify numerical correctness.
        kernel_kwargs: Keyword arguments for verification call.
        expected: Expected output for verification.

    Raises:
        AssertionError: If verification fails.
    """
    if cache_dir is not None:
        variant_idx = depth_counts.get(node.depth, 0)
        depth_counts[node.depth] = variant_idx + 1
        filename = f"{func_name}_d{node.depth}_v{variant_idx}.py"
        source = get_source(node.func)
        (cache_dir / filename).write_text(f'"""variant d{node.depth} v{variant_idx}"""\n' + source)

    if verify:
        assert kernel_kwargs is not None and expected is not None
        actual = node.func(**kernel_kwargs)
        np.testing.assert_allclose(actual, expected)


def search(
    func: Callable,
    transforms: list[Transform],
    num_targets: float,
    seed: int | None,
    min_depth: int,
    save_cache: Path | str | None,
    verify: bool,
    kernel_kwargs: dict[str, Any] | None,
) -> list[Callable]:
    """Search the transform state graph for unique kernel variants.

    Expands random frontier nodes until ``num_targets`` unique nodes at or
    beyond ``min_depth`` are discovered, or the frontier is exhausted.
    Use ``math.inf`` for exhaustive search.

    Every node in the graph is a valid kernel — not just terminal leaves.
    Nodes are unique by construction since the graph deduplicates by
    source.  Termination is guaranteed even for exhaustive search because
    the set of unique source states is finite.

    When ``save_cache`` is set, each qualifying variant is written to disk
    as it is discovered using the naming pattern
    ``{func_name}_d{depth}_v{variant_count}.py``.  When ``verify`` is
    True, each variant is checked against ``func(**kernel_kwargs)`` inline.

    Args:
        func: Root tiled function to start from.
        transforms: Transforms whose opportunities define the search graph.
        num_targets: Number of unique nodes with ``depth >= min_depth`` to
            collect.  ``math.inf`` expands the full graph (exhaustive).
        seed: Random seed for reproducibility.  ``None`` means unseeded RNG.
        min_depth: Minimum number of transforms that must have been applied
            for a node to count toward ``num_targets`` and be included in
            the results.
        save_cache: Directory path for writing variant source files as they
            are discovered.  ``None`` disables caching.
        verify: When True, verify each variant against
            ``func(**kernel_kwargs)`` using ``np.testing.assert_allclose``.
        kernel_kwargs: Keyword arguments passed to each variant for
            verification.  Required when ``verify`` is True.

    Returns:
        List of unique callables, deduplicated by source.

    Raises:
        ValueError: If ``verify`` is True but ``kernel_kwargs`` is None.
    """
    if verify and kernel_kwargs is None:
        raise ValueError("verify=True requires kernel_kwargs to be provided")

    cache_dir: Path | None = None
    if save_cache is not None:
        cache_dir = Path(save_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)

    expected: np.ndarray | None = None
    if verify:
        assert kernel_kwargs is not None
        expected = func(**kernel_kwargs)

    func_name = getattr(func, "__name__", "variant")
    depth_counts: dict[int, int] = {}

    graph = _TransformGraph(func, transforms)
    rng = random.Random(seed) if seed is not None else random.Random()

    qualifying: list[_Node] = []

    root_node = next(iter(graph.nodes.values()))
    if root_node.depth >= min_depth:
        _process_variant(root_node, func_name, depth_counts, cache_dir, verify, kernel_kwargs, expected)
        qualifying.append(root_node)

    while len(qualifying) < num_targets and graph.frontier:
        new_node = graph.expand_one(rng)
        if new_node is not None and new_node.depth >= min_depth:
            _process_variant(new_node, func_name, depth_counts, cache_dir, verify, kernel_kwargs, expected)
            qualifying.append(new_node)

    return [node.func for node in qualifying]
