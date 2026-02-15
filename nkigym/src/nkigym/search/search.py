"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph using
program-tuple deduplication (no source rendering in the hot path).

Compilation to callable is deferred until results are returned.
Every variant is verified against the root function for numerical
correctness.
"""

import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nkigym.ir import Program, ir_to_callable, ir_to_source
from nkigym.tiling import generate_tiled_ir
from nkigym.transforms.base import Transform
from nkigym.utils import callable_to_source, import_func


def _collect_opportunities_ir(program: Program, transforms: list[Transform]) -> list[tuple[Transform, Any]]:
    """Collect all available transform opportunities for a program.

    Calls ``analyze_ir()`` on each transform and flattens the results into
    ``(transform, option)`` pairs.

    Args:
        program: Program tuple to analyze.
        transforms: List of transforms to query for opportunities.

    Returns:
        Flat list of ``(transform, option)`` pairs across all transforms.
    """
    opportunities: list[tuple[Transform, Any]] = []
    for transform in transforms:
        for option in transform.analyze_ir(program):
            opportunities.append((transform, option))
    return opportunities


@dataclass
class _Node:
    """A node in the transform state graph.

    Attributes:
        program: Program tuple for this state (used as dedup key).
        opportunities: All ``(transform, option)`` pairs available here.
        unexplored: Indices into ``opportunities`` not yet expanded.
        depth: Depth at which this node was first discovered.
    """

    program: Program
    opportunities: list[tuple[Transform, Any]]
    unexplored: list[int]
    depth: int


@dataclass
class _TransformGraph:
    """Lazily-expanded transform state graph.

    Nodes are deduplicated by program tuple (hashable, no source rendering
    needed).  The ``frontier`` tracks nodes with unexplored opportunity
    indices.  Every node (not just terminal leaves) is a valid kernel
    candidate.

    Attributes:
        nodes: Map from program tuple to ``_Node``.
        edges: Map from ``(parent_program, opp_idx)`` to child program tuple.
        frontier: Program tuples of nodes with unexplored opportunities.
        _frontier_index: Map from frontier program to its index in
            ``frontier``, enabling O(1) add/remove.
        transforms: Transforms defining the search space.
    """

    nodes: dict[Program, _Node] = field(default_factory=dict)
    edges: dict[tuple[Program, int], Program] = field(default_factory=dict)
    frontier: list[Program] = field(default_factory=list)
    _frontier_index: dict[Program, int] = field(default_factory=dict)
    transforms: list[Transform] = field(default_factory=list)

    def __init__(self, program: Program, transforms: list[Transform]) -> None:
        """Initialize the graph with a root node.

        Args:
            program: Root program tuple.
            transforms: Transforms whose opportunities define the graph.
        """
        self.nodes = {}
        self.edges = {}
        self.frontier = []
        self._frontier_index = {}
        self.transforms = transforms

        self._add_node(program, depth=0)

    def _frontier_add(self, program: Program) -> None:
        """Add a program to the frontier in O(1).

        Args:
            program: Program tuple to add.
        """
        self._frontier_index[program] = len(self.frontier)
        self.frontier.append(program)

    def _frontier_remove(self, program: Program) -> None:
        """Remove a program from the frontier in O(1) via swap-and-pop.

        Args:
            program: Program tuple to remove.
        """
        idx = self._frontier_index.pop(program)
        last = self.frontier[-1]
        if idx < len(self.frontier) - 1:
            self.frontier[idx] = last
            self._frontier_index[last] = idx
        self.frontier.pop()

    def _add_node(self, program: Program, depth: int) -> bool:
        """Register a new node in the graph.

        Args:
            program: Program tuple for this state.
            depth: Discovery depth.

        Returns:
            True if a new node was created, False if already present.
        """
        if program in self.nodes:
            return False

        opportunities = _collect_opportunities_ir(program, self.transforms)

        if not opportunities:
            self.nodes[program] = _Node(program=program, opportunities=[], unexplored=[], depth=depth)
            return True

        unexplored = list(range(len(opportunities)))
        self.nodes[program] = _Node(program=program, opportunities=opportunities, unexplored=unexplored, depth=depth)
        self._frontier_add(program)
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
        parent_program = rng.choice(self.frontier)
        node = self.nodes[parent_program]

        idx_pos = rng.randrange(len(node.unexplored))
        opp_idx = node.unexplored.pop(idx_pos)

        transform, option = node.opportunities[opp_idx]
        child_program = transform.transform_ir(node.program, option)

        self.edges[(parent_program, opp_idx)] = child_program

        if not node.unexplored:
            self._frontier_remove(parent_program)

        is_new = self._add_node(child_program, depth=node.depth + 1)
        if is_new:
            return self.nodes[child_program]
        return None


def _process_variant(
    node: _Node, depth_counts: dict[int, int], cache_dir: Path, kernel_kwargs: dict[str, Any], expected: np.ndarray
) -> None:
    """Save and verify a single qualifying variant.

    Compilation to callable is deferred until this point: only called
    when saving to disk or verifying numerical correctness.

    Args:
        node: The graph node representing this variant.
        depth_counts: Mutable map from depth to number of variants already
            saved at that depth.  Updated in place.
        cache_dir: Directory to write source files.
        kernel_kwargs: Keyword arguments for the kernel call.
        expected: Expected output for verification.

    Raises:
        AssertionError: If verification fails.
    """
    variant_idx = depth_counts.get(node.depth, 0)
    depth_counts[node.depth] = variant_idx + 1
    filename = f"nkigym_d{node.depth}_v{variant_idx}.py"
    source = ir_to_source(node.program)
    (cache_dir / filename).write_text(source)

    func = ir_to_callable(node.program)
    actual = func(**kernel_kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def search(
    func: Callable,
    transforms: list[Transform],
    num_targets: float,
    seed: int | None,
    min_depth: int,
    save_cache: Path,
    kernel_kwargs: dict[str, Any],
) -> list[Program]:
    """Search the transform state graph for unique kernel variants.

    Expands random frontier nodes until ``num_targets`` unique nodes at or
    beyond ``min_depth`` are discovered, or the frontier is exhausted.
    Use ``math.inf`` for exhaustive search.

    The search operates entirely on program tuples for deduplication and
    transform application. Compilation to callable is deferred until
    results are returned.

    Every node in the graph is a valid kernel -- not just terminal leaves.
    Nodes are unique by construction since the graph deduplicates by
    program tuple.  Termination is guaranteed even for exhaustive search
    because the set of unique program states is finite.

    Each qualifying variant is written to disk as it is discovered using
    the naming pattern ``nkigym_d{depth}_v{variant_count}.py``.
    Each variant is verified against ``func(**kernel_kwargs)`` for
    numerical correctness.

    Args:
        func: Root tiled function to start from.
        transforms: Transforms whose opportunities define the search graph.
        num_targets: Number of unique nodes with ``depth >= min_depth`` to
            collect.  ``math.inf`` expands the full graph (exhaustive).
        seed: Random seed for reproducibility.  ``None`` means unseeded RNG.
        min_depth: Minimum number of transforms that must have been applied
            for a node to count toward ``num_targets`` and be included in
            the results.
        save_cache: Directory for writing variant source files.
        kernel_kwargs: Keyword arguments for the kernel call.

    Returns:
        List of unique callables, deduplicated by program tuple.
    """
    if save_cache.exists():
        shutil.rmtree(save_cache)
    save_cache.mkdir(parents=True)
    input_path = save_cache / "nkigym_input.py"
    input_path.write_text("import numpy as np\n" + "import nkigym\n" + callable_to_source(func))
    imported_func = import_func(input_path, func.__name__)
    expected = imported_func(**kernel_kwargs)

    output_dtype = next(iter(kernel_kwargs.values())).dtype
    program = generate_tiled_ir(func, kernel_kwargs, output_dtype)
    root_source = ir_to_source(program)
    root_func_path = save_cache / "nkigym_root.py"
    root_func_path.write_text(root_source)

    depth_counts: dict[int, int] = {}

    graph = _TransformGraph(program, transforms)
    rng = random.Random(seed) if seed is not None else random.Random()

    qualifying: list[Program] = []

    root_node = next(iter(graph.nodes.values()))
    if root_node.depth >= min_depth:
        _process_variant(root_node, depth_counts, save_cache, kernel_kwargs, expected)
        qualifying.append(root_node.program)

    while len(qualifying) < num_targets and graph.frontier:
        new_node = graph.expand_one(rng)
        if new_node is not None and new_node.depth >= min_depth:
            _process_variant(new_node, depth_counts, save_cache, kernel_kwargs, expected)
            qualifying.append(new_node.program)

    return qualifying
