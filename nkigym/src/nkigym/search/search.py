"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph using
program-tuple deduplication (no source rendering in the hot path).

Every new node is verified against the root function for numerical
correctness.  Qualifying variants (depth >= min_depth) are saved to disk.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nkigym.ir import GymProgram, program_to_source, source_to_program
from nkigym.tiling import tile_program
from nkigym.transforms.base import Transform
from nkigym.utils import callable_to_source, import_func
from nkigym.utils.source import source_to_callable

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 500


def _collect_opportunities_ir(program: GymProgram, transforms: list[Transform]) -> list[tuple[Transform, Any]]:
    """Collect all available transform opportunities for a program.

    Calls ``analyze_ir()`` on each transform and flattens the results into
    ``(transform, option)`` pairs.

    Args:
        program: GymProgram tuple to analyze.
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
        program: GymProgram tuple for this state (used as dedup key).
        opportunities: All ``(transform, option)`` pairs available here.
        unexplored: Indices into ``opportunities`` not yet expanded.
        depth: Depth at which this node was first discovered.
    """

    program: GymProgram
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
        edges: Map from ``(parent_program, opp_idx)`` to child program.
        frontier: GymProgram tuples with unexplored opportunities.
        _frontier_index: Frontier program to index for O(1) removal.
        transforms: Transforms defining the search space.
        total_expansions: Count of ``expand_one`` calls.
    """

    nodes: dict[GymProgram, _Node] = field(default_factory=dict)
    edges: dict[tuple[GymProgram, int], GymProgram] = field(default_factory=dict)
    frontier: list[GymProgram] = field(default_factory=list)
    _frontier_index: dict[GymProgram, int] = field(default_factory=dict)
    transforms: list[Transform] = field(default_factory=list)
    total_expansions: int = 0

    def __init__(self, program: GymProgram, transforms: list[Transform]) -> None:
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
        self.total_expansions = 0

        self._add_node(program, depth=0)

    def _frontier_add(self, program: GymProgram) -> None:
        """Add a program to the frontier in O(1).

        Args:
            program: GymProgram tuple to add.
        """
        self._frontier_index[program] = len(self.frontier)
        self.frontier.append(program)

    def _frontier_remove(self, program: GymProgram) -> None:
        """Remove a program from the frontier in O(1) via swap-and-pop.

        Args:
            program: GymProgram tuple to remove.
        """
        idx = self._frontier_index.pop(program)
        last = self.frontier[-1]
        if idx < len(self.frontier) - 1:
            self.frontier[idx] = last
            self._frontier_index[last] = idx
        self.frontier.pop()

    def _add_node(self, program: GymProgram, depth: int) -> bool:
        """Register a new node in the graph.

        Args:
            program: GymProgram tuple for this state.
            depth: Discovery depth.

        Returns:
            True if a new node was created, False if already present.
        """
        is_new = program not in self.nodes
        if is_new:
            opportunities = _collect_opportunities_ir(program, self.transforms)
            unexplored = list(range(len(opportunities)))
            self.nodes[program] = _Node(
                program=program, opportunities=opportunities, unexplored=unexplored, depth=depth
            )
            if opportunities:
                self._frontier_add(program)
        return is_new

    def expand_one(self, rng: random.Random) -> tuple[bool, GymProgram]:
        """Expand one unexplored opportunity from a random frontier node.

        Picks a random frontier node, picks a random unexplored opportunity
        index, applies the transform, and records the edge.

        Args:
            rng: Random number generator for selection.

        Returns:
            Tuple of (is_new, child_program).  is_new is True when the
            child was a previously unseen program.
        """
        self.total_expansions += 1

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
        return is_new, child_program


def _verify_node(node: _Node, kernel_kwargs: dict[str, Any], expected: np.ndarray) -> None:
    """Compile and run a node, assert numerical correctness.

    Args:
        node: Graph node to verify.
        kernel_kwargs: Keyword arguments for the kernel call.
        expected: Expected output array.

    Raises:
        AssertionError: If numerical verification fails.
    """
    source = program_to_source(node.program)
    func = source_to_callable(source, node.program.name)
    actual = func(**kernel_kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def _save_variant(node: _Node, depth_counts: dict[int, int], cache_dir: Path) -> None:
    """Write a qualifying variant source file to disk.

    Args:
        node: Graph node to save.
        depth_counts: Map from depth to variant count; updated in place.
        cache_dir: Directory to write the source file.
    """
    variant_idx = depth_counts.get(node.depth, 0)
    depth_counts[node.depth] = variant_idx + 1
    filename = f"nkigym_d{node.depth}_v{variant_idx}.py"
    source = program_to_source(node.program)
    (cache_dir / filename).write_text(source)


def _format_depth_table(depth_node_counts: dict[int, int], min_depth: int) -> str:
    """Format depth distribution as a two-row aligned table.

    Uses a uniform column width (minimum 4) so columns stay stable
    across successive progress lines.  Non-qualifying and qualifying
    depths are separated by ``|``.

    Args:
        depth_node_counts: Map from depth to node count.
        min_depth: Minimum depth threshold for qualifying variants.

    Returns:
        Two-line string with aligned depth and node-count rows.
    """
    sorted_items = sorted(depth_node_counts.items())
    depths = [str(d) for d, _ in sorted_items]
    counts = [str(c) for _, c in sorted_items]
    col_w = max([4] + [len(s) for s in depths] + [len(s) for s in counts])

    left_d: list[str] = []
    left_c: list[str] = []
    right_d: list[str] = []
    right_c: list[str] = []
    for i, (d, _) in enumerate(sorted_items):
        cell_d = depths[i].rjust(col_w)
        cell_c = counts[i].rjust(col_w)
        if d >= min_depth:
            right_d.append(cell_d)
            right_c.append(cell_c)
        else:
            left_d.append(cell_d)
            left_c.append(cell_c)

    sep = " | " if left_d and right_d else ""
    depth_line = " ".join(left_d) + sep + " ".join(right_d)
    count_line = " ".join(left_c) + sep + " ".join(right_c)
    return f"  depth  {depth_line}\n  nodes  {count_line}"


def _log_progress(unique_count: int, qualifying_count: int, depth_node_counts: dict[int, int], min_depth: int) -> None:
    """Log periodic search progress with depth distribution table."""
    table = _format_depth_table(depth_node_counts, min_depth)
    logger.info("Progress: %d unique (%d qualifying)\n%s", unique_count, qualifying_count, table)


def _prepare_root(func: Callable, save_cache: Path, kernel_kwargs: dict[str, Any]) -> tuple[GymProgram, np.ndarray]:
    """Tile the user function and compute expected output.

    Writes ``nkigym_input.py`` and ``nkigym_root.py`` to the cache
    directory.

    Args:
        func: Root user function.
        save_cache: Directory for writing files.
        kernel_kwargs: Keyword arguments for the kernel call.

    Returns:
        Tuple of (tiled GymProgram, expected output array).
    """
    input_path = save_cache / "nkigym_input.py"
    input_path.write_text("import numpy as np\nimport nkigym\n" + callable_to_source(func))
    imported_func = import_func(input_path, func.__name__)
    expected = imported_func(**kernel_kwargs)

    output_dtype = next(iter(kernel_kwargs.values())).dtype
    input_shapes = {k: v.shape for k, v in kernel_kwargs.items()}
    source = callable_to_source(func)
    program = tile_program(source_to_program(source, input_shapes, output_dtype))
    (save_cache / "nkigym_root.py").write_text(program_to_source(program))
    return program, expected


def _run_search(
    graph: _TransformGraph,
    rng: random.Random,
    min_depth: int,
    num_targets: float,
    save_cache: Path,
    kernel_kwargs: dict[str, Any],
    expected: np.ndarray,
) -> list[GymProgram]:
    """Execute search loop: verify every node, save qualifying ones.

    Returns:
        Qualifying GymProgram list.
    """
    qualifying: list[GymProgram] = []
    depth_counts: dict[int, int] = {}
    depth_node_counts: dict[int, int] = {0: 1}
    last_progress_count = 0

    root_node = next(iter(graph.nodes.values()))
    _verify_node(root_node, kernel_kwargs, expected)
    if root_node.depth >= min_depth:
        _save_variant(root_node, depth_counts, save_cache)
        qualifying.append(root_node.program)

    while len(qualifying) < num_targets and graph.frontier:
        is_new, child_program = graph.expand_one(rng)
        if not is_new:
            continue
        child_node = graph.nodes[child_program]
        child_depth = child_node.depth
        depth_node_counts[child_depth] = depth_node_counts.get(child_depth, 0) + 1
        _verify_node(child_node, kernel_kwargs, expected)
        if child_depth >= min_depth:
            _save_variant(child_node, depth_counts, save_cache)
            qualifying.append(child_node.program)
        unique_count = len(graph.nodes)
        if unique_count - last_progress_count >= PROGRESS_INTERVAL:
            _log_progress(unique_count, len(qualifying), depth_node_counts, min_depth)
            last_progress_count = unique_count

    _log_search_summary(graph, qualifying, min_depth)
    return qualifying


def _log_search_summary(graph: _TransformGraph, qualifying: list[GymProgram], min_depth: int) -> None:
    """Log final search summary with deduplication statistics."""
    unique = len(graph.nodes)
    expansions = graph.total_expansions
    duplicates = expansions - (unique - 1)
    if expansions > 0:
        dedup_pct = duplicates / expansions * 100
    else:
        dedup_pct = 0.0
    logger.info("Search complete: %d unique programs, %d qualifying (depth >= %d)", unique, len(qualifying), min_depth)
    logger.info("Deduplication: %d expansions, %d duplicates (%.1f%%)", expansions, duplicates, dedup_pct)


def search(
    func: Callable,
    transforms: list[Transform],
    num_targets: float,
    seed: int,
    min_depth: int,
    save_cache: Path,
    kernel_kwargs: dict[str, Any],
) -> list[GymProgram]:
    """Search the transform state graph for unique kernel variants.

    Expands random frontier nodes until ``num_targets`` unique nodes at or
    beyond ``min_depth`` are discovered, or the frontier is exhausted.
    Every node is verified for numerical correctness.

    Args:
        func: Root tiled function.
        transforms: Transforms defining the search graph.
        num_targets: Qualifying variant target (``math.inf`` for exhaustive).
        seed: Random seed for reproducibility.
        min_depth: Minimum transform depth for qualifying variants.
        save_cache: Directory for variant source files.
        kernel_kwargs: Keyword arguments for the kernel call.

    Returns:
        List of qualifying GymProgram tuples.
    """
    save_cache.mkdir(parents=True, exist_ok=True)
    program, expected = _prepare_root(func, save_cache, kernel_kwargs)

    graph = _TransformGraph(program, transforms)
    rng = random.Random(seed)
    logger.info("Search root: %d stmts, %d opportunities", len(program.stmts), len(graph.nodes[program].opportunities))

    return _run_search(graph, rng, min_depth, num_targets, save_cache, kernel_kwargs, expected)
