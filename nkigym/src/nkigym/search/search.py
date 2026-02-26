"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph using
program-tuple deduplication (no source rendering in the hot path).

Every new node is verified against the root function for numerical
correctness.  Qualifying variants (depth >= min_depth) are saved to disk.
"""

import ast
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nkigym.codegen.gym_to_nki import lower_to_nki
from nkigym.codegen.loop_rolling import roll_loops
from nkigym.ir import GymProgram, program_to_source, source_to_program
from nkigym.search.compile import (  # noqa: F401
    CompilationPool,
    SearchResults,
    VariantResult,
    _capture_error,
    run_on_hardware,
)
from nkigym.search.interpret import interpret_program
from nkigym.search.mac_count import compute_mac_count
from nkigym.search.report import SearchReport
from nkigym.tiling import tile_program
from nkigym.transforms.base import Transform
from nkigym.utils import callable_to_source, import_func

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 500
_WARMUP = 10
_ITERS = 100

_NL_AFFINE_RANGE = ast.Attribute(value=ast.Name(id="nl", ctx=ast.Load()), attr="affine_range", ctx=ast.Load())


class _RangeToAffine(ast.NodeTransformer):
    """Replace ``range(N)`` with ``nl.affine_range(N)`` in for-loop iterators."""

    def visit_For(self, node: ast.For) -> ast.For:
        """Rewrite for-loop iterator from range to nl.affine_range."""
        self.generic_visit(node)
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            it.func = ast.copy_location(_NL_AFFINE_RANGE, it.func)
        return node


def _use_affine_range(source: str) -> str:
    """Replace ``range`` with ``nl.affine_range`` in for-loop iterators.

    Args:
        source: NKI kernel source code string.

    Returns:
        Source with for-loop ranges replaced by nl.affine_range.
    """
    tree = ast.parse(source)
    tree = _RangeToAffine().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _collect_opportunities_ir(program: GymProgram, transforms: list[Transform]) -> list[tuple[Transform, Any]]:
    """Collect all ``(transform, option)`` pairs available for a program."""
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
        variant_idx: Per-depth variant counter, assigned when qualifying.
    """

    program: GymProgram
    opportunities: list[tuple[Transform, Any]]
    unexplored: list[int]
    depth: int
    variant_idx: int


@dataclass
class _TransformGraph:
    """Lazily-expanded transform state graph.

    Attributes:
        nodes: Map from program tuple to ``_Node``.
        edges: Map from ``(parent_program, opp_idx)`` to child program.
        frontier: GymProgram tuples with unexplored opportunities.
        transforms: Transforms defining the search space.
        total_programs_visited: Total programs visited (root + expansions).
    """

    nodes: dict[GymProgram, _Node] = field(default_factory=dict)
    edges: dict[tuple[GymProgram, int], GymProgram] = field(default_factory=dict)
    frontier: list[GymProgram] = field(default_factory=list)
    _frontier_index: dict[GymProgram, int] = field(default_factory=dict)
    transforms: list[Transform] = field(default_factory=list)
    total_programs_visited: int = 0

    def __init__(self, program: GymProgram, transforms: list[Transform]) -> None:
        """Initialize the graph with a root node."""
        self.nodes = {}
        self.edges = {}
        self.frontier = []
        self._frontier_index = {}
        self.transforms = transforms
        self.total_programs_visited = 1
        self._add_node(program, depth=0)

    def _frontier_add(self, program: GymProgram) -> None:
        """Add a program to the frontier in O(1)."""
        self._frontier_index[program] = len(self.frontier)
        self.frontier.append(program)

    def _frontier_remove(self, program: GymProgram) -> None:
        """Remove a program from the frontier in O(1) via swap-and-pop."""
        idx = self._frontier_index.pop(program)
        last = self.frontier[-1]
        if idx < len(self.frontier) - 1:
            self.frontier[idx] = last
            self._frontier_index[last] = idx
        self.frontier.pop()

    def _add_node(self, program: GymProgram, depth: int) -> bool:
        """Register a new node in the graph.

        Returns:
            True if a new node was created, False if already present.
        """
        is_new = program not in self.nodes
        if is_new:
            opportunities = _collect_opportunities_ir(program, self.transforms)
            unexplored = list(range(len(opportunities)))
            self.nodes[program] = _Node(
                program=program, opportunities=opportunities, unexplored=unexplored, depth=depth, variant_idx=-1
            )
            if opportunities:
                self._frontier_add(program)
        return is_new

    def expand_one(self, rng: random.Random) -> tuple[bool, GymProgram]:
        """Expand one unexplored opportunity from a random frontier node.

        Returns:
            Tuple of (is_new, child_program).
        """
        self.total_programs_visited += 1
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


def _verify_node(node: _Node, kernel_kwargs: dict[str, Any], expected: np.ndarray, tol: float) -> None:
    """Interpret a node's program directly and check numerical correctness."""
    actual = interpret_program(node.program, kernel_kwargs)
    if np.max(np.abs(actual - expected)) > tol:
        raise AssertionError("Variant failed numerical verification")


def _assign_variant_idx(node: _Node, depth_counts: dict[int, int]) -> None:
    """Assign the next per-depth variant index to a qualifying node.

    Args:
        node: The node to assign a variant index to.
        depth_counts: Mutable counter tracking next index per depth.
    """
    node.variant_idx = depth_counts.get(node.depth, 0)
    depth_counts[node.depth] = node.variant_idx + 1


def _save_variant(node: _Node, cache_dir: Path) -> tuple[str, str]:
    """Write variant source files to disk.

    Returns:
        Tuple of (nki_path, error).
    """
    gym_dir = cache_dir / "nkigym"
    gym_dir.mkdir(parents=True, exist_ok=True)
    source = program_to_source(node.program)
    (gym_dir / f"nkigym_d{node.depth}_v{node.variant_idx}.py").write_text(source)
    nki_dir = cache_dir / "nki"
    nki_dir.mkdir(parents=True, exist_ok=True)
    nki_path = str(nki_dir / f"nki_d{node.depth}_v{node.variant_idx}.py")
    error = ""
    try:
        nki_source = _use_affine_range(roll_loops(lower_to_nki(node.program)))
        Path(nki_path).write_text(nki_source)
    except Exception as e:
        error = _capture_error(e)
    return (nki_path, error)


@dataclass
class _SearchContext:
    """Bundled parameters for the search loop."""

    save_cache: Path
    kernel_kwargs: dict[str, Any]
    expected: np.ndarray
    pool: CompilationPool
    report: SearchReport
    tol: float


def _save_and_submit(node: _Node, ctx: _SearchContext, errors: list[VariantResult]) -> None:
    """Save variant to disk and submit to compilation pool if enabled."""
    nki_path, error = _save_variant(node, ctx.save_cache)
    ctx.report.add_variant(nki_path, node.depth, node.variant_idx)
    if error:
        errors.append(
            VariantResult(
                nki_path=nki_path,
                min_ms=0.0,
                mean_ms=0.0,
                p50_ms=0.0,
                p99_ms=0.0,
                mac_count=0,
                mfu=0.0,
                correct=False,
                error=error,
            )
        )
        ctx.report.update_variant(nki_path, status="error", error=error)
    if ctx.pool is not None and not error:
        ctx.report.update_variant(nki_path, status="compiled")
        ctx.pool.submit(nki_path)


def _emit_progress(
    report: SearchReport,
    root_node: _Node,
    graph: _TransformGraph,
    qualifying_programs: int,
    min_depth: int,
    depth_dist: dict[int, int],
) -> None:
    """Write current search progress to the JSON report."""
    report.update_search(
        root_stmts=len(root_node.program.stmts),
        root_opportunities=len(root_node.opportunities),
        unique_programs=len(graph.nodes),
        qualifying_programs=qualifying_programs,
        min_depth=min_depth,
        total_programs_visited=graph.total_programs_visited,
        depth_distribution=depth_dist,
    )


def _prepare_root(func: Callable, save_cache: Path, kernel_kwargs: dict[str, Any]) -> tuple[GymProgram, np.ndarray]:
    """Tile the user function and compute expected output.

    Writes ``nkigym_input.py`` and ``nkigym_root.py`` to the cache directory.
    """
    gym_dir = save_cache / "nkigym"
    gym_dir.mkdir(parents=True, exist_ok=True)
    input_path = gym_dir / "nkigym_input.py"
    input_path.write_text("import numpy as np\nimport nkigym\n" + callable_to_source(func))
    imported_func = import_func(input_path, func.__name__)
    output_dtype = next(iter(kernel_kwargs.values())).dtype
    expected = imported_func(**kernel_kwargs).astype(output_dtype)
    input_shapes = {k: v.shape for k, v in kernel_kwargs.items()}
    source = callable_to_source(func)
    program = tile_program(source_to_program(source, input_shapes, output_dtype))
    (gym_dir / "nkigym_root.py").write_text(program_to_source(program))
    return program, expected


def _run_search(
    graph: _TransformGraph, rng: random.Random, min_depth: int, num_targets: float, ctx: _SearchContext
) -> tuple[list[GymProgram], list[VariantResult]]:
    """Execute search loop: verify every node, save qualifying ones."""
    qualifying: list[GymProgram] = []
    lowering_errors: list[VariantResult] = []
    depth_counts: dict[int, int] = {}
    depth_dist: dict[int, int] = {0: 1}
    last_progress_count = 0

    root_node = next(iter(graph.nodes.values()))
    _verify_node(root_node, ctx.kernel_kwargs, ctx.expected, ctx.tol)
    if root_node.depth >= min_depth:
        _assign_variant_idx(root_node, depth_counts)
        _save_and_submit(root_node, ctx, lowering_errors)
        qualifying.append(root_node.program)

    while len(qualifying) < num_targets and graph.frontier:
        is_new, child_program = graph.expand_one(rng)
        if not is_new:
            continue
        child_node = graph.nodes[child_program]
        depth_dist[child_node.depth] = depth_dist.get(child_node.depth, 0) + 1
        _verify_node(child_node, ctx.kernel_kwargs, ctx.expected, ctx.tol)
        if child_node.depth >= min_depth:
            _assign_variant_idx(child_node, depth_counts)
            _save_and_submit(child_node, ctx, lowering_errors)
            qualifying.append(child_node.program)
        unique_count = len(graph.nodes)
        if unique_count - last_progress_count >= PROGRESS_INTERVAL:
            _emit_progress(ctx.report, root_node, graph, len(qualifying), min_depth, depth_dist)
            last_progress_count = unique_count

    _emit_progress(ctx.report, root_node, graph, len(qualifying), min_depth, depth_dist)
    return (qualifying, lowering_errors)


def _make_pool(
    program: GymProgram, kernel_kwargs: dict[str, Any], expected: np.ndarray, save_cache: Path
) -> CompilationPool:
    """Create a CompilationPool for parallel NKI compilation."""
    first_val = next(iter(kernel_kwargs.values()))
    return CompilationPool(
        func_name=program.name,
        input_shapes={k: v.shape for k, v in kernel_kwargs.items()},
        input_dtype_name=first_val.dtype.name,
        output_name="output",
        output_shape=expected.shape,
        output_dtype_name=expected.dtype.name,
        cache_dir=save_cache,
    )


def _finalize_benchmark(
    pool: CompilationPool,
    func_name: str,
    kernel_kwargs: dict[str, np.ndarray],
    expected: np.ndarray,
    report: SearchReport,
    mac_count: int,
) -> list[VariantResult]:
    """Wait for compilation and run hardware benchmarks."""
    compile_results = pool.wait_all()
    pool.shutdown()
    succeeded = sum(1 for cr in compile_results if not cr.error)
    report.set_compilation(succeeded=succeeded, failed=len(compile_results) - succeeded)
    variant_results = run_on_hardware(compile_results, func_name, kernel_kwargs, expected, _WARMUP, _ITERS, mac_count)
    _update_report_variants(report, variant_results)
    report.sort_variants()
    return variant_results


def _update_report_variants(report: SearchReport, results: list[VariantResult]) -> None:
    """Update variant entries in the report from benchmark results."""
    for r in results:
        status = "benchmarked" if not r.error else "error"
        report.update_variant(
            r.nki_path,
            status=status,
            min_ms=r.min_ms,
            mean_ms=r.mean_ms,
            p50_ms=r.p50_ms,
            p99_ms=r.p99_ms,
            mac_count=r.mac_count,
            mfu=r.mfu,
            correct=r.correct,
            error=r.error if r.error else None,
        )


def _log_search_summary(report: SearchReport, qualifying: int, results: list[VariantResult]) -> None:
    """Log final search summary and report path.

    Args:
        report: The search report that was written to disk.
        qualifying: Number of qualifying variants found.
        results: All variant results including errors.
    """
    succeeded = sum(1 for r in results if not r.error)
    failed = len(results) - succeeded
    logger.info("Search complete: %d qualifying, %d succeeded, %d failed", qualifying, succeeded, failed)
    logger.info("Results: %s", report.path)


def search(
    func: Callable,
    transforms: list[Transform],
    num_targets: float,
    seed: int,
    min_depth: int,
    save_cache: Path,
    kernel_kwargs: dict[str, Any],
) -> SearchResults:
    """Search the transform state graph for unique kernel variants.

    Args:
        func: Root tiled function.
        transforms: Transforms defining the search graph.
        num_targets: Qualifying variant target (``math.inf`` for exhaustive).
        seed: Random seed for reproducibility.
        min_depth: Minimum transform depth for qualifying variants.
        save_cache: Directory for variant source files.
        kernel_kwargs: Keyword arguments for the kernel call.

    Returns:
        SearchResults with qualifying variants and benchmark data.
    """
    save_cache.mkdir(parents=True, exist_ok=True)
    report = SearchReport(save_cache / "results.json")
    program, expected = _prepare_root(func, save_cache, kernel_kwargs)
    mac_count = compute_mac_count(program)

    graph = _TransformGraph(program, transforms)
    rng = random.Random() if seed < 0 else random.Random(seed)
    logger.info("Search root: %d stmts, %d opportunities", len(program.stmts), len(graph.nodes[program].opportunities))

    pool = _make_pool(program, kernel_kwargs, expected, save_cache)
    tol = 1e-4 + 1e-4 * float(np.max(np.abs(expected)))
    ctx = _SearchContext(
        save_cache=save_cache, kernel_kwargs=kernel_kwargs, expected=expected, pool=pool, report=report, tol=tol
    )
    qualifying, lowering_errors = _run_search(graph, rng, min_depth, num_targets, ctx)
    variant_results = _finalize_benchmark(pool, program.name, kernel_kwargs, expected, report, mac_count)

    all_results = lowering_errors + variant_results
    _log_search_summary(report, len(qualifying), all_results)
    return SearchResults(variants=qualifying, variant_results=all_results)
