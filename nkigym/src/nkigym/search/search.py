"""Transform graph search.

Provides graph-based search over the space of transform application
sequences.  A ``_TransformGraph`` lazily expands the state graph using
kernel-tuple deduplication (no source rendering in the hot path).

Only qualifying variants (depth >= min_depth) are saved to disk, verified
for numerical correctness, and submitted for hardware benchmarking.
Shallow nodes are explored but not persisted or tested.
"""

import logging
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nkigym.codegen import codegen, dce, normalize
from nkigym.codegen.types import NKIKernel
from nkigym.search.compile import (  # noqa: F401
    CompilationPool,
    SearchResults,
    VariantResult,
    _capture_error,
    run_on_hardware,
)
from nkigym.search.report import SearchReport
from nkigym.transforms.base import NKITransform

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 50
_WARMUP = 10
_ITERS = 100


def _collect_opportunities(kernel: NKIKernel, transforms: list[NKITransform]) -> list[tuple[NKITransform, Any]]:
    """Collect all ``(transform, option)`` pairs available for a kernel."""
    opportunities: list[tuple[NKITransform, Any]] = []
    for transform in transforms:
        for option in transform.analyze(kernel):
            opportunities.append((transform, option))
    return opportunities


@dataclass
class _Node:
    """A node in the transform state graph.

    Attributes:
        kernel: NKIKernel for this state (used as dedup key).
        opportunities: All ``(transform, option)`` pairs available here.
        unexplored: Indices into ``opportunities`` not yet expanded.
        depth: Depth at which this node was first discovered.
        variant_idx: Per-depth variant counter, assigned when qualifying.
    """

    kernel: NKIKernel
    opportunities: list[tuple[NKITransform, Any]]
    unexplored: list[int]
    depth: int
    variant_idx: int


@dataclass
class _TransformGraph:
    """Lazily-expanded transform state graph with depth-uniform sampling.

    Attributes:
        nodes: Map from kernel tuple to ``_Node``.
        edges: Map from ``(parent_kernel, opp_idx)`` to child kernel.
        _depth_buckets: Frontier kernels grouped by depth.
        _bucket_pos: Map from frontier kernel to position in its bucket.
        transforms: Transforms defining the search space.
        total_kernels_visited: Total kernels visited (root + expansions).
    """

    nodes: dict[NKIKernel, _Node] = field(default_factory=dict)
    edges: dict[tuple[NKIKernel, int], NKIKernel] = field(default_factory=dict)
    _depth_buckets: dict[int, list[NKIKernel]] = field(default_factory=dict)
    _bucket_pos: dict[NKIKernel, int] = field(default_factory=dict)
    transforms: list[NKITransform] = field(default_factory=list)
    total_kernels_visited: int = 0

    def __init__(self, kernel: NKIKernel, transforms: list[NKITransform]) -> None:
        """Initialize the graph with a root node."""
        self.nodes = {}
        self.edges = {}
        self._depth_buckets = {}
        self._bucket_pos = {}
        self.transforms = transforms
        self.total_kernels_visited = 1
        self._add_node(kernel, depth=0)

    def _frontier_add(self, kernel: NKIKernel) -> None:
        """Add a kernel to its depth bucket in O(1)."""
        depth = self.nodes[kernel].depth
        bucket = self._depth_buckets.setdefault(depth, [])
        self._bucket_pos[kernel] = len(bucket)
        bucket.append(kernel)

    def _frontier_remove(self, kernel: NKIKernel) -> None:
        """Remove a kernel from its depth bucket in O(1) via swap-and-pop."""
        depth = self.nodes[kernel].depth
        bucket = self._depth_buckets[depth]
        idx = self._bucket_pos.pop(kernel)
        last = bucket[-1]
        if idx < len(bucket) - 1:
            bucket[idx] = last
            self._bucket_pos[last] = idx
        bucket.pop()
        if not bucket:
            del self._depth_buckets[depth]

    def _add_node(self, kernel: NKIKernel, depth: int) -> bool:
        """Register a new node. Returns True if new."""
        is_new = kernel not in self.nodes
        if is_new:
            opportunities = _collect_opportunities(kernel, self.transforms)
            unexplored = list(range(len(opportunities)))
            self.nodes[kernel] = _Node(
                kernel=kernel, opportunities=opportunities, unexplored=unexplored, depth=depth, variant_idx=-1
            )
            if opportunities:
                self._frontier_add(kernel)
        return is_new

    def expand_one(self, rng: random.Random) -> tuple[bool, NKIKernel]:
        """Expand one opportunity using depth-uniform sampling.

        Uniformly samples a depth, then randomly picks a frontier node
        from that depth group. Ensures equal expansion rate across all
        depths regardless of per-depth population size.

        Returns:
            Tuple of (is_new, child_kernel).
        """
        self.total_kernels_visited += 1
        d = rng.choice(list(self._depth_buckets.keys()))
        parent_kernel = rng.choice(self._depth_buckets[d])
        node = self.nodes[parent_kernel]
        idx_pos = rng.randrange(len(node.unexplored))
        opp_idx = node.unexplored.pop(idx_pos)
        transform, option = node.opportunities[opp_idx]
        child_kernel = normalize(dce(transform.apply(node.kernel, option)))
        self.edges[(parent_kernel, opp_idx)] = child_kernel
        if not node.unexplored:
            self._frontier_remove(parent_kernel)
        is_new = self._add_node(child_kernel, depth=node.depth + 1)
        return is_new, child_kernel


def _verify_node(node: _Node, sim_kwargs: dict[str, Any], expected: np.ndarray) -> None:
    """Simulate a node's kernel and check numerical correctness."""
    actual = node.kernel.simulate(sim_kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _assign_variant_idx(node: _Node, depth_counts: dict[int, int]) -> None:
    """Assign the next per-depth variant index to a qualifying node.

    Args:
        node: The node to assign a variant index to.
        depth_counts: Mutable counter tracking next index per depth.
    """
    node.variant_idx = depth_counts.get(node.depth, 0)
    depth_counts[node.depth] = node.variant_idx + 1


def _save_variant(node: _Node, cache_dir: Path) -> tuple[str, str]:
    """Write variant NKI source file to disk.

    Returns:
        Tuple of (nki_path, error).
    """
    nki_dir = cache_dir / "nki"
    nki_dir.mkdir(parents=True, exist_ok=True)
    nki_path = str(nki_dir / f"nki_d{node.depth}_v{node.variant_idx}.py")
    error = ""
    try:
        nki_source = node.kernel.render()
        Path(nki_path).write_text(nki_source)
    except Exception as e:
        error = _capture_error(e)
    return (nki_path, error)


@dataclass
class _SearchContext:
    """Bundled parameters for the search loop."""

    save_cache: Path
    kernel_kwargs: dict[str, Any]
    sim_kwargs: dict[str, Any]
    expected: np.ndarray
    pool: CompilationPool
    report: SearchReport


def _save_node(node: _Node, ctx: _SearchContext, errors: list[VariantResult]) -> tuple[str, str]:
    """Save variant to disk and register in report. Returns (nki_path, error)."""
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
    return nki_path, error


def _submit_to_pool(nki_path: str, error: str, ctx: _SearchContext) -> None:
    """Submit a saved variant to the compilation pool."""
    if ctx.pool is not None and not error:
        ctx.report.update_variant(nki_path, status="compiled")
        ctx.pool.submit(nki_path)


def _emit_progress(
    report: SearchReport,
    root_node: _Node,
    graph: _TransformGraph,
    qualifying_kernels: int,
    min_depth: int,
    depth_dist: dict[int, int],
) -> None:
    """Write current search progress to the JSON report."""
    root_stmts = sum(len(b.body) for b in root_node.kernel.blocks)
    report.update_search(
        root_stmts=root_stmts,
        root_opportunities=len(root_node.opportunities),
        unique_programs=len(graph.nodes),
        qualifying_programs=qualifying_kernels,
        min_depth=min_depth,
        total_programs_visited=graph.total_kernels_visited,
        depth_distribution=depth_dist,
    )


def _prepare_root(
    func: Callable, save_cache: Path, kernel_kwargs: dict[str, Any]
) -> tuple[NKIKernel, np.ndarray, dict[str, Any]]:
    """Codegen the user function and compute expected output.

    Writes ``nki_root.py`` to the cache directory.
    Returns kernel, expected output, and float64 sim kwargs.
    """
    nki_dir = save_cache / "nki"
    nki_dir.mkdir(parents=True, exist_ok=True)
    sim_kwargs = {k: v.astype(np.float64) for k, v in kernel_kwargs.items()}
    expected = func(**sim_kwargs)
    kernel = codegen(func, kernel_kwargs)
    (nki_dir / "nki_root.py").write_text(kernel.render())
    return kernel, expected, sim_kwargs


def _run_search(
    graph: _TransformGraph, rng: random.Random, min_depth: int, num_targets: float, ctx: _SearchContext
) -> tuple[list[NKIKernel], list[VariantResult]]:
    """Execute search loop: verify every node, save qualifying ones."""
    qualifying: list[NKIKernel] = []
    lowering_errors: list[VariantResult] = []
    depth_counts: dict[int, int] = {}
    depth_dist: dict[int, int] = {0: 1}
    last_progress_count = 0

    root_node = next(iter(graph.nodes.values()))
    if root_node.depth >= min_depth:
        _assign_variant_idx(root_node, depth_counts)
        nki_path, error = _save_node(root_node, ctx, lowering_errors)
        _verify_node(root_node, ctx.sim_kwargs, ctx.expected)
        _submit_to_pool(nki_path, error, ctx)
        qualifying.append(root_node.kernel)

    _emit_progress(ctx.report, root_node, graph, len(qualifying), min_depth, depth_dist)

    while len(qualifying) < num_targets and graph._depth_buckets:
        is_new, child_kernel = graph.expand_one(rng)
        if not is_new:
            continue
        child_node = graph.nodes[child_kernel]
        depth_dist[child_node.depth] = depth_dist.get(child_node.depth, 0) + 1
        if child_node.depth >= min_depth:
            _assign_variant_idx(child_node, depth_counts)
            nki_path, error = _save_node(child_node, ctx, lowering_errors)
            _verify_node(child_node, ctx.sim_kwargs, ctx.expected)
            _submit_to_pool(nki_path, error, ctx)
            qualifying.append(child_node.kernel)
        unique_count = len(graph.nodes)
        if unique_count - last_progress_count >= PROGRESS_INTERVAL:
            _emit_progress(ctx.report, root_node, graph, len(qualifying), min_depth, depth_dist)
            last_progress_count = unique_count

    _emit_progress(ctx.report, root_node, graph, len(qualifying), min_depth, depth_dist)
    return (qualifying, lowering_errors)


def _make_pool(
    kernel: NKIKernel, kernel_kwargs: dict[str, Any], expected: np.ndarray, save_cache: Path
) -> CompilationPool:
    """Create a CompilationPool for parallel NKI compilation."""
    first_val = next(iter(kernel_kwargs.values()))
    return CompilationPool(
        func_name=kernel.name,
        input_shapes={k: v.shape for k, v in kernel_kwargs.items()},
        input_dtype_name=first_val.dtype.name,
        output_name="output",
        output_shape=expected.shape,
        output_dtype_name=first_val.dtype.name,
        cache_dir=save_cache,
    )


def _finalize_benchmark(
    pool: CompilationPool,
    func_name: str,
    kernel_kwargs: dict[str, np.ndarray],
    expected: np.ndarray,
    report: SearchReport,
    mac_count: int,
    input_dtype_name: str,
) -> list[VariantResult]:
    """Wait for compilation and run hardware benchmarks."""
    compile_results = pool.wait_all()
    pool.shutdown()
    succeeded = sum(1 for cr in compile_results if not cr.error)
    report.set_compilation(succeeded=succeeded, failed=len(compile_results) - succeeded)
    variant_results = run_on_hardware(
        compile_results, func_name, kernel_kwargs, expected, _WARMUP, _ITERS, mac_count, input_dtype_name
    )
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
    transforms: list[NKITransform],
    num_targets: float,
    seed: int,
    min_depth: int,
    save_cache: Path,
    kernel_kwargs: dict[str, Any],
) -> SearchResults:
    """Search the transform state graph for unique kernel variants.

    Args:
        func: Root user function.
        transforms: Transforms defining the search graph.
        num_targets: Qualifying variant target (``math.inf`` for exhaustive).
        seed: Random seed for reproducibility.
        min_depth: Minimum transform depth for qualifying variants.
        save_cache: Directory for variant source files.
        kernel_kwargs: Keyword arguments for the kernel call.

    Returns:
        SearchResults with qualifying variants and benchmark data.
    """
    shutil.rmtree(save_cache, ignore_errors=True)
    save_cache.mkdir(parents=True)
    report = SearchReport(save_cache / "results.json")
    kernel, expected, sim_kwargs = _prepare_root(func, save_cache, kernel_kwargs)
    mac_count = kernel.mac_count
    input_dtype_name = next(iter(kernel_kwargs.values())).dtype.name

    graph = _TransformGraph(kernel, transforms)
    root_stmts = sum(len(b.body) for b in kernel.blocks)
    rng = random.Random() if seed == 0 else random.Random(seed)
    logger.info("Search root: %d stmts, %d opportunities", root_stmts, len(graph.nodes[kernel].opportunities))
    pool = _make_pool(kernel, kernel_kwargs, expected, save_cache)
    ctx = _SearchContext(
        save_cache=save_cache,
        kernel_kwargs=kernel_kwargs,
        sim_kwargs=sim_kwargs,
        expected=expected,
        pool=pool,
        report=report,
    )
    qualifying, lowering_errors = _run_search(graph, rng, min_depth, num_targets, ctx)
    variant_results = _finalize_benchmark(
        pool, kernel.name, kernel_kwargs, expected, report, mac_count, input_dtype_name
    )
    all_results = lowering_errors + variant_results
    _log_search_summary(report, len(qualifying), all_results)
    return SearchResults(variants=qualifying, variant_results=all_results)
