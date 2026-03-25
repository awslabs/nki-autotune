"""Combinatorial schedule search over kernel generation strategies.

Enumerates all valid schedules as the cross-product of loop orders,
op placements, and blocking factors.  Each unique schedule is rendered
to NKI source, compiled, and benchmarked on hardware.
"""

import logging
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nkigym.codegen.analysis import _Analysis, _OpCall, analyze_dims
from nkigym.codegen.parse import find_func_def, parse_body
from nkigym.codegen.passes import _PassAssignment, assign_passes
from nkigym.schedule.enumerate import enumerate_all
from nkigym.schedule.render import render_schedule
from nkigym.schedule.types import Schedule
from nkigym.search.compile import (
    CompilationPool,
    SearchResults,
    VariantResult,
    _capture_error,
    _make_benchmark_cfg,
    stream_compile_and_run,
)
from nkigym.search.report import SearchReport
from nkigym.simulate import simulate_kernel
from nkigym.utils.source import callable_to_source

logger = logging.getLogger(__name__)

_WARMUP = 10
_ITERS = 100


@dataclass
class _Workload:
    """Parsed user workload: op calls, params, analysis, function name."""

    analysis: _Analysis
    op_calls: list[_OpCall]
    params: tuple[str, ...]
    func_name: str
    pa: _PassAssignment


def _parse_workload(func: Callable, kernel_kwargs: dict[str, Any]) -> _Workload:
    """Parse a user function into workload components.

    Args:
        func: User function using ``nkigym.<op>(...)`` calls.
        kernel_kwargs: Input arrays keyed by parameter name.

    Returns:
        Parsed workload with analysis and op calls.
    """
    source = callable_to_source(func)
    func_def = find_func_def(source)
    params = tuple(arg.arg for arg in func_def.args.args)
    op_calls = parse_body(func_def)
    input_shapes = tuple(kernel_kwargs[p].shape for p in params)
    analysis = analyze_dims(op_calls, params, input_shapes)
    pa = assign_passes(op_calls, analysis)
    return _Workload(analysis=analysis, op_calls=op_calls, params=params, func_name=func_def.name, pa=pa)


def _compute_mac_count(analysis: _Analysis) -> int:
    """Compute total multiply-accumulate count from dimension analysis.

    MACs only exist for workloads with reduction dimensions (e.g. matmul).
    Returns 0 for pure element-wise workloads (no reduction dims).

    Args:
        analysis: Dimension analysis result.

    Returns:
        Total MAC count, or 0 if no reduction dims.
    """
    all_counts = {**analysis.tile_counts, **analysis.reduction_tile_counts}
    total = 1 if analysis.reduction_dims else 0
    for dim_id, count in all_counts.items():
        total *= count * analysis.dim_tile_sizes[dim_id]
    return total


@dataclass
class _SearchContext:
    """Bundled parameters for the search pipeline."""

    save_cache: Path
    kernel_kwargs: dict[str, Any]
    expected: np.ndarray
    pool: CompilationPool
    report: SearchReport
    workload: _Workload


def _save_variant(idx: int, schedule: Schedule, ctx: _SearchContext) -> tuple[str, str, str]:
    """Write variant NKI source file to disk. Returns (nki_path, nki_source, error)."""
    nki_dir = ctx.save_cache / "nki"
    nki_dir.mkdir(parents=True, exist_ok=True)
    nki_path = str(nki_dir / f"nki_v{idx}.py")
    nki_source = ""
    error = ""
    try:
        w = ctx.workload
        nki_source = render_schedule(w.analysis, schedule, w.op_calls, w.params, w.func_name, w.pa)
        Path(nki_path).write_text(nki_source)
    except Exception as e:
        error = _capture_error(e)
    return (nki_path, nki_source, error)


def _failure_result(nki_path: str, error: str) -> VariantResult:
    """Create a VariantResult for a variant that failed verification."""
    return VariantResult(
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


def _cpu_verify(nki_source: str, func_name: str, kernel_kwargs: dict[str, Any], expected: np.ndarray) -> str:
    """Verify kernel correctness via CPU simulation.

    Simulates the rendered NKI kernel using numpy at float64 and
    compares against the expected reference output.

    Returns:
        Empty string on success, error traceback on failure.
    """
    error = ""
    try:
        actual = simulate_kernel(nki_source, func_name, kernel_kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    except Exception as e:
        error = _capture_error(e)
    return error


def _save_and_submit(idx: int, schedule: Schedule, ctx: _SearchContext, errors: list[VariantResult]) -> None:
    """Save variant, verify correctness via CPU sim, and submit for compilation."""
    nki_path, nki_source, error = _save_variant(idx, schedule, ctx)
    ctx.report.add_variant(nki_path, depth=0, variant_idx=idx)
    if not error:
        error = _cpu_verify(nki_source, ctx.workload.func_name, ctx.kernel_kwargs, ctx.expected)
    if error:
        errors.append(_failure_result(nki_path, error))
        ctx.report.update_variant(nki_path, status="error", error=error)
    elif ctx.pool is not None:
        ctx.report.update_variant(nki_path, status="compiled")
        ctx.pool.submit(nki_path)


def _select_schedules(all_schedules: list[Schedule], num_targets: int, seed: int) -> list[Schedule]:
    """Select up to num_targets schedules, sampling randomly if space is larger.

    Args:
        all_schedules: All valid enumerated schedules.
        num_targets: Maximum number of schedules to benchmark.
        seed: Random seed (0 for system entropy).

    Returns:
        Selected schedules.
    """
    rng = random.Random() if seed == 0 else random.Random(seed)
    selected = rng.sample(all_schedules, min(num_targets, len(all_schedules)))
    return selected


def _run_search(schedules: list[Schedule], ctx: _SearchContext) -> list[VariantResult]:
    """Render, save, and submit all selected schedules.

    Returns:
        List of lowering errors (schedules that failed to render).
    """
    lowering_errors: list[VariantResult] = []
    for idx, schedule in enumerate(schedules):
        _save_and_submit(idx, schedule, ctx, lowering_errors)
    ctx.report.update_search(
        unique_schedules=len(schedules),
        qualifying_schedules=len(schedules) - len(lowering_errors),
        total_visited=len(schedules),
    )
    return lowering_errors


def _make_pool(
    func_name: str, kernel_kwargs: dict[str, Any], expected: np.ndarray, save_cache: Path
) -> CompilationPool:
    """Create a CompilationPool for parallel NKI compilation."""
    first_val = next(iter(kernel_kwargs.values()))
    return CompilationPool(
        func_name=func_name,
        input_shapes={k: v.shape for k, v in kernel_kwargs.items()},
        input_dtype_name=first_val.dtype.name,
        output_name="hbm_tensor_0",
        output_shape=expected.shape,
        output_dtype_name=first_val.dtype.name,
        cache_dir=save_cache,
    )


def _finalize_benchmark(
    ctx: _SearchContext, func_name: str, mac_count: int, input_dtype_name: str
) -> list[VariantResult]:
    """Stream compilations into hardware benchmarks."""
    cfg = _make_benchmark_cfg(
        func_name, ctx.kernel_kwargs, ctx.expected.shape, _WARMUP, _ITERS, mac_count, input_dtype_name
    )
    succeeded, failed, variant_results = stream_compile_and_run(ctx.pool, cfg)
    ctx.report.set_compilation(succeeded=succeeded, failed=failed)
    _update_report_variants(ctx.report, variant_results)
    ctx.report.sort_variants()
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


def _enumerate_and_select(
    workload: _Workload, kernel_kwargs: dict[str, Any], num_targets: int, seed: int
) -> list[Schedule]:
    """Enumerate valid schedules and select a subset for benchmarking."""
    all_schedules = enumerate_all(workload.analysis, workload.op_calls, workload.params, workload.pa.passes_per_dim)
    selected = _select_schedules(all_schedules, num_targets, seed)
    logger.info("Enumerated %d valid schedules, selected %d", len(all_schedules), len(selected))
    return selected


def search(
    func: Callable, num_targets: int, seed: int, save_cache: Path, kernel_kwargs: dict[str, Any]
) -> SearchResults:
    """Search the combinatorial schedule space for kernel variants.

    Enumerates all valid schedules from loop orders x op placements x blocking,
    selects up to ``num_targets`` for benchmarking, renders each to NKI source,
    compiles, and benchmarks on hardware.

    Args:
        func: User function using ``nkigym.<op>(...)`` calls.
        num_targets: Maximum number of schedules to benchmark.
        seed: Random seed for selection (0 for system entropy).
        save_cache: Directory for output artifacts.
        kernel_kwargs: Input arrays keyed by parameter name.

    Returns:
        Combined search and benchmark results.
    """
    shutil.rmtree(save_cache, ignore_errors=True)
    save_cache.mkdir(parents=True)
    report = SearchReport(save_cache / "results.json")
    workload = _parse_workload(func, kernel_kwargs)
    sim_kwargs = {k: v.astype(np.float64) for k, v in kernel_kwargs.items()}
    expected = func(**sim_kwargs)
    mac_count = _compute_mac_count(workload.analysis)
    input_dtype_name = next(iter(kernel_kwargs.values())).dtype.name
    selected = _enumerate_and_select(workload, kernel_kwargs, num_targets, seed)

    pool = _make_pool(workload.func_name, kernel_kwargs, expected, save_cache)
    ctx = _SearchContext(
        save_cache=save_cache,
        kernel_kwargs=kernel_kwargs,
        expected=expected,
        pool=pool,
        report=report,
        workload=workload,
    )
    lowering_errors = _run_search(selected, ctx)
    variant_results = _finalize_benchmark(ctx, workload.func_name, mac_count, input_dtype_name)
    all_results = lowering_errors + variant_results
    succeeded = sum(1 for r in all_results if not r.error)
    failed = len(all_results) - succeeded
    logger.info("Search complete: %d variants, %d succeeded, %d failed", len(selected), succeeded, failed)
    logger.info("Results: %s", report.path)
    return SearchResults(variants=selected, variant_results=all_results)
