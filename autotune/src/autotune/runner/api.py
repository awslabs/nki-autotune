"""Local API for NKI kernel profiling.

Runs ON a Trn2 box, in-process. Compiles + benchmarks a set of NKI
kernels and writes the standard cache layout (sources, results.json,
per-kernel profiler JSON) under ``cache_dir``.
"""

import logging
import os
import time

from autotune.runner.driver import run_pipeline
from autotune.runner.output import (
    ProfileOutput,
    write_compiler_logs,
    write_kernel_sources,
    write_results_json,
)
from autotune.runner.types import KernelJob

logger = logging.getLogger(__name__)


def profile(
    kernels: dict[str, KernelJob],
    cache_dir: str,
    seed: int,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
) -> ProfileOutput:
    """Compile + benchmark NKI kernels in-process on this Trn2 box.

    Args:
        kernels: Map of kernel filename to KernelJob.
        cache_dir: Directory to write sources / results.json / per-kernel
            profiler JSON. Empty string skips all disk output.
        seed: RNG seed for deterministic input tensor generation.
        neuron_platform_target: Neuron platform target (e.g. "trn2"). Set
            into NEURON_PLATFORM_TARGET_OVERRIDE for the run.
        collect_detailed_profile: Capture the full per-instruction
            neuron-profile JSON + NEFF/NTFF into each kernel's cache
            subfolder (tens of MB per kernel).

    Returns:
        ProfileOutput with results, compiler logs, and elapsed time.
    """
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = neuron_platform_target
    collect_logs = bool(cache_dir)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        write_kernel_sources(cache_dir, kernels)

    t0 = time.monotonic()
    results, compiler_logs = run_pipeline(
        kernels,
        seed=seed,
        collect_compiler_logs=collect_logs,
        collect_detailed_profile=collect_detailed_profile,
    )
    elapsed_s = time.monotonic() - t0
    logger.info("Profile complete: %d results in %.1fs", len(results), elapsed_s)

    if cache_dir:
        write_kernel_sources(cache_dir, kernels)
        write_compiler_logs(cache_dir, compiler_logs)
        write_results_json(cache_dir, len(kernels), results, elapsed_s)
        logger.info("Cache saved to %s", cache_dir)

    return ProfileOutput(
        results=results,
        compiler_logs=compiler_logs,
        elapsed_s=elapsed_s,
        cache_dir=cache_dir,
    )
