"""Convenience API for remote NKI kernel profiling."""

from autotune.runner.output import ProfileOutput
from autotune.runner.remote import RemoteProfiler
from autotune.runner.types import KernelJob, ProfileConfig, ProfileResult


def _run_and_cache(profiler: RemoteProfiler, kernels: dict[str, KernelJob], cache_dir: str) -> list[ProfileResult]:
    """Execute profiling and optionally save cache."""
    results = profiler.profile(kernels)
    if cache_dir:
        profiler.save_cache(cache_dir, kernels, results)
    return results


def remote_profile(
    kernels: dict[str, KernelJob],
    hosts: list[str],
    cache_dir: str,
    warmup: int = 5,
    iters: int = 20,
    config: ProfileConfig = ProfileConfig(),
) -> ProfileOutput:
    """Profile NKI kernels across remote Trainium hosts.

    Single-call API that distributes kernel compilation and benchmarking
    to remote workers, optionally saving results to a cache directory.

    Args:
        kernels: Map of kernel filename to KernelJob.
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        cache_dir: Save results/sources/logs here (empty to skip).
        warmup: Number of warmup iterations before timing.
        iters: Number of benchmark iterations.
        config: Optional infra tuning (SSH timeout, platform target, etc.).

    Returns:
        ProfileOutput with results, compiler logs, and elapsed time.
    """
    profiler = RemoteProfiler(
        hosts=hosts,
        venv_python=config.venv_python,
        ssh_timeout_sec=config.ssh_timeout_sec,
        neuron_platform_target=config.neuron_platform_target,
        warmup=warmup,
        iters=iters,
        seed=config.seed,
        _collect_compiler_logs=bool(cache_dir),
    )
    results = _run_and_cache(profiler, kernels, cache_dir)

    return ProfileOutput(
        results=results,
        compiler_logs=profiler.compiler_logs,
        elapsed_s=profiler._last_elapsed,
        hosts=hosts,
        cache_dir=cache_dir,
    )
