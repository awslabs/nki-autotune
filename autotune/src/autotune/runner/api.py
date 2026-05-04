"""Convenience API for remote NKI kernel profiling."""

from autotune.runner.output import ProfileOutput
from autotune.runner.remote import RemoteProfiler, write_kernel_sources
from autotune.runner.types import KernelJob, ProfileResult


def _run_and_cache(profiler: RemoteProfiler, kernels: dict[str, KernelJob], cache_dir: str) -> list[ProfileResult]:
    """Execute profiling and optionally save cache."""
    if cache_dir:
        write_kernel_sources(cache_dir, kernels)
    results = profiler.profile(kernels)
    if cache_dir:
        profiler.save_cache(cache_dir, kernels, results)
    return results


def remote_profile(
    kernels: dict[str, KernelJob],
    hosts: list[str],
    cache_dir: str,
    seed: int,
    neuron_platform_target: str,
    venv_python: str,
    collect_detailed_profile: bool,
) -> ProfileOutput:
    """Profile NKI kernels across remote Trainium hosts.

    Single-call API that distributes kernel compilation and benchmarking
    to remote workers, optionally saving results to a cache directory.

    Args:
        kernels: Map of kernel filename to KernelJob.
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        cache_dir: Save results/sources/logs here (empty to skip).
        seed: RNG seed for deterministic input tensor generation.
        neuron_platform_target: Neuron platform target (e.g. "trn2").
        venv_python: Path to the Python executable on remote hosts.
        collect_detailed_profile: Capture the full per-instruction
            ``neuron-profile`` JSON into each kernel's cache subfolder.
            Tens of MB per kernel.

    Returns:
        ProfileOutput with results, compiler logs, and elapsed time.
    """
    profiler = RemoteProfiler(
        hosts=hosts,
        venv_python=venv_python,
        neuron_platform_target=neuron_platform_target,
        seed=seed,
        _collect_compiler_logs=bool(cache_dir),
        collect_detailed_profile=collect_detailed_profile,
    )
    results = _run_and_cache(profiler, kernels, cache_dir)

    return ProfileOutput(
        results=results,
        compiler_logs=profiler.compiler_logs,
        elapsed_s=profiler._last_elapsed,
        hosts=hosts,
        cache_dir=cache_dir,
    )
