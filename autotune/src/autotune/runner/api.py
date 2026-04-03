"""Convenience API for remote NKI kernel profiling."""

from autotune.runner.output import ProfileOutput
from autotune.runner.remote import RemoteProfiler
from autotune.runner.types import ProfileConfig, ProfileResult


def _run_and_cache(
    profiler: RemoteProfiler,
    kernels: dict[str, str],
    cache_dir: str,
    cfg: ProfileConfig,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> list[ProfileResult]:
    """Execute profiling and optionally save cache."""
    results = profiler.profile(
        kernels=kernels,
        input_specs=input_specs,
        scalar_params=cfg.scalar_params,
        mac_count=cfg.mac_count,
        seed=cfg.seed,
        golden_source=cfg.golden_source,
        golden_func_name=cfg.golden_func_name,
        atol=cfg.atol,
        rtol=cfg.rtol,
    )
    if cache_dir:
        profiler.save_cache(cache_dir, kernels, results)
    return results


def remote_profile(
    kernels: dict[str, str],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    hosts: list[str],
    cache_dir: str,
    config: ProfileConfig,
) -> ProfileOutput:
    """Profile NKI kernels across remote Trainium hosts.

    Single-call API that distributes kernel compilation and benchmarking
    to remote workers, optionally saving results to a cache directory.

    Args:
        kernels: Map of kernel filename to source code string.
        input_specs: Map of param name to (shape, dtype_str).
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        cache_dir: Save results/sources/logs here (empty to skip).
        config: Optional tuning for benchmarking, correctness, and infra.

    Returns:
        ProfileOutput with results, compiler logs, and elapsed time.
    """
    profiler = RemoteProfiler(
        hosts=hosts,
        venv_python=config.venv_python,
        ssh_timeout_sec=config.ssh_timeout_sec,
        neuron_platform_target=config.neuron_platform_target,
        warmup=config.warmup,
        iters=config.iters,
        _collect_compiler_logs=bool(cache_dir),
    )
    results = _run_and_cache(profiler, kernels, cache_dir, config, input_specs)

    return ProfileOutput(
        results=results,
        compiler_logs=profiler.compiler_logs,
        elapsed_s=profiler._last_elapsed,
        hosts=hosts,
        cache_dir=cache_dir,
    )
