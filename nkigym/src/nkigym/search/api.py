"""remote_search: sample unique kernel variants and profile them on remote Trainium hosts."""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileConfig
from nkigym.kernel_ir import build_ir
from nkigym.search.sampler import sample_variants

_WARMUP = 10
_ITERS = 100


def remote_search(
    func: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    golden_source: str,
    golden_func_name: str,
    hosts: list[str],
    cache_dir: str,
    num_variants: int,
    atol: float,
    rtol: float,
    seed: int | None = None,
    config: ProfileConfig = ProfileConfig(),
) -> ProfileOutput:
    """Sample ``num_variants`` unique variants from ``func`` and profile them.

    Builds a seed ``KernelIR`` for ``func`` internally, then
    rejection-samples ``(group_dim_orders, tensor_placements)`` via
    ``sample_variants``, dumping each sampled IR to
    ``<cache_dir>/kernels/<name>/`` as it is drawn, then wraps every
    variant in a ``KernelJob`` and delegates to ``remote_profile``,
    which writes the kernel source and compiler log to the same
    per-variant directory. Benchmark warmup and iters are fixed at
    ``10`` and ``100``.

    Args:
        func: Math function using NKIOp classes. Dimension analysis
            and the seed op graph are derived from this.
        input_specs: ``{param_name: (shape, dtype)}`` used by every
            variant.
        golden_source: Source code of the reference numpy function.
        golden_func_name: Reference function's name.
        hosts: SSH hostnames for remote profiling.
        cache_dir: Directory to cache kernel sources + results.
        num_variants: Number of unique variants to sample.
        atol: Absolute tolerance for CPU-sim correctness.
        rtol: Relative tolerance for CPU-sim correctness.
        seed: RNG seed for sampling. ``None`` = nondeterministic.
        config: Infra settings (SSH timeout, venv path, etc.).

    Returns:
        ProfileOutput with per-variant timing and correctness.
    """
    rng = random.Random(seed)
    initial_kernel = build_ir(func, input_specs, seed=seed)
    variants = sample_variants(initial_kernel, num_variants, rng, cache_dir=Path(cache_dir))
    kernels: dict[str, KernelJob] = {
        f"{name}.py": KernelJob(
            source=source,
            input_specs=input_specs,
            golden_source=golden_source,
            golden_func_name=golden_func_name,
            atol=atol,
            rtol=rtol,
        )
        for name, _ir, source in variants
    }
    return remote_profile(
        kernels=kernels, hosts=hosts, cache_dir=cache_dir, warmup=_WARMUP, iters=_ITERS, config=config
    )
