"""Driver for the ``"tune"`` stage of ``nkigym_compile``.

Loads the synthesised ``f_nkigym``, builds the canonical
:class:`KernelModule`, then dispatches on ``rewrites``:

* ``rewrites`` is a list → **explicit path** — apply each rewrite in
  order, render to ``kernel_tuned.py``, CPU-sim-check inline. No HW
  profile. Used by tests and backward-compatible callers.

* ``rewrites`` is ``None`` → **batch path** — enumerate the rewrite
  graph via :func:`nkigym.tune.batch.enumerate_pool`, uniformly
  sample ``num_kernels`` states, render each into
  ``kernel_tuned_{idx:04d}.py``, hand the dict to
  ``autotune.remote_profile`` for remote CPU-sim + HW profile,
  raise on any CPU-sim failure.
"""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import KernelModule
from nkigym.codegen.render import render
from nkigym.tune import KernelRewrite
from nkigym.tune.batch import enumerate_pool, sample_pool

_MAX_POOL_MULTIPLIER = 100


def _adapt_specs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, dict]:
    """Convert tuple-form input specs to dict-form expected by canonical builder.

    Args:
        input_specs: ``{name: (shape_tuple, dtype_str)}``.

    Returns:
        ``{name: {"shape": shape_tuple, "dtype": dtype_str}}``.
    """
    return {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in input_specs.items()}


def run_tune(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    rewrites: list[KernelRewrite] | None,
    seed: int,
    load_f_nkigym: Callable[[Path], Callable[..., np.ndarray]],
    cpu_sim_check: Callable[[str, str, Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], None],
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
    trace_output_shape: Callable[[Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], tuple[int, ...]],
    assert_no_cpu_sim_failures: Callable[[Path], None],
    atol: float,
    rtol: float,
) -> None:
    """Apply structural rewrites or batch-sample and profile.

    See module docstring for the two dispatch paths.

    Args:
        f_numpy: Plain-numpy reference — CPU-sim golden (explicit
            path) and output-shape trace source (batch path).
        input_specs: Per-param ``(shape, dtype_str)``.
        cache_path: Directory holding ``f_nkigym.py``; receives
            ``kernel_tuned.py`` (explicit) or
            ``kernel_tuned_{idx:04d}.py`` + ``results.json`` (batch).
        rewrites: List → explicit path; ``None`` → batch path.
        seed: Seeds the random draw and the ``remote_profile`` input
            tensor generation.
        load_f_nkigym: Loader from ``compile.py`` — injected to keep
            the stage driver independent.
        cpu_sim_check: fp32-simulation correctness helper (explicit
            path only).
        num_kernels: Batch-path kernel count. Ignored on the explicit
            path.
        hosts: ``remote_profile`` SSH hostnames. Ignored on the
            explicit path.
        venv_python: Python executable path on remote hosts.
        neuron_platform_target: Neuron target, e.g. ``"trn2"``.
        collect_detailed_profile: Forwarded to ``remote_profile``.
        trace_output_shape: Helper that runs ``f_numpy`` once to
            recover the output shape for ``KernelJob``.
        assert_no_cpu_sim_failures: Helper that reads ``results.json``
            and raises ``AssertionError`` on any ``cpu_sim_passed ==
            False`` entry.
        atol: CPU-sim abs tolerance forwarded to ``KernelJob``.
        rtol: CPU-sim rel tolerance forwarded to ``KernelJob``.
    """
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"tune requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = load_f_nkigym(f_nkigym_path)
    module = build_canonical_module(f_nkigym, _adapt_specs(input_specs))

    if rewrites is None:
        _run_batch(
            f_numpy=f_numpy,
            f_nkigym=f_nkigym,
            input_specs=input_specs,
            cache_path=cache_path,
            module=module,
            seed=seed,
            num_kernels=num_kernels,
            hosts=hosts,
            venv_python=venv_python,
            neuron_platform_target=neuron_platform_target,
            collect_detailed_profile=collect_detailed_profile,
            trace_output_shape=trace_output_shape,
            assert_no_cpu_sim_failures=assert_no_cpu_sim_failures,
            atol=atol,
            rtol=rtol,
        )
    else:
        _run_explicit(
            f_numpy=f_numpy,
            f_nkigym=f_nkigym,
            input_specs=input_specs,
            cache_path=cache_path,
            module=module,
            rewrites=rewrites,
            cpu_sim_check=cpu_sim_check,
        )


def _run_explicit(
    *,
    f_numpy: Callable[..., np.ndarray],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    module: KernelModule,
    rewrites: list[KernelRewrite],
    cpu_sim_check: Callable[[str, str, Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], None],
) -> None:
    """Apply ``rewrites`` deterministically and CPU-sim-check the result."""
    for r in rewrites:
        if not r.is_legal(module):
            raise ValueError(f"{r!r} illegal on current state")
        module = r.apply(module)
    kernel_source = render(module)
    (cache_path / "kernel_tuned.py").write_text(kernel_source)
    cpu_sim_check(kernel_source, f_nkigym.__name__, f_numpy, input_specs)


def _run_batch(
    *,
    f_numpy: Callable[..., np.ndarray],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    module: KernelModule,
    seed: int,
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
    trace_output_shape: Callable[[Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], tuple[int, ...]],
    assert_no_cpu_sim_failures: Callable[[Path], None],
    atol: float,
    rtol: float,
) -> None:
    """Enumerate the rewrite pool, render ``num_kernels`` samples, profile on HW."""
    rng = random.Random(seed)
    pool = enumerate_pool(module=module, max_pool_size=_MAX_POOL_MULTIPLIER * num_kernels, rng=rng)
    sampled = sample_pool(pool, num_kernels=num_kernels, rng=rng)

    output_shape = trace_output_shape(f_numpy, input_specs)
    nkigym_source = (cache_path / "f_nkigym.py").read_text()

    kernels: dict[str, KernelJob] = {}
    for idx, sampled_module in enumerate(sampled):
        source = render(sampled_module)
        name = f"kernel_tuned_{idx:04d}.py"
        (cache_path / name).write_text(source)
        kernels[name] = KernelJob(
            source=source,
            func_name=f_nkigym.__name__,
            output_shape=output_shape,
            input_specs=input_specs,
            nkigym_source=nkigym_source,
            nkigym_func_name=f_nkigym.__name__,
            atol=atol,
            rtol=rtol,
        )

    remote_profile(
        kernels=kernels,
        hosts=hosts,
        cache_dir=str(cache_path),
        seed=seed,
        neuron_platform_target=neuron_platform_target,
        venv_python=venv_python,
        collect_detailed_profile=collect_detailed_profile,
    )

    assert_no_cpu_sim_failures(cache_path / "results.json")
