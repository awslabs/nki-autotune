"""Unified nkigym pipeline: dispatch → canonical → sample → verify → profile.

``nkigym_compile`` is the single public entry point. It dispatches by
reading ``__nkigym_kernel__`` off the passed callable:

* ``@nkigym_kernel``-decorated → skip synthesis, use ``f`` directly.
* plain numpy callable         → run :func:`compile_numpy_to_nkigym`,
                                 write ``<cache>/f_nkigym.py``,
                                 load, :func:`_verify_fns` against
                                 the numpy reference, then continue.

Shared tail:

1. :func:`build_canonical_module` from ``f_nkigym``.
2. Render the canonical module into ``<cache>/kernel.py``; run
   :func:`_verify` — fp32 CPU sim vs ``f_nkigym``. Raise on mismatch.
3. ``enumerate_pool`` + ``sample_pool`` → ``num_kernels`` random
   variants.
4. For each variant ``i``: render → ``<cache>/kernel_tuned_{i:04d}.py``;
   run :func:`_verify`. Raise on mismatch.
5. If ``hosts == []``: write a minimal ``results.json`` index and
   return; no hardware profiling.
6. Else: build a ``KernelJob`` dict and call
   ``autotune.remote_profile``; ``results.json`` comes from there.
"""

import importlib.util
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render
from nkigym.synthesis import compile_numpy_to_nkigym
from nkigym.tune.batch import enumerate_pool, sample_pool
from nkigym.tune.verify import _draw_fp32_inputs, _verify, _verify_fns

_MAX_POOL_MULTIPLIER = 100


def nkigym_compile(
    f: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool = False,
    seed: int = 0,
) -> None:
    """Compile ``f`` to random NKI kernel variants, local-verify, profile.

    Dispatches on ``f``:
      * ``@nkigym_kernel``-decorated  → use directly.
      * plain numpy                   → synthesise f_nkigym first.

    Args:
        f: Either a ``@nkigym_kernel``-decorated callable (used
            directly) or a plain-numpy reference (triggers synthesis).
        input_specs: ``{param_name: (shape, dtype_str)}`` matching
            ``f``'s positional parameters.
        cache_dir: Directory for all artifacts — created if missing.
        num_kernels: Number of random variants to draw and (if
            ``hosts`` is non-empty) profile.
        hosts: SSH hostnames for ``remote_profile``. Empty list skips
            profiling entirely; verification still runs.
        venv_python: Python executable on remote hosts.
        neuron_platform_target: Neuron target, e.g. ``"trn2"``.
        collect_detailed_profile: Forwarded to ``remote_profile``.
        seed: Drives random sampling and (via ``remote_profile``)
            remote input-tensor generation.

    Raises:
        AssertionError: The canonical render, any sampled render, or
            (numpy-input branch only) the synthesised ``f_nkigym``
            vs ``f_numpy`` check fails fp32 CPU sim.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    f_nkigym = _resolve_f_nkigym(f, input_specs, cache_path)

    module = build_canonical_module(f_nkigym, _to_canonical_specs(input_specs))
    canonical_source = render(module)
    (cache_path / "kernel.py").write_text(canonical_source)
    _verify(canonical_source, f_nkigym, input_specs)

    rng = random.Random(seed)
    pool = enumerate_pool(module=module, max_pool_size=_MAX_POOL_MULTIPLIER * num_kernels, rng=rng)
    sampled = sample_pool(pool, num_kernels=num_kernels, rng=rng)

    kernels: dict[str, KernelJob] = {}
    output_shape = _trace_output_shape(f_nkigym, input_specs)
    for idx, sampled_module in enumerate(sampled):
        name = f"kernel_tuned_{idx:04d}.py"
        try:
            source = render(sampled_module)
        except Exception as e:
            (cache_path / f"{Path(name).stem}.render_error.txt").write_text(f"{type(e).__name__}: {e}\n")
            continue
        (cache_path / name).write_text(source)
        _verify(source, f_nkigym, input_specs)
        kernels[name] = KernelJob(
            source=source, func_name=f_nkigym.__name__, output_shape=output_shape, input_specs=input_specs
        )

    if not hosts:
        _write_stub_results_json(cache_path, kernels)
        return

    remote_profile(
        kernels=kernels,
        hosts=hosts,
        cache_dir=str(cache_path),
        seed=seed,
        neuron_platform_target=neuron_platform_target,
        venv_python=venv_python,
        collect_detailed_profile=collect_detailed_profile,
    )


def _resolve_f_nkigym(
    f: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], cache_path: Path
) -> Callable[..., np.ndarray]:
    """Dispatch: return ``f`` directly if tagged, else synthesise + verify."""
    if getattr(f, "__nkigym_kernel__", False):
        return f
    source = compile_numpy_to_nkigym(f, input_specs)
    (cache_path / "f_nkigym.py").write_text(source)
    f_nkigym = _load_f_nkigym(cache_path / "f_nkigym.py")
    _verify_fns(f_nkigym, f, input_specs)
    return f_nkigym


def _load_f_nkigym(path: Path) -> Callable[..., np.ndarray]:
    """Load ``path`` as a module and return its ``f_nkigym`` callable."""
    spec = importlib.util.spec_from_file_location("_nkigym_compile_f_nkigym", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not spec_from_file_location for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, "f_nkigym", None)
    if not callable(func):
        raise ValueError(f"{path!s} does not define a callable `f_nkigym`")
    return func


def _to_canonical_specs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, dict]:
    """Convert ``(shape, dtype)`` tuples to the dict form canonical expects."""
    return {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in input_specs.items()}


def _trace_output_shape(
    f_nkigym: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[int, ...]:
    """Call ``f_nkigym`` once on fp32 inputs to recover the output HBM shape."""
    inputs = _draw_fp32_inputs(input_specs)
    result = f_nkigym(**inputs)
    if isinstance(result, tuple):
        result = result[0]
    return tuple(np.asarray(result).shape)


def _write_stub_results_json(cache_path: Path, kernels: dict[str, KernelJob]) -> None:
    """Write an index-only ``results.json`` when ``hosts == []`` (no profiling)."""
    kernel_entries = [
        {"kernel_name": name, "kernel_path": f"{Path(name).stem}/{Path(name).stem}.py"} for name in sorted(kernels)
    ]
    data = {
        "metadata": {"num_kernels": len(kernels), "wallclock_s": 0.0, "hosts": []},
        "metrics": {},
        "kernels": kernel_entries,
    }
    (cache_path / "results.json").write_text(json.dumps(data, indent=2))
