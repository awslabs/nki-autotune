"""``nkigym_compile``: synthesis → eager codegen → tune driver.

Single public entry point for the nkigym compilation pipeline. Runs
all three stages in order on every call and caches their artifacts
under ``cache_dir``:

``<cache_dir>/``

* ``f_nkigym.py``              — synthesised ``f_nkigym`` math function.
* ``kernel.py``                — rendered canonical eager NKI kernel.
* ``kernel_tuned_{idx:04d}.py``  — batch-path rendered samples
  (``rewrites=None``; default).
* ``kernel_tuned.py``          — explicit-path single rendered kernel
  (``rewrites=[...]``).
* ``results.json``             — per-kernel + aggregate profile data,
  written by ``autotune.remote_profile`` on the batch path.

``initial_codegen`` runs an automatic fp32 CPU-sim accuracy check.
The batch tune path runs the same check per kernel on remote workers
and raises if any kernel diverges.
"""

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.mermaid import dump_forest_mermaid
from nkigym.codegen.render import render
from nkigym.synthesis import compile_numpy_to_nkigym
from nkigym.tune import KernelRewrite
from nkigym.tune.stage import run_tune

_ATOL = 5e-3
_RTOL = 5e-3
_SIM_SEED = 0


def nkigym_compile(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool = False,
    rewrites: list[KernelRewrite] | None = None,
    seed: int = 0,
) -> None:
    """Run the full nkigym compile pipeline: synthesis → initial_codegen → tune.

    Args:
        f_numpy: Plain-numpy reference. Drives synthesis (input),
            ``initial_codegen``'s inline CPU-sim check (golden), and
            the batch path's output-shape trace + kernel-level
            correctness golden.
        input_specs: ``{param_name: (shape, dtype_str)}``; order
            matches ``f_numpy``'s positional parameters by name.
        cache_dir: Artifact directory. Created if missing;
            pre-existing files for each stage are overwritten.
        num_kernels: Batch path — number of kernels to sample + profile.
            Ignored when ``rewrites`` is not ``None``.
        hosts: SSH hosts for ``remote_profile``. Ignored when
            ``rewrites`` is not ``None``.
        venv_python: Python executable on remote hosts.
        neuron_platform_target: Neuron target (e.g. ``"trn2"``).
        collect_detailed_profile: Collect the full per-instruction
            profile + NEFF/NTFF per kernel. Off by default.
        rewrites: ``None`` → batch path (default). List →
            deterministic explicit path, writes
            ``kernel_tuned.py`` only, no HW profile.
        seed: Drives sampling (batch path) and ``remote_profile``
            input-tensor generation.

    Raises:
        AssertionError: Any sampled kernel fails CPU sim (batch) or
            the canonical ``initial_codegen`` CPU sim fails.
        ValueError: ``rewrites`` list has an illegal atom on the
            current state.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    _run_synthesis(f_numpy, input_specs, cache_path)
    _run_initial_codegen(f_numpy, input_specs, cache_path)
    run_tune(
        f_numpy=f_numpy,
        input_specs=input_specs,
        cache_path=cache_path,
        rewrites=rewrites,
        seed=seed,
        load_f_nkigym=_load_f_nkigym,
        cpu_sim_check=_cpu_sim_check,
        num_kernels=num_kernels,
        hosts=hosts,
        venv_python=venv_python,
        neuron_platform_target=neuron_platform_target,
        collect_detailed_profile=collect_detailed_profile,
        trace_output_shape=_trace_output_shape,
        assert_no_cpu_sim_failures=_assert_no_cpu_sim_failures,
        atol=_ATOL,
        rtol=_RTOL,
    )


def _run_synthesis(
    f_numpy: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], cache_path: Path
) -> None:
    """Synthesise ``f_nkigym`` via the Claude Agent SDK and write ``f_nkigym.py``."""
    source = compile_numpy_to_nkigym(f_numpy, input_specs)
    (cache_path / "f_nkigym.py").write_text(source)


def _run_initial_codegen(
    f_numpy: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], cache_path: Path
) -> None:
    """Render the eager NKI kernel, write ``kernel.py`` and ``forest_initial.mmd``, and CPU-sim-check."""
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"initial_codegen requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = _load_f_nkigym(f_nkigym_path)
    op_graph = parse_and_resolve(f_nkigym, input_specs)
    forest = build_canonical_forest(op_graph)
    kernel_source = render(op_graph, forest=forest)
    (cache_path / "kernel.py").write_text(kernel_source)
    (cache_path / "forest_initial.mmd").write_text(dump_forest_mermaid(forest=forest, op_graph=op_graph))
    _cpu_sim_check(kernel_source, f_nkigym.__name__, f_numpy, input_specs)


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


def _cpu_sim_check(
    kernel_source: str,
    func_name: str,
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Execute the rendered kernel through ``nki.simulate`` at fp32 and compare to numpy.

    Drops the user-declared bf16/fp16 dtypes in favour of fp32 so the
    accuracy check isn't dominated by low-precision rounding — matches
    the ``simulate_one`` convention in ``autotune.runner.benchmark``.
    """
    sim_source = kernel_source.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)  # noqa: S102
    kernel_fn = ns[func_name]
    inputs = _draw_fp32_inputs(input_specs)
    actual = nki.simulate(kernel_fn)(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = f_numpy(**inputs)
    max_abs = float(np.abs(actual - expected).max())
    max_rel = float((np.abs(actual - expected) / (np.abs(expected) + _ATOL)).max())
    tolerance_ok = np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL)
    print(f"[nkigym_compile] cpu_sim max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")
    if not tolerance_ok:
        raise AssertionError(
            f"initial_codegen CPU-sim mismatch vs numpy reference: "
            f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} "
            f"(atol={_ATOL}, rtol={_RTOL})"
        )


def _draw_fp32_inputs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, np.ndarray]:
    """Draw reproducible fp32 inputs for the CPU-sim accuracy check."""
    rng = np.random.default_rng(_SIM_SEED)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in input_specs.items()}


def _trace_output_shape(
    f_numpy: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[int, ...]:
    """Trace the numpy reference once to recover the output shape.

    ``KernelJob`` requires the output HBM shape on the coordinator so
    workers skip unreliable AST parsing. The same random fp32 inputs
    used by :func:`_cpu_sim_check` drive the trace — if the numpy
    reference is deterministic on shape this is stable regardless of
    seed.
    """
    inputs = _draw_fp32_inputs(input_specs)
    result = f_numpy(**inputs)
    if isinstance(result, tuple):
        result = result[0]
    return tuple(result.shape)


def _assert_no_cpu_sim_failures(results_json_path: Path) -> None:
    """Raise ``AssertionError`` when any kernel in ``results.json`` failed CPU sim.

    ``remote_profile`` writes a ``results.json`` whose ``"kernels"``
    array has one entry per kernel with a boolean ``"cpu_sim_passed"``.
    Per the design spec, CPU-sim divergence on any batched kernel is
    a bug; HW compile/runtime failures are tolerated (emitted into
    ``results.json`` but not raised here).
    """
    data = json.loads(results_json_path.read_text())
    failed = [k["kernel_name"] for k in data["kernels"] if not k["cpu_sim_passed"]]
    if failed:
        raise AssertionError(
            f"CPU-sim failures in batch tune stage: {failed}. " f"See {results_json_path} for details."
        )
