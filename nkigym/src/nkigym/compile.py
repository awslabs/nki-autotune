"""``nkigym_compile``: synthesis → eager codegen driver.

Single public entry point for the nkigym compilation pipeline. Runs the
selected ``stages`` in order, caching each stage's artifact under
``cache_dir``:

``<cache_dir>/``

* ``f_nkigym.py``  — synthesised ``f_nkigym`` math function
  (produced by ``"synthesis"``).
* ``kernel.py``    — rendered eager NKI kernel (produced by
  ``"initial_codegen"``).

``"initial_codegen"`` also runs an automatic CPU-simulation accuracy
test: it executes the rendered kernel through ``nki.simulate`` at fp32
against random inputs drawn from ``input_specs`` and raises
``AssertionError`` if the result diverges from ``f_numpy`` beyond the
default tolerance.
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen import render_eager
from nkigym.synthesis import compile_numpy_to_nkigym
from nkigym.tune import KernelRewrite
from nkigym.tune.stage import run_tune

_ATOL = 5e-3
_RTOL = 5e-3
_SIM_SEED = 0

_KNOWN_STAGES = ("synthesis", "initial_codegen", "tune")


def nkigym_compile(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    stages: list[str],
    rewrites: list[KernelRewrite] | None = None,
    seed: int = 0,
) -> None:
    """Run the nkigym compilation pipeline through the given ``stages``.

    Args:
        f_numpy: Plain-numpy reference — drives both synthesis (input)
            and the CPU-sim accuracy check (golden).
        input_specs: ``{param_name: (shape, dtype)}`` for every parameter.
            Must match ``f_numpy``'s positional parameters by name + order.
        cache_dir: Directory to write stage artifacts. Created if missing;
            pre-existing files from prior runs are preserved so stages
            can be replayed independently.
        stages: Ordered list of stage names to run. Supported:
            ``"synthesis"``, ``"initial_codegen"``, ``"tune"``. Later
            stages consume prior-stage artifacts from ``cache_dir``.
        rewrites: Consumed only by the ``"tune"`` stage. When ``None``,
            the stage randomises from :func:`enumerate_fusion_atoms`
            using ``seed``. When a list is given, rewrites are applied
            in order; each rewrite's legality is checked against the
            current ``(op_graph, forest)`` state.
        seed: Seeds the random rewrite draw when ``rewrites`` is ``None``.

    Raises:
        ValueError: An unknown stage name is in ``stages``, a stage's
            required input artifact is missing from ``cache_dir``, or an
            explicit rewrite is illegal on the current state.
        AssertionError: The CPU-sim result of the rendered kernel
            diverges from ``f_numpy`` beyond the fp32 tolerance.
    """
    for stage in stages:
        if stage not in _KNOWN_STAGES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {_KNOWN_STAGES}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        if stage == "synthesis":
            _run_synthesis(f_numpy, input_specs, cache_path)
        elif stage == "initial_codegen":
            _run_initial_codegen(f_numpy, input_specs, cache_path)
        elif stage == "tune":
            run_tune(
                f_numpy,
                input_specs,
                cache_path,
                rewrites=rewrites,
                seed=seed,
                load_f_nkigym=_load_f_nkigym,
                cpu_sim_check=_cpu_sim_check,
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
    """Render the eager NKI kernel, write ``kernel.py``, and CPU-sim-check it."""
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"initial_codegen requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = _load_f_nkigym(f_nkigym_path)
    kernel_source = render_eager(f_nkigym, input_specs)
    (cache_path / "kernel.py").write_text(kernel_source)
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
