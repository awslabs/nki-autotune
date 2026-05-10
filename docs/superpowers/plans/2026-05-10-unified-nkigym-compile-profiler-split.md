# Unified `nkigym_compile` + autotune-as-profiler-only — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `run_tune` + `nkigym_compile` into a single `nkigym_compile` entry point that dispatches on an `@nkigym_kernel` wrapper tag; remove all CPU-sim plumbing from the autotune runner (profiler-only); rewrite example drivers; delete `scripts/` and `debug_kernel_0000_chain.py`.

**Architecture:** Tag-dispatch at `nkigym.compile.nkigym_compile` — numpy-input branch runs synthesis first, then the shared pipeline: build canonical IR → render `kernel.py` → local fp32 CPU-verify → enumerate+sample `num_kernels` random variants → render each into `kernel_tuned_{idx:04d}.py` + local fp32 CPU-verify → hand the set to `autotune.remote_profile`. Any local verify failure aborts the run. `autotune.remote_profile` now only compiles + benchmarks; `KernelJob` and `ProfileResult` lose all CPU-sim fields. When `hosts == []`, skip profiling and write a minimal `results.json` index.

**Tech Stack:** Python 3.12, nkigym IR (in-repo), NKI + nkipy (AWS Trainium runtime, `~/venvs/kernel-env`), pytest.

**Related spec:** `docs/superpowers/specs/2026-05-09-nkigym-tune-and-profiler-split-design.md`.

**Convention for every commit in this plan:**
```
<type>: <subject>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 1: Tag the `@nkigym_kernel` decorator wrapper

The unified entry point dispatches by reading `__nkigym_kernel__` off the passed function. Set the attribute on the wrapper so decorated callables can be distinguished from plain numpy callables.

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py:192-217`
- Test: `test/ops/test_nkigym_kernel_tag.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/ops/__init__.py` if missing (empty file), then create `test/ops/test_nkigym_kernel_tag.py`:

```python
"""Tag assertion for the nkigym_kernel decorator wrapper."""

import numpy as np

from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


def test_nkigym_kernel_sets_tag_on_wrapper():
    """Decorated wrappers expose __nkigym_kernel__ = True."""

    @nkigym_kernel
    def identity(x):
        sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        hbm = NKIAlloc(location="hbm", shape=(128, 128), dtype="bfloat16")()
        NKILoad()(src=x, dst=sbuf)
        NKIStore()(src=sbuf, dst=hbm)
        return hbm

    assert getattr(identity, "__nkigym_kernel__", False) is True


def test_plain_function_has_no_tag():
    """Undecorated callables lack the tag."""

    def plain(x: np.ndarray) -> np.ndarray:
        return x

    assert getattr(plain, "__nkigym_kernel__", False) is False
```

Check the exact path — if `test/ops/` does not exist, create the directory plus `__init__.py`. Other test subdirs follow the `test/<subpkg>/__init__.py + test_*.py` convention already.

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/ops/test_nkigym_kernel_tag.py -v
```

Expected: `test_nkigym_kernel_sets_tag_on_wrapper` FAILS (`False is True`). The plain test passes.

- [ ] **Step 3: Set the tag on the wrapper**

Edit `nkigym/src/nkigym/ops/base.py`:

```python
def nkigym_kernel(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``func`` as an nkigym kernel and enforce load / store discipline.

    Tags every ``np.ndarray`` argument with ``role="param"`` on entry,
    so any non-``NKILoad`` op that touches it raises ``TypeError`` from
    the offending op's call site. After ``func`` returns, asserts the
    return value is a ``_RoleArray`` with ``role="stored"`` — i.e. the
    last op was ``NKIStore`` — otherwise raises ``TypeError`` at the
    return site.

    The returned wrapper carries ``__nkigym_kernel__ = True`` so public
    dispatchers (``nkigym_compile``) can distinguish it from plain numpy
    callables.

    Preserves the wrapped function's signature and source; downstream
    consumers that rely on ``inspect.signature`` / ``inspect.getsource``
    (``build_ir``, the synthesis prompt builder, etc.) keep working.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tagged_args = tuple(_tag_as_param(a) for a in args)
        tagged_kwargs = {k: _tag_as_param(v) for k, v in kwargs.items()}
        result = func(*tagged_args, **tagged_kwargs)
        role = _operand_role(result)
        if role != "stored":
            raise TypeError(f"{func.__name__} returned role={role!r}; kernel must end with NKIStore before `return`")
        return result

    wrapper.__nkigym_kernel__ = True
    return wrapper
```

Only two changes: the docstring paragraph about the tag, and the `wrapper.__nkigym_kernel__ = True` line before `return wrapper`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/ops/test_nkigym_kernel_tag.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/base.py test/ops/__init__.py test/ops/test_nkigym_kernel_tag.py
git commit -m "$(cat <<'EOF'
feat(nkigym): tag @nkigym_kernel wrappers with __nkigym_kernel__ = True

Enables nkigym_compile to dispatch between numpy-input (needs synthesis)
and live @nkigym_kernel input (use directly) by a single attribute check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Introduce `nkigym/tune/verify.py`

Local fp32 CPU-sim helper. Takes the rendered kernel source and a live `f_nkigym` callable; runs `nki.simulate` at fp32 and compares against calling `f_nkigym` with the same fp32 inputs. This replaces the removed worker-side `compute_golden` + `simulate_one` pair and the old `compile._cpu_sim_check`.

**Files:**
- Create: `nkigym/src/nkigym/tune/verify.py`
- Test: `test/tune/test_verify.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/tune/test_verify.py`:

```python
"""Local fp32 CPU-sim verify helper."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.verify import _verify, _verify_fns


K, M, N = 128, 128, 128


@nkigym_kernel
def _lhsT_matmul(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


def test_verify_passes_for_canonical_render():
    module = build_canonical_module(
        _lhsT_matmul,
        {name: {"shape": shape, "dtype": dt} for name, (shape, dt) in _INPUT_SPECS.items()},
    )
    source = render(module)
    _verify(source, _lhsT_matmul, _INPUT_SPECS)


def test_verify_raises_on_mismatch():
    def wrong_kernel(lhs_T, rhs):
        return np.zeros((M, N), dtype=np.float32)

    module = build_canonical_module(
        _lhsT_matmul,
        {name: {"shape": shape, "dtype": dt} for name, (shape, dt) in _INPUT_SPECS.items()},
    )
    source = render(module)
    with pytest.raises(AssertionError, match="max_abs"):
        _verify(source, wrong_kernel, _INPUT_SPECS)


def test_verify_fns_passes_when_fns_agree():
    def f_numpy(lhs_T, rhs):
        return lhs_T.T @ rhs

    _verify_fns(_lhsT_matmul, f_numpy, _INPUT_SPECS)


def test_verify_fns_raises_when_fns_disagree():
    def f_numpy(lhs_T, rhs):
        return lhs_T @ rhs

    with pytest.raises(AssertionError, match="f_nkigym vs f_numpy"):
        _verify_fns(_lhsT_matmul, f_numpy, _INPUT_SPECS)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/tune/test_verify.py -v
```

Expected: all four FAIL with `ImportError` — `nkigym.tune.verify` doesn't exist yet.

- [ ] **Step 3: Implement the helper**

Create `nkigym/src/nkigym/tune/verify.py`:

```python
"""Local fp32 CPU-sim verification for nkigym-rendered kernels.

Two helpers, both local (no remote round-trip, no autotune dependency):

* :func:`_verify` runs ``nki.simulate`` on the rendered kernel source
  at fp32 and compares element-wise against ``f_nkigym(**inputs)``.
  ``f_nkigym`` is executed directly — its ``NKIOp`` operations have
  pure-numpy ``__call__`` implementations, so the math-function itself
  serves as the golden reference.

* :func:`_verify_fns` compares two live callables at fp32. Used on the
  numpy-input branch of ``nkigym_compile`` to confirm synthesis
  produced a math-correct ``f_nkigym`` before spending cycles tuning
  it.

Both raise ``AssertionError`` on divergence outside ``atol=rtol=5e-3``.
"""

from collections.abc import Callable

import nki
import numpy as np

_ATOL = 5e-3
_RTOL = 5e-3
_SIM_SEED = 0


def _draw_fp32_inputs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, np.ndarray]:
    """Draw reproducible fp32 inputs shaped by ``input_specs``.

    Ignores declared dtypes — the CPU-sim contract is fp32 everywhere,
    matching the renderer's ``nl.bfloat16 -> nl.float32`` rewrite below.
    """
    rng = np.random.default_rng(_SIM_SEED)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in input_specs.items()}


def _rewrite_to_fp32(kernel_source: str) -> str:
    """Return ``kernel_source`` with bf16 / fp16 dtypes forced to fp32.

    Matches the convention used throughout the project for CPU-sim runs:
    the hardware path keeps the user's declared dtypes; the simulator
    runs fp32 end-to-end so accuracy is not dominated by rounding.
    """
    return kernel_source.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")


def _verify(
    kernel_source: str,
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Run ``kernel_source`` through ``nki.simulate`` and compare to ``f_nkigym``.

    Raises ``AssertionError`` if the element-wise diff exceeds
    ``atol=rtol=5e-3``.
    """
    sim_source = _rewrite_to_fp32(kernel_source)
    ns: dict = {}
    exec(sim_source, ns)  # noqa: S102
    kernel_fn = ns[f_nkigym.__name__]
    inputs = _draw_fp32_inputs(input_specs)
    actual = nki.simulate(kernel_fn)(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = f_nkigym(**inputs)
    max_abs = float(np.abs(actual - expected).max())
    max_rel = float((np.abs(actual - expected) / (np.abs(expected) + _ATOL)).max())
    if not np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL):
        raise AssertionError(
            f"kernel vs f_nkigym: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(atol={_ATOL}, rtol={_RTOL})"
        )


def _verify_fns(
    f_nkigym: Callable[..., np.ndarray],
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Check that synthesised ``f_nkigym`` agrees with ``f_numpy`` at fp32.

    Raises ``AssertionError`` on divergence.
    """
    inputs = _draw_fp32_inputs(input_specs)
    nk = f_nkigym(**inputs)
    np_ = f_numpy(**inputs)
    if isinstance(nk, tuple):
        nk = nk[0]
    if isinstance(np_, tuple):
        np_ = np_[0]
    max_abs = float(np.abs(nk - np_).max())
    max_rel = float((np.abs(nk - np_) / (np.abs(np_) + _ATOL)).max())
    if not np.allclose(nk, np_, atol=_ATOL, rtol=_RTOL):
        raise AssertionError(
            f"f_nkigym vs f_numpy: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(atol={_ATOL}, rtol={_RTOL})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/tune/test_verify.py -v
```

Expected: all four PASS. If the rendered kernel source contains `nl.bfloat16` tokens and `nki.simulate` evaluates correctly, the canonical render will match. The mismatch tests will raise with `max_abs=...`.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/verify.py test/tune/test_verify.py
git commit -m "$(cat <<'EOF'
feat(nkigym): add local fp32 CPU-sim verify helpers

_verify compares a rendered kernel source against a live @nkigym_kernel
callable via nki.simulate at fp32; _verify_fns compares two callables
directly (synthesised f_nkigym vs reference f_numpy). Both raise
AssertionError on divergence outside atol=rtol=5e-3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `compile.py` as the unified tag-dispatch driver

Merge the old `compile.py` + `tune/stage.py` + the batch-path pipeline into a single file. The new `nkigym_compile` does: dispatch → canonical build → render `kernel.py` → `_verify` → sample `num_kernels` → render + `_verify` each → either skip profiling (`hosts==[]`) with a stub `results.json`, or call `remote_profile`.

**Files:**
- Rewrite: `nkigym/src/nkigym/compile.py`

- [ ] **Step 1: Rewrite `compile.py` end-to-end**

Replace the whole file with:

```python
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
from nkigym.tune.verify import _verify, _verify_fns

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
    for idx, sampled_module in enumerate(sampled):
        source = render(sampled_module)
        name = f"kernel_tuned_{idx:04d}.py"
        (cache_path / name).write_text(source)
        _verify(source, f_nkigym, input_specs)
        kernels[name] = KernelJob(
            source=source,
            func_name=f_nkigym.__name__,
            output_shape=_trace_output_shape(f_nkigym, input_specs),
            input_specs=input_specs,
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
    f: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
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
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> tuple[int, ...]:
    """Call ``f_nkigym`` once on fp32 inputs to recover the output HBM shape."""
    from nkigym.tune.verify import _draw_fp32_inputs

    inputs = _draw_fp32_inputs(input_specs)
    result = f_nkigym(**inputs)
    if isinstance(result, tuple):
        result = result[0]
    return tuple(result.shape)


def _write_stub_results_json(cache_path: Path, kernels: dict[str, KernelJob]) -> None:
    """Write an index-only ``results.json`` when ``hosts == []`` (no profiling)."""
    kernel_entries = [
        {
            "kernel_name": name,
            "kernel_path": f"{Path(name).stem}/{Path(name).stem}.py",
        }
        for name in sorted(kernels)
    ]
    data = {
        "metadata": {"num_kernels": len(kernels), "wallclock_s": 0.0, "hosts": []},
        "metrics": {},
        "kernels": kernel_entries,
    }
    (cache_path / "results.json").write_text(json.dumps(data, indent=2))
```

Key points for the engineer reading this:

- `_trace_output_shape` lives here, not in `verify.py`, because it's part of the compile driver; the import of `_draw_fp32_inputs` is intentionally local (avoids re-exporting a private name).
- The old `compile.py` had injectable callables (`load_f_nkigym`, `cpu_sim_check`, `trace_output_shape`, `assert_no_cpu_sim_failures`) — all deleted. There's no caller that needs to inject different implementations.
- The stub `results.json` schema mirrors `remote.py:_write_results_json` structure (metadata + metrics + kernels) but with empty metrics. See Task 8 for the matching changes in `remote.py`.

- [ ] **Step 2: Delete `tune/stage.py`**

```bash
git rm nkigym/src/nkigym/tune/stage.py
```

- [ ] **Step 3: Run existing tests to catch broken imports**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen test/tune -x -q 2>&1 | tail -40
```

Expected: any test that still imports `run_tune` or `nkigym.tune.stage` fails. Grep confirms no such tests exist today (`test/` never imports `run_tune`). If one surfaces, patch it.

Also smoke-test the imports:

```bash
python -c "from nkigym.compile import nkigym_compile; print(nkigym_compile)"
python -c "import nkigym; print(nkigym.nkigym_compile)"
```

Expected: both print a function reference — the lazy `__getattr__` on `nkigym/__init__.py` keeps working unchanged.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/compile.py nkigym/src/nkigym/tune/stage.py
git commit -m "$(cat <<'EOF'
refactor(nkigym): unify compile/tune into tag-dispatched nkigym_compile

Fold tune/stage.py into compile.py. Dispatches on f.__nkigym_kernel__:
decorated callables are used directly; plain numpy functions trigger
synthesis + _verify_fns check. Removes the injected-callable scaffolding
(load_f_nkigym/cpu_sim_check/trace_output_shape/assert_no_cpu_sim_failures
parameters) and the dual explicit/batch dispatch paths. When hosts==[],
write a minimal results.json and skip profiling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Slim `KernelJob` and `ProfileResult` in `autotune.runner.types`

Remove every CPU-sim-related field from the public autotune types. After this task, `autotune` only models "compile + benchmark on hardware."

**Files:**
- Modify: `autotune/src/autotune/runner/types.py`
- Test: repo-wide smoke — `pytest test/ -x -q`

- [ ] **Step 1: Edit `types.py`**

Make these edits:

Drop `nkigym_source`, `nkigym_func_name`, `atol`, `rtol`, `skip_cpu_sim` from `KernelJob`. Also remove them from the docstring.

```python
class KernelJob(NamedTuple):
    """Per-kernel configuration for remote profiling.

    Attributes:
        source: NKI kernel source code string.
        func_name: Name of the ``@nki.jit`` function inside ``source``.
        output_shape: Shape of the kernel's HBM output tensor. Supplied
            by the caller (traced once on the coordinator for
            nkigym-generated kernels, hardcoded for reference kernels)
            to avoid unreliable AST parsing on the worker.
        input_specs: Map of param name to (shape, dtype_str).
        neuronx_cc_args: Extra flags forwarded to neuronx-cc via
            ``CompileOptions.set_pipeline_options(*args)``. Empty for
            nkigym-generated kernels; hand-allocated reference kernels
            (e.g. nkilib's ``attention_cte``) typically need
            ``("enable-linear-scan-allocation=false",
            "enable-instruction-scheduling=false")`` — equivalent to
            nkilib's ``disable_backend_optimizations()``.
        lnc: Logical NeuronCore count (1 or 2).
    """

    source: str
    func_name: str
    output_shape: tuple[int, ...]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    neuronx_cc_args: tuple[str, ...] = ()
    lnc: int = 1
```

Drop the `cpu_sim: dict` field from `ProfileResult`:

```python
class ProfileResult(NamedTuple):
    """Benchmark result for a single kernel.

    ``profiler_summary`` is the raw ``neuron-profile view
    --output-format summary-json`` dict for the post-compiler NTFF
    trace. Every number we report — wall-clock time, MFU/MBU/ceiling,
    engine-active times, cycle counts, DMA bytes — lives there.

    ``profile_detailed``, ``neff_b64``, and ``ntff_b64`` are collected
    only when the caller passes ``collect_detailed_profile=True`` to
    :func:`remote_profile`.

    All optional fields are ``None`` when compile or hardware
    execution failed, or when detailed collection is off.
    """

    kernel_name: str
    hardware_output: str
    profiler_summary: dict | None = None
    profile_detailed: dict | None = None
    neff_b64: str | None = None
    ntff_b64: str | None = None
```

Delete `_sim_not_run`. Simplify the failure constructors:

```python
def make_failure(kernel_name: str, hardware_output: str) -> ProfileResult:
    """Create a failed ProfileResult with a null ``profiler_summary``."""
    return ProfileResult(kernel_name=kernel_name, hardware_output=hardware_output)


def compile_failure_result(cr: CompileResult) -> ProfileResult:
    """Convert a failed CompileResult into a ProfileResult."""
    return make_failure(cr.kernel_name, cr.error)
```

- [ ] **Step 2: Run pytest on the autotune package**

```bash
pytest test/ -x -q 2>&1 | tail -20
```

Expected: imports break in `worker.py`, `benchmark.py`, `baseline.py`, `remote.py` — those touch `_sim_not_run` or `cpu_sim=`. That's the point: the next tasks will fix each file. Leave the repo in a failing state for now (checkpoint commit below).

- [ ] **Step 3: Commit (interim)**

```bash
git add autotune/src/autotune/runner/types.py
git commit -m "$(cat <<'EOF'
refactor(autotune): slim KernelJob and ProfileResult (cpu_sim fields dropped)

Drop nkigym_source/nkigym_func_name/atol/rtol/skip_cpu_sim from KernelJob
and cpu_sim from ProfileResult. _sim_not_run, and the cpu_sim kwarg on
make_failure/compile_failure_result, are removed.

This commit intentionally leaves the repo in a temporarily broken state
— worker.py/benchmark.py/baseline.py/remote.py still reference the
removed fields; they are cleaned up in the following commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Strip CPU-sim from `autotune.runner.benchmark`

Delete `compute_golden`, `simulate_one`, `_rewrite_to_fp32`, and the `cpu_sim` parameter / `_sim_not_run` usage on `benchmark_one`.

**Files:**
- Modify: `autotune/src/autotune/runner/benchmark.py`

- [ ] **Step 1: Edit `benchmark.py`**

1. In the imports near the top, drop `re`, `tempfile`'s unused halves are fine, and reshape the `autotune.runner.types` import to remove `_sim_not_run`:

```python
from autotune.runner.types import (
    CompileResult,
    OutputSpec,
    ProfileResult,
    capture_error,
    resolve_dtype,
    tensor_inputs,
)
```

2. Delete the `simulate_one`, `_rewrite_to_fp32`, and `compute_golden` functions in their entirety (currently lines 170–255 — verify by reading the file). The test imports from earlier tasks never touched these; no replacements needed.

3. Edit `benchmark_one` — drop the `cpu_sim` parameter and stop passing it into `ProfileResult`:

```python
def benchmark_one(
    spike: BaremetalExecutor,
    cr: CompileResult,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    out: OutputSpec,
    collect_detailed_profile: bool = False,
) -> ProfileResult:
    """Run the compiled kernel once with ``save_trace=True`` and attach the
    profiler's JSON output(s) to a :class:`ProfileResult`.

    Wall-clock timing and utilization come entirely from the profiler
    summary. When ``collect_detailed_profile`` is ``True`` the full
    per-instruction trace is also attached as ``profile_detailed``.
    """
    profiler_summary: dict[str, Any] | None = None
    profile_detailed: dict[str, Any] | None = None
    neff_b64: str | None = None
    ntff_b64: str | None = None
    hardware_output = f"{list(out.shape)} {out.dtype}"
    try:
        compiled = create_compiled_kernel(cr.neff_path, cr.nki_path, func_name, kernel_kwargs, out)
        profiler_summary, profile_detailed, ntff_b64 = _collect_profiler_outputs(
            spike, compiled, kernel_kwargs, collect_detailed_profile
        )
        if collect_detailed_profile:
            with open(cr.neff_path, "rb") as f:
                neff_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        hardware_output = capture_error(e)
    return ProfileResult(
        kernel_name=cr.kernel_name,
        hardware_output=hardware_output,
        profiler_summary=profiler_summary,
        profile_detailed=profile_detailed,
        neff_b64=neff_b64,
        ntff_b64=ntff_b64,
    )
```

4. Clean up module imports that are now unused: `import re` and `from pathlib import Path` were only used by the deleted helpers. Remove them. Keep `numpy as np` only if still referenced (it's not once `simulate_one`/`compute_golden` are gone — double check, and remove if unused).

- [ ] **Step 2: Run a targeted check**

```bash
python -c "from autotune.runner.benchmark import benchmark_one; print(benchmark_one)"
```

Expected: prints a function object (imports succeed).

- [ ] **Step 3: Commit**

```bash
git add autotune/src/autotune/runner/benchmark.py
git commit -m "$(cat <<'EOF'
refactor(autotune): drop CPU-sim path from benchmark.py

Remove simulate_one, compute_golden, _rewrite_to_fp32, and the cpu_sim
parameter on benchmark_one. autotune's only job on the worker is
compile + hardware benchmark; CPU sim now lives in nkigym.tune.verify.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Delete `autotune.runner.compare`

The only function in this file (`assert_close`) is only used by the deleted CPU-sim path.

**Files:**
- Delete: `autotune/src/autotune/runner/compare.py`

- [ ] **Step 1: Confirm no remaining callers**

```bash
grep -rn "from autotune.runner.compare\|runner.compare\|assert_close" /home/ubuntu/nki-autotune --include="*.py" | grep -v __pycache__
```

Expected: zero hits (worker.py drops its import in Task 7).

If any hits appear: stop and report — something is wrong with the plan order.

- [ ] **Step 2: Delete the file**

```bash
git rm autotune/src/autotune/runner/compare.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(autotune): delete runner/compare.py

Only assert_close lived here, only used by the removed CPU-sim path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Strip CPU-sim from `autotune.runner.worker`

Remove the entire CPU-sim branch in `_process_kernel_job`, the sim-failure short-circuit in `_run_pipeline`, and the `_cpu_sim_status` helper.

**Files:**
- Modify: `autotune/src/autotune/runner/worker.py`

- [ ] **Step 1: Edit imports**

Replace the `from autotune.runner.benchmark ...` and `from autotune.runner.compare ...` imports:

```python
from autotune.runner.benchmark import benchmark_one, generate_tensors
```

Remove `from autotune.runner.compare import assert_close`. Remove `compute_golden` and `simulate_one` from the benchmark import.

Also remove these from the `autotune.runner.types` import list: `compile_failure_result` (still used below — keep), `make_failure` (still used — keep), but nothing else.

- [ ] **Step 2: Delete `_cpu_sim_status`**

It occupies lines 55–73 (verify via Read). Remove the whole function.

- [ ] **Step 3: Rewrite `_process_kernel_job`**

Replace with:

```python
def _process_kernel_job(kname: str, job: dict[str, Any], seed: int, nki_dir: Path) -> dict[str, Any]:
    """Process a single kernel job: write source, generate tensors.

    Returns a dict with all per-kernel data needed for compile and benchmark.
    """
    source = job["source"]
    filename = kname if kname.endswith(".py") else f"{kname}.py"
    nki_path = nki_dir / filename
    nki_path.write_text(source)

    func_name = job["func_name"]
    output_shape = tuple(job["output_shape"])
    neuronx_cc_args = tuple(job.get("neuronx_cc_args", ()))

    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in job["tensor_specs"].items()}
    kwargs = generate_tensors(tensor_specs, seed)

    input_dtype_name = next(iter(job["tensor_specs"].values()))[1]
    return {
        "nki_path": str(nki_path),
        "func_name": func_name,
        "output_shape": output_shape,
        "kwargs": kwargs,
        "input_dtype_name": input_dtype_name,
        "neuronx_cc_args": neuronx_cc_args,
        "lnc": int(job.get("lnc", 1)),
    }
```

- [ ] **Step 4: Rewrite `_benchmark_compiled` — drop `cpu_sim` arg**

```python
def _benchmark_compiled(
    compile_futures: list[Future],
    spike: BaremetalExecutor,
    kernel_data: dict[str, dict[str, Any]],
    collect_detailed_profile: bool,
) -> tuple[list[CompileResult], list[ProfileResult], list[ProfileResult]]:
    """Benchmark each kernel as it finishes compiling.

    Returns:
        Tuple of (compile_results, compile_errors, hw_results).
    """
    compile_results: list[CompileResult] = []
    compile_errors: list[ProfileResult] = []
    hw_results: list[ProfileResult] = []
    for f in as_completed(compile_futures):
        cr = f.result()
        compile_results.append(cr)
        kd = kernel_data[cr.kernel_name]
        if cr.error:
            compile_errors.append(compile_failure_result(cr))
            continue
        out = OutputSpec(
            name=_OUTPUT_TENSOR_NAME, shape=kd["output_shape"], dtype=resolve_dtype(kd["input_dtype_name"])
        )
        hw_results.append(
            benchmark_one(
                spike,
                cr,
                kd["func_name"],
                kd["kwargs"],
                out,
                collect_detailed_profile=collect_detailed_profile,
            )
        )
    return compile_results, compile_errors, hw_results
```

- [ ] **Step 5: Rewrite `_run_pipeline` — drop sim-failure branch**

Replace the sim-failure accumulation block (currently lines 252–268, verify) with a direct progression:

```python
def _run_pipeline(payload: dict[str, Any]) -> tuple[list[ProfileResult], dict[str, str]]:
    """Execute the per-kernel compile + benchmark pipeline.

    Returns:
        Tuple of (all_results, compiler_logs).
    """
    kernel_jobs = payload["kernel_jobs"]
    seed = payload["seed"]
    config = payload["config"]
    t_start = time.monotonic()

    first_job = next(iter(kernel_jobs.values()))
    work_dir = Path(f"/tmp/autotune-{first_job['func_name']}")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    nki_dir = work_dir / "nki"
    nki_dir.mkdir()
    neff_dir = work_dir / "neff"
    neff_dir.mkdir()

    neuron_cores = detect_neuron_cores()
    logger.info("Worker ready: %d kernels, %d CPU, %d NC", len(kernel_jobs), os.cpu_count() or 1, neuron_cores)

    kernel_data: dict[str, dict[str, Any]] = {
        kname: _process_kernel_job(kname, job, seed, nki_dir) for kname, job in kernel_jobs.items()
    }

    executor, futures = _submit_compilations(kernel_data, neff_dir)
    all_results, compiler_logs = _run_hw_benchmarks(executor, futures, kernel_data, config, neff_dir)
    logger.info("Done: %d results (%.1fs)", len(all_results), time.monotonic() - t_start)

    return all_results, compiler_logs
```

(`make_failure` is no longer referenced from worker; confirm the import tightens. `_fail_host` in `remote.py` still calls `make_failure` — keep in `types.py` import list there.)

- [ ] **Step 6: Remove the final `make_failure` import from `worker.py` if unused**

Verify: does the new `worker.py` still reference `make_failure`? If not, drop it from the `from autotune.runner.types import ...` block. Keep `compile_failure_result`, `CompileResult`, `OutputSpec`, `ProfileResult`, `ensure_venv_on_path`, `resolve_dtype`.

- [ ] **Step 7: Smoke-check the imports**

```bash
python -c "from autotune.runner import worker; print(worker.worker_main)"
```

Expected: prints a function object.

- [ ] **Step 8: Commit**

```bash
git add autotune/src/autotune/runner/worker.py
git commit -m "$(cat <<'EOF'
refactor(autotune): strip CPU-sim path from worker.py

Remove _cpu_sim_status, the CPU-sim branch in _process_kernel_job,
the sim-failure short-circuit in _run_pipeline, and every cpu_sim
kwarg. Worker becomes: parse -> compile -> benchmark -> emit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Strip CPU-sim from `autotune.runner.remote`

Drop the CPU-sim-related payload fields and the `cpu_sim_passed` accounting in `results.json`.

**Files:**
- Modify: `autotune/src/autotune/runner/remote.py`

- [ ] **Step 1: Edit `_launch_ssh_workers` payload construction**

Around line 144, replace the per-kernel payload dict with:

```python
payload["kernel_jobs"][n] = {
    "source": job.source,
    "func_name": job.func_name,
    "output_shape": list(job.output_shape),
    "tensor_specs": {name: (list(shape), dt) for name, (shape, dt) in job.input_specs.items()},
    "neuronx_cc_args": list(job.neuronx_cc_args),
    "lnc": job.lnc,
}
```

(No `nkigym_source`, `nkigym_func_name`, `atol`, `rtol`, `skip_cpu_sim`.)

- [ ] **Step 2: Edit `_kernel_index_row`**

Drop `cpu_sim_passed` from the returned dict:

```python
def _kernel_index_row(
    r: ProfileResult, total_time_s: float | None, mfu: float | None, mbu: float | None, ceiling: float | None
) -> dict:
    """Slim per-kernel index row for results.json — no embedded dicts."""
    stem = Path(r.kernel_name).stem
    return {
        "kernel_name": r.kernel_name,
        "kernel_path": f"{stem}/{stem}.py",
        "hardware_output": r.hardware_output,
        "total_time_s": total_time_s,
        "mfu_estimated_percent": mfu,
        "mbu_estimated_percent": mbu,
        "mfu_max_achievable_estimated_percent": ceiling,
    }
```

- [ ] **Step 3: Edit `_write_results_json` — drop `passed_cpu_sim`**

Delete the `passed_cpu_sim = ...` counter and its usage. Simplify the accounting loop:

```python
def _write_results_json(
    cache_dir: str, num_kernels: int, results: list[ProfileResult], hosts: list[str], wallclock_s: float
) -> None:
    """Write results.json (index) and split detailed per-kernel JSONs."""
    _write_per_kernel_profiles(cache_dir, results)

    extracted: dict[str, tuple[float | None, float | None, float | None, float | None]] = {}
    success = sbuf_oom = psum_oom = 0
    for r in results:
        total = (r.profiler_summary or {}).get("total_time")
        total_s = float(total) if isinstance(total, (int, float)) else None
        mfu = profiler_percent(r.profiler_summary, "mfu_estimated_percent")
        mbu = profiler_percent(r.profiler_summary, "mbu_estimated_percent")
        ceiling = profiler_percent(r.profiler_summary, "mfu_max_achievable_estimated_percent")
        extracted[r.kernel_name] = (total_s, mfu, mbu, ceiling)
        hw_ok = total_s is not None
        if hw_ok:
            success += 1
        if not hw_ok and "Out of memory in sbuf" in r.hardware_output:
            sbuf_oom += 1
        if not hw_ok and "Out of memory in psum" in r.hardware_output:
            psum_oom += 1

    successes = [(r, extracted[r.kernel_name]) for r in results if extracted[r.kernel_name][0] is not None]
    times = [e[0] for _, e in successes]
    mfus = [e[1] for _, e in successes if e[1] is not None]
    mbus = [e[2] for _, e in successes if e[2] is not None]

    kernel_entries = [
        _kernel_index_row(r, *extracted[r.kernel_name])
        for r in sorted(results, key=lambda r: _kernel_sort_key(r.kernel_name))
    ]

    best_kernel = min(successes, key=lambda s: s[1][0])[0].kernel_name if successes else None
    worst_kernel = max(successes, key=lambda s: s[1][0])[0].kernel_name if successes else None
    results_data = {
        "metadata": {"num_kernels": num_kernels, "wallclock_s": wallclock_s, "hosts": hosts},
        "metrics": {
            "best_total_time_s": min(times) if times else None,
            "worst_total_time_s": max(times) if times else None,
            "best_kernel": best_kernel,
            "worst_kernel": worst_kernel,
            "best_mfu": max(mfus) if mfus else None,
            "worst_mfu": min(mfus) if mfus else None,
            "best_mbu": max(mbus) if mbus else None,
            "worst_mbu": min(mbus) if mbus else None,
            "success": success,
            "sbuf_oom": sbuf_oom,
            "psum_oom": psum_oom,
        },
        "kernels": kernel_entries,
    }
    with open(os.path.join(cache_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
```

- [ ] **Step 4: Smoke-check the imports**

```bash
python -c "from autotune.runner.remote import RemoteProfiler; print(RemoteProfiler)"
```

Expected: prints the dataclass.

- [ ] **Step 5: Commit**

```bash
git add autotune/src/autotune/runner/remote.py
git commit -m "$(cat <<'EOF'
refactor(autotune): drop CPU-sim fields from remote payload and results.json

Remove nkigym_source/nkigym_func_name/atol/rtol/skip_cpu_sim from the
SSH payload. Drop cpu_sim_passed from the per-kernel index row and the
aggregate metrics block in results.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Clean up `autotune.runner.baseline`

`baseline.py` constructs a `ProfileResult` with `cpu_sim=_sim_not_run()` — that field no longer exists.

**Files:**
- Modify: `autotune/src/autotune/runner/baseline.py`

- [ ] **Step 1: Edit the result construction (around line 142)**

```python
result = ProfileResult(
    kernel_name=kernel_name,
    hardware_output=hardware_output,
    profiler_summary=profiler_summary,
    profile_detailed=profile_detailed,
    neff_b64=neff_b64,
    ntff_b64=ntff_b64,
)
```

- [ ] **Step 2: Remove `_sim_not_run` from imports**

```python
from autotune.runner.types import ProfileResult, capture_error, ensure_venv_on_path
```

- [ ] **Step 3: Update the docstring fragment at line 113**

Replace the paragraph that says "``cpu_sim`` field is marked 'not run'" with:

```
Returns:
    A ``(ProfileResult, nki_source)`` pair. ``nki_source`` is the NKI
    text emitted by the compiler's tensorizer (``lower_to_nki``);
    empty on failure.
```

- [ ] **Step 4: Smoke-check**

```bash
python -c "from autotune.runner.baseline import profile_numpy_baseline; print(profile_numpy_baseline)"
```

Expected: prints a function object.

- [ ] **Step 5: Commit**

```bash
git add autotune/src/autotune/runner/baseline.py
git commit -m "$(cat <<'EOF'
refactor(autotune): drop _sim_not_run usage from baseline.py

The cpu_sim field no longer exists on ProfileResult; baseline's numpy
path stops constructing it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Full repo regression — pytest

At this point every import should resolve. Run the whole test suite.

**Files:** none (validation only)

- [ ] **Step 1: Run pytest**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/ -x -q 2>&1 | tail -30
```

Expected: all tests pass. Failure modes to watch for:
- A test that read `cpu_sim` off a `ProfileResult` — update the test to drop that field.
- A test that constructed a `KernelJob` with `nkigym_source=...` — replace with the new slim constructor.

Fix any such tests inline; each test file fix is its own commit per the project convention.

- [ ] **Step 2: If tests fail, fix and commit each one**

Example commit message:

```bash
git commit -m "$(cat <<'EOF'
fix(test): drop cpu_sim/nkigym_source usage after autotune slim-down

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: If tests pass, proceed — no commit needed for this task.**

---

## Task 11: Rewrite `examples/matmul_lhsT_rhs.py` as a `nkigym_compile` driver

**Files:**
- Rewrite: `examples/matmul_lhsT_rhs.py`

- [ ] **Step 1: Replace the whole file**

```python
"""Tune the ``lhs_T @ rhs`` matmul end-to-end via ``nkigym_compile``.

Defines the canonical ``@nkigym_kernel`` math function and hands it to
``nkigym_compile``. Writes the canonical kernel + 100 random variants
into the cache dir, local fp32 CPU-verifies each, then profiles on the
gym fleet.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

from nkigym import nkigym_compile
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` with first-class buffer declarations."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


if __name__ == "__main__":
    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs_tune")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    nkigym_compile(
        f=matmul_lhsT_rhs_nkigym,
        input_specs=INPUT_SPECS,
        cache_dir=CACHE_ROOT,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
    print(f"[matmul_lhsT_rhs] canonical kernel: {CACHE_ROOT / 'kernel.py'}")
    print(f"[matmul_lhsT_rhs] results.json:     {CACHE_ROOT / 'results.json'}")
```

- [ ] **Step 2: Commit**

```bash
git add examples/matmul_lhsT_rhs.py
git commit -m "$(cat <<'EOF'
feat(examples): matmul_lhsT_rhs as a nkigym_compile driver

Replace the render-only demo with the full tune pipeline (canonical +
100 random variants, local fp32 verify, remote profile).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Rewrite `examples/matmul_lhs_rhs.py` as a `nkigym_compile` driver

**Files:**
- Rewrite: `examples/matmul_lhs_rhs.py`

- [ ] **Step 1: Replace the whole file**

```python
"""Tune ``lhs @ rhs`` (with inline transpose) end-to-end via ``nkigym_compile``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

from nkigym import nkigym_compile
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose

M, K, N = 2048, 2048, 2048


@nkigym_kernel
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """``lhs @ rhs`` — transposes lhs to the stationary operand layout."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    lhs_T_psum = NKIAlloc(location="psum", shape=(K, M), dtype="float32")()
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKITranspose()(src=lhs_sbuf, dst=lhs_T_psum)
    NKITensorCopy()(src=lhs_T_psum, dst=lhs_T_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


if __name__ == "__main__":
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs_tune")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    nkigym_compile(
        f=matmul_lhs_rhs_nkigym,
        input_specs=INPUT_SPECS,
        cache_dir=CACHE_ROOT,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
    print(f"[matmul_lhs_rhs] canonical kernel: {CACHE_ROOT / 'kernel.py'}")
    print(f"[matmul_lhs_rhs] results.json:     {CACHE_ROOT / 'results.json'}")
```

- [ ] **Step 2: Commit**

```bash
git add examples/matmul_lhs_rhs.py
git commit -m "$(cat <<'EOF'
feat(examples): matmul_lhs_rhs as a nkigym_compile driver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update `examples/rmsnorm_matmul.py` to pass `f=`

The numpy-input branch stays synthesis-driven, but the keyword name is now `f` (not `f_numpy`).

**Files:**
- Modify: `examples/rmsnorm_matmul.py`

- [ ] **Step 1: Edit the `nkigym_compile` call**

Replace the call body around line 53:

```python
    nkigym_compile(
        f=rmsnorm_matmul_numpy,
        input_specs=INPUT_SPECS,
        cache_dir=cache_dir,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
```

(Only the `f_numpy=` kwarg becomes `f=`. No other changes.)

- [ ] **Step 2: Commit**

```bash
git add examples/rmsnorm_matmul.py
git commit -m "$(cat <<'EOF'
refactor(examples): rmsnorm_matmul uses unified f= kwarg

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Delete `scripts/` and `debug_kernel_0000_chain.py`

**Files:**
- Delete: `scripts/` (directory)
- Delete: `debug_kernel_0000_chain.py`

- [ ] **Step 1: Confirm nothing imports from `scripts/`**

```bash
grep -rn "from scripts\|import scripts\|scripts\." /home/ubuntu/nki-autotune --include="*.py" | grep -v __pycache__
```

Expected: zero hits.

- [ ] **Step 2: Delete both**

```bash
git rm -r scripts/
git rm debug_kernel_0000_chain.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore: delete scripts/ and debug_kernel_0000_chain.py

scripts/tune_matmul_lhsT_rhs.py is superseded by examples/matmul_lhsT_rhs.py.
debug_kernel_0000_chain.py was top-level scratch never referenced anywhere.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: End-to-end smoke run (local-only)

Exercise `nkigym_compile` on a `@nkigym_kernel` input with a small shape, no gym hosts. This validates the dispatch, canonical render, verify, sampling, and per-variant verify.

**Files:** none (manual verification)

- [ ] **Step 1: Run the smoke test**

```bash
source ~/venvs/kernel-env/bin/activate
python - <<'PY'
import shutil
from pathlib import Path

from nkigym import nkigym_compile
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K = M = N = 128


@nkigym_kernel
def tiny(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


cache = Path("/tmp/nkigym_compile_smoke")
shutil.rmtree(cache, ignore_errors=True)
cache.mkdir()
nkigym_compile(
    f=tiny,
    input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
    cache_dir=cache,
    num_kernels=4,
    hosts=[],
    venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
    neuron_platform_target="trn2",
    seed=0,
)
print("smoke OK:", sorted(p.name for p in cache.iterdir()))
PY
```

Expected output (order may vary):

```
smoke OK: ['kernel.py', 'kernel_tuned_0000.py', 'kernel_tuned_0001.py', 'kernel_tuned_0002.py', 'kernel_tuned_0003.py', 'results.json']
```

The `results.json` should contain a `metadata.num_kernels: 4`, `metrics: {}`, and one entry per kernel. If `_verify` raises for a sampled variant, the run aborts with `AssertionError` — that's expected behavior and not a bug in this plan. If it raises consistently with a small-shape MatMul, investigate the renderer, not the plan.

- [ ] **Step 2: Inspect `results.json`**

```bash
python -c "import json; print(json.dumps(json.load(open('/tmp/nkigym_compile_smoke/results.json')), indent=2))"
```

Expected: the stub schema described in Task 3 (`metadata` + empty `metrics` + `kernels` list with `kernel_name` and `kernel_path`).

- [ ] **Step 3: No commit needed (validation only).**

---

## Task 16: End-to-end HW smoke run (optional, one gym host)

Skip if gym access isn't available. Otherwise, this validates that the profiler path still works after the autotune slim-down.

**Files:** none (manual verification)

- [ ] **Step 1: Temporarily reduce `num_kernels` in `examples/matmul_lhsT_rhs.py` to 4 and `hosts` to `["gym-1"]`**

Use a scratch branch or just run with a local edit; do not commit this.

- [ ] **Step 2: Run**

```bash
source ~/venvs/kernel-env/bin/activate
python examples/matmul_lhsT_rhs.py
```

Expected: completes in a few minutes; `/home/ubuntu/cache/matmul_lhsT_rhs_tune/results.json` contains `metadata.num_kernels=4`, `metrics.success>=1`, no `passed_cpu_sim` field (that field is gone).

- [ ] **Step 3: Revert the temporary edit.** No commit.

---

## Task 17: Update learnings

Record the architectural shift so future sessions don't have to rediscover it.

**Files:**
- Modify: `.claude/rules/learnings.md`

- [ ] **Step 1: Add a single entry under `## Architecture`**

Append:

```markdown
- **Unified `nkigym_compile`**: single public entry point; tag-dispatch on `f.__nkigym_kernel__` picks synthesis-first vs direct. Local fp32 verify in `nkigym/tune/verify.py` runs on canonical + every sampled variant; any miss aborts the run. Autotune worker is profile-only: `KernelJob` / `ProfileResult` carry no CPU-sim fields, `results.json` drops `passed_cpu_sim`. `hosts=[]` writes a stub `results.json` and skips remote. *(2026-05-10 ET)*
```

- [ ] **Step 2: Commit**

```bash
git add .claude/rules/learnings.md
git commit -m "$(cat <<'EOF'
docs(learnings): record unified nkigym_compile + profiler-only autotune

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final checkpoint

- [ ] `pytest test/ -q` green.
- [ ] `python examples/matmul_lhsT_rhs.py` end-to-end if gym access available (Task 16).
- [ ] `ls scripts 2>/dev/null` returns an error — directory deleted.
- [ ] `ls debug_kernel_0000_chain.py 2>/dev/null` returns an error — file deleted.
- [ ] `grep -rn "cpu_sim\|nkigym_source\|simulate_one\|compute_golden\|assert_close\|_sim_not_run\|skip_cpu_sim" autotune/ --include="*.py"` returns zero hits.
- [ ] `grep -rn "run_tune\|from nkigym.compile import _" nkigym/ test/ --include="*.py"` returns zero hits.
