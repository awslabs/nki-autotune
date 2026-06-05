# Local profiling backend + SSH/Kaizen transport shells — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the autotune profiling backend to local (on-box) mode and add two pure-bash transport shells (SSH + Kaizen) that sync code, set up the env, run an arbitrary command, and download artifacts to the laptop.

**Architecture:** Three independent layers. (1) `autotune/src/autotune/runner/` becomes in-process on a Trn2 box — delete the SSH coordinator/fan-out in `remote.py`, relocate its cache-writers to `output.py`, expose a local `profile()` in `api.py`. (2) Two bash scripts in `transport/` each do sync → env-setup → exec → download, injecting `--cache-root-dir`. (3) Driver scripts require `--cache-root-dir`. Stay on the public `nki` + `nkipy` + `spike` stack (private-nki-staging deferred).

**Tech Stack:** Python 3.10+ (`autotune` package, public `nki`/`nkipy`/`spike`, `numpy`, `ml_dtypes`), bash, `rsync` (SSH), `kaizen` CLI + `s5cmd` (Kaizen).

**Spec:** `docs/superpowers/specs/2026-06-05-local-backend-and-transport-shells-design.md`

---

## ⚠️ Verification environment — READ FIRST

**The dev laptop cannot run any of this code.** It has no `numpy`, `ml_dtypes`,
`nki`, `nkipy`, `spike`, or `shellcheck` — even `autotune.runner.types` fails
to import (`ModuleNotFoundError: ml_dtypes`). The standard local-TDD loop
("run pytest, watch it fail, make it pass") **does not apply** to the Python
tasks here.

Verification splits in two:

- **Locally available:** `bash -n <script>` (syntax check — confirmed works),
  `git`, `grep`, reading files, and `python -c "import ast; ast.parse(open(p).read())"`
  for Python *syntax* (parses without importing).
- **On-box only (kernel-env on a Trn2 box, via a transport shell or already
  on the box):** importing `autotune`, running the `profile()` path, running
  the example drivers, and any test that imports the package.

Each task states which checks are **local** and which are **on-box (deferred)**.
Do the local checks inline. Collect the on-box checks into the end-to-end
verification (Task 12) — run them once a box is reachable. **Do not claim a
Python task "passes" from the laptop; claim "syntax-parses locally, import
deferred to on-box."**

There are **no existing tests referencing the runner** (verified:
`grep -rl 'remote_profile\|RemoteProfiler\|worker_main' test/` → empty), so
nothing breaks by deleting `remote.py`. New pure-Python unit tests added here
(Task 11) are import-light where possible but still require kernel-env to run.

---

## File structure (decisions locked here)

**Layer 1 — `autotune/src/autotune/runner/`**
- `remote.py` → **deleted**. Cache-writer helpers move to `output.py`; the rest (SSH/bundle/fan-out) is dropped.
- `output.py` → gains the relocated cache writers; loses the `hosts` field on `ProfileOutput`.
- `api.py` → `remote_profile`/`RemoteProfiler` replaced by a local `profile()`.
- `worker.py` → keeps the compile→benchmark core (`_process_kernel_job`, `_submit_compilations`, `_benchmark_compiled`, `_collect_compiler_logs`, `_run_hw_benchmarks`, `_run_pipeline`); loses `worker_main`, `_parse_payload`, `_setup_env`. Renamed to `driver.py` (it is no longer a remote worker).
- `baseline.py` → keeps `profile_numpy_baseline`; loses `baseline_worker_main`.
- `types.py` → loses `_DEFAULT_VENV_PYTHON` (remote-only). **Keeps** `ensure_venv_on_path` (used on-box by `compile.py`/`baseline.py`/`driver.py`).
- `compile.py`, `benchmark.py`, `detect.py` → unchanged.
- `__init__.py` (package) → docstring rewritten to the local API.

**Layer 2 — new `transport/`**
- `transport/ssh_host.sh`, `transport/kaizen.sh` → 4-step transports.
- `transport/common.sh` → shared constants (cache roots, venv-activate line) + arg parsing, sourced by both.
- `install_neuron.sh` → one edit: configurable `VENV_DIR`.

**Layer 3 — drivers**
- `examples/matmul_lhsT_rhs.py`, `examples/kernel_transforms_repro.py` → require `--cache-root-dir`.

**Tests**
- `test/runner/test_output.py` → new; covers the relocated cache-writer + the `hosts`-free summary.

---

## Task 1: Relocate cache-writer helpers from `remote.py` into `output.py`

Move the six cache-writer helpers (and the constants they use) verbatim into
`output.py`, which already owns the `ProfileOutput` / results-display surface.
This must happen **before** deleting `remote.py` so nothing is lost.

**Files:**
- Modify: `autotune/src/autotune/runner/output.py`

- [ ] **Step 1: Add imports and module constants to the top of `output.py`**

Replace the current import block (lines 1-6) of `output.py` with:

```python
"""Profile output formatting, display, and cache-layout writers."""

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from autotune.runner.types import ProfileResult, profiler_percent

_PROFILE_SUMMARY_FILE = "profile_summary.json"
_PROFILE_DETAILED_FILE = "profile.json"
_NEFF_FILE = "file.neff"
_NTFF_FILE = "profile.ntff"
```

- [ ] **Step 2: Append the relocated cache-writer helpers to the end of `output.py`**

Append verbatim (these are moved unchanged from `remote.py`):

```python
def write_kernel_sources(cache_dir: str, kernels: dict) -> None:
    """Write kernel source files to ``<cache>/<stem>/<stem>.py``."""
    for kname, job in kernels.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, f"{stem}.py"), "w") as f:
            f.write(job.source)


def write_compiler_logs(cache_dir: str, compiler_logs: dict[str, str]) -> None:
    """Write compiler logs to ``<cache>/<stem>/log-neuron-cc.txt``."""
    for kname, log_text in compiler_logs.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, "log-neuron-cc.txt"), "w") as f:
            f.write(log_text)


def _kernel_sort_key(kernel_name: str) -> tuple[int, int, str]:
    """Natural sort key for kernel names.

    Names shaped like ``<prefix>_<N>.py`` sort numerically by ``N`` (so
    ``kernel_2.py`` precedes ``kernel_10.py``); names without a numeric
    suffix sort alphabetically after all numerically-sorted names.
    """
    stem = Path(kernel_name).stem
    tail = stem.rsplit("_", 1)[-1]
    key = (0, int(tail), stem) if tail.isdigit() else (1, 0, stem)
    return key


def _write_per_kernel_profiles(cache_dir: str, results: list[ProfileResult]) -> None:
    """Move per-kernel profiler dicts into their subfolders.

    Always writes ``<cache>/<stem>/profile_summary.json``. When detailed
    collection was on, the subfolder also receives ``profile.json`` (full
    trace), ``file.neff`` (compiled binary), and ``profile.ntff`` (raw trace)
    so offline ``neuron-profile view`` runs work without a gym host.
    """
    for r in results:
        stem = Path(r.kernel_name).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        if r.profiler_summary is not None:
            with open(os.path.join(variant_dir, _PROFILE_SUMMARY_FILE), "w") as f:
                json.dump(r.profiler_summary, f, indent=2)
        if r.profile_detailed is not None:
            with open(os.path.join(variant_dir, _PROFILE_DETAILED_FILE), "w") as f:
                json.dump(r.profile_detailed, f)
        if r.neff_b64 is not None:
            with open(os.path.join(variant_dir, _NEFF_FILE), "wb") as f:
                f.write(base64.b64decode(r.neff_b64))
        if r.ntff_b64 is not None:
            with open(os.path.join(variant_dir, _NTFF_FILE), "wb") as f:
                f.write(base64.b64decode(r.ntff_b64))


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


def write_results_json(
    cache_dir: str, num_kernels: int, results: list[ProfileResult], wallclock_s: float
) -> None:
    """Write results.json (index) and split per-kernel JSONs into subfolders."""
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
        "metadata": {"num_kernels": num_kernels, "wallclock_s": wallclock_s},
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

Note the deliberate changes from the `remote.py` originals: `_write_compiler_logs`
→ public `write_compiler_logs`; `_write_results_json` → public `write_results_json`
and its signature **drops the `hosts` parameter** (local mode has no hosts), so
`metadata` no longer carries `"hosts"`. These names are referenced by Tasks 3-5.

- [ ] **Step 3: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/output.py').read()); print('parse OK')"`
Expected: `parse OK`

(Import check is **on-box deferred** — `output.py` imports `types.py` which needs `ml_dtypes`.)

- [ ] **Step 4: Commit**

```bash
git add autotune/src/autotune/runner/output.py
git commit -m "refactor: relocate cache-writers from remote.py into output.py (hosts-free)"
```

---

## Task 2: Drop the `hosts` field from `ProfileOutput`

`ProfileOutput.hosts` is vestigial in local mode. Remove it and the summary
line that prints it.

**Files:**
- Modify: `autotune/src/autotune/runner/output.py`

- [ ] **Step 1: Remove the `hosts` field and update the docstring**

In `output.py`, change the `ProfileOutput` dataclass body. Replace:

```python
    Attributes:
        results: Per-kernel profiling results.
        compiler_logs: Map of kernel name to compiler log text.
        elapsed_s: Total wallclock time in seconds.
        hosts: SSH hostnames used for profiling.
        cache_dir: Path to cache directory, if saved.
    """

    results: list[ProfileResult]
    compiler_logs: dict[str, str]
    elapsed_s: float
    hosts: list[str]
    cache_dir: str = ""
```

with:

```python
    Attributes:
        results: Per-kernel profiling results.
        compiler_logs: Map of kernel name to compiler log text.
        elapsed_s: Total wallclock time in seconds.
        cache_dir: Path to cache directory, if saved.
    """

    results: list[ProfileResult]
    compiler_logs: dict[str, str]
    elapsed_s: float
    cache_dir: str = ""
```

- [ ] **Step 2: Remove the hosts summary line**

In `_format_summary`, delete the line:

```python
    lines.append(f"  Hosts:      {len(output.hosts)} ({', '.join(output.hosts)})")
```

- [ ] **Step 3: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/output.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add autotune/src/autotune/runner/output.py
git commit -m "refactor: drop vestigial hosts field from ProfileOutput"
```

---

## Task 3: Convert `worker.py` → `driver.py` (in-process local core)

`worker.py`'s compile→benchmark pipeline is kept; its SSH-worker shell
(`worker_main`, `_parse_payload`, `_setup_env`, the stdout hijack) is dropped.
Rename the file to `driver.py` to reflect it is no longer a remote worker, and
have it accept Python objects directly instead of a JSON payload.

**Files:**
- Create: `autotune/src/autotune/runner/driver.py`
- Delete: `autotune/src/autotune/runner/worker.py`

- [ ] **Step 1: Create `driver.py`**

```python
"""In-process NKI kernel compile + benchmark pipeline.

Runs ON a Trn2 box. Given a set of KernelJobs, compiles each to NEFF
(parallel ProcessPool), benchmarks on Neuron hardware, and returns
per-kernel ProfileResults plus compiler logs. No SSH, no bundling — the
whole driver runs in-process on the box where nki + nkipy are installed.
"""

import logging
import os
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from nkipy.runtime import BaremetalExecutor

from autotune.runner.benchmark import benchmark_one, generate_tensors
from autotune.runner.compile import compile_one, init_compile_worker
from autotune.runner.detect import detect_neuron_cores
from autotune.runner.types import (
    CompileResult,
    KernelJob,
    OutputSpec,
    ProfileResult,
    compile_failure_result,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

_OUTPUT_TENSOR_NAME = "hbm_tensor_0"


def _prepare_kernel(kname: str, job: KernelJob, seed: int, nki_dir: Path) -> dict[str, Any]:
    """Write a kernel's source to disk and generate its input tensors."""
    filename = kname if kname.endswith(".py") else f"{kname}.py"
    nki_path = nki_dir / filename
    nki_path.write_text(job.source)

    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in job.input_specs.items()}
    kwargs = generate_tensors(tensor_specs, seed)
    input_dtype_name = next(iter(job.input_specs.values()))[1]
    return {
        "nki_path": str(nki_path),
        "func_name": job.func_name,
        "output_shape": tuple(job.output_shape),
        "kwargs": kwargs,
        "input_dtype_name": input_dtype_name,
        "neuronx_cc_args": tuple(job.neuronx_cc_args),
        "lnc": int(job.lnc),
    }


def _submit_compilations(
    kernel_data: dict[str, dict[str, Any]], neff_dir: Path
) -> tuple[ProcessPoolExecutor, list[Future]]:
    """Submit all kernels for parallel compilation."""
    cpu_cores = os.cpu_count() or 1
    compile_workers = min(max(cpu_cores - 1, 1), len(kernel_data))
    executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=init_compile_worker)
    futures: list[Future] = []
    for kname, kd in kernel_data.items():
        compile_dir = neff_dir / Path(kname).stem
        compile_dir.mkdir(parents=True, exist_ok=True)
        input_shapes = {k: v.shape for k, v in kd["kwargs"].items() if hasattr(v, "ndim") and v.ndim > 0}
        futures.append(
            executor.submit(
                compile_one,
                kname,
                kd["nki_path"],
                kd["func_name"],
                input_shapes,
                kd["input_dtype_name"],
                _OUTPUT_TENSOR_NAME,
                kd["output_shape"],
                kd["input_dtype_name"],
                str(compile_dir),
                {},
                kd["neuronx_cc_args"],
                kd["lnc"],
            )
        )
    return executor, futures


def _benchmark_compiled(
    compile_futures: list[Future],
    spike: BaremetalExecutor,
    kernel_data: dict[str, dict[str, Any]],
    collect_detailed_profile: bool,
) -> tuple[list[CompileResult], list[ProfileResult], list[ProfileResult]]:
    """Benchmark each kernel as it finishes compiling."""
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
                spike, cr, kd["func_name"], kd["kwargs"], out, collect_detailed_profile=collect_detailed_profile
            )
        )
    return compile_results, compile_errors, hw_results


def _collect_compiler_logs(compile_results: list[CompileResult], neff_dir: Path, collect: bool) -> dict[str, str]:
    """Gather compiler log files if collection is enabled."""
    logs: dict[str, str] = {}
    for cr in compile_results if collect else []:
        stem = Path(cr.kernel_name).stem
        log_path = neff_dir / stem / "log-neuron-cc.txt"
        if log_path.exists():
            logs[cr.kernel_name] = log_path.read_text()
    return logs


def run_pipeline(
    kernels: dict[str, KernelJob],
    seed: int,
    collect_compiler_logs: bool,
    collect_detailed_profile: bool,
) -> tuple[list[ProfileResult], dict[str, str]]:
    """Compile + benchmark every kernel in-process on this box.

    Returns (results, compiler_logs).
    """
    lncs = {int(job.lnc) for job in kernels.values()}
    if len(lncs) > 1:
        raise RuntimeError(f"Batch mixes lnc values {lncs}; submit one lnc per profile() call")
    lnc = next(iter(lncs))
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0" if lnc == 1 else "0,1"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)

    first_job = next(iter(kernels.values()))
    work_dir = Path(f"/tmp/autotune-{first_job.func_name}")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    nki_dir = work_dir / "nki"
    nki_dir.mkdir()
    neff_dir = work_dir / "neff"
    neff_dir.mkdir()

    neuron_cores = detect_neuron_cores()
    logger.info("Driver ready: %d kernels, %d CPU, %d NC", len(kernels), os.cpu_count() or 1, neuron_cores)

    kernel_data = {kname: _prepare_kernel(kname, job, seed, nki_dir) for kname, job in kernels.items()}

    executor, futures = _submit_compilations(kernel_data, neff_dir)
    with BaremetalExecutor(verbose=0) as spike:
        compile_results, compile_errors, hw_results = _benchmark_compiled(
            futures, spike, kernel_data, collect_detailed_profile
        )
    executor.shutdown(wait=True)

    compiler_logs = _collect_compiler_logs(compile_results, neff_dir, collect_compiler_logs)
    return compile_errors + hw_results, compiler_logs
```

Differences from `worker.py` worth noting: `_setup_env` set
`NEURON_PLATFORM_TARGET_OVERRIDE` from the payload; that env var is now the
caller's/transport's responsibility (the shells export it), so it is dropped
here. The `lnc` env-var handling moves into `run_pipeline` (it was in
`_run_hw_benchmarks`). The hardcoded `NEURON_PLATFORM_TARGET_OVERRIDE="trn2"`
override from `_run_hw_benchmarks` is dropped — `neuron_platform_target` is set
once by the caller (Task 4).

- [ ] **Step 2: Delete `worker.py`**

```bash
git rm autotune/src/autotune/runner/worker.py
```

- [ ] **Step 3: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/driver.py').read()); print('parse OK')"`
Expected: `parse OK`

(Import deferred on-box — `driver.py` imports `nkipy`.)

- [ ] **Step 4: Commit**

```bash
git add autotune/src/autotune/runner/driver.py
git commit -m "refactor: worker.py -> driver.py, in-process local compile+benchmark core"
```

---

## Task 4: Rewrite `api.py` with a local `profile()` entry

Replace `remote_profile`/`RemoteProfiler` with an in-process `profile()` that
calls `driver.run_pipeline` and writes the cache via the relocated writers.

**Files:**
- Modify: `autotune/src/autotune/runner/api.py` (full rewrite)

- [ ] **Step 1: Replace the entire contents of `api.py`**

```python
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
```

- [ ] **Step 2: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/api.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
git add autotune/src/autotune/runner/api.py
git commit -m "feat: local in-process profile() replacing remote_profile"
```

---

## Task 5: Port `baseline.py` to local-only (drop the SSH worker entry)

`profile_numpy_baseline` already runs locally on a box. Drop only
`baseline_worker_main` and its now-unused imports.

**Files:**
- Modify: `autotune/src/autotune/runner/baseline.py`

- [ ] **Step 1: Delete `baseline_worker_main` and the `__main__` guard**

Remove the entire `def baseline_worker_main() -> None:` function (lines
~151-201) and the trailing:

```python
if __name__ == "__main__":
    baseline_worker_main()
```

- [ ] **Step 2: Prune imports now unused**

`baseline_worker_main` was the only user of `base64`, `json`, `sys`, and
`ensure_venv_on_path` within `baseline.py`'s top-of-file imports — but
`ensure_venv_on_path()` is **also called at line 174 inside
`baseline_worker_main` only**, so it goes too. Verify each before removing:

```bash
grep -nE "base64|json|sys|ensure_venv_on_path" autotune/src/autotune/runner/baseline.py
```

After deleting the worker function, `base64` IS still used by
`profile_numpy_baseline` (the `neff_b64` branch). `json` and `sys` are not.
`ensure_venv_on_path` is not. Update the import block to:

```python
import base64
import logging
import os
import shutil
from typing import Any, Callable

import numpy as np
from nkipy.core.compile import CompilationTarget, compile_to_neff, lower_to_nki, trace
from nkipy.runtime import BaremetalExecutor, CompiledKernel

logger = logging.getLogger(__name__)

from autotune.runner.benchmark import _collect_profiler_outputs, generate_tensors
from autotune.runner.types import ProfileResult, capture_error
```

- [ ] **Step 3: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/baseline.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Confirm no dangling references to the removed symbol**

Run: `grep -rnE "baseline_worker_main" autotune/ test/ examples/ || echo "clean"`
Expected: `clean`

- [ ] **Step 5: Commit**

```bash
git add autotune/src/autotune/runner/baseline.py
git commit -m "refactor: baseline.py local-only, drop SSH worker entry"
```

---

## Task 6: Trim `types.py` (drop remote-only `_DEFAULT_VENV_PYTHON`)

`_DEFAULT_VENV_PYTHON` was used only by `remote.py`. `ensure_venv_on_path`
stays (used by `compile.py`, `baseline.py` was pruned in Task 5, and
`driver.py` no longer imports it — but `compile.py` still does).

**Files:**
- Modify: `autotune/src/autotune/runner/types.py`

- [ ] **Step 1: Confirm `_DEFAULT_VENV_PYTHON` has no remaining users**

Run: `grep -rnE "_DEFAULT_VENV_PYTHON" autotune/`
Expected: only the definition in `types.py` (Task 3/4 already removed the `remote.py` users; `remote.py` is deleted in Task 7).

If `remote.py` still exists at this point and references it, that's fine —
Task 7 deletes `remote.py`. Re-run this grep after Task 7.

- [ ] **Step 2: Remove the `_DEFAULT_VENV_PYTHON` constant**

In `types.py`, delete:

```python
_DEFAULT_VENV_PYTHON = "/home/ubuntu/venvs/kernel-env/bin/python"
```

Keep `ensure_venv_on_path()` exactly as-is (still used by `compile.py:86`).

- [ ] **Step 3: Update the `ProfileResult` docstring reference to `remote_profile`**

In `types.py`, the `ProfileResult` docstring says
"only when the caller passes ``collect_detailed_profile=True`` to
:func:`remote_profile`". Change `remote_profile` → `profile`.

- [ ] **Step 4: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/runner/types.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 5: Commit**

```bash
git add autotune/src/autotune/runner/types.py
git commit -m "refactor: drop remote-only _DEFAULT_VENV_PYTHON from types"
```

---

## Task 7: Delete `remote.py` and update the package docstring

**Files:**
- Delete: `autotune/src/autotune/runner/remote.py`
- Modify: `autotune/src/autotune/__init__.py`

- [ ] **Step 1: Confirm nothing imports `remote.py` anymore**

Run:
```bash
grep -rnE "runner\.remote|from autotune.runner.remote|import remote|RemoteProfiler|remote_profile|remote_numpy_baseline" autotune/ test/ examples/ | grep -v "__init__.py"
```
Expected: empty (Task 4 rewrote `api.py`; the only remaining hits should be in `autotune/src/autotune/__init__.py`'s docstring, fixed next).

- [ ] **Step 2: Delete the file**

```bash
git rm autotune/src/autotune/runner/remote.py
```

- [ ] **Step 3: Rewrite the package docstring in `autotune/src/autotune/__init__.py`**

Replace the whole file with:

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKI kernel profiling backend (local, on-box).

Compiles and benchmarks NKI kernels in-process on a Trainium box::

    from autotune.runner.api import profile

    output = profile(
        kernels={"copy_v0.py": job0, "copy_v1.py": job1},
        cache_dir="/home/ubuntu/autotune_cache/copy",
        seed=42,
        neuron_platform_target="trn2",
        collect_detailed_profile=False,
    )
    print(output)

To run on a remote box, drive this through ``transport/ssh_host.sh`` or
``transport/kaizen.sh`` (sync code, set up the env, execute, download
artifacts).
"""
```

- [ ] **Step 4: Local syntax check**

Run: `python -c "import ast; ast.parse(open('autotune/src/autotune/__init__.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 5: Final dead-reference sweep**

Run:
```bash
grep -rnE "RemoteProfiler|remote_profile|remote_numpy_baseline|worker_main|_get_worker_bundle|\.hosts\b|hosts=" autotune/ | grep -v "\.pyc"
```
Expected: empty.

- [ ] **Step 6: Commit**

```bash
git add -A autotune/
git commit -m "refactor: delete remote.py SSH coordinator; local-only backend"
```

---

## Task 8: Make `install_neuron.sh` `VENV_DIR` configurable

Single edit — the only change to the env-setup script (private-nki-staging
deferred, so the install body stays public nki + nkipy + spike).

**Files:**
- Modify: `install_neuron.sh:15`

- [ ] **Step 1: Make `VENV_DIR` overridable via env var**

Replace line 15:

```bash
VENV_DIR="/home/ubuntu/venvs/kernel-env"
```

with:

```bash
VENV_DIR="${AUTOTUNE_VENV_DIR:-/home/ubuntu/venvs/kernel-env}"
```

- [ ] **Step 2: Local syntax check**

Run: `bash -n install_neuron.sh && echo "bash -n OK"`
Expected: `bash -n OK`

- [ ] **Step 3: Commit**

```bash
git add install_neuron.sh
git commit -m "chore: make install_neuron.sh VENV_DIR configurable"
```

---

## Task 9: Transport shared library `transport/common.sh`

Shared constants + arg parsing + helpers sourced by both transports.

**Files:**
- Create: `transport/common.sh`

- [ ] **Step 1: Create `transport/common.sh`**

```bash
#!/usr/bin/env bash
#
# Shared constants and helpers for the autotune transport shells.
# Sourced by ssh_host.sh and kaizen.sh — not executed directly.

# --- Hardcoded cache roots (edit here to change for all transports) ---
# MUST live under $HOME on the remote: only $HOME is S3-backed on Kaizen
# and visible to the reverse s5cmd sync. /ustore/* is ephemeral/invisible.
transport_cache_root_dir="\$HOME/autotune_cache"
local_cache_root_dir="/workplace/weittang/autotune_cache"

# Local path to the nki-autotune repo (the dir to sync to the box).
repo_root_dir="/workplace/weittang/nki-autotune"

# Remote subdir under the box's $HOME where the repo lands.
remote_repo_subdir="nki-autotune"

# Line that activates the kernel-env venv on the remote box before running.
# Overridable via AUTOTUNE_REMOTE_ACTIVATE for boxes with a different layout
# (e.g. the Kaizen py312 conda image).
remote_activate="${AUTOTUNE_REMOTE_ACTIVATE:-source \$HOME/venvs/kernel-env/bin/activate}"

# Neuron platform target exported on the box for the run.
neuron_platform_target="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"

# Files synced INTO the repo on the box should exclude these.
sync_excludes=(.git __pycache__ "*.pyc" .pytest_cache .mypy_cache build .venv node_modules)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Parse --cmd / --no-setup (shared between both transports). Each transport
# parses its own --host / --name first and passes the rest here via "$@".
USER_CMD=""
NO_SETUP=0
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cmd) USER_CMD="$2"; shift 2 ;;
            --no-setup) NO_SETUP=1; shift ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    [[ -n "$USER_CMD" ]] || die "--cmd is required"
}

# The full remote command: activate venv, cd into the repo, run the user's
# command with --cache-root-dir injected.
remote_run_cmd() {
    printf '%s && cd ~/%s && export NEURON_PLATFORM_TARGET_OVERRIDE=%s && %s --cache-root-dir %s' \
        "$remote_activate" "$remote_repo_subdir" "$neuron_platform_target" \
        "$USER_CMD" "$transport_cache_root_dir"
}

# The remote env-setup command (idempotent).
remote_setup_cmd() {
    printf 'AUTOTUNE_VENV_DIR=$HOME/venvs/kernel-env bash ~/%s/install_neuron.sh --local' \
        "$remote_repo_subdir"
}
```

- [ ] **Step 2: Local syntax check**

Run: `bash -n transport/common.sh && echo "bash -n OK"`
Expected: `bash -n OK`

- [ ] **Step 3: Commit**

```bash
git add transport/common.sh
git commit -m "feat: transport/common.sh — shared transport constants + helpers"
```

---

## Task 10: `transport/ssh_host.sh` and `transport/kaizen.sh`

The two 4-step transports.

**Files:**
- Create: `transport/ssh_host.sh`
- Create: `transport/kaizen.sh`

- [ ] **Step 1: Create `transport/ssh_host.sh`**

```bash
#!/usr/bin/env bash
#
# SSH transport: sync repo -> set up env -> run --cmd -> download artifacts.
#
# Usage:
#   transport/ssh_host.sh --host <h> --cmd "python xxx.py" [--no-setup]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=transport/common.sh
source "$SCRIPT_DIR/common.sh"

HOST=""
rest=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done
[[ -n "$HOST" ]] || die "--host is required"
parse_common_args "${rest[@]}"

# rsync exclude flags from the shared list.
rsync_excludes=()
for e in "${sync_excludes[@]}"; do
    rsync_excludes+=(--exclude "$e")
done

echo "==> [1/4] Syncing $repo_root_dir/ -> $HOST:~/$remote_repo_subdir/"
rsync -az --delete "${rsync_excludes[@]}" \
    "$repo_root_dir/" "$HOST:$remote_repo_subdir/"

if [[ "$NO_SETUP" -eq 0 ]]; then
    echo "==> [2/4] Setting up env on $HOST (idempotent)"
    ssh "$HOST" "$(remote_setup_cmd)"
else
    echo "==> [2/4] Skipping env setup (--no-setup)"
fi

echo "==> [3/4] Executing on $HOST"
ssh "$HOST" "$(remote_run_cmd)"

echo "==> [4/4] Downloading artifacts -> $local_cache_root_dir/"
mkdir -p "$local_cache_root_dir"
# transport_cache_root_dir contains a literal $HOME — expand it on the remote.
remote_cache="$(ssh "$HOST" "echo $transport_cache_root_dir")"
rsync -az "$HOST:$remote_cache/" "$local_cache_root_dir/"
echo "==> Done. Artifacts in $local_cache_root_dir/"
```

- [ ] **Step 2: Create `transport/kaizen.sh`**

```bash
#!/usr/bin/env bash
#
# Kaizen transport: sync repo -> set up env -> run --cmd -> download artifacts.
#
# Usage:
#   transport/kaizen.sh --name <desktop> --cmd "python xxx.py" [--no-setup]
#
# Prerequisites (caller's responsibility — fails loud if missing):
#   - mwinit -o done; ~/.aws/config has kaizen-access + cluster-role profiles
#   - s5cmd on PATH; kaizen CLI installed
#   - the named desktop is already RUNNING (this script does NOT start one)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=transport/common.sh
source "$SCRIPT_DIR/common.sh"

NAME=""
rest=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done
[[ -n "$NAME" ]] || die "--name is required"
parse_common_args "${rest[@]}"

command -v kaizen >/dev/null 2>&1 || die "kaizen CLI not on PATH"
command -v s5cmd  >/dev/null 2>&1 || die "s5cmd not on PATH"

echo "==> Resolving desktop $NAME (s3SyncUri + region)"
INFO="$(AWS_PROFILE=kaizen-access kaizen desktop info --name "$NAME" --output json 2>&1 | tail -1)"
echo "$INFO" | grep -q '"RUNNING"' || die "desktop $NAME is not RUNNING (run: kaizen desktop start ...)"
S3_URI="$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["s3SyncUri"])')"
REGION="$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["region"])')"
case "$S3_URI" in */) ;; *) S3_URI="$S3_URI/" ;; esac

echo "==> [1/4] Syncing $repo_root_dir/ -> desktop \$HOME/$remote_repo_subdir/"
s5cmd_excludes=()
for e in "${sync_excludes[@]}"; do
    s5cmd_excludes+=(--exclude "*/$e/*")
done
AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync "${s5cmd_excludes[@]}" \
    "$repo_root_dir/" "$S3_URI$remote_repo_subdir/"

if [[ "$NO_SETUP" -eq 0 ]]; then
    echo "==> [2/4] Setting up env on desktop (idempotent)"
    AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_setup_cmd)"
else
    echo "==> [2/4] Skipping env setup (--no-setup)"
fi

echo "==> [3/4] Executing on desktop"
AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_run_cmd)"

echo "==> [4/4] Downloading artifacts -> $local_cache_root_dir/"
mkdir -p "$local_cache_root_dir"
# transport_cache_root_dir is $HOME/autotune_cache; under $HOME it maps to the
# S3 sync prefix path 'autotune_cache/'. Reverse-sync with poll/retry because
# the $HOME->S3 export lags up to ~60s.
remote_rel="autotune_cache/"
for attempt in $(seq 1 12); do
    AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync \
        "$S3_URI$remote_rel" "$local_cache_root_dir/" || true
    if [[ -f "$local_cache_root_dir/results.json" ]]; then
        echo "==> results.json present after $attempt attempt(s)"
        break
    fi
    echo "    waiting for reverse S3 export (attempt $attempt/12)..."
    sleep 10
done
echo "==> Done. Artifacts in $local_cache_root_dir/"
```

- [ ] **Step 3: Make both executable + local syntax check**

```bash
chmod +x transport/ssh_host.sh transport/kaizen.sh
bash -n transport/ssh_host.sh && bash -n transport/kaizen.sh && echo "bash -n OK"
```
Expected: `bash -n OK`

- [ ] **Step 4: Commit**

```bash
git add transport/ssh_host.sh transport/kaizen.sh
git commit -m "feat: SSH + Kaizen transport shells (sync/setup/exec/download)"
```

---

## Task 11: Unit test for the relocated cache writers + hosts-free output

A pure-Python test for `write_results_json` + `ProfileOutput.__str__` (no nki).
**Requires kernel-env to run** (imports `autotune.runner.types` → `ml_dtypes`);
mark as on-box-deferred for execution, but write it now.

**Files:**
- Create: `test/runner/__init__.py`
- Create: `test/runner/test_output.py`

- [ ] **Step 1: Create the test package marker**

```bash
mkdir -p test/runner
touch test/runner/__init__.py
```

- [ ] **Step 2: Write `test/runner/test_output.py`**

```python
"""Unit tests for the local cache-writer + ProfileOutput summary."""

import json

from autotune.runner.output import ProfileOutput, write_results_json
from autotune.runner.types import ProfileResult


def _summary_result(name: str, total_time: float, mfu: float) -> ProfileResult:
    """A ProfileResult shaped like a successful HW run."""
    return ProfileResult(
        kernel_name=name,
        hardware_output="[128, 512] float32",
        profiler_summary={
            "total_time": total_time,
            "mfu_estimated_percent": mfu,
        },
    )


def test_write_results_json_no_hosts_key(tmp_path):
    """results.json metadata must not carry a 'hosts' key in local mode."""
    results = [_summary_result("k_0.py", 0.001, 0.9)]
    write_results_json(str(tmp_path), num_kernels=1, results=results, wallclock_s=1.5)

    data = json.loads((tmp_path / "results.json").read_text())
    assert "hosts" not in data["metadata"]
    assert data["metadata"]["num_kernels"] == 1
    assert data["metadata"]["wallclock_s"] == 1.5
    assert data["metrics"]["success"] == 1
    assert data["kernels"][0]["kernel_name"] == "k_0.py"


def test_write_results_json_sorts_kernels_numerically(tmp_path):
    """kernel_2 must precede kernel_10 in the index."""
    results = [
        _summary_result("kernel_10.py", 0.002, 0.5),
        _summary_result("kernel_2.py", 0.001, 0.5),
    ]
    write_results_json(str(tmp_path), num_kernels=2, results=results, wallclock_s=1.0)
    data = json.loads((tmp_path / "results.json").read_text())
    names = [k["kernel_name"] for k in data["kernels"]]
    assert names == ["kernel_2.py", "kernel_10.py"]


def test_profile_output_str_has_no_hosts_line():
    """The human summary must not reference hosts (field removed)."""
    out = ProfileOutput(
        results=[_summary_result("k_0.py", 0.001, 0.9)],
        compiler_logs={},
        elapsed_s=2.0,
        cache_dir="",
    )
    text = str(out)
    assert "Hosts:" not in text
    assert "Succeeded:" in text
```

- [ ] **Step 3: Local syntax check**

Run: `python -c "import ast; ast.parse(open('test/runner/test_output.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: On-box run (deferred — record the command)**

On a kernel-env box:
```bash
PYTHONPATH=autotune/src:nkigym/src pytest test/runner/test_output.py -v
```
Expected: 3 passed. **Do not check this box off until run on-box.**

- [ ] **Step 5: Commit**

```bash
git add test/runner/__init__.py test/runner/test_output.py
git commit -m "test: cache-writer + hosts-free ProfileOutput summary"
```

---

## Task 12: Driver scripts require `--cache-root-dir`

Update the two example drivers to take `--cache-root-dir` instead of a
hardcoded `CACHE_DIR`. These are nkigym rollout/repro scripts (they do not call
`profile()`), so the change is purely the cache-root convention so the
transports can drive them uniformly.

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py`
- Modify: `examples/kernel_transforms_repro.py`

- [ ] **Step 1: `matmul_lhsT_rhs.py` — add argparse, replace hardcoded CACHE_DIR**

At the top of the file, add `import argparse` next to the other stdlib imports
(`import importlib.util`, `import os`, `import random`, `import shutil`).

Replace the `if __name__ == "__main__":` block's first lines:

```python
if __name__ == "__main__":
    CACHE_DIR = "/home/ubuntu/cache/matmul_lhsT_rhs"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
```

with:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root-dir", required=True)
    args = parser.parse_args()
    CACHE_DIR = os.path.join(args.cache_root_dir, "matmul_lhsT_rhs")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
```

- [ ] **Step 2: `kernel_transforms_repro.py` — same treatment**

`_check_numerics` reads the module-global `CACHE_DIR` (set in `__main__`); keep
that contract. Add `import argparse` next to its stdlib imports (`import
importlib.util`, `import os`, `import shutil`, `import sys`).

Replace:

```python
if __name__ == "__main__":
    CACHE_DIR = "/home/ubuntu/cache/kernel_transforms_repro"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
```

with:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root-dir", required=True)
    args = parser.parse_args()
    CACHE_DIR = os.path.join(args.cache_root_dir, "kernel_transforms_repro")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
```

- [ ] **Step 3: Local syntax check**

```bash
python -c "import ast; ast.parse(open('examples/matmul_lhsT_rhs.py').read()); ast.parse(open('examples/kernel_transforms_repro.py').read()); print('parse OK')"
```
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add examples/matmul_lhsT_rhs.py examples/kernel_transforms_repro.py
git commit -m "feat: drivers require --cache-root-dir (drop hardcoded CACHE_DIR)"
```

---

## Task 13: Update `AGENTS.md` / `CLAUDE.md` invocation notes

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a transport section to `AGENTS.md`**

After the "Repository Setup" section in `AGENTS.md`, add:

```markdown
## Running on Trainium (transports)

The profiling backend (`autotune/src/autotune/runner/`) runs **on a Trn2
box** (`from autotune.runner.api import profile`). `nkigym` also requires
`nki` installed, so drivers run on the box, not the laptop.

Drive a box from the laptop with a transport shell — each syncs the repo,
sets up the env (idempotent; `--no-setup` to skip), runs the command, and
downloads artifacts to the local cache root:

    transport/ssh_host.sh --host gym-1   --cmd "python examples/matmul_lhsT_rhs.py"
    transport/kaizen.sh   --name trn2-exp --cmd "python examples/matmul_lhsT_rhs.py"

Cache roots are hardcoded in `transport/common.sh`
(`transport_cache_root_dir` under the box's `$HOME`; `local_cache_root_dir`
on the laptop). The shell injects `--cache-root-dir`. Already on a box?
Run directly: `python examples/matmul_lhsT_rhs.py --cache-root-dir <dir>`.
```

- [ ] **Step 2: Note the driver convention in `CLAUDE.md`**

In `CLAUDE.md`, under the "Development Environment" section, append:

```markdown

Driver scripts (examples) require `--cache-root-dir`. To run on remote
Trainium, use `transport/ssh_host.sh` or `transport/kaizen.sh` (see
AGENTS.md → Running on Trainium).
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: document local backend + transport shells"
```

---

## Task 14: End-to-end verification (on-box; collects all deferred checks)

This task runs **on a kernel-env Trn2 box** (reach it via a transport shell or
work on the box directly). It is the single place all deferred import/run
checks are paid off.

- [ ] **Step 1: Package imports cleanly**

On the box:
```bash
PYTHONPATH=autotune/src:nkigym/src python -c "from autotune.runner.api import profile; from autotune.runner.output import write_results_json, ProfileOutput; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 2: Unit tests pass**

```bash
PYTHONPATH=autotune/src:nkigym/src pytest test/runner/test_output.py -v
```
Expected: 3 passed. (Now check off Task 11 Step 4.)

- [ ] **Step 3: Existing suite still green (no runner regressions)**

```bash
PYTHONPATH=autotune/src:nkigym/src pytest test/ -q
```
Expected: same pass/skip counts as before this change (the runner had no tests; nkigym/transform suite unaffected).

- [ ] **Step 4: Example driver runs end-to-end with --cache-root-dir**

```bash
PYTHONPATH=autotune/src:nkigym/src python examples/kernel_transforms_repro.py --cache-root-dir /tmp/autotune_cache_test
```
Expected: exits 0, prints "All 15 ladder rungs ... reproduced", and
`/tmp/autotune_cache_test/kernel_transforms_repro/kernel.py` exists.

- [ ] **Step 5: Transport dry-run (syntax + arg parsing only, no box)**

Locally (no SSH/Kaizen needed):
```bash
bash -n transport/common.sh transport/ssh_host.sh transport/kaizen.sh && echo "shells OK"
transport/ssh_host.sh --cmd "x" 2>&1 | grep -q "host is required" && echo "ssh arg-guard OK"
transport/kaizen.sh --name n 2>&1 | grep -q "cmd is required" && echo "kaizen arg-guard OK"
```
Expected: `shells OK`, `ssh arg-guard OK`, `kaizen arg-guard OK`.

- [ ] **Step 6: Full transport smoke (requires a reachable box — optional but recommended)**

With a running Kaizen desktop or SSH host:
```bash
transport/kaizen.sh --name <desktop> --cmd "python examples/kernel_transforms_repro.py"
ls /workplace/weittang/autotune_cache/kernel_transforms_repro/kernel.py
```
Expected: the command completes and the artifact appears in the **local**
cache root, proving sync → setup → exec → download.

---

## Self-review notes

- **Spec coverage:** Layer 1 strip = Tasks 1-7; `install_neuron.sh` edit =
  Task 8; transports = Tasks 9-10; `--cache-root-dir` drivers = Task 12;
  docs = Task 13; tests + e2e = Tasks 11, 14. Deferral of
  private-nki-staging/baseline-rewrite is honored (no task touches the nki/
  nkipy import surface beyond deletion of dead SSH code).
- **Verification honesty:** every Python task uses `ast.parse` locally and
  defers import/run to Task 14 (laptop lacks numpy/ml_dtypes/nki) — stated
  up front.
- **Type/name consistency:** `write_results_json` (no `hosts` arg),
  `write_compiler_logs`, `write_kernel_sources`, `run_pipeline(kernels, seed,
  collect_compiler_logs, collect_detailed_profile)`, and `profile(kernels,
  cache_dir, seed, neuron_platform_target, collect_detailed_profile)` are used
  identically across Tasks 1, 3, 4, 11. `ProfileOutput` constructed without
  `hosts` in Task 4 matches its field removal in Task 2.
```

