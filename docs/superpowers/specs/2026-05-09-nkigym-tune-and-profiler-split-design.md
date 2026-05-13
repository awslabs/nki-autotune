# Unified `nkigym_compile` + autotune-as-profiler-only

**Date:** 2026-05-09
**Status:** Design
**Scope:** Collapse `run_tune` + `nkigym_compile` into a single `nkigym_compile`
entry point that dispatches on whether its input is a live
`@nkigym_kernel` callable or plain numpy. Move local CPU verify into
nkigym. Strip CPU-sim plumbing from autotune. Rewrite example drivers.
Delete `scripts/`.

---

## Goals

1. **Single public entry point** — `nkigym_compile(f, input_specs, cache_dir, num_kernels, hosts, ...)`. Tag-dispatch on `f`:
   - `f` is `@nkigym_kernel`-decorated → skip synthesis, use directly.
   - `f` is plain numpy → run synthesis, write `<cache>/f_nkigym.py`, load, verify.
   Then the common pipeline: canonical render → local verify → sample `num_kernels` variants → render + local-verify each → `remote_profile` (or skip if `hosts=[]`).
2. **Profiler boundary**: `autotune.remote_profile` does one job — compile + benchmark NKI sources on remote Trainium hosts. No CPU sim, no golden, no tolerances in the payload.
3. **Delete `scripts/`** and the stale `debug_kernel_0000_chain.py`.
4. **Rewrite `examples/*.py`** to be end-to-end tune drivers (not render-only demos). All three call `nkigym_compile` with the same shape; only the input type differs.

## Non-goals

- No changes to the tune sampler (`enumerate_pool`, `sample_pool`, atom legality, hash_forest).
- No changes to the rendering path (`build_initial_ir`, `render`).
- No changes to remote SSH orchestration, bootstrap, or per-kernel cache layout other than the fields removed.
- No changes to existing unit tests in `test/codegen/` and `test/tune/` — those use the IR API directly and never touched `run_tune`.

---

## Architecture

Two layers with a clean boundary — nkigym owns everything that knows what `f_nkigym` means; autotune only knows how to ship sources to a gym and read back `profile_summary.json`.

### `nkigym.compile.nkigym_compile` (public API, unified)

```python
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
    """Compile f → random kernel variants, local-verify, profile.

    Dispatches on ``f``:
      * ``@nkigym_kernel``-decorated  → use directly.
      * plain numpy                   → synthesise f_nkigym first.
    """
```

Pipeline:

1. **Dispatch**: if `getattr(f, "__nkigym_kernel__", False)`, set `f_nkigym = f`. Else:
   - `source = compile_numpy_to_nkigym(f, input_specs)`
   - Write `<cache_dir>/f_nkigym.py = source`.
   - `f_nkigym = _load_f_nkigym(<cache_dir>/f_nkigym.py)`.
   - `_verify_fns(f_nkigym, f_numpy=f, input_specs)` — raise on mismatch.
2. `module = build_initial_ir(f_nkigym, _to_canonical_specs(input_specs))`.
3. `source₀ = render(module)` → write `<cache_dir>/kernel.py`.
4. `_verify(source₀, f_nkigym, input_specs)` — raise `AssertionError` on mismatch.
5. `pool = enumerate_pool(module, max_pool_size=100*num_kernels, rng=rng)`.
   `sampled = sample_pool(pool, num_kernels, rng)`.
6. For each `(i, Mᵢ)`:
   - `sourceᵢ = render(Mᵢ)` → write `<cache_dir>/kernel_tuned_{i:04d}.py`.
   - `_verify(sourceᵢ, f_nkigym, input_specs)` — raise on mismatch, naming the kernel file.
7. If `hosts == []`: write a minimal `results.json` (index only, no profile metrics). Return.
8. Else: build `kernels: dict[str, KernelJob]` — one entry per sampled kernel with the slimmed `KernelJob` fields (see below). Call `remote_profile(kernels, hosts, cache_dir, seed, neuron_platform_target, venv_python, collect_detailed_profile)`.

Local verify fails the whole run. HW failures in `remote_profile` are tolerated and reported in `results.json` (unchanged behavior for HW failures).

### Tag for dispatch: `@nkigym_kernel`

`nkigym_kernel` (in `nkigym/ops/base.py`) sets `wrapper.__nkigym_kernel__ = True` on the returned `functools.wraps`-decorated callable. `nkigym_compile` checks that attribute. A plain numpy callable won't have it; a decorated one will.

### `nkigym.tune.verify._verify` (new helper)

```python
def _verify(
    kernel_source: str,
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """fp32 CPU sim vs f_nkigym golden; raise AssertionError on mismatch."""
```

- Replace `nl.bfloat16` / `nl.float16` with `nl.float32` in `kernel_source` (matches current `_cpu_sim_check` and worker `simulate_one`).
- `exec` the fp32 source into a namespace, pull out `kernel_fn = ns[f_nkigym.__name__]`.
- Draw reproducible fp32 inputs (`np.random.default_rng(_SIM_SEED)`) shaped by `input_specs`.
- `actual = nki.simulate(kernel_fn)(**inputs)`; if tuple take `actual[0]`.
- `expected = f_nkigym(**inputs)` — `f_nkigym` runs through the `NKIOp.__call__` numpy interpretation; same math that `compute_golden` runs today by exec'ing `nkigym_source`. Since we have the callable, no exec needed.
- Compare with `np.allclose(actual, expected, atol=5e-3, rtol=5e-3)`. Raise with max abs / max rel diff on mismatch.

Tolerances are package-level constants (`_ATOL = _RTOL = 5e-3`), identical to today.

### `_verify_fns` helper (numpy-input path only)

```python
def _verify_fns(
    f_nkigym: Callable[..., np.ndarray],
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Check that synthesised f_nkigym agrees with f_numpy at fp32."""
```

Draw the same fp32 inputs `_verify` uses. Call both; compare with `atol=rtol=5e-3`. Raises `AssertionError` on mismatch. Guards against synthesis producing math-wrong nkigym programs before we spend cycles tuning them.

### `autotune.remote_profile` (slimmed)

`remote_profile` retains its signature (called by `nkigym_compile`), but the payload it sends to workers loses every CPU-sim field. Signature unchanged:

```python
def remote_profile(
    kernels: dict[str, KernelJob],
    hosts: list[str],
    cache_dir: str,
    seed: int,
    neuron_platform_target: str,
    venv_python: str,
    collect_detailed_profile: bool,
) -> ProfileOutput: ...
```

`seed` still drives remote input-tensor generation for benchmarking (deterministic `generate_tensors` call).

---

## Data type changes

### `autotune.runner.types.KernelJob`

**Before** (11 fields):
```python
class KernelJob(NamedTuple):
    source: str
    func_name: str
    output_shape: tuple[int, ...]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    nkigym_source: str
    nkigym_func_name: str
    atol: float
    rtol: float
    neuronx_cc_args: tuple[str, ...] = ()
    skip_cpu_sim: bool = False
    lnc: int = 1
```

**After** (6 fields):
```python
class KernelJob(NamedTuple):
    source: str
    func_name: str
    output_shape: tuple[int, ...]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    neuronx_cc_args: tuple[str, ...] = ()
    lnc: int = 1
```

### `autotune.runner.types.ProfileResult`

**Drop** `cpu_sim: dict` field. Worker never populates it. `hardware_output` keeps its role (compile errors, runtime failures, empty string on success).

### `results.json`

- `metrics.passed_cpu_sim` → removed.
- Per-kernel `kernel_entries[i].cpu_sim_passed` → removed.
- Everything else unchanged.

---

## Files changed

### Delete

| Path | Reason |
|---|---|
| `scripts/` (whole dir) | Per request — `tune_matmul_lhsT_rhs.py` is the only inhabitant and it's superseded by the new `examples/matmul_lhsT_rhs.py`. |
| `debug_kernel_0000_chain.py` | Stale, top-level, not referenced anywhere. |
| `autotune/src/autotune/runner/compare.py` | Only `assert_close`, only used by the worker's deleted CPU-sim path. |
| `nkigym/src/nkigym/tune/stage.py` | Folded into `nkigym/compile.py` (unified driver). |

### New

| Path | Purpose |
|---|---|
| `nkigym/src/nkigym/tune/verify.py` | `_verify`, `_verify_fns`, `_draw_fp32_inputs` + tolerances. Local-only; no autotune dependency. |

### Modify

| Path | Change |
|---|---|
| `nkigym/src/nkigym/ops/base.py` | `nkigym_kernel` sets `wrapper.__nkigym_kernel__ = True` on the returned wrapper. |
| `nkigym/src/nkigym/compile.py` | Rewrite as the unified tag-dispatch driver. Imports `build_initial_ir` / `render` / `enumerate_pool` / `sample_pool` / `remote_profile` / `KernelJob` / `_verify` / `_verify_fns` / `_load_f_nkigym` / `compile_numpy_to_nkigym`. Body runs the 8-step pipeline above. |
| `nkigym/src/nkigym/tune/stage.py` | **Delete.** Its body (minus the two dispatch paths we're dropping) folds into `compile.py`. |
| `nkigym/src/nkigym/tune/__init__.py` | Drop any export of `run_tune`. |
| `nkigym/src/nkigym/__init__.py` | Lazy `__getattr__` unchanged — only `nkigym_compile` is exposed. |
| `autotune/src/autotune/runner/types.py` | Slim `KernelJob` and `ProfileResult` per above. `make_failure`/`compile_failure_result` lose their `cpu_sim` kwarg. |
| `autotune/src/autotune/runner/worker.py` | Remove `_cpu_sim_status`, `compute_golden`/`simulate_one`/`assert_close` imports, CPU-sim branch in `_process_kernel_job`, sim-failure short-circuit in `_run_pipeline`. Worker becomes: parse → compile (all kernels) → benchmark → emit. |
| `autotune/src/autotune/runner/benchmark.py` | Delete `compute_golden`, `simulate_one` and their imports. Keep `benchmark_one`, `generate_tensors`. |
| `autotune/src/autotune/runner/remote.py` | Drop `nkigym_source`/`nkigym_func_name`/`atol`/`rtol`/`skip_cpu_sim` from payload in `_launch_ssh_workers`. Drop `cpu_sim_passed` field + `passed_cpu_sim` counter in `_write_results_json`. |
| `autotune/src/autotune/runner/api.py` | Unchanged signature. Internal call path unchanged. |
| `examples/matmul_lhsT_rhs.py` | Rewrite to define `@nkigym_kernel matmul_lhsT_rhs_nkigym(...)`, call `nkigym_compile(f=matmul_lhsT_rhs_nkigym, ..., num_kernels=100, hosts=[...])`. Cache: `/home/ubuntu/cache/matmul_lhsT_rhs_tune/`. |
| `examples/matmul_lhs_rhs.py` | Same treatment. Cache: `/home/ubuntu/cache/matmul_lhs_rhs_tune/`. |
| `examples/rmsnorm_matmul.py` | Public call is now `nkigym_compile(f=rmsnorm_matmul_numpy, ...)` — same name, numpy input, `nkigym_compile` takes the synthesis branch. Cache unchanged. |

---

## Error handling

| Situation | Behavior |
|---|---|
| Canonical `_verify` fails | `AssertionError("canonical kernel vs f_nkigym: max_abs=X, max_rel=Y ...")`. No variants produced. |
| Any variant `_verify` fails | `AssertionError("kernel_tuned_NNNN.py vs f_nkigym: ...")`. No `remote_profile` call. |
| `hosts == []` | Skip `remote_profile`. Write `results.json` with `metadata.hosts=[]`, `metadata.wallclock_s=0.0`, `metadata.num_kernels`, `metrics={}` (or zero'd aggregates), `kernels=[{kernel_name, kernel_path}, ...]`. Return normally. |
| Remote HW compile error / SBUF OOM / PSUM OOM | Logged in `results.json`, **no raise**. |
| Synthesis fails (numpy-input branch) | Propagates from `compile_numpy_to_nkigym`. |
| `f_numpy` vs `f_nkigym` mismatch (numpy-input branch) | `AssertionError("synthesised f_nkigym vs f_numpy: ...")`. No tune run. |

---

## Rollout

Single commit (or a small stack if review prefers) — no migration period needed since nothing outside the repo consumes these APIs.

1. Add `__nkigym_kernel__ = True` tag in `nkigym_kernel` decorator.
2. Add `nkigym/tune/verify.py`.
3. Rewrite `compile.py` to the unified tag-dispatch driver.
4. Delete `tune/stage.py`; clean `tune/__init__.py`.
5. Rewrite the two matmul examples; update `rmsnorm_matmul.py` if needed.
6. Strip autotune CPU-sim plumbing.
7. Delete `scripts/`, `debug_kernel_0000_chain.py`, `compare.py`.
8. Run `pytest test/` + one example end-to-end (matmul_lhsT_rhs with `num_kernels=4`, one gym host) to confirm HW path still works.

## Alternatives considered

- **Two separate entry points `nkigym_tune` and `nkigym_compile`**: rejected — both run the same pipeline (verify + canonical + sample + profile); the only difference is an optional synthesis pre-step. Splitting them hides that they're the same operation and forces callers to pick the "right" name based on input type.
- **Explicit `f_nkigym=` / `f_numpy=` kwargs (exactly one required)**: rejected — more verbose at call sites; tag-based dispatch is one attribute read and preserves positional form. `@nkigym_kernel` already marks the wrapper; the tag just exposes that marker.
- **Keep `compute_golden`/`simulate_one` in autotune for "someone might want it"**: rejected per project code-style (dead code must go); nkigym verify runs in-process with the live callable, so the exec-based form is not reusable without adaptation anyway.
- **Keep explicit-rewrites path**: rejected — no production caller uses it; tests call `build_initial_ir` + `render` + rewrites directly.
- **`nkigym_compile` takes a path, not a callable**: rejected — callers already have the decorated function in scope; forcing a disk round-trip adds no value. The numpy-input branch still writes `f_nkigym.py` for persistence, then loads it before the tune step.
