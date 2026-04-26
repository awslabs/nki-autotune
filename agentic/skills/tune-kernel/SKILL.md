---
name: tune-kernel
description: Hand-tune an nkigym KernelIR for any nkigym math function to beat the nkipy compiler baseline.
---

Tunes one nkigym kernel by editing `KernelIR` knobs and measuring on Trainium. Generic over any nkigym math function.

**Goal: reach `TARGET_MFU` (set by the user in `workload.py`).** The skill keeps iterating — new rewrite variants, new knob sweeps, new batches — until the running-best MFU (see `summary.json → tuning.batch_<latest>.mfu`) is ≥ `TARGET_MFU`. That MFU threshold is the one and only stop criterion.

## Reference material

- `nkigym/src/nkigym/kernel_ir/ir.py` — dataclasses (`KernelIR`, `Op`, `PhysicalBuffer`, `BufferScope`, `NumBuffers`)
- `nkigym/src/nkigym/kernel_ir/build.py` — `build_ir` parser
- `nkigym/src/nkigym/kernel_ir/validate.py` — `is_valid(ir)` boolean + `validity_report(ir)` structured fix hints
- `nkigym/src/nkigym/kernel_ir/sample.py` — `sample(ir, rng)` joint random draw over all 5 knobs; `knob_signature(ir)` for dedup
- `nkigym/src/nkigym/kernel_ir/rewrites/` — available IR rewrites + `enumerate_rewrite_combinations(ir, rewrites)`
- `nkigym/src/nkigym/ops/<op>.py` — gadget + op class for each NKIOp
- `nkigym/src/nkigym/codegen/render.py` — how knobs map to emitted code
- `autotune/src/autotune/runner/tune_session.py` — `dump_baseline`, `submit_batch` for this skill
- `autotune/src/autotune/runner/api.py` — `remote_profile` one-shot entry
- `autotune/src/autotune/runner/remote.py` — `RemoteProfiler`, `remote_numpy_baseline`
- `/home/ubuntu/venvs/kernel-env/lib/python3.12/site-packages/nki/` — NKI Python API (`isa/`, `language/`, `simulator.py`)
- `/home/ubuntu/shared_workplace/KaenaCompiler/neuronxcc` — compiler source
- `/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary` — hand-written reference kernels

## Invocation

Invoked as `tune-kernel <workload.py>` (e.g. `tune-kernel /home/ubuntu/cache/matmul_lhs_rhs/matmul_lhs_rhs.py`). `workload.py` MUST define **exactly these six names** at module top level with these exact spellings:

| name | type | purpose |
|---|---|---|
| `f_nkigym` | `Callable` ending `return <name>`, using only `NKIOp` subclasses | the math function being tuned |
| `f_numpy` | `Callable[..., np.ndarray]` | same math in numpy, for the nkipy compiler baseline |
| `INPUT_SPECS` | `dict[str, tuple[tuple[int, ...], str]]` | `{param_name: (shape, dtype_str)}` per positional param |
| `CACHE_ROOT` | `str` or `pathlib.Path` | tuning cache root directory |
| `HOSTS` | `list[str]` | SSH hostnames (kernels round-robin across them; one host is fine, e.g. `["gym-3"]`) |
| `TARGET_MFU` | `float` | target MFU percentage — skill keeps iterating until reached |

If any required name is missing, the skill MUST abort with a clear error listing the missing names — no fallback, no guessing.

```python
# workload.py example
import numpy as np
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose

def f_nkigym(lhs, rhs):
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output

def f_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs @ rhs

M, K, N = 2048, 2048, 2048
INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
CACHE_ROOT = "/home/ubuntu/cache/matmul_lhs_rhs_agentic"
HOSTS = ["gym-3"]           # or ["gym-1", "gym-2", "gym-3"] to parallelize
TARGET_MFU = 89.0           # keep iterating until running-best MFU ≥ this
```

Run `source ~/venvs/kernel-env/bin/activate` first.

## Step 0 — Load the user's workload file

`<workload.py>` is the path the user passed to the skill.

```python
import importlib.util, sys

spec = importlib.util.spec_from_file_location("workload", "<workload.py>")
mod = importlib.util.module_from_spec(spec)
sys.modules["workload"] = mod            # required for inspect.getmodule(f_nkigym)
spec.loader.exec_module(mod)

required = ["f_nkigym", "f_numpy", "INPUT_SPECS", "CACHE_ROOT", "HOSTS", "TARGET_MFU"]
missing = [n for n in required if not hasattr(mod, n)]
if missing:
    raise RuntimeError(f"workload.py is missing required names: {missing}. "
                       f"See Invocation section.")

f_nkigym    = mod.f_nkigym
f_numpy     = mod.f_numpy
INPUT_SPECS = mod.INPUT_SPECS
CACHE_ROOT  = mod.CACHE_ROOT
HOSTS       = mod.HOSTS
TARGET_MFU  = float(mod.TARGET_MFU)
assert callable(f_nkigym) and callable(f_numpy)
assert isinstance(INPUT_SPECS, dict)
assert isinstance(HOSTS, list) and all(isinstance(h, str) for h in HOSTS) and HOSTS
```

## Step 1 — Dump the nkipy baseline

```python
from autotune.runner.tune_session import dump_baseline

baseline = dump_baseline(f_numpy, f_nkigym, INPUT_SPECS, CACHE_ROOT,
                         host=HOSTS[0], target_mfu=TARGET_MFU)
print(f"nkipy: {baseline.mfu:.3f}%  {baseline.min_ms:.4f} ms  target: {TARGET_MFU:.3f}%")
```

`dump_baseline` runs the numpy function through `nkipy` + `neuronx-cc` on one host and writes:

- `<CACHE_ROOT>/nkipy_baseline/nkipy_baseline.py` — compiler-emitted NKI source
- `<CACHE_ROOT>/nkipy_baseline/baseline.json` — `{source, mfu, min_ms}`
- `<CACHE_ROOT>/summary.json` — seeded with `function`, `baseline`, `target_mfu`

Read `nkipy_baseline.py` to see the compiler's tiling / buffer layout / instruction choice. **Start by mirroring it.** Translate what the compiler picked into your first IR configuration — tile shapes → `ltiles_per_block`, loop nest order → `dim_order`, which buffers live across the outer loop → `buffer_scopes` / `emission_depth`, any ping-pong / rotation → `num_buffers`. The compiler's output is already a strong local optimum; starting there gives an informed floor to tune *from*, rather than climbing from a random default. Submit this mirror as the first kernel in batch 0 alongside the rewrite-variant defaults.

## Step 2 — Build the canonical KernelIR

```python
from nkigym.kernel_ir import build_ir
ir = build_ir(f_nkigym, INPUT_SPECS)
print(repr(ir))
```

Read `repr(ir)` for: `dimensions` (dim ids + roles), `physical_buffers` (SBUF buffer names + tile shape), `ops` (op graph). Your knob dicts key off the buffer names printed here.

## Step 3 — Enumerate IR-rewrite variants

```bash
ls nkigym/src/nkigym/kernel_ir/rewrites/
```

Rewrites restructure the op graph and change which buffers exist. Every subset of applicable rewrites gives a distinct starting point for knob tuning — tune over all of them. `enumerate_rewrite_combinations(ir, rewrites)` yields one `(name, variant_ir)` per subset, deduping no-op rewrites (whose patterns aren't present) against `"base"`.

```python
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.rewrites import LoadTranspose, enumerate_rewrite_combinations

base_ir = build_ir(f_nkigym, INPUT_SPECS)
variants = dict(enumerate_rewrite_combinations(base_ir, [LoadTranspose()]))
"""For K rewrites, up to 2**K variants: {"base", "LoadTranspose",
"LoadTranspose+OnlineFusion", ...}. Subsets whose IR repr matches
an earlier variant are dropped."""
for name, ir in variants.items():
    print(name)
    print(repr(ir))   # physical_buffers + ops differ per variant → knob dicts key off them
```

A rewrite leaves the IR unchanged if its pattern isn't present, so always including the rewrite is safe; keeping both `base` and `<rewrite>-applied` versions is what matters. See `examples/matmul_lhs_rhs.py` for the pattern.

When sweeping knobs (step 4), produce one set of candidate IRs per variant — the best knob setting under one rewrite configuration is not necessarily the best under another.

## Step 4 — Set the 5 knobs

`KernelIR` has exactly five tunable fields. Before writing values, reason about what each knob physically does to the emitted kernel — knobs interact with each other:

- **`dim_order: list[str]`** — permutation of dim ids. Controls the loop nest order, which controls **what loops are open at each buffer's first use**. Store fires once all `ACCUMULATION`/`SEQUENTIAL` loops close (at position = `first_acc_position`); any dim to the right of that is already closed when the output lands in HBM. Reducing dim innermost = reuse of loaded operands; reducing dim outermost = drain-per-operand, shorter accumulator lifetime.
- **`ltiles_per_block: dict[str, int]`** — how many logical tiles one block iteration processes per dim. Raises `allocate_buffers` sizes (base = `Π tile × ltiles_per_block`) and shrinks the outer `num_blocks` loop trip count. Matmul inner body emits `ltiles[m] × ltiles[n]` PSUM tiles simultaneously; transpose emits `ltiles[P] × ltiles[F]`; those sums share the 8-bank PSUM budget.
- **`buffer_scopes: dict[str, BufferScope]`** — sizing (NOT emission placement): `INNER` = 0 full-span dims (one tile), `MIDDLE` = outermost-in-`dim_order` axis per-block + others full, `OUTER` = all full. Matmul accumulators MUST cover every output dim whose `dim_order` position ≥ `first_acc_position`; transpose src/dst pairs MUST share a scope.
- **`num_buffers: dict[str, NumBuffers]`** — rotation for DMA/compute overlap. `NumBuffers(num_p_buffers=N, ...)` emits `[i_block_<p_axis> % N]` ping-pong slots. Rotation axis MUST be open at the buffer's `first_use_depth`; rotation multiplies live bytes by `N`.
- **`emission_depth: dict[str, int]`** — loop depth at which `allocate_buffers` emits. `0` = kernel top (lifetime = whole kernel), `N` = inside `N` outer loops (lifetime = that loop iteration). Must be ≤ `first_use_depth(buf)`.

**The knobs are not independent.** Changing `dim_order` shifts `first_use_depth` for every buffer, which invalidates prior `emission_depth` and `num_buffers` choices; widening `buffer_scopes` can violate the accumulator-coverage rule after a `dim_order` swap. Treat each (dim_order, ltiles, scopes, num_buffers, emission_depth) as one joint configuration.

Assign directly to `ir.dim_order`, `ir.ltiles_per_block`, `ir.buffer_scopes`, `ir.num_buffers`, `ir.emission_depth`. Only set entries you want to override; anything missing uses `build_ir` defaults. Buffer-keyed dicts key off the names printed in `repr(ir)`.

## Step 5 — Validate

Prefer `validity_report(ir)` over `is_valid(ir)`: a `False` from the boolean API silently discards IRs that are one knob away from working — which is most of the interesting neighborhood of any good kernel. The report returns one `ValidityFailure` per failed check with a concrete fix hint:

```python
from nkigym.kernel_ir.validate import validity_report
report = validity_report(ir)
if report:
    for f in report:
        print(f"{f.check}({f.buffer}): {f.detail}\n  → {f.fix_hint}")
    """Nudge the named knob rather than discarding the IR — the hint
    points at exactly which field to change."""
    continue
```

The four checks are:

| check | what it guards | typical fix |
|---|---|---|
| `emission_depth_ceiling` | `emission_depth[buf] ≤ first_use_depth(buf)` | lower `emission_depth[buf]`, or change `dim_order`/`scope` to push `first_use_depth` deeper |
| `rotation_axis_closed` | `num_*_buffers` axis must be open at first use | set that axis's `num_*_buffers=None`, or move the axis earlier in `dim_order` |
| `transpose_scope_mismatch` | transpose src/dst share one scope | copy one side's scope to the other |
| `accumulator_coverage` | matmul accumulator spans full extent on closed output dims | widen accumulator scope, or move the uncovered dim before any `ACCUMULATION` dim in `dim_order` |

## Step 6 — Submit a batch

```python
from autotune.runner.tune_session import submit_batch

results = submit_batch([ir], f_nkigym, INPUT_SPECS, CACHE_ROOT, hosts=HOSTS)           # one kernel
results = submit_batch([ir_a, ir_b, ir_c], f_nkigym, INPUT_SPECS, CACHE_ROOT, hosts=HOSTS)  # sweep

for r in results:
    print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  "
          f"mfu={r.mfu}  min_ms={r.min_ms}")
    if not r.mfu:
        print(r.hardware_output[-500:])
```

`submit_batch` deduplicates each IR in the list against every prior batch under `CACHE_ROOT` (matched on `repr(ir)`). Cache hits reuse the original `ProfileResult` — no re-render, no re-compile, no re-profile — and the returned `ProfileResult.kernel_name` still points at the original `batch_<bid>_kernel_<kid>.py`. If every IR in the list is a cache hit, no new batch directory is created and `summary.json` is left untouched. Fresh IRs get the next monotonic `batch_id` and `kernel_id`, are rendered, shipped in one SSH round-trip to `HOSTS`, and written as:

```
<CACHE_ROOT>/
  batch_<bid>/
    kernel_<kid>/
      batch_<bid>_kernel_<kid>.py                  # rendered source
      batch_<bid>_ir_<kid>.md                      # KernelIR dump (repr)
      batch_<bid>_kernel_<kid>_log-neuron-cc.txt   # compiler log
    batch_<bid>_results.json                       # backend-shape results for the batch
  summary.json                                     # updated: tuning.batch_<bid> running-best
```

Resumable: rerunning the skill on an existing `CACHE_ROOT` reads `summary.json` + `batch_*_results.json` to pick the next ids and the running-best carries forward.

`ProfileResult` fields of interest:

- `cpu_sim.passed` — numerical match vs `f_nkigym` golden. `False` → HW skipped.
- `min_ms` — best of 100 iters (built-in `warmup=10, iters=100` from `ProfileConfig`/`BenchmarkConfig`; one submit = one valid comparison point, replicate only for deltas under ~0.1pp). `None` → compile or runtime failure.
- `mfu` — compute utilization.
- `mfu_max_achievable_estimated_percent` — roofline ceiling.
- `roofline_efficiency` — `mfu / mfu_max_achievable`.
- `mbu_estimated_percent` — HBM bandwidth utilization.
- `hardware_output` — neuronx-cc stderr.

**Reading `roofline_efficiency`** — the ceiling already bakes in memory-vs-compute bound for this arithmetic intensity, so it isolates scheduling headroom from algorithmic limits:
- Low MFU + low roofline eff → tuning headroom remains. Check `tensor_engine_active_time_percent`; gaps point at PSUM bank contention, missing DMA/TE overlap, scheduler-induced instruction gaps, unrotated buffers, or transpose scheduled as a separate gadget. Keep sweeping the 5 knobs.
- Low MFU + high roofline eff → near the memory-bound ceiling for this AI. Step up to rewrite-level changes: grow tile footprint, extend operand reuse, or apply a fusion rewrite.

Open `<CACHE_ROOT>/batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py` after each submit to spot render bugs.

`summary.json` schema:

```
{
  "function": "<name>",
  "baseline": {"source": "nkipy", "mfu": ..., "min_ms": ...},
  "target_mfu": 89.0,
  "tuning": {
    "batch_0": {"running_best_kernel": "batch_0_kernel_0.py", "mfu": 89.21},
    "batch_1": {"running_best_kernel": "batch_1_kernel_2.py", "mfu": 89.33}
  }
}
```

## Step 7 — Iterate until `TARGET_MFU`

Loop until target hit:

```python
import json
from pathlib import Path

def running_best_mfu() -> float:
    s = json.loads((Path(CACHE_ROOT) / "summary.json").read_text())
    tuning = s.get("tuning", {})
    if not tuning:
        return 0.0
    last = tuning[next(reversed(tuning))]
    return float(last.get("mfu") or 0.0)

while running_best_mfu() < TARGET_MFU:
    irs = propose_next_batch()           # your search strategy below
    submit_batch(irs, f_nkigym, INPUT_SPECS, CACHE_ROOT, hosts=HOSTS)
    print(f"running best: {running_best_mfu():.3f}%  target: {TARGET_MFU}%")
```

`propose_next_batch` does three things per iteration: read feedback → form a hypothesis → pick a mutation mode.

**1. Read feedback.** Load `<CACHE_ROOT>/batch_<latest>/batch_<latest>_results.json`, rank entries by `mfu` (descending, `None` last), and classify the bottleneck. If *every* entry has `min_ms is None`, the batch compiled/ran nothing — address failures before any MFU tuning: read `hardware_output` of the most common failure mode and fix the offending knob. Otherwise look at the top-3 successful kernels:

- Low MFU + low `roofline_efficiency` → TE scheduling gaps (PSUM bank contention, missing rotation, transpose scheduled as a separate gadget, `neuronx_cc_args` scheduler-off not set). Implicates `num_buffers`, `emission_depth`, or `dim_order`.
- Low MFU + high `roofline_efficiency` or high `mbu_estimated_percent` → memory-bound at this AI. Implicates `ltiles_per_block`, `buffer_scopes`, or a fusion rewrite.
- Mixed success/failure in one batch → read `hardware_output` of failures; the offending knob is usually named in the stderr (OOM, `NCC_*` code, unbound rotation index). Narrow the next sweep to avoid that failure region.

A batch proposed without first reading the prior batch's profiler fields is wasted compute. Whenever running-best MFU is more than ~5pp below the nkipy baseline, also diff `<CACHE_ROOT>/nkipy_baseline/nkipy_baseline.py` against your best rendered `.py` — the compiler's tile shapes, buffer placement, rotation depth, and instruction choices (e.g. inline transpose via `nc_matmul(is_transpose=True)` rather than a separate `transpose_block`) are a strong prior; translate what you see into knob settings before mutating further.

**2. Pick a mutation mode** based on whether the hypothesis names a single knob:

- **Targeted mode** — one knob clearly implicated. Sweep 5-10 settings of that knob per rewrite variant, holding others fixed near the running best. Use when the bottleneck diagnosis is specific (e.g. "unrotated `sbuf_rhs` is causing TE gap" → sweep `num_buffers["sbuf_rhs"]`).
- **Joint sample mode** — no single knob obviously implicated, or individual sweeps have all regressed from a local maximum (knobs interact; coordinate descent can't escape). Draw joint samples via `nkigym.kernel_ir.sample`.

The first batch is a special case: `submit_batch(list(variants.values()), ...)` (reusing the `variants` dict from Step 3) establishes a per-variant floor before any feedback exists. Read the rendered `.py` after every submit (`<CACHE_ROOT>/batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py`) to catch render bugs early.

**Joint sample mode reference** (reusing `variants` from Step 3):

```python
import random
from nkigym.kernel_ir.sample import sample, knob_signature

rng = random.Random()  # nondeterministic — resumed sessions advance new ground
seen: set = set()
batch: list[KernelIR] = []
while len(batch) < 20:
    base = rng.choice(list(variants.values()))
    candidate = sample(base, rng)   # joint draw — satisfies validate.is_valid by construction
    sig = knob_signature(candidate)
    if sig in seen:
        continue
    seen.add(sig)
    batch.append(candidate)
```

`sample.py` draws every knob conditioned on prior draws (PSUM cap on `ltiles`, transpose-pair matching on `scopes`, SBUF budget + open-axis on joint `emission_depth` × `num_buffers`) — every sample is valid by construction, no reject loop needed. `submit_batch` dedups against prior batches automatically, so re-proposed configurations are free.

**3. Justify each candidate before submitting.** The sampler and validator already cover validity; the question here is *performance* relevance — does this candidate target the bottleneck from step 1? If running-best is far behind the nkipy baseline, does it borrow tile shapes or rotation depth from `nkipy_baseline.py`? Bias selection toward candidates that address a named bottleneck; drop the rest.

**Stopping condition**: `running_best_mfu() >= TARGET_MFU`. A plateau in one batch means draw more joint samples or flip modes — not give up.
