---
name: tune-kernel
description: Hand-tune an nkigym KernelIR for any nkigym math function to beat the nkipy compiler baseline.
---

Tunes one nkigym kernel. Two mutation surfaces are both first-class: the 3 `KernelIR` knobs, and the rendered NKI source itself (direct hand-edits). Generic over any nkigym math function.

**Goal: reach `TARGET_MFU` (set by the user in `workload.py`).** The skill keeps iterating — new rewrite variants, new knob sweeps, new batches, new hand-edits — until the running-best MFU (see `summary.json → tuning.<latest>.mfu`) is ≥ `TARGET_MFU`. That MFU threshold is the one and only stop criterion. Pick whichever surface the profiler feedback implicates; knob-tuning and source-editing share the same cache, summary, and stop criterion.

## Reference material

- `/home/ubuntu/nki-autotune/nkigym/src/nkigym/` — IR, codegen, ops, rewrites, search (`KernelIR`, `build_ir`, `validity_report`, `sample`/`knob_signature`, `enumerate_rewrite_combinations`, `render_ir`, `inline_gadgets`, `func_source_with_imports`, `compute_mac_count`)
- `/home/ubuntu/nki-autotune/autotune/src/autotune/runner/` — `dump_baseline`, `submit_batch`, `remote_profile`, `KernelJob`
- `/home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.md` — canonical walkthrough of the new IR semantics and code-gen contract (start here when unsure)
- `/home/ubuntu/venvs/kernel-env/lib/python3.12/site-packages/nki/` — NKI Python API (`isa/`, `language/`, `simulator.py`)
- `/home/ubuntu/shared_workplace/KaenaCompiler/neuronxcc` — compiler source
- `/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary` — hand-written reference kernels

## Two mutation surfaces

You pick freely each iteration:

1. **IR knobs (`submit_batch`)** — edit `KernelIR.loop_order / ltiles_per_block / buffer_scopes`, optionally apply an `IRRewrite`, render through `render_ir`. Use this when the bottleneck diagnosis maps cleanly onto one of the 3 knobs (tile footprint → `ltiles_per_block`; loop nest structure / drain rhythm → `loop_order`; per-dim buffer extent → `buffer_scopes`). Cheap, structured, and `validity_report` catches most mistakes before a round-trip.
2. **Direct source edits (`remote_profile`)** — take the rendered `.py` (from the mirror step, or from the running-best kernel in the cache) and hand-edit it. Use this when the IR/renderer can't express what you need: a custom instruction schedule, bank-specific `address=` annotations, reordered DMA issues, inlined pragmas, or anything the compiler log points at that isn't a 3-knob shift. Start from `render_ir(ir)` + `inline_gadgets(...)` as scaffolding when that helps; start from scratch when it doesn't.

These are peers — neither is the fallback. Read the profiler fields (`roofline_efficiency`, `mbu_estimated_percent`, `tensor_engine_active_time_percent` from the NTFF summary, `hardware_output` stderr) and **choose the surface that targets the bottleneck**. Bouncing between surfaces within one tuning session is expected.

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

Read `nkipy_baseline.py` to see the compiler's tiling / buffer layout / instruction choice. **Start by mirroring it.** Translate what the compiler picked into your first IR configuration — tile shapes → `ltiles_per_block`, block/tile loop interleaving → `loop_order`, per-axis extent of each buffer (one tile vs one block vs full) → `buffer_scopes`. The compiler's output is already a strong local optimum; starting there gives an informed floor to tune *from*, rather than climbing from a random default. Submit this mirror as the first kernel in batch 0 alongside the rewrite-variant defaults.

## Step 2 — Build the canonical KernelIR

```python
from nkigym.kernel_ir import build_ir
ir = build_ir(f_nkigym, INPUT_SPECS)
print(repr(ir))
```

Read `repr(ir)` for: `dimensions` (dim ids + size/tile), `physical_buffers` (SBUF and PSUM buffer names, per-op-output including implicit PSUM allocations, plus tile shapes), `operators` (op graph with `dim_role` per op). Your knob dicts key off the buffer names printed here.

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

A rewrite leaves the IR unchanged if its pattern isn't present, so always including the rewrite is safe; keeping both `base` and `<rewrite>-applied` versions is what matters.

When sweeping knobs (step 4), produce one set of candidate IRs per variant — the best knob setting under one rewrite configuration is not necessarily the best under another.

## Step 4 — Set the 3 knobs

`KernelIR` has exactly three tunable fields. Everything else (buffer emission depth, op fire depth, PSUM drain placement) is **derived** by mechanical lowering — the renderer does not invent schedules, and the knobs cannot override derived values. Before writing values, reason about what each knob physically does to the emitted kernel; knobs interact.

- **`loop_order: list[str]`** — permutation of `{d}.block` and `{d}.tile` entries, one pair per dim. For `N` dims → `2N` entries, with the invariant `{d}.block` precedes `{d}.tile` → `(2N)! / 2^N` valid orderings (90 for 3 dims). Controls which loops are open at each depth, which controls both *when* each op can fire (intrinsic tile-level requirements demand certain `.tile` loops open) and *where* each buffer is allocated (a buffer's emission depth is the tightest depth where every access is covered). The `{d}.block` / `{d}.tile` split is first-class — separating them is how you reorder tile-level work independently of block-level blocking.
- **`ltiles_per_block: dict[str, int]`** — how many logical tiles sit inside one block iteration per dim. Divisors of `num_ltile[d]`. `{d}.block` trip = `num_ltile[d] / ltiles_per_block[d]`; `{d}.tile` trip = `ltiles_per_block[d]`. Raises the per-block buffer footprint and shrinks the outer block-loop trip count.
- **`buffer_scopes: dict[str, dict[str, Scope]]`** — per-dim extent map for every dim the buffer carries. Three choices per dim, independent across dims:
  - `PER_TILE` — one tile along that axis. Alloc must live inside `{d}.tile`.
  - `PER_BLOCK` — one block (`ltiles_per_block[d]` tiles) along that axis. Alloc must live inside `{d}.block`.
  - `FULL` — entire `num_ltile[d]` tiles along that axis. Alloc must live outside `{d}.block`.

  Load-destination buffers and PSUM accumulators are tunable; SBUF accumulators downstream of a reducing op are partially derived — reducing dims of their producers are pinned to `FULL` by codegen and omitted from the IR knob surface. Codegen lowers each buffer to a single `nl.ndarray` (no multi-buffer banks in the IR — the Neuron compiler handles rotation / double-buffering on its own).

**The knobs are not independent.** Changing `loop_order` shifts which depths each `{d}.block` / `{d}.tile` sits at, which changes every buffer's emission depth and every op's fire depth; a `buffer_scopes` pick that was feasible under one order can cross bounds under another. Treat each (loop_order, ltiles_per_block, buffer_scopes) as one joint configuration.

Assign directly to `ir.loop_order`, `ir.ltiles_per_block`, `ir.buffer_scopes`. Only set entries you want to override; anything missing uses `build_ir` defaults. Buffer-keyed dicts key off the names printed in `repr(ir)`.

**Two options on PSUM's reducing-dim scope** — this is the highest-leverage knob:

- **Option A (`PSUM.K = FULL`)** — PSUM holds the entire K reduction. Drain = `nisa.dma_copy(dst=sbuf, src=psum)` (dtype-narrowing copy, no accumulation). Requires PSUM emission outside every K loop; only feasible if `loop_order` keeps both `K.block` and `K.tile` inside PSUM's lifetime.
- **Option B (`PSUM.K = PER_BLOCK`)** — PSUM holds one K.block's partial sum. Drain = `nisa.tensor_tensor(dst=sbuf, data1=sbuf, data2=psum, op=nl.add)` — folds into an SBUF K-FULL accumulator. PSUM alloc lives inside K.block but outside K.tile.

`PSUM.K = PER_TILE` is invalid: zero-init per K.tile would wipe the partial sum.

## Step 5 — Validate

Prefer `validity_report(ir)` over a boolean check: a `False` silently discards IRs that are one knob away from working — which is most of the interesting neighborhood of any good kernel. The report returns one `ValidityFailure` per failed check with a concrete fix hint:

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

**Invalid-IR detection is mechanical: every derivation step is a constraint resolution.**

Each buffer's emission depth is a per-dim `[lower_bound, upper_bound]` interval computed from its scope:
- `PER_TILE` on dim d → ≥ depth of `{d}.tile`
- `PER_BLOCK` on dim d → ≥ depth of `{d}.block`, ≤ depth just inside `{d}.block`
- `FULL` on dim d → ≤ depth just outside `{d}.block`

Each op's fire depth is `max` of two sources:
- **Operand availability** — `max(operand.emission_depth for operand in operands)`
- **Op-intrinsic tile requirement** — the op must fire inside the `{d}.tile` loop of every dim its ISA instruction requires. (`nc_matmul` needs all of K.tile, M.tile, N.tile; `dma_copy` needs the P-axis `.tile`; the free axis is unconstrained.)

Additional constraints checked by the validator:
- **Accumulator-close** for store ops: must fire outside every reducing loop of the producer.
- **PSUM reducing-dim scope**: `PER_TILE` rejected; `FULL` requires PSUM placement outside every K-loop; `PER_BLOCK` requires placement between K.block and K.tile.
- **Dtype compatibility** across producer/consumer edges (e.g. `tensor_scalar.operand0/1` must be fp32).
- **Reducing-dim coverage** — any downstream accumulator must carry every reducing dim of its producer (codegen pins those to `FULL` implicitly).

Any contradiction (`lower > upper` on a depth interval, dtype mismatch, missing reducing-dim coverage, operand not in scope at the op's computed fire depth, etc.) rejects the IR at derivation time. No ad-hoc fixes, no silent rewrites. **Nudging one knob based on the report's `fix_hint` is almost always enough to resolve a single contradiction** — discard only when multiple independent contradictions implicate the same knob in opposite directions.

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
- Low MFU + low roofline eff → tuning headroom remains. Check `tensor_engine_active_time_percent`; gaps point at PSUM bank contention, missing DMA/TE overlap, scheduler-induced instruction gaps, or drain placement that stalls the compute pipeline. Keep sweeping the 3 knobs.
- Low MFU + high roofline eff → near the memory-bound ceiling for this AI. Step up to rewrite-level changes: grow tile footprint (`ltiles_per_block`), extend operand reuse (`buffer_scopes` PER_BLOCK → FULL on a parallel dim), or apply a fusion rewrite.

Open `<CACHE_ROOT>/batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py` after each submit to spot render bugs — and to use as the starting point for a direct source edit in the next iteration (step 6b).

## Step 6b — Submit a hand-edited source

When the bottleneck calls for something the 3 knobs can't reach, edit NKI source directly and ship it through `remote_profile`. The layout under `CACHE_ROOT` mirrors `submit_batch`'s so both paths coexist cleanly.

Starting points:
- **Scaffold from an IR**: `source = inline_gadgets(render_ir(ir))` — gives you a fully-inlined, self-contained kernel to edit.
- **Scaffold from the running best**: glob for the rendered `.py` of the highest-`mfu` entry in `summary.json → tuning` — batch kernels sit at `batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py`, hand-edits at `<edit_id>/kernel/kernel.py`. Good when iterating on a kernel that's already close.
- **Scaffold from the compiler**: read `<CACHE_ROOT>/nkipy_baseline/nkipy_baseline.py` when the compiler's low-level schedule is the thing you want to replicate or tweak.
- **Scaffold from nkilib**: hand-written references in `/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary` are often the right model for custom allocation or instruction layout.

Edit the source — change tile shapes, rewrite per-op code, insert `address=` annotations, reorder DMA, swap `nisa.*` instructions, anything the profiler points at.

Then build a `KernelJob` and submit via `remote_profile`:

```python
from pathlib import Path
from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.search.api import func_source_with_imports
from nkigym.search.mac import compute_mac_count

edited_source: str = ...           # your hand-edited NKI source
func_name: str = ...               # the @nki.jit function name inside edited_source
output_shape: tuple = ...          # HBM output shape (trace once or read from an IR)

job = KernelJob(
    source=edited_source,
    func_name=func_name,
    output_shape=output_shape,
    input_specs=INPUT_SPECS,
    nkigym_source=func_source_with_imports(f_nkigym),
    nkigym_func_name=f_nkigym.__name__,
    mac_count=compute_mac_count(f_nkigym, INPUT_SPECS),
    atol=1e-2,
    rtol=1e-2,
    neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
)

"""Pick a unique label. Keep it under CACHE_ROOT so it sits next to
batch_*/ entries; use an edit_<N>/ prefix to avoid colliding with
submit_batch's batch_<bid>/ ids."""
edit_id = "edit_0"
edit_dir = Path(CACHE_ROOT) / edit_id
edit_dir.mkdir(parents=True, exist_ok=True)
(edit_dir / "NOTES.md").write_text("starting point + hypothesis + what changed\n")

"""remote_profile writes <cache_dir>/<stem>/<stem>.py; using
kernel_name='kernel.py' with cache_dir=<CACHE_ROOT>/edit_<N> yields
<CACHE_ROOT>/edit_<N>/kernel/kernel.py — the layout matches batch kernels."""
kernel_name = "kernel.py"
output = remote_profile(
    kernels={kernel_name: job},
    hosts=HOSTS,
    cache_dir=str(edit_dir),
)
r = output.results[0]
print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')} mfu={r.mfu} min_ms={r.min_ms}")
if not r.mfu:
    print(r.hardware_output[-500:])

"""Path used in summary.json → relative to CACHE_ROOT."""
rendered_path = f"{edit_id}/kernel/kernel.py"
```

`remote_profile` writes `<edit_dir>/kernel/kernel.py`, `<edit_dir>/kernel/log-neuron-cc.txt`, and `<edit_dir>/results.json` — the same per-kernel shape `submit_batch` produces (minus the `ir.md` sibling, which doesn't apply to hand-edits).

After a hand-edit lands, update `summary.json → tuning` to keep the running-best visible to step 7's stop check:

```python
import json
from pathlib import Path

summary_path = Path(CACHE_ROOT) / "summary.json"
summary = json.loads(summary_path.read_text())
summary.setdefault("tuning", {})
"""running_best_kernel points at the file that achieved this MFU."""
prior_best = None
prior_mfu = None
for entry in summary["tuning"].values():
    m = entry.get("mfu")
    if m is not None and (prior_mfu is None or m > prior_mfu):
        prior_mfu = m
        prior_best = entry.get("running_best_kernel")
best_name = rendered_path
best_mfu = r.mfu
if prior_mfu is not None and (best_mfu is None or prior_mfu > best_mfu):
    best_name, best_mfu = prior_best, prior_mfu
summary["tuning"][edit_id] = {"running_best_kernel": best_name, "mfu": best_mfu}
summary_path.write_text(json.dumps(summary, indent=2))
```

Hand-edits have no IR to dedup against, so it's on you to not re-submit the same source. Keep a per-edit `NOTES.md` with "starting point / hypothesis / what changed" so future iterations can tell at a glance whether a previous edit already tried this mutation.

`summary.json` schema:

```
{
  "function": "<name>",
  "baseline": {"source": "nkipy", "mfu": ..., "min_ms": ...},
  "target_mfu": 89.0,
  "tuning": {
    "batch_0": {"running_best_kernel": "batch_0_kernel_0.py",  "mfu": 89.21},
    "batch_1": {"running_best_kernel": "batch_1_kernel_2.py",  "mfu": 89.33},
    "edit_0":  {"running_best_kernel": "edit_0/kernel/kernel.py", "mfu": 90.12}
  }
}
```

Keys are heterogeneous: `batch_<bid>` entries are written by `submit_batch`, `edit_<N>` entries are written by step 6b's hand-edit path. Step 7's `running_best_mfu()` takes the max across all of them regardless of kind.

## Step 7 — Iterate until `TARGET_MFU`

Loop until target hit:

```python
import json
from pathlib import Path

def running_best_mfu() -> float:
    """Max mfu across every tuning entry — works for both submit_batch
    batches and hand-edit entries, regardless of insertion order."""
    s = json.loads((Path(CACHE_ROOT) / "summary.json").read_text())
    mfus = [e.get("mfu") for e in s.get("tuning", {}).values() if e.get("mfu") is not None]
    return float(max(mfus)) if mfus else 0.0

while running_best_mfu() < TARGET_MFU:
    """Pick ONE mode per iteration based on latest feedback (see below):
       - submit_batch(irs, ...) for knob sweeps / joint samples
       - remote_profile(...) for a direct source edit (step 6b)"""
    propose_and_submit()
    print(f"running best: {running_best_mfu():.3f}%  target: {TARGET_MFU}%")
```

Each iteration does three things: read feedback → form a hypothesis → pick a mutation mode (one of three: targeted knob, joint knob sample, or direct source edit).

**1. Read feedback.** Load `<CACHE_ROOT>/batch_<latest>/batch_<latest>_results.json` (or the hand-edit's `<CACHE_ROOT>/<edit_id>/results.json`), rank entries by `mfu` (descending, `None` last), and classify the bottleneck. If *every* entry has `min_ms is None`, the batch compiled/ran nothing — address failures before any MFU tuning: read `hardware_output` of the most common failure mode and fix the offending knob or source line. Otherwise look at the top-3 successful kernels:

- Low MFU + low `roofline_efficiency` → TE scheduling gaps (PSUM bank contention, drain-induced stalls, transpose scheduled as a separate gadget, `neuronx_cc_args` scheduler-off not set). Implicates `loop_order` (reordering can move `{d}.tile` loops to change drain rhythm and expose more parallelism), `buffer_scopes` (PER_BLOCK → FULL on a PARALLEL dim to cut reload traffic), **or** a source-level issue (instruction order, `address=` placement) that IR knobs don't reach.
- Low MFU + high `roofline_efficiency` or high `mbu_estimated_percent` → memory-bound at this AI. Implicates `ltiles_per_block`, `buffer_scopes` widening, a fusion rewrite, **or** a source-level change to DMA overlap / prefetch depth.
- Mixed success/failure in one batch → read `hardware_output` of failures; the offending knob or instruction is usually named in the stderr (OOM, `NCC_*` code). Narrow the next sweep or source edit to avoid that failure region.

A batch proposed without first reading the prior batch's profiler fields is wasted compute. Whenever running-best MFU is more than ~5pp below the nkipy baseline, also diff `<CACHE_ROOT>/nkipy_baseline/nkipy_baseline.py` against your best rendered `.py` — the compiler's tile shapes, buffer placement, drain rhythm, and instruction choices (e.g. inline transpose via `nc_matmul(is_transpose=True)` rather than a separate `nc_transpose` op) are a strong prior. Translate what you see either into knob settings **or** directly into a hand-edited source variant, whichever maps more cleanly.

**2. Pick a mutation mode.** Three modes — choose by what the feedback implicates:

- **Targeted knob mode** (`submit_batch`) — one knob clearly implicated. Sweep 5-10 settings of that knob per rewrite variant, holding others fixed near the running best. Use when the bottleneck diagnosis is specific and maps onto a knob (e.g. "K.tile innermost is crowding PSUM drain" → sweep `loop_order` permutations that move `K.tile` to mid-depth; "reload traffic on sbuf_lhs_T dominates" → sweep `buffer_scopes[sbuf_lhs_T]` with the M-dim stepped through PER_TILE / PER_BLOCK / FULL).
- **Joint knob sample mode** (`submit_batch`) — no single knob obviously implicated, or individual knob sweeps have all regressed from a local maximum (knobs interact; coordinate descent can't escape). Draw joint samples via `nkigym.kernel_ir.sample`.
- **Direct source edit mode** (`remote_profile`, step 6b) — the bottleneck points at something IR knobs can't change. Examples: compiler log shows `pf_transpose_insts_for_io=0` when the baseline has many; PSUM address rotation is wrong after the allocator; the baseline NKI inlines an instruction pattern that the renderer doesn't emit. Edit `<CACHE_ROOT>/<starting_point>.py` directly — IR scaffolding is optional, not required.

The first batch is a special case: `submit_batch(list(variants.values()), ...)` (reusing the `variants` dict from Step 3) establishes a per-variant floor before any feedback exists. Read the rendered `.py` after every submit (`<CACHE_ROOT>/batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py` or `<CACHE_ROOT>/<edit_id>/<stem>/<stem>.py`) to catch bugs early.

**Joint sample mode reference** (reusing `variants` from Step 3):

```python
import random
from nkigym.kernel_ir.sample import sample, knob_signature

rng = random.Random()  # nondeterministic — resumed sessions advance new ground
seen: set = set()
batch: list[KernelIR] = []
while len(batch) < 20:
    base = rng.choice(list(variants.values()))
    candidate = sample(base, rng)   # joint draw — satisfies validity_report by construction
    sig = knob_signature(candidate)
    if sig in seen:
        continue
    seen.add(sig)
    batch.append(candidate)
```

`sample.py` draws every knob conditioned on prior draws (`loop_order` constraint `{d}.block` before `{d}.tile`, `ltiles_per_block[d]` from divisors of `num_ltile[d]`, `buffer_scopes[B]` per-dim picks respecting PSUM accumulator rules, emission-depth feasibility after the joint draw) — every sample is valid by construction, no reject loop needed. `submit_batch` dedups against prior batches automatically, so re-proposed configurations are free.

**Direct source edit mode reference** (step 6b):

```python
"""Pick starting point — running best, compiler baseline, or a fresh
inline_gadgets(render_ir(ir)) render. Glob the cache for the highest-MFU
rendered .py if you want to edit the current leader."""
candidates = list(Path(CACHE_ROOT).rglob("*.py"))
"""Apply your hypothesis (example: switch PSUM drain from tensor_tensor
to dma_copy after changing PSUM's K scope from PER_BLOCK to FULL)."""
start_src = best_candidate.read_text()
edited = start_src.replace(
    "nisa.tensor_tensor(dst=sbuf_out[p, m, f0:f1], data1=sbuf_out[p, m, f0:f1], "
    "data2=psum_out[p, 0, f0:f1], op=nl.add)",
    "nisa.dma_copy(dst=sbuf_out[p, m, f0:f1], src=psum_out[p, 0, f0:f1])",
)

"""Ship via step-6b KernelJob + remote_profile."""
```

Keep edits small and single-purpose. A diff of one dozen lines is easier to roll back than a rewritten file. Record the hypothesis in `<edit_dir>/NOTES.md` so the next iteration can read what was tried.

**3. Justify each candidate before submitting.** The sampler and validator cover IR-knob validity; source edits have no validator at all — the compiler and CPU sim are the only gates. The performance question is the same across all three modes: *does this candidate target the bottleneck from step 1?* If running-best is far behind the nkipy baseline, does the candidate borrow tile shapes, drain rhythm, or instruction patterns from `nkipy_baseline.py`? Bias selection toward candidates that address a named bottleneck; drop the rest.

**Stopping condition**: `running_best_mfu() >= TARGET_MFU`. A plateau in one batch or edit means flip modes (knob sweep → joint sample → source edit, in any direction) — not give up.
