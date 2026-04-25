---
name: tune-kernel
description: Hand-tune an nkigym KernelIR for any nkigym math function to beat the nkipy compiler baseline.
---

Tunes one nkigym kernel by editing `KernelIR` knobs and measuring on Trainium. Generic over any nkigym math function.

**Goal: reach `TARGET_MFU` (set by the user in `workload.py`).** The skill keeps iterating — new rewrite variants, new knob sweeps, new batches — until the running-best MFU (see `summary.json → tuning.batch_<latest>.mfu`) is ≥ `TARGET_MFU`. That MFU threshold is the one and only stop criterion.

## Reference material

- `nkigym/src/nkigym/kernel_ir/ir.py` — dataclasses (`KernelIR`, `Op`, `PhysicalBuffer`, `BufferScope`, `NumBuffers`)
- `nkigym/src/nkigym/kernel_ir/build.py` — `build_ir` parser
- `nkigym/src/nkigym/kernel_ir/validate.py` — `is_valid(ir)` gate
- `nkigym/src/nkigym/kernel_ir/rewrites/` — available graph rewrites
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

Read `nkipy_baseline.py` to see the compiler's tiling / buffer layout / instruction choice.

## Step 2 — Build the canonical KernelIR

```python
from nkigym.kernel_ir import build_ir
ir = build_ir(f_nkigym, INPUT_SPECS)
print(repr(ir))
```

Read `repr(ir)` for: `dimensions` (dim ids + roles), `physical_buffers` (SBUF buffer names + tile shape), `ops` (op graph). Your knob dicts key off the buffer names printed here.

## Step 3 — Enumerate graph-rewrite variants

```bash
ls nkigym/src/nkigym/kernel_ir/rewrites/
```

Rewrites restructure the op graph and change which buffers exist. Every subset of applicable rewrites gives a distinct starting point for knob tuning — tune over all of them.

```python
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.rewrites import LoadTranspose

base_ir = build_ir(f_nkigym, INPUT_SPECS)
variants = {
    "base":  base_ir,
    "fused": LoadTranspose()(base_ir),
    # add one entry per additional rewrite / combination as the library grows
}
for name, ir in variants.items():
    print(name)
    print(repr(ir))   # physical_buffers + ops differ per variant → knob dicts key off them
```

A rewrite leaves the IR unchanged if its pattern isn't present, so always including the rewrite is safe; keeping both `base` and `<rewrite>-applied` versions is what matters. See `examples/matmul_lhs_rhs.py` for the pattern.

When sweeping knobs (step 4), produce one set of candidate IRs per variant — the best knob setting under one rewrite configuration is not necessarily the best under another.

## Step 4 — Set the 5 knobs

`KernelIR` has exactly five tunable fields:

- **`dim_order: list[str]`** — permutation of dim ids. Controls loop nest. Store fires at first `ACCUMULATION` position.
- **`ltiles_per_block: dict[str, int]`** — tiles per block per dim. `num_blocks[d] = num_ltile[d] / ltiles[d]`. Watch the 8-bank PSUM budget.
- **`buffer_scopes: dict[str, BufferScope]`** — `INNER` (0 full dims) / `MIDDLE` (1 full, outermost-in-order per-block) / `OUTER` (all full).
- **`num_buffers: dict[str, NumBuffers]`** — `NumBuffers(num_p_buffers, num_f_buffers)`. `None` = no rotation, `N` = `N` ping-pong copies.
- **`emission_depth: dict[str, int]`** — `0` = kernel top, `N` = inside `N` loops.

```python
from nkigym.kernel_ir import BufferScope, NumBuffers
ir.dim_order = ["d0", "d2", "d1"]
ir.ltiles_per_block = {"d0": 8, "d1": 8, "d2": 1}
ir.buffer_scopes = {"sbuf_output": BufferScope.MIDDLE}
ir.num_buffers = {"sbuf_lhs_T": NumBuffers(4, 4), "sbuf_rhs": NumBuffers(8, None)}
ir.emission_depth = {"sbuf_lhs_T": 0, "sbuf_rhs": 2, "sbuf_output": 1}
```

Only set entries you want to override. Anything missing uses `build_ir` defaults.

## Step 5 — Validate

```python
from nkigym.kernel_ir.validate import is_valid
if not is_valid(ir):
    continue  # skip this IR
```

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

`submit_batch` assigns the next monotonic `batch_id` and `kernel_id`, renders each IR, ships the batch in one SSH round-trip to `HOSTS`, and writes:

```
<CACHE_ROOT>/
  batch_<bid>/
    kernel_<kid>/
      batch_<bid>_kernel_<kid>.py                  # rendered source
      batch_<bid>_ir_<kid>.py                      # KernelIR dump (repr)
      batch_<bid>_kernel_<kid>_log-neuron-cc.txt   # compiler log
    batch_<bid>_results.json                       # backend-shape results for the batch
  summary.json                                     # updated: tuning.batch_<bid> running-best
```

Resumable: rerunning the skill on an existing `CACHE_ROOT` reads `summary.json` + `batch_*_results.json` to pick the next ids and the running-best carries forward.

`ProfileResult` fields of interest:

- `cpu_sim.passed` — numerical match vs `f_nkigym` golden. `False` → HW skipped.
- `min_ms` — best of 100 iters. `None` → compile or runtime failure.
- `mfu` — compute utilization.
- `mfu_max_achievable_estimated_percent` — roofline ceiling.
- `roofline_efficiency` — `mfu / mfu_max_achievable`.
- `mbu_estimated_percent` — HBM bandwidth utilization.
- `hardware_output` — neuronx-cc stderr.

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

## Step 7 — Measurement

Built-in `warmup=10, iters=100` (`ProfileConfig`, `BenchmarkConfig`). One submit = one valid comparison point. Replicate only for deltas under ~0.1pp.

## Step 8 — Iterate until `TARGET_MFU`

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

Search strategy for `propose_next_batch`:

1. First batch: one plain `build_ir(...)` IR per rewrite variant from Step 3. Establishes a per-variant floor.
2. Each subsequent batch sweeps one knob (5-10 variants per axis) per rewrite variant — the best knob setting under one rewrite is not the best under another.
3. Read the rendered `.py` every time (`<CACHE_ROOT>/batch_<bid>/kernel_<kid>/batch_<bid>_kernel_<kid>.py`) to catch render bugs early.
4. Coarse-sweep `ltiles` / `dim_order` / `buffer_scopes` first, then refine `num_buffers` / `emission_depth`.

**Stopping condition**: `running_best_mfu() >= TARGET_MFU`. Keep sweeping new rewrite variants, ltile combos, emission depths, and rotation depths until the target is hit.
