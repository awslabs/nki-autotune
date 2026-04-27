---
name: profile-nki
description: Run and profile hand-written NKI kernels on Trainium until one passes accuracy and hits TARGET_MFU.
---

Iterate on raw `@nki.jit` kernels the agent writes from scratch, profiling each on Trainium, until one kernel passes accuracy against `f_numpy` AND reaches `mfu >= TARGET_MFU`. This skill covers only how to run and profile — writing NKI is the agent's job (reference: `/home/ubuntu/venvs/kernel-env/lib/python3.12/site-packages/nki/`).

## The loop

1. **Propose** one or more new kernel `.py` files for a new batch.
2. **Profile** the batch with one `remote_profile` call.
3. **Feedback** — inspect each `ProfileResult`, write one running-best entry to `summary.json`, check the stop criterion.
4. **Iterate** — use the feedback to propose the NEXT batch. Back to step 1.

Stop when `summary.json`'s latest `tuning.batch_<bid>.mfu >= TARGET_MFU`.

## Rules

- **Never delete a failed kernel.** Compile error / OOM / accuracy miss / low MFU — the file stays on disk as-is. The failure is the record.
- **Never mutate a profiled kernel.** Once `remote_profile` has been called on a file, do NOT edit / overwrite / delete / rename / move it. Fixes go into a fresh file in the NEXT batch.
- **A batch is exactly one `remote_profile` call.** Submit → feedback → submit = two batches, different `bid`s.
- **`(bid, kid)` never reuses.** Fresh monotonic `bid` per batch; fresh monotonic `kid` per kernel (never resets across batches).
- **No peeking at other NKI kernels.** Agent may only read files it wrote under `<CACHE_ROOT>/`. No reading from `kernel_library/`, `examples/`, `nkilib/`, `nkigym/`, prior caches, or any other kernel source.
- **No files outside `<CACHE_ROOT>/`.** All kernel sources, driver scripts, and artifacts live under `<CACHE_ROOT>/`.

## Invocation

Invoked as `profile-nki <workload.py>`. `workload.py` MUST define:

| name | type | purpose |
|---|---|---|
| `f_numpy` | `Callable[..., np.ndarray]` | golden reference |
| `INPUT_SPECS` | `dict[str, tuple[tuple[int, ...], str]]` | `{param_name: (shape, dtype_str)}`, key order matches `f_numpy` params |
| `CACHE_ROOT` | `str` \| `Path` | everything lives here |
| `HOSTS` | `list[str]` | Trainium SSH hostnames |
| `TARGET_MFU` | `float` | stop threshold |

Abort with the list of missing names if any are absent.

Run `source ~/venvs/kernel-env/bin/activate` first.

## Cache layout

```
<CACHE_ROOT>/
  summary.json
  batch_<bid>/
    batch_<bid>_kernel_<kid>.py
    batch_<bid>_kernel_<kid>/log-neuron-cc.txt
    results.json
```

`summary.json`:

```json
{
  "function": "<f_numpy.__name__>",
  "baseline": null,
  "target_mfu": 89.0,
  "tuning": {
    "batch_0": {"running_best_kernel": null, "mfu": null},
    "batch_1": {"running_best_kernel": "batch_1_kernel_2.py", "mfu": 4.93}
  }
}
```

`tuning.batch_<bid>` is the running-best kernel across **all batches up to `bid`** with `cpu_sim.passed is True` and `mfu is not None`. Both fields `null` if none qualify. Monotonically non-decreasing.

## Prelude (run once)

```python
"""Step 0 — load workload."""
import importlib.util, sys, json, inspect, numpy as np
from pathlib import Path
from autotune.runner.types import resolve_dtype, KernelJob
from autotune.runner.api import remote_profile

spec = importlib.util.spec_from_file_location("workload", "<workload.py>")
mod  = importlib.util.module_from_spec(spec); sys.modules["workload"] = mod
spec.loader.exec_module(mod)

required = ["f_numpy", "INPUT_SPECS", "CACHE_ROOT", "HOSTS", "TARGET_MFU"]
missing  = [n for n in required if not hasattr(mod, n)]
if missing: raise RuntimeError(f"workload.py missing: {missing}")

f_numpy     = mod.f_numpy
INPUT_SPECS = mod.INPUT_SPECS
CACHE_ROOT  = Path(str(mod.CACHE_ROOT)); CACHE_ROOT.mkdir(parents=True, exist_ok=True)
HOSTS       = list(mod.HOSTS)
TARGET_MFU  = float(mod.TARGET_MFU)

"""Step 1 — derive output_shape and mac_count."""
dummy = {n: np.zeros(s, dtype=resolve_dtype(d)) for n, (s, d) in INPUT_SPECS.items()}
output_shape = tuple(f_numpy(**dummy).shape)
mac_count    = <derive from f_numpy by hand: sum m*k*n per A@B; 0 for elementwise/reduction/transpose>

"""Step 2 — init summary.json."""
summary_path = CACHE_ROOT / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
    "function": f_numpy.__name__, "baseline": None, "target_mfu": TARGET_MFU, "tuning": {},
}
summary_path.write_text(json.dumps(summary, indent=2))

golden_src = "import numpy as np\n\n" + inspect.getsource(f_numpy)
```

## Iteration loop

### 1. Propose

```python
def next_kid(root: Path) -> int:
    seen = [int(p.stem.rsplit("_", 1)[-1]) for p in root.rglob("batch_*_kernel_*.py")]
    return max(seen) + 1 if seen else 0

bid       = len(summary["tuning"])
start_kid = next_kid(CACHE_ROOT)
kids      = list(range(start_kid, start_kid + <num_kernels in this batch>))
batch_dir = CACHE_ROOT / f"batch_{bid}"; batch_dir.mkdir(parents=True, exist_ok=True)
```

For each `kid` in `kids`, use the `Write` tool to create `batch_<bid>/batch_<bid>_kernel_<kid>.py`. Each file:

- Single `@nki.jit` function; positional params match `INPUT_SPECS` order.
- Imports only from `nki`, `nki.isa`, `nki.language`.
- Output allocated in HBM (`buffer=nl.shared_hbm`) and returned.

Record each function name as `func_name` for Step 2.

### 2. Profile

```python
kernels = {
    f"batch_{bid}_kernel_{kid}.py": KernelJob(
        source=(batch_dir / f"batch_{bid}_kernel_{kid}.py").read_text(),
        func_name=<name>,
        output_shape=output_shape,
        input_specs=INPUT_SPECS,
        nkigym_source=golden_src,
        nkigym_func_name=f_numpy.__name__,
        mac_count=mac_count,
        atol=1e-2, rtol=1e-2,
        neuronx_cc_args=(),
    )
    for kid in kids
}
output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(batch_dir))
```

After this call, every `.py` in this batch is frozen.

### 3. Feedback

Per-kernel `ProfileResult` fields:

- `cpu_sim["passed"]` — accuracy; `False` → HW skipped, see `cpu_sim["error"]`.
- `mfu` — `None` on compile/runtime failure; see `hardware_output` + `log-neuron-cc.txt`.
- `min_ms`, `mfu_max_achievable_estimated_percent`, `roofline_efficiency`, `hardware_output`.

Write one running-best entry for this batch:

```python
best_name, best_mfu = None, None
for rpath in sorted(CACHE_ROOT.glob("batch_*/results.json")):
    for r in json.loads(rpath.read_text()):
        if not r.get("cpu_sim", {}).get("passed"): continue
        mfu = r.get("mfu")
        if mfu is None: continue
        if best_mfu is None or mfu > best_mfu:
            best_name, best_mfu = r["kernel_name"], mfu

summary["tuning"][f"batch_{bid}"] = {"running_best_kernel": best_name, "mfu": best_mfu}
summary_path.write_text(json.dumps(summary, indent=2))

if best_mfu is not None and best_mfu >= TARGET_MFU:
    print("DONE"); exit()
```

### 4. Iterate

Use the feedback to design the next batch. Back to step 1. Never touch prior `batch_<bid>/` directories.
