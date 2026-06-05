# Local profiling backend + SSH/Kaizen transport shells

**Date:** 2026-06-05
**Status:** Approved (design); pending implementation plan
**Branch:** `dev_1`

## Problem

The autotune profiling backend (`autotune/src/autotune/runner/`) is a
laptop-side *coordinator*: it runs `nkigym` locally to generate NKI source
strings, base64-bundles the `autotune` package, and pipes it plus a JSON
job payload over **SSH stdin** to one or more remote Trainium hosts, which
compile + benchmark and return JSON.

Two facts break this model for the current workflow:

1. **`nkigym` cannot run on the laptop.** Every op module hard-imports
   `nki.isa` at module load (`nkigym/ops/{matmul,load,store,tensor_scalar,
   transpose,dma_transpose}.py`, plus `nkigym/synthesis/simulate_nki.py`).
   `import nki` fails locally (`ModuleNotFoundError`); the `kernel-env`
   referenced in `.claude/rules/references.md` does not exist on the dev box.
   So the laptop literally cannot run the coordinator — kernel generation,
   compilation, and benchmarking must all happen *on the box*.

2. **Execution targets are now Kaizen desktops and SSH gym hosts**, reached
   by two different transports (Kaizen has no SSH-stdin pipe — only
   `s5cmd` sync + `kaizen desktop connect --cmd`).

The desired workflow: develop only in `nki-autotune`; run the whole driver
on a Trn2 box (Kaizen or SSH); get result artifacts back on the laptop.

## Goals

- Strip the profiling backend to **local mode** — it assumes it runs on a
  Trn2 box, in-process, no SSH / bundling / multi-host fan-out.
- Add **two command-agnostic transport shells** (SSH + Kaizen) that each
  do: (1) code sync, (2) env setup, (3) execute an arbitrary `--cmd`,
  (4) download artifacts to the laptop.
- Drive everything from a `--cache-root-dir` convention so artifacts land
  in a known place on the box and sync back to a known place on the laptop.

## Non-goals / deferred

- **Do NOT switch to `private-nki-staging`.** Stay on the public
  `nki` + `nkipy` + `spike` stack. Rationale below.
- **Do NOT port or rewrite the numpy baseline.** It stays on nkipy's
  `baremetal_jit` path, unchanged except for losing its SSH wrapper.
- No Python transport abstraction — the transports are **pure bash**.
- No auto-provisioning of Kaizen desktops (no auto `kaizen desktop start`).

### Why `private-nki-staging` is deferred

`private-nki-staging`'s `pyproject.toml` declares `name = "nki"` — it *is*
the `nki` package, built from MLIR/CMake source. Switching to it was
investigated and rejected for "minimal change now":

- **Shifted compile API.** The worker imports
  `from nki.compiler.driver import CompileOptions, compile_bir_to_neff,
  compile_to_bir`. In private-nki-staging, `driver.py` exposes only
  `compile_to_bir`; `CompileOptions` / `CompiledKernel` /
  `extract_perf_metrics` moved to `nki.compiler.ncc_driver`. The compile
  path would need a rewrite.
- **No `nkipy` / `spike`.** The runtime (`BaremetalExecutor`,
  `HLOModule`/`HLOTensor`) lives in `nkipy`, which private-nki-staging does
  not provide; its analogue is `nrtpy.SpikeModel` + `nki.profiling`. The
  benchmark path would need a rewrite.
- **No numpy→NEFF path.** nkipy's `baremetal_jit` (numpy → HLO →
  neuronx-cc → NEFF) and `lower_to_nki` have no equivalent in
  private-nki-staging. `@nki.jit` and `nki.framework.torch_xla`
  (`TorchXlaKernel`, an INTERNAL custom-call wrapper) both require an
  *already hand-written NKI kernel*; neither compiles a plain numpy/torch
  function. A genuine compiler baseline would need the heavy public
  `torch_neuronx` stack as a new dependency.
- **Not pip-installable trivially.** It is a custom-LLVM source build
  (`scripts/dev_setup.py`) or CodeArtifact wheels (`nki` + `nrtpy`), and
  still shells out to an external `neuronx-cc`.

Keeping public `nki` + `nkipy` means **zero compile/benchmark API churn**.
The private migration is a separate, larger effort.

## Architecture: three independent layers

```
LAYER 3  Driver scripts (xxx.py)            require --cache-root-dir
            │ calls
LAYER 1  Local profiling backend            in-process on a Trn2 box
            ▲ invoked by
LAYER 2  Transport shells (ssh_host.sh,     sync → setup → exec → download
         kaizen.sh) — command-agnostic
```

The transport is generic: it runs *any* `--cmd`, knowing nothing about
nkigym or profiling. "The driver" is whatever script the caller points it
at. When already on a Trn2 box, Layers 2–3 are skipped — run the driver
directly.

---

## Layer 1 — Strip the profiling backend to local mode

`autotune/src/autotune/runner/` assumes it runs **on a Trn2 box**. No SSH,
no base64 bundle, no host fan-out, no stdin/stdout JSON protocol.

### Delete (laptop-coordinator / multi-host machinery, all in `remote.py`)

`_get_worker_bundle`, `_BOOTSTRAP_SCRIPT`, `_BASELINE_BOOTSTRAP_SCRIPT`,
`_feed_stdin`, `_fail_host`, `_build_ssh_cmd`, `_build_host_assignments`,
`_launch_ssh_workers`, `_read_host_output`, `_collect_host_outputs`,
`_host_error_message`, `_parse_host_result`, `_process_host_outputs`, the
`RemoteProfiler` dataclass, `remote_profile`, and the SSH body of
`remote_numpy_baseline`.

### Keep & relocate

- **Cache-writer helpers** currently in `remote.py` —
  `write_kernel_sources`, `_write_compiler_logs`, `_kernel_sort_key`,
  `_write_per_kernel_profiles`, `_kernel_index_row`, `_write_results_json`
  — **move to `output.py`** (where `ProfileOutput` and the table formatters
  already live). This is the artifact contract the shells download; the
  on-disk layout (`<cache>/<stem>/<stem>.py`, `results.json`,
  `profile_summary.json`, optional `profile.json`/`file.neff`/`profile.ntff`)
  is **unchanged**.
- `compile.py`, `benchmark.py`, `detect.py` — unchanged (already local,
  already public-nki). They keep importing `nki` + `nkipy`.
- `worker._run_pipeline` (the compile→benchmark→collect core) becomes the
  body of the new local entry. `worker_main`, `_parse_payload`,
  `_setup_env`, and the stdout-hijack JSON wrapper are removed. The
  per-kernel `ProcessPoolExecutor` parallelism is **kept** (intra-box).

### New public entry (`api.py`)

```python
def profile(
    kernels: dict[str, KernelJob],
    cache_dir: str,
    seed: int,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
) -> ProfileOutput
```

Runs in-process on the box: per-kernel compile (parallel) → benchmark →
write the standard cache layout under `cache_dir`. No `hosts`, no
`venv_python`. `profile_numpy_baseline` (in `baseline.py`) is already local
— keep it, delete only `baseline_worker_main`.

### Type / surface cleanups

- `types.py`: drop the SSH-payload framing. `KernelJob` keeps
  `source, func_name, output_shape, input_specs, neuronx_cc_args, lnc`.
  `_DEFAULT_VENV_PYTHON` and `ensure_venv_on_path` (PATH munging for the
  bundled-worker case) are removed if unused after the strip.
- `output.py`: `ProfileOutput.hosts: list[str]` is vestigial in local mode.
  Drop it (and the `hosts` line from `__str__`/summary). `cache_dir` stays.
- `__init__.py`: rewrite the module docstring/example from
  `remote_profile(..., hosts=[...])` to the local `profile(..., cache_dir=...)`.

---

## Layer 2 — Two transport shells

New directory `transport/`. Two standalone bash scripts, **identical
interface**, each performing the same four steps. They run *any* command.

```
transport/ssh_host.sh --host <h>  --cmd "<bash>" [--no-setup]
transport/kaizen.sh   --name <n>  --cmd "<bash>" [--no-setup]
```

### Hardcoded cache roots (top of each script)

```bash
transport_cache_root_dir="$HOME/autotune_cache"             # on the remote box
local_cache_root_dir="/workplace/weittang/autotune_cache"   # on the laptop
```

`transport_cache_root_dir` **must be under `$HOME`**: on Kaizen only `$HOME`
is S3-backed and visible to `s5cmd`; `/ustore/ssd`, `/ustore/ebs` are
ephemeral and invisible to the reverse sync. Both shells use the same
`$HOME`-relative default for symmetry (SSH has no such constraint).

### The four steps

| Step | `ssh_host.sh` | `kaizen.sh` |
|---|---|---|
| **1. Code sync** | `rsync -az --delete` `nki-autotune/` → `host:~/nki-autotune/`, excluding `.git`, `__pycache__`, `*.pyc`, caches | `kaizen desktop sync --src nki-autotune/ --dst nki-autotune/` (`s5cmd`, skill's exclude set) |
| **2. Env setup** | `ssh host 'bash ~/nki-autotune/install_neuron.sh --local'` — idempotent; `--no-setup` skips | `connect --cmd 'bash ~/nki-autotune/install_neuron.sh --local'` — idempotent; `--no-setup` skips |
| **3. Execute** | `ssh host '<activate venv> && cd ~/nki-autotune && <cmd> --cache-root-dir $transport_cache_root_dir'` | `connect --cmd '<activate venv> && cd ~/nki-autotune && <cmd> --cache-root-dir $transport_cache_root_dir'` |
| **4. Download** | `rsync -az host:$transport_cache_root_dir/ $local_cache_root_dir/` | reverse `s5cmd sync` `s3://…/autotune_cache/` → `$local_cache_root_dir/`, with a short poll/retry until `results.json` appears (reverse export lags ≤60 s) |

### Command rewriting

The shell appends `--cache-root-dir "$transport_cache_root_dir"` to the
user's `--cmd`. So:

```
transport/kaizen.sh --name xxx --cmd "python xxx.py"
```

runs, on the remote Trn2 box:

```
python xxx.py --cache-root-dir <transport_cache_root_dir>
```

and then syncs `<transport_cache_root_dir>` back to `local_cache_root_dir`
(diff only — incremental sync, not a full re-copy).

### Behavior decisions

- **Idempotent, skippable setup.** `install_neuron.sh --local` already
  guards each step; first run provisions, subsequent runs are cheap.
  `--no-setup` skips step 2 once a box is known-good.
- **Reverse sync, not `cat`.** Kaizen download uses `s5cmd` (S3→local) with
  a poll/retry on `results.json`, not `connect --cmd 'cat …'` — this
  preserves the full per-kernel artifact tree, not just the index.
- **Preconditions are the caller's job.** The scripts check for
  prerequisites (`mwinit`/profiles for Kaizen; reachable host for SSH; a
  running desktop) and **fail loud** with a clear message. They do not
  auto-start a desktop.

### `install_neuron.sh` (env setup) — minimal edit

Because private-nki-staging is deferred, the script's install body is
**reused as-is** (public Neuron SDK + `nki` + `nkipy`/`spike` from source).
The only change: make `VENV_DIR` configurable (env var with the current
`/home/ubuntu/venvs/kernel-env` as default) so it is not pinned to one
absolute path across boxes.

---

## Layer 3 — Driver scripts

Driver scripts require a `--cache-root-dir` kwarg and write all artifacts
under it, replacing today's hardcoded
`CACHE_DIR = "/home/ubuntu/cache/..."`.

Three invocation modes, no code change between them:

```bash
# already on a Trn2 box — Layer 1 directly
python examples/matmul_lhsT_rhs.py --cache-root-dir /home/ubuntu/autotune_cache

# via SSH gym host
transport/ssh_host.sh --host gym-1 --cmd "python examples/matmul_lhsT_rhs.py"

# via Kaizen desktop
transport/kaizen.sh --name trn2-exp --cmd "python examples/matmul_lhsT_rhs.py"
```

### Honest note on the current examples

`examples/matmul_lhsT_rhs.py` and `examples/kernel_transforms_repro.py` are
**nkigym rollout + CPU-sim** scripts. They do *not* call the profiling
backend today (only `__init__.py`'s docstring references the old
`remote_profile`). For them, `--cache-root-dir` is purely the cache-root
convention (replacing the hardcoded `CACHE_DIR`). A script that actually
profiles on hardware would call the new Layer-1 `profile()`; writing such a
profiling driver is **out of scope** for this change — the examples are
updated only to honor the `--cache-root-dir` convention so the transports
can drive them uniformly.

---

## File-by-file change list

**Layer 1 — strip to local (`autotune/src/autotune/runner/`)**
- `remote.py` → **deleted**; cache-writer helpers relocated to `output.py`.
- `api.py` → `remote_profile`→`profile` (in-process; drop `hosts`/`venv_python`/`RemoteProfiler`).
- `worker.py` → keep `_run_pipeline` core as the local driver entry; delete `worker_main` + stdin/stdout JSON wrapper, `_parse_payload`, `_setup_env`.
- `baseline.py` → keep `profile_numpy_baseline`; delete `baseline_worker_main`.
- `types.py` → drop SSH-payload framing; remove `_DEFAULT_VENV_PYTHON`/`ensure_venv_on_path` if unused post-strip.
- `output.py` → receive the relocated cache writers; drop the `hosts` field from `ProfileOutput`.
- `__init__.py` → rewrite docstring/example to the local `profile(...)` API.
- `compile.py`, `benchmark.py`, `detect.py` → unchanged.

**Layer 2 — transports (new `transport/`)**
- `transport/ssh_host.sh`, `transport/kaizen.sh` → new 4-step scripts; hardcoded cache roots; inject `--cache-root-dir`.
- `install_neuron.sh` → single edit: configurable `VENV_DIR` (default unchanged). Install body kept (public nki+nkipy+spike).

**Layer 3 — drivers**
- `examples/matmul_lhsT_rhs.py`, `examples/kernel_transforms_repro.py` → require `--cache-root-dir`; drop hardcoded `CACHE_DIR`.

**Tests / docs**
- No tests reference the runner today (verified) — nothing to update there.
  Add coverage as the new `profile()` surface stabilizes (out of scope for
  the strip itself).
- `AGENTS.md` / `CLAUDE.md` → update invocation notes (local run +
  transport shells; `--cache-root-dir`).

## Risks / open points

- **Kaizen reverse-sync lag (≤60 s).** Mitigated by poll/retry on
  `results.json` in step 4. If a run produces no `results.json` (e.g. a
  pure rollout/sim driver), the poll must have a bounded timeout and still
  sync whatever is present.
- **Venv activation line in step 3.** The exact `source .../activate` (or
  conda activate, on the Kaizen py312 image) differs per box. The shells
  take it from a hardcoded/overridable constant alongside the cache roots.
- **`ProfileOutput.hosts` removal** may ripple into `output.py`'s `__str__`
  — verify the summary table still renders.
```

