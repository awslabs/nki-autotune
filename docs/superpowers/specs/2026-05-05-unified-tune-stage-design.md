---
date: 2026-05-05
status: draft
---

# Unified Tune Stage — Frontier-Expansion Kernel Sampler

## Core decision

Replace the current one-kernel-per-call `run_tune` batch path with a
**frontier-expansion graph sampler**. The rewrite graph — every
kernel reachable from the canonical `(op_graph, forest)` by any
sequence of legal atoms — is enumerated as a single dedup'd pool,
keyed by `hash_forest`. Enumeration proceeds by maintaining a
frontier of pool nodes with unexplored outgoing atoms; each
iteration picks one frontier node uniformly, picks one of its
unexplored atoms uniformly, applies it, and adds the result to the
pool when the hash is new. After enumeration terminates we uniformly
sample `num_kernels` states from the pool, render each, and hand the
resulting `(filename → KernelJob)` dict to `autotune`'s
`remote_profile`. No iteration, no agentic reasoning — one round of
randomized graph exploration followed by bulk CPU-sim + HW profile.

Enumeration terminates when **either** `len(pool) >= 100 *
num_kernels` **or** the frontier is empty (whole reachable graph
enumerated). The `100×` factor gives downstream dedup'd uniform
sampling a meaningful draw pool even when the reachable set is
large; when it is small, we fall through to frontier-empty and
sample from whatever we have.

## Invariants

1. **Every pooled state is reachable and semantically valid.** Atoms
   are only returned by the per-atom enumerators when they are
   legal at the frontier node, and `apply` preserves semantics.
   CPU-sim divergence from `f_numpy` on *any* sampled kernel is a bug
   — raise `AssertionError` listing the failing kernels.

2. **Pool is a unified graph, not a set of trajectories.** Once a
   state is in the pool its outgoing atoms are enumerated exactly
   once (across the whole run, not once per "walker"). Every
   `(node, outgoing_atom)` pair is tried at most once; destinations
   already in the pool are discarded without expanding further from
   them through *that* edge (they may still have unexplored
   outgoing edges themselves).

3. **Order-invariant at exhaustion.** When enumeration runs to
   frontier-empty, the pool equals the full reachable set
   regardless of rng draw order. When enumeration stops early at
   the `100×` cap, the pool is a random-order-BFS-partial sample.

4. **Deterministic given seed.** `random.Random(seed)` drives
   frontier-node pick, atom pick, and final pool sampling. Same
   seed + same inputs → byte-equal `kernel_tuned_*.py` filenames
   and contents.

5. **`nkigym` boundary preserved.** The tune stage imports from
   `autotune.runner.api`; the reverse remains forbidden. No
   `nkigym.codegen` import from `autotune`.

## Rejected alternatives

- **Independent random walks from the root.** `R` walkers each take
  a random greedy trajectory, pool the visited states. Two biases:
  (i) path-multiplicity — states reachable by many distinct paths
  get disproportionate visits; (ii) walker termination rules
  (global-seen vs. local-cycle) shape the pool in hard-to-reason
  ways. Frontier expansion spends one enumeration budget per
  `(node, edge)` pair regardless of how many paths lead to either
  endpoint.

- **Depth-stratified random walks** (original user proposal: two-pass,
  estimate `max_depth`, sample `d ~ U(0, max_depth)` per kernel,
  apply exactly `d` steps). Same path-multiplicity bias as
  independent walks, plus an unstable `max_depth` estimate.

- **Deterministic BFS.** Canonical BFS of the same graph would
  yield identical pools in the exhaustion case. We randomize the
  frontier/atom pick so that the partial pool produced when the
  `100×` cap fires is a uniform random partial enumeration rather
  than a shallow-biased one.

- **MCMC / Metropolis-Hastings.** Uniform stationary distribution
  requires reversibility; `FuseLoops` has no inverse in the current
  rewrite set. Adding `UnfuseLoops` purely to make MCMC sound is
  premature — revisit if the downstream profile distribution
  reveals a pooling bias worth fixing.

- **Agentic reasoning during tune.** Premature — no baseline exists
  to measure against. Frontier expansion is the baseline an agentic
  approach will be compared to.

- **Drop explicit `rewrites` path.** Existing `test_compile.py`
  relies on it for deterministic end-to-end coverage of rewrite
  composition. Retained as optional kwarg for backward compatibility.

## Public API

### `nkigym_compile`

```python
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
```

**Behaviour change:**

- `stages` parameter removed. Always runs
  synthesis → initial_codegen → tune in order. Per-stage cache
  reuse is unchanged from today: each stage's helper overwrites
  its own artifact (`f_nkigym.py`, `kernel.py`, `kernel_tuned*.py`)
  unconditionally, and that behaviour is preserved — no
  skip-on-hit logic introduced here. Callers who previously relied
  on `stages=["tune"]` to replay tune against a hand-placed
  `f_nkigym.py` must now populate the full cache or invoke
  `run_tune` directly.
- New required params: `num_kernels`, `hosts`, `venv_python`,
  `neuron_platform_target`.
- New optional: `collect_detailed_profile` (forwarded to
  `remote_profile`).
- `rewrites` retained optional. When non-None, tune takes the
  deterministic explicit-apply path (no HW profile, writes
  `kernel_tuned.py` singular). When `None`, tune takes the batch
  frontier-expansion path.
- Return type stays `None`. Callers read results from
  `cache_dir/results.json` written by `remote_profile`.

### `run_tune` dispatcher

Lives at `nkigym/tune/stage.py`. Two paths:

| `rewrites` | Path | Output |
|---|---|---|
| `list[KernelRewrite]` | explicit | `cache_path/kernel_tuned.py`; CPU-sim check inline |
| `None` | batch | `cache_path/kernel_tuned_{0000..N-1}.py`; remote_profile writes `results.json`; raises on any CPU-sim failure |

Explicit path is unchanged from today. Batch path described next.

### Batch sampler: `nkigym/tune/batch.py`

New module — two pure functions, no I/O.

```python
def enumerate_pool(
    op_graph: OpGraph,
    forest: LoopForest,
    max_pool_size: int,
    rng: random.Random,
) -> dict[int, tuple[OpGraph, LoopForest]]:
    """Enumerate the reachable rewrite-graph via randomized frontier expansion.

    Maintains a frontier: pool nodes that still have un-tried outgoing
    atoms. Each iteration:

      1. Pick a frontier node uniformly: h = rng.choice(frontier_keys).
      2. Pop a random atom from frontier[h]'s unexplored-atoms list
         (swap-and-pop: `j = rng.randrange(len(atoms)); atoms[j], atoms[-1] = atoms[-1], atoms[j]; atom = atoms.pop()`).
      3. If frontier[h] is now empty, remove h from frontier.
      4. (og', f') = atom.apply(pool[h]).
      5. h' = hash_forest(f').
      6. If h' not in pool: pool[h'] = (og', f'); frontier[h'] = all
         legal atoms at (og', f'). Otherwise discard; no frontier
         entry written.

    Terminates when len(pool) >= max_pool_size OR frontier is empty.

    Returns the pool dict. Pool always contains the starting state
    keyed by hash_forest(forest). The atom list for each frontier
    node is the concatenation enumerate_fusion_atoms(f) +
    enumerate_reorder_atoms(f) at the moment the node was added.
    """


def sample_pool(
    pool: dict[int, tuple[OpGraph, LoopForest]],
    num_kernels: int,
    rng: random.Random,
) -> list[tuple[OpGraph, LoopForest]]:
    """Sample num_kernels states uniformly without replacement.

    If len(pool) < num_kernels: warn (warnings.warn) and return all
    pool values. Caller decides whether to proceed or raise.
    """
```

**Implementation notes.**

- *Frontier representation.* `dict[int, list[KernelRewrite]]` —
  dict for O(1) deletion by key, list for O(1) random-pop via
  swap-and-pop. Iterating `list(frontier)` once per iteration is
  acceptable at the scale we target (≤100×num_kernels nodes);
  optimize if profiling shows it matters.
- *Atom list snapshot.* The per-node atom list is snapshot at pool
  insertion time and never re-enumerated. This is safe because
  `enumerate_fusion_atoms`/`enumerate_reorder_atoms` are pure
  functions of the forest state, and our frontier nodes are
  immutable `(OpGraph, LoopForest)` tuples.
- *No per-edge dedup on atoms.* Two different atoms at the same
  node may `apply` to the same destination hash (e.g. commutative
  swaps reach the same permutation). We accept the redundant
  apply + hash computation — enumeration is cheap, destinations
  past the first hit are simply discarded in step 6.

### Driver: `_run_batch` in `stage.py`

```
parse_and_resolve(f_nkigym, input_specs)   → op_graph
build_canonical_forest(op_graph)           → forest
rng = random.Random(seed)
pool = enumerate_pool(op_graph, forest, max_pool_size=100 * num_kernels, rng)
sampled = sample_pool(pool, num_kernels, rng)

output_shape = _trace_output_shape(f_numpy, input_specs)   # existing helper or add one
nkigym_source = (cache_path / "f_nkigym.py").read_text()

kernels = {}
for idx, (og, f) in enumerate(sampled):
    source = render(og, forest=f)
    name = f"kernel_tuned_{idx:04d}.py"
    (cache_path / name).write_text(source)
    kernels[name] = KernelJob(
        source=source,
        func_name=f_nkigym.__name__,
        output_shape=output_shape,
        input_specs=input_specs,
        nkigym_source=nkigym_source,
        nkigym_func_name=f_nkigym.__name__,
        atol=_ATOL,
        rtol=_RTOL,
    )

remote_profile(
    kernels=kernels,
    hosts=hosts,
    cache_dir=str(cache_path),
    seed=seed,
    neuron_platform_target=neuron_platform_target,
    venv_python=venv_python,
    collect_detailed_profile=collect_detailed_profile,
)

_assert_no_cpu_sim_failures(cache_path / "results.json")
```

`_assert_no_cpu_sim_failures` reads `results.json`, filters for
entries where `cpu_sim.passed is False`, and raises
`AssertionError` listing the offending kernel names when any are
found. HW compile/runtime failures (non-zero `hardware_output`
with `cpu_sim.passed is True`) are not raised — they reflect
expected SBUF/PSUM OOMs that downstream consumers filter.

## Cache layout

```
cache_dir/
  f_nkigym.py                     # synthesis
  kernel.py                       # initial_codegen
  kernel_tuned_0000.py            # batch: N rendered kernels
  ...
  kernel_tuned_{N-1:04d}.py
  results.json                    # written by remote_profile
  compiler_logs/                  # remote_profile artifact
  ...
```

Explicit rewrites path continues to write `kernel_tuned.py` (no
index, no `results.json`).

## Edge cases

- **No legal atoms at canonical state.** `enumerate_pool` inserts
  the initial state with empty atom list → frontier empty after
  the first step's cleanup → immediate termination with pool of
  size 1. `sample_pool(N)` warns and returns that single state.
  Batch path still runs (1 kernel profiled).

- **Reachable set smaller than `max_pool_size`.** Frontier empties
  before the cap fires. Pool equals the full reachable set.
  `sample_pool(N)` returns `min(N, |S|)` distinct states (warns on
  under-fill).

- **Reachable set larger than `max_pool_size`.** Cap fires; pool
  is a random-order-BFS-partial sample of size `100 * N`.
  `sample_pool(N)` draws `N` distinct states.

- **Reorder atoms reaching pooled destinations.** Common case when
  the graph has commutative diamonds. The apply runs, the hash is
  computed, and the pool lookup discards the destination. No
  frontier entry is written and no further expansion happens
  through that edge. Other outgoing atoms at the source node
  (added to its frontier list at insertion time) remain eligible.

- **`FuseLoops` irreversible.** Fusion edges are one-way; the
  reverse state is simply not in the reachable set and the
  sampler never considers it. No special handling needed.

- **CPU-sim tolerance.** `_ATOL = _RTOL = 5e-3` (current
  `compile.py` values). Batch path uses same values via
  `KernelJob.atol`/`rtol`.

## Tests

### Unit (`test/codegen/test_batch.py`, new file)

- `test_enumerate_pool_deterministic` — fixed seed + fixed
  starting state → byte-equal sorted pool keys across two runs.
- `test_enumerate_pool_includes_initial` — pool always contains
  `hash_forest(starting_forest)`.
- `test_enumerate_pool_exhausts_small_graph` — on a starting
  state where the reachable set is known-small (e.g. a matmul
  with a single fusion atom available → pool size = 2; or a
  curated fixture with computable `|S|`), run with
  `max_pool_size` much larger than `|S|` → pool equals the full
  reachable set regardless of seed.
- `test_enumerate_pool_cap_respected` — set `max_pool_size=k`
  on a graph with `|S| > k` → `len(pool) == k` exactly.
- `test_enumerate_pool_no_legal_atoms` — canonical state with
  no legal atoms → pool of size 1, no error.
- `test_sample_pool_exact_fill` — pool of 10, N=5 → returns 5
  distinct states.
- `test_sample_pool_under_fill_warns` — pool of 3, N=5 → emits
  `UserWarning`, returns all 3.
- `test_sample_pool_deterministic` — fixed pool + seed → same
  sample.

### Integration (`test/codegen/test_compile.py`, amend)

- `test_nkigym_compile_batch_path_mocks_remote_profile` — patch
  `nkigym.tune.stage.remote_profile`, call
  `nkigym_compile(..., num_kernels=3, rewrites=None, ...)`, assert
  the patched profiler was called with exactly 3
  `kernel_tuned_{0000,0001,0002}.py` entries and each `KernelJob`
  has the right `func_name` / `output_shape` / `nkigym_source`.
- `test_nkigym_compile_batch_raises_on_cpu_sim_failure` — mock
  `remote_profile` to write a `results.json` with one entry
  flagged `cpu_sim.passed=False`; assert `AssertionError` mentions
  the failing kernel's filename.
- Existing `test_tune_*` tests using `rewrites=[...]` migrate to
  pass `num_kernels=1, hosts=[], venv_python="", neuron_platform_target=""`
  (or use `run_tune` module-internal directly — preferred for
  deterministic tests that don't exercise the HW knobs). Test
  coverage for explicit-rewrites semantics unchanged.
- Drop tests that pass `stages=[...]` for unknown-stage
  validation; the parameter no longer exists.

### Examples

- `examples/rmsnorm_matmul.py` and `examples/matmul_*.py`: update
  the `nkigym_compile` call to drop `stages=...` and pass
  `num_kernels`, `hosts`, `venv_python`, `neuron_platform_target`.

## Migration checklist

Existing callers of `nkigym_compile(stages=[...], rewrites=None)`
break. Concrete migrations:

1. `examples/rmsnorm_matmul.py:52` — replace
   `stages=["synthesis","initial_codegen","tune"]` with
   `num_kernels=N` + hardware kwargs.
2. `examples/matmul_lhsT_rhs.py`, `examples/matmul_lhs_rhs.py` do
   not call `nkigym_compile` — no change needed.
3. `test/codegen/test_compile.py` — see Tests section above.
4. No `agentic/` callers observed; re-grep at implementation time.

## Out of scope for this spec

- `run_tune_batch` exposed as a separate public symbol (current
  plan: internal helper on the batch path inside `stage.py`).
- Configurable `max_pool_size` multiplier (fixed at `100×`
  `num_kernels` here; promote to a kwarg if downstream workflows
  want different sampling-vs-exploration trade-offs).
- Weighted atom selection (uniform now; biased sampling if future
  priors warrant).
- Per-edge atom dedup (skip atoms whose destination hash is
  already pooled without running `apply`). Adds a precomputed
  destination-hash per atom; only worth it if apply is expensive
  relative to hashing, which is not the case today.
- Rank/filter/select inside the tune stage — consumers read
  `results.json` and decide.
- `UnfuseLoops` / reversibility work for future MCMC variant.
- New rewrite atoms (`tiles_per_block`, `hoist`, `multi_buffer`,
  `sw_pipelining`) — they plug into the existing atom-union in
  `enumerate_pool` by extending the concatenation at node
  insertion.
