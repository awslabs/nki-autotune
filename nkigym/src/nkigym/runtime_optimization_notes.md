# Runtime Optimization Notes

Baseline: 2048×2048×2048 matmul, `num_targets=100`, `seed=42`, `min_depth=20`.
Baseline wall time: **837s** (measured via `/fsx/weittang/gym_cache_no_loop`).
Baseline correctness: 100/100 correct variants, best MFU=0.1373, best min_ms=1.591.

All optimizations verified against baseline (≤1.5% MFU/latency difference due to machine fluctuation).

## Optimization 1: Replace `dataclasses.fields()` with Direct Methods

**Problem:** `dataclasses.fields()` was the #1 bottleneck — 82M calls, 148s total.
Called from generic `_get_dst_name`, `_stmt_input_names`, `_rename_stmt`, `_rename_ref`
helpers in DCE, normalize, and transforms. Each call does module-level lookup + tuple
allocation for field descriptors.

**Solution:** Added direct methods to NKIOp subclasses: `dst_name()`, `tensor_names()`,
`input_names()`, `renamed()`. Each method accesses fields directly by name with
identity-based short-circuit (returns `self` when unchanged). Also added
`TensorRef.renamed()` with the same pattern.

**Files changed:** `ops/base.py`, `ops/alloc.py`, `ops/dma_copy.py`, `ops/matmul.py`,
`ops/activation.py`, `ops/tensor_copy.py`, `ir/tensor.py`, `codegen/dce.py`,
`codegen/types.py`, `transforms/block_merge.py`.

**Result:** 837s → **472s** (44% faster).

## Optimization 2: O(n) Adjacency Detection via End-to-Start Matching

**Problem:** `OperandMergeTransform._find_dma_pairs` used `itertools.combinations(n, 2)`
for O(n²) pairwise adjacency checking. For large blocks with hundreds of DMA loads, this
dominated analysis time.

**Solution:** Added `find_adjacent_pairs()` in `block_merge.py`. Groups items by
non-target dimensions, then uses a `by_end` dict to match end positions to start positions
in O(n) per dimension. Updated `_find_dma_pairs` to use it.

**Files changed:** `transforms/block_merge.py`, `transforms/operand_merge.py`.

**Result:** 472s → **432s** (9% faster, 48% faster than baseline).

## Optimization 3: Consecutive-Only Data Reuse Pairs

**Problem:** `DataReuseTransform._find_reuse_pairs` emitted all O(n²) combinations of
identical DMA loads. For a root kernel with 256 blocks sharing the same source slice, this
produced 32K+ options (most redundant since applying one invalidates others).

**Solution:** Changed from `combinations(refs, 2)` to consecutive-only pairs:
`TransformOption(refs[i], refs[i+1])`. Reduces root opportunities from 61440 to 7680
(87% reduction) while preserving search space coverage — consecutive pairs are sufficient
since the search applies them iteratively.

**Files changed:** `transforms/data_reuse.py`, `test/golden/nki_data_reuse.py`.

**Result:** Combined with Opt 2: 472s → **427s** (total 49% faster than baseline).

## Optimization 4: DCE Backward Scan

**Problem:** Dead code elimination used a worklist-based approach with `producers` dict,
`stmt_map` dict, `live_set`, and explicit `_enqueue` function. For 539 DCE calls, the
worklist machinery (`_find_live_names` 55s, `_propagate` 24s, `_enqueue` 5s) totaled ~84s
profiled time.

**Solution:** Replaced with a simple backward scan. Statements are in topological order
(producers before consumers), so a single reverse iteration propagates liveness without
any intermediate data structures:

```python
live_names = {"output"}
for block in reversed(kernel.blocks):
    for stmt in reversed(block.body):
        if stmt.dst_name() in live_names:
            for name in stmt.input_names():
                live_names.add(name)
```

Removed `_propagate` and `_enqueue` functions entirely.

**Files changed:** `codegen/dce.py`.

**Result:** 427s → **413s** (profiled DCE: 61s → 14s cumulative). Total **51% faster**
than baseline.

## Optimization 5: NKIOp Hash Caching (Reverted)

**Problem:** Profile showed 81M `hash()` calls at 11s. Frozen dataclasses recompute hash
from all fields on every call.

**Attempted:** Added explicit `__hash__` with `__dict__`-based caching to all 5 NKIOp
subclasses. First call computes and stores via `object.__setattr__`; subsequent calls
return cached value.

**Result:** **Net negative** — reverted. Python-level `__hash__` dispatch + `dict.get`
overhead (17s + 7s) exceeded the C-level frozen dataclass hash (11s). CPython's tuple hash
already caches at each nesting level, making NKIOp-level caching counterproductive.

## Optimization 6: Remove Last `dataclasses.fields()` in block_merge

**Problem:** `_stmt_names()` in `block_merge.py` still used the old `dataclasses.fields()`
pattern for extracting tensor names from statements.

**Solution:** Replaced with direct `stmt.tensor_names()` call. Removed `dataclasses` and
`TensorRef` imports from block_merge.

**Files changed:** `transforms/block_merge.py`.

**Result:** Minor cleanup, folded into measurement cycle.

## Optimization 7: Lazy Normalization

**Problem:** `normalize()` was called after every transform+DCE in `expand_one` (540 times),
costing 46s cumulative. It scans all stmts twice (collect names + rename). But the search
produced 0 duplicates — normalize's dedup benefit was zero. Only 101 qualifying kernels
actually needed normalized output for NKI source rendering.

**Solution:** Moved `normalize()` from `expand_one` to `_save_variant`. The search loop
now operates on non-normalized kernels (structurally unique per transform path), and
normalizes only when writing NKI source to disk for compilation.

**Files changed:** `search/search.py`.

**Result:** 413s → **399s** (3.4% faster). Total **52% faster** than baseline.

## Optimization 8: Inline Adjacency Matching + Single-Pass DMA Map

**Problem:** `_match_ends_to_starts` was called 4.5M times from `_adjacent_on_dim`,
adding Python function call overhead. `_build_dma_alloc_map` used two passes over block
stmts (one for allocs, one for DMAs).

**Solution:** Inlined `_match_ends_to_starts` body into `_adjacent_on_dim` (eliminated
4.5M function calls). Merged `_build_dma_alloc_map` two-pass into single-pass, leveraging
the invariant that allocs always precede their consumers.

**Files changed:** `transforms/block_merge.py`, `transforms/operand_merge.py`.

**Result:** ~1-2s savings (within measurement noise). Minor micro-optimization.

## Optimization 9: Grouped Transform Interface

**Problem:** Opt 3 (consecutive-only data reuse pairs) restricted merges to `refs[i], refs[i+1]`.
This missed non-consecutive pairs (e.g., `refs[0]` with `refs[5]`), limiting the transform
search space.

**Solution:** Changed `NKITransform.analyze()` to return groups of mutually-exclusive options
instead of flat option lists. Added `sample_group(group, rng)` and `group_budget(group)` to
base class for per-group sampling and budget control. Search's `_add_node` builds `unexplored`
by repeating each group index by its budget. OperandMerge returns groups of TransformOption;
DataReuse returns StmtRef groups with lazy pair derivation (see Opt 10).

**Files changed:** `transforms/base.py`, `transforms/data_reuse.py`,
`transforms/operand_merge.py`, `search/search.py`, test goldens.

**Result:** Interface change only; performance measured with Opt 10.

## Optimization 10: Lazy Pair Derivation via Combinatorial Unranking

**Problem:** Opt 9 restored C(n,2) pair enumeration for DataReuse, creating 43M
TransformOption objects (24s in `__new__`) across 589 `_find_reuse_groups` calls.
Only 588 pairs were actually used by the search.

**Solution:** DataReuse.analyze() returns StmtRef ref groups (no pre-computed pairs).
`sample_group()` uses `rng.randrange(C(n,2))` + combinatorial unranking to derive the
k-th pair in `combinations(refs, 2)` order on demand. `group_budget()` returns C(n,2)
mathematically. RNG consumption is identical to `rng.choice(list_of_pairs)` (both call
`_randbelow(n)` once), so search behavior is preserved exactly.

Unranking formula: for index k in a group of n refs, find row i where
`row_start(i) = i*(2n-i-1)/2`, then `j = i + 1 + (k - row_start(i))`.
Returns `TransformOption(refs[i], refs[j])`.

**Files changed:** `transforms/data_reuse.py`.

**Result:** 419s → **397s** (5.3% faster). Eliminates 43M object allocations.
MFU diff 1.41% vs baseline (within tolerance). Total **53% faster** than baseline.

## Optimization 11: Weighted Budget Sampling

**Problem:** `_add_node` materialized a 61K-entry flat `unexplored` list (repeating group
indices by budget), and `expand_one` used O(n) `list.pop(random_pos)`. For 589 nodes,
this created 36M+ list entries total.

**Solution:** Replaced `unexplored: list[int]` with `budgets: list[int]` (one entry per group,
~512 entries) and `total_budget: int`. Added `_weighted_sample()` that picks a random target
in `[0, total_budget)` and walks the budgets list to find the group — O(groups) instead of
O(total_budget). RNG consumption identical (one `randrange` call).

**Files changed:** `search/search.py`.

**Result:** ~397s (within measurement noise of Opt 10). Reduces per-node memory from
61K list entries to 512. Minor micro-optimization.

## Summary

| Step | Wall Time | vs Baseline | Status |
|------|-----------|-------------|--------|
| Baseline | 837s | — | |
| Opt 1: Direct NKIOp methods | 472s | -44% | kept |
| Opt 2: O(n) adjacency detection | 432s | -48% | kept |
| Opt 3: Consecutive-only data reuse pairs | — | — | **reverted** (changes search) |
| Opt 4: DCE backward scan | 413s | -51% | kept |
| Opt 5: Hash caching | — | — | **reverted** (net negative) |
| Opt 6: block_merge cleanup | 413s | -51% | kept |
| Opt 7: Lazy normalization | — | — | **re-enabled** (see Opt 17) |
| Opt 8: Inline + single-pass | ~413s | -51% | kept |
| Opt 9: Grouped transform interface | — | — | **reverted** (changes search) |
| Opt 10: Lazy pair unranking | — | — | **reverted** (depends on Opt 9) |
| Opt 11: Weighted budget sampling | — | — | **reverted** (depends on Opt 9) |
| Opt 12: Id-based node lookup | ~413s | -51% | kept |
| Opt 13: Block-level caching + single-pass DCE | ~391s | -53% | kept |
| Opt 14: Deferred batch simulation | ~378s | -55% | kept |
| Opt 15: Non-blocking shutdown | ~370s | -56% | kept |
| Opt 16: Streaming compile→hardware | ~363s | -57% | kept |
| Opt 17: Re-enable lazy normalization | ~374s | -55% | kept |
| Opt 18: Lazy opportunity indexing | ~358s | -57% | kept |
| **Post-revert** | **~451s** | **-46%** | |

Opts 3, 9, 10, 11 reverted because they changed search behavior (different
RNG consumption, different pair enumeration). Opt 7 (lazy normalization) was
initially reverted but re-enabled as Opt 17 after verifying it produces
identical kernels. After revert+re-enable, generated kernels match baseline
`/fsx/weittang/gym_cache_no_loop` exactly (verified via file diff,
MFU=0.1373, 100/100 correct).

## Optimization 12: Id-based Node Lookup

**Problem:** `_node_by_id` used `id(kernel)` but the dedup check `kernel not in self._dedup`
still hashed the full kernel (recursive NamedTuple hash, 25M calls, 3.2s). Also, `expand_one`
and other hot paths looked up nodes via the `_dedup` dict which hashes on every access.

**Solution:** Added `_node_by_id: dict[int, _Node]` keyed by `id(kernel)` for O(1) lookups
in all non-dedup paths. Only `_add_node`'s `kernel not in self._dedup` still hashes. Exposed
`node_for(kernel)` and `node_count` property. `_bucket_pos` also keyed by `id(kernel)`.

**Files changed:** `search/search.py`.

**Result:** Hash calls dropped from 88M → 25M (71% reduction, 10.9s → 3.2s). Wall time
within measurement noise (~397-399s). Micro-optimization.

## Optimization 13: Block-level Analysis Caching + Single-pass DCE

**Problem:** Transform analysis (`_collect_opportunities`) scanned ALL blocks for every new
kernel (589 calls × ~256 blocks × ~84 stmts = 12.6M iterations). But transforms only modify
1 block per apply — 255/256 blocks are identical Python objects from the parent kernel.

Additionally, DCE used two separate full scans: `_find_live_names` (backward, 6.2s) then
`_filter_blocks` (forward, 5.4s). And `_filter_blocks` always created new block objects via
`_replace()`, even when no stmts were removed — defeating any future identity-based caching.

**Solution (3 changes):**

1. **Single-pass DCE**: Combined `_find_live_names` + `_filter_blocks` into one backward scan
   (`_backward_filter`). Each block is processed once: backward scan propagates liveness AND
   collects live stmts simultaneously. When all stmts in a block are live (common case for
   unchanged blocks), returns the original body tuple, preserving block object identity.

2. **DataReuse block cache**: Added `_block_cache: dict[int, list]` to `DataReuseTransform`.
   `_find_reuse_groups` checks `id(block)` in cache before scanning. Cache hit → reuse
   precomputed `(key, StmtRef)` entries. Only new/changed blocks trigger full body scans.
   Cross-block grouping (by DMA source key) is always recomputed (cheap dict operations on
   cached entries).

3. **OperandMerge block cache**: Same pattern. `_find_merge_groups` checks `id(block)` cache
   before calling `_find_dma_groups` + `_find_compute_groups`. Cache stores per-block
   `list[list[TransformOption]]`.

**Why it works:** Transforms create new kernels via `replace_block()`, which reuses unchanged
block objects (same `id()`). DCE identity preservation ensures unchanged blocks keep their
`id()` through the DCE pass. The search graph holds all kernels alive, so cached block ids
are never reused by different objects.

**Files changed:** `codegen/dce.py`, `transforms/data_reuse.py`, `transforms/operand_merge.py`.

**Result:** Search loop 187s → **119s** (36% faster). Transform analysis 42s → ~4s (90%
reduction). DCE 11.6s → 8.4s. Function calls 299M → 151M (49% fewer). Wall time ~402s →
**391s** due to compilation pipeline absorbing savings (search finishes faster, waits longer
for workers). Total **53% faster** than baseline. MFU diff 1.37%.

## Optimization 14: Deferred Batch Simulation

**Problem:** In the search loop, `_verify_node` (simulate + allclose) ran inline for each
qualifying kernel before `_submit_to_pool`. With 100 qualifying kernels at ~0.76s each,
simulation added ~76s to the search loop critical path, delaying compilation submissions.

**Solution:** Moved simulation from inline in the search loop to `_finalize_benchmark`.
New flow per qualifying kernel: `save → submit → append to pending_verify`.
`_run_search` returns `pending_verify`; `_finalize_benchmark` calls `_verify_batch`
before `wait_all`, so verification overlaps with ongoing compilations. Replaced
`_verify_node` with `_verify_batch` that iterates over all pending nodes.

The search loop (excluding verify) now runs in ~49s vs ~119s before. The 66s verify batch
runs while compilations are still in progress.

**Files changed:** `search/search.py`.

**Result:** 391s → **378s** (3.4% faster). Total **55% faster** than baseline.
MFU diff 1.38% vs baseline (within tolerance). 100/100 correct.

## Optimization 15: Non-blocking Process Pool Shutdown

**Problem:** `CompilationPool.shutdown()` and per-core hardware executors used
`shutdown(wait=True)`, joining all worker processes before returning. After
`wait_all()`, all futures are resolved and workers are idle — joining them adds
unnecessary blocking time (~18.5s across 191 compile workers + 100 hardware
executors).

**Solution:** Changed both to `shutdown(wait=False)`. Workers terminate
asynchronously. Also removed pre-existing `# type: ignore[import-untyped]`
comments from compile.py (added `reportMissingImports = "none"` to
`pyproject.toml [tool.pyright]` so pyright passes without suppression).

**Files changed:** `search/compile.py`, `pyproject.toml`.

**Result:** ~370s (within compilation variance, ~2% improvement). Minor
optimization — savings are small relative to compilation variance (±100s).

## Optimization 16: Streaming Compile→Hardware Dispatch

**Problem:** Hardware benchmarking ran sequentially after all compilations
completed: `wait_all(230s) → hardware(31s)`. The 31s hardware phase added
directly to the critical path even though NeuronCores are idle during
compilation (CPU-only).

**Solution:** Replaced the sequential `wait_all` + `run_on_hardware` with
`stream_compile_and_run`. As each compilation future resolves via
`as_completed`, the NEFF is immediately dispatched to a Neuron core via a
shared `ProcessPoolExecutor(max_workers=128)`. Hardware benchmarks run
concurrently with remaining compilations.

Also consolidated per-core hardware dispatch: removed 100 separate
single-worker executors, replaced with one shared executor. Core pinning
moved from `initializer` to the worker function (`_run_core_worker` now
takes `core_id` as first argument and sets `NEURON_RT_VISIBLE_CORES`).

Extracted `_make_benchmark_cfg` and `_collect_hw_results` helpers. Removed
`_dispatch_to_cores`, `run_on_hardware`, `_split_into_groups`,
`_set_neuron_core` (all superseded by streaming).

**Files changed:** `search/compile.py`, `search/search.py`.

**Result:** ~370s → **363s** (1.9% faster). Hardware (20s) now fully
overlapped with compilation tail. Total **57% faster** than baseline.
MFU diff 1.47% vs baseline. 100/100 correct.

## Optimization 17: Re-enable Lazy Normalization

**Problem:** After reverting Opts 3, 9, 10, 11, normalize was back in `expand_one`
(called 589 times, ~40s cumulative). Additionally, normalize destroys block object
identity — every block gets a new NamedTuple with renamed tensors — which defeats
the block-level analysis caching from Opt 13. With normalize in expand_one, block
cache hit rate dropped to near zero.

**Why it was originally reverted:** Opt 7 was reverted alongside Opts 3/9 because
the combination changed search behavior. However, lazy normalize alone does NOT
change search behavior: the search produced 0 duplicates in practice, so normalize's
dedup effect was zero. The flat `unexplored` list indexing is identical regardless
of whether normalize runs in `expand_one` or `_save_variant`.

**Verification:** Ran `/fsx/weittang/e2e_runtime_cache_41` with lazy normalize.
All 101 kernel files are byte-identical to baseline (`/fsx/weittang/e2e_runtime_cache_39`).
MFU=0.13725 (baseline=0.13727, diff=0.01%). 100/100 correct, 0 failed.

**Solution:** Moved `normalize()` from `expand_one` back to `_save_variant`. The
search loop uses `dce(transform.apply(...))` only. Normalize runs at save time
when writing NKI source to disk (101 calls vs 589).

**Files changed:** `search/search.py`.

**Result:** ~451s → **374s** (17% faster than post-revert). Total **55% faster**
than baseline. Savings come from two sources: (1) eliminating 488 unnecessary
normalize calls (~40s), and (2) restoring block identity for Opt 13 caching
(~30s saved from cache hits).

## Optimization 18: Lazy Opportunity Indexing

**Problem:** `_collect_opportunities` materialized all 61K+ `(transform, option)`
tuples per node by iterating `analyze()` results. For DataReuse, this created
35.5M TransformOption objects (12.2s `__new__`) and 33.5M genexpr iterations
(6.3s) via `combinations(refs, 2)`. The append loop in `_collect_opportunities`
added 19.2s self time. Total: ~38s. Only ~542 options were ever accessed.

**Solution (2 changes):**

1. **`_LazyPairSeq`** in `data_reuse.py`: Virtual `Sequence[TransformOption]`
   that stores ref groups and computes C(n,2) pairs on demand via combinatorial
   unranking. Same ordering as `combinations(refs, 2)` — given flat index k,
   finds group via `bisect`, then unranks via the formula:
   `i = (2n-1 - isqrt((2n-1)^2 - 8k)) // 2`, `j = i + 1 + (k - row_start)`.
   `_find_reuse_pairs` returns `_LazyPairSeq(groups)` instead of materializing
   all pairs.

2. **`_LazyOpportunities`** in `search.py`: Stores `(transform, options)` segments
   without iterating them. `__getitem__` uses `bisect` to find the segment, then
   delegates to the segment's `__getitem__`. `_collect_opportunities` builds this
   in O(transforms) instead of O(total_options).

**Why search behavior is identical:** The total count `len(opportunities)` is the
same (61440 for root). The mapping from flat index to `(transform, option)` is
identical — same group ordering (dict insertion order), same pair ordering within
groups (unranking matches combinations). RNG consumption is unchanged.

**Files changed:** `transforms/data_reuse.py`, `transforms/base.py`,
`search/search.py`.

**Result:** 374s → **367s** (1.9% faster). Function calls 243M → 144M (41%
fewer). `__new__` calls 35.5M → 2.2M (94% reduction, 12.2s → 0.3s).
`_collect_opportunities` vanished from top-50 profile (was 19.2s self).
`_run_search` 97s → 60s. All kernel files byte-identical to baseline.
MFU=0.13727. 100/100 correct.

## Remaining Bottlenecks (Profile at Opt 18)

| Function | Self Time | Calls | Notes |
|----------|-----------|-------|-------|
| `_thread.lock.acquire` | 202s | 752 | Compilation wait (dominant) |
| `matmul.simulate` | 55s | 410K | Correctness verification (inherent) |
| `numpy.zeros` | 18s | 868K | Simulation alloc (inherent) |
| `dma_copy.simulate` | 11s | 843K | Simulation copy (inherent) |
| `tensor.renamed` | 10s | 2.9M | Normalize rename pass |
| `_backward_filter` | 7.6s | 132K | DCE single-pass |
| `hash` built-in | 2.9s | 23M | Kernel dedup |
| `_add_node` | 3.3s | 543 | Node registration |

**Critical path**: `_run_search(60s) → _verify_batch(99s) →
stream_compile_and_run(205s)`

The compilation phase (205s) dominates. Search (60s) + verify (99s) are
fully overlapped with compilation. Hardware benchmarking (22s) is hidden
behind the compilation tail.

**Critical path analysis**: The wall time equals the total compilation
time. All CPU work (search + verify = 159s) is fully overlapped with
compilation (~360s). Even if ALL search and verify time were eliminated,
the wall time would still be ~210s (compilation wait in
`stream_compile_and_run`).

Verified by comparing sequential vs threaded verify: moving verify to a
background thread doesn't help because verify (99s) is already hidden
within the compilation window (compilations run from t=0 to t=~360).

**Conclusion**: Achieved **57% reduction** from 837s to ~358s (clean run
without cProfile). The wall time is at the compilation floor. Further
improvements require: (1) faster neuronxcc compilation, (2) fewer
compilations (compile only promising variants), or (3) loop rolling to
reduce NKI source size for the compiler.

## Compilation Floor Investigation (Post Opt 18)

Investigated three approaches to break through the compilation floor:

### Attempt A: Local NVMe Compilation (rejected)

**Hypothesis:** FSx Lustre writes (17MB per compilation × 101 concurrent)
cause I/O contention.

**Test:** Compiled to local NVMe (`/tmp`), copied only NEFF back to FSx.
I/O benchmark: NVMe 2.4 GB/s vs Lustre 989 MB/s.

**Result:** 358s → **402s** (12% slower). The copy + cleanup overhead
exceeded any I/O savings. FSx handles concurrent writes well enough.

### Attempt B: Thread-Limiting Environment Variables (rejected)

**Hypothesis:** neuronxcc uses OMP/MKL threading that could be constrained.

**Test:** Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2` in worker initializer.

**Result:** 358s → **364s** (no improvement). neuronxcc's parallelism
is not controlled by these environment variables.

### Attempt C: Reduced Worker Count (rejected)

**Hypothesis:** Each neuronxcc compilation spawns ~5000 threads/processes
(measured: 128 main threads + 4795 children per compilation). With 101
concurrent compilations, that's ~500K threads on 192 CPUs. Reducing to
32 workers would lower contention.

**Measurement:** Single compilation (idle machine): **52s**.
With 191 workers (101 concurrent): **337s** spread (6.5× contention).

**Test:** 32 workers.

**Result:** 358s → **436s** (22% slower). Per-compilation time improved
(less contention) but serialization (batching 101 tasks through 32
workers) dominated. The throughput-optimal point is at maximum
concurrency (191 workers) despite the massive thread oversubscription.

### Key Findings

| Metric | Value |
|--------|-------|
| Single-compilation time | 52s |
| Threads per compilation | ~5000 |
| 101-concurrent wall time | ~337s |
| Parallelism efficiency | 15.6× on 192 CPUs |
| NKI source size | 22,355–22,541 lines per file |

The compilation floor is fundamental: neuronxcc processes 22K-line NKI
files with internal parallelism that cannot be tuned externally. The
`--internal-compiler-debug-mode=penguin` flag is required for NKI IR
compilation (not optional despite the "debug mode" warning).

### Attempt D: Compiler Optimization Level -O1 (rejected)

**Hypothesis:** `-O1` "enables core performance optimizations while
aiming to minimize compile-time" per `neuronx-cc --help`.

**Test:** Compiled best baseline variant (22K lines) with `-O1` vs `-O2`.

**Result:** `-O1` 52.3s, `-O2` 51.1s — identical within noise.
MFU identical (0.13716). NEFF sizes identical (400,384 bytes). The
optimization level has no measurable effect on NKI kernel compilation.

### Attempt E: Loop Rolling — Full (rejected, MFU regression)

**Hypothesis:** Replacing 256 helper functions with `nl.affine_range`
loops would shrink source from 22K to ~100 lines, drastically
reducing compilation time.

**Test:** Manually created rolled kernel with `nl.affine_range(16)` ×
`nl.affine_range(16)` calling one `_rolled_block` helper. Required
`nisa.tensor_copy` (PSUM→SBUF staging) before final `nisa.dma_copy`
— neuronxcc rejects direct PSUM→HBM DMA inside loop bodies.

**Result:** 101 lines, **6.9s compile** (7.5× faster). But
**MFU=0.0745** vs baseline 0.1373 — a 1.84× performance regression.
The `nl.affine_range` loop constrains the compiler's global
instruction scheduling, preventing cross-iteration overlap that the
unrolled version achieves.

### Attempt F: Loop Rolling — Partial, Outer Only (rejected)

**Hypothesis:** Rolling only the outer loop (16 iterations) while
keeping 16 inner blocks unrolled per iteration would give the compiler
more scheduling freedom within each iteration body.

**Test:** Outer `nl.affine_range(16)` with 16 unrolled inner block
helper functions.

**Result:** 1423 lines, **9.3s compile**. But **MFU=0.0744** —
identical degradation to full rolling. The `nl.affine_range` loop
itself is the bottleneck, regardless of inner body structure.

### Attempt G: Parallel Compilation Scaling Analysis

Measured compilation scaling with 1–32 concurrent identical tasks:

| Concurrent | Wall (s) | Per-task (s) | Throughput |
|-----------|----------|-------------|------------|
| 1         | 50.7     | 50.4        | 0.020/s    |
| 2         | 54.0     | 53.1        | 0.037/s    |
| 4         | 59.3     | 58.5        | 0.067/s    |
| 8         | 70.4     | 68.6        | 0.114/s    |
| 16        | 94.9     | 92.5        | 0.169/s    |
| 32        | 146.8    | 142.7       | 0.218/s    |

Key resource measurements per compilation:
- **Peak RSS:** 6.19 GB (768 GB total → 128 concurrent fits in RAM)
- **CPU usage:** ~1 effective CPU (47s user / 52s real with 8-CPU
  `taskset` — unchanged from unconstrained)
- **Thread count:** ~5000 (but almost all idle)

Thread-limiting env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS`, `TBB_NUM_THREADS`,
all set to 1) had **zero effect** on scaling — the compiler's
threading is intrinsic and not controlled by standard env vars.

The scaling bottleneck is kernel thread scheduling overhead with
500K+ mostly-idle threads, not CPU, memory, or I/O. The current
"all tasks at once" strategy is optimal: serializing into batches
is strictly worse.

### Conclusion

The compilation floor at ~358s is **hard** — no external tuning
(thread limits, CPU affinity, optimization levels, I/O paths, worker
counts) can improve it. Loop rolling achieves 7× faster compilation
but degrades MFU by 1.84×, violating the baseline correctness
constraint. The `nl.affine_range` construct fundamentally limits the
compiler's global instruction scheduling vs fully unrolled code.

Possible future paths (not currently actionable):
1. **neuronxcc improvements** — compiler team could optimize NKI
   tracing/tensorization for large unrolled kernels
2. **Fewer compilations** — compile only promising variants (requires
   a heuristic for pre-filtering without running on hardware)
3. **NEFF caching** — cache compiled NEFFs by source hash across runs
   (helps repeated profiling, not first run)
