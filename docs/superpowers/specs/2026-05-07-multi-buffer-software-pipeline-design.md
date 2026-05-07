# Multi-Buffer + Software Pipeline

*Date: 2026-05-07*
*Status: Draft for review*

## 1. Context and Goal

Initial-codegen places every op in its own loopnest, so every intermediate
is cross-loopnest and must materialize the full tile range to persist
values across loopnest boundaries. After `FuseLoops` brings a producer
and its consumers under a shared outer loop, those intermediates become
intra-loopnest on that dim — the buffer no longer needs the full range
and could drop to a single-iteration slot (plus optional multi-buffer
degree for software pipelining).

The current renderer does **not** exploit that shrinkage. `_sbuf_shape`
(`nkigym/src/nkigym/codegen/render.py:131-144`) reads `num_tiles(d)`
unconditionally from `op_graph.dims`, so every tuned kernel in the
rmsnorm_matmul pool carries bloated allocations. For instance,
`/home/ubuntu/cache/rmsnorm_matmul_compile/kernel_tuned_0000.py` fuses
`activation_reduce → tensor_reduce → activation` under a common `d0`
outer loop but still allocates `sbuf_sum_sq = nl.ndarray((128, 16, 1))`
when `(128, 1, 1)` would suffice. All sampled kernels are currently
hitting SBUF OOM — this bloat is the primary contributor.

This design:

1. Derives a `required_tiles(tensor, d)` quantity from forest+dep at
   render time. `FuseLoops`/`ReorderLoops` automatically shrink it by
   restructuring the forest — no new atom needed for basic shrinkage.
2. Adds a `buffer_degree[tensor, d]` field on `Tensor`, defaulting to 1,
   controlled by a new `MultiBuffer(tensor, dim, degree)` atom. When
   `degree > 1`, the tensor gets additional slots modulo'd by iteration
   — extra live-across-iteration capacity for software pipelining.
3. Adds a `pipeline_depth: int` field on `LoopNode`, defaulting to 1,
   controlled by a new `SoftwarePipeline(loop_path, depth)` atom. When
   `depth > 1`, the renderer emits a prologue + skewed body + epilogue
   sequence for that loop, with per-leaf iteration offsets computed
   mechanically from the op dep graph.

Shrinkage and multi-buffer are **decoupled**: fusion alone shrinks intra-
loopnest buffers to `required_tiles = 1`; `MultiBuffer` then grows them
back up by a deliberate factor when pipelining demands it. Cross-
loopnest tensors are pinned at `required_tiles = num_tiles(d)` and never
become candidates for multi-buffer degree above 1.

## 2. Architecture

Two new rewrite atoms in `nkigym/tune/`:

- `MultiBuffer(tensor_name, dim_id, degree)` — rewrites
  `op_graph.tensors[tensor_name].buffer_degree[dim_id]` to any divisor
  of `lca_trip_product(tensor, dim_id)`. Forest untouched.
- `SoftwarePipeline(loop_path, depth)` — clones the `LoopNode` at
  `loop_path` with `pipeline_depth = depth`; preserves `dim_id`,
  `trip_count`, `role`, `children`, `reduce_op`, `name` unchanged.
  Children shared by reference. Pipeline structure is a render-time
  concern — the tree shape is untouched.

Three IR fields:

- `OpGraph.tensors[name].buffer_degree: dict[dim_id, int]` — new field on
  the `Tensor` dataclass, default `{d: 1 for d in tensor.dim_ids}`.
  Persisted knob.
- `LoopNode.pipeline_depth: int = 1` — new field, persisted knob.
- `Tensor.required_tiles(dim_id, forest, dep) -> int` — derived, not
  stored. Computed by the renderer as
  `num_tiles(d) / lca_trip_product` where `lca_trip_product` is the
  product of `trip_count` over all `d`-iterating `LoopNode`s above the
  LCA of the tensor's producer + all consumers in the current forest.

State hash:

- `hash_state(op_graph, forest)` replaces today's `hash_forest(forest)`.
  Folds per-tensor `buffer_degree` into the hash alongside the forest
  canonical key. `pipeline_depth` is already part of `LoopNode` so it
  folds into the forest hash naturally via `_canonical_key`. This is the
  first rewrite that mutates `op_graph` — the existing `hash_forest`
  docstring already anticipates this ("once graph rewrites land, extend
  the hash to include op_graph").

Composition:

- `FuseLoops`, `ReorderLoops`, `MultiBuffer`, `SoftwarePipeline` are
  mutually orthogonal atoms. Sampler enumerates them independently.
- Fusion atoms restructure the forest → `required_tiles` drops for
  newly-intra-loopnest tensors at render time (automatic).
- `MultiBuffer` atoms become meaningful when `lca_trip_product > 1`
  (intra-loopnest tensors). The enumerator filters cross-loopnest
  tensors out, since their `lca_trip_product = 1` pins
  `degree` at `1`.
- `SoftwarePipeline` reads both forest and graph state to check skew
  vs. available total slots.

## 3. Components

Files touched:

### Graph-side

- `nkigym/codegen/graph.py` — `Tensor` dataclass gains
  `buffer_degree: dict[str, int]`. Populated in `parse_and_resolve` as
  `{d: 1 for d in tensor.dim_ids}` on every tensor.
- `nkigym/codegen/dep_graph.py` — unchanged. `OpGraph.dep` already
  carries `producer` and `consumers` maps that the legality and
  derivation code consume.

### Renderer

- `nkigym/codegen/render.py:_sbuf_shape` — replace
  `num_tiles(d)` reads with
  `required_tiles(tensor, d, forest, dep) * tensor.buffer_degree[d]`.
  The `required_tiles` function is new — a pure derivation over the
  current forest and dep graph.
- `nkigym/codegen/render.py:_slot_expr` — signature gains an explicit
  `total_slots: int` parameter (computed by the caller as
  `required_tiles * buffer_degree`). Wraps the sum-of-ordinals in
  `(sum) % total_slots`. When `total_slots == 1`, the wrap simplifies
  to a constant `0`. When `total_slots == num_tiles(d)`, the wrap
  simplifies to the raw sum (no-op). The simplification pass avoids
  cosmetic `% N` clutter in emitted source.
- `nkigym/codegen/render.py:_emit_node` for `LoopNode` with
  `pipeline_depth > 1` — emits three phases. Let `D = pipeline_depth`,
  `N = trip_count`, and each leaf carry a stage `s ∈ [0, D-1]`:
  1. **Prologue** (`D - 1` unrolled iters, `i_pro ∈ [0, D-2]`): at
     iter `i_pro`, every leaf at stage `s ≤ i_pro` fires with ancestor
     `loop_var` bound to the integer constant `i_pro - s` (emitted as
     a literal, no generated `for`). Leaves at stage `s > i_pro` do
     not fire.
  2. **Pipelined body** (`for i in range(D-1, N):`): every leaf fires
     once, with ancestor `loop_var` substituted by `(i - s)` via
     `_slot_expr`'s new `stage_offset` arg set to `-s`.
  3. **Epilogue** (`D - 1` unrolled iters, `i_epi ∈ [0, D-2]`): at
     iter `i_epi`, every leaf at stage `s > i_epi` fires with ancestor
     `loop_var` bound to the integer constant `N + i_epi - s`. Leaves
     at stage `s ≤ i_epi` do not fire.

  Total firings per leaf across all three phases = `N` (invariant: each
  op iterates exactly `trip_count` times, matching the un-pipelined
  case).
- `nkigym/codegen/render.py:assign_stages(loop_node, dep) -> dict[body_leaf_id, int]`
  — new helper. Longest-path walk over the subtree's producer→consumer
  DAG derived from `dep`. First leaves get stage 0; downstream leaves
  get `max(upstream_stages) + 1`. Used by both pipeline emission and
  `SoftwarePipeline.is_legal`.

### Loop forest

- `nkigym/codegen/loop_forest.py:LoopNode` — new field
  `pipeline_depth: int = 1`. Included in `_canonical_key`.
- `nkigym/codegen/loop_forest.py:check_invariant` — unchanged.
  `pipeline_depth` is a rendering concern; `trip_count` still represents
  the total iteration range, which the invariant checks.

### New rewrite modules

- `nkigym/tune/multi_buffer.py` — new file.
  - `MultiBuffer` dataclass with `is_legal` and `apply`.
  - `enumerate_multi_buffer_atoms(op_graph, forest)` — for each tensor,
    for each dim, for each divisor of `lca_trip_product` (excluding
    current `buffer_degree`), emit one atom. Cross-loopnest on dim
    (`lca_trip_product == 1`) emits nothing.
- `nkigym/tune/software_pipeline.py` — new file.
  - `SoftwarePipeline` dataclass with `is_legal` and `apply`.
  - `enumerate_software_pipeline_atoms(op_graph, forest)` — for each
    `LoopNode` in the forest, for each `depth ∈ [1, chain_length]`
    (excluding current `pipeline_depth`) where `chain_length` is the
    longest stage chain in the subtree, emit one atom. `depth = 1`
    atoms reset a previously-pipelined loop back to un-pipelined.

### Sampler

- `nkigym/tune/batch.py:enumerate_pool` — atom list becomes
  `fusion + reorder + multi_buffer + software_pipeline`. Pool dict keyed
  by `hash_state(op_graph, forest)` instead of `hash_forest(forest)`.
- `nkigym/codegen/loop_forest.py:hash_forest` → renamed `hash_state` with
  an extra `op_graph` parameter. The forest portion of the hash still
  uses `_canonical_key`; the graph portion hashes the tuple
  `((tensor_name, frozenset((d, deg) for d, deg in buffer_degree.items()))
  for tensor in op_graph.tensors.values())`.

## 4. Data Flow

### Starting state (no rewrites applied)

- Every op in its own loopnest.
- Every intermediate cross-loopnest on every `d`.
- `required_tiles(t, d) = num_tiles(d)` for all intermediates (lca is
  forest root).
- `buffer_degree[t, d] = 1` default.
- `pipeline_depth = 1` on every `LoopNode`.
- Rendered output matches today's kernel bit-identically (slot expr
  simplifies to raw sum; shape matches `num_tiles(d)`).

### After `FuseLoops` bringing AR + tensor_reduce + activation under a shared d0

- `sbuf_sum_sq`: producer = AR's closing tensor_reduce leaf; consumer =
  activation leaf. Both under the fused d0 `LoopNode`. LCA is that
  `LoopNode`. `lca_trip_product(d0) = 16`. `required_tiles(d0) = 1`.
- Renderer emits `sbuf_sum_sq = nl.ndarray((128, 1, 1), ...)` with slot
  `[0:128, 0, 0:1]`. Current bug (shape `(128, 16, 1)` and slot
  `[0:128, i_d0_0 + i_d0_1, 0:1]`) fixed automatically.
- `sbuf_lhs`: producer = load (still its own loopnest); consumer = AR
  and tensor_scalar. LCA = forest root. `lca_trip_product = 1`.
  `required_tiles = 16`. Unchanged from today.

### After `MultiBuffer("sbuf_sum_sq", "d0", 2)` on top of fusion

- `buffer_degree["d0"] = 2`. `lca_trip_product = 16`, so `degree ≤ 16`
  — legal.
- Total slots: `required_tiles * buffer_degree = 1 * 2 = 2`.
- Renderer emits `(128, 2, 1)` with slot
  `[0:128, (i_d0_0 + i_d0_1) % 2, 0:1]`.
- Enables `SoftwarePipeline` on the outer fused d0 loop for schedules
  that skew `sbuf_sum_sq` by 1 across iterations.

### After `SoftwarePipeline((0,), 2)` on top

- Legal: `sbuf_sum_sq`'s skew at depth 2 is 1; `total_slots(sbuf_sum_sq,
  d0) = 2 ≥ skew + 1 = 2`. OK.
- Forest: the target `LoopNode.pipeline_depth = 2`. Children untouched.
- Renderer at that node:
  1. Prologue iter (`i_pro = 0`): AR leaves fire with ancestor bound to
     `0`. Activation leaf does NOT fire (stage 1, `i_pro < stage`).
  2. Pipelined body `for i_d0_0 in range(1, 16):`: AR leaves fire with
     ancestor bound to `i_d0_0`; activation leaf fires with ancestor
     bound to `i_d0_0 - 1` (stage offset `-1`).
  3. Epilogue iter: activation leaf fires with ancestor bound to `15`.

### Cross-loopnest tensors pinned

- `sbuf_normed` in tune_0000: producer sits in the fused d0 loop;
  consumer sits in a d1-outer transpose nest. No shared d0 ancestor
  below the forest root for the consumer path → `lca_trip_product(d0) =
  1` → `required_tiles(d0) = num_tiles(d0) = 16`. Shape stays `(128, 16,
  ...)`. `MultiBuffer("sbuf_normed", "d0", degree>1)` filtered out by
  the enumerator.

### State hash

```python
hash_state(op_graph, forest) = hash((
    forest_canonical_key(forest),
    tuple(
        (t.name, tuple(sorted(t.buffer_degree.items())))
        for t in op_graph.tensors.values()
    ),
))
```

Self-move dedup: applying `MultiBuffer(t, d, current_degree)` yields the
same state — caught by the sampler's hash check. Applying
`SoftwarePipeline(loop, 1)` is filtered by `is_legal` (depth must be
≥ 2).

## 5. Legality and Error Handling

### `MultiBuffer(tensor_name, dim_id, degree)`

- `tensor_name ∈ op_graph.tensors` — else `False`.
- `dim_id ∈ tensors[tensor_name].dim_ids` — else `False`.
- `1 ≤ degree ≤ lca_trip_product(tensor, dim_id, forest, dep)` — else
  `False`.
- `degree` divides `lca_trip_product` — else `False`.
  (Ensures the modulo-indexed cycle aligns with the dim's tile count.)

`lca_trip_product` is derived. Cross-loopnest tensors have it equal to 1
and accept only `degree = 1` — an identity atom, filtered as a no-op
self-move.

### `SoftwarePipeline(loop_path, depth)`

- `loop_path` resolves to a `LoopNode` — else `False`.
- `depth ≥ 1` — else `False`. `depth = 1` is the reset-to-unpipelined
  form (valid only when current `pipeline_depth > 1`; the self-move
  case is filtered by `hash_state` dedup).
- `depth ≠ current pipeline_depth` — else `False` (self-move).
- `loop.trip_count ≥ depth` — else `False` (need ≥ 1 body iter after
  prologue).
- For `depth ≥ 2`: target `LoopNode` has ≥ 2 stages in
  `assign_stages(loop, dep)` — else `False` (nothing to pipeline).
- For `depth ≥ 2`: every tensor `t` live-across-iteration in the
  subtree has `total_slots(t, loop.dim_id) ≥ skew(t, depth) + 1`.
  Else `False`.

`skew(t, depth)` is computed from `assign_stages`:
- `producer_stage` = stage of the leaf writing `t` (unique by SSA).
- `consumer_stages` = stages of leaves reading `t`.
- `skew = max(consumer_stages) - producer_stage`.
- Linear chains: every tensor has `skew = 1` regardless of depth.
- Skip consumers (residuals): some tensors have `skew > 1` at higher
  depths, demanding larger `buffer_degree`.

### Error discipline

- `apply()` on an illegal atom raises `ValueError` — matches the
  renderer's "unknown `(op_kind, phase)` MUST raise" doctrine. Sampler
  pre-filters with `is_legal`, so reaching `apply()` on an illegal atom
  indicates an enumerator bug.
- `required_tiles` denominator not dividing `num_tiles(d)` — raises
  `ValueError` at render. Indicates an unsupported forest shape;
  surfaces loudly rather than silently mis-sizing the buffer.
- CPU-sim failures are **not** retroactive legality signals. The
  invariants above are strict enough that any rewrite passing them
  produces correctness-preserving source — same guarantee as the
  existing `FuseLoops` / `ReorderLoops` atoms.

### Scope boundaries (what this design does NOT do)

- No automatic skew inference per op — stages are derived only from
  `dep` edges, which is sufficient for linear chains and diamond
  patterns present in rmsnorm_matmul, attention, and MLP kernels.
- No cross-loop pipelining — `SoftwarePipeline` targets a single
  `LoopNode`; inter-nest overlap requires fusion first.
- No ring-buffer degrees that exceed `lca_trip_product` — bounded above
  by the tile count along the dim. Higher degrees are semantically
  meaningless (slots never cycle through within one live range).

## 6. Testing

### `test/codegen/test_multi_buffer_unit.py` (new)

- `MultiBuffer.is_legal` false on nonexistent tensor/dim, `degree < 1`,
  `degree > lca_trip_product`, non-divisor degree.
- `MultiBuffer.is_legal` true for legal parameters including cross-
  loopnest (`degree = 1`).
- `MultiBuffer.apply` clones only the targeted tensor; other tensors
  shared by reference.
- `enumerate_multi_buffer_atoms` — returns expected atoms for a small
  handcrafted `op_graph + forest`; cross-loopnest tensors yield zero
  atoms.
- Frontier sampler with only `MultiBuffer` atoms: determinism under
  seeded `Random`.

### `test/codegen/test_software_pipeline_unit.py` (new)

- `SoftwarePipeline.is_legal` false on non-LoopNode path, depth < 2,
  trip < depth, < 2 stages, insufficient `total_slots`.
- `SoftwarePipeline.apply` clones only the targeted `LoopNode`; sets
  `pipeline_depth`; children preserved by reference.
- `assign_stages`: linear chain `A → B → C` yields `{A:0, B:1, C:2}`;
  diamond `A → {B, C} → D` yields `{A:0, B:1, C:1, D:2}`; skip consumer
  receives `skew > 1`.
- `enumerate_software_pipeline_atoms` returns `(path, depth)` for
  `depth ∈ [2, chain_length]` over every LoopNode with ≥ 2 stages.

### `test/codegen/test_render_derivation.py` (new)

- Starting-state snapshot: every intermediate allocates `num_tiles(d)`;
  slot exprs are raw sums (no `% N` clutter).
- After fusion under rmsnorm_matmul's AR+activation: `sbuf_sum_sq`
  allocates `(128, 1, 1)`; d0 slot is constant `0`. **This test catches
  the current buffer-bloat bug.**
- After `MultiBuffer("sbuf_sum_sq", "d0", 2)` on top of fusion: shape
  `(128, 2, 1)`; slot `[0:128, (i_d0_0 + i_d0_1) % 2, 0:1]`.
- After `SoftwarePipeline` depth=2 on the outer fused d0 loop:
  prologue + skewed body + epilogue source snapshot byte-matches
  expected emission.
- Renderer raises `ValueError` on contrived forests where
  `lca_trip_product` doesn't divide `num_tiles(d)`.

### `test/codegen/test_multi_buffer_cpu_sim.py` (new)

- rmsnorm_matmul starting state: CPU-sim vs numpy passes (baseline).
- After fusion: CPU-sim passes, `sbuf_sum_sq` shrunk.
- After `MultiBuffer` on top: CPU-sim passes.
- After `SoftwarePipeline` on top: CPU-sim passes.
- Negative: applying `SoftwarePipeline` without sufficient
  `buffer_degree` raises at `is_legal`; bypassing `is_legal` raises at
  render.

### `test/codegen/test_batch.py` (extended)

- Extend frontier-sampler fixtures to include `MultiBuffer` +
  `SoftwarePipeline` atoms.
- Seeded-RNG determinism preserved.
- Spot-check: ≥ 1 sampled kernel has non-default `buffer_degree`; ≥ 1
  has `pipeline_depth > 1`.

### Out of scope for this design

- HW OOM / MFU regression testing — validated through the normal
  autotune gates once the rewrites land.
- Full autotune end-to-end run — follow-up task after the unit tests
  ship.
