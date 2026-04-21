# Online-fusion phased plan

Target: a genuine, generic pattern-rewrite infrastructure on the op
graph that supports online fusion today and load/transpose fusion
and other peephole transforms next.

## Status at a glance

| Sub-step | State |
|---|---|
| 1a — driver + protocol | DONE |
| 1b — ScaleSpec + AccumulatorSpec | DONE |
| 1c — multi-accumulator + toposort insert | DONE |
| 1d — `exp_bias` render body | DONE |
| 1e — shape bump + 4-example verification | DONE |
| Phase 2 — Load/Store as graph nodes | not started |
| Phase 3 — Load+Transpose → DMATranspose peephole | not started |

Phase 1 results — all 4 examples sim 50/50, HW within noise:

| Example | Shape | Sim | HW | Best MFU |
|---|---|---|---|---|
| matmul/lhsT_rhs | 2048³ | 50/50 | 50/50 | 58.57% |
| matmul/lhs_rhs | 2048³ | 50/50 | 50/50 | 60.15% |
| matmul/lhs_rhsT | 2048³ | 50/50 | 50/50 | 63.16% |
| double_matmul | 2048³ | 50/50 | 22/50 | 39.80% (HW compiler flakiness) |
| **rmsnorm_matmul** (online-fused) | 2048³ | **50/50** | **50/50** | **20.88%** |
| **attention** (online-fused) | 512×128 | **50/50** | **50/50** | **4.41%** |

---

## Core design decision: composite-as-one-node

Every pattern-rewrite — online fusion today, load+transpose
fusion tomorrow, any future peephole — replaces a matched
subgraph with **one unified composite `NKIOp` subclass**. The
subgraph's semantics live entirely inside the composite — not
spread across multiple peer nodes.

The pattern is the same across all rewrites:

1. User writes the vanilla computation (e.g. rmsnorm + matmul,
   vanilla attention, load-then-transpose) as `NKIOp` calls in a
   math function.
2. `build_op_graph` produces the initial op graph from the user's
   calls.
3. Op-graph-level analysis runs every registered `PatternRewrite`
   until a fixpoint. Each pattern's `match` finds applications;
   each `apply` replaces a matched subgraph with ONE composite
   `NKIOp` node.
4. The composite flows through the rest of the pipeline exactly
   like any other op — sampler, validator, renderer treat it as
   a regular `NKIOp` subclass with its own `OPERAND_AXES` /
   `BLOCKING_AXES` / `format_isa_call` (or in the composite
   case, a dedicated render path).

Concrete composite classes:

* **`NKIOnlineFusionChain`** — subsumes an entire X +
  Accumulation subgraph (detected via `detect_online_fusion`).
  Per-match subclasses carry `OPERAND_AXES` / `OUTPUT_AXES` /
  `BLOCKING_AXES` / `TILE_LIMITS` plus the parametric
  `SCALE_SPEC` / `ACCUMULATOR_SPECS`. All X-step, σ-chain, and
  accumulator bodies live in the codegen render path, not in the
  graph. Applies to rmsnorm+matmul (role `rsqrt_then_mul`),
  attention softmax (role `exp_bias`), softmax+matmul (role
  `reciprocal_then_mul`), etc.
* **`NKIDMATranspose`** — subsumes a Load+Transpose pair
  (Phase 3). Per-match subclass encodes the DMA geometry; the
  composite's `format_isa_call` emits `nisa.dma_transpose`.

Why unified composites rather than peer-ops + annotations:

1. **Pure DAG.** No in-place writes, no loop-carry edges at the
   graph level. Running-state buffers are codegen-local details
   of the composite's render body, not IR tensors.
2. **Composes naturally with further patterns.** A composite
   appears to subsequent patterns as just another `NKIOp` — it
   has inputs, outputs, blocking axes, tile limits. Chains of
   online fusions, or a composite feeding into a DMA-transpose
   fusion, fall out of the iterative driver with zero extra
   machinery.
3. **Codegen locality.** Everything a composite needs to emit
   lives in ONE render function per composite class. The rest
   of the renderer doesn't special-case; it sees a regular op.

**The user's math function is never modified.** Everything
happens at the op-graph level: the user writes vanilla
attention, `build_op_graph` produces the vanilla DAG, pattern
rewrites fold matched subgraphs into composites, and codegen
renders whatever `NKIOp` classes the post-rewrite graph contains.

**OpGraph is the full lowered computation.** DMA ops (load, store,
DMA transpose) live as first-class `NKIOp` nodes alongside compute
ops. Every transformation is a `PatternRewrite` on this graph:

```
PatternRewrite.match(da, graph) -> list[MatchInstance]
PatternRewrite.apply(da, graph, instance) -> (da', graph')
```

The driver `apply_rewrites_until_fixpoint(da, graph, patterns)`
runs every pattern until a full pass over all patterns yields zero
matches. Pattern ordering matters only insofar as cascades — a
later pattern exposing a match for an earlier one triggers another
pass through the outer loop.

Composite nodes (online-fusion chain, fused DMA transpose) are
themselves `NKIOp` subclasses and participate in future pattern
matches without special-casing.

---

## Phase 1 — Parametric online fusion as a `PatternRewrite`

**Goal**: online fusion works on `rsqrt_then_mul` AND `exp_bias`
(attention's softmax) through ONE parametric rewrite + render
path, wired through the generic pattern driver.

### 1a. Driver + protocol — DONE
- `kernel_ir/pattern_rewrite.py`: `PatternRewrite` protocol,
  `MatchInstance` marker, `apply_rewrites_until_fixpoint` driver.
- Runtime-checkable protocols for structural typing.
- `build_ir` calls the driver with `[OnlineFusionPattern()]`.

### 1b. Parametric ScaleSpec + AccumulatorSpec — DONE
- `kernel_ir/online_fusion_spec.py`: `ScaleSpec` (combinator,
  init, delta op, inverse chain, sigma kind) and `AccumulatorSpec`
  (kind, output role, source op, chunk name).
- Rewrite builds these specs at match-apply time. No more
  string-typed `scale_role` dispatch in the rewrite body.
- Composite subclass (`NKIOnlineFusionChain`) carries the specs
  as class attrs; render path dispatches on them directly.
- One unified composite class (not per-role subclasses). The
  dynamic subclass per match only varies in axes/tile_limits +
  the specs; there's ONE base class.

### 1c. Multi-accumulator support — DONE
- `AccumulatorSpec` tuple — one per accumulator absorbed by the
  match. Render emits σ once, then per-accumulator chunk compute +
  `scalar_tensor_tensor` σ-correct-and-accumulate.
- Composite has N external outputs (one running_out per
  accumulator). External inputs = union of absorbed ops' inputs
  minus internally-produced tensors.
- Output role names use positional ``out_{i}`` so multi-output
  composites disambiguate.
- Graph insertion: composite inserts at the earliest legal
  survivor index (post external-input producers, pre external-
  output consumers) — `_composite_insert_position`. Required so
  edge rebuild (index-sorted) records every consumer edge.

### 1d. `exp_bias` render body — DONE
- Running state stores NEGATED max (reference-kernel convention):
  init +inf, combine via ``minimum``, `tensor_reduce(..., negate=True)`
  for the delta op.
- σ via one `nisa.activation(exp, data=prev, bias=running, scale=-1.0)`
  = `exp(max_old - max_new)`.
- `activation_reduce` accumulator uses `bias=running` with default
  scale — `exp(S + running) = exp(S - max_new)` directly.
- `nc_matmul` accumulator, when preceded by an `activation_reduce`:
  reuses the `exp_S` tile as stationary source (via
  `_matmul_stationary_source`) — no extra normalization needed.
- `source_kwargs` captured at detection time and stored in
  `AccumulatorSpec` so render reads them without a stale index
  lookup.
- Per-op arrays in `_reassemble_dim_analysis` now mirror
  `_reassemble_graph`'s composite insertion slot, so
  `per_op_blocking_dims` tracks the new op-graph order and
  `_check_blocking_innermost` correctly forces acc_dim
  innermost.

### 1e. Example shape bump + verification — DONE
- `examples/rmsnorm_matmul.py`: M=N=K=2048.
- All 4 examples sim 50/50. HW 50/50 on 5/6 groups
  (double_matmul at 22/50 HW is known compiler-pipeline
  flakiness on gym hosts, orthogonal to Phase 1 work).

---

## Phase 2 — Load/Store as first-class graph nodes

**Goal**: DMA ops join the op graph as explicit nodes. Codegen
reads the Load/Store nodes instead of synthesizing them from
tensor metadata. No behavior change — pure refactor.

### 2a. NKIOp subclasses for DMA
- `NKILoad`: HBM → SBUF copy. Takes an HBM tensor, produces an
  SBUF tensor. Its `format_isa_call` emits `nisa.dma_copy` (or
  `nisa.dma_transpose` for the fused transpose variant — but
  that's Phase 3's pattern).
- `NKIStore`: SBUF → HBM. Inverse of `NKILoad`.

### 2b. Graph insertion at build time
- `build_op_graph` (or a post-build pass) inserts one `NKILoad`
  per kernel-input tensor, one `NKIStore` per return tensor.
- Edges: `NKILoad.output` feeds every compute op that reads the
  kernel input. `NKIStore.input` is the producer of the return
  tensor.
- Tensor metadata: kernel inputs get BOTH an HBM tensor entry
  (the externally-provided parameter) and an SBUF tensor entry
  (the Load's output). `DimAnalysis.tensors` carries both with
  distinct names.

### 2c. Codegen realignment
- `render_sbuf_buffers` / `render_hbm_loads` / `render_hbm_store`:
  replace with per-node rendering. Each `NKILoad`/`NKIStore` emits
  through its `format_isa_call`. SBUF allocations still handled
  by the buffer machinery (tier, multi-buffer) but keyed on the
  Load's output tensor.

### 2d. Verification — no regression
- Byte-identical rendered output to Phase 1e's baseline for all
  4 examples (at fixed `PYTHONHASHSEED`). Pure refactor — if not
  byte-identical, something's semantically off.

---

## Phase 3 — Load+Transpose → DMATranspose peephole

**Goal**: the existing `find_fused_transposes` fused-DMA-transpose
logic becomes a pattern rewrite in the same driver, producing a
composite node just like `NKIOnlineFusionChain`. Demonstrates the
composite-as-one-node contract for a second, smaller pattern, and
validates that the pattern driver composes rewrites cleanly.

### 3a. `NKIDMATranspose` composite class
- One unified `NKIOp` subclass per the composite-as-one-node
  contract. Takes an HBM-tensor input (the load's source),
  produces an SBUF-tensor output (the transposed tile), emits
  `nisa.dma_transpose` via its `format_isa_call`.
- Per-match dynamic subclass via a `make_dma_transpose_class`
  mirror of `make_online_fusion_class`: fills `OPERAND_AXES`
  (swapped per transpose semantics), `OUTPUT_AXES`,
  `BLOCKING_AXES=frozenset()`, `TILE_LIMITS` from the matched
  pair's geometry.
- Appears to later passes as a regular `NKIOp` — the renderer's
  generic op-emission path handles it with no special casing.

### 3b. `LoadTransposePattern` rewrite
- `PatternRewrite` implementation that matches `NKILoad →
  NKITranspose` where the load's output is consumed only by
  the transpose. `apply` absorbs both into one
  `NKIDMATranspose` composite node via the same graph surgery
  helpers the online-fusion pattern uses (`_reassemble_graph`,
  `_rebuild_edges`). Registered alongside `OnlineFusionPattern`
  in the driver's pattern list.

### 3c. Remove legacy fused-transpose path
- Delete `find_fused_transposes` and `elided` kwarg threading in
  codegen. Render path no longer special-cases — it just emits
  whatever `NKIOp` classes the graph contains, including
  composites.

### 3d. Verification
- 4-example regression. Expected: matmul/double_matmul pick up
  the transpose fusion via the new pattern (they currently do
  through the legacy path); rmsnorm_matmul/attention unchanged
  (their transpose ops are already inside `NKIOnlineFusionChain`
  composites, which aren't eligible for further Load+Transpose
  matching — transpose is an INTERNAL operation of the online-
  fusion composite, not a separate op for the pattern matcher).

---

## Testing strategy

Every phase follows the same cadence:

1. **Incremental implementation** — each sub-step (1a, 1b, ...)
   leaves the tree rendering + running.
2. **Smoke test** after each sub-step — render a seed IR, confirm
   it doesn't blow up.
3. **4-example regression** at end of each phase — full serial
   remote run. Required: sim 50/50 on all; HW within noise of
   baseline OR better.
4. **Commit boundaries at phase ends only** — never mid-phase.

Phase 1 adds visible improvements (online fusion actually
running). Phases 2-3 are pure refactors — success = no regression.

---

## Guardrails against prior failure modes

1. **Composite as graph node, not splice** — one atomic
   replacement, no in-place updates to pre-existing ops.
2. **Parametric specs, not hardcoded branches** — render path
   dispatches on `ScaleSpec` fields, not `scale_role` strings.
3. **Edges always rebuilt after rewrite** — no empty edges list
   slipping through (this caused the lhs_rhsT regression
   earlier).
4. **Kernel inputs retained unconditionally** — prevents
   "KeyError: 'a'" in tensor_buffers when the rewrite cleans up
   absorbed-op-only tensors.
5. **No op-index order assumption in render** — either maintain
   topological order at rewrite time or make the renderer
   toposort explicitly.

---

## Open design notes

- **Multi-accumulator vs recursive fusion**: attention is ONE X,
  TWO accumulators sharing σ. My `OnlineFusionCandidate` already
  expresses this; the rewrite needs to emit both accumulator
  bodies in one render pass. Truly chained cases (X1 → X2 where
  X2 is another blocking reducer) would require the driver's
  iterative behavior — `match` after the first `apply` exposes
  X2 as a new candidate. We don't have a live example of this
  today; the iterative infrastructure supports it without extra
  work.
- **Composite as new-X**: the composite's class-level
  `BLOCKING_AXES = frozenset()`, so by default it's never matched
  as an X producer. If a future rewrite needs composite-as-X, it
  can override this on its subclass.
- **Scope of "X" in match**: today the detector starts from
  producer-reducer ops. Composites produced by prior rewrites
  have empty BLOCKING_AXES so they're not re-matched. This
  guarantees fixpoint termination trivially for the online-fusion
  pattern.
