# TVM TensorIR Knowledge (for the nkigym port)

Findings from reading the TVM (TIRx fork) source at `/home/weittang/workplace/tvm`
(symlinked from `/workplace/weittang/tvm`). Namespace is `tvm.s_tir` / `tvm.tirx`,
NOT classic `tvm.tir`. Every claim is tagged with its evidence level:

- **[SRC]** — read directly in TVM source, file:line cited.
- **[PROBE]** — confirmed by RUNNING TVM (built). The strongest evidence.
- **[INFERRED]** — reasoned from source, NOT executed; a hypothesis, may be wrong.

> **Method note (hard-won, 2026-06-09):** I flip-flopped THREE times on the
> init-domination question by reading source fragments and reasoning between them
> ("no check" → "stage-pipeline gate" → "region_cover"). Source reading is necessary
> but NOT sufficient. For any claim about what a primitive *rejects* or *emits*,
> prefer a **[PROBE]** over an **[INFERRED]** trace. Negative claims ("TVM has no
> check for X", "TVM would emit a wrong kernel") are the most trap-prone: a shallow
> trace of one function misses gates in its callees/preconditions. When a conclusion
> contradicts a system's core invariant (e.g. "correctness-preserving schedule
> primitives silently emit wrong kernels"), the prior is that the TRACE is wrong.

## How to actually run TVM (probe harness)

**TVM is NOT built or installed ANYWHERE on the Kaizen desktop** ([PROBE]
2026-06-09, exhaustive): not in `/opt/conda` (default), not in
`~/venvs/kernel-env`, no `libtvm*.so` under `$HOME`, no `tvm` package dir, source
not synced to `$HOME`. **`install.sh` does NOT build/install TVM** (only the Neuron
SDK + spike). The repo's oracle tests (`test/transforms/_tvm_struct_oracle.py`,
`importorskip("tvm")`) therefore SKIP on this desktop; the "oracle-verified vs a
BUILT TVM" note in `learnings.md` was a PRIOR/DIFFERENT environment, not this one.

**Consequence: [PROBE]-grade evidence is currently UNAVAILABLE** without first
building the TIRx-fork TVM (heavy CMake/C++ build) on the desktop. Until then,
init-domination claims rest on [SRC]+[INFERRED] only — see the open question below.

Probe script (ready, awaiting a built TVM): `test/transforms/_tvm_init_domination_probe.py`
— uses the same TIRx surface as `_tvm_struct_oracle.py` (`tvm.te.create_prim_func`,
`tvm.s_tir.Schedule`). Run via `transport/remote_pytest.sh` once TVM is built.

## Reduction block `init` — the model [SRC]

- `init` is a field of `SBlockNode` (`include/tvm/tirx/stmt.h`), never registered in
  `stmt2ref` (`state.cc` visits `block->body`, not `block->init`). → **not
  independently schedulable while welded; no `BlockRV`/`LoopRV` can name it.**
- Under `compute_at`, the whole `SBlock` (init included) is re-emitted via
  `ScopeReconstructor::MakeNewLoop` (`compute_at.cc:296-297`) — init travels with the
  block; no path separates it.
- `decompose_reduction(block, loop_rv)` is the ONLY primitive that extracts init into
  its own block (`reduction.cc`): allocates a new `*_init` `SBlockNode` (line 31,33),
  gives it ONLY data-parallel iter_vars (line 223 skips non-`kDataPar`) → init block is
  a **complete block**, inserts it right before the chosen `loop_rv`, registers it, and
  returns a schedulable `SBlockRV` (`reduction.cc:123`; `concrete_schedule.cc`
  `CreateRV<SBlockRV>`; `schedule.py -> SBlockRV`).
- Default init lowering (no `decompose_reduction`): `LowerInitBlock`
  (`src/s_tir/transform/lower_init_block.cc:49-66`) prepends init INTO the block body
  wrapped in `if (reduce_var == reduce_var.min)` (line 45). → memset executes at the
  **innermost** position, predicated to first reduction iteration — NOT hoisted.
- `decompose_reduction`'s `loop_rv` is a **schedulable choice**; `LoopHeightError`
  (`reduction.cc:109-158`) only REJECTS a loop lower than the reduction loops, does not
  compute placement.

> **TVM's model: init welded by default → `decompose_reduction` makes it a first-class
> schedulable block.** Two states, one transform between.

### Corrections (claims I made then retracted)
- ❌ "TVM derives init placement." **WRONG** — default is predicated-in-place; hoisting
  is an explicit `decompose_reduction(block, loop)` choice. [SRC]
- ❌ "TVM cannot express two-level init (outer-once / inner-per-ko)." **WRONG** — both
  inits exist after rfactor, each with its own `T.init()`. Equal expressiveness to ours;
  difference is *mechanism* (welded+decompose vs our always-sibling memset), not
  capability. [SRC]

## `compute_at` legality [SRC]

Explicit checks (`compute_at.cc:700-718`): (1) `GetScopeRoot(require_stage_pipeline=true)`,
(2) `CheckCompleteOrReductionBlock` (PERMITS reductions), (3) `NotInSameScopeError`,
(4) `CheckNotOutputBlock` (compute_at only). Then `FindInsertionPoint` enforces the
producer-consumer DAG ordering. **No check literally named "reduction axis covered" /
"init dominates"** — `_check_no_reduction_axis_covered` is OURS (nkigym), not a port.

The load-bearing implicit gate is **`region_cover` / `stage_pipeline`**
(`state.cc:204 CheckRegionCoverAndStagePipeline`): for each consumer, the producer's
WRITTEN region must cover the consumer's READ region within the scope
(`ProducerCoversConsumer`, line 321; producer lower-bound, consumer upper-bound, lines
296/315). Coverage fail → `region_cover=false` → `stage_pipeline=false` (line 331) →
`require_stage_pipeline=true` rejects. `VerifyCachedFlags` (`analysis/verify.cc:177`)
re-validates after EVERY primitive.

### Correction
- ❌ "TVM ALLOWS sinking a decomposed init into the reduction loop → silently produces a
  wrong kernel." Claimed from a shallow `FindInsertionPoint` subagent trace that MISSED
  the `region_cover`/`stage_pipeline` gate. **[INFERRED, suspected wrong]** — the
  `region_cover` [SRC] analysis predicts TVM REJECTS it. **NEEDS [PROBE] to confirm**
  (probe written, not yet run against a located TVM). Do not cite "TVM allows it."
  Open nuance the trace exposed: region_cover checks region *coverage*, and a memset
  sunk into K still *writes* `C[i,j]` — so whether coverage alone catches the re-zeroing
  is exactly what the probe must settle.

## `rfactor` [SRC]

Produces TWO blocks and is **TERMINAL** (no downstream fuse-to-single-accumulator
anywhere in TVM — checked `primitive/`, `meta_schedule/`, `dlight/`, tests):
- **rf-block**: writes `rf_buffer` of shape `[factor, *shape(C)]` (`CreateRFactorBuffers`);
  factored loop iter_var flipped `kCommReduce → kDataPar` (`reduction.cc:42`).
- **wb-block**: reduces `rf_buffer` over the factored axis into `C`; factored loop is
  `kCommReduce` (`reduction.cc:170`). Each block carries its own `T.init()`
  (`CreateBlockInit`, `reduction.cc:758-770`).
- **Does NOT reorder loops.** `CreateLoopOutsideRfactorBlock` (`reduction.cc`) reuses
  the original `loops` in original order, only renaming loop vars (`copy_with_suffix("")`)
  and re-wrapping outer→inner. Factored axis stays in place; only its iter_var TYPE flips.
- Canonical matmul output: `tests/python/s_tir/schedule/test_tir_schedule_rfactor.py:69-94`
  — input grid (line 56) == rf-block grid (line 75), byte-identical → proof of no reorder.
- TVM keeps the multi-slot `rf_buffer` ALIVE (never folds): the factored axis is meant to
  bind to **concurrent threads** (`dlight/gpu/reduction.py:193-205`:
  `split → rfactor → reorder → bind(threadIdx) → set_scope(local) →
  decompose_reduction(rf,r) → reverse_compute_at(wb,bx)`).

## Implications for nkigym (the port)

- **Our canonical IR is permanently in TVM's post-`decompose_reduction` state**: memset
  is always a separate sibling block. We never have the welded form → we never port
  `SBlockNode.init` / `LowerInitBlock` / `decompose_reduction`; we get init-schedulability
  for free from one uniform primitive set.
- **Cost:** a sibling memset CAN be orphaned (sunk into the reduction loop) — representable
  here, unlike TVM's welded form. So we carry an explicit check
  (`_check_no_reduction_axis_covered`, or its replacement) that TVM expresses structurally
  (welded init) + via `region_cover`.
- **Block-granular vs loop-granular dependency (KEY corrected understanding):**
  block-granular CAN express init-domination, but ONLY with **region-coverage analysis**
  (`region_cover`), NOT block-ordering edges alone. Our 2026-06-02 block-granular bug
  (memset sunk into K) was the *absence of region_cover*, not a granularity limit. So
  "mirror TVM block-granular" is viable IF we also port `region_cover` + `stage_pipeline`
  + `VerifyCachedFlags`. Our shipped loop-granular carry-edge model
  (`memset → K-loop` edge) gives the EQUIVALENT guarantee via a different representation.
  *(This whole bullet is [SRC]+[INFERRED]; the decisive behavior awaits [PROBE].)*

## ⏳ TO REVISIT: block-granular dependency model (parked 2026-06-09)

**We will revisit whether to adopt TVM's block-granular dependency model** (mirroring
`region_cover` / `stage_pipeline`) vs. keeping our shipped loop-granular carry-edge
model. Parked, NOT decided. Status when parked:

- The two models give the **equivalent** init-domination guarantee (see the bullet
  above); the choice is representation + faithfulness, not correctness. *[SRC]+[INFERRED]*
- Our current loop-granular model **works and ships** — RFactor and everything else
  proceed on it. This is purely a possible future refactor.
- It is the **blocker for the RFactor→fused fold** (narrowing
  `_check_no_reduction_axis_covered`; see the RFactor spec §7). The fold stays blocked
  until this is resolved.
- **Prerequisite to decide it properly:** run `_tvm_init_domination_probe.py` against a
  BUILT TVM (does TVM reject sinking a decomposed init into the reduction loop, via
  `region_cover`/`stage_pipeline`?). TVM is not built on the current desktop — see
  "How to actually run TVM" above. The probe verdict, not another source trace, should
  drive the decision (per the method note).

When revisiting: start from the probe result, then weigh (a) port `region_cover` +
go block-granular, vs (b) keep loop-granular and narrow `_check_no_reduction_axis_covered`.

## Trn2-vs-GPU specialization (why our rfactor diverges in OUTPUT, not mechanism)

TVM's rfactor exposes **spatial/concurrent** parallelism (slots → threads). Trn2 has ONE
Tensor Engine → reductions are **temporally pipelined**, not spatially parallel → the
multi-slot rf_buffer is pure waste. So our rfactor mechanism = TVM's, but our useful
OUTPUT folds to a single reused accumulator (fold = ComputeAt + compact_shapes in our IR;
TVM never folds because it wants the slots for threads). Same "faithful mechanism,
single-target output specialization" pattern as SoftwarePipeline.
