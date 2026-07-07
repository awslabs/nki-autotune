# Region-cover dependency, unified CodeMotion, and BufferLayout — design

**Date:** 2026-06-25
**Status:** design (approved in brainstorming; pending written-spec review)
**Target:** reproduce `examples/manual_transforms.py` (`kernel_0 … kernel_27`)
exactly, by a transform-driven ladder (the way `examples/kernel_transforms.py`
drives the shipped transforms to `kernel_target`).

## Problem

`examples/manual_transforms.py` is the new, finer-grained target ladder for the
2048³ bf16 `matmul_lhsT_rhs` workload. Its inline comments name the operation
each rung applies: `Split`, `Reorder`, `Buffer layout`, `Code motion`, the
recurring `Side effect: automatic buffer scope tightening and compaction`, and a
final `Apply RFactor` (k26→k27). The shipped transforms cannot reproduce it, for
five reasons (four the user confirmed, plus RFactor — see below):

1. **Heuristic dependency edges are too fragile.** `nkigym/ir/dependency.py`
   carries synthetic `CARRY` (init-dominates-reduction) and `COVER`
   (consumer-needs-all-tiles) edge kinds plus a `skip_cover_loops` escape hatch
   and the `_carry_loops_of_leaf` / `_tiled_write_loops_of_leaf` /
   `_reads_independently_of_loop` heuristics. This machinery has flip-flopped
   historically (the 2026-06-02 memset-sunk-into-K bug; the B1/B2 over-rejection
   barriers).
2. **The manual ladder is not representable.** The `Buffer layout` rungs render a
   Python *list* of separate `nl.ndarray` tiles indexed by a leading subscript
   (`psum_prod[i_d1_0][0:128, 0, 0:512]`). The `Buffer` dataclass and the codegen
   cannot express this — `_emit_alloc` emits exactly one `nl.ndarray` per buffer.
   This is the measured k22 "46.6% vs 90.7%, all buffer representation" gap.
3. **Side effects are not tracked.** Code motion's scope-tighten (`place_buffers`)
   and compaction (`compact_shapes`) run after every move and feed codegen, but are
   invisible to `Dependency` (rebuilt from scratch each time), so legality never
   sees the compacted footprint.
4. **Dependency is re-derived from scratch.** `Dependency(tree)` re-runs every
   region/role/RMW heuristic after every transform.
5. **RFactor is under-verified and suspected incomplete/buggy** (user-flagged as a
   first-class goal, twice). Evidence from the code, not inference:
   - **Zero coverage of late application.** Every case in `test_rfactor.py` runs
     `Split(K)` → `RFactor(ko)` immediately, on the **fully-packed canonical** form
     (`split_k_ir()` in `_rfactor_fixtures.py`). Nothing exercises the k26 state the
     manual ladder applies RFactor to (M tiled `i_d1_0×i_d1_1`, N split, **list**
     buffers, loads sunk). The suspected defect lives exactly where there is no test.
   - **The one byte-exact fixture is not `kernel_27`.** `kernel_rfactor_ko.py` emits
     the **packed** `(128,16,2048)` fused form; `kernel_27` emits a **list** of
     `(128,1,512)` tiles with a per-`i_d1_1` PSUM. Reproducing `kernel_27` is *not*
     "apply today's RFactor late" — RFactor must emit the per-tile fold the ladder
     actually contains.
   - **Stale terminology in the live tests.** `test_rfactor.py` docstrings still
     describe the abandoned multi-slot "rf-block/wb-block/`psum_rf`" form while the
     code does the fused `sbuf_rfactor`/`tensor_tensor` form — the suite passes but
     documents a form that no longer exists.
   So RFactor's correctness is **established only for one early, packed
   configuration**. This work treats RFactor as a first-class, **diagnosis-first**
   goal (§6), not a downstream build step.

## Goal

A transform-driven ladder emits each `kernel_0 … kernel_27` **byte-exact**
(rendered IR == the hand kernel) and **CPU-sim clean** vs the numpy golden, and
the runnable rungs profile on real Trn2 hardware. Ground truth is empirical on
gym-1 via `transport/ssh_host.sh` (CPU-sim + byte-exact ladder + HW MFU +
`pytest`), **not** a TVM oracle — the region-cover decision is made, so the
parked TVM probe in `tvm_knowledge.md` is moot and not a dependency of this work.

Four **co-equal** workstreams, each a first-class goal (not derived from the others):

1. **Region-cover dependency model** (§1–2) — replace the fragile CARRY/COVER edge
   heuristics; carry footprints, recompute ordering on demand.
2. **Unified `CodeMotion`** (§3) — fold ComputeAt + ReverseComputeAt; rename to
   code_motion.
3. **`BufferLayout` + side-effect tracking** (§4) — list-of-tiles representation
   and the compaction/scope-tighten side-effect, made footprint-visible.
4. **RFactor debug + correct** (§6) — the user-flagged goal. Diagnose what RFactor
   actually does across the ladder's states (it is verified only for one early,
   packed config), find the incompleteness/bug, and correct it so RFactor reproduces
   `kernel_27`. **Diagnosis precedes fix** — the precise defect is not yet known.

## Non-goals

- No change to `_move`'s verbatim same-prefix residual splice (correct, byte-exact).
- No change to Split / Reorder / Fuse / SoftwarePipeline internals. **RFactor is the
  exception** — it is a debug-and-correct goal (§6); its `_emit_rmw` internals are
  expected to change once diagnosis (Task 6.1) names the defect.
- No change to the `arith` substrate or the `nki` ops.
- No TVM build / probe. The validation loop is gym-1, not a TVM comparison.

## Scope

Strictly: the dependency/legality model, the code-motion transform surface, the
buffer-layout representation + transform, and the codegen for the list-of-tiles
form. Everything else is untouched.

## Anchor

The dependency model exists for **one job: gate code motion**. Its canonical
rule is *"a producer cannot move after its consumer, nor a consumer before its
producer."* Every other concept in today's model (the `CARRY` / `COVER` edge
kinds, `skip_cover_loops`) is machinery bolted on to approximate that rule for
the reduction and tiling cases. The revamp replaces all of it with the single
question region-cover already answers.

## 1. Dependency model: one coverage rule

Today `Dependency` carries three edge families, all re-derived after every
transform: **RAW/WAW/WAR** flow hazards, **CARRY** ("init dominates the reduction
loop"), and **COVER** ("consumer needs every tile a loop produces"), plus the
`skip_cover_loops` hatch for when a move legitimately dissolves a COVER edge.

The revamp collapses these into one question, asked only when code motion
proposes a move:

> Move block **B** under loop **L** at slot **i**. For every producer→consumer
> pair touching B, does the producer's **written region still cover the
> consumer's read region** at B's new execution position?

The two "special" edges fall out of this — they stop being separate concepts:

- **COVER *is* region-cover.** A producer inside `L` writes one tile per
  iteration; a consumer reading the full extent on an axis `L` does not index is
  not covered until `L` finishes, so it must stay outside `L`. This is exactly
  `_add_coverage_edges` / `_reads_independently_of_loop` today, restated as the
  general coverage query rather than a precomputed edge.
- **CARRY *is* region-cover against a live range.** A reduction block RMWs its
  accumulator loop-invariantly (matmul `psum_prod`, the two-stage fold's
  `sbuf_prod`), so the accumulator's **live range spans the whole reduction
  loop**. Any *other* writer of that buffer (the init memset) must sit entirely
  outside that live range; sinking it *into* the loop splits the range and
  re-zeros per iteration → NaN. This is the `_carry_loops_of_leaf` /
  `_own_carry_loop_nids` guard, restated as a live-range containment check.

So the model is: **per-block read/write footprints** (region evaluated in their
enclosing-loop context) + **one coverage query**. The accumulator live-range is
the single subtle invariant carried explicitly — it is the exact thing that bit
the project before (memset-sunk-into-K, k16/k17 NaN), so it is *named*, not
implicit in an edge-kind.

### Direction must come from the pre-move program

The load-bearing bug-guard in today's code: producer→consumer **direction** is
frozen from the original program (`first_backward_edge_for_insertion` reads
directions from `ir.dependency`, positions from the proposed splice). Re-deriving
direction from a *moved* tree would flip a sunk producer's RAW edge into a WAR and
hide the violation (matmul reads uninitialised data → NaN).

Recompute-on-demand satisfies this for free: the legality check runs **before**
the move, so the current tree *is* the pre-move program. Directions are derived
from current footprints; the proposed position is tested analytically (the same
half-integer-span technique as `first_backward_edge_for_insertion`, now
coverage-driven). The spec keeps this invariant explicit so it is not lost in the
refactor.

## 2. How footprints are carried on the IR

Today `Dependency` conflates two jobs in one rebuilt-from-scratch graph:

1. **Ordering** — who-runs-before-whom (RAW/WAW/WAR). Used by `analyze()` to bound
   where a block may land (the `(lp, fc]` insertion-gap in `_legal_indices`).
2. **Coverage** — is *this specific move* sound. Bolted on as `CARRY`/`COVER`.

The revamp splits them, because coverage is a **query**, not a stored fact:

- **Footprints become first-class.** A block already carries `reads`/`writes` as
  `BufferRegion`s in loop-var space; the footprint completes that with the
  enclosing-loop extents that bound each free var (the data `_BlockInfo` caches
  today). Footprints live on the block and stay correct as the tree mutates,
  rather than being re-scanned into a fresh `_BlockInfo` set each call.
- **Coverage is not stored.** `CARRY`/`COVER` vanish as edge kinds. When code
  motion proposes a move, region-cover is computed directly from the footprints of
  B and its partners on shared buffers, at B's new position. Not cached; recomputed
  per candidate (cheap — only B's partners, not the whole graph).
- **Ordering is recomputed on demand.** `KernelIR` **loses its `dependency`
  field**. There is no stored dependency object. `analyze()` derives
  producer/consumer from block footprints when it needs them (O(blocks²), trivial
  at ladder scale). Zero drift risk — nothing persistent to invalidate.

This directly closes pain points 1 (heuristic edges gone — coverage computed, not
encoded), 3 (compaction updates footprints; the next coverage query reads them),
and 4 (nothing re-derived globally).

## 3. `code_motion`: one unified transform

Today `ComputeAt` (sink producer) and `ReverseComputeAt` (lift consumer) are two
`Transform` classes that already share `_move`; they differ only in `is_reverse`
(structurally inert) and which dependency face they check (conditions 5a vs 5b in
`compute_at_legality.md`). The ladder calls both simply `# Code motion`.

The revamp folds them into one `CodeMotion` transform in a `code_motion.py`
module:

- `CodeMotionOption(block_nid, target_loop_nid, index)` — no direction flag.
- **Direction is inferred**: if `target_loop_nid` sits in the subtree of a
  *consumer* of B, it is a sink (old ComputeAt); if under a *producer* of B, a lift
  (old ReverseComputeAt). Since `is_reverse` was already structurally inert, this
  only selects which side the coverage query walks — and the one region-cover rule
  (§1) handles both faces identically, so the 5a/5b split disappears.
- `analyze()` enumerates both move kinds (producer-sink and consumer-lift) and
  filters by the single coverage check.
- **Deleted (superseded):** `compute_at.py`, `reverse_compute_at.py`,
  `compute_at_legality.md`, and the 5a/5b condition split.

`_move`'s verbatim same-prefix residual splice is **unchanged** — only the legality
wrapper and class structure collapse. Callers (`examples/kernel_transforms.py`, the
transform tests) swap `ComputeAt`/`ReverseComputeAt` for `CodeMotion`. The manual
ladder (`manual_transforms.py`) is hand-written NKI — the target, not a caller — so
it is untouched.

## 4. `BufferLayout` transform + the compaction side-effect

The blocker my dive found: **codegen physically cannot emit the list-of-tiles
form.** `Buffer` carries only `shape/dtype/location/versions`, and
`codegen/body.py:_emit_alloc` emits exactly one `nl.ndarray` per buffer. The
ladder's `psum_prod = [nl.ndarray((128,1,512),...) for _ in range(16)]` with leading
subscript `psum_prod[i_d1_0][0:128, 0, 0:512]` has no representation. This is the
measured k22 "46.6% vs 90.7%, all buffer representation" gap.

The ladder shows two **distinct** mechanisms; the design keeps them distinct:

### (a) `BufferLayout` — an explicit user transform

Rungs k6, k13, k21, k26 are standalone `# Buffer layout` (no code motion). The
transform converts a buffer between **packed** `nl.ndarray((128, T, F))` and
**list-of-tiles** `[nl.ndarray((128, deg, F)) for _ in range(T)]` — the multi-buffer
lever: list length = liveness-scheduled separately by the allocator; middle dim =
co-resident degree.

**Representation:** add one field `Buffer.num_tiles: int = 1`. Default `1` =
today's single ndarray, byte-identical. `num_tiles > 1` splits the leading
partition-tiles into a Python list of `num_tiles` ndarrays; the renderer emits the
list comprehension and peels the middle-dim index into a leading list subscript.
`versions` (pipelining) and `num_tiles` (allocation granularity) compose — they are
orthogonal levers (degree vs count, per the learnings).

**Invariant:** list length × per-tile middle dim = total tile count
(`num_tiles × deg == num_p_tiles × versions`), so packed and list forms always
describe the same storage.

### (b) Scope-tighten + compaction — an automatic side-effect of code motion

Rungs k12, k16, k20, k25 pair `# Side effect: …compaction` with `# Code motion`.
When a block sinks under a loop, its buffer declaration moves inside that loop
(scope-tighten, today's `place_buffers`) and its shape shrinks to the live slice
(compaction, today's `compact_shapes`). This already works **for packed shapes**;
the revamp:

1. extends compaction to the list form (shrink per-tile shape; adjust `num_tiles`),
   and
2. makes it **update footprints** so the next coverage query (§1) sees the compacted
   region — closing pain point 3 (side effects not tracked).

## 5. Build plan, sequencing, and testing

Rung-by-rung, each step gated on gym-1 before the next (per "verify each rung before
the next / one fix per run"):

1. **`Buffer.num_tiles` + codegen** — list-of-tiles emission. Gate: a hand IR with
   `num_tiles>1` renders byte-exact to the `[nl.ndarray(...) for _ in range(...)]`
   form of k6/k13/k21/k26.
2. **Region-cover legality core** — footprint coverage query + accumulator
   live-range invariant. Gate: existing transform unit tests stay green; it must
   reject everything the old `CARRY`/`COVER` rejected (verified against the
   k16/k17-NaN cases) and no more (over-rejection re-blocks fold-inlining).
3. **`CodeMotion` unification** — fold the two classes, delete old modules, route
   legality through #2. Gate: `kernel_transforms.py` ladder + CPU-sim unchanged.
4. **`BufferLayout` transform** — packed↔list, footprint-aware. Gate: standalone
   Buffer-layout rungs reproduce.
5. **Side-effect tracking** — compaction/scope-tighten update footprints; drop the
   standalone `Dependency` object and `KernelIR.dependency`; `analyze` recomputes
   ordering on demand. **Blast radius (verified by grep):** EVERY transform ends
   `apply` with `new_ir.dependency = Dependency(new_ir.tree)` —
   `split`/`reorder`/`fuse`/`software_pipeline`/`rfactor` + the two code-motion
   classes. All those lines are removed. `software_pipeline` also *reads*
   `ir.dependency.must_precede(...)` and `ir.dependency.info(leaf)` for stage
   legality (step 5 of this list); these repoint to the recompute-on-demand ordering
   helper + footprint accessor. `ir.dump` calls `self.dependency.dump(cache_dir)` — the
   dependency diagram dump is dropped (or re-derived on demand for the diagram only).
6. **RFactor debug + correct** — diagnosis-first; see §6. Task 6.1 (the gym-1
   diagnostic harness across early/mid/late states) starts **early, in parallel** —
   its early+mid cases need no other step. Only the late (list-buffer) case waits on
   step 4.
7. **Full ladder** — a transform-driven ladder reproducing all of
   `kernel_0 … kernel_27` byte-exact + CPU-sim clean on gym-1.

**Deletions (supersede, not shim):** `compute_at.py`, `reverse_compute_at.py`,
`compute_at_legality.md`, the `CARRY`/`COVER`/`skip_cover_loops` machinery in
`dependency.py`, and `KernelIR.dependency`. Each is checked against "is this the sole
record of shipped behavior" before removal.

**Validation loop (gym-1 via `transport/ssh_host.sh`):** CPU-sim
(`simulate_fp32`) per rung; byte-exact ladder (rendered IR == hand kernel,
AST-canonical oracle); HW MFU (`autotune.runner.profile`); `pytest`. The controller
owns all remote runs. No TVM oracle.

**Risk:** region-cover must reject *exactly* the set the carry/cover heuristics
rejected — no more (over-rejection re-blocks fold-inlining; B1/B2 history), no less
(under-rejection → NaN rungs). The gym-1 ladder is the arbiter.

## 6. RFactor: debug + correct (first-class, diagnosis-first)

User-flagged as an important goal, twice. The precise defect is **not yet known**
("I highly suspect the current implementation is not complete/buggy"), so this
section is **diagnosis-first**: it does not assert a bug-and-fix it cannot yet prove.
The Problem-section evidence establishes only that RFactor is *under-verified* — it
passes for one early, packed config and has no test at the ladder states that matter.

### Structural contract (the BEFORE → AFTER template)

RFactor is defined by a fixed transformation of **structural roles relative to the
factored loop `ko`**, NOT by buffer geometry:

```
BEFORE (one-stage):                  AFTER (two-stage):
    init_one_stage()                     init_two_stage_0()
    for ko in range(ko_trip):            for ko in range(ko_trip):
        for ki in range(ki_trip):            init_two_stage_1()
            run_op()                         for ki in range(ki_trip):
    drain_one_stage()                            run_op()
                                             drain_two_stage_0()
                                         drain_two_stage_1()
```

The three input roles, located by position relative to `ko`:
- **`init_one_stage`** — the accumulator-init block(s) *before* `ko` (canonical:
  `memset(acc1)`, acc1 = the PSUM accumulator the run-op writes).
- **`run_op`** — the reduction body *inside* `ki` (the matmul).
- **`drain_one_stage`** — the accumulator-consumer block(s) *after* `ko` (canonical:
  `tensor_copy(acc1 → acc2)`, acc2 = the SBUF output buffer).

The fixed rewrite recipe over those roles (acc2 = cross-`ko` SBUF accumulator,
staging = transient SBUF for the PSUM→SBUF copy the fold needs):
- **`init_two_stage_0`** ← `init_one_stage` **retargeted** to zero acc2, kept before
  `ko`.
- **`init_two_stage_1`** ← **new** per-`ko` `memset(acc1)`, spliced as `ko`'s first
  child (before the `ki` nest). The run-op's reduction axis flips ACCUMULATION →
  PARALLEL (each `ko` is an independent partial; `ki` HW-accumulates in acc1).
- **`drain_two_stage_0`** ← **new** per-`ko` fold spliced as `ko`'s last children:
  `tensor_copy(acc1 → staging)` then `tensor_tensor(acc2 = combiner(acc2, staging))`.
  Carries `ko` as ACCUMULATION (the closing cross-`ko` reduction on acc2).
- **`drain_two_stage_1`** ← the residual after `ko`; **None** in the matmul case
  (acc2 already holds the stored output; the original drain copy is absorbed into the
  fold).

Mapped to the manual ladder's `Apply RFactor` rung (k26 → k27): acc1 = `psum_prod`,
acc2 = `sbuf_prod`, staging = `sbuf_rfactor`, combiner = `add`, `drain_two_stage_1`
empty.

**Why this matters for the rewrite:** because the recipe keys off *roles located by
position relative to `ko`*, not buffer shapes, the SAME recipe must apply whether
buffers are packed or list-of-tiles (§4) and whether the nest is shallow (early) or
deep (late). The diagnosis (6.1) and correction (6.2) are framed against this
contract: **(1) role location** — does RFactor correctly find init/run/drain by
position? — and **(2) stage emission** — does it derive each new block's geometry
from the *located* roles rather than hardcoded constants?

### Task 6.1 — Diagnostic harness (gym-1) BEFORE any change

Extend the `manual_transforms.py` / `kernel_transforms.py` pattern to drive RFactor
across the states the ladder actually visits, render + CPU-sim each, and record what
breaks. Concretely, apply `RFactor(ko)` to:

- the **early packed** state (today's `split_k_ir()`) — the one known-good config;
- a **mid-ladder** state (M tiled `i_d1_0×i_d1_1`, N split, packed buffers);
- the **late k26** state (list buffers via §4 `num_tiles`, loads sunk) — the manual
  ladder's actual `Apply RFactor` input.

The output is a concrete failure table (exception / wrong-nest / NaN / byte-mismatch
per state), captured in the cache for inspection. **This replaces my earlier
inferred bug list** — the harness reports the real defect; the design does not
pre-judge it.

**RESULT (gym-1, 2026-06-25 — harness `examples/rfactor_states.py`, plan
`2026-06-25-rfactor-debug-and-correct.md`):** RFactor is **correct on both packed
states** — `early_packed` and `mid_packed` (M-tiled) each `sim=pass` (1.373e-04), with
`mid_packed` rendering a proper two-stage form. The plan's hypothesis (M-tiling breaks
the hardcoded `m_var`/`m_tiles`) was **disproved**: on a packed `(128,16,2048)` buffer
the drain sweeps all 16 tiles in one flat loop regardless of matmul M-tiling. **The
defect is confined to the LATE-LIST state** (`num_tiles>1`, per-tile free 512), which
is gated behind §4 BufferLayout and not yet representable. So §6.2 (below) is NOT done
in the RFactor plan — it is **folded into the BufferLayout plan**, where the late-list
state first exists. The harness is retained as a permanent regression guard.

### Task 6.2 — Correct from the diagnosis (FOLDED INTO THE BUFFERLAYOUT PLAN)

6.1 showed the only defect is the late-list state, which needs §4 `num_tiles` buffers
to exist first. So this correction lands **in the BufferLayout plan**, not the RFactor
plan. The confirmed scope: make `_emit_rmw` **list-buffer aware** — derive `free_extent`
from the matmul `dst` region's per-tile free width (512), not the packed buffer shape
(2048); handle `num_tiles>1` on `psum`; stage `sbuf_rfactor` at the per-tile shape; and
emit the drain as the per-tile `(128,1,512)` list form `kernel_27` contains rather than
a flat 16-tile sweep. The packed-state geometry (`m_var`/`m_tiles` flat sweep) is
**verified correct and unchanged** — only the list path is new. The
`ir.dependency = Dependency(tree)` rebuild on `_emit_rmw`'s last line is still removed
once §2 drops the stored dependency object. `_rfactorable` is unchanged.

### Task 6.3 — Refresh the stale suite

`test_rfactor.py` docstrings describe the abandoned multi-slot "rf-block/wb-block"
form; update them to the fused `sbuf_rfactor`/`tensor_tensor` reality and add the
late-application cases the suite lacks (the gap that hid the defect).

### Gate

RFactor must apply correctly across the harness states:
- LATE (k26→k27): reproduces `kernel_27` byte-exact + CPU-sim clean.
- EARLY (today's `split_k_ir`): unchanged byte-exact vs `kernel_rfactor_ko.py`,
  existing `test_rfactor.py` green.

All gated on gym-1. RFactor is the one transform whose *internals* change (vs §3's
wrapper-only collapse), so it carries its own before/after byte-exact fixtures.

### Sequencing note

RFactor-late operates on §4 `num_tiles` list buffers, so Task 6.1's late case cannot
fully run until BufferLayout (step 4) lands. But 6.1's **early + mid** cases need
none of that and run first — so RFactor diagnosis starts immediately and is not
gated behind the rest of the work. If 6.1 shows the defect is independent of list
buffers, the fix can land early against packed states and only the list-aware part
waits on BufferLayout.

## Open questions / explicitly deferred

- **SoftwarePipeline interaction with `num_tiles`.** `versions` and `num_tiles`
  compose by construction (§4 invariant), but the manual ladder does not exercise a
  pipelined list buffer. Not in scope; flagged so a later pipeline rung over a list
  buffer is a known untested combination.
- **`SoftwarePipeline` dependency repoint** (cross-reference to §5 step 5, called
  out here because it is the most error-prone repoint): it is the one transform that
  *reads* `ir.dependency` (`must_precede`, `info(leaf)`), not only rebuilds it, so its
  two reads must repoint to the recompute-on-demand ordering helper + footprint
  accessor, gated by its own `test_software_pipeline.py` regression.
