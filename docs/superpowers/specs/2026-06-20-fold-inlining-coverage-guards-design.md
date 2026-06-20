# Fold-inlining: refine two coverage guards to reach kernel_target

**Status:** design (2026-06-20). Follows
`2026-06-20-fold-inlining-blocker-analysis.md`, which characterized the
~45pp MFU gap (k15 46.08% vs kernel_target 90.81%) as structural. This spec
designs the two legality-guard refinements that close it, with the gap
re-confirmed by fresh gym-1 probes this session.

## Goal

Drive canonical `f_nkigym` → a `kernel_target`-equivalent (the fold inlined
into the matmul's innermost M loop) using the EXISTING transform set
(Split, Reorder, RFactor, ComputeAt/ReverseComputeAt). No new transform
types. The blockers are two over-conservative legality predicates, not
missing transforms.

## Evidence (probed this session, gym-1)

A step-by-step probe drove canonical through the N-outermost nest:

- **k0–k9 ALL sim-clean (~1.4e-4):** `Split(K→ko2,ki8)` → `Split(M→Mo4,Mi4)`
  → `Reorder×6` (bubble N=`i_d2_0` outermost, sink ki=`i_d0_1` innermost; ki
  descends by ADJACENT swaps — Reorder rejects non-adjacent) → reaches nest
  `N > ko > Mo > Mi > ki` → **`RFactor(ko)` applies sim-clean there** (k9
  max_abs 1.37e-4). This overturns the stale "RFactor sim-FAILs after
  Split(M)" claim (that was pre-fused-RFactor). Loop interchange + RFactor
  are NOT the gap.
- At k9 the memset/drain/fold sit as a separate full-M, full-N `[N,M]` sweep
  after the matmul's ko-body — correct but un-inlined.

The two co-location moves that would inline them are both REJECTED:

| Move | Transform | Rejecting check | Message |
|---|---|---|---|
| fold → matmul Mi | ReverseComputeAt | `_check_no_reduction_axis_covered` (`_code_motion.py:127`) | "would cover reduction axis 'd0' (ACCUMULATION) with enclosing loops [('i_d0_0', 2)]" |
| copy → matmul Mi | ReverseComputeAt → Dependency | frozen COVER edge via `first_backward_edge_for_insertion` (`dependency.py:288`) | "reorders dependency edge 21→30 backward" |

A follow-up probe confirmed: splitting the copy's region on N (`d2`)
narrows it correctly to `[*, i_d2_1*512 : +512]`, but co-location is STILL
rejected with the same edge — because the copy then owns its OWN inner
`i_d2_1`, independent of the matmul's shared outer `i_d2_0`, so the COVER
edge persists. Narrowing the region is necessary but not sufficient; the
drain must INHERIT the matmul's `i_d2_0`, which the move's region-regen does
— but the guard judges the pre-move (frozen) edge.

## Root cause (unifying both barriers)

Both guards reject moves that the move ITSELF makes legal. `_move`'s
region-regen (`_domain_solve.regen_and_rebind`) rebinds a covered dim to the
target's loops; `solve_iter_domains` returns this as
`solved[dim].target_loops`. But both guards evaluate the PRE-move structure
and miss that the move will re-cover the dim. Fix: each guard consults
`solved` (already computed in `_check_move_realizable`) to recognize the
re-covered dim.

## Barrier 2 — COVER-aware backward-edge check (`ir/dependency.py`)

**Defect:** `_first_backward` (`dependency.py:204`) iterates `self.graph.edges()`
without edge data and ranks every incident edge by span. A frozen
`COVER: L → moved` edge (added by `_add_coverage_edges` when the consumer
reads a buffer region wider than the producer writes per `L`-iteration) is
treated as a hard hazard. When the move binds `moved`'s covered dim to `L`,
that edge is dissolved by the move — but the span check fires on the frozen
edge → spurious backward → reject.

**Fix:** make the backward-edge query COVER-aware.

- `_first_backward` reads `self.graph.edges(data=True)` and inspects
  `attrs["kind"]`.
- For a `COVER` edge `L → moved` (where `moved` is the moved leaf and `L` a
  producer-loop nid), SKIP it when `solved` shows the move binds the moved
  block's covered dim to `L` — i.e. `L`'s `(loop_var, extent)` is in
  `solved[dim].target_loops` for some covered `dim` of the moved block.
- All other edge kinds (RAW, WAW, WAR, CARRY, and COVER edges from loops NOT
  in the moved block's `target_loops`) stay frozen and are checked exactly as
  today.

**Data flow:** `first_backward_edge_for_insertion` (the
no-mutation insertion query, `dependency.py:124`) and `_first_backward` gain
the covered-loop set as a parameter. `_check_move_preserves_dependencies`
(`_code_motion.py:160`) already runs `_check_move_realizable` first, which
computes `solved`; thread the `{loop_var → covered}` set (or `solved`
directly) from there into the dependency query. This is a signature addition
across the two files; no behavior change to the span math itself.

**Why narrow:** the skip triggers ONLY when `L` is literally a target loop
the move re-covers for the moved block. A COVER edge from an unrelated loop,
or for a different buffer, is untouched. The discriminator cannot
accidentally drop a true producer→consumer hazard because RAW/WAW/WAR are a
different `kind` and never skipped.

**Risk:** LOW. B2 governs PARALLEL ops (memset, tensor_copy). A wrong
loosening yields a stale READ (caught by sim), never silent accumulation
corruption.

## Barrier 1 — init-domination discriminator (`transforms/_code_motion.py`)

**Defect:** `_check_no_reduction_axis_covered` (`_code_motion.py:127`)
rejects covering ANY ACCUMULATION axis. But covering `ko` is SAFE when the
moved block's own init dominates the covering loop (the fold accumulates
across its ENCLOSING `ko`, with `sbuf_prod` memset hoisted outside — the
kernel_target structure), and UNSAFE only when the covering loop is foreign
(a producer's prefetch K merely sharing the `i_d0_0` name → init no longer
dominates → NaN).

**Fix:** refine the guard to consult the init-domination signal already in
the dependency graph. For each covered ACCUMULATION dim:

1. Identify the moved block's RMW/accumulator buffer via `RMW_OPERANDS` (the
   same gate `_carry_loops_of_leaf` uses to find the carried buffer — the
   fold's `sbuf_prod` in `data1`/`dst`).
2. Check whether that buffer's init-writer (the memset) sits OUTSIDE the
   covering target loop and carries into it — i.e. the existing
   `memset → ko` CARRY edge (`_add_carry_edges`, `dependency.py:265`) is
   present for that loop.
3. **Carry edge present →** init dominates → covering is the safe
   enclosing-reduction case → ALLOW. **Absent →** foreign loop → REJECT
   (current behavior preserved).

**Why this discriminator:** it is the SAME init-domination invariant the
carry edge was built to express — the guard consults the graph signal rather
than inventing a new criterion. A foreign `i_d0_0` has no carry edge into the
fold's accumulator, so it still rejects.

**Risk:** HIGHER — this is the NaN-capable guard. A wrong loosening silently
corrupts (sim NaN, or worse, last-write-wins masking). Mitigations in the
validation plan are mandatory, not optional.

## The fold-inlining ladder (extends probed-clean k0–k9)

These rungs are the NEW **N-outermost** ladder (a distinct path from the
shipped **ko-outermost** k0–k15 in `examples/kernel_transforms.py`; the
rung numbers below are local to this path, not that file's). It replaces the
shipped ladder once both guards are refined.

```
k0–k9   Split(K)→Split(M)→Reorder×6→RFactor(ko)   [PROVEN sim-clean]
─── B2 enables: ───
k10  Split(memset psum, d2→4×512)                 narrow init to per-N-tile
k11  Split(copy, d2→4×512)                          narrow drain to per-N-tile
k12  ReverseComputeAt(memset → matmul i_d2_0)      co-locate init (inherits N-tile)
k13  ReverseComputeAt(copy → matmul i_d2_0)        co-locate drain → PSUM (128,1,512)
─── B1 enables: ───
k14  Split(fold, d2→4×512) + Split(fold, M→4×4)    match matmul tile prefix
k15  ReverseComputeAt(fold → matmul i_d1_1)        inline fold under innermost M
─── then: ───
k16  ComputeAt(store → i_d2_0)                       sink store under N
```

Exact rung order/locators are pinned empirically during implementation (the
probe harness is the tool); co-location under the SHARED OUTER `i_d2_0` is the
key difference from the shipped k15 ladder's N-innermost approach, and B2 is
what makes it legal.

## Component boundaries

- `ir/dependency.py` — `_first_backward` + `first_backward_edge_for_insertion`:
  COVER-skip, `solved`-driven. Independently testable: construct a tree with a
  frozen COVER edge, assert the query skips it iff the loop is a target loop.
- `transforms/_code_motion.py` — `_check_no_reduction_axis_covered`:
  carry-edge discriminator. Already receives `ir` + `solved`. Independently
  testable: fold-block move (carry present → allow) vs foreign-loop move
  (absent → reject).
- Neither touches transform MECHANICS (`_move`, `regen_and_rebind`) — only the
  two legality predicates. Transform-mechanic tests are unaffected.

## Validation plan

Per the repo's verification learnings (golden-verdict gate; sim is a weaker
oracle than dep-order; loud failures only):

1. **Golden-verdict capture FIRST (before any code change):** record the
   verdicts of known-ILLEGAL moves — foreign-loop ko-coverage, memset sunk
   into K, drain sunk into K, consumer-before-producer. Each MUST stay
   rejected after the change. A loosening that flips any `illegal→ok` while
   sim stays green is the failure mode to catch.
2. **Per-rung gym-1 sim:** every new rung k10–k16 renders + sim-PASS
   (~1.4e-4). Gate EACH B1 candidate individually (the guard prevents NaN; a
   wrong loosening corrupts).
3. **Regression:** all existing k0–k15 in `examples/kernel_transforms.py` +
   the transform unit suite stay green (run via `transport/remote_pytest.sh`,
   `PYTHONPATH=.:nkigym/src:autotune/src`).
4. **HW MFU:** measure k13 (PSUM shrunk, fold still a sweep) and k15/k16 (fold
   inlined) on gym-1; expect movement 46% → toward kernel_target's 90%.
5. **Stop condition:** if any guard loosening admits a previously-rejected
   illegal move, HALT and reassess — do not chase MFU through a broken guard.

## Out of scope (YAGNI / independent work)

- Buffer-list multi-buffer encoding (the `[nl.ndarray((128,1,512)) ...]` lever).
- Codegen-time declaration-position pass (buffer scope tightening — behaviorally
  inert on HW; the Neuron allocator does liveness placement).
- Per-Mo lhs_T reload `(128,1,512)` tuning (blocker doc step 4; secondary, after
  the fold inlines).
