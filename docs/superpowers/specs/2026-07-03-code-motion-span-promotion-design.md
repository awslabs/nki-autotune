# Unified CodeMotion + legality by span-promotion — design

**Date:** 2026-07-03
**Status:** design (brainstorming; supersedes the *barrier-model* portions of
`2026-06-25-dependency-model-from-ladder-design.md` — §"The model (concrete)",
§"Pressure test", and the init/drain-bracket barrier. The ladder walk, the
access-granularity decisions, and the frozen-direction invariant from that doc
SURVIVE and are re-stated here.)

## Two coupled changes in one plan

1. **Merge `ComputeAt` + `ReverseComputeAt` into one `CodeMotion` transform.** The two
   faces are already one operation: `_move` and `_check_move_preserves_dependencies`
   take an `is_reverse` flag but NEVER branch on it (it is documented "structurally
   inert"), and the two `analyze()` methods are byte-identical modulo the option class
   name. `_legal_indices` already uses `producers` AND `consumers` symmetrically (the
   insertion gap `(lp, fc]` is bounded below by the last producer, above by the first
   consumer) — producer-sink vs consumer-lift is EMERGENT from the dependency graph,
   not a caller-chosen mode. So the merge is a straight union, not a reconciliation.
2. **Replace the barrier/CARRY/COVER legality machinery with span-promotion** (below).

These are coupled because the merge removes the only real asymmetry between the faces
— the output-block guard — and that removal is what makes the k11→k12 store-sink legal
under one transform (see "Output-block guard" below).

## What changes vs the June-25 doc

The June-25 doc concluded with a **barrier model**: an accumulator's `rmw` access,
per enclosing brackets-only loop, defines an init→L→drain bracket that acts as a
barrier (no foreign writer/reader inside L), and a brackets-only rmw loop with no
closing init is illegal replication. That model needed per-accumulator init/drain
bookkeeping and had, by its own account, flip-flopped three times and required a
pressure test to find two gaps.

This design replaces that barrier model with a **single span-promotion rule** layered
on the *already-shipped* backward-edge span test. It is strictly less machinery: no
init/drain bracket location, no axis-role consult in the ordering check, no
CARRY/COVER edge kinds, no `skip_cover_loops` escape hatch. The correctness argument
is one law (producer-before-all-consumers) plus one place the symbolic test is fooled
(a loop-carried region), fixed in one place (`span()`).

**No unrolling.** An earlier iteration of this design proposed an unrolled concrete
oracle as test-time ground truth. That is dropped: the check is purely symbolic, and
validation rests on the existing pinned-verdict tests + the gym-1 ladder (see
"Validation"). There is no new oracle module and no O(trip-product) enumeration
anywhere.

## Scope

Two things change, both in the code-motion concern:

- **The transform surface:** `ComputeAt` + `ReverseComputeAt` → one `CodeMotion`
  (`CodeMotionOption(block_nid, target_loop_nid, index)`, no direction flag).
- **The legality machinery:** the `CARRY`/`COVER` edge families, `skip_cover_loops`,
  and the reduction-axis-covered guard → the span-promotion rule.

Out of scope / preserved:

- `KernelIR.dependency` is **NOT dropped**. SoftwarePipeline reads it
  (`must_precede`, `info`), and `CodeMotion.analyze` reads `producers`/`consumers` for
  slot enumeration (`_legal_indices`).
- The base RAW/WAW/WAR precedence relation, `info`, and `touches_by_tensor` STAY.
- Split / Reorder / Fuse / RFactor internals, `_move`'s verbatim splice, the `arith`
  substrate, the `nki` ops — untouched.
- The `CodeMotion` merge is NOT a new capability: it is the union of two existing
  transforms. The only behavioral change from the merge alone is dropping the
  output-block guard (justified below); everything else is a rename + de-duplication.

## The one law

On the fully-executed program there is a single ordering law:

> **Every write of a buffer region executes before every read/rmw of the overlapping
> region that originally followed it** (producer-before-all-consumers), for each pair
> whose original order is frozen from the pre-move program.

Everything the old CARRY/COVER/barrier machinery encoded is this one law, observed at
different loop-span shapes. Code motion is legal iff, after the proposed splice, no
frozen producer→consumer pair flips backward.

## Where the symbolic test is fooled (the only hard case)

The shipped check (`Dependency.first_backward_edge_for_insertion`) already tests this
law via a **preorder span**: each node has a span `[start, end]` over the tree (a leaf
is a point; a loop spans its whole subtree), and an edge `a → b` ("a before b") is
satisfied iff `span(a).end < span(b).start`. Directions come from the pre-move graph;
spans are read at the proposed position.

This is correct **except** when a buffer region is **carried across a loop** — live
across every iteration of an enclosing loop L because it is read-modified-written
loop-invariantly over L (matmul's `psum` accumulator across K; the two-stage fold's
`sbuf_prod` across `ko`). Collapsing such an access to a single leaf-point loses that
its live range is the *whole* of L. Two concrete failures the leaf-point test misses:

- **memset sunk INTO K** (the historical NaN bug). Frozen edge `memset → matmul` on
  `psum`. As leaf-points, memset is lexically the first child under K, so
  `memset.end < matmul.start` holds — "legal". But `psum` is carried across K, so the
  matmul's live range is all of K; a memset inside K re-zeros the partial sum every
  iteration. The move must be rejected.
- **drain/store sunk INTO `ko`** (coverage). The fold RMWs `sbuf_prod` across `ko`; a
  store that reads the whole `sbuf_prod` sunk inside `ko` reads a partial sum. Per
  iteration `fold_j → store_j` is satisfied, yet the value is wrong.

Both are the same defect: an access to a carried region, sitting inside the loop that
carries it, compared as a point instead of against the carried live range.

## The fix: span-promotion

Add ONE rule to `span(nid)`. An access's span grows from its leaf-point to loop **L**'s
full span (the span L already has as an enclosing loop) iff all three hold:

1. **L encloses the access** on the *proposed* tree (L is a ForNode ancestor of the
   access at its evaluated position).
2. **The access is invariant across L** — L's loop-var does not appear in any axis
   offset (`lo`) of the access's region. (An access L *indexes* — its offset contains
   L's var — moves in lockstep with its partner and needs no promotion; that is the
   base point/point case.)
3. **The access's tensor is carried across L** (definition below).

After promoting both endpoints of a frozen edge as applicable, apply the unchanged
test `span(P).end < span(C).start`; a violated pair is the first backward edge → reject.

**The moved leaf's enclosing loops are the TARGET's, not its old parent's.** In
`first_backward_edge_for_insertion` the moved leaf is scored at the proposed position:
its enclosing ForNodes for condition 1 are `target_loop_nid` and the target's ancestor
ForNodes (`self._tree.ancestors(target_loop_nid)`), not the loops it sat under before
the move. This is exactly what makes the memset-into-K case promote: after the proposed
splice the memset's enclosing loops INCLUDE the matmul's K loop, so — memset invariant
across K, `psum` carried across K — the memset promotes to K-span. A partner that is
not the moved leaf keeps its own tree-position enclosers (with the shipped
exclude-moved-subtree / grow-enclosers adjustments already in that method).

### "Carried across L" (the load-bearing definition)

> Tensor `T` is **carried across loop L** iff there exists an `rmw` access of `T`
> (an operand in `RMW_OPERANDS`: matmul's `psum`, `tensor_tensor`'s accumulator) that
> is **invariant across L** (L's var absent from its offset) and is **enclosed by L**
> on the proposed tree.

This is a yes/no property of `(T, L)`, read from the tree on demand. It deliberately
distinguishes:

- **carried** — matmul `psum` across K, fold `sbuf_prod` across `ko`: RMW'd
  loop-invariantly. Foreign invariant accesses to `T` inside L are promoted → the
  init (memset) and drain (store) are forced outside L.
- **NOT carried, pure read** — `sbuf_lhs_T` read (never RMW'd) by the matmul. Sinking
  the `lhs_T` load under the N loop it does not index is a legal per-N *reload*: the
  load stays a point, ordered before each matmul, no promotion. (This is the
  duplication case; its *replication-of-an-accumulation* sibling is rejected
  structurally, see below.)
- **NOT carried, write-only staging** — `sbuf_rfactor` (the copy writes it, the fold
  reads it, nobody RMWs it). Not carried → copy-before-fold stays plain point
  ordering. No spurious promotion.

Using **rmw** (not "written invariantly") is what keeps the legal reload and the
staging buffer from being falsely promoted. This was confirmed as the intended
definition during brainstorming.

## Ordering vs. replication: two different questions

Span-promotion answers an **ordering** question: does a frozen producer→consumer pair
flip backward at the proposed position? Three of the June-25 "four facts" (#1 RAW,
#2 init-domination, #4 coverage) are all this one ordering question at different span
shapes, and all are handled by span-promotion.

Fact #3 — **replication** — is NOT an ordering question. It is the move *duplicating*
a block across a loop the block does not bind (a target loop-var absent from the
block's own nest). Duplicating a pure producer is a benign reload; duplicating an
ACCUMULATION re-runs a reduction into an un-reinitialised accumulator (garbled). This
is about the splice creating new copies, not about reordering existing accesses, so it
stays a **structural guard**: `_check_no_reduction_replicated` (kept unchanged). Its
error string `"replicates a reduction"` is what
`test_compute_at_rejects_replicating_reduction_over_untiled_output_dim` matches.

## Output-block guard: DROPPED (the merge's one behavioral change)

`ComputeAt` today forbids relocating the block whose ISA leaf writes the return tensor
(`compute_at.py`: `region.tensor == ir.return_name` → raise). `ReverseComputeAt` has
**no such guard**. This asymmetry is the only real behavioral difference between the
faces, and it must be dropped in the merge — not kept — because:

- **k11→k12 sinks the output store.** In `manual_transforms.py`, k11→k12 relocates
  `dma_copy(sbuf_prod → hbm_out)` from its own `for i_d2_0: for i_d1_0` loop INTO the
  matmul's `i_d2_0` body (right after the drain); its `i_d2_0` merges with the
  matmul's, its `i_d1_0` stays a residual loop. `hbm_out` IS the return tensor. This
  rung is legal, sim-clean, and REQUIRED.
- It works today **because** it is a `ReverseComputeAt` (the guard-less face). A merged
  `CodeMotion` applying the guard universally would forbid k11→k12.
- Dropping the guard is safe: the store is governed by dependency + span-promotion like
  any other block. Its legal placements are exactly the ordering-preserving ones — the
  drain writes the `sbuf_prod` slice in the SAME `i_d2_0` iteration the store reads it
  (RAW satisfied); `sbuf_prod` is NOT carried (pure write by drain, pure read by store,
  no rmw) so no promotion, just point ordering. Sinking the store above its drain, or
  into a loop whose slice is not yet produced, flips a RAW edge backward → correctly
  rejected. The guard was never what protected that.

This also matches the memory note listing "store-sink LEGALITY blocker (output block
pinned at root)" as future work: the merge + guard-drop is what unblocks it. No new
capability is added — the store-sink was already reachable via `ReverseComputeAt`;
the merge just stops the redundant guard from being re-imposed.

## Worked verdicts (the pinned cases)

Directions frozen from the pre-move program; spans read at the proposed position.

- **memset → matmul on `psum`, K = `i_d0_0`** (`psum` carried across K):
  - memset OUTSIDE K (legal): memset not enclosed by K → point before K →
    `pt < K.start` ✓ → **allow**.
  - memset sunk INTO K, first child (NaN bug): both promoted to K-span →
    `K.end < K.start` ✗ → **reject**.
- **fold covering its OWN `ko`** (`test_reverse_compute_at_allows_fold_covering_its_own_ko`):
  the fold's self-dependence across its own `ko` is not a frozen edge (a block never
  reorders its own iterations; Reorder's SEQUENTIAL guard blocks the only transform
  that could). The foreign edges are `memset(sbuf_prod) → fold` and `fold → store`;
  both remain correctly ordered under promotion → the fold covering its own `ko` is
  **allowed**.
- **store → (reads whole `sbuf_prod`) sunk into `ko`** (coverage, fact #4): frozen edge
  `fold → store` on `sbuf_prod` (`sbuf_prod` carried across `ko`). The fold rmw's
  `sbuf_prod` invariantly across `ko`, so its span promotes to the whole ko-loop.
  - store sunk INSIDE `ko`: `fold.end (= ko.end) < store.start` is false (the store
    sits within ko, so `store.start > ko.start`; and even without promoting the store,
    ko.end is not `< ` a point inside ko) → backward → **reject**.
  - store left OUTSIDE `ko`: a point after ko → `fold.end (= ko.end) < store.start`
    holds → **allow**.
- **matmul replicated under an untiled output dim**
  (`test_compute_at_rejects_replicating_reduction_over_untiled_output_dim`): rejected
  by `_check_no_reduction_replicated` (structural), not by span-promotion. Verdict
  preserved.
- **matmul sunk under a foreign loop covering its K axis**
  (`test_compute_at_rejects_covering_matmul_reduction_axis`): must still **reject**.
  ⚠️ The deleted `_check_no_reduction_axis_covered` emitted the "reduction axis"
  message this test's `match=` regex names first. After the change the rejection must
  arrive via a surviving path. The backward-edge span message
  ("`reorders dependency edge …`") matches the regex's `reorder` alternative; the
  prefix message ("`… Split / Reorder …`") does NOT (case-sensitive `re.search`).
  **Which path fires is a gym-1 finding**, and the `match=` regex is re-pointed to the
  surviving message if needed (a preserve-the-verdict rewrite — see the test-slimming
  note). Do not assume from a hand-trace which guard rejects.
- **memset sink across the block wall, sims correct**
  (`test_compute_at_memset_sink_across_block_wall_sims_correct`): the memset's M is
  covered by the enclosing matmul M loop (prefix merge), not replicated; span
  ordering on `psum`/`sbuf_prod` holds → **allow** + sim-clean.
- **store sunk into the matmul's `i_d2_0` (k11→k12)**: frozen edge
  `drain → store` on `sbuf_prod`. `sbuf_prod` is written by the drain and read by the
  store, neither rmw → NOT carried → no promotion. Per `i_d2_0` iteration the drain
  writes the slice the store reads, so `drain.end < store.start` holds at the sunk
  position → **allow**. (The old output-block guard would have rejected this outright;
  dropping it lets dependency ordering decide, and ordering permits it.)

## Concrete changes

### `nkigym/src/nkigym/ir/dependency.py`

- **Add** span-promotion to the `span()` closures in `first_backward_edge` and
  `first_backward_edge_for_insertion`. `span(nid)` first computes the leaf-point (or
  subtree) position exactly as today, then, for each enclosing ForNode L on the
  evaluated tree, if the access is invariant across L (L's var absent from every
  region `lo`) and its tensor is carried across L (an `rmw` of that tensor, invariant
  across L, enclosed by L), widen the returned span to include L's full span. The
  carried/invariance predicates are computed from `_BlockInfo.read_regions` /
  `write_regions` + `RMW_OPERANDS` + `to_affine` on the region `lo`s — the same
  primitives the deleted helpers used, applied on demand inside `span()` rather than
  materialised as edges.
- **Delete** `_add_carry_edges`, `_add_coverage_edges`, `_carry_loops_of_leaf`,
  `_tiled_write_loops_of_leaf`, `_reads_independently_of_loop`, and the
  `skip_cover_loops` parameter on `first_backward_edge` /
  `first_backward_edge_for_insertion` / `_first_backward` (with its COVER-skip branch).
  `_build` ends after the base RAW/WAW/WAR hazard walk — it no longer calls
  `_add_carry_edges`.
- **Keep** `_build`'s base hazard walk, `info`, `touches_by_tensor`, `must_precede`,
  `producers`/`consumers`, the preorder-span technique, and the frozen-direction
  contract.

### Transform surface: merge into `code_motion.py`

- **Rename** `transforms/_code_motion.py` → `transforms/code_motion.py` (it now carries
  a public export). It keeps the shared `_move`, `_check_same_loop_prefix`,
  `_check_no_reduction_replicated`, `_check_move_preserves_dependencies`, and gains the
  public `CodeMotion` Transform class + `CodeMotionOption` dataclass.
- **`CodeMotionOption(block_nid, target_loop_nid, index)`** — no `is_reverse`.
- **`CodeMotion.apply`** = re-check legality, deep-copy, `_move` (drop the `is_reverse`
  parameter throughout), `place_buffers`, `compact_shapes`, rebuild `Dependency`.
- **`CodeMotion.analyze`** = the shared enumeration (the byte-identical body from the
  two faces, emitting `CodeMotionOption`).
- **`CodeMotion._check_legality`** = the structural checks (target in graph, target is
  a ForNode, block in graph, target not a descendant of the block) + span-promotion
  ordering via `_check_move_preserves_dependencies`. **No output-block guard.**
- **Delete** `transforms/compute_at.py` and `transforms/reverse_compute_at.py`.
- **`transforms/__init__.py`**: drop `ComputeAt`/`ComputeAtOption`/`ReverseComputeAt`/
  `ReverseComputeAtOption` exports; add `CodeMotion`/`CodeMotionOption`.
- **Legality doc**: rename `compute_at_legality.md` → `code_motion_legality.md` and
  update it (it is already partly stale — references `_check_move_realizable`, which
  the shipped code replaced with `_check_same_loop_prefix`; fix while renaming).

### Legality machinery in `code_motion.py`

- **Delete** `_own_carry_loop_nids`, `_check_no_reduction_axis_covered`, `is_reverse`
  (throughout), and the `skip_cover_loops` construction in
  `_check_move_preserves_dependencies`. The check calls
  `first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)` with no
  `skip_cover_loops`.
- **Keep** `_check_same_loop_prefix` (structural prefix merge) and
  `_check_no_reduction_replicated` (structural fact #3 guard). `_check_same_loop_prefix`
  currently calls `_check_no_reduction_axis_covered` (line ~159) — that call is
  removed; the reduction-covering verdict is now delivered by span-promotion in the
  ordering check.

### Callers to update

- **`examples/kernel_transforms.py`**: the 7 `ComputeAtOption`/`ReverseComputeAtOption`
  call sites → `CodeMotionOption` (same `block_nid`/`target_loop_nid`/`index`
  arguments; drop nothing else — the moves are unchanged).
- **`examples/matmul_lhsT_rhs.py`**: same rename where it constructs the options.
- **Test files** (`test/transforms/test_compute_at.py`,
  `test_reverse_compute_at.py`, `test_code_motion.py`, and the `_fixtures.py` /
  `_pipeline_fixtures.py` / `test_reorder.py` / `test_split.py` uses): update imports
  and option construction to `CodeMotion`/`CodeMotionOption`. `test_compute_at.py` and
  `test_reverse_compute_at.py` fold into `test_code_motion.py` (their distinct-face
  premise is gone); every assertion's verdict is preserved (test-slimming rule).

### Direction stays frozen (unchanged invariant)

Directions come from `ir.dependency` built on the pre-move program; only spans read the
proposed tree. Re-deriving directions on the moved tree would flip a sunk producer's
RAW into WAR and hide the violation (matmul reads uninitialised data → NaN). This is
the one invariant carried verbatim from the shipped model; span-promotion does not
touch it.

## Migration hazard (verified by grep; resolved on gym-1, not guessed)

`software_pipeline.py:108` calls `ir.dependency.must_precede(ls, ld)` (leaf→leaf),
which reads `self._closure` — today the transitive closure of a graph that INCLUDES
CARRY/COVER edges routed through loop-nids (`writer → loop → reader`). Removing those
builders could drop a transitive `writer → reader` relation SoftwarePipeline observes,
though the direct base RAW edge usually already supplies it. Two safeguards, and which
applies is a gym-1 finding:

- **(a) clean delete** — if `test_software_pipeline.py` stays green after the
  edge-builders are removed from `_build`, the base RAW closure sufficed; delete them.
- **(b) relocate** — if a `must_precede` answer changes, the carried/coverage relation
  is computed on demand inside the code-motion span-promotion path (where it already
  lives after this change) and the CARRY/COVER edges are simply *not* rebuilt in
  `_build`; SoftwarePipeline's static closure is then unchanged because the base RAW
  edges it actually relied on are untouched. (I.e. the relation moves fully into
  `span()`, not back into `_build`.)

The plan's first diagnostic step runs `test_software_pipeline.py` on gym-1 with the
edge-builders removed to resolve (a) vs (b) before finalising the delete.

## Validation (no oracle)

Correctness rests on the existing gate — nothing new to build. (Test names below carry
`compute_at`/`reverse_compute_at` prefixes today; after the merge they migrate into
`test_code_motion.py` under `CodeMotion`-named tests. It is the VERDICT each pins that
must survive, not the name.)

1. **The pinned verdict tests** (currently in `test/transforms/test_code_motion.py`
   plus the two face-specific files):
   `..._rejects_covering_matmul_reduction_axis` (reject),
   `..._allows_fold_covering_its_own_ko` (allow the subtle self-domination),
   `..._rejects_replicating_reduction_over_untiled_output_dim` (reject, structural),
   `..._memset_sink_across_block_wall_sims_correct` (allow + sim). Plus a NEW verdict to
   add: **the store-sink (k11→k12 shape) is ALLOWED under `CodeMotion`** — the case the
   dropped output guard would have rejected.
2. **`test/ir/test_dependency.py`** — the direct span tests
   (`test_first_backward_edge_flags_memset_sunk_under_kloop`,
   `..._allows_load_under_kloop`, `..._frozen_directions_catch_parallel_producer_flip`,
   the `for_insertion` agreement test). These reach into internals being deleted
   (`_carry_loops_of_leaf` at lines 185/205); they are **rewritten to assert the new
   check's verdict** (or its surviving entry point), not deleted — the verdicts they
   pin are preserved. This is the June-25 obligation restated.
3. **`k0 … k27` byte-exact rebuild + CPU-sim clean on gym-1** via
   `examples/manual_transforms.py` (and the transform-driven ladder in
   `examples/kernel_transforms.py`).
4. **`test_software_pipeline.py` green** after the edge-builder removal (the migration
   hazard gate).

Every intermediate rung must be correct (the composability law): a wrong intermediate
means the check under- or over-rejected — fix the check, not the ladder.

## Test-slimming note

Per the user-locked rule, rewriting the internal-reaching tests
(`test_dependency.py` lines 185/205, and any `match=` string that named the old
reduction-axis error) PRESERVES every assertion's *verdict* — it re-points them at the
new entry point / new error string, it does not drop coverage.

## Build order

Two workstreams — legality (span-promotion) and surface (merge). Sequenced so the
suite is green at each gym-1 checkpoint. Legality first (behavior-only, faces
untouched), then the merge (rename + de-dup + guard-drop) on top.

1. **Add span-promotion** to `dependency.py`'s two `span()` closures (carried +
   invariance predicates on demand), leaving the CARRY/COVER builders in place for now.
   Unit-test the memset-into-K reject, load-under-N allow, and store-into-ko reject
   directly against the new `span()`.
2. **Switch** `_check_move_preserves_dependencies` to call
   `first_backward_edge_for_insertion` without `skip_cover_loops`; drop the
   `_check_no_reduction_axis_covered` call from `_check_same_loop_prefix`. Both faces
   still exist; run the four pinned verdicts + ladder on gym-1 — legality is now
   span-promotion end to end.
3. **Diagnostic on gym-1:** run `test_software_pipeline.py` with the CARRY/COVER
   builders disabled to resolve migration-hazard (a) vs (b).
4. **Delete** the CARRY/COVER builders, the three `_carry`/`_tiled_write`/`_reads_*`
   helpers, `skip_cover_loops`, `_own_carry_loop_nids`,
   `_check_no_reduction_axis_covered`. Rewrite the internal-reaching tests to the new
   entry points. (Grep-confirm no remaining caller before each removal.) Ladder + suite
   green on gym-1.
5. **Merge the surface:** rename `_code_motion.py` → `code_motion.py`, add `CodeMotion`
   + `CodeMotionOption`, drop `is_reverse` throughout, drop the output-block guard.
   Delete `compute_at.py` / `reverse_compute_at.py`; update `__init__.py`. Update the
   callers (`examples/kernel_transforms.py`, `examples/matmul_lhsT_rhs.py`) and fold
   the test files into `test_code_motion.py`. Rename `compute_at_legality.md` →
   `code_motion_legality.md` and de-stale it.
6. **Full ladder + suite green on gym-1** — `k0…k27` byte-exact + CPU-sim (crucially
   k11→k12 store-sink passes under `CodeMotion` with no output guard); the pinned
   verdicts; `test_software_pipeline.py`.

## Non-goals

- No unrolling / no concrete oracle module (dropped from the earlier iteration).
- No change to `_move`'s verbatim splice, Split/Reorder/Fuse/RFactor internals, the
  `arith` substrate, or the `nki` ops.
- No NEW code-motion capability. `CodeMotion` is the union of the two existing faces;
  the store-sink it "unblocks" was already reachable via `ReverseComputeAt` — the merge
  only stops the redundant output guard from being re-imposed. Actively *pursuing* the
  store-sink for MFU (and the region-rebase / list-of-tiles work around it) stays
  separate follow-on.
- `KernelIR.dependency` is not dropped; SoftwarePipeline + `analyze` slot enumeration
  still read it. Dropping the stored graph entirely is a later plan gated on migrating
  SoftwarePipeline.
