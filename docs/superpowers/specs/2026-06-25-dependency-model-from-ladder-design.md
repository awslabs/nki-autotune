# Dependency model, redesigned from the ladder — design

**Date:** 2026-06-25
**Status:** design (brainstorming; supersedes the dependency portions of
`2026-06-25-dependency-region-cover-codemotion-bufferlayout-design.md` §1–2 —
this is the from-scratch redesign that spec asked for, reasoned from the
transforms in `examples/manual_transforms.py`).

## Why redesign

The shipped `nkigym/ir/dependency.py` accreted three edge families (RAW/WAW/WAR
flow + synthetic `CARRY` + `COVER`) plus a `skip_cover_loops` escape hatch and the
`_carry_loops_of_leaf` / `_tiled_write_loops_of_leaf` / `_reads_independently_of_loop`
heuristics. That set is hard to reason about and has flip-flopped historically. The
goal here is a model derived **bottom-up from what the ladder transforms actually
need** — nothing more — so every check has a transform that justifies it.

## Method: the dependency model serves exactly ONE caller

The only consumer of dependency information that gates legality is **code motion**
(`ComputeAt`/`ReverseComputeAt` today, unified `CodeMotion` later). Split, Reorder,
Fuse, Buffer-layout, and RFactor either do not reorder leaf execution across a shared
buffer, or carry their own structural legality. So the design question is precise:

> **When code motion relocates a block, which dependency facts decide whether the new
> position preserves the program's meaning?**

We answer it by walking every transform in `examples/manual_transforms.py` (the
byte-exact target) and, for each, writing down the dependency fact that made that rung
legal — or that would have made a nearby illegal move illegal. The union of those
facts is the model; anything not needed by some rung is not in the model.

## Scope (decided): the code-motion LEGALITY FUNCTION only

A grep of the shipped transforms shows **two** legality consumers of dependency info, not
one:

- **Code motion** — the hard query: "if I splice leaf X under loop L at slot i, does any
  producer→consumer pair flip backward?" Needs proposed-position ordering + the
  accumulator barrier. THIS is what we redesign.
- **SoftwarePipeline** — two SIMPLE queries on the CURRENT tree: `must_precede(a, b)` and
  `info(leaf)` (per-leaf reads/writes). No proposed-position, no barrier.
- Split / Reorder / Fuse / RFactor — only the end-of-`apply` `Dependency(tree)` rebuild;
  they do not consult it for legality.

So there are really two layers: a **static footprint/precedence layer** over the current
tree (what SoftwarePipeline uses, and what code motion's `analyze()` uses for slot
enumeration) and a **code-motion legality function** built on top (the proposed-position
+ barrier logic). **This redesign is scoped to the second layer ONLY.** The static layer
(today's `Dependency` precedence + `info` + `touches_by_tensor`) is treated as a GIVEN —
kept, lightly refactored if needed, NOT deleted. Consequences:

- `KernelIR.dependency` is **NOT dropped** in this work (SoftwarePipeline still needs it).
- `CARRY`/`COVER`/`skip_cover_loops` — these live in the code-motion legality path, so
  they ARE replaced by the new barrier logic. But the base RAW/WAW/WAR precedence relation
  and `info`/`touches_by_tensor` STAY (SoftwarePipeline + `analyze` slot enumeration).
- The four-facts model below describes the code-motion legality function; it reads the
  static layer's footprints, it does not replace them.
- A later plan may unify or drop the static layer once SoftwarePipeline is also migrated;
  that is explicitly out of scope here.

## The ladder's transforms (what each does to execution order)

`kernel_0 … kernel_27`, annotated by the comments in the file:

- **Split** (e.g. k1→k2): sub-tiles one loop into two. Leaf execution order over each
  buffer region is unchanged — a producer still precedes its consumer. **No dependency
  check needed** (structural: trip-product invariant).
- **Reorder** (e.g. k0→k1): swaps an adjacent loop pair above one block. Permutes the
  iteration order of a SINGLE block's own loops; does not move one block relative to
  another. **No cross-block dependency check** — Reorder's own legality (perfect-nest,
  no SEQUENTIAL leaf on the swap dim) is structural and already shipped.
- **Buffer layout** (e.g. k5→k6): packed ndarray ↔ list-of-tiles. Pure allocation-form
  change; touches no execution order. **No dependency check.**
- **Code motion** (e.g. k8→k9, k15→k16, k19→k20, k24→k25): relocates one block under
  another's loop. **This is the only transform whose legality is a dependency
  question.** Detailed below.
- **RFactor** (k26→k27): splices NEW blocks (per-`ko` memset, copy, fold) into the `ko`
  loop and retargets the init. It is a structural rewrite by a fixed recipe, but the
  blocks it creates and the init-domination it relies on are exactly the facts code
  motion later must respect, so RFactor's output defines dependency obligations even
  though RFactor itself does not call the checker.

So the model is justified entirely by **code motion**, with RFactor's output as the
source of the hardest case (a reduction accumulator's init/drain).

## Walking the code-motion rungs

Each rung below: what moved, why it is legal, and the dependency fact that decides it.
A buffer access is a `(tensor, region)` where `region` is per-axis `(lo, width)` in
loop-var space; the **footprint** of a leaf at a tree position is its region evaluated
over the loops enclosing it.

### Rung A — k8→k9: sink the drain `tensor_copy` into the matmul's N loop

k8 has the matmul nest `for i_d2_0: for i_d0_0…: nc_matmul(... psum_prod ...)` followed
by a separate full-extent drain `for i_d1_0: tensor_copy(psum_prod → sbuf_prod)`. k9
sinks the drain to the END of the `i_d2_0` body, so it drains the N-tile the matmul just
produced, per N iteration.

- **What this requires:** the matmul (producer of `psum_prod`) must still execute before
  the drain (consumer) for every element the drain reads. After the move the drain reads
  `psum_prod[*, *, i_d2_0*512 : +512]` — exactly the slice the matmul wrote in this
  `i_d2_0` iteration. Legal.
- **What would make it ILLEGAL:** sinking the drain ABOVE the matmul in the `i_d2_0`
  body (it would read `psum_prod` before this iteration's matmul writes it), or to a
  position where it reads a slice no matmul iteration has produced yet.
- **Dependency fact #1 — producer-before-consumer, per shared region.** For every pair
  (writer W, reader R) of a buffer where the original program ran W before R, the new
  position must keep W before R *for the overlapping region*. This is the RAW core.

### Rung B — k15→k16: sink the psum `memset` into the matmul's N loop

k15's `memset(psum_prod)` sits OUTSIDE the `i_d2_0` loop (zeroes the whole accumulator
once). k16 sinks it to the FIRST child of the `i_d2_0` body — but note `psum_prod`
became per-N-tile `(128,1,512)` and the matmul's K reduction (`i_d0_0`) is now INSIDE
`i_d2_0`. The memset zeroes this N-tile's psum before the K accumulation over it.

- **What this requires:** the memset (init of the accumulator) must dominate the K
  reduction that accumulates into it — execute before the first `nc_matmul` of this tile,
  and NOT re-execute inside the K loop (re-zeroing mid-accumulation destroys the partial
  sum → NaN). In k16 the memset sinks under `i_d2_0` but stays OUTSIDE the K loop
  (`i_d0_0`), so it dominates correctly.
- **What would make it ILLEGAL:** sinking the memset INSIDE the K loop (`i_d0_0`) — it
  would re-zero `psum_prod` every K iteration, so only the last K-block survives → NaN.
  This is the historical k16/k17-NaN bug.
- **Dependency fact #2 — an accumulator's init must dominate its reduction loop, and may
  not enter it.** The matmul RMWs `psum_prod` loop-invariantly across the K (ACCUMULATION)
  loop, so `psum_prod`'s value is LIVE across the entire K loop. Any other writer of
  `psum_prod` (the memset) must sit ENTIRELY OUTSIDE that live range — before it, never
  within. (This is what the old `CARRY` edge encoded.)

### Rung C — k19→k20 and k24→k25: sink the operand loads into the matmul nest

k19's `rhs` load is a full up-front `for i_d0_0: dma_copy(rhs → sbuf_rhs)`. k20 sinks it
under the matmul's `i_d2_0` (and k25 sinks `lhs_T` under `i_d1_0`), so each operand tile
is streamed just before the matmul that consumes it.

- **What this requires:** the load (producer of `sbuf_rhs`) must execute before the
  matmul (consumer) for the tile the matmul reads. Same producer-before-consumer as #1.
- **The subtlety — duplication on a non-indexed loop:** `lhs_T` is N-invariant, yet k25
  sinks it UNDER the N loop, so it RELOADS per N-block. That is legal because a pure
  reload re-writes the same buffer (idempotent producer); it is NOT legal for an
  accumulation block (re-running a reduction per non-indexed iteration corrupts).
- **Dependency fact #3 — replicating a pure producer over a loop it does not index is
  legal (idempotent reload); replicating an accumulation is not.** (This is what the old
  `_check_no_reduction_replicated` guarded.)

### Rung D — k26→k27 (RFactor output): the drain must cover what the consumer reads

RFactor's `drain_two_stage_0` does `tensor_copy(psum → sbuf_rfactor)` then
`tensor_tensor(sbuf_prod = sbuf_prod + sbuf_rfactor)` per `ko`. The closing store reads
the WHOLE `sbuf_prod` (all `ko` folded in). If a later code motion tried to sink that
store INTO the `ko` loop, it would read `sbuf_prod` before all `ko` partials are folded.

- **What this requires:** a consumer reading a buffer over an extent produced ACROSS a
  loop must stay OUTSIDE that loop until every iteration's contribution is in.
- **Dependency fact #4 — coverage: a consumer that reads a region wider than one
  iteration's write of an enclosing loop must execute after that loop completes.** (This
  is what the old `COVER` edge encoded.)

## The four facts, unified

Facts #1–#4 are all the SAME question asked at the moved block's new position:

> Does every byte the moved block reads (or writes) still sit on the correct side of
> every byte some other block writes (or reads) to the same buffer — i.e. is the
> producer→consumer order preserved for the OVERLAPPING region, accounting for the loops
> that bracket each access?

- #1 RAW order = the base case (point vs point).
- #2 init-domination = the producer (init) vs a reader whose live range is a whole loop
  (point vs loop-span).
- #3 replication = a producer duplicated over a loop it does not index (idempotent ⇒ ok;
  accumulation ⇒ the duplicated writes collide on one region ⇒ not ok).
- #4 coverage = a consumer vs a producer whose writes are spread across a loop
  (loop-span vs point).

So the model needs exactly two primitives: **(a) per-access footprints** (region +
the loops that bracket it), and **(b) a region-overlap-and-order test** that, given two
accesses and the proposed new position, answers "is the original producer→consumer order
preserved over the overlap?" The four facts are that one test applied with different
span shapes (point/loop) on each side. No separate CARRY/COVER edge kinds; they are the
loop-span cases of the single test.

## Granularity: what are the endpoints A and B

A dependency `A → B` is not between two instructions in the abstract — in this IR a
dependency exists ONLY between two accesses to the **same buffer**. The buffer mediates;
without a shared buffer there is no edge. So the endpoints are **accesses**, not blocks:

```
Access = (leaf_nid, buffer, side, region)
```

- **Per-operand, not per-leaf.** One leaf contributes one access per operand. The matmul
  leaf is THREE accesses: read `sbuf_lhs_T`, read `sbuf_rhs`, rmw `psum_prod`. Conflating
  them into one block-node (today's granularity) loses which operand an edge is about —
  and fact #2 is specifically about the `psum_prod` access, so it cannot even be stated
  at block granularity. This is why "access" beats "block".
- **NOT iteration-point.** We do not split an access into one node per `(i_d2_0, …)`
  integer point (the polyhedral model). That exactness needs Presburger machinery we are
  not porting. Instead the region is **symbolic-affine**: loop vars appear IN the offset
  (`psum_prod[…, i_d2_0*512 : +512]`), so "does loop L index this access?" is decidable
  as "does L's var appear in the offset?" — enough to separate a tiling loop (var in
  offset) from an invariant/live-across loop (var absent) without enumerating points.

### Unit of movement vs unit of dependency

Code motion relocates a whole **leaf** (block), which owns several accesses. So the
endpoint of a dependency is an access, but the thing that MOVES is the leaf: the legality
check iterates over the moved leaf's accesses and, for each, checks its dependencies
against every other access to that buffer.

## The model (concrete)

### Access and footprint

```
Access = (leaf_nid, tensor, side, region)        # side ∈ {read, write, rmw}
        region: tuple[(lo: Expr, width: int), ...]  in loop-var space
```

**`side` has three values, not two.** An operand both read and written by one
instruction — the matmul's `psum_prod`, `tensor_tensor`'s accumulator — is `rmw`, the set
the ops already declare as `RMW_OPERANDS`. This makes fact #2 readable DIRECTLY: an `rmw`
access whose region offset is invariant across an enclosing ACCUMULATION loop IS the
accumulator live-range; the init (a plain `write` of the same tensor) must dominate it.
Modelling rmw as a separate read+write would force reconstructing "same value carried
across iterations" by pairing them — re-deriving what `rmw` states outright.

**Overlap = affine-interval disjointness** (reuse the shipped `nkigym/ir/interval.py`).
Two accesses to one buffer conflict iff their regions are NOT provably disjoint: disjoint
iff some axis's `(lo, width)` intervals cannot overlap over the loop-var box. This is the
exact test today's `Dependency._provably_disjoint` already uses — conservative (never
reports a false "disjoint", so a real edge is never dropped), and it handles the ladder's
tile offsets (distinct `psum_prod` tiles, N-tile slices). Not exact/polyhedral; not
"same-tensor ⇒ always conflict" (that would reject the legal disjoint-tile moves the
ladder needs).

For ordering we need, per access, the loops that BRACKET it and how each relates to its
region. Both are **derived from the tree on demand**, nothing stored on the access (so
nothing to invalidate after a mutation):

- **Bracketing loops** = the access leaf's ForNode ancestors at the tree being evaluated.
- **Loop-access relation is BINARY** — for an access and an enclosing loop L:
  - **L INDEXES the access** iff L's loop var appears in some region-axis `lo` (L tiles
    or sweeps the access; distinct iterations touch distinct slices).
  - **L BRACKETS-only** iff L's var is absent from every offset (the access is reproduced
    identically per iteration — replicated, or its value live-invariant across L).

The binary relation + `side` is necessary but — as the **pressure test below proves** —
NOT sufficient for the accumulation cases (#2/#3): a brackets-only `rmw` loop is legal
(HW-accumulation) or illegal (replication) depending on whether it is CLOSED by an
init/drain bracket, which the axis role alone does not reveal. The corrected facts:

- **#1 order:** the point/point base case, no loop relation needed. Reads off positions
  alone.
- **#4 coverage:** a `write` that L INDEXES paired with a `read` that L BRACKETS-only ⇒
  the read needs every L iteration's slice, so it must sit outside L. Reads off the binary
  relation + side.
- **#2 / #3 (accumulation):** require locating the accumulator's init (`write`) and drain
  (`read`) and which loops they bracket — see "Pressure test" below for why role-alone
  fails and what the corrected barrier model is.

### The single check: `order_preserved(producer, consumer, tree)`

Given the original program's producer→consumer **direction** (frozen — see below) and a
candidate `tree` (the program after the proposed move), the move is legal iff, for every
ordered pair (P, C) of accesses to the same tensor with overlapping regions:

1. **Point/point (RAW/WAW/WAR, fact #1):** P's tree position precedes C's. Position uses
   the preorder span technique already in `first_backward_edge_for_insertion`
   (`span(P).end < span(C).start`).
2. **Accumulator barrier (fact #2, BOTH directions):** if a tensor has an `rmw` access
   bracketed by a brackets-only loop L that is CLOSED — an init (`write`) dominates L and
   a drain (`read`) post-dominates L, both outside L — then the init→L→drain bracket is a
   BARRIER: no foreign writer OR reader of the overlapping region may sit inside L. (The
   draft captured only "no foreign writer"; the pressure test added "no foreign reader" —
   a drain sunk into L reads a partial sum.)
3. **Replication / unclosed accumulation (fact #3):** a brackets-only `rmw` or `write`
   access over a loop L with NO closing init inside L is replicated over L. Idempotent for
   a pure producer (legal reload); a collision for an accumulation (illegal). "Closed by
   an init inside L" — NOT the axis role — is the legal-`ki` vs illegal-foreign-`L`
   discriminator (pressure test, Gap 1).
4. **Coverage (fact #4):** if P writes a tensor region tiled by a loop L (P's write
   offset depends on L's var) and C reads that tensor over an extent NOT indexed by L,
   then C must sit OUTSIDE L (after it completes).

All four read the SAME footprints; they differ only in whether each side's relevant span
is a point (a leaf) or a loop (an ACCUMULATION live range / a tiling loop). The
implementation is one function that classifies each (P, C) pair by these span shapes and
applies the matching comparison — not four edge kinds materialized into a graph.

### Direction is frozen from the pre-move program (load-bearing)

The producer→consumer DIRECTION of each pair is read from the CURRENT (pre-move) tree,
then the ordering is tested at the PROPOSED position. Re-deriving direction from the
moved tree would flip a sunk producer's RAW edge into a WAR and hide the violation
(matmul reads uninitialised data → NaN). Recompute-on-demand satisfies this for free:
the legality check runs BEFORE the move, so the current tree IS the pre-move program.
This is the one invariant carried verbatim from the shipped model
(`first_backward_edge_for_insertion`'s contract).

### What the legality function stores (nothing) vs what the static layer keeps

The **code-motion legality function** stores nothing: given the current `KernelIR` and a
proposed `(leaf, target_loop, index)`, it derives footprints from the tree on demand,
freezes directions from the current tree, and answers legal/illegal. No `CARRY`/`COVER`
edge kinds, no `skip_cover_loops` — those replaced by the barrier logic.

The **static layer is untouched by scope** (see Scope section): `KernelIR.dependency`,
the RAW/WAW/WAR precedence relation, `info`, and `touches_by_tensor` STAY — SoftwarePipeline
and `analyze`'s slot enumeration read them. This redesign removes the CARRY/COVER/skip
machinery from the code-motion path and replaces it with the barrier function; it does NOT
remove the base graph. (Dropping the stored graph entirely is a later plan, gated on
migrating SoftwarePipeline too.)

## Equivalence obligation (the gate, not a shim)

The redesigned check must return the **same legal/illegal verdict** as the shipped model
on every code-motion move the suite and the full ladder exercise — these are the pinned
hard cases:

- `test_compute_at_rejects_covering_matmul_reduction_axis` (fact #2: reject).
- `test_reverse_compute_at_allows_fold_covering_its_own_ko` (fact #2: ALLOW the fold
  covering its OWN ko — the init it itself dominates; the subtle "allow" that over-broad
  guards get wrong).
- `test_compute_at_rejects_replicating_reduction_over_untiled_output_dim` (fact #3:
  reject).
- `test_compute_at_memset_sink_across_block_wall_sims_correct` (fact #1/#2: allow).
- Every `kernel_0…kernel_27` rung rebuilds byte-exact + CPU-sim clean on gym-1.

Two suite tests reach into the OLD internals (`_check_same_loop_prefix` import;
error-message `match=` strings). Those are rewritten to assert the new check's verdict
(or, where the loop-prefix structural check still lives, its surviving entry point), not
deleted — the verdicts they pin are preserved.

## Build order (rung-by-rung, gym-1-gated)

Scoped to the code-motion legality function; the static layer (`KernelIR.dependency`
precedence + `info` + `touches_by_tensor`) stays for SoftwarePipeline.

1. **Barrier + ordering check, as a NEW pure module** (`transforms/_code_motion_legality.py`
   or similar), reading footprints from the tree on demand. Unit-test the four facts
   (including the barrier's both-directions and the closed-vs-unclosed discriminator)
   directly with hand-built trees.
2. **Equivalence harness:** run BOTH the shipped code-motion legality path (CARRY/COVER/
   skip) and the new barrier check on every code-motion move in the suite + the ladder;
   assert identical verdicts. Test scaffold (deleted at the end), NOT a production shim.
3. **Switch code motion** (`_check_move_preserves_dependencies`) to the new barrier check;
   **delete** the CARRY/COVER/`skip_cover_loops` machinery and the
   `_carry_loops_of_leaf`/`_tiled_write_loops_of_leaf`/`_reads_independently_of_loop`
   heuristics THAT ONLY THE CODE-MOTION PATH USED. **Keep** `KernelIR.dependency`, the
   base RAW/WAW/WAR precedence, `info`, `touches_by_tensor` — SoftwarePipeline + `analyze`
   slot enumeration still read them. (Verify by grep that a deleted helper has no
   remaining caller outside the code-motion path before removing it.)
4. **Delete the equivalence harness**; full ladder + suite green on gym-1.

**Migration hazard (verified by grep, must be handled in step 3):** CARRY/COVER edges are
currently built UNCONDITIONALLY inside `Dependency._build` (`_add_carry_edges` /
`_add_coverage_edges`) and folded into `self._closure`. SoftwarePipeline's
`must_precede(a, b)` reads that closure — so today it can observe ordering relations that
CARRY/COVER contributed. Removing those edge-builders therefore MIGHT change a
`must_precede` answer SoftwarePipeline relies on. Two safeguards: (a) the equivalence
harness (step 2) must also assert `test_software_pipeline.py` stays green after the edge
removal, not just code-motion verdicts; (b) if a `must_precede` answer does change, the
CARRY/COVER builders are moved INTO the code-motion legality module (computed on demand
there) rather than deleted from `_build`, so the static closure SoftwarePipeline reads is
unchanged. Which of (a-clean-delete) or (b-relocate) applies is a gym-1 finding, not a
guess — the plan's first diagnostic step resolves it.

## Non-goals

- No change to `_move`'s verbatim splice, Split/Reorder/Fuse internals, the `arith`
  substrate, or the `nki` ops.
- No new transform. (The `CodeMotion` unification is a SEPARATE plan; this one keeps the
  two existing faces and only swaps what their legality calls.)
- No TVM port literal-for-literal — region-cover is the inspiration, but the model is
  derived from THIS ladder's needs, validated on gym-1, not against a TVM oracle.

## Pressure test — two gaps found (the binary classification was too strong)

Tracing the hardest rungs against the model exposed two real defects. Recording them
honestly; they change the model.

### Gap 1 — `(side, indexes/brackets-only, role)` does NOT discriminate legal accumulation from illegal replication

Two cases have the **identical** access signature but **opposite** verdicts:

- **Legal (post-RFactor matmul, k27):** nest `for ko: memset(psum); for ki: matmul(psum += …)`.
  The matmul's `psum` rmw offset has neither `ko` nor `ki`, so both are brackets-only.
  RFactor flips the matmul's K role to **PARALLEL**. So `ki` is `(rmw, brackets-only,
  PARALLEL)` and psum legitimately HW-accumulates over it. LEGAL.
- **Illegal (`test_compute_at_rejects_replicating_reduction_over_untiled_output_dim`):**
  a foreign output loop `L` brackets-only the matmul's `psum` rmw, `L` also a PARALLEL
  axis. Replicating the K accumulation into one un-re-init'd psum. ILLEGAL.

Both are `(rmw, brackets-only, PARALLEL)`. The binary relation + axis role cannot tell
them apart. **What actually discriminates:** the legal `ki` is *closed* — its accumulation
is bracketed by an init (memset) that dominates it and a drain (copy) that post-dominates
it, both inside the same `ko`. The foreign `L` is *not closed* — no init inside `L`
re-establishes the accumulator per `L`-iteration. So the legality of a brackets-only rmw
loop depends on **whether that loop sits inside the accumulator's init→drain bracket** —
the relationship the old `CARRY` edge / `own_carry_loop_nids` computed. It is NOT
derivable from `(side, brackets-only, role)` alone.

### Gap 2 — fact #2 was one-directional; the drain (reader) rule is missing

Fact #2 as drafted: "every other WRITER (init) must sit outside the live-range loop." It
says nothing about READERS. But a reader of an accumulator sunk INTO the reduction loop
reads a partial sum — and point/point fact #1 is satisfied per iteration (`fold_j` before
`store_j`) yet the value is wrong. The old model had BOTH `CARRY` directions
(`init_writer → loop` AND `loop → consumer`); the design captured only the first.

### Revised model: the accumulator live-range is a BARRIER with an init/drain bracket

Replace the one-directional fact #2 + the "role discriminates replication" assumption
with: an rmw access defines, per enclosing brackets-only loop `L`, an **accumulation over
`L`** that is **valid iff it is closed** — an init (a `write` of the same tensor)
dominates `L` (sits outside, before) and a drain (a `read`) post-dominates `L` (sits
outside, after). Then:

- the init→`L`→drain bracket is a **barrier**: no foreign writer or reader of the
  overlapping region may sit inside `L` (fact #2, now both directions);
- a brackets-only rmw loop with NO closing init inside it is illegal replication (fact #3)
  — this is the post-RFactor `ki` (closed by the per-`ko` memset ⇒ legal) vs the foreign
  `L` (not closed ⇒ illegal) discriminator, computed from the init/drain positions, NOT
  the axis role.

This means the model **does** need to locate, for an accumulator, its init and drain
accesses and which loops they bracket — modest bookkeeping (read from the tree on demand,
still no persistent graph), but MORE than the pure `(side, binary-relation, role)` read
the earlier draft claimed. The access granularity and the on-demand derivation survive;
the over-simplified "four facts collapse with zero accumulator bookkeeping" claim does
not. **This is the central correction the pressure test produced.**

### What survives unchanged

- Endpoints are per-operand accesses; `side ∈ {read, write, rmw}`; overlap = affine
  intervals; loop context derived on demand. (Granularity decisions all hold.)
- Direction frozen from the pre-move program (the NaN guard).
- No persistent `Dependency` graph / no `KernelIR.dependency`; recomputed per query.
- The loop-carried self-dependence of an accumulation across its OWN reduction loop is
  still NOT an edge: code motion never reorders a single block's own iterations, and
  Reorder's structural SEQUENTIAL guard blocks the only transform that could. (Verified
  in the trace; an explicit assumption, not an omission.)

## Open question for review

The build order above stages an equivalence harness (option "equivalence-harness first")
rather than a direct switch. If the preference is the more aggressive direct switch
(matching "removals over shims"), steps 1–3 collapse: build the check, switch, delete the
old model in one plan, with the gym-1 ladder + golden-verdict as the only net. The
harness is a test scaffold either way — flagged here so the rollout shape is a conscious
choice before the plan is written.
