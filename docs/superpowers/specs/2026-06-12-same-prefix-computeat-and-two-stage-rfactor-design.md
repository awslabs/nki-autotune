# Same-Prefix ComputeAt + Two-Stage RFactor

*Design date: 2026-06-12*

Two coupled transform changes, validated byte-exact against the hand-written
reference ladder in `examples/kernel_transforms.py` (k0..k8):

1. **Tighten `ComputeAt` / `ReverseComputeAt`** to a *same loop prefix* legality
   rule (k4→k5, k6→k7).
2. **Rewrite `RFactor`** to emit the *fused two-stage-accumulation* form directly
   (k7→k8), as a generic structural transform that looks up op/Trn-specific
   gadgets from a new reduction-recipe registry.

Plus the test re-grounding both require: the suite still targets the previous
k0..k14 ladder, which the new k0..k8 reference file replaced.

> **Supersedes `docs/superpowers/specs/2026-06-07-rfactor-transform-design.md`
> and deletes the `2026-06-09-rfactor-spec-compliance-fix.md` plan.** Those
> specified RFactor as TVM's *multi-slot terminal form* (rf-buffer of
> `[factor, *shape(out)]` + a separate write-back block), with the fused
> single-accumulator reached *downstream* by a later `ComputeAt` fold. Per the
> decision to treat k7→k8 as one RFactor rung, RFactor now emits the fused form
> directly; the downstream-fold path is removed. Surviving contributions folded
> in below: the `RFACTOR_RECIPE = "rmw" | "slot"` taxonomy and the
> `REDUCE_COMBINATOR` reducer declaration (both already shipped on the ops), and
> the legality framing (assoc+commutative reducer; never gate resource capacity).

## Decisions (at-a-glance)

1. **`ComputeAt`/`ReverseComputeAt` legality tightens to *same loop prefix*:** the
   target's enclosing loop nest (outermost down to `target_loop_nid`), as an
   ordered `(dim, extent)` sequence, must be an exact prefix of the moved block's
   own loop nest. Inner residual loops on dims the target does not iterate stay
   legal; a partial split of a shared dim, or a different loop order, is rejected
   loudly ("Split / Reorder first").
2. **`RFactor` is one-stage → two-stage accumulation.** Generic structural
   skeleton (§2.1); all buffer/ISA specifics come from a recipe lookup.
3. **Op/Trn-specific gadgets live in a standalone reduction-recipe registry**
   (`nkigym/ops/reduction_recipes.py`), keyed by `op_cls.RFACTOR_RECIPE`. The
   transform never names `psum`, `memset`, `tensor_copy`, `tensor_tensor`, or the
   staging buffer.
4. **The fused form is the RFactor atom's output** (not a downstream fold). k8 is
   byte-exact to `RFactor` applied to the co-located k7 state.
5. **RFactor requires the co-located (PSUM-hoisted) k7 state** as precondition:
   the one-stage init must be the run-op block's preceding sibling and the
   one-stage drain its following sibling, under a shared per-output-tile prefix.
   A flat / un-hoisted state is rejected loudly.
6. **Legality = correctness + dep-order + ISA well-formedness only; never resource
   capacity** (user-locked, carried over from the prior spec).
7. **Emission is verified on hardware before it is locked** (§4): the constructed
   k8 IR must render byte-exact, CPU-sim correct, and pass the dependency model —
   confirmed by a desktop scout, not by static reasoning.

## Scope

| # | Change | Primary files |
|---|--------|---------------|
| A | Same-prefix `ComputeAt`/`ReverseComputeAt` | `transforms/_code_motion.py` (+ `_domain_solve.py` helpers), `transforms/compute_at_legality.md` |
| B | Two-stage `RFactor` + recipe registry | `transforms/rfactor.py`, new `ops/reduction_recipes.py`, this spec (supersede 06-07), delete 06-09 plan, retire `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` + `examples/matmul_rfactor.py` |
| C | Re-ground tests to k0..k8 | `test/transforms/_fixtures.py`, `test_compute_at.py`, `test_reverse_compute_at.py`, `test_rfactor.py`, `test_split.py`, `test/ir/test_dependency.py`, `test_software_pipeline.py`, `_ladder_scout.py`, `examples/kernel_transforms_repro.py`, `test/runner/test_output.py` |

Out of scope: the `"slot"` recipe (activation_reduce) — registered but raises
`NotImplementedError`; the k0..k4 rungs use already-shipped transforms unchanged.

---

## 1. Same loop prefix — `ComputeAt` / `ReverseComputeAt`

### 1.1 The problem with the current rule

`_check_move_realizable` → `solve_iter_domains` checks legality **per dim, by
product**: for each moved dim it requires the target's coverage to *divide* the
moved extent, then regenerates a residual loop for the leftover
(`_domain_solve.py`). Two over-permissive consequences:

- **Partial coverage of a shared dim is accepted.** Target covers `d_M=8`, moved
  block has `d_M=16` → the solver emits a `residual=2` loop. The existing
  `test_compute_at_partial_coverage` documents this path producing a numerically
  *wrong* residual nest.
- **Loop order is ignored.** Per-dim products cannot distinguish an enclosing
  nest `[d2, d1]` from `[d1, d2]`; the solver collapses both to the same
  per-dim coverage.

### 1.2 The rule

Build two **interleaved, ordered** loop sequences (outermost → innermost, across
all dims), each as a list of `(dim, extent)`:

- `target_seq` — every `ForNode` from the outermost enclosing loop down to and
  including `target_loop_nid`.
- `moved_seq` — the moved block's own loop nest.

> A move is **legal iff `target_seq` is an exact element-wise `(dim, extent)`
> prefix of `moved_seq`.**

`moved_seq[len(target_seq):]` is the residual inner nest. Because the match is a
*prefix*, the residual can only hold dims the target does not drive, each at full
extent — never a partial split of a shared dim. Any mismatch (extent, dim, or
interleave order) → loud `TransformLegalityError` instructing the caller to
`Split` / `Reorder` first.

### 1.3 Why this fits the new ladder

The k0..k8 ladder is authored so every compute_at move is exact-prefix:

- **k3→k4, k5→k6** `Split` the memset / drain on N *first* (`d2 → (4,512)`), so
  their loop nest becomes `[d2(4), d1(16)]`.
- **k4→k5** sinks the memset under the matmul's `i_d1_0`, whose enclosing prefix
  is exactly `[d2(4), d1(16)]` — exact prefix, zero residual.
- **k6→k7** lifts the drain `tensor_copy` `[d2(4), d1(16)]` the same way.

The Split-first discipline the new rule *enforces* is precisely what the ladder
already does by hand. The previously-buggy partial-residual path becomes
unreachable: the only residual any survivor produces is a full-extent inner dim
(the clean regen case).

### 1.4 Implementation

One shared helper invoked from `_check_move_realizable` (both faces already route
through it):

```python
def _check_same_loop_prefix(ir, block_nid, target_loop_nid) -> None:
    target_seq = _ordered_loop_seq(enclosing_dim_loops(...))   # [(dim, extent), ...]
    moved_seq  = _ordered_loop_seq(dim_loops_of_block(...))
    if target_seq != moved_seq[:len(target_seq)]:
        raise TransformLegalityError("ComputeAt requires the target's enclosing "
            "loop nest to be an exact prefix of the moved block's loops; "
            f"target={target_seq} is not a prefix of moved={moved_seq} "
            "(Split / Reorder the mismatched loop first)")
```

`enclosing_dim_loops` / `dim_loops_of_block` already return the per-dim
`(loop_var, extent)` chains crossing BlockNode walls (`_domain_solve.py`); the new
helper flattens them into the single outer→inner interleaved sequence the chain
order implies, and compares as ordered lists rather than per-dim products.
`solve_iter_domains` / `regen_and_rebind` / `_move` are **unchanged** — we only
restrict which candidates reach them; every survivor hits the clean
full-extent-residual regen path. `analyze()` already filters on
`TransformLegalityError`, so the tighter check simply yields fewer options.

`compute_at_legality.md` §5-cov is updated: the "coverage divides" bullet is
replaced by the same-prefix rule, and §-composition's "partial-coverage cases
must Split first" promise becomes the enforced contract rather than guidance.

---

## 2. RFactor = one-stage → two-stage accumulation

### 2.1 The general skeleton (transform-owned, op-agnostic)

A reduction is *one-stage accumulation*: init the accumulator once, reduce, drain
once. RFactor factors the reduction loop into `ko`/`ki` and restructures it into
*two-stage accumulation* — `ki` produces a partial, `ko` sums the partials. This
is the entire transform; it is the TVM-faithful structural part.

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

The transform's responsibilities, all structural:

1. **Match BEFORE.** The run-op block's RMW operand (via `op_cls.RMW_OPERANDS`) is
   the accumulator; `init_one_stage` is the preceding sibling that writes it;
   `drain_one_stage` is the following sibling that reads it. The `ko`/`ki` nest is
   the factored reduction (a prior `Split` produced `ko`).
2. **Re-lay control flow to AFTER.** Place each gadget at its position relative to
   `ko`: `init_two_stage_0` before `ko`; `init_two_stage_1` first inside `ko`
   (before `ki`); `drain_two_stage_0` last inside `ko` (after `ki`);
   `drain_two_stage_1` after `ko`.
3. **Flip the factored-axis role.** In the run-op block, `ko` becomes PARALLEL
   (each `ko` is an independent partial); the new stage-2 accumulate block carries
   `ko` as ACCUMULATION (its cross-`ko` carry is the second-stage reduction). `ki`
   is untouched (still the stage-1 reduction). Legal only because the reducer is
   associative + commutative (declared by `REDUCE_COMBINATOR`).
4. **Re-derive geometry.** `place_buffers` (LCA) + rebuild `Dependency`.

The transform never names a buffer or an ISA op. It asks the recipe to *emit* the
four gadgets and to *declare* its transient buffers, then splices them.

### 2.2 Precondition (the co-located k7 state)

RFactor requires the one-stage init and drain already co-located with the run-op
block under a shared per-output-tile prefix (the PSUM-hoisted k7 shape that
same-prefix `ComputeAt`/`ReverseComputeAt` build):

```
for i_d2_0(4):
  for i_d1_0(16):
    memset(psum[d1, d2])                         # init_one_stage  — preceding sibling
    for i_d0_0(2): for i_d0_1(8): nc_matmul(...) # run_op nest     (ko=d0_0, ki=d0_1)
    tensor_copy(sbuf_prod[d1,d2] <- psum[d1,d2]) # drain_one_stage — following sibling
```

If the init is not the preceding sibling writing the accumulator, or the drain not
the following sibling reading it, under the run-op block's prefix → loud
`TransformLegalityError` ("ComputeAt the init/drain first"). `analyze()` only
offers the factorable outer reduction loop (`ko`) of a co-located run-op block.

### 2.3 The reduction-recipe registry (op/Trn-specific)

New module `nkigym/ops/reduction_recipes.py`. The transform fills a context from
the matched BEFORE state and looks up the recipe by `op_cls.RFACTOR_RECIPE`:

```python
@dataclass(frozen=True)
class TwoStageContext:
    """Everything a recipe needs to emit gadgets, derived from the BEFORE match."""
    accumulator: BufferRegion      # run_op RMW operand region (matmul dst = psum_prod)
    output: BufferRegion           # drain_one_stage dst region (sbuf_prod tile)
    ko_var: str
    m_var: str                     # partition-tile loop var of the output tile
    free_extent: int
    combinator: ReduceCombinator   # from op_cls.REDUCE_COMBINATOR

class ReductionRecipe(Protocol):
    def transient_buffers(self, ctx) -> tuple[Buffer, ...]: ...
    def init_two_stage_0(self, ctx) -> BlockNode: ...
    def init_two_stage_1(self, ctx) -> BlockNode: ...
    def drain_two_stage_0(self, ctx) -> tuple[BlockNode, ...]: ...
    def drain_two_stage_1(self, ctx) -> tuple[BlockNode, ...]: ...

REGISTRY: dict[str, ReductionRecipe] = {"rmw": RmwRecipe(), "slot": SlotRecipe()}
```

`RmwRecipe` (matmul on Trn2) emits:

| gadget | emission | rationale |
|--------|----------|-----------|
| `transient_buffers` | `sbuf_rfactor (128, 1, 512)` in SBUF, declared in the `d1` block | staging tile; lives one `(d1,d2)` tile |
| `init_two_stage_0` | `memset(out_sbuf, identity)` — before `ko` | the **2nd** accumulator; two-stage needs two inits |
| `init_two_stage_1` | `memset(psum, identity)` — first inside `ko` | re-zero the per-`ko` PSUM partial |
| `run_op` | `nc_matmul` (transform leaves it; `ki` accumulates in PSUM) | stage-1 reduction is HW `+=` in PSUM |
| `drain_two_stage_0` | `tensor_copy(sbuf_rfactor <- psum)` **then** `tensor_tensor(out_sbuf = combiner(out_sbuf, sbuf_rfactor))` — last inside `ko` | `tensor_tensor` cannot read a PSUM operand, so the PSUM partial must land in SBUF first (`nc_matmul`/PSUM-specific) |
| `drain_two_stage_1` | `()` | final value already accumulated in `out_sbuf` |

`identity` (`0.0`) and `combiner` (`"add"`) come from the matmul's shipped
`REDUCE_COMBINATOR`. `SlotRecipe` raises `NotImplementedError` (deferred).

Emitting AFTER for `rmw` yields k8 byte-for-byte: `memset(sbuf_prod)` outside
`ko`; inside `ko` → `memset(psum)`, the `ki`-matmul nest, `tensor_copy(rf)`,
`tensor_tensor(+=)`; loops not reordered; `sbuf_rfactor (128,1,512)` declared in
the `d1` block.

### 2.4 What stays / what goes

- **Reused unchanged:** `RFACTOR_RECIPE`/`REDUCE_COMBINATOR` ClassVars,
  `NKITensorTensor`, the `analyze` "factorable reduction loop" enumeration, the
  deep-copy/re-check `apply` contract.
- **Replaced:** the entire shipped `_emit_rmw` multi-slot machinery
  (`_grow_psum_and_add_rf`, `_flip_and_slot_matmul`'s slot-indexing,
  `_nest_memset`/`_nest_drain` flat helpers, `_insert_writeback`,
  `_partition_tiles`, `_slot_region`). The fused emission needs no `[factor, …]`
  growth, no slot index, no separate flat wb-block.

---

## 3. Dependency model — what must hold, verified not assumed

The fused k8 leans on three behaviors of the shipped `ir/dependency.py`. The
learnings forbid settling legality claims by static source reasoning, so §4 gates
these on a hardware scout before the emission is locked. The *expectations*:

- **`memset(psum)` inside the now-PARALLEL `ko` is legal.** It inits the stage-1
  partial; the reduction loop it sits inside is `ki`, not its own. The
  "carried-init cannot sink into its own reduction loop" guard keys on the
  carrying loop's role — `ko` is flipped PARALLEL in the run-op block, so the
  per-`ko` memset is an ordinary init, not a re-zero of `ki`'s accumulation.
- **`ko` carries the `sbuf_prod` reduction** through the stage-2 `tensor_tensor`
  (RMW on `out_sbuf`, constant address across `ko`). `_carry_loops_of_leaf`
  already distinguishes a *carried* operand (loop-invariant address — here
  `sbuf_prod`) from a *swept* one (`sbuf_rfactor` is a single constant slot, also
  carried-invariant here, but it is a stage-1 product re-read each `ko`, not the
  accumulator). The carry edge must land on `sbuf_prod`.
- **`memset(sbuf_prod)` outside `ko` dominates** that reduction (init → `ko` carry
  edge), and the final value is live after `ko` for the store.

If the scout shows any of these is wrong, the fix is in the *emission shape*
(roles/positions the recipe and transform produce), not a loosening of
`dependency.py` — which this work leaves untouched (a self-review gate).

---

## 4. Hardware verification gate (Task 1 of the plan)

Before rewriting `rfactor.py`, a **controller-run desktop scout** constructs the
target k8 IR — simplest path: hand-edit the post-k7 tree into the two-stage shape,
or build it through the new transform-under-development — then confirms on the
Kaizen desktop (no local env):

1. **Byte-exact:** `render(k8_ir)` matches `KT.kernel_8` under the AST-canonical
   oracle (`assert_matches_hand`).
2. **Numeric:** CPU-sims `== lhs_T.T @ rhs` (atol=rtol=5e-3).
3. **Dependency:** the §3 expectations hold — `Dependency(k8_ir)` builds, the
   per-`ko` `memset(psum)` is accepted, the `ko`→`sbuf_prod` carry + init
   domination are present.

GATE: all three pass → proceed to rewrite `rfactor.py` against the verified
target. Any fail → stop, report the specific failure, reassess. Delete the scout.

---

## 5. Test re-grounding to k0..k8

The suite still targets the previous k0..k14 ladder (references to `kernel_10`,
`kernel_12`, `kernel_14`, `kernel_15`, `kernel_partial`, and the old
`build_ladder_state` rung sequence whose `rung_6_7` relied on a *d2 residual* —
exactly the partial-coverage move §1 now forbids). It is red independent of this
work. Re-grounding:

- **`_fixtures.build_ladder_state`** — rewrite to the new k0..k8 atom trace
  (Split / Reorder / ComputeAt / ReverseComputeAt / RFactor). The exact
  `(transform, literal-nid option)` sequence is **derived empirically** via the
  scout pattern (`_ladder_scout.py`; deterministic build → stable nids), each rung
  sim-checked — not guessed. k0→k4 use shipped transforms; k4→k5, k6→k7 exercise
  the same-prefix rule; k7→k8 the new RFactor.
- **`test_compute_at` / `test_reverse_compute_at`** — re-point at k0..k8; the
  partial-coverage test flips from "documents the residual bug" to "asserts
  same-prefix rejection (→ Split first)".
- **`test_rfactor`** — re-point at the k7→k8 rung; byte-exact gate becomes
  `render(RFactor(k7)) == KT.kernel_8`. The `kernel_rfactor_ko.py` fixture is
  retired (k8 *is* the fixture now); add a precondition-rejection test (flat state
  → loud raise).
- **`test_split`, `test/ir/test_dependency`, `test_software_pipeline`,
  `test/runner/test_output`** — update stale kernel references / ladder-state
  indices to the k0..k8 set.
- **`_ladder_scout.py`, `examples/kernel_transforms_repro.py`** — update to k0..k8
  (drop the stale k9..k14 references).
- **Retire** `examples/matmul_rfactor.py` (its before/after demo is now the
  k7→k8 rung in the ladder).

GATE: full suite green except the 2 known pre-existing `mmdc` failures
(`test_dump_tree_runs_on_canonical_ir`, `test_fuse_tensorize_matmul_n_renders_and_sims`).
All runs remote via `transport/remote_pytest.sh` (`AWS_PROFILE=kaizen-access`);
controller owns remote runs, subagents edit + commit only.

---

## 6. Invariants preserved

- Single return per function; loud failures only (no silent no-op-on-bad-input).
- Transform legality never gates resource capacity (PSUM/SBUF fit) — decision 6.
- `dependency.py` and `_code_motion.py`'s `_move`/`regen_and_rebind` core are
  untouched by RFactor; §1 only *adds* a rejection in `_check_move_realizable`.
- Block-count-changing emission is RFactor's prerogative (it builds the gadget
  blocks); `ComputeAt`/`ReverseComputeAt` remain block-count-preserving.
- The hand fixture encodes the SPEC shape (k8 by hand), never a captured render —
  the byte-exact gate means "matches the reference ladder", per the
  2026-06-09 learnings correction.

## 7. Risks

- **Dependency expectations (§3) wrong.** Mitigated by the §4 hardware gate before
  any `rfactor.py` rewrite. Highest-uncertainty item; do not lock emission first.
- **`build_ladder_state` nid drift.** The trace hardcodes literal nids from a
  deterministic build; if a transform changes node-allocation order the nids
  shift. Mitigated by deriving them from a fresh scout run and sim-checking every
  rung (a wrong nid fails loudly, not silently).
- **Recipe context under-specified.** If `TwoStageContext` misses a field a future
  recipe needs, the registry's `Protocol` makes the gap a type error at the call
  site, not a silent wrong emission. Start minimal (only what `rmw` needs); extend
  when `slot` lands.
