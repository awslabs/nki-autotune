# ComputeAt / ReverseComputeAt — Greenfield Rewrite (region-regen)

## Status (2026-06-02)

NOT STARTED. Supersedes the move-mechanics portions of
`2026-05-29-compute-at-design.md` (Part B/C), which shipped a
pre-`normalize_block` move engine that is silently wrong on tiled IR.
This is a **greenfield rewrite** of both transforms against the current
IR (post-SSA, post-unified-Split, no-trip-1, `normalize_block` as the
universal reconciler).

Part A (region overlap: `interval.py`, `buffer_placement.py`, region-gated
`dependency.py`) and Part C codegen (`compact.py`:
`compact_shapes`/`rebased_region`) are SHIPPED and REUSED unchanged.

## Why a rewrite (not a patch)

The existing `_code_motion._compute_at_impl` + `ReverseComputeAt` predate
this session's IR model. Two defects, confirmed by reproduction:

1. **`_loop_var_to_axis` inverts only bare-`Var` iter_values** — its own
   docstring says "a fully-Split axis is treated as uncovered." On tiled
   IR (`iter_values = i_d1_0*512 + i_d1_1*128` after Split) covered axes
   are missed, so the collapse silently no-ops: a lift "succeeds" but the
   block stays put with a stale loop. On canonical IR it only *looked*
   correct because the moved block's loop names coincidentally matched the
   target's.
2. **No `normalize_block` call, no loud partial-coverage rejection.** The
   move never reconciles names/trip-1/bindings, and an illegal
   partial-coverage move neither moves correctly nor raises — violating
   "all errors must be loud."

`ComputeAt` (forward, the producer-sink) **does not exist at all** — it is
5 of the 7 ladder rungs.

## The hand ladder is the oracle (byte-exact gate)

`kernel_transforms.py` (`kernel_0..kernel_14`, 15/15 CPU-sim pass) is the
ground truth. Every code-motion rung must be reproduced **byte-exact**:
`render(apply(before_IR))` equals the hand `kernel_N` source.

**The comparison is normalize-then-equate** (a NEW shared test helper —
no such helper exists today; the shipped Split "byte-exact" test only
asserts individual rendered substrings, and `test_render_equivalence.py`
only fp32-sims). The helper canonicalizes both sides before `==`:
- strip comments and blank lines;
- `black`-format both (so line-wrapping never causes a false diff);
- rename-normalize the two known skews — the kernel function name
  (`nki_f_matmul` ↔ `kernel_N`) and the accumulator buffer (`psum_prod`
  ↔ `psum_acc`) — to fixed placeholders.

These two name skews are the ONLY tolerated differences; everything else
(loop structure, indices, widths, alloc shapes, op order) must match
character-for-character. The same helper is reused to retrofit the
existing Split/Fuse byte tests onto full-source comparison.

Code-motion transitions (verified against the current file):

| Rung | Transform | Move | Buffer effect |
|---|---|---|---|
| k1→k2 | **ComputeAt** | sink load_lhsT under matmul (d0,d1) | `sbuf_lhs_T`→(128,1,128) |
| k3→k4 | **ComputeAt** | sink load_rhs under matmul d2 | `sbuf_rhs`→(128,1,512) |
| k6→k7 | **ComputeAt** | sink memset under matmul d1 | — |
| k7→k8 | **ComputeAt** | sink both loads under inner d2 | — |
| k9→k10 | **ComputeAt** | sink memset under d2 | — |
| k11→k12 | **ReverseComputeAt** | lift tensor_copy under matmul d2 | **`psum_acc`→(128,1,512)** (PSUM hoist) |
| k13→k14 | **ReverseComputeAt** | lift store under d2 | `sbuf_prod`→(128,1,512) |

(Non-move rungs are already covered: k0→k1/k2→k3/k4→k5/k10→k11/k12→k13
tensorize-Split; k5→k6/k8→k9 Reorder.)

Every ladder rung is **full coverage** (each move is preceded by the Split
that makes the covered dim's extent exact). Region-regen (below) reduces
to wholesale collapse on full-coverage input, so it reproduces these rungs
exactly. The **partial-coverage** capability has no ladder rung, so this
spec **adds one new hand kernel** as its oracle (see Verification).

## Decisions (settled in brainstorming)

1. **Greenfield.** Delete `_code_motion.py` + `reverse_compute_at.py`;
   rewrite fresh. Reuse the *legality logic* (it is correct) and the
   shipped `place_buffers`/`compact_shapes`/`Dependency`/`normalize_block`.
2. **Two faces, one core.** `ComputeAt` + `ReverseComputeAt` are thin
   `Transform` subclasses; one shared `_move(ir, block_nid,
   target_loop_nid, index, is_reverse)`. They differ ONLY in:
   - dependency-check direction — forward checks **consumers** (5a),
     reverse checks **producers** (5b);
   - **output-block guard** — forward only (condition 4).
   The insertion-gap computation `(lp, fc]` is identical for both.
   (Mirrors TVM's `ComputeAtOrReverseComputeAtImpl(..., is_compute_at)`.)
3. **Option = free triple** `(block_nid, target_loop_nid, index)`.
   `analyze` enumerates EVERY legal `index` in the gap `(lp, fc]` as a
   distinct action — not one canonical slot. `-1/-2` remain only as
   hand-authored `apply` conveniences, never emitted by `analyze`.
4. **Region-regen, TVM-style** (NOT full-coverage-only). The move derives
   each moved-block iter-var's domain from the required region and
   regenerates residual loops, so a `range(16)` block sinks under a
   `range(4)` target with an auto-generated `range(4)` residual. Replaces
   the old spec's "full-coverage-only; Split composes" non-goal.

## Architecture

### Files

```
nkigym/src/nkigym/transforms/
├── _code_motion.py         # REWRITE — _move(ir, block_nid, target_loop_nid, index, is_reverse)
│                           #   + region-regen helpers (required-region, solve-domain, regen-loops)
├── compute_at.py           # NEW — ComputeAt, ComputeAtOption (forward; analyze + 6-condition legality)
├── reverse_compute_at.py   # REWRITE — ReverseComputeAt, ReverseComputeAtOption (thin face)
├── __init__.py             # EDIT — export ComputeAt + ComputeAtOption
└── compute_at_legality.md  # EDIT — drop init/NKIAlloc/dst=; refresh "What is a block?" for SSA IR

kernel_transforms.py        # EDIT — add ONE partial-coverage hand kernel (the region-regen oracle)

test/transforms/
├── _ladder_compare.py          # NEW — normalize-then-equate helper (strip comments, black-format, rename-normalize the kernel-fn + psum_acc/psum_prod skews) shared by all byte-exact tests
├── test_compute_at.py          # NEW — forward: byte-exact k1→k2/k3→k4/k6→k7/k7→k8/k9→k10; legality; partial-cover
└── test_reverse_compute_at.py  # REWRITE — reverse: byte-exact k11→k12/k13→k14; PSUM-hoist E2E; legality
```

### Option payloads

```python
@dataclass(frozen=True)
class ComputeAtOption(TransformOption):
    """Sink producer ``block_nid`` under ``target_loop_nid`` at ``index``."""
    block_nid: int
    target_loop_nid: int
    index: int

@dataclass(frozen=True)
class ReverseComputeAtOption(TransformOption):
    """Lift consumer ``block_nid`` under ``target_loop_nid`` at ``index`` (mirror)."""
    block_nid: int
    target_loop_nid: int
    index: int
```

### The six legality conditions (ported from `compute_at_legality.md`)

Unchanged in substance; ported into `ComputeAt._check_legality` /
`ReverseComputeAt._check_legality`, raising `TransformLegalityError`:

1. **Acyclic deps** — precondition (assert).
2. **Block well-formed** — moved block's ISA leaf has no `SEQUENTIAL`-role
   iter_var (`role_of`).
3. **Target not ancestor of block** — `target_loop_nid ∉
   descendants(block_nid)` and vice versa (disjoint root-child subtrees).
4. **Block is not the output** — *ComputeAt only*: the moved producer's
   writes don't include `ir.return_name`.
5. **Dependency preservation** (the directional rule):
   - **5a ComputeAt**: every **consumer** of the moved producer is a
     descendant of `target_loop_nid` OR in a root-sibling with pre-order
     index `> target_root_index`. (uses `Dependency.consumers`)
   - **5b ReverseComputeAt**: every **producer** of the moved consumer is
     a descendant of `target_loop_nid` OR in a root-sibling with index
     `< target_root_index`. (uses `Dependency.producers`)
   Port `_check_producers_visited` / `_root_sibling_of` verbatim; add the
   mirror `_check_consumers_visited` for forward.
6. **Insertion gap exists** — among the target loop body's children, the
   legal range is `(last_producer_position, first_consumer_position]`;
   reject if empty. `analyze` enumerates each slot in it.

### The move mechanic — region-regen (`_move`)

`apply` deep-copies, re-checks legality, calls `_move`, then
`place_buffers` → `compact_shapes` → rebuild `Dependency`.

`_move(ir, block_nid, target_loop_nid, index, is_reverse)` mutates the
tree in five steps:

1. **Required region.** Compute the buffer region the move must cover:
   the union, over the moved block's dependency neighbors *that anchor the
   move* (consumers for forward, producers for reverse) restricted to the
   target's enclosing loops, of the shared tensor's accessed region — as
   an integer set in the target's loop vars. (Our iter_values are simple
   affines `Σ loopⱼ·strideⱼ`, no floordiv/floormod, so the integer-set
   layer is narrower than TVM's general `arith::IntSet`.)
2. **Solve moved-block domains.** For each moved-block iter-var, invert its
   access expression against the required region to solve its `[min,
   extent)` domain — the analogue of TVM's `SolveBlockVarDomain`. A
   covered axis solves to the target's loop var directly; an uncovered
   axis solves to a residual domain.
3. **Regenerate loops.** Build fresh ForNodes from the solved domains
   (TVM's `MakeNewLoop`): a domain with extent 1 contributes no loop (no
   trip-1, per our rule); extent>1 contributes a residual ForNode below
   the insertion point. Rebind the moved block's iter_values + reads +
   writes + leaf operand_bindings to the new loop vars (region arithmetic
   in element space, as `normalize_block` does).
4. **Splice** the moved block (with regenerated residual nest) under
   `target_loop_nid` at `index`, preserving sibling order.
5. **`normalize_block`** on the fork block — drop any trip-1, dense-rename
   loop vars, recompute iter_values + region lo's. This is the universal
   reconciler the old engine skipped; it makes the rendered output match
   the hand ladder's dense names.

On **full-coverage** input (every ladder rung), step 2 solves every
covered axis to the target var with no residual, so steps 3–5 reduce to
"collapse covered loops + splice + normalize" — byte-exact to the hand
kernel. On **partial coverage** (`range(16)` under `range(4)`), step 2
solves the uncovered remainder to a `range(4)` residual that step 3
regenerates.

### Buffer geometry descends automatically

After `_move`, `apply` calls the shipped `place_buffers` (LCA descent) +
`compact_shapes` (bbox shape). The **PSUM hoist** falls out at k11→k12:
once tensor_copy lifts under the matmul's d2, the three `psum_acc`-touching
blocks co-locate, the LCA descends into the matmul block, and
`compact_shapes` shrinks `psum_acc` to `(128,1,512)`. No special-casing.

### Renderer — unchanged

`body.py` already rebases via `compact.rebased_region` and allocates via
`physical_shape`. No renderer change expected; if a rung fails to match
the hand kernel, the fix belongs in `_move`/`normalize_block`, not the
renderer (the renderer is mechanical lowering and stays loud on bad IR).

## Verification

Byte-exact gate per the doctrine "same means same" — build the before-IR,
apply ONE transform, render, compare to the hand kernel name-agnostically,
AND fp32-sim against the numpy golden at `atol=rtol=5e-3`.

`test/transforms/test_compute_at.py` (NEW):
- **Byte-exact forward rungs**: k1→k2, k3→k4, k6→k7, k7→k8, k9→k10. Each
  builds the before-state (canonical + the prior ladder transforms),
  applies one `ComputeAt`, asserts rendered source == hand `kernel_N`.
- **Partial-coverage region-regen** (NEW oracle): build a before-state
  with a moved block over `range(16)` and a target over `range(4)`, apply
  `ComputeAt`, assert it matches the NEW hand kernel (added to
  `kernel_transforms.py`) with an auto-generated `range(4)` residual +
  fp32-sim.
- **Legality rejections**: one per condition (2 SEQUENTIAL-role, 3
  ancestor, 4 output-block, 5a consumer-after-target, 6 empty gap).
- **analyze enumerates every legal index** in a multi-slot gap.
- input-IR-preservation.

`test/transforms/test_reverse_compute_at.py` (REWRITE):
- **Byte-exact reverse rungs**: k11→k12 (the PSUM hoist) and k13→k14.
- **PSUM-hoist assertion**: after k11→k12, `psum_acc` descended into the
  matmul block's `alloc_buffers` AND compacted to `(128,1,512)`.
- **Legality rejections**: 5b producer-before-target violation; ancestor;
  non-ForNode target.
- input-IR-preservation.

`kernel_transforms.py` (EDIT): add one partial-coverage hand kernel,
CPU-sim-verified in `_main`, as the region-regen byte oracle.

## Out of scope

- multi_buffer, software_pipeline, decompose/compose_reduction.
- Multi-leaf compound blocks (each block keeps one ISA leaf).
- Wiring the two transforms into `examples/matmul_lhsT_rhs.py`'s MDP list
  — a trivial follow-up once both pass byte-exact, tracked separately.

## Risk

The integer-set domain solver (steps 1–2) is the one genuinely new piece;
everything else is structural reuse. Mitigation: our affines are far
simpler than TIR's (no floordiv/floormod), the full-coverage path (all 7
ladder rungs) exercises the degenerate case where the solver returns the
target var directly, and the one partial-coverage hand kernel pins the
residual-regeneration path against a byte oracle. Each rung gates on
byte-exact + sim before the next.
