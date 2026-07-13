# RFactor k28→k29 + intermediate-state manual ladder — design

**Date:** 2026-07-07
**Status:** design (awaiting review)
**Goal:** (A) clarify `examples/manual_transforms.py` by hand-authoring the
`kernel_i_intermediate` state that precedes each auto place+compact "side effect,"
and (B) rewrite the `RFactor` transform so a machine-driven `k28→k29` rung
reproduces the hand `kernel_29` byte-exact, CPU-sim clean, and HW-profiled on gym-1.

**Depends on / supersedes:** completes the k28→k29 (formerly k26→k27) RFactor rung
deferred by both `2026-06-25-rfactor-debug-and-correct.md` (Task 2 → "folded into the
BufferLayout plan") and `2026-07-03-buffer-layout-transform-design.md` ("k26→k27
RFactor + Gap 2 — OUT of scope"). Builds on the fused two-stage RFactor of
`2026-06-12-same-prefix-computeat-and-two-stage-rfactor-design.md` (§2), whose
`ko`-anchored emission this generalizes to a `ki`-anchored one.

## Root cause (gym-1-verified, 2026-07-07)

Applying the **shipped** `RFactor(ko)` to the real k28 IR (built via
`kernel_transforms._build_ladder`) does **not** raise — it returns and renders a
**structurally wrong** kernel that CPU-sim rejects:

```
[sim] RAISED AssertionError: Out-of-bound access for tensor `unnamed` on
      dimension 2: index range [512, 1023] exceed dimension size of 512.
```

The three diagnostic states (all `Split(K→ko,ki)`) differ only in where `ki` sits:

| state | matmul nest under `ko` | `ki` position | shipped RFactor |
|---|---|---|---|
| early-packed (`kernel_rfactor_ko.py`, only pinned fixture) | `ko > ki > M > N` | directly under `ko` | correct (byte-exact) |
| mid-packed (sim-only test) | `ko > ki > Mo > Mi > N` | directly under `ko` | correct (sim) |
| **k28** (target) | `ko > Mo > Mi > ki` | **innermost** | **wrong → OOB sim** |

The shipped `_nest_*` / `_splice_block_under_ko` helpers hardcode the early/mid
coincidence — splice gadgets directly under `ko`, fabricate a flat 16-trip `i_d1_0`
partition loop, keep psum a packed/list-16 accumulator, and author a bare `0:512`
free offset. On k28 that (a) mis-places the gadgets (they belong inside `Mi`,
bracketing `ki`), (b) keeps psum list-16 instead of the per-tile list-1 `kernel_29`
needs, and (c) makes the free offset inconsistent with the matmul's `i_d2_0*512` →
the matmul's free offset fails `compact_shapes`'s anchor test → renders un-rebased →
OOB.

The manual `kernel_29` target itself CPU-sims correct (`max_abs=1.37e-04`, gym-1).

## A. Intermediate-state manual ladder (`examples/manual_transforms.py`)

### The split

`CodeMotion.apply` and `RFactor.apply` each end with an automatic buffer step after
the structural edit — `place_buffers` (LCA scope) then `compact_shapes` (shape
shrink) — surfacing today under the `# Side effect: automatic buffer scope
tightening and compaction` comment. Each such rung is therefore **two arrows**:

```
prev  --structural move/emit-->  kernel_i_intermediate  --place_buffers+compact_shapes-->  kernel_i
```

`kernel_i_intermediate` = the structural edit only, with buffers left at their
**prior scope, shape, and indexing**. `kernel_i` = current kernel (post-side-effect),
**unchanged**.

### Rungs split (+6 hand kernels)

Every CodeMotion + RFactor rung, per the locked decision:

| rung | transform | intermediate = | final (unchanged) = |
|---|---|---|---|
| k10 | CodeMotion (drain-sink) | == kernel_10 (tail is a no-op here) | kernel_10 |
| k13 | CodeMotion (store-sink) | store sunk; `sbuf_prod` still `(128,16,2048)`, indexed `[:, i_d1_0, i_d2_0*512:+512]` | `sbuf_prod` `(128,16,512)`, `0:512` |
| k17 | CodeMotion (psum-memset sink) | memset sunk; `psum_prod` still `(128,1,2048)`×16 | `(128,1,512)`×16 |
| k22 | CodeMotion (rhs-load sink) | load sunk; `sbuf_rhs` still `(128,16,2048)` | `(128,8,512)` |
| k27 | CodeMotion (lhs_T-load sink) | load sunk; `sbuf_lhs_T` still `(128,16,2048)` | `(128,8,512)` |
| k29 | RFactor | two-stage control-flow restructure; **k28's buffers verbatim** — psum list-16 addressed `[i_d1_0*4+i_d1_1]`, `sbuf_rfactor` list-16 at `ko`-body scope | psum list-1 `[0]`, `sbuf_rfactor` list-1, both scoped in `Mi` (per-tile) |

k10_intermediate is byte-identical to kernel_10 (added only for the uniform
"every CodeMotion+RFactor rung" rule the user chose).

### Ground-truth authoring (locked rule)

Every `kernel_i_intermediate` is **authored BY HAND** from the spec shape — reasoned
from what the structural move produces, **NOT** captured from a transform's render
(`learnings.md`: "the hand fixture MUST encode the SPEC shape, NOT the transform's
own captured render"). The hand ladder is the ground truth the driven transforms are
later verified against, so it must not be a transform's own output.

k29_intermediate reading (confirmed with user): the two-stage restructure —
`memset(psum)` as `ki`'s preceding sibling, the `ki` matmul nest unchanged,
`tensor_copy(psum→sbuf_rfactor)` + `tensor_tensor(sbuf_prod += sbuf_rfactor)` as
`ki`'s following siblings, all inside `Mi` (`i_d1_1`) — with k28's buffers retained
(psum list-16 addressed `[i_d1_0*4+i_d1_1]`, sbuf_rfactor list-16). `kernel_29` is
then the post-tighten state (psum list-1 `[0]`, sbuf_rfactor list-1, both scoped in
`Mi`) — the existing hand kernel, unchanged.

### Discovery + sort

Widen the discovery regex `^kernel_\d+$` → `^kernel_\d+(_intermediate)?$` in
`manual_transforms.py::_discover_kernels`, and sort by `(int_id,
intermediate-before-plain)` so `kernel_13_intermediate` precedes `kernel_13`.
`_kernel_source` (AST extraction by name) already works for any function name.

### Verification (gym-1)

Run `manual_transforms.py` on gym-1. It does two things per kernel, intermediates
included (the existing `_main` → `_check_numerics` + `_profile_on_hw` path picks up
every discovered kernel with no harness change):

1. **CPU-sim** (`simulate_fp32`, fp32) vs the numpy golden — **the only exit-non-zero
   gate.** Every kernel, intermediate and final, must `pass=True`.
2. **HW profile** on gym-1 (`autotune.runner.profile`) — compiles + runs + records
   MFU/latency and compile/run status for **every** kernel, intermediates included.
   Note `profile()` does NOT numerically validate against a golden (no `allclose` in
   the pipeline — see `autotune/runner/api.py`/`driver.py`); it reports HW
   compile+execution success and MFU. Correctness therefore stays with CPU-sim (1);
   the HW run adds "does it compile+run on gym-1, and at what MFU."

The HW result is **reported, not gated** for the intermediates: several revert a
compaction to a full-extent PSUM/SBUF buffer that BIR-verify rejects BY DESIGN (the
locked "transform legality never gates resource capacity — full-extent-buffer rungs
exit 70 at BIR-verify, the intended pruning" rule), so a blanket HW-pass bar would
contradict that invariant. Those failures surface in the profile failure summary and
do NOT abort the run. The **final** kernels keep their existing expectation (the k28
endpoint + k29 near the ~90% champion MFU). Concretely, the intermediate HW outcomes
are recorded in `<cache>/manual_transforms/` and read off the printed table — expect
the full-extent-PSUM intermediates (k10/k13, packed `(128,16,2048)` psum) and
full-extent-SBUF intermediates (k22/k27, `(128,16,2048)`) among the non-fatal
failures; the k17/k29 intermediates' fit is confirmed empirically by the run, not
asserted up front.

## B. RFactor rewrite: gadgets bracket `ki`, sized to the `ki`-subtree footprint

### Principle

Per the user's framing, the only structurally-relevant loops are `ko` (factored,
given) and `ki` (the innermost reduction loop, whose sole descendant is the run-op).
Everything else is opaque body. RFactor emits the fused two-stage form
(`2026-06-12` §2.1) but anchors the gadgets to **`ki`**, not `ko`:

```
init_two_stage_0()      # retarget pre-ko memset: psum → sbuf_prod (before ko; position unchanged)
for ko:
    ...opaque body (Mo, loads)...
    init_two_stage_1()  # memset(R)                       -> inserted as ki's PRECEDING sibling
    for ki: run_op()    # matmul — UNCHANGED
    drain_two_stage_0() # copy(R→sbuf_rfactor); fold(sbuf_prod = add(sbuf_prod, sbuf_rfactor))
                        #                                  -> inserted as ki's FOLLOWING siblings
drain_two_stage_1()     # None
```

"Beside `ki`" = child of `ki`'s parent ForNode. In early-packed that parent **is
`ko`**, so this reduces to the shipped under-`ko` splice → `kernel_rfactor_ko.py`
stays byte-exact. In k28 it is `Mi` → the `kernel_29` shape.

### Footprint R (the one derived quantity)

R = the accumulator region the `ki`-subtree writes over one full execution of `ki`
(enclosing loops fixed). It is read off the loops **strictly between `ki` and the
matmul leaf** and the matmul `dst` region:

- **partition(`d1`)-role loops** between `ki` and the matmul → **materialized** as
  gadget partition loops (early-packed: the `M`(16) loop → today's 16-trip sweep).
- **free(`d2`)-role loops** between `ki` and the matmul → **absorbed** into the op
  free width (memset / tensor_copy / tensor_tensor free cap ≥ 2048 → one wide op;
  early-packed → 2048-wide, matching `kernel_rfactor_ko.py`).
- **k28: no loops between `ki` and matmul** → R = a single `(128, 512)` tile;
  gadgets are loopless single ops; `psum_prod` / `sbuf_rfactor` are per-tile.

This subsumes the shipped hardcoded `m_var="i_d1_0"` / `m_tiles=M//128` / packed
`free_extent`, which are exactly R for the early-packed nest.

### The matmul leaf is never rewritten

Gadgets and the `psum_prod` / `sbuf_rfactor` transient decls are placed at `ki`'s
scope; RFactor's tail (`place_buffers` + `compact_shapes` + the renderer's
`rebased_region`) then tightens k28's `psum[i_d1_0*4+i_d1_1, i_d2_0*512:+512]` to the
rendered `psum_prod[0][0:128, 0, 0:512]` with **no surgery on the run-op**, honoring
"only `ko`/`ki` matter."

### The three buffers' `list_len`, and the `compact_shapes` fix

k28→k29 changes buffer *layout*, not just scope/shape. The three transients:

| buffer | k28 | k29 (target) | how it changes |
|---|---|---|---|
| `psum_prod` | list-16 `(2048,512)` (16 live M-tiles across the whole `ko` body) | **list-1** `(128,512)` (one live tile inside `Mi`) | scope descends to `Mi` (place_buffers) → live footprint 16 tiles → 1 → `compact_shapes` shrinks leading axis AND `list_len` |
| `sbuf_prod` | list-16 `(2048,512)` | **list-16** (unchanged — the cross-`ko` accumulator, all 16 M-tiles live over `ko`) | untouched |
| `sbuf_rfactor` | — | **list-1** `(128,512)` | created by the emission with `list_len=1` |

**The `compact_shapes` gap (gym-1-verified) and its fix.** When `psum_prod`'s scope
descends into `Mi`, its live footprint collapses from 16 M-tiles to 1, so
`compact_shapes` shrinks its leading logical axis `2048→128` (tile-count T: 16→1).
But `compact_shapes` today recomputes only the logical *shape* and never touches
`list_len`; with `list_len` still 16 and T now 1, `per_tile_physical_shape` asserts
*"list_len 16 does not divide tile-dim 1."* **Fix: teach `compact_shapes` to shrink
`list_len` together with the leading tile-count axis** — exactly the remedy the
BufferLayout spec pre-authorized ("If `compact_shapes` ever DOES mis-shrink a list
buffer's leading dim, fix it there — compact the tile axis in `list_len`-consistent
units, shrinking `list_len` explicitly — loud, no silent clamp"). Concretely: when
`_compact_one` reduces a buffer's leading (partition-tile) extent, clamp its
`list_len` to `min(list_len, new_T)` so `list_len` divides the new T (loud-assert if
it does not divide cleanly). This is the ONLY list buffer in the whole ladder whose
leading axis shrinks (the BufferLayout spec confirms every other leading-axis shrink
happens on a packed buffer before it is listed), so the blast radius is this one case.

**Consequence for the split (§A).** With the fix in `compact_shapes`, RFactor's tail
stays the plain `place_buffers` + `compact_shapes` pair (symmetric with
`CodeMotion.apply`), and the `k29_intermediate → kernel_29` side-effect arrow is
uniform with the CodeMotion rungs: `kernel_29_intermediate` carries **k28's buffers
verbatim** (psum list-16, addressed `[i_d1_0*4+i_d1_1]`; `sbuf_rfactor` list-16,
declared at the `ko`-body scope) with only the control flow restructured; the side
effect then does scope-descent + shape-shrink + `list_len` 16→1 together to reach
`kernel_29`.

**Tail order (in `_emit_rmw`):** after splicing gadgets, retargeting init, and
creating `sbuf_rfactor`, run (1) `place_buffers`, (2) `compact_shapes` (now shrinks
psum shape AND `list_len`), (3) rebuild `Dependency`. For the **early-packed** nest R
spans 16 partition tiles under a materialized loop and the psum stays its existing
single packed instance (no leading-axis shrink), so `compact_shapes` changes no
`list_len` and the result matches today's byte-exact `kernel_rfactor_ko.py`.

### Surface unchanged

`analyze` / `_rfactorable` / `_check_legality` are unchanged — still "a ForNode
binding an ACCUMULATION axis of an rmw op with a REDUCE_COMBINATOR." Only the
internal emission (`_nest_memset` / `_nest_copy` / `_nest_combine` /
`_partition_block` / `_partition_region` / `_splice_block_under_ko`) is rewritten to
the `ki`-anchored, footprint-R form. `RFactorOption` is unchanged.

### Legality note (span-promotion, unchanged)

The emitted shape is the same fused two-stage form the span-promotion dependency
model already validates: the `memset(psum)` sits as `ki`'s **preceding sibling**
(before the `ki` loop, not inside it) — an ordinary per-partial init, since `ki`
accumulates in PSUM via HW `+=` and the memset dominates rather than re-inits that
accumulation; the `ko`-carried reduction re-emerges on the `tensor_tensor` fold
(`sbuf_prod`), and `init_two_stage_0` dominates it. No change to `dependency.py`.
RFactor rebuilds `Dependency` at the end of `apply` as it does today.

## Verification plan (gym-1, in order)

1. **No regression:** `test/transforms/test_rfactor.py` — `test_apply_byte_exact`
   (early-packed vs `kernel_rfactor_ko.py`) + the mid-packed sim test stay green.
2. **New byte-exact gate:** drive the k28 IR (`kernel_transforms._build_ladder`
   endpoint) → `RFactor(ko)` → `assert_matches_hand(render(...), kernel_29)` +
   CPU-sim clean. Both references kept (early-packed AND k28→k29), per the locked
   decision.
3. **Driven ladder:** append the RFactor rung to `kernel_transforms._build_ladder`;
   the full driven ladder is byte-exact to manual k0…k29 (finals) + CPU-sim clean +
   HW-profiled. Intermediates have no driven counterpart yet — the byte-exact gate
   maps each driven rung to its `kernel_i` (finals only); this is a manual-ladder +
   sim refactor for section A (the driven side is untouched there).

## Scope

| # | Change | Files |
|---|--------|-------|
| A1 | Hand-author 6 `kernel_i_intermediate` kernels | `examples/manual_transforms.py` |
| A2 | Widen discovery regex + sort | `examples/manual_transforms.py::_discover_kernels` |
| A3 | gym-1 CPU-sim gate (intermediates may fail HW compile — non-fatal) | run only |
| B1 | Rewrite `_emit_rmw` emission to `ki`-anchored + footprint-R; tail = `place_buffers` + `compact_shapes` + `Dependency` | `nkigym/src/nkigym/transforms/rfactor.py` |
| B2 | Teach `compact_shapes` to shrink `list_len` with a shrinking leading tile-count axis (loud on non-divisor) | `nkigym/src/nkigym/codegen/compact.py` |
| B3 | New k28→k29 byte-exact + sim test; keep early-packed green | `test/transforms/test_rfactor.py`, `test/transforms/_rfactor_fixtures.py` |
| B4 | Append the RFactor rung to the driven ladder | `examples/kernel_transforms.py` |
| B5 | Unit test for the `compact_shapes` `list_len` shrink | `test/transforms/test_buffer_layout.py` or `test/codegen/test_compact.py` |

Out of scope: driven-ladder reproduction of the `_intermediate` states (would need a
structural-only apply path on CodeMotion/RFactor — a separate future change); the
`slot` reduction recipe; `versions × list_len` composition.

## Invariants preserved

- Single return per function; loud failures only (no silent no-op-on-bad-input).
- Transform legality never gates resource capacity (the full-extent intermediate
  kernels are valid outputs; HW profiling prunes them).
- The matmul run-op leaf is never rewritten by RFactor (region surgery is confined
  to the gadgets + the automatic place/compact tail).
- Hand fixtures encode the SPEC shape, authored by hand, never a captured render.
- `analyze` / `apply` / `RFactorOption` signatures unchanged; `dependency.py`
  untouched.
