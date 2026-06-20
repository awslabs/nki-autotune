# Fold-inlining blocker: why the merged ladder stalls at 46% MFU

**Status:** analysis of a parked perf gap (2026-06-20). The fused RFactor is shipped
and correct (`fbb1f5e`); the single merged ladder `examples/kernel_transforms.py`
reaches a HW-running `kernel_target`-equivalent at **46.08% MFU** (`113d630`,
`5a602a1`). `kernel_target` (hand) is 90.81%. This doc characterizes precisely why
the transform path cannot yet close that ~45pp gap, so the follow-on starts from
evidence, not a fresh investigation.

## What's shipped and correct

`canonical → Split(K) → RFactor(ko) → Reorder×3 → Split-d2×3 → Reorder×3 →
ReverseComputeAt(memset, copy under matmul i_d1_0) → ComputeAt×2 (loads)`. All
k0..k15 + kernel_target CPU-sim PASS (~1.4e-4). HW: k15 46.08%, k13 37.43%,
kernel_target 90.81%. The two-stage fold + per-tile PSUM `(128,1,512)` + streamed
loads are all transform-derived in ONE IR — the old "merge blocker" (multi-slot
RFactor corrupting under Split(M)) is fully resolved.

## The remaining gap is structural, not numeric

`kernel_target` inlines the `tensor_tensor` ko-fold **into the matmul's innermost
loop body** (`for i_d1_1: { matmul; tensor_copy; tensor_tensor }`), so the per-tile
PSUM partial is folded into `sbuf_output` immediately and TensorEngine/Vector work
pipelines across `ko`. The shipped ladder leaves the fold as a **separate `[N,M]`
sweep** after the matmul's ko-body, which (a) serializes the two accumulation
stages and (b) keeps `sbuf_rfactor` full-extent `(128,16,2048)`. That is the bulk
of the 46%→90% gap (plus a secondary one: lhs_T reloaded full-width vs
kernel_target's `(128,1,512)` per-Mo slabs).

## Finding: the N-outermost nest is reachable and RFactor-clean

The shipped ladder uses ko-OUTERMOST (`ko > N > M`). `kernel_target` is
N-OUTERMOST (`N > ko > Mo > Mi > ki`), with `ko` SECOND so the fold can sit at the
innermost M, reducing across the *enclosing* `ko`. **This nest IS reachable**
(probed on gym-1, all sim-clean): from `split_k` do `Split(M→4,4)` then
`Reorder×4` to bubble N outermost then `Reorder×2` to sink ki — reaching
`N > ko > Mo > Mi > ki` while still single-stage — **then `RFactor(ko)` applies
sim-clean there**, emitting the fold inside `ko` at the M level. This is the right
precondition for fold-inlining and should replace the ko-outermost ladder when the
two barriers below are lifted.

## Barrier 1 — reduction-axis-coverage guard rejects covering the fold's ko

`_check_no_reduction_axis_covered` (`transforms/_code_motion.py:127`) rejects ANY
move that covers an ACCUMULATION axis of the moved block. The fold carries `ko` as
ACCUMULATION (its cross-`ko` carry into `sbuf_prod` is the second-stage reduction).
Moving it under the matmul's inner M loop reports `ko` as *covered* → reject.

But `kernel_target` proves covering the fold's `ko` is LEGAL: `sbuf_prod` /
`sbuf_output` is allocated OUTSIDE `ko` (by `init_two_stage_0`), so the init
dominates and re-running the fold per-`ko` correctly accumulates over the ENCLOSING
`ko`. The guard cannot distinguish:
- **covering by an enclosing loop the block legitimately reduces across** (the fold
  over its own outer `ko` — SAFE, the kernel_target structure), from
- **covering by a DIFFERENT block's loop that merely shares the `i_d0_0` name** (a
  producer's prefetch K loop — UNSAFE, init no longer dominates → NaN; the case the
  guard was built for).

The discriminator is whether the covering loop is the one whose carry edge the
block's own init dominates. The guard currently keys only on role==ACCUMULATION +
`target_loops` non-empty, which conflates the two. (This is the same
reduction-axis-coverage area the `tvm_knowledge.md` method note warns has been
flip-flopped on — change it only with a [PROBE], i.e. a gym-1 sim, per case.)

## Barrier 2 — frozen COVER edge from the pre-tiling full-N psum region

Even co-locating just the (PARALLEL, unguarded) `tensor_copy` is rejected:
`move(... ) reorders dependency edge backward`. Root cause: RFactor emits the
`memset(psum)` and drain `tensor_copy` writing/reading the FULL-N region
`psum[tile, 0:2048]` (N is not tiled at RFactor time). After the N/M splits the
matmul writes per-N-tile `psum[tile, i_d2_0*512:+512]`, but the copy still reads
`0:2048`. That width mismatch is a COVER edge (consumer reads wider than the
producer writes per N-iter), frozen in `ir.dependency` at construction. Moving the
copy under the matmul's per-N-tile loop then reads as backward.

The fix is to NARROW the copy/memset's psum region to per-N-tile BEFORE the move —
i.e. they must be split on N (d2→4×512) AND have their region rebased so the inner
`i_d2_1` they own becomes the matmul's shared outer `i_d2_0`. The shipped
ko-outermost ladder sidesteps this by splitting d2 and co-locating under `i_d1_0`
(where N is innermost), which is why it reaches HW-runnable; the N-outermost path
needs the copy to INHERIT the enclosing `i_d2_0` rather than own an `i_d2_1`.

## Recommended follow-on (when resumed)

1. Adopt the N-outermost ladder (probed reachable) as the RFactor precondition.
2. Barrier 2 first (lower risk, PARALLEL op): make the co-location narrow the
   copy/memset psum region to the enclosing N-tile (region rebase), so the frozen
   COVER edge is per-tile and the move is forward. Verify copy co-locates → PSUM
   `(128,1,512)` with the fold still a sweep; measure MFU.
3. Barrier 1 (higher risk): refine `_check_no_reduction_axis_covered` to permit
   covering an ACCUMULATION axis by the loop the block's own init dominates (the
   outer-carried reduction), while still rejecting a foreign covering loop. Gate
   EACH candidate on a gym-1 sim (the guard prevents NaN; a wrong loosening
   silently corrupts). Then inline the fold under `i_d1_1`.
4. Secondary: per-Mo lhs_T reload `(128,1,512)` (a tighter ComputeAt + compact).

GATE for each step: gym-1 CPU-sim PASS + HW MFU measured. Stop if a guard loosening
admits any previously-rejected illegal move (capture golden verdicts first).
