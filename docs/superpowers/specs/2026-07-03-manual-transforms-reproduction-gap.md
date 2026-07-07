# Reproducing `manual_transforms.py` (k0…k27) by transform steps — gap analysis

**Date:** 2026-07-03
**Status:** scoping / investigation (read-only; no code). Answers: "can we reproduce
`examples/manual_transforms.py` by rewriting the transform steps in
`examples/kernel_transforms.py` now?"

## Verdict

**Not exactly — one transform (`BufferLayout`) is unshipped, and the RFactor
ordering differs.** The span-promotion + `CodeMotion`-merge work just shipped fixed
code-motion *legality*; it did not add the missing transform. `kernel_transforms.py`
already reproduces a `kernel_target`-**equivalent** (its k25, all rungs CPU-sim clean)
via a DIFFERENT step sequence, but reproducing `manual_transforms.py`'s exact k0…k27
rung-for-rung needs new capability.

## The two ladders are different sequences to (nearly) the same place

| | `manual_transforms.py` (k0…k27) | `kernel_transforms.py` `_build_ladder` |
|---|---|---|
| Steps | 27 (hand-written kernels) | 25 (transform-driven) |
| RFactor | **LAST** (k26→k27), on fully-tiled **list** form | **EARLY** (step 9), on packed form |
| Buffer form | explicit **`BufferLayout`** steps (packed→list) ×4 | **`compact_shapes`** auto (stays packed) |
| Reaches | two-stage fold + per-tile list tiles | two-stage fold + per-tile packed tiles |

## Rung-by-rung: what `manual_transforms.py` needs vs. what is shipped

Shipped transforms: **Split, Fuse, Reorder, CodeMotion, RFactor, SoftwarePipeline**.
(`CodeMotion` = the merged former ComputeAt/ReverseComputeAt.)

| Rung | Transform (manual's comment) | Shipped? | Note |
|---|---|---|---|
| k0→k1 | Reorder (K>M>N → N>K>M) | ✅ Reorder | |
| k1→k2 | Split (K→ko,ki) | ✅ Split | |
| k2→k3 | Split (M→Mo,Mi) | ✅ Split | |
| k3→k4 | Reorder | ✅ Reorder | |
| k4→k5 | Reorder | ✅ Reorder | |
| **k5→k6** | **Buffer layout** (psum packed→list-of-16) | ❌ **MISSING** | no `BufferLayout` transform |
| k6→k7 | Split (drain d2) | ✅ Split | |
| k7→k8 | Reorder | ✅ Reorder | |
| k8→k9 | Code motion (drain sink) | ✅ CodeMotion | |
| k9→k10 | Split (store d2) | ✅ Split | |
| k10→k11 | Reorder | ✅ Reorder | |
| k11→k12 | Code motion (**store** sink) | ✅ CodeMotion | the dropped-output-guard case; now legal |
| **k12→k13** | **Buffer layout** (sbuf_prod→list) | ❌ **MISSING** | |
| k13→k14 | Split (psum memset d2) | ✅ Split | |
| k14→k15 | Reorder | ✅ Reorder | |
| k15→k16 | Code motion (psum memset sink) | ✅ CodeMotion | init-domination (per-ko) — span-promotion handles |
| k16→k17 | Split (rhs load d2) | ✅ Split | |
| k17→k18 | Split/Reorder (rhs load) | ✅ | |
| k18→k19 | Reorder | ✅ Reorder | |
| k19→k20 | Code motion (rhs load sink) | ✅ CodeMotion | |
| **k20→k21** | **Buffer layout** (sbuf_rhs→list) | ❌ **MISSING** | |
| k21→k22 | Split (lhs_T load d1) | ✅ Split | |
| k22→k23 | Split | ✅ Split | |
| k23→k24 | Reorder | ✅ Reorder | |
| k24→k25 | Code motion (lhs_T load sink) | ✅ CodeMotion | |
| **k25→k26** | **Buffer layout** (sbuf_lhs_T→list) | ❌ **MISSING** | |
| k26→k27 | **RFactor (LAST)** | ⚠️ RFactor exists but **runs EARLY** in the shipped ladder | ordering question |

## The two real gaps

### Gap 1 — `BufferLayout` transform (the packed ↔ list-of-tiles primitive) — UNSHIPPED

4 of the 27 rungs (k5→k6, k12→k13, k20→k21, k25→k26) are "Buffer layout": rewriting a
packed `nl.ndarray((128, T, F))` into a Python `list` of `T` separate
`nl.ndarray((128, 1, F))` tiles. This is a real, distinct transform:

- **Data-model + codegen support EXISTS** (`Buffer.num_tiles`, `per_tile_physical_shape`,
  list-of-tiles `_emit_alloc` / `render_buffer_region` — shipped in the "buffer
  num_tiles + codegen" plan; `num_tiles=1` is byte-identical to packed).
- **No transform SETS `num_tiles`** — the "BufferLayout plan" was written but never
  implemented (ledger: *"No transform sets num_tiles yet (BufferLayout plan does)"*).
- This is the MFU lever ([[nki-buffer-allocation-granularity]]): list = liveness-scheduled
  / multi-buffered; packed = serialized. `kernel_transforms.py` reaches per-tile SHAPES
  via `compact_shapes` but keeps them PACKED, so it is a `kernel_target`-*equivalent* in
  compute body, not byte-identical in buffer form.

### Gap 2 — RFactor ordering (LAST vs EARLY)

`manual_transforms.py` applies RFactor as the FINAL step (k26→k27), on a fully-tiled,
list-form, one-stage-accumulator kernel. `kernel_transforms.py` applies it EARLY
(step 9), and the learnings note *"RFactor must go EARLY (LATE → KeyError)"*. So
`manual_transforms.py`'s RFactor-last order may not be reachable with the shipped
RFactor without work. (This needs a gym-1 probe to confirm whether RFactor-last on the
tiled+list form raises or works — do not assume.)

## What IS reproducible today

`kernel_transforms.py`'s existing `_build_ladder` (25 shipped-transform steps) already
drives canonical → a `kernel_target`-equivalent, **all rungs CPU-sim clean on gym-1**
(verified 2026-07-03). It reaches the two-stage fold + per-tile shapes + streamed loads
— the same COMPUTE as `kernel_target`, differing only in buffer FORM (packed vs list)
and the fold-inlining perf detail. So the *behavior* of the manual ladder's endpoint is
reproducible; the exact *k0…k27 rung sequence and buffer form* is not, pending Gap 1
(and possibly Gap 2).

## Recommendation

To reproduce `manual_transforms.py` step-for-step: **implement the `BufferLayout`
transform** (Gap 1) — it is the single missing primitive behind all 4 unshipped rungs,
and its data-model/codegen substrate already exists. Then resolve the RFactor ordering
(Gap 2) with a gym-1 probe. Both are their own scoped plans (brainstorm → plan →
execute), not a mechanical rewrite of the existing ladder steps.
