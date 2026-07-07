# `BufferLayout` transform + k0→k26 reproduction — design

**Date:** 2026-07-03 (updated 2026-07-04)
**Status:** design (awaiting review)
**Goal:** generalize the buffer tile-axis representation to an arbitrary
`list_len × per_tile` factorization, ship the `BufferLayout` transform that
**re-factorizes** a buffer's existing tiles into any such form, and use it to reproduce
`examples/manual_transforms.py` **k0→k26 byte-exact**, driven by the shipped transforms.
k26→k27 (RFactor) and Gap 2 (RFactor ordering) are explicitly OUT.

**Decisions locked (user):**
- **Uniform-list rendering** (option A) — EVERY sbuf/psum buffer renders as a Python
  list, including `list_len == 1` (a list-of-one), so codegen has ONE path and no
  bare-`nl.ndarray` special-case. **Verified perf-neutral on gym-1** (2026-07-04): the
  `manual_transforms.py` A/B probes k28 (= k27 with its two single-tile b=1 buffers as
  list-of-1) and k29 (= k25 with its packed `(128,8,512)` b=1 buffer as list-of-1)
  matched their bare originals within noise — k27 90.79% vs k28 90.77%, k25 75.11% vs
  k29 75.27%, identical latency, matching CPU-sim max_abs. So the list-of-1 wrapper
  compiles to the same NEFF. **Consequence:** the hand reference `manual_transforms.py`
  is rewritten to uniform-list form (done alongside this spec) and byte-exact is
  established against that machine-canonical reference.
- **`BufferLayout` conserves the total tile count** — it only redistributes what a
  buffer already holds across `(a, b)`; it never creates tiles (no double-buffering —
  that is `versions`, owned by SoftwarePipeline / a future multi-buffer transform).

## The generalized tile-axis model (a·b factorization)

A SBUF/PSUM buffer's physical shape is `(P, T, F)` where `P` is the partition extent
(**P ≤ 128**, the hardware max — see "Partition axis" below), `T` is the tile count (the
middle physical dim), and `F` the free extent. `BufferLayout` factorizes the **tile
axis** `T` — orthogonal to `P`, which it never touches:

```
list_len (b) × per_tile (a)  =  T        # a Python list of b ndarrays, each (P, a, F)
```

and a tile `t ∈ [0, T)` is addressed `buf[t // a][0:P, t % a, F]`. The two endpoints
already work today; the middle (`a>1 AND b>1`) is the generalization this plan adds:

| form | a (per_tile) | b (list_len) | render | status |
|---|---|---|---|---|
| packed | T | 1 | `buf[0][0:P, t, F]` (list-of-1) | render change (was bare `nl.ndarray`) |
| general | a | b, `a·b=T` | `buf[t//a][0:P, t%a, F]` | **NEW (renderer rejects a>1)** |
| full split | 1 | T | `buf[t][0:P, 0, F]` | ships (unit-tested) |

Every row is a list — `b=1` is a list-of-one (`buf[0]`), not a bare ndarray. This is
option A (uniform rendering), verified perf-neutral above.

**Naming:** the shipped field `Buffer.num_tiles` already means **b** (list length):
`_emit_alloc` emits `for _ in range(num_tiles)` and `per_tile_physical_shape` returns
`(P, T // num_tiles, F)`, so `a = T // num_tiles`. This plan renames
`num_tiles → list_len` for honesty (it is NOT the total tile count; it is b). Bounded
mechanical rename: 23 references across `tree.py`, `codegen/body.py`, `codegen/compact.py`,
`test/ir/test_node_labels.py`, `test/codegen/test_body.py`.

### Partition axis: P ≤ 128, currently hardcoded to exactly 128

`P` is the NeuronCore partition extent, whose hardware MAX is 128 — a buffer may use
FEWER (a matmul with M < 128, a reduction over a short axis, etc.). The shipped code
today treats it as **exactly** 128, hardcoded in five places: `physical_shape`
(`tree.py:161-163`, which also asserts `leading % 128 == 0` — so P < 128 is not even
representable), the renderer (`body.py:329,338,341`: literal `0:128` + a
`hi.value == PARTITION_DIM` assert), compaction (`compact.py:129-131`), and interval
analysis (`interval.py:118-121`).

**This is orthogonal to `BufferLayout`.** The transform factorizes the tile axis `T`
and passes the partition slice through unchanged (`0:P`); it neither reads nor writes
`P`. So the transform, its option, legality, and conservation are P-agnostic as written
— substitute `P` for the literal `128` in this doc's slices and nothing else changes.
Whether the shipped `PARTITION_DIM`-is-128 hardcoding should become a per-buffer `P ≤ 128`
is a **separate, pre-existing concern** (it predates this plan and affects
`physical_shape`/render/compact/interval, none of which `BufferLayout` modifies). It is
called out here for correctness of notation but is **out of scope** — see "Out of scope".

**Why general `a·b`:** each factorization of `T` is a distinct on-chip multi-buffer
granularity — the autotuning knob ([[nki-buffer-allocation-granularity]]): b = number of
independently liveness-scheduled tiles, a = co-resident degree per list slot. The manual
k0→k26 ladder itself only uses the two endpoints (`a=1` full split, `b=1` packed) — no
intermediate `a>1` rung — so general `a>1` is capability for the IR/transform model
beyond what byte-exact k0→k26 strictly needs, but it is the right first-class model and
costs little once the renderer handles `t//a` / `t%a`.

## Conservation — `BufferLayout` re-factorizes, never creates tiles

`BufferLayout` **conserves T** (the total tile count = `a·b`). It only redistributes a
buffer's existing tiles across the `(a, b)` factorization:

- `[P,2,F]×4` (a=2,b=4, T=8) → `[P,1,F]×8` (a=1,b=8, T=8) ✓ — same 8 tiles, regrouped.
- `[P,2,F]×4` → `[P,2,F]×8` (T=16) ✗ — that CREATES 8 new tiles (double-buffering).

Creating tiles is a *different lever* (buffer versions / multi-buffer degree), owned by
**SoftwarePipeline** (sets `Buffer.versions`) and any future multi-buffer transform — NOT
`BufferLayout`. This is why `analyze` offers only divisors of the CURRENT T and `apply`
holds `versions` fixed: the transform is closed over the set of layouts with the same
total storage. The `versions>1` rejection (below) enforces the boundary structurally —
a versioned buffer's tile count already encodes double-buffering, so re-factorizing it
would entangle the two levers.

## Uniform-list rendering (option A — one codegen path)

EVERY sbuf/psum buffer renders as a Python list. A `b=1` buffer is a list-of-one:
`buf = [nl.ndarray((P,T,F), …) for _ in range(1)]`, accessed `buf[0][0:P, t, F]`. There
is NO bare-`nl.ndarray` branch — `_emit_alloc` and `render_buffer_region` always emit the
list form, `list_len` (b) driving the list length and `a = T // b` the per-tile middle.

Why this over the bare-`b=1` special-case: it removes a renderer branch and makes the
IR's "buffers are stored as lists" model literal in the codegen. The cost — the hand
`manual_transforms.py` no longer matches (it used bare ndarrays for every b=1 buffer) —
is paid once by rewriting that reference to uniform-list form (a human wouldn't write
`for _ in range(1)`, so the reference is machine-canonical, which is already the repo's
stance: "Renderer output IS the canonical style — normalize hand kernels to it"). The
**HW probe above proved this changes no NEFF / no MFU**, so the rewrite is safe.
`shared_hbm` params/outputs keep their bare 2D form (no tile axis; never listed).

## Why k0→k26 is the clean cut

RFactor is applied ONLY at the final k26→k27 rung. Every one of the 26 rungs k0→k26
is one of exactly four transforms:

| Transform | Rungs | Count | Shipped? |
|---|---|---|---|
| Reorder | k1, k4, k5, k8, k11, k15, k19, k24 | 8 | ✅ |
| Split | k2, k3, k7, k10, k14, k17, k18, k22, k23 | 9 | ✅ |
| CodeMotion | k9, k12, k16, k20, k25 | 5 | ✅ (merged, span-promotion) |
| **BufferLayout** | k6, k13, k21, k26 | 4 | ❌ **this plan** |

So k0→k26 needs exactly one new transform and **completely sidesteps Gap 2**: there
is no RFactor in the chain, so the "RFactor EARLY vs LAST" question never arises. k26
is the single-accumulator, fully-tiled, all-list-buffer state — the pre-RFactor
endpoint.

**Implementation finding (2026-07-04, shipped):** driving this on gym-1 revealed two
of the manual "# Reorder" rungs (k0→k1 matmul N-bubble, and the rhs-load N-bubble)
each move a loop up TWO levels, but the shipped `Reorder` only swaps an ADJACENT
parent-child pair. Each therefore needs two atomic Reorders, and the manual ladder was
normalized to carry the intermediate kernel (per the user: "fill the manual ladder with
the atomic reorder missing steps"). Net: the manual ladder is now **k0…k29** (RFactor
at k29; +2 vs the original k0…k27), the pre-RFactor endpoint is **k28** (was k26), and
BufferLayout rungs are **k7/k14/k23/k28**. The driven ladder is 28 transforms
reproducing k0…k28 byte-exact + CPU-sim + HW-MFU (shipped `33b4c44`). "k0→k26"
throughout this doc predates that finding; read it as "k0→k28 (the RFactor-free
prefix)". Also normalized: two decorative `"""init_one_stage"""` string-literals in the
endpoint kernel that the renderer does not emit.

The rung-type table above comes from the read-only gap analysis
(`2026-07-03-manual-transforms-reproduction-gap.md`). "Uses a shipped transform type"
is NOT proof the shipped transform emits byte-identical output at that rung — the
learnings are explicit that this is unprovable short of byte-exact repro. **Driving the
chain and diffing each rung on gym-1 is the proof**; a diverging rung is a finding for
implementation (a transform gap, or a manual kernel needing normalization), not a
failure of this plan's scope.

## The `BufferLayout` transform

### Key fact

`BufferLayout` is a **pure field-set**: it sets one buffer's `list_len` and changes
neither regions nor tree structure — only allocation granularity. The `BufferRegion`
carries the tile index `t` in iter-var space; the renderer projects it to
`buf[t//a][0:P, t%a, F]` from `list_len` alone (`a = T // list_len`). Setting the
field flips the render with zero region surgery — the same mechanism by which
`SoftwarePipeline` sets the sibling `Buffer.versions`.

The data model already stores the factorization (`Buffer.num_tiles` = b,
`per_tile_physical_shape` = `(P, T//b, F)`; commits `9fbd180`, `6351f08`). Codegen
ships the `a=1` full-split endpoint (unit-tested) but **rejects the general `a>1`
middle** (`body.py:331-336`) and today emits `b=1` as a bare `nl.ndarray` — both
addressed by the renderer work below (general `a>1` + uniform list-of-1). No transform
sets the field today (grep: zero assignments).

### Option

```python
@dataclass(frozen=True)
class BufferLayoutOption(TransformOption):
    tensor: str      # buffer name to relayout
    list_len: int    # target b; a = T // b is derived. 1 = packed, T = full split.
```

`list_len` (b) is the sole knob; `a` is always `T // b`. One field keeps the option
hashable and the invariant `a·b = T` structural.

### `analyze(ir) -> list[BufferLayoutOption]`

For each buffer with `location in ("sbuf", "psum")` and `versions == 1`, let
`T = buf.physical_shape()[1]`. Offer `(tensor, b)` for **every divisor b of T** with
`b != buf.list_len` (skip the current layout — no-op). T=16 → b ∈ {1,2,4,8,16}; each is
a distinct multi-buffer granularity. Nothing for `shared_hbm` (no tile axis),
`versions > 1` (does not compose), or `T == 1` (only b=1 exists).

### `apply(ir, option) -> KernelIR`

1. Re-check legality; raise `TransformLegalityError` on violation (loud, no recovery).
2. `new_ir = deepcopy(ir)`.
3. `_set_list_len` — replace the owning block's `alloc_buffers` entry via
   `replace(buf, list_len=...)`, mirroring `SoftwarePipeline._set_versions` (a buffer
   lives on exactly one block).
4. `new_ir.dependency = Dependency(new_ir.tree)` — contract uniformity; a no-op in
   effect (regions unchanged), documenting that buffer layout does not affect deps.

No `place_buffers` / `compact_shapes` rerun inside `apply`: the buffer neither moves nor
changes logical shape at the moment of relayout (matches `SoftwarePipeline`).

### Legality (structural only)

Reject, loud, via `TransformLegalityError`:

- `tensor` not declared in any block.
- `buf.location == "shared_hbm"` — no tile axis.
- `buf.versions > 1` — does not compose (two distinct tile-dim multipliers).
- `T % list_len != 0` — b must divide T (else `a` is not integral).
- `list_len == buf.list_len` — no-op.

Per the locked learning, legality gates correctness / dep-order / ISA-wellformedness
ONLY, never resource capacity. Behavior is provably unchanged (same bytes, same order);
there is no numeric or dependency concern — the checks are all structural renderability
guards.

### Renderer work (general a>1 middle + uniform b=1 list-of-1)

Two changes to `codegen/body.py`, both removing special-cases:

1. **Uniform list (drop the bare-`b=1` branch).** `_emit_alloc` today emits a bare
   `nl.ndarray` when `num_tiles == 1` (`body.py:246`) and `render_buffer_region` renders
   the packed `buf[0:P, t, F]` form; make BOTH always emit the list — `b=1` becomes
   `[… for _ in range(1)]` accessed `buf[0][…]`. `shared_hbm` stays bare (no tile axis).
2. **General `a>1` middle.** `render_buffer_region` emits the tile index as a leading
   `[t]` subscript with a literal middle `0`, guarded to per-tile middle == 1
   (`body.py:331-336`). Generalize: leading subscript `t // a`, middle index `t % a`.
   Both are non-affine in `t` when `a>1` (like the pipeline `% versions` rotation), so
   they render via the non-normalising `_format_raw` path, not `to_affine`. When the
   region's tile expr aligns with the split — `t = outer·a + inner` (e.g.
   `i_d1_0·4 + i_d1_1`, `a=4`) — the arith substrate must fold
   `(outer·a+inner)//a → outer` and `%a → inner`, giving the natural
   `buf[i_d1_0][0:P, i_d1_1, F]`.

`_emit_alloc` already emits `(P, a, F)` per tile via `per_tile_physical_shape`. Gate:
unit tests for (1) `b=1` → list-of-1 alloc + `buf[0][…]` access, and (2) `a>1` covering
an aligned index (folds clean) and a bare `t` (emits literal `t//a`, `t%a`).

## List-form composability through the chain (in scope — traced against the target)

`compact_shapes` reruns only inside CodeMotion (+ RFactor) — Split/Reorder/Fuse do not.
The CodeMotion rungs are **k9, k12, k16, k20, k25**. `psum_prod` becomes a list at k6,
so from k9 on, every CodeMotion reruns `place_buffers` + `compact_shapes` with a list
buffer present. This interaction is **untested** (no transform sets `list_len` today),
so it must be verified — but tracing the manual kernels shows it is EXERCISED ONLY IN
ITS SAFE FORMS, and what "safe" produces is known:

- **`place_buffers`** — safe: `replace`s whole `Buffer` objects, so `list_len` survives;
  never references it.
- **`compact_shapes`** — recomputes each buffer's logical 2D shape (so `list_len`
  survives the `replace`) but *without* consulting `list_len`. From `_axis_span`
  (`compact.py:113-132`), two cases:
  - **Free-axis compaction** — changes only the free dim; `list_len` still divides
    `leading//128`. **Safe.** This is the one shape-change to a list buffer in the whole
    ladder: **k15→k16** shrinks `psum_prod` `(128,1,2048)×16 → (128,1,512)×16` (memset
    sunk under `i_d2_0`). k9/k12/k20/k25 leave list buffers idempotent.
  - **Leading/partition-axis compaction** — `_axis_span` returns `(hi+1)*128`; if a
    CodeMotion made a tile-indexing loop a compaction *anchor*, the leading dim would
    shrink below `list_len × a × 128` and `per_tile_physical_shape` would assert. **This
    NEVER fires in the manual target**, for two structural reasons:
    1. The sbuf operands are compacted while still PACKED, then listed as the terminal
       op on them: `sbuf_prod` shrinks at k12 (packed) → lists at k13; `sbuf_rhs` shrinks
       to `(128,8,512)` at k20 (packed, leading 16→8) → lists at k21; `sbuf_lhs_T`
       likewise k25→k26. Every leading-axis shrink is on the packed path.
    2. `psum_prod` (a list before it is compacted) keeps all 16 M-tiles live (it is the
       accumulator over the full M range), so `list_len=16` is invariant; only the free
       axis shrinks.
    The subtle safe case is **k25**: `sbuf_rhs[i_d0_1]` is written by the load and read
    by the matmul in DISJOINT `i_d0_1` loop-nests (distinct `ForNode` nids, same name).
    `_anchor_loop_vars` intersects ancestor *nodes*, not names (`compact.py:59`), so
    `i_d0_1` is not an anchor and the list-of-8 survives — exactly why the hand kernel
    keeps `sbuf_rhs` as `[…]×8` at k25 rather than collapsing to one tile.

**Task:** confirm the driven chain reproduces this. Unit-test the k16 case
(`compact_shapes` shrinks a list buffer's free axis, `per_tile_physical_shape` stays
consistent, `list_len` preserved) and the k25 case (disjoint-nest tile axis is not
anchored → list length survives). If `compact_shapes` ever DOES mis-shrink a list
buffer's leading dim, fix it there (compact the tile axis in `list_len`-consistent
units, shrinking `a` and/or `list_len` explicitly) — loud, no silent clamp. Gate before
wiring the full chain.

## The driven ladder (k0→k26) and the byte-exact gate

`examples/kernel_transforms.py` is the home for this (per the locked "three examples"
learning: it drives shipped transforms to reproduce the manual ladder 1-to-1). Its
current `_build_ladder` reaches a target-*equivalent* via a DIFFERENT order that applies
RFactor EARLY; k0→k26 has no RFactor, so this is a **new 26-step chain in manual order**
(replacing / superseding the current `_build_ladder`). Each step is a
`Split / Reorder / CodeMotion / BufferLayout` on hand-picked stable nids
(`build_initial_ir` is deterministic → nids hardcodable, per the repo idiom).

The byte-exact proof — currently the example only CPU-sims, it does NOT diff against the
manual kernels — wires the existing oracle `assert_matches_render` (from
`test/transforms/_ladder_compare.py`, AST-canonical) between each driven rung's render
and the corresponding `manual_transforms.kernel_N` source. Per rung:

1. `assert_matches_render(render(driven_kᵢ), source_of(manual kernel_i))` — byte-exact.
2. CPU-sim the render in fp32 vs `lhs_T.T @ rhs` (unchanged from today).
3. HW-profile the render on Trn2 (`autotune.runner.profile`, scheduler + linear-scan
   OFF — same path `manual_transforms.py` uses) and assert the driven rung's MFU matches
   the corresponding manual kernel's within noise (byte-exact renders ⇒ identical NEFF ⇒
   identical MFU; this is the end-to-end confirmation, and it also re-proves the k26
   endpoint reaches the champion's ~90% rather than merely CPU-sim-clean).

The "manual kernel_i source" is the **uniform-list `manual_transforms.py`** (rewritten
alongside this spec: every sbuf/psum buffer a list, `b=1` as list-of-1). The rewrite is
the machine-canonical reference the driven ladder must reproduce.

### Gate (gym-1, via `transport/ssh_host.sh`)

- Unit: `test/codegen/test_body.py` — `b=1` renders as list-of-1 (`[… for _ in
  range(1)]` + `buf[0][0:P, t, F]`); general `a>1` render (aligned index folds to
  `buf[outer][…, inner, …]`; bare `t` emits `buf[t//a][…, t%a, …]`). Existing packed
  `b=1` assertions are updated to the list-of-1 form.
- Unit: `test/transforms/test_buffer_layout.py` — analyze enumerates every divisor of T;
  apply sets `list_len`; every reject path; round-trip identity; conservation (T
  unchanged across any apply); one standalone k5→k6 `assert_matches_render` against
  manual k6.
- Unit: list-form compaction — the k16 case (free-axis shrink of a list buffer
  preserves `list_len` + `per_tile_physical_shape`) and the k25 case (disjoint-nest tile
  axis is not a compaction anchor → list length survives).
- Example: all 26 driven rungs `assert_matches_render` to manual k0…k26 AND CPU-sim
  clean AND HW-MFU-match each manual rung within noise (the runnable rungs — the
  full-extent-buffer rungs the BIR verifier rejects are expected non-compiles, same as
  `manual_transforms.py`). This is the deliverable proof of "reproduce k0→k26".

## Build sequencing (each gated on gym-1 before the next)

0. **Rewrite `manual_transforms.py` to uniform-list form** — every sbuf/psum buffer a
   list, `b=1` as list-of-1 (`[… for _ in range(1)]`, `buf[0][…]`); `shared_hbm` bare.
   Establishes the machine-canonical reference the byte-exact gate diffs against. Gate:
   all kernels CPU-sim clean + HW-profile unchanged vs the pre-rewrite MFU (the probe
   already showed neutrality; this confirms it across all rungs). **Done alongside this
   spec** — probes k28/k29 deleted after they served their purpose.
1. **Rename `num_tiles → list_len`** across the 5 files (mechanical; tests stay green —
   pure rename, no behavior change).
2. **Generalize the renderer** — uniform list-of-1 for `b=1` (drop the bare-`nl.ndarray`
   branch) AND general `a>1` (`t//a`, `t%a`; arith folding of aligned indices). Gate:
   the codegen unit tests above. NOTE: this is the render change that makes the shipped
   codegen match the step-0 reference; every existing byte-exact test that asserted a
   bare `b=1` buffer is updated to list-of-1 here. **Blast radius (grep-verified): 13
   bare-tile-buffer assertions across 4 files** — `test/codegen/test_body.py`,
   `test/codegen/test_render.py`, `test/codegen/test_compact.py`,
   `test/transforms/test_ladder_compare.py`. `test_split`/`test_reorder`/`test_fuse` do
   not assert rendered decls, so they are unaffected. Update all 13 to the list-of-1 form.
3. **`BufferLayout` transform** (analyze/apply/legality). Gate: `test_buffer_layout.py`
   incl. standalone k5→k6.
4. **List-form compaction composability** — `BufferLayout → CodeMotion`. Fix
   `compact_shapes` if it mis-shrinks. Gate: the compaction unit test.
5. **Driven k0→k26 ladder** in `kernel_transforms.py` + wire `assert_matches_render` per
   rung against the step-0 reference. Gate: 26 rungs byte-exact + CPU-sim clean + HW-MFU
   matches each manual rung within noise, on gym-1.

Step 0 is the example rewrite (this turn); steps 1–4 are pure `nkigym`/test work; step 5
is the driven-ladder rewrite. A diverging rung in step 5 is a finding (transform gap or
manual-kernel normalization), triaged then.

## Out of scope (deferred, own plans)

- **k26→k27 RFactor + Gap 2** — RFactor ordering (manual LAST vs shipped EARLY) is an
  unconfirmed separate unknown needing its own gym-1 probe. k0→k26 avoids it entirely.
- **`versions` × `list_len` composition** — rejected today; a future lever.
- **Per-buffer `P ≤ 128` (vs the shipped hardcoded 128)** — a pre-existing concern in
  `physical_shape`/render/compact/interval, none of which `BufferLayout` touches (it
  factorizes the orthogonal tile axis). The manual ladder is all-P=128, so it does not
  arise here; generalizing `PARTITION_DIM` is its own plan. See "Partition axis" above.

## Relation to prior design

Supersedes §4(a) + §5 step 4 of
`2026-06-25-dependency-region-cover-codemotion-bufferlayout-design.md` (the BufferLayout
transform) and folds in that doc's §4(b) list-form compaction as the composability task.
The region-cover dependency + CodeMotion unification from that doc already shipped via
span-promotion (`fc3043b`).
