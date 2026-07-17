# `BufferCompaction` transform — make the silent place+compact+rebase loud

**Date:** 2026-07-14
**Status:** design (awaiting review)
**Goal:** lift the anonymous `place_buffers` + `compact_shapes` tail (and the
render-time `rebased_region`) into a first-class, per-buffer **`BufferCompaction`**
transform, materialized in the IR. Decouple `CodeMotion` and `RFactor` so their
`apply` does the structural edit only. Verification target: the driven ladder
(`examples/kernel_transforms.py`) reproduces `manual_transforms.py` **k0…k32**
rung-for-rung, byte-exact + CPU-sim clean on gym-1.

**Out of scope (deferred):** k33 = RFactor's byte-exact reproduction. RFactor loses
its tail here (becomes structural-only), but its before/after templates and the
`list_len`-shrink compaction case k33 needs are a separate future change. k33 is the
*only* RFactor rung, so k0…k32 is the entire non-RFactor ladder.

## The silent side effect

Both `CodeMotion.apply` (`code_motion.py:274-275`) and `RFactor.apply`
(`rfactor.py:170-171`) end with:

```python
place_buffers(new_ir.tree)    # descend every buffer's decl to the LCA of its users
compact_shapes(new_ir.tree)   # shrink every buffer's logical shape (+ list_len clamp)
new_ir.dependency = Dependency(new_ir.tree)
```

Nothing in the transform's name or option says "…and then I re-place and re-shape
every buffer in the tree." A third, hidden piece runs at **render** time:
`rebased_region` (`body.py:272`) subtracts a buffer's anchor loop vars from each
index so a compacted buffer is addressed within its single live instance.

Compaction is thus **three pieces split across two places**:

| piece | what | today |
|---|---|---|
| **place** | descend the decl to the LCA of its touchers | `place_buffers`, in the tail |
| **shape** | shrink logical shape to the access bbox | `compact_shapes`, in the tail |
| **rebase** | subtract anchor loops from each index | `rebased_region`, **inferred at render** |

The manual ladder already treats compaction as its own rung — each `# Code motion`
rung (k13/k18/k24/k30) is structural-only, and the next `# Buffer compaction` rung
(k14/k19/k25/k31) does place+shape+rebase. The transforms do not: they fuse the move
and the three compaction pieces into one `apply`, and the rebase is never in the IR.

## Design

**One principle: compaction is materialized in the IR, per buffer, atomically.** The
IR state (decl scope + logical shape + region frame) *is* the record of whether a
buffer is compacted — no render-time inference, no `compacted` flag.

### 1. `BufferCompaction(tensor)` — a per-buffer atomic move

Mirrors `BufferLayout`'s single-`tensor` surface (`buffer_layout.py`). `analyze`
offers one option per sbuf/psum buffer whose placed+compacted form differs from its
current tree form. `apply(ir, BufferCompactionOption(tensor))`:

1. **place** — descend *this buffer's* decl to the LCA block of its touchers (the
   per-buffer slice of `place_buffers`);
2. **compact** — shrink *this buffer's* logical shape to the bbox of its access
   regions over its LCA scope (the per-buffer slice of `_compact_one`);
3. **re-normalize** — recompute *this buffer's* access-region offsets from the
   extent-fit rule (§2a): now that the buffer's logical extent has shrunk, the outer
   loops beyond its capacity are dropped, so the offset falls into the single live
   instance automatically. No separate "rebase/subtract-anchors" step — the local
   frame is what the extent-aware offset computes;
4. rebuild `Dependency`.

There is **no `rebase_regions_of`** — the local frame is a consequence of the
extent-fit offset in `normalize_block`, not a materialized subtraction bolted on
afterward. Loud on bad input (unknown tensor, `shared_hbm`, already-compacted → no-op
is a legality error, matching `BufferLayout`).

**No `list_len` shrink for k0…k32.** Every compaction rung keeps its tile count
(sbuf_prod 1→1, psum_prod 16→16, sbuf_rhs 1→1, sbuf_lhs_T 1→1 — verified). The
`_clamp_list_len_to_tiles` case only fires at k33 (RFactor psum 16→1), which is out
of scope. `BufferCompaction` retains the clamp helper (harmless idempotent no-op
when the tile count is unchanged) but the k0…k32 ladder never exercises a shrink.

### 2. Render emits regions verbatim, and decl position follows the owning block

Two coupled render-path changes make the structural-only vs compacted split
byte-exact. **Both are no-ops for every kernel produced today** (where the compaction
tail always ran), so they only bite on the new structural-only states.

**(a) Region frame — extent-fit offset in `normalize_block`, emitted verbatim.**
Delete the `rebased_region` wrap at `body.py:272`; the renderer emits each region
**as stored on the tree**. The offset stored on the tree is the *correct
scope-relative* offset, computed once by `normalize_block._recompute_region` (which
every offset-touching transform — Split/Fuse/Reorder/CodeMotion — already calls).

The old `_recompute_region` rebuilt each offset as the **global** tile-space affine
`Σ_j loop_j · Π(inner extents)` over **every** loop the block binds on that dim —
i.e. it always addressed the full-extent buffer at root, ignoring the buffer's shape.
That is the bug behind k24: after `psum` is compacted at k19, the k24 rhs-load sink
re-runs `normalize_block` on the matmul's fork block, and the global recompute
re-derives `i_d2_0*512` on the compacted `psum`'s free axis (while the sibling memset
stays `0:512`) → mixed frame → wrong render.

Replace it with an **extent-fit** offset. On each axis, capacity = `extent // width`
tiles (for the partition axis, `extent // 128`). Keep the **innermost** loops on that
dim whose cumulative trip product ≤ capacity; **drop the outer loops beyond capacity**
— those select which instance is live, not a position within it. The kept loops build
the same `Σ loop · Π(inner)` affine.

- **Full-extent buffer** (canonical, structural-only, un-compacted): span == extent →
  capacity covers every loop → nothing dropped → **byte-identical to today's
  recompute**. A structural-only store-sink renders `sbuf_prod[:, i_d1_0,
  i_d2_0*512:+512]` into `(128,16,2048)` = **hand k13**. Split/Fuse/Reorder on full
  buffers are unchanged.
- **Compacted buffer**: capacity shrank with the shape → the outer instance-selecting
  loop is dropped → `sbuf_prod[:, i_d1_0, 0:512]` into `(128,16,512)` = **hand k14**,
  and the compacted `psum` stays `0:512` through the k24 sink because *every*
  `normalize` now computes the same local frame — no separate rebase to clobber.

The rule reads only shape + loop trips, never offsets, so it is non-circular and
idempotent.

**(b) Decl position — region-aware hoist above carried loops.**
`_alloc_emit_anchors` (`body.py:50-79`) originally computed decl *scope* as
`_lca_nodes(tree, leaves)` — the LCA of the buffer's touching leaves. That alone
cannot distinguish hand k13 from hand k14: in BOTH, `sbuf_prod`'s two touchers
(drain + store) sit under `i_d2_0`, so the touchers' LCA is `i_d2_0` in both — yet
k13 (full-shape) declares the buffer at **top level** while k14 (compacted) declares
it **inside `i_d2_0`**. (An early attempt used the buffer's `alloc_buffers` owning
block instead; `place_buffers` walks the LCA up to the nearest BlockNode, which is a
root child for both states, so that put BOTH at top and broke k14 — gym-1-verified.)

The discriminator is whether the buffer's region still **references** the enclosing
loop var — which the extent-fit offset (§2a) already decides. A full-shape `sbuf_prod`
(k13) writes `[:, i_d1_0, i_d2_0*512:+512]` — its offset carries `i_d2_0` (capacity
covers the loop), so it is live across the whole `i_d2_0` loop and its decl must
**hoist above** `i_d2_0` (to top). A compacted `sbuf_prod` (k14) has offset
`[:, i_d1_0, 0:512]` (extent-fit dropped `i_d2_0`), so it is re-created each iteration
and its decl sits **inside** `i_d2_0`, at the touchers' LCA.

Rule: **scope = LCA of touchers, then hoisted above every enclosing loop whose
`loop_var` appears in any of the buffer's region offsets.** No-op for canonical/base
(offsets there never carry an enclosing loop var beyond capacity → stay at LCA / root),
unifies k13 (carries `i_d2_0` → hoist to top) and k14 (extent-fit dropped `i_d2_0` →
decl at `i_d2_0`), and needs neither `_owning_block` nor a bare `_lca_nodes`. It reads
the same offsets the extent-fit rule produces, so decl position and offset frame are
consistent by construction — one mechanism, not two.

`rebased_region` is deleted from the render path (its job is subsumed by the
extent-fit offset). `compact_shapes` (whole-tree) is retired in favor of the
per-buffer path; its unit-tested internals (`_compact_one`, `_anchor_loop_vars`,
`_axis_span`, `_offsets_consistently`) are reused by `BufferCompaction`.

### 3. `CodeMotion` and `RFactor` become structural-only

Drop `place_buffers` + `compact_shapes` from both `apply` tails; each does the
structural edit (`_move` / `_emit_rmw`) + `Dependency` rebuild only.

- **CodeMotion**: a sink leaves the moved block's buffers at their prior scope,
  shape, and global-frame index → renders the hand `# Code motion` rungs exactly.
  The following explicit `BufferCompaction` rung compacts.
- **RFactor**: also loses the tail (per "remove the tail from RFactor too"). Its
  byte-exact reproduction (k33) is deferred — the emission authors gadgets whose
  offsets are set up for a *subsequent* compaction, and the k33 psum needs the
  `list_len` 16→1 shrink. Reproducing k33 is left to the RFactor-template redesign;
  this spec only requires that RFactor's `apply` no longer runs the tail, so the
  render-path change (deleting `rebased_region`) does not depend on it.

### 4. Dependency frame — the one reversed invariant

`compact.py:12-13` documents a deliberate choice: tree regions stayed **global-frame**
so `Dependency` reasoned in one consistent coordinate frame. Under extent-fit, a
compacted buffer's tree offset is **scope-relative** (the instance-selecting loops are
dropped), so `Dependency` (rebuilt at the end of `apply`) sees the compacted buffer's
true physical reuse across those loops — a genuine WAR/WAW carry the global frame hid.

Assessment: this is *more* physically faithful (it matches the emitted kernel) and
*safe for k0…k32* — every rung after a compaction is a `BufferLayout` field-set or a
`Split`/`Reorder`/`CodeMotion` on a *different* buffer, none of which relies on the
compacted buffer's pre-compaction global frame to license a move. The gym-1 CPU-sim
+ byte-exact gate is the proof, not this argument. Flagged because it reverses a
documented decision. Note this frame change now applies to **every** transform's
offset (extent-fit lives in `normalize_block`), not just `BufferCompaction`'s output —
but it only diverges from the old global offset for a buffer whose extent is smaller
than its full loop span, i.e. a buffer that has been through `BufferCompaction`. Full
buffers are byte-identical, so Split/Fuse/Reorder dependency graphs are unchanged.

### 5. Driven ladder → manual k0…k32

Rewrite `examples/kernel_transforms.py::_build_ladder` to the restructured manual
numbering, ONE transform per rung, byte-exact to `manual_transforms.kernel_i`:

| rungs | transform |
|---|---|
| k1–k6 | Reorder ×2, Split ×2, Reorder ×2 → matmul nest `N>ko>Mo>Mi>ki` |
| k7 | BufferLayout psum_prod → list-16 |
| k8–k9 | drain tensor_copy: Split d2, Reorder |
| **k10** | **CodeMotion** (drain sink) — structural-only |
| k11–k12 | store: Split d2, Reorder |
| **k13** | **CodeMotion** (store sink) — structural-only |
| **k14** | **BufferCompaction** sbuf_prod → `(128,16,512)`, scope `i_d2_0` |
| k15 | BufferLayout sbuf_prod → list-16 |
| k16–k17 | psum memset: Split d2, Reorder |
| **k18** | **CodeMotion** (psum-memset sink) — structural-only |
| **k19** | **BufferCompaction** psum_prod → `(128,1,512)`, scope `i_d2_0` |
| k20–k23 | rhs load: Split d0, Split d2, Reorder ×2 |
| **k24** | **CodeMotion** (rhs-load sink) — structural-only |
| **k25** | **BufferCompaction** sbuf_rhs → `(128,8,512)`, scope `i_d0_0` |
| k26 | BufferLayout sbuf_rhs → list-8 |
| k27–k29 | lhs_T load: Split d1, Split d0, Reorder |
| **k30** | **CodeMotion** (lhs_T-load sink) — structural-only |
| **k31** | **BufferCompaction** sbuf_lhs_T → `(128,8,512)`, scope `i_d1_0` |
| k32 | BufferLayout sbuf_lhs_T → list-8 |

The byte-exact gate (`assert_matches_hand(render(ir), manual_transforms.kernel_i)`)
and the CPU-sim + HW-profile loop are unchanged; only `_build_ladder` and its rung
count (32 steps) change. The RFactor step is dropped (k33 deferred).

Note the five CodeMotion rungs (k10/k13/k18/k24/k30) that were previously
compaction-fused now render their **structural-only** intermediate, matching the
manual ladder's separate `# Code motion` rungs. k10 is a structural no-op that still
renders identically to hand k10 (its buffer does not move scope) — the drain
tensor_copy sinks under `i_d2_0`, and its buffers (`psum_prod` read, `sbuf_prod`
write) are already scoped such that no shape change results, so structural-only ==
compacted here.

### 6. Tests

- **New** `test/transforms/test_buffer_compaction.py`: `analyze` enumerates the
  compactable buffers; `apply` on each of the four ladder buffers yields the k14 /
  k19 / k25 / k31 shape + local-frame regions; idempotence; loud rejects (unknown
  tensor, shared_hbm, no-op). Fold `test/codegen/test_compact.py`'s surviving cases
  (canonical no-op, per-leaf extents, `_emit_alloc` follows shape) in — PRESERVE
  every assertion (test-slimming = dedup, not drop).
- **`test/transforms/test_code_motion.py`**: assert `apply` now returns the
  structural-only form (buffer at prior scope/shape/global index). Where a case
  needs the compacted output, chain `BufferCompaction().apply(...)` and keep the
  existing compacted assertion.
- **`test/transforms/test_rfactor.py`**: RFactor is now structural-only. The
  compacted fixtures (`kernel_rfactor_ko.py`) are out of this spec's scope (k33);
  mark the k33 byte-exact case pending with a note pointing at the RFactor-template
  redesign — do NOT fake-green it. The non-compacting structural assertions (role
  flip, gadget placement, `Dependency`) stay.
- **`test/transforms/test_render_equivalence.py`, `test/ir/test_buffer_placement.py`,
  `test/transforms/test_buffer_layout.py`**: re-run; adjust only if they assumed the
  render-time rebase. `canonical_build`'s initial `place_buffers` is untouched.

## Scope

| # | Change | Files |
|---|--------|-------|
| 1 | **Extent-fit offset** in `_recompute_region` (drop the global-span recompute; keep innermost loops that fit `extent // width`) — the shared fix all offset-touching transforms inherit | `nkigym/src/nkigym/transforms/_normalize.py` |
| 2 | `BufferCompaction` transform (per-buffer place + compact + re-normalize); **no `rebase_regions_of`** | `nkigym/src/nkigym/transforms/buffer_compaction.py` (new), `nkigym/src/nkigym/codegen/compact.py` (per-buffer place+compact helper; retire whole-tree `compact_shapes` + delete `rebase_regions_of`) |
| 3 | Export `BufferCompaction` / `BufferCompactionOption` | `nkigym/src/nkigym/transforms/__init__.py` |
| 4 | Delete `rebased_region` call; region-aware decl hoist (`_hoisted_scope`); render regions verbatim | `nkigym/src/nkigym/codegen/body.py` |
| 5 | CodeMotion `apply` → structural-only (drop the tail) | `nkigym/src/nkigym/transforms/code_motion.py` |
| 6 | RFactor `apply` → drop the tail (k33 repro deferred) | `nkigym/src/nkigym/transforms/rfactor.py` |
| 7 | `_build_ladder` → manual k0…k32 with explicit BufferCompaction rungs | `examples/kernel_transforms.py` |
| 8 | Tests (new + adapted): unit-test the extent-fit offset directly | `test/transforms/`, `test/codegen/`, `test/ir/` |

## Invariants preserved

- Single return per function; loud failures only (no silent no-op-on-bad-input; a
  no-op compaction is a legality error).
- Transform legality never gates resource capacity (full-extent intermediate rungs
  remain valid outputs; HW profiling prunes them).
- Byte-exact gate = `render(apply(...)) == hand kernel`, AST-canonical; the hand
  ladder is the ground truth, authored by hand, never a captured render.
- `analyze`/`apply` signatures follow the `Transform` base; `BufferCompactionOption`
  is a frozen dataclass mirroring `BufferLayoutOption`.

## Verification plan (gym-1, in order)

1. Unit: `test_buffer_compaction.py` + adapted `test_code_motion.py` green.
2. No regression: full `test/transforms/` + `test/codegen/` + `test/ir/` suite green
   (RFactor k33 case marked pending, not failing).
3. Driven ladder: `python examples/kernel_transforms.py --cache …` on gym-1 — every
   rung k0…k32 byte-exact to `manual_transforms.kernel_i` AND CPU-sim clean; HW
   profile reported (full-extent intermediates may fail BIR-verify — non-fatal).
