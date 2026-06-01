# Unified Split + Loop-Var Normalization + No Trip-1 Loops

## Status (2026-06-01)

NOT STARTED. Supersedes the partial tensorize fix
(`2026-06-01-tensorize-split-fuse-fix.md`, shipped `afa84fa`→`b000a7e`):
that fixed the *region arithmetic* but left two structural defects that
strict byte-exact verification against `kernel_transforms.py` exposed —
double-suffix loop names (`i_d1_0_0`) and a leftover trip-1 wrapper. This
spec reshapes the iteration model so the transforms produce IR
**byte-identical** to the hand-written ladder.

Supersedes the learning *"Canonical KEEPS trip-1 ForNodes"* — the new
rule is **no trip-1 ForNodes anywhere**.

## The model

A dim's iteration is a factorization `(trip₀, …, trip_{k-1},
tensorize_size)` whose product is the dim extent. In the `KernelTree`:

- **k ForNodes** — one per trip, in nest order, parent→child. A ForNode
  exists ONLY when its trip > 1 (trip-1 factors contribute no loop).
- **tensorize_size** — the innermost factor, stored as the access
  `BufferRegion` width on the ISA leaf.
- **loop_var names** — dense, position-in-dim: `i_d{dim}_{N}`, where N is
  the loop's ordinal among that dim's ForNodes, outer→inner. Never the
  nested `i_d{dim}_0_0` form.
- **iter_values** — the block binds each iter_var to the affine over its
  dim's loop vars: a dim with loops `i_d1_0 (t0)`, `i_d1_1 (t1)` and tile
  `w` binds element-offset `(i_d1_0*t1 + i_d1_1) * w`. The renderer reads
  these for the region `lo`.

### Worked examples (from the user)

```
(16, 8, 128)  --split trip i_d0_1 (8)->(2,4)-->  (16, 2, 4, 128)
   i_d0_0(16) i_d0_1(8) tile128             i_d0_0(16) i_d0_1(2) i_d0_2(4) tile128

(16, 8, 128)  --split trip i_d0_0 (16)->(4,4)--> (4, 4, 8, 128)
                                            i_d0_0(4) i_d0_1(4) i_d0_2(8) tile128
                                              ^ inserted    ^ old 8-loop renumbered 1->2

(16, 1024)    --split tensorize 1024->(16,64)--> (16, 16, 64)
   i_d0_0(16) tile1024                       i_d0_0(16) i_d0_1(16) tile64

(2048,)       --split tensorize 2048->(16,128)--> (16, 128)   [= k0->k1 load d1]
   no loop, tile2048                         i_d0_0(16) tile128, index i_d0_0*128
```

The last is k0→k1: canonical load d1 is trip-1 → **no ForNode**, just
tensorize 2048. Split → one trip-16 loop, tile 128 → **exactly k1**. The
double-suffix and trip-1-wrapper both vanish because the IR never stores
them.

## Decisions (settled in brainstorming)

1. **One unified Split.** No more "outer-trip vs tensorize" flavors:
   `SplitOption` names a factor (a trip ForNode, or the tensorize_size)
   and replaces it with sub-factors. The current `target_axis=None` vs
   set distinction is removed.
2. **Naming model (B): stored-but-normalized.** `loop_var` stays a
   `ForNode` field (so `dependency`/`reorder`/`_code_motion` keep keying
   off it), but every transform re-normalizes its block's loop vars to
   the dense `i_d{dim}_{N}` scheme. Not a render-time derivation.
3. **No trip-1 ForNodes anywhere** — including `canonical_build`. A
   trip-1 axis is pure tensorize_size with no loop.
4. **Byte-exact verification.** The gate is `render(apply(before)) ==
   <hand kernel> source` (modulo comments/formatting), for transitions
   whose before-state is buildable today (k0→k1). The 4 tensorize splits
   whose before-states need compute_at (k2/k4/k10/k12) are deferred and
   stated as such — NOT proxied.

## Architecture

### `normalize_block(tree, block_nid)` — the new shared pass

Owns both invariants (no trip-1, dense names). After any Split/Fuse
mutates a block's ForNode chain, the transform calls it:

1. **Drop trip-1 ForNodes** in the block's subtree: for each ForNode with
   `extent == 1`, re-link its children to its parent (preserving sibling
   order via `_replace_in_parent_children`), remove the node.
2. **Renumber dense per dim**: walk the block's ForNode chain outer→inner;
   for each dim (identified via `block.axis_map` / the iter_var the loop
   binds), assign `i_d{dim}_{N}` with N = running count for that dim.
3. **Rewrite iter_values + region lo's**: rebuild the block's
   `iter_values` as the affine over the renamed loops (`Σ loopᵢ ·
   stride_below_i`), and substitute the new names into every descendant
   ISA leaf's `operand_bindings` and the block's reads/writes.

This single function replaces the ad-hoc `f"{loop_var}_{i}"` naming in
Split, the `_stem`/`_fused` logic in Fuse, and absorbs what
`_tile_region.py` did (the element-space arithmetic moves here as the
stride computation). One place enforces correctness.

### Unified `Split`

The option shape is unchanged from today — `SplitOption(target_nid,
factors, target_axis)` — but the *mechanics* converge so there is no
behavioral fork:

- `target_nid` is a **ForNode** (split a trip): replace it with
  `len(factors)` ForNodes (extents = `factors`); the tile is unchanged.
- `target_nid` is an **ISA leaf** + `target_axis` = the dim whose
  **tensorize_size** to split: the leading `factors[:-1]` become new
  ForNodes (trips), `factors[-1]` becomes the new tile width.

Both paths then call `normalize_block`, which is where they converge:
drop any resulting trip-1 ForNodes, assign dense `i_d{dim}_{N}` names,
and rewrite iter_values + region lo's. The current "outer-trip vs
tensorize" *behavioral* distinction (different naming, different region
math) disappears — only the "which factor am I splitting" input differs,
and `normalize_block` handles both identically downstream.

### Unified `Fuse` (the exact inverse — same treatment)

Fuse is Split's inverse and has the SAME two defects, so it gets the same
fix in the same change:

- **Merge two trips** (inverse of trip-split): `i_d0_1(2), i_d0_2(4)` →
  `i_d0_1(8)`. Today `_do_outer_trip` names the result `i_d{dim}_fused`
  (via `_fused_loop_var = _stem + "_fused"`) — the Fuse analogue of the
  `i_d1_0_0` naming wart. After the fix, `normalize_block` renames it
  dense (`i_d0_1`).
- **Absorb a trip into the tile** (inverse of tensorize-split):
  `i_d1_0(16) + tile128` → `tile2048`. The absorbed trip becomes part of
  the tile, leaving **trip-1 → the loop is dropped entirely** by
  `normalize_block`. This is the k1→k0 reverse (a load's single d1 loop
  collapses back to a full-width loopless access). The region-widen math
  for this path was fixed this session (`widen_region_axis`) and moves
  into `normalize_block`.

Both Fuse paths call `normalize_block` exactly like Split. The ad-hoc
`_same_loop_axis` / `_stem` / `_fused_loop_var` naming helpers are
deleted — `normalize_block` is the sole namer. `_same_loop_axis` (used by
`Fuse.analyze` to find fuseable adjacent same-dim ForNodes) is replaced by
"adjacent ForNodes binding the same concrete dim" via the block's
`axis_map`, not stem-string comparison.

**Round-trip is the key cross-check:** Split a factor then Fuse it back
must return byte-identical IR (the dense names make this exact, where
`_fused`/`_0_0` previously diverged). Tested both directions:
`Split(2048→16,128)` then `Fuse` → `(2048,)`; `Split(trip 8→2,4)` then
`Fuse` → `(8,)`.

### `canonical_build` — stop emitting trip-1 loops

`canonical_build.py:90-92` computes `trip = extent // tile` and
unconditionally adds a ForNode. Change: **add the ForNode only when
`trip > 1`**. A trip-1 axis still gets its iter_var + the tile-width
region; it just has no loop. The renderer already fires the op at "loop
depth" — with one axis loopless, the op sits one level shallower, which
is correct (matches hand-k0, now trip-1-free).

### Renderer — unchanged

`body.py` walks ForNodes and fires the op at leaf depth; with no trip-1
nodes, it naturally emits fewer loops. `iter_values`/region `lo` (already
normalized) drive the slice. No renderer change expected — verify.

## Verification

The committed `kernel_transforms.py` is now trip-1-free and
self-consistent (15/15 CPU-sim pass). Tests assert **byte-exact** rendered
output:

- **Canonical k0**: `render(build_canonical_ir())` matches hand-k0's op
  structure (no trip-1 loops; loads/memset/copy/store have only their
  real loops). Update the ~9 test files asserting the old trip-1
  structure (`test_render`, `test_body`, `test_node_labels`,
  `test_dependency`, `test_code_motion`, `test_compact`, `test_split`,
  `test_ir_extensions`, `test_tile_region`).
- **k0→k1**: build canonical, `Split` the load's d1 tensorize 2048→(16,128),
  assert rendered load block == hand-k1's load block (`for i_d1_0 in
  range(16): dma_copy(... i_d1_0*128 : +128)`) — byte-exact, no
  `i_d1_0_0`, no trip-1 wrapper.
- **Naming unit**: split a trip-16 loop →(2,8); assert the two loops are
  `i_d1_0`, `i_d1_1` (dense), and a 3-factor split gives `i_d1_0,
  i_d1_1, i_d1_2`. Insertion+renumber case (split outer, old inner
  renumbers) per the worked examples.
- **Fuse merge-trips naming**: fuse `i_d0_1(2), i_d0_2(4)` → assert the
  result is `i_d0_1(8)` (dense), NOT `i_d0_1_fused`.
- **Fuse absorb-into-tile → trip-1 drop**: fuse a load's single d1 loop
  `i_d1_0(16) + tile128` → assert tile becomes 2048 and the ForNode is
  GONE (trip-1 dropped) — the k1→k0 reverse; render byte-matches the
  trip-1-free k0 load.
- **Split↔Fuse round-trip**: `Split(2048→16,128)` then `Fuse` == original
  `(2048,)`; `Split(trip 8→2,4)` then `Fuse` == `(8,)`. Byte-exact render
  both ends (the dense naming makes this exact where `_fused`/`_0_0`
  previously diverged).
- **Deferred**: k2→k3, k4→k5, k10→k11, k12→k13 need compute_at to build
  their before-states; explicitly listed as unverifiable-until-compute_at.
  No proxy.

## Files

```
nkigym/src/nkigym/transforms/
├── _normalize.py     # NEW — normalize_block(tree, block_nid): drop trip-1, dense rename, rewrite iter_values + regions
├── split.py          # EDIT — unified mechanics; both paths -> ForNodes + tile + normalize_block; drop f"{lv}_{i}" naming
├── fuse.py           # EDIT — both paths -> normalize_block; delete _same_loop_axis/_stem/_fused_loop_var/_widen_region_axis (analyze finds adjacent same-dim ForNodes via axis_map)
└── _tile_region.py   # ABSORB/REMOVE — element-space stride math moves into _normalize

nkigym/src/nkigym/ir/canonical_build.py  # EDIT — emit ForNode only when trip > 1

test/...              # EDIT — byte-exact render assertions; update trip-1-structure expectations (~9 files)
```

## Implementation order

1. **canonical_build drops trip-1** + update canonical-structure tests →
   render matches hand-k0. Gate: full suite green on the new canonical.
2. **`normalize_block`** + unit tests (drop-trip-1, dense rename,
   iter_value/region rewrite) in isolation.
3. **Split** routes through `normalize_block`; byte-exact k0→k1 test;
   naming + insertion-renumber tests. Remove `_tile_region` once absorbed.
4. **Fuse** routes through `normalize_block`; Split↔Fuse round-trip.
5. Full suite byte-exact green; re-confirm `kernel_transforms.py` ladder
   semantics unchanged.

## Out of scope

- compute_at / the 4 tensorize transitions needing it (deferred, stated).
- Reorder mechanics (verified correct; but it must keep names valid —
  after a reorder, `normalize_block` may renumber if two same-dim loops
  swap; confirm reorder calls it or is naming-stable).
- Multi-output-axis ops beyond the matmul fixture (handled generically by
  axis_map, tested on matmul).

## Risk

This changes canonical IR shape (trip-1 loops gone) — broad test churn
(~9 files). Mitigated by: the change is mechanical (fewer loops), the hand
ladder is now the byte oracle, and each step gates on the full suite.
`_tile_region.py` (shipped this session) is absorbed, not wasted — its
element-space stride math becomes `normalize_block`'s rewrite core.
