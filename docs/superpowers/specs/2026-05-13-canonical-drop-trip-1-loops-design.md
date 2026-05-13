# Canonical: Drop Trip-1 Outer Loops for Unbounded-MAX Axes

## Problem

Canonical build currently emits two nested `ForNode`s per op axis:
- Outer trip loop with extent `axis_extent / MAX_TILE_SIZE`.
- Inner tile loop with extent `MAX_TILE_SIZE` (elided by renderer).

For unbounded axes (`MAX_TILE_SIZE = None`), `outer_trip = 1` and `inner_extent = axis_extent`. The outer is a no-op wrapper that adds nothing beyond noise:

1. **Atom traces become longer than necessary.** To reach the structure of `kernel_1` in `kernel_transforms.py` (lhs_T load's M axis tiled to 128), the trace is canonical `(1, 2048) → Fuse → (2048) → Split(factor=128) → (16, 128)`. The first state is an artifact of how canonical emits the axis, not a meaningful scheduling decision.

2. **Trip-1 loops aren't splittable.** `Split.is_legal` requires `1 < factor < extent`. A trip-1 outer can't be directly split — Fuse-then-Split is the only path, which exercises more atom composition than the user's intent (just tile this axis).

3. **The hand-written `kernel_0` doesn't have trip-1 outers wrapping the visible slice width.** Canonical's output matches byte-for-byte today only because the renderer elides the inner tile loop, leaving the trip-1 outer as a cosmetic `for ... in range(1):`. This is surface-level parity achieved through two layers of compensating nonsense.

## Target Model

Each op axis emits **at most one `ForNode`** at canonical-build time. The number of loops depends on the op's declared bound for that axis:

- **Bounded axis** (`MAX_TILE_SIZE[axis] = N`): emit one `ForNode` with `trip = axis_extent / N`. The ISA call's slice width along this axis is `N` (the tile), encoded directly in `BufferAccess.iter_var_coeffs` + `AccessRange.extent = N`.
- **Unbounded axis** (`MAX_TILE_SIZE[axis] = None` or absent): emit **no `ForNode`** for this axis. The ISA call's slice width is the full axis extent. `BufferAccess.iter_var_coeffs` has no entry for this axis; `AccessRange.extent = axis_extent`.

**Consequences:**

- For a bounded axis, canonical is already the "tiled" state (one loop, trip = # tiles). Further tiling is one Split call.
- For an unbounded axis, canonical has no loop. The natural way to bring an axis under a loop is `ComputeAt` on a consumer's loop of the same axis — the producer follows the consumer's scheduling, matching hand-written idioms.

**Example (lhs_T load, matmul M axis at extent 2048):**

Before (today):
```
for i_d1_outer in range(1):       # trip-1 artifact
  for i_d1_inner in range(2048):  # elided by renderer, encoded as slice width
    nisa.dma_copy(..., slice[0:2048])
```

After:
```
nisa.dma_copy(..., slice[0:2048])  # no d1 loop at all
```

Matmul's M axis (bounded, MAX=128) before:
```
for i_d1_outer in range(16):       # trip = 2048/128 = 16
  for i_d1_inner in range(128):    # elided
    nisa.nc_matmul(..., slice[i_d1_outer*128 : i_d1_outer*128 + 128])
```

After:
```
for i_d1_outer in range(16):       # unchanged — bounded case already only emits one visible loop
  nisa.nc_matmul(..., slice[i_d1_outer*128 : i_d1_outer*128 + 128])
```

The change is invisible for bounded axes (renderer already elided the inner). For unbounded axes, the trip-1 outer disappears from emitted source and from the IR.

## IR Changes

**`SBlock.iter_vars`:** a block has iter-vars only for axes with one or more enclosing `ForNode`s. An unbounded axis with no loop contributes no iter-var to the block's `iter_vars` list. Today each block has 2 iter-vars per axis (outer + inner); after the refactor, blocks have 1 iter-var per bounded axis and 0 iter-vars per unbounded axis.

**`BufferAccess.iter_var_coeffs`:** entries for unbounded axes disappear. `AccessRange.extent` for an unbounded axis equals the full axis extent (unchanged in meaning; just no longer decomposed into outer-trip × inner-tile).

**`AccessRange.extent`:** for a bounded axis, equals `MAX_TILE_SIZE[axis]` (the tile). For an unbounded axis, equals `axis_extent` (the full range). Semantically this is `axis_extent / product(ancestor_trips)` where the product is 1 for unbounded axes and `axis_extent / MAX_TILE_SIZE` (= trip count) for bounded, giving `MAX_TILE_SIZE` back — consistent with today.

## Renderer Changes

The renderer's "innermost tile loop elision" logic disappears. Every `ForNode` now emits a `for` header unconditionally — there are no elided loops left to skip.

`_innermost_tile_iter_var_ids` and `EmitCtx.innermost_tile_ids` become dead code. Remove them.

## Canonical Builder Changes

`_make_sblock` in `canonical.py`:
- For each axis in `op.touched_axes`:
  - If `op.axis_tile[axis_id] < axis_extent` (bounded): allocate one iter-var with `extent = axis_extent / tile`. Build one ForNode above the SBlock for this axis.
  - Else (`op.axis_tile[axis_id] == axis_extent`, unbounded): allocate zero iter-vars. No ForNode.

BufferAccess coefficients:
- For bounded axes: one iter-var per axis, coefficient = `tile`. AccessRange.extent = `tile`.
- For unbounded axes: no iter-var entry. AccessRange.extent = full axis extent.

## Atom Changes

**Split, Fuse, Reorder, ComputeAt, ReverseComputeAt:** all operate on live `ForNode`s. Semantics unchanged. Legality checks tighten naturally because illegal operations (e.g., Split on a non-existent loop) can't be constructed.

**No new atom introduced.** The pattern "tile an unbounded axis" is expressed as `ComputeAt(producer_block, consumer_loop_on_that_axis)`, which already exists. The producer's enclosing loop chain inherits the consumer's loop for that axis, bringing the producer under a loop without allocating one directly.

## Pre-existing Atom Operations

- `ComputeAt(lhs_T_load, matmul_d1_outer)` — moves load inside matmul's d1 loop (trip 16). Load's d1 axis access pattern gains the d1 iter-var automatically because it sits below matmul's d1 outer now. This is what `kernel_1` represents structurally.

- `Split(loop, factor)` on a bounded-axis loop — same as today, since bounded axes already emit one loop.

## Test Migration

**`test/tune/test_axis_identity.py`:** today's tests find a `(outer=trip-1, inner=tile)` d1 pair on the lhs_T load (NKILoad d1 is unbounded). After the change that pair doesn't exist. Same-axis Fuse tests need a fixture that constructs the `(outer, inner)` state explicitly via an initial `Split` on a bounded axis. Example: matmul M canonical trip=16 → `Split(factor=4)` → `(4, 4)` → `Fuse` → `(16)` → `Split(factor=8)` → `(2, 8)`. This exercises the same atom interaction without relying on canonical pathology.

**`test/tune/test_fuse.py`:** same considerations. Most existing Fuse tests use canonical + immediate fuse on unbounded-axis outer; they'd have nothing to fuse post-change. Update fixtures to introduce the fusable pair via a preceding Split.

**`test/tune/test_split.py`:** many tests split canonical outer loops. For bounded axes, no change. For unbounded axes, the loop no longer exists — those tests either switch to bounded-axis fixtures or get deleted.

**`test/codegen/test_canonical.py`:** tests that assert "2 iter-vars per axis" must update to "1 iter-var per bounded axis, 0 per unbounded axis." Regression test `test_canonical_emits_outer_and_inner_tile_loops` is about to be false — remove or invert.

**`test/codegen/test_render_tile_elision.py`:** the renderer no longer elides anything. Remove this file entirely.

**`test/tune/test_fuse_eager_rewrite.py`:** `test_fuse_then_split_renders_and_cpu_sims` depends on finding the canonical d1 pair on lhs_T load. Update to the new fixture pattern (A above).

**`test/ops/test_tile_bounds.py`:** unaffected — tests class attrs, not canonical behavior.

## Out of Scope

- Growing a loop on a no-loop axis via a dedicated atom. `ComputeAt` covers the real use cases; a `Tile(axis, factor)` atom is deferred unless a workload emerges that needs it.
- Non-tiled execution of bounded axes. Today a bounded axis always emits a loop at canonical; if a workload needs to fire one giant ISA call instead of iterating, that's a separate design question.

## Success Criteria

- Canonical matmul module has **fewer** ForNodes than today (trip-1 outers gone for unbounded axes).
- `examples/matmul_lhsT_rhs.py` step 0 canonical renders with no trip-1 `for ... in range(1):` loops for unbounded axes (lhs_T load's M, rhs load's N, memset's N, tensor_copy's N, store's N).
- `kernel_transforms.py`'s kernel_0 is no longer byte-identical to the rendered canonical — it's cleaner (fewer loops). The structural equivalence to hand-written `kernel_0` holds after mechanical rewrite of `kernel_transforms.py` to drop the trip-1 wrappers in the reference.
- 15 kernels in `kernel_transforms.py` still CPU-sim pass (after the reference is updated to drop trip-1 outers).
- Full test suite passes (after test migration).
- `examples/matmul_lhsT_rhs.py` reaches at least one post-canonical step via a real atom (e.g., `ComputeAt(lhs_T_load, matmul_d1_outer)`), demonstrating the cleaner atom sequence.

## Migration Risk

- **`kernel_transforms.py` reference kernels** have trip-1 `for ... in range(1):` outers today. These must be rewritten to drop those outers (mechanical edit).
- **Existing atoms' legality checks** that look up loops by extent may need updating. Most should be robust because they check for specific extents (not "the trip-1 outer").
- **`SBlock.iter_vars` consumers** (atoms, renderer's `_innermost_tile_iter_var_ids`, etc.) need audits to handle variable iter-var counts per axis.
