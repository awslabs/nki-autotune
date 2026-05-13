# Canonical: Drop Trip-1 Outer for Unbounded-MAX Axes

## Problem

Canonical build emits two nested `ForNode`s per op axis:
- Outer trip loop with extent `axis_extent / MAX_TILE_SIZE`.
- Inner tile loop with extent `MAX_TILE_SIZE`.

For unbounded axes (`MAX_TILE_SIZE = None`), `outer.trip = 1` and `inner.extent = axis_extent`. The trip-1 outer is pure noise:

1. **Atom traces become longer than necessary.** To reach `kernel_1` (lhs_T load's M tiled at 128), canonical `(1, 2048) → Fuse → (2048) → Split(128) → (16, 128)`. The `(1, 2048)` state is canonical pathology, not a meaningful scheduling choice.
2. **Trip-1 loops aren't splittable** (`Split.is_legal` requires `1 < factor < extent`). Fuse-then-Split becomes the only path; composition bugs stack up.
3. **Emitted source has `for ... in range(1):` wrappers** around trip-1 outers for unbounded axes. Cosmetic noise in the rendered NKI kernel.

## Target Model

**Preserve the TVM-faithful "innermost loop = tile" invariant.** Every axis has at least one ForNode; the innermost-per-axis ForNode's extent is the ISA slice width (what the renderer bakes into the call as the tile).

- **Bounded axis** (`MAX_TILE_SIZE[axis] = N`, `N < axis_extent`): two ForNodes — outer with trip `axis_extent / N`, inner with trip `N`. **Unchanged from today.**
- **Unbounded axis** (`MAX_TILE_SIZE[axis] = None` or `MAX >= axis_extent`): **one ForNode** with trip `axis_extent`. Drop the trip-1 outer; keep the tile loop. The whole axis is one tile.

### Examples

Matmul M (MAX=128, extent=2048):

```
for outer in range(16) [outer trip]
  for inner in range(128) [tile — elided by renderer]
    nc_matmul(..., slice[outer*128 : outer*128 + 128])
```

Unchanged — bounded axis already emits (trip, tile).

lhs_T load F (MAX=None, extent=2048):

Before:
```
for outer in range(1) [trip-1 noise]
  for inner in range(2048) [tile — elided]
    dma_copy(..., slice[0:2048])
```

After:
```
for inner in range(2048) [tile — elided by renderer]
  dma_copy(..., slice[0:2048])
```

One ForNode. Rendered NKI source has NO `for` loop for that axis (tile loop is elided; no outer wrapper).

## IR Changes

**`KernelModule` structure:** unchanged.

**`SBlock.iter_vars`:** bounded axes contribute 2 iter-vars (outer + inner). Unbounded axes contribute **1** iter-var (the tile). Today both contribute 2. Consumers that walked "2 iter-vars per axis" must handle variable counts — but it's still always "at least 1 per axis".

**`BufferAccess.iter_var_coeffs`:**
- Bounded axis: two entries `{outer.var_id: tile, inner.var_id: 1}`. Extent = tile. Unchanged.
- Unbounded axis: one entry `{tile_iv.var_id: 1}`. Extent = full axis extent. Today has `{outer: extent, inner: 1}` → simplify to `{inner: 1}` (drop the trip-1 outer's entry whose coefficient * trip=1 contributes 0 anyway).

**`AccessRange.extent`:**
- Bounded axis: `tile = MAX_TILE_SIZE`. Unchanged.
- Unbounded axis: full axis extent. Unchanged in meaning.

## Canonical Builder Change

`_make_sblock` in `canonical.py`. Current structure:

```python
for axis_id in op.touched_axes:
    total = module.axes[axis_id].total_size
    tile = op.axis_tile[axis_id]
    outer_extent = total // tile
    inner_extent = tile
    v_outer = module.allocate_iter_var(axis_id, outer_extent, role)
    v_inner = module.allocate_iter_var(axis_id, inner_extent, role)
    # always emit both
```

New:

```python
for axis_id in op.touched_axes:
    total = module.axes[axis_id].total_size
    tile = op.axis_tile[axis_id]
    if tile < total:
        """Bounded axis: outer trip + inner tile (two ForNodes)."""
        v_outer = module.allocate_iter_var(axis_id, total // tile, role)
        v_inner = module.allocate_iter_var(axis_id, tile, role)
        iter_vars.append(v_outer)
        iter_vars.append(v_inner)
    else:
        """Unbounded axis (tile == total): one ForNode for the whole axis."""
        v_tile = module.allocate_iter_var(axis_id, total, role)
        iter_vars.append(v_tile)
```

`_build_buffer_access` — coefficient logic per axis:
- Bounded: `{outer.var_id: tile, inner.var_id: 1}`, extent = tile.
- Unbounded: `{tile_iv.var_id: 1}`, extent = total.

Tree wrap: unchanged, except unbounded axes contribute one ForNode instead of two. Structure is `for each axis: ForNode(outer?) > ForNode(tile) > ... > SBlock`, with the outer only present for bounded axes.

## Renderer Change

**None.** `_innermost_tile_iter_var_ids` stays as-is — it collects the per-axis last iter-var per SBlock; whether the SBlock has 1 or 2 iter-vars per axis, the last one is the tile, and the renderer elides it. For unbounded axes, the only loop (now the sole tile ForNode) is elided; no Python `for` is emitted for that axis. For bounded axes, the outer emits as `for outer in range(trip):`; the inner elides.

## Atom Changes

**None structural.** Existing legality checks already use "innermost-per-axis in affected SBlock" (fixed after Task 6 of the TVM-style split+fuse work). That check works the same whether the axis has 2 or 1 iter-vars.

**Split legality in practice:**
- **Bounded axis**, e.g. matmul M canonical `(16, 128)`: splitting the outer (trip=16) is legal when factors preserve total and innermost ≥ MIN. `Split(outer, factor=4)` → `(4, 4, 128)` legal; inner 128 stays ≥ 128. Splitting the inner (trip=128) is constrained: `Split(inner, factor=N)` must have `N ≥ 128`; only `(128)` stays, so no meaningful split.
- **Unbounded axis**, e.g. lhs_T load F canonical `(2048)`: `Split(tile, factor=128)` → `(16, 128)` legal; new innermost 128 ≥ MIN=128. `Split(tile, factor=4)` → `(512, 4)` illegal; 4 < MIN=128.

This matches your framing exactly.

## Test Migration

Smaller than the over-aggressive version: only test fixtures and assertions for unbounded axes change.

- `test/codegen/test_canonical.py`: tests asserting "4 iter-vars per SBlock" (for 2D ops like load with P+F) become "3 iter-vars" (P outer + P inner + F tile). Update counts.
- `test/tune/test_axis_identity.py`: `_find_lhs_t_load_d1_pair` today returns the `(outer_trip_1, inner_tile_2048)` pair. After refactor, unbounded F has only the single tile ForNode. The test becomes: find the *single* d1 iter-var on the lhs_T load, confirm it has extent = 2048 = full axis extent. The Fuse/Split tests that operated on this pair need new fixtures — use matmul's bounded M axis: canonical `(16, 128)`, Split outer with factor=4 → `(4, 4, 128)` for a same-axis pair.
- `test/tune/test_fuse.py`, `test_fuse_eager_rewrite.py`: analogous — fixtures that relied on unbounded-axis trip-1-outer pairs need to construct the pair on a bounded axis instead.
- `test/codegen/test_render_tile_elision.py`: keep; renderer elision still exists and still matters for the tile loop.
- Other tests (`test_split.py`, `test_reorder.py`, `test_compute_at.py`, `test_reverse_compute_at.py`, rfactor tests): audit for unbounded-axis trip-1 assumptions; most target bounded-axis loops (matmul K/M/N) and are unaffected.

## `kernel_transforms.py` Reference

The 16 hand-written kernels have `for i_dN_0 in range(1):` trip-1 wrappers for unbounded axes. These were cosmetic parity with the old renderer. After the refactor, the renderer no longer emits trip-1 wrappers. Mechanical rewrite: drop `for ... in range(1):` lines and substitute their loop var with `0` in child slice expressions.

## Example Driver

`examples/matmul_lhsT_rhs.py`: step 0 canonical has no trip-1 loops. To reach a k1-like state (lhs_T load tiled at 128), `ComputeAt(lhs_T_load, matmul_d1_outer)` is natural — the load block sits inside matmul's d1 outer (trip=16). Its existing tile loop stays the same. One-atom reach.

## Success Criteria

- Canonical render of the matmul kernel has zero `range(1)` in Python source.
- Unbounded axes have one ForNode per axis in canonical IR; bounded axes have two (outer trip + inner tile).
- `SBlock.iter_vars` has 1 iter-var per unbounded axis + 2 per bounded axis.
- Renderer elision remains; every axis's innermost-per-SBlock loop elides into the ISA slice width.
- `kernel_transforms.py`'s 15 kernels still CPU-sim pass after mechanical rewrite to drop trip-1 wrappers.
- `test/` suite passes after test migration (modulo pre-existing `test_batch` failure).
- `examples/matmul_lhsT_rhs.py` 2-step driver (canonical → ComputeAt) passes.

## What Does NOT Change

- Bounded axis canonical structure: still outer trip + inner tile.
- Renderer: `_innermost_tile_iter_var_ids` still collects per-axis-last iter-vars; still elides them.
- `EmitCtx.innermost_tile_ids`, `_emit_affine_start`, `_emit_index_expr`: unchanged.
- Split, Fuse, Reorder, ComputeAt, ReverseComputeAt atom implementations: unchanged.
