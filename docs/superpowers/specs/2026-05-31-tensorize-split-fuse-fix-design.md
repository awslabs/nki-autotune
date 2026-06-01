# Fix Tensorize Split / Fuse (region re-tiling)

## SUPERSEDED (2026-06-01)

The region-arithmetic fix described here SHIPPED (`afa84fa`→`b000a7e`) but
strict byte-exact verification then exposed two structural defects it did
not address: double-suffix loop names (`i_d1_0_0`) and leftover trip-1
wrappers. The full fix is `2026-06-01-unified-split-loopvar-normalization`
(plan `2026-06-01-unified-split-fuse-normalization`, shipped
`e0d5034`→`19bd624`): a dim is a factorization `(trip…, tensorize_size)`;
no trip-1 ForNodes anywhere; dense `i_d{dim}_{N}` names via a shared
`normalize_block` pass that absorbed this spec's `_tile_region` lo-math.
Split↔Fuse now round-trip byte-exact. The text below is retained for
history; do not implement it.

## Status (2026-05-31) — original, now superseded

NOT STARTED. Prerequisite for resuming the compute_at plan
(`2026-05-31-compute-at.md`): the k0→k1/k2→k3/k4→k5/k10→k11/k12→k13
transitions in `kernel_transforms.py` are all **tensorize-Split**, which
is currently broken. The `BlockNode.axis_map` foundation is already
SHIPPED (`fd4ebfa`); this spec fixes the region-rewrite bugs on top of it.

## Problem — verified by build → apply → render → CPU-sim

I built each before-IR, applied the transform, rendered, and simulated
against the numpy golden. Findings:

| Transform · flavor | Status |
|---|---|
| Split outer-trip, Fuse outer-trip, **Reorder** | ✅ correct (no fix) |
| **Split tensorize** | ❌ broken — 4/4 ladder splits NaN; memset false-passes |
| **Fuse tensorize** | ❌ broken — matmul N-fuse → maxabs 233 |

One conceptual bug — **trip-count space vs. element space confusion** —
manifests as four symptoms, all in the tensorize paths:

1. **`Split.analyze` offered nothing** (`_current_tensorize_width`) —
   FIXED by the shipped `axis_map` foundation (`fd4ebfa`): the concrete
   axis (`d2`) now translates to the abstract op-axis (`N`) for the
   `OPERAND_AXES` lookup.
2. **Split `lo` not rescaled** (`Split._do_tensorize` / `_shrink_region`):
   shrinks the *width* (2048→512) but leaves the region `lo` at the OLD
   tile stride (`i_d2_0 * 2048` instead of `* 512`). The new tile loop
   strides by the old tile size → walks out-of-bounds. The leaf
   `operand_bindings` `lo` is never substituted at all.
3. **Fuse widen no-ops** (`Fuse._widen_region_axis`): receives the
   concrete axis `d2`, checks `"d2" in ("K","N")` → False → returns the
   region UNCHANGED, so the tile never widens.
4. **Fuse width is trip count, not elements** (`Fuse._do_tensorize`):
   passes `absorbed_extent = prod(trips) = 4` as the new width; the
   correct widened width is `prod(trips) × current_tile = 4 × 512 = 2048`.

Why memset Split "passes": its out-of-bounds zeroing writes are mooted by
the matmul overwriting all of PSUM afterward — a false positive that
masked the identical bug. Why no test caught any of this: every shipped
Split/Fuse test uses the **outer-trip** flavor; the one tensorize test
(`test_fuse_tensorize_absorbs_loop_into_leaf_tile`) asserts **topology
only** — never renders or sims.

## Root cause

`_affine_split` builds loop bindings in **iter-var / element space**
(`i_d2_0_0 * 512`), but a `BufferRegion` `lo` is stored in **tile-scaled
space** (`loop_var * tile_width`). Split's `substitute` injects the
element-space affine into the tile-scaled `lo`, composing the two scales
(`* 2048 * ...`). Fuse's `_widen_region_axis` sets width to a trip count
rather than an element extent. Both transforms treat "the number that
multiplies the loop var" and "the tile width" as the same thing; they are
not. The fix routes both through ONE element-space tile-arithmetic helper.

## Design — shared tile-arithmetic helper

Add `nkigym/src/nkigym/transforms/_tile_region.py` (new) with the single
source of truth for re-tiling a `BufferRegion` axis in **element space**.
Both Split (narrowing) and Fuse (widening) call it, so they provably
round-trip.

### The region model (element space)

For a buffer axis, a tiled access is `lo = base + Σ(loop_var_i ·
stride_i)`, `width = tile`. The **stride of a loop over a tile of width
`w` is `w`** (consecutive tiles are `w` apart). Re-tiling splits/merges
this:

- **Narrow (Split)** a tile of width `W` into `factors = (f_0, …, f_{n-1})`
  (product `W`): the innermost tile becomes `f_{n-1}`; new outer loop
  vars `v_0…v_{n-2}` get strides `f_{n-1}·∏(f_{i+1..n-2})` … i.e. each
  new loop strides by the element-size of everything inside it. The
  region `lo` gains `Σ vᵢ·strideᵢ`; `width` becomes `f_{n-1}`.
- **Widen (Fuse)** absorbs loops with trips `(t_0…t_{k-1})` back: the
  tile width becomes `W · ∏tᵢ`; the absorbed loop-var terms drop from
  `lo` (their contribution is now covered by the wider tile).

### Helper API

```python
def retile_axis_lo(lo: Expr, old_tile: int, new_loop_vars: list[str],
                   new_tile: int) -> Expr:
    """Rewrite a region's axis `lo` when its tile of width `old_tile` is
    split into `len(new_loop_vars)+1` factors with innermost width
    `new_tile`. Returns `lo` with the OLD loop-var stride rebased to the
    new tile and the new outer loop-var terms added (each striding by its
    element-size). Element space throughout."""

def widen_axis(lo: Expr, absorbed_loop_vars: list[str], old_tile: int,
               num_trips: int) -> tuple[Expr, int]:
    """Inverse: absorb `absorbed_loop_vars` (total `num_trips`) into the
    tile. Returns (lo with those terms dropped, new_tile = old_tile*num_trips)."""
```

(Exact signatures may be refined in the plan; the invariant is that both
operate on element-space `(lo, width)` and are inverses.)

### Concrete-axis targeting

Both transforms locate the affected axis on each operand via
`block.axis_map` (shipped): the concrete `target_axis` (`d2`) → abstract
op-axis (`N`) → the region range index for that operand (an operand that
doesn't carry the axis, e.g. matmul `stationary` for `N`, is skipped —
it has no range on that axis). This replaces the broken
`"d2" in OPERAND_AXES` checks in `_shrink_region` (Split) and
`_widen_region_axis` (Fuse).

### What changes

- **Split `_do_tensorize`**: compute the new `lo` for the affected axis
  on BOTH the block reads/writes AND the leaf `operand_bindings`, via
  `retile_axis_lo`; set width to `factors[-1]`. Replace `_shrink_region`
  (width-only, axis-name-blind) with the axis-targeted helper.
- **Fuse `_do_tensorize`**: widen via `widen_axis` (width =
  `old_tile × prod(trips)`), targeting the axis through `axis_map`;
  replace `_widen_region_axis`. Drop the absorbed loop vars from `lo`
  through the helper, not a blanket `Const(0)` substitution (which
  zeroes the term but leaves the width wrong).
- **`_current_tensorize_width`** (Split): already fixed by `axis_map`
  (shipped); confirm it reads the affected operand's current tile.

## Verification — render + sim, and round-trip (the missing gate)

The topology-only tests are why this rotted. Every tensorize test
renders + CPU-sims against `lhs_T.T @ rhs` at `atol=rtol=5e-3`.

`test/transforms/test_split.py`:
- **Tensorize-Split each ladder op** (the preserved failing test
  `test_split_tensorize_load_d1_to_16x128`, plus rhs·d2, memset·d2,
  tensor_copy·d2, store·d2): apply → render → sim pass; assert the
  affected dst tile width shrank (e.g. 2048→128 or 2048→512) AND the new
  loop var appears in the rendered slice with the correct stride.
- **memset in-bounds**: after Split, the memset writes within the PSUM
  free axis (no out-of-bounds offset) — the false-pass becomes a real
  pass.

`test/transforms/test_fuse.py`:
- **Tensorize-Fuse matmul·N** (`4×512 → 2048`): apply → render → sim pass
  (replaces the topology-only `test_fuse_tensorize_absorbs_loop_into_leaf_tile`,
  which should be upgraded to render+sim).

`test/transforms/test_tile_region.py` (NEW):
- **Round-trip**: `retile_axis_lo` then `widen_axis` (same factors)
  returns the original `(lo, width)`. Unit-level, several axis shapes.
- Element-space unit checks: narrow `W=2048` by `(16,128)` →
  innermost tile 128, new loop stride 128; widen back → 2048.

## Layout

```
nkigym/src/nkigym/transforms/
├── _tile_region.py   # NEW — retile_axis_lo, widen_axis (element-space, inverses)
├── split.py          # EDIT — _do_tensorize uses retile_axis_lo on block + leaf; drop _shrink_region
└── fuse.py           # EDIT — _do_tensorize uses widen_axis; drop _widen_region_axis

test/transforms/
├── test_tile_region.py   # NEW — round-trip + element-space units
├── test_split.py         # EDIT — tensorize render+sim per ladder op (restore the shelved failing test)
└── test_fuse.py          # EDIT — upgrade the tensorize test to render+sim
```

The shelved failing test is at `/tmp/failing_tensorize_apply_test.py`
(`test_split_tensorize_load_d1_to_16x128`) — restore it; it is the
canonical TDD red for Split tensorize.

## Implementation order

1. `_tile_region.py` + `test_tile_region.py` (the helper, round-trip
   unit-tested in isolation — no IR needed).
2. Split `_do_tensorize` → restore + pass the load·d1 render+sim test;
   add the other ladder splits.
3. Fuse `_do_tensorize` → upgrade its test to render+sim; round-trip
   Split↔Fuse on d2.
4. Full suite green; then resume the compute_at plan (Task 5.5 is now
   subsumed/SHIPPED — the compute_at plan's Task 6 can chain a real
   tensorize-Split for its k0→k1 setup).

## Out of scope

- Reorder (verified correct), Split/Fuse outer-trip (verified correct).
- compute_at itself (separate plan; this unblocks its k1 test setup).
- Multi-factor (>2) tensorize splits beyond what `_factorizations`
  already enumerates — the helper handles N factors, but tests focus on
  the 2-factor ladder cases.
