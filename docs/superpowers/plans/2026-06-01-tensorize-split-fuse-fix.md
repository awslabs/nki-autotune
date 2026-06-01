# Tensorize Split / Fuse Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken tensorize flavor of `Split` and `Fuse` so re-tiling a buffer axis (e.g. load d1 `2048 → 16×128`) produces correct IR that renders + CPU-sims, via one shared element-space tile-arithmetic helper that Split and Fuse round-trip through.

**Architecture:** A `BufferRegion` axis stores `(lo, width)` where `lo = base + Σ(loop_var·stride)` in element space and a loop over a width-`w` tile strides by `w`. Tensorize-Split narrows a tile into factors (adds inner loop vars at the new sub-tile stride; width → innermost factor); tensorize-Fuse widens (absorbs loop vars back; width → tile×trips). Both go through one helper so they are provable inverses. The affected axis is located per-operand via the shipped `BlockNode.axis_map` (concrete→abstract).

**Tech Stack:** Python 3.12, networkx, numpy, pytest. Activate `~/venvs/kernel-env`. Spec: `docs/superpowers/specs/2026-05-31-tensorize-split-fuse-fix-design.md`.

---

## Ground truth (verified live on current `dev_1`, HEAD `37bc283`)

- **Shipped foundation (`fd4ebfa`):** `BlockNode.axis_map: dict[abstract→concrete]` (e.g. load `{P:d0, F:d1}`, matmul `{K:d0, M:d1, N:d2}`), carried through every BlockNode rebuild. `Split._current_tensorize_width(leaf, block, concrete_axis)` already inverts it; `Split.analyze` now OFFERS tensorize options (verified). `SplitOption.target_axis` is the CONCRETE dim (`"d1"`).
- **The bug (verified by build→apply→render→sim):** tensorize Split NaNs (4/4 ladder ops; memset false-passes); tensorize Fuse → maxabs 233. Root cause is element-space vs trip-space confusion in the region rewrite. Reorder + both outer-trip flavors are correct (untouched).
- **Element-space math, verified by hand** (`to_affine`/`from_affine` in `nkigym/ir/expr.py`):
  - Split load `lo = i_d1_0*2048` (width 2048) by `(16,128)`, new loop var `i_d1_0_0`: correct result is `lo = i_d1_0*2048 + i_d1_0_0*128`, width 128. **The OLD loop var's stride (2048) stays; the NEW inner loop var is added at stride 128 (the innermost factor).**
  - Fuse back (drop `i_d1_0_0`, width 128×16): `lo = i_d1_0*2048`, width 2048. Round-trips.
- **The two Split bugs in `_do_tensorize`** (`split.py:160-199`): (a) it builds `new_value_factor = _affine_split([*new_loop_vars, base_loop_var], factors)` in element space, then `substitute(base_loop_var → new_value_factor)` into the region `lo` — but the region `lo` is ALREADY `base_loop_var*2048` (tile-scaled), so substitution composes scales → `*262144`; (b) the leaf `operand_bindings` only get `_shrink_region` (width-only), never the lo update.
- **The two Fuse bugs in `_do_tensorize`** (`fuse.py:175-216`): (a) `_widen_region_axis(region, op_cls, slot, target_axis, ...)` checks `target_axis in OPERAND_AXES[slot]` but `target_axis` is CONCRETE `d2` while OPERAND_AXES is abstract `(K,N)` → no-op; (b) it passes `absorbed_extent = prod(trips) = 4` as the new width, but correct width is `prod(trips)×old_tile = 2048`.
- **Preserved TDD red:** the failing tensorize-apply test is shelved at `/tmp/failing_tensorize_apply_test.py` (`test_split_tensorize_load_d1_to_16x128`). Restore it in Task 2.
- **`expr` API:** `to_affine(expr) -> dict[str|None, int]` (None key = const term), `from_affine(coeffs) -> Expr`, `substitute(expr, {name: Expr})`, `Var(name=)`, `Const(value=)`, `Mul(left=,right=)`, `format_expr(expr)`.

## File Structure

- `nkigym/src/nkigym/transforms/_tile_region.py` (NEW) — the shared element-space helper: `narrow_region_axis` (Split) + `widen_region_axis` (Fuse), inverses, plus `retile_region` (locates the axis on a region via the op's `OPERAND_AXES` and applies a narrow/widen closure — used by both Split and Fuse for leaf bindings AND block regions, so the axis-index lookup lives in ONE place). Owns ALL re-tiling arithmetic. ~110 lines.
- `split.py` — `_do_tensorize` calls `narrow_region_axis` on block regions AND leaf bindings; delete the broken `_shrink_region`.
- `fuse.py` — `_do_tensorize` calls `widen_region_axis`; delete the broken `_widen_region_axis`.
- `test/transforms/test_tile_region.py` (NEW) — helper round-trip + element-space units.
- `test/transforms/test_split.py`, `test_fuse.py` — tensorize render+sim tests (replace topology-only).

---

## Task 1: The shared element-space helper `_tile_region.py`

**Files:**
- Create: `nkigym/src/nkigym/transforms/_tile_region.py`
- Test: `test/transforms/test_tile_region.py` (NEW)

The single source of truth for re-tiling one buffer-region axis. Pure functions over `(lo: Expr, width: int)` in element space; no tree/IR.

- [ ] **Step 1: Write the failing round-trip + element-space tests**

Create `test/transforms/test_tile_region.py`:

```python
"""Unit tests for the element-space tile-arithmetic helper."""

from __future__ import annotations

from nkigym.ir.expr import Const, Mul, Var, format_expr
from nkigym.transforms._tile_region import narrow_region_axis, widen_region_axis


def test_narrow_adds_inner_loop_var_at_subtile_stride():
    """Splitting a width-2048 tile (lo=i_d1_0*2048) by (16,128) keeps the old loop var's
    stride and adds the new inner loop var at the innermost factor's stride (128)."""
    lo = Mul(left=Var(name="i_d1_0"), right=Const(value=2048))
    new_lo, new_width = narrow_region_axis(lo, 2048, ["i_d1_0_0"], (16, 128))
    assert format_expr(new_lo) == "i_d1_0 * 2048 + i_d1_0_0 * 128"
    assert new_width == 128


def test_widen_drops_loop_vars_and_scales_width():
    """Fusing the inner loop var back (16 trips) drops its term and widens the tile 128->2048."""
    lo = Mul(left=Var(name="i_d1_0"), right=Const(value=2048))
    """Start from the post-narrow lo."""
    narrowed, _ = narrow_region_axis(lo, 2048, ["i_d1_0_0"], (16, 128))
    back_lo, back_width = widen_region_axis(narrowed, ["i_d1_0_0"], 128, 16)
    assert format_expr(back_lo) == "i_d1_0 * 2048"
    assert back_width == 2048


def test_narrow_then_widen_round_trips():
    """narrow then widen with matching factors returns the original (lo, width)."""
    lo = Mul(left=Var(name="i_d2_0"), right=Const(value=2048))
    narrowed, w = narrow_region_axis(lo, 2048, ["i_d2_0_0"], (4, 512))
    assert w == 512
    back_lo, back_w = widen_region_axis(narrowed, ["i_d2_0_0"], 512, 4)
    assert format_expr(back_lo) == "i_d2_0 * 2048"
    assert back_w == 2048


def test_narrow_three_factors():
    """A 3-factor split (8, 2, 128) of a 2048 tile adds two inner loop vars with strides
    256 (=2*128) and 128, width 128."""
    lo = Mul(left=Var(name="i_d1_0"), right=Const(value=2048))
    new_lo, new_width = narrow_region_axis(lo, 2048, ["i_d1_0_0", "i_d1_0_1"], (8, 2, 128))
    assert format_expr(new_lo) == "i_d1_0 * 2048 + i_d1_0_0 * 256 + i_d1_0_1 * 128"
    assert new_width == 128


def test_retile_region_narrows_the_matching_axis_only():
    """retile_region finds the F-axis range on a load dst (P,F) region and narrows it;
    a region whose op-operand lacks the axis is returned unchanged."""
    from math import prod

    from nkigym.ir.tree import BufferRegion
    from nkigym.ops.load import NKILoad
    from nkigym.transforms._tile_region import retile_region

    """Load dst: (P=i_d0_0 width128, F=i_d1_0*2048 width2048). Narrow F by (16,128)."""
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=((Var(name="i_d0_0"), Const(value=128)), (Mul(left=Var(name="i_d1_0"), right=Const(value=2048)), Const(value=2048))),
    )
    out = retile_region(region, NKILoad, "F", lambda lo, w: narrow_region_axis(lo, w, ["i_d1_0_0"], (16, 128)))
    assert format_expr(out.ranges[1][0]) == "i_d1_0 * 2048 + i_d1_0_0 * 128"
    assert out.ranges[1][1].value == 128
    """Axis P untouched."""
    assert format_expr(out.ranges[0][0]) == "i_d0_0" and out.ranges[0][1].value == 128
```

Add `from nkigym.transforms._tile_region import narrow_region_axis, widen_region_axis, retile_region` to the test file's imports.

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_tile_region.py -v`
Expected: FAIL — `nkigym.transforms._tile_region` does not exist.

- [ ] **Step 3: Implement the helper**

Create `nkigym/src/nkigym/transforms/_tile_region.py`:

```python
"""Element-space re-tiling arithmetic for tensorize Split / Fuse.

A buffer-region axis is ``(lo, width)`` where ``lo`` is an affine
combination of loop vars in ELEMENT space and a loop iterating a tile of
``width`` elements strides by ``width``. Splitting a tile into factors
adds inner loop vars (each striding by the element-size of everything
nested inside it) and shrinks the tile to the innermost factor; fusing
loops back drops those terms and widens the tile. :func:`narrow_region_axis`
and :func:`widen_region_axis` are inverses, so tensorize Split and Fuse
round-trip.
"""

from __future__ import annotations

from math import prod

from nkigym.ir.expr import Const, Expr, from_affine, to_affine


def narrow_region_axis(lo: Expr, old_tile: int, new_loop_vars: list[str], factors: tuple[int, ...]) -> tuple[Expr, int]:
    """Split a tile of width ``old_tile`` into ``factors`` (product == old_tile).

    ``new_loop_vars`` are the ``len(factors) - 1`` new OUTER loop vars
    (the innermost factor stays bound to the existing loop var already in
    ``lo`` and needs no new var). Each new loop var ``i`` strides by
    ``prod(factors[i+1:])`` — the element-size of everything nested below
    it. The existing terms in ``lo`` are preserved unchanged. Returns
    ``(new_lo, new_width=factors[-1])``.
    """
    assert prod(factors) == old_tile, f"factors {factors} product != old_tile {old_tile}"
    assert len(new_loop_vars) == len(factors) - 1, f"{len(new_loop_vars)} loop vars for {len(factors)} factors"
    coeffs = dict(to_affine(lo))
    for i, var in enumerate(new_loop_vars):
        stride = prod(factors[i + 1 :])
        coeffs[var] = coeffs.get(var, 0) + stride
    return from_affine(coeffs), factors[-1]


def widen_region_axis(lo: Expr, absorbed_loop_vars: list[str], old_tile: int, num_trips: int) -> tuple[Expr, int]:
    """Absorb ``absorbed_loop_vars`` (total ``num_trips`` iterations) back into the tile.

    Inverse of :func:`narrow_region_axis`: drops the absorbed loop-var
    terms from ``lo`` (their footprint is now covered by the wider tile)
    and widens the tile to ``old_tile * num_trips``. Returns
    ``(new_lo, new_width)``.
    """
    coeffs = dict(to_affine(lo))
    for var in absorbed_loop_vars:
        coeffs.pop(var, None)
    return from_affine(coeffs), old_tile * num_trips


def retile_region(region: BufferRegion, op_cls: type, abstract_axis: str | None, rewrite) -> BufferRegion:
    """Apply ``rewrite(lo, width) -> (new_lo, new_width)`` to ``region``'s range on
    ``abstract_axis``, returning the region unchanged if the op's operand for this
    region's tensor does not carry that axis.

    Locates the axis index by matching ``region.tensor`` to the operand slot whose
    ``OPERAND_AXES`` contains ``abstract_axis``. Block regions and leaf operand
    bindings share axis order (both built from ``OPERAND_AXES``), so this works for
    both. ``rewrite`` is a closure over the element-space helper (narrow or widen).
    """
    if abstract_axis is None:
        return region
    idx: int | None = None
    for _slot, axes in op_cls.OPERAND_AXES.items():
        if abstract_axis in axes and axes.index(abstract_axis) < len(region.ranges):
            idx = axes.index(abstract_axis)
            break
    if idx is None:
        return region
    lo, width = region.ranges[idx]
    assert isinstance(width, Const), f"region width must be Const; got {width!r}"
    new_lo, new_width = rewrite(lo, width.value)
    new_ranges = list(region.ranges)
    new_ranges[idx] = (new_lo, Const(value=new_width))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))
```

> **NOTE:** `retile_region` assumes every operand carrying `abstract_axis` puts it
> at the same range index (true here — load `(P,F)`, matmul `(K,M)/(K,N)/(M,N)` all
> place a given axis consistently). It uses the FIRST slot containing the axis to
> find the index; since all slots agree, that's correct. Add
> `from nkigym.ir.tree import BufferRegion` to the helper's imports.

> **VERIFIED (alignment check, do not skip):** block `reads`/`writes` regions are
> keyed by tensor name and map 1:1 to leaf operand slots with identical axis order
> (confirmed: load block `reads=[lhs_T]`, `writes=[sbuf_lhs_T]`, leaf `{src:lhs_T,
> dst:sbuf_lhs_T}`, both 2D `(P,F)`, F-axis index 1). So Split/Fuse retile block
> regions and leaf bindings with the SAME `retile_region` call.

- [ ] **Step 4: Run the tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_tile_region.py -v`
Expected: PASS (4 tests). If `from_affine` orders terms differently than the expected strings (e.g. omits the `* 1` or sorts vars), read the actual `format_expr` output and correct the expected strings to match — do NOT change `from_affine`.

- [ ] **Step 5: Full suite (standalone module, no wiring yet)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS (158 + 4 new = 162). Paste the real summary line.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/_tile_region.py test/transforms/test_tile_region.py
git commit -m "Add _tile_region: element-space narrow/widen helper for tensorize Split/Fuse

narrow_region_axis (Split) adds inner loop vars at sub-tile strides + shrinks
the tile to the innermost factor; widen_region_axis (Fuse) drops absorbed loop
vars + scales the tile by trips. Inverses, unit-tested round-trip. Standalone."
```

---

## Task 2: Fix `Split._do_tensorize` to use `narrow_region_axis`

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py` (`_do_tensorize`, remove `_shrink_region`)
- Test: `test/transforms/test_split.py` (restore the shelved render+sim test)

- [ ] **Step 1: Restore the shelved failing test**

The shelved test is at `/tmp/failing_tensorize_apply_test.py`. Append it to `test/transforms/test_split.py` (check its imports — it needs `importlib.util`, `numpy`, `render`, `simulate_fp32`, `build_canonical_ir`, `INPUT_SPECS`, `ISANode`, `Const`; add any missing to the test file's imports). The test:

```python
def test_split_tensorize_load_d1_to_16x128(tmp_path):
    """Tensorize-Split the load's d1 free-axis tile 2048 -> (16, 128): the load's dst tile
    shrinks to 128, gains a 16-trip loop, and the kernel renders + sims correctly."""
    import importlib.util

    import numpy as np

    from test.transforms._fixtures import build_canonical_ir, INPUT_SPECS

    from nkigym.codegen import render
    from nkigym.ir.expr import Const
    from nkigym.ir.tree import ISANode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    load_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    new_leaf = next(
        n for n in new_ir.tree.preorder()
        if isinstance(new_ir.tree.data(n), ISANode) and new_ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    dst = new_ir.tree.data(new_leaf).operand_bindings["dst"]
    assert any(isinstance(w, Const) and w.value == 128 for _lo, w in dst.ranges), dst.ranges
    src = render(new_ir)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run to verify it fails (NaN)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_split.py::test_split_tensorize_load_d1_to_16x128 -v`
Expected: FAIL — `assert_allclose` mismatch (NaN), the pre-existing `_do_tensorize` bug.

- [ ] **Step 3: Rewrite `_do_tensorize` to use `narrow_region_axis` on block regions AND leaf bindings**

Read `split.py` `_do_tensorize` (currently lines ~160-199). Keep the loop-insertion part (it creates `new_loop_vars` and the new ForNodes — that's correct). Replace the region-rewrite part — from the `"""Shrink the leaf's operand_bindings..."""` block through the `new_block` construction — with this, using the shared `retile_region` helper (Task 1) for BOTH leaf bindings and block regions:

```python
        """Translate the concrete target_axis (e.g. d1) to the abstract op-axis (e.g. F)."""
        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        abstract_axis = inverse_axis_map.get(option.target_axis)
        old_tile = prod(option.factors)

        def _narrow(lo, width):
            return narrow_region_axis(lo, old_tile, new_loop_vars, option.factors)

        new_bindings = {
            slot: retile_region(region, leaf.op_cls, abstract_axis, _narrow)
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=block.iter_values,
            reads=tuple(retile_region(r, leaf.op_cls, abstract_axis, _narrow) for r in block.reads),
            writes=tuple(retile_region(w, leaf.op_cls, abstract_axis, _narrow) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
            axis_map=block.axis_map,
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block
```

`retile_region` (Task 1) ignores the `width` arg in `_narrow` because `narrow_region_axis` derives the new width from `factors`; the closure signature `(lo, width)` matches what `retile_region` calls. Block regions and leaf bindings both go through the same `retile_region` — the VERIFIED alignment (load block `reads=[lhs_T]`/`writes=[sbuf_lhs_T]`, F-axis index 1) means both retile correctly.

Notes:
- `iter_values` does NOT need rewriting: the new loop vars are fresh and the loop structure carries them; the renderer reads the region `lo`, which `retile_region` updated. (If a render test shows otherwise, STOP and report — but the hand-verified math says it doesn't.)
- Delete the now-unused `_shrink_region` function and the old `new_value_factor = _affine_split(...)` + `substitute(base_loop_var → ...)` region logic. Keep `_affine_split` (still used by `_do_outer_trip` — confirm via grep).
- Imports: add `from nkigym.transforms._tile_region import narrow_region_axis, retile_region` and `from math import prod` (if not already imported).

- [ ] **Step 4: Run the restored test**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_split.py::test_split_tensorize_load_d1_to_16x128 -v`
Expected: PASS (dst tile 128, render+sim matches numpy). If it still NaNs, STOP and report NEEDS_CONTEXT with the rendered load line — do not loosen tolerance.

- [ ] **Step 5: Add the other ladder-op tensorize tests**

Add to `test/transforms/test_split.py` (same render+sim shape, parametrized over op + axis + factors):

```python
import pytest


@pytest.mark.parametrize(
    "op_name, which, axis, factors",
    [
        ("NKILoad", 1, "d2", (4, 512)),       # k2->k3: load rhs N
        ("NKIMemset", 0, "d2", (4, 512)),     # k4->k5: memset N
        ("NKITensorCopy", 0, "d2", (4, 512)), # k10->k11: drain N
        ("NKIStore", 0, "d2", (4, 512)),      # k12->k13: store N
    ],
)
def test_split_tensorize_ladder_ops_render_and_sim(tmp_path, op_name, which, axis, factors):
    """Each tensorize-Split in the kernel_transforms ladder renders + sims correctly."""
    import importlib.util

    import numpy as np

    from test.transforms._fixtures import build_canonical_ir, INPUT_SPECS

    from nkigym.codegen import render
    from nkigym.ir.tree import ISANode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    leaves = [
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == op_name
    ]
    new_ir = Split().apply(ir, SplitOption(target_nid=leaves[which], factors=factors, target_axis=axis))
    src = render(new_ir)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

(NOTE: the second `NKILoad` is `rhs`; confirm pre-order order is lhs_T then rhs — it is, per canonical build source order.)

- [ ] **Step 6: Run all Split tests + full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_split.py -v && python -m pytest test/ -q`
Expected: PASS. If any ladder op NaNs, STOP and report which + the rendered op line.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/transforms/split.py test/transforms/test_split.py
git commit -m "Fix Split tensorize: rewrite region lo via element-space narrow_region_axis

_do_tensorize substituted an element-space affine into the already-tile-scaled
region lo (double-scaling the stride to *262144) and never touched the leaf
operand_bindings lo. Now retile both block regions and leaf bindings on the
affected axis (located via axis_map) through narrow_region_axis. Deletes the
width-only _shrink_region. All 5 ladder tensorize-Splits render + fp32-sim."
```

---

## Task 3: Fix `Fuse._do_tensorize` to use `widen_region_axis`

**Files:**
- Modify: `nkigym/src/nkigym/transforms/fuse.py` (`_do_tensorize`, remove `_widen_region_axis`)
- Test: `test/transforms/test_fuse.py` (upgrade the topology-only tensorize test to render+sim; add round-trip)

- [ ] **Step 1: Write the failing render+sim test (replace topology-only)**

In `test/transforms/test_fuse.py`, replace `test_fuse_tensorize_absorbs_loop_into_leaf_tile`'s topology-only assertion with a render+sim gate, and add a Split↔Fuse round-trip test:

```python
def test_fuse_tensorize_matmul_n_renders_and_sims(tmp_path):
    """Tensorize-Fuse the matmul's innermost N loop (i_d2_0, 4 trips) back into the tile
    (512 -> 2048): renders + sims correctly. (Topology-only assertion was insufficient —
    it never caught that the tile width stayed 512.)"""
    import importlib.util

    import numpy as np

    from test.transforms._fixtures import build_canonical_ir, INPUT_SPECS

    from nkigym.ir.expr import Var
    from nkigym.ir.tree import BlockNode, ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Fuse, FuseOption

    ir = build_canonical_ir()
    leaf_nid = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    parent_for = ir.tree.parent(leaf_nid)  # i_d2_0, the innermost matmul loop
    mb = next(
        ir.tree.data(a) for a in reversed(ir.tree.ancestors(leaf_nid))
        if isinstance(ir.tree.data(a), BlockNode) and ir.tree.data(a).iter_vars
    )
    target_axis = next(
        iv.axis for iv, v in zip(mb.iter_vars, mb.iter_values)
        if isinstance(v, Var) and v.name == ir.tree.data(parent_for).loop_var
    )
    fused = Fuse().apply(ir, FuseOption(target_nids=(parent_for, leaf_nid), target_axis=target_axis))
    src = render(fused)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_split_then_fuse_tensorize_round_trips(tmp_path):
    """Tensorize-Split the load d1 (2048->16x128) then tensorize-Fuse it back == original;
    both intermediate and final render + sim correctly."""
    import importlib.util

    import numpy as np

    from test.transforms._fixtures import build_canonical_ir, INPUT_SPECS

    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    load_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    split_ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    """The load now has a new inner ForNode above the leaf; fuse it back."""
    new_leaf = next(
        n for n in split_ir.tree.preorder()
        if isinstance(split_ir.tree.data(n), ISANode) and split_ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    inner_for = split_ir.tree.parent(new_leaf)
    assert isinstance(split_ir.tree.data(inner_for), ForNode)
    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(inner_for, new_leaf), target_axis="d1"))
    src = render(fused_ir)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

Keep `test_fuse_outer_trip_inverts_split` (it tests the working outer-trip flavor). Delete the old topology-only `test_fuse_tensorize_absorbs_loop_into_leaf_tile`.

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_fuse.py::test_fuse_tensorize_matmul_n_renders_and_sims -v`
Expected: FAIL — `assert_allclose` mismatch (tile width stayed 512; maxabs ~233).

- [ ] **Step 3: Rewrite `Fuse._do_tensorize` to use `widen_region_axis`**

Read `fuse.py` `_do_tensorize` (lines ~175-228). It keeps `absorbed_extent = prod(trips)` and `absorbed_loop_vars` (correct — keep them). Replace BOTH the leaf-binding widen (currently `_widen_region_axis(...)`) AND the block region rewrite (currently `substitute(... Const(0))`, which drops the lo term but leaves width wrong) with the shared `retile_region` helper (Task 1) and a `_widen` closure:

```python
        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        abstract_axis = inverse_axis_map.get(option.target_axis)

        def _widen(lo, width):
            return widen_region_axis(lo, absorbed_loop_vars, width, absorbed_extent)

        new_bindings = {
            slot: retile_region(region, leaf.op_cls, abstract_axis, _widen)
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=block.iter_values,
            reads=tuple(retile_region(r, leaf.op_cls, abstract_axis, _widen) for r in block.reads),
            writes=tuple(retile_region(w, leaf.op_cls, abstract_axis, _widen) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
            axis_map=block.axis_map,
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block
```

Here `retile_region`'s closure DOES use the `width` arg (`widen_region_axis` needs the current tile width). Same shared helper as Split — block regions and leaf bindings both go through it. Delete `_widen_region_axis`. Delete the old `substitute(... Const(0))` block-region rewrite and its `substitutions` dict (the `_widen` closure now drops the absorbed loop vars from `lo` AND widens the tile). Keep the `iter_values` handling consistent with Split (no rewrite needed — but if the existing code substituted Const(0) into iter_values too, preserve that for the iter_value binding, since the absorbed loop var is genuinely gone from the loop structure). Add `from nkigym.transforms._tile_region import widen_region_axis, retile_region`.

> **VERIFY:** after fuse, the matmul `moving`/`dst` N-axis width is 2048 and the `i_d2_0` term is gone from their `lo`. `stationary` (axes K,M — no N) is returned unchanged by `retile_region`. If the matmul still sims wrong, STOP and report the rendered matmul line.

- [ ] **Step 4: Run the fuse tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_fuse.py -v`
Expected: PASS (matmul-N render+sim, round-trip, outer-trip-inverts all green). If NaN/mismatch, STOP and report the rendered matmul line.

- [ ] **Step 5: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS. Paste the real summary line.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/fuse.py test/transforms/test_fuse.py
git commit -m "Fix Fuse tensorize: widen tile via element-space widen_region_axis

_widen_region_axis no-opped (concrete d2 not in abstract OPERAND_AXES) and set
width to the trip count (4) not elements (2048). Now widen both leaf bindings
and block regions on the axis-map-located axis through widen_region_axis
(width = old_tile * trips, absorbed loop vars dropped from lo). Upgrades the
topology-only test to render+sim; adds a Split<->Fuse tensorize round-trip."
```

---

## Task 4: Whole-ladder regression + resume-readiness

**Files:**
- Test: `test/transforms/test_render_equivalence.py` (add a tensorize chain) — optional consolidation
- Modify: `docs/superpowers/plans/2026-05-31-compute-at.md` (mark Task 5.5 SHIPPED via this plan)

- [ ] **Step 1: Verify the full canonical → k1 setup the compute_at plan needs**

Run this sanity script (not a committed test — a readiness check):

```bash
source ~/venvs/kernel-env/bin/activate && PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -c "
from test.transforms._fixtures import build_canonical_ir
from nkigym.ir.tree import ISANode
from nkigym.transforms import Split, SplitOption
from nkigym.codegen import render
ir = build_canonical_ir()
leaf = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n),ISANode) and ir.tree.data(n).op_cls.__name__=='NKILoad')
new = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16,128), target_axis='d1'))
print([l for l in render(new).splitlines() if 'sbuf_lhs_T = nl.ndarray' in l or 'dst=sbuf_lhs_T' in l][:2])
"
```
Expected: `sbuf_lhs_T` still allocates full (compaction not in this plan) but the load's dst slice reads `[..., i_d1_0*..., (i_d1_0_0)*128 : +128]`-shaped indexing with width 128 and BOTH loop vars present — i.e. k1 is reachable. (This confirms compute_at Task 6's k0→k1 step will work.)

- [ ] **Step 2: Mark compute_at plan's Task 5.5 as shipped-by-this-plan**

In `docs/superpowers/plans/2026-05-31-compute-at.md`, Task 5.5's header note: append "SUPERSEDED — the axis_map foundation shipped (`fd4ebfa`) and the tensorize region-rewrite is fixed by `docs/superpowers/plans/2026-06-01-tensorize-split-fuse-fix.md`. When resuming, Task 6's k0→k1 setup uses the now-working tensorize-Split."

- [ ] **Step 3: Full suite final + commit the doc note**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: all green.

```bash
git add docs/superpowers/plans/2026-05-31-compute-at.md
git commit -m "Mark compute_at Task 5.5 superseded by the tensorize fix plan"
```

---

## Self-Review (completed)

**Spec coverage:** shared element-space helper → Task 1; Split `lo` rescale (both block + leaf) → Task 2; Fuse no-op + width → Task 3; render+sim on every tensorize test + round-trip → Tasks 2-3; the `axis_map` translation is the shipped foundation, consumed in Tasks 2-3. Reorder/outer-trip explicitly untouched. The memset-in-bounds check is covered by the ladder parametrize in Task 2 Step 5 (memset·d2 now sims correctly).

**Placeholder scan:** every code step has full code, including the shared `retile_region` helper (Task 1) that both Split and Fuse use for leaf bindings AND block regions — the earlier per-task block-region soft spot is gone (one helper, full code, unit-tested in Task 1, axis alignment verified live). No remaining placeholders.

**Type consistency:** `narrow_region_axis(lo, old_tile, new_loop_vars, factors) -> (Expr, int)` and `widen_region_axis(lo, absorbed_loop_vars, old_tile, num_trips) -> (Expr, int)` used consistently across Tasks 1-3. `target_axis` is concrete throughout; `abstract_axis` is the translated value. `BlockNode.axis_map`, `BufferRegion.ranges` as `(lo, width)`, `Const(value=)` all match shipped code.
