# Unified Split/Fuse + Loop-Var Normalization + No Trip-1 Loops — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the iteration model so Split/Fuse produce IR byte-identical to the hand-written `kernel_transforms.py` ladder: no trip-1 ForNodes anywhere (incl. canonical), dense position-in-dim loop names (`i_d{dim}_{N}`, never `i_d1_0_0`/`i_d1_fused`), via one shared `normalize_block` pass that both transforms call.

**Architecture:** A dim's iteration is a factorization `(trip₀,…,trip_{k-1}, tensorize_size)`; one ForNode per trip>1, tensorize_size = the access BufferRegion width. After any Split/Fuse mutates a block's ForNode chain, `normalize_block(tree, block_nid)` drops trip-1 ForNodes, renumbers each dim's loops dense, and rewrites iter_values + region lo's. Canonical build stops emitting trip-1 loops. The shipped `_tile_region.py` element-space math is absorbed into `normalize_block`.

**Tech Stack:** Python 3.12, networkx, numpy, pytest. Activate `~/venvs/kernel-env`. Spec: `docs/superpowers/specs/2026-06-01-unified-split-loopvar-normalization-design.md`. Hand oracle: `kernel_transforms.py` (now trip-1-free, 15/15 sim-pass).

---

## Ground truth (verified on `dev_1`, HEAD `58643cd`, suite 170 green)

- **Canonical emits 5 trip-1 loops today** (`render(build_canonical_ir())` shows `for i_d1_0 in range(1)` ×1 + `for i_d2_0 in range(1)` ×4 — load lhs_T d1, load rhs d2, memset d2, tensor_copy d2, store d2). The matmul has no trip-1.
- **trip-1 emission site:** `canonical_build.py:88-92` — `tile = extent if max_tile is None else max_tile; trip = extent // tile; <add ForNode(extent=trip)>`. The fix: add the ForNode only when `trip > 1`.
- **Split naming bug:** `split.py:119` `new_loop_vars = [f"{target.loop_var}_{i}" ...]` → `i_d1_0_0`. `_do_outer_trip` (108-159) and `_do_tensorize` (~160) are the two flavors.
- **Fuse naming bug:** `fuse.py` `_fused_loop_var = _stem + "_fused"` (260) → `i_d1_fused`. Helpers `_same_loop_axis`(242), `_stem`(252), `_fused_loop_var`(259); flavors `_do_outer_trip`(106), `_do_tensorize`(176).
- **Shipped `_tile_region.py`** (`narrow_region_axis`/`widen_region_axis`/`retile_region`) — element-space stride math; absorb into `normalize_block`.
- **Test churn (verified — only 4 files reference trip-1/`_fused`, plus golden renders):**
  - `test/codegen/test_render.py`, `test/codegen/test_body.py` — golden canonical render (currently includes the 5 trip-1 loops) → update expected text.
  - `test/ir/test_dependency.py:124,166` — hand-built `ForNode(extent=1)` in disjoint-region fixtures (NOT canonical; keep — they test the dep graph directly, trip-1 is legal in a hand-built tree).
  - `test/ir/test_ir_extensions.py:258`, `test/ir/test_node_labels.py:84` — `ForNode(loop_var, extent)` shape/label (UNCHANGED by this work — keep).
  - `test/transforms/test_tile_region.py` — absorbed/removed with `_tile_region`.
  - `test/transforms/test_split.py`, `test_fuse.py` — rewrite for dense naming + byte-exact.
- **`KernelTree`/`ForNode`/`BlockNode` API:** `ForNode(loop_var: str, extent: int)`; `BlockNode(iter_vars, iter_values, reads, writes, alloc_buffers, annotations, axis_map)`; `block.axis_map: dict[abstract→concrete]`; `tree.children/parent/ancestors/descendants/preorder/blocks/data`, `tree.graph.add_edge/remove_edge/remove_node`, `_replace_in_parent_children`, `_block_local_descendants` (in `_tree_ops.py`). `expr`: `Var(name=)`, `Const(value=)`, `Mul`, `substitute`, `to_affine`, `from_affine`, `format_expr`.

## Sequencing

canonical-drops-trip-1 FIRST (it changes the baseline render every later test builds on), then `normalize_block`, then Split routes through it, then Fuse, then absorb `_tile_region`. Each task gates on the full suite.

---

## Task 1: canonical_build stops emitting trip-1 ForNodes

**Files:**
- Modify: `nkigym/src/nkigym/ir/canonical_build.py` (`_build_subblock`, the loop-emission line ~88-92)
- Modify: `test/codegen/test_render.py`, `test/codegen/test_body.py` (golden renders lose the 5 trip-1 loops)
- Test: `test/ir/test_canonical_no_trip1.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `test/ir/test_canonical_no_trip1.py`:

```python
"""Canonical IR contains no trip-1 ForNodes (the 'no trip-1 anywhere' rule)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode


def test_canonical_has_no_trip1_loops():
    """Every ForNode in canonical IR has extent > 1; trip-1 axes are loopless
    (pure tensorize_size on the access)."""
    ir = build_canonical_ir()
    trip1 = [
        ir.tree.data(n).loop_var
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).extent == 1
    ]
    assert trip1 == [], f"canonical still emits trip-1 loops: {trip1}"


def test_canonical_load_d1_is_loopless_full_width():
    """The lhs_T load's d1 (M) axis is trip-1 -> no loop; its sbuf_lhs_T write spans
    the full 2048 free width in one access."""
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    load = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    """Only the d0 (K) loop encloses the load leaf; no d1 loop."""
    loops = [ir.tree.data(a).loop_var for a in ir.tree.ancestors(load) if isinstance(ir.tree.data(a), ForNode)]
    assert loops == ["i_d0_0"], loops
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_canonical_no_trip1.py -v`
Expected: FAIL — canonical currently emits `i_d1_0`/`i_d2_0` trip-1 loops.

- [ ] **Step 3: Skip trip-1 ForNodes in `_build_subblock`**

Read `canonical_build.py` `_build_subblock`. The loop-emission block (~84-92) is:

```python
    parent_for_loops: int = block_nid
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        trip = extent // tile
        loop_var = loop_var_names[abstract]
        for_nid = tree.add_node(ForNode(loop_var=loop_var, extent=trip), parent=parent_for_loops)
        parent_for_loops = for_nid
```

Change the `add_node` to be conditional on `trip > 1`:

```python
    parent_for_loops: int = block_nid
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        trip = extent // tile
        if trip > 1:
            loop_var = loop_var_names[abstract]
            for_nid = tree.add_node(ForNode(loop_var=loop_var, extent=trip), parent=parent_for_loops)
            parent_for_loops = for_nid
```

The iter_var, iter_value binding, and the operand region (with its tile width) are UNCHANGED — a trip-1 axis keeps its iter_var and full-width region; it just gets no ForNode. The region's `lo` for a trip-1 axis was `loop_var * tile` with the loop never iterating; with no loop, the `loop_var` is unbound. CHECK: does `_build_region` produce a `lo` referencing the now-absent loop_var? Read `_build_region` — if the trip-1 axis's region lo is `Var(loop_var)*tile` or a bare `Const(0)`, confirm what renders. For the canonical load d1 (tile=2048=full extent), the lo should be `Const(0)` (offset 0, full width). VERIFY by rendering; if the lo references an unbound `i_d1_0`, fix `_build_region` to emit `Const(0)` for a trip-1 axis (lo = 0 since the single tile starts at 0).

- [ ] **Step 4: Run the no-trip1 test + render the canonical to see the new shape**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_canonical_no_trip1.py -v`
Then render to capture the new golden:
```bash
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -c "
from test.transforms._fixtures import build_canonical_ir
from nkigym.codegen import render
print(render(build_canonical_ir()))
"
```
Expected: the no-trip1 test passes; the rendered kernel has NO `for ... in range(1)` lines (load/memset/copy/store fire directly under their real loop, full-width access). Confirm it matches `kernel_transforms.py` kernel_0's structure (now trip-1-free).

- [ ] **Step 5: Update the golden-render tests**

Run `python -m pytest test/codegen/test_render.py test/codegen/test_body.py -v` to see which assertions broke. For each that compares rendered text against an expected string containing `for i_d?_0 in range(1):`, update the expected to the new trip-1-free render (drop the `range(1)` loop line; the op's slice becomes full-width `0:2048`). Use the actual render output from Step 4 as the source of truth. Do NOT weaken assertions — update the expected text to the correct new output.

- [ ] **Step 6: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS. If `test_dependency.py`'s hand-built `ForNode(extent=1)` fixtures (lines 124,166) fail, they should NOT — they construct IR directly and trip-1 is legal in a hand-built tree; only canonical-BUILD avoids it. If they fail for another reason, investigate. Paste the real summary line.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/ir/canonical_build.py test/ir/test_canonical_no_trip1.py test/codegen/test_render.py test/codegen/test_body.py
git commit -m "canonical_build: emit a ForNode only when trip > 1 (no trip-1 loops)

A trip-1 axis keeps its iter_var + full-width region but gets no loop, so
canonical IR is trip-1-free and matches the (trip-1-free) kernel_transforms
ladder. Updates golden renders. Supersedes the 'canonical KEEPS trip-1'
learning."
```

---

## Task 2: `normalize_block` — the shared invariant-enforcing pass

**Files:**
- Create: `nkigym/src/nkigym/transforms/_normalize.py`
- Test: `test/transforms/test_normalize.py` (NEW)

`normalize_block(tree, block_nid)`: after a transform mutates a block's ForNode chain, (1) drop trip-1 ForNodes, (2) renumber each dim's loops dense `i_d{dim}_{N}`, (3) rewrite iter_values + all region lo's to the new names with correct element strides.

- [ ] **Step 1: Write failing tests**

Create `test/transforms/test_normalize.py`:

```python
"""Unit tests for normalize_block: drop trip-1, dense rename, rewrite bindings."""

from __future__ import annotations

from nkigym.ir.expr import Const, Var, format_expr
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.transforms._normalize import normalize_block


def _two_loop_d1_block():
    """Hand-build a block whose d1 axis has TWO loops named non-densely (i_d1_0_0, i_d1_0_1)
    over a tile-128 load — the post-split-bug shape — plus a trip-1 loop to be dropped."""
    tree = KernelTree()
    block = BlockNode(
        iter_vars=(IterVar(axis="d1", dom=(0, 2048), role=AxisRole.PARALLEL),),
        iter_values=(Var(name="i_d1_0_0"),),
        reads=(),
        writes=(BufferRegion(tensor="sbuf", ranges=((Var(name="i_d1_0_0"), Const(value=2048)),)),),
        axis_map={"F": "d1"},
    )
    bnid = tree.add_node(block, parent=tree.root)
    outer = tree.add_node(ForNode(loop_var="i_d1_0_0", extent=2), parent=bnid)
    inner = tree.add_node(ForNode(loop_var="i_d1_0_1", extent=8), parent=outer)
    leaf = tree.add_node(
        ISANode(op_cls=NKILoad, operand_bindings={"dst": BufferRegion(tensor="sbuf", ranges=((Var(name="i_d1_0_1"), Const(value=128)),))}),
        parent=inner,
    )
    return tree, bnid, outer, inner, leaf


def test_normalize_renames_dense():
    """Two d1 loops named i_d1_0_0/i_d1_0_1 -> dense i_d1_0/i_d1_1."""
    tree, bnid, outer, inner, leaf = _two_loop_d1_block()
    normalize_block(tree, bnid)
    assert tree.data(outer).loop_var == "i_d1_0"
    assert tree.data(inner).loop_var == "i_d1_1"


def test_normalize_drops_trip1():
    """A trip-1 ForNode is removed; its child re-links to its parent."""
    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), axis_map={})
    bnid = tree.add_node(block, parent=tree.root)
    real = tree.add_node(ForNode(loop_var="i_d0_0", extent=16), parent=bnid)
    triv = tree.add_node(ForNode(loop_var="i_d1_0", extent=1), parent=real)
    leaf = tree.add_node(ISANode(op_cls=NKILoad, operand_bindings={}), parent=triv)
    normalize_block(tree, bnid)
    """trip-1 gone; leaf now child of the real loop."""
    from nkigym.ir.tree import ForNode as FN
    remaining = [tree.data(n).loop_var for n in tree.preorder(bnid) if isinstance(tree.data(n), FN)]
    assert remaining == ["i_d0_0"]
    assert tree.parent(leaf) == real
```

- [ ] **Step 2: Run to verify fail**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_normalize.py -v`
Expected: FAIL — `_normalize` does not exist.

- [ ] **Step 3: Implement `normalize_block`**

Create `nkigym/src/nkigym/transforms/_normalize.py`:

```python
"""Loop-var + trip-1 normalization for a block subtree.

After a transform (Split/Fuse) mutates a block's ForNode chain,
:func:`normalize_block` restores the two IR invariants:

* No trip-1 ForNodes — a trip-1 loop is removed, its children re-linked to
  its parent (the axis becomes loopless; its extent folds into the access
  tile width, already set by the transform).
* Dense position-in-dim names — each dim's surviving ForNodes are named
  ``i_d{dim}_{N}`` with N the loop's ordinal among that dim's loops,
  outer-to-inner. iter_values and every region ``lo`` in the block are
  rewritten to the new names with correct element strides.
"""

from __future__ import annotations

from math import prod

from nkigym.ir.expr import Const, Expr, Var, from_affine, substitute, to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children


def normalize_block(tree: KernelTree, block_nid: int) -> None:
    """Drop trip-1 ForNodes and re-densify loop-var names in this block's subtree."""
    _drop_trip1(tree, block_nid)
    _rename_dense(tree, block_nid)


def _drop_trip1(tree: KernelTree, block_nid: int) -> None:
    """Remove every trip-1 ForNode under the block, re-linking children to the parent."""
    trivial = [
        n for n in _block_local_descendants(tree, block_nid)
        if isinstance(tree.data(n), ForNode) and tree.data(n).extent == 1
    ]
    for nid in trivial:
        parent = tree.parent(nid)
        children = tree.children(nid)
        _replace_in_parent_children(tree, parent, [nid], children)
        tree.graph.remove_node(nid)


def _rename_dense(tree: KernelTree, block_nid: int) -> None:
    """Rename each dim's ForNodes to dense i_d{dim}_{N}; rewrite iter_values + regions.

    The loop->dim map comes from the block's iter_values: a ForNode's old
    loop_var appears in exactly one iter_var's affine binding. We rebuild the
    rename by walking the ForNode chain outer->inner and counting per dim.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    counters: dict[str, int] = {}
    renames: dict[str, str] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        dim = old_to_dim.get(data.loop_var)
        if dim is None:
            continue
        n = counters.get(dim, 0)
        new_name = f"i_{dim}_{n}"
        counters[dim] = n + 1
        if new_name != data.loop_var:
            renames[data.loop_var] = new_name
        tree.graph.nodes[nid]["data"] = ForNode(loop_var=new_name, extent=data.extent)
    if not renames:
        return
    subs: dict[str, Expr] = {old: Var(name=new) for old, new in renames.items()}
    new_block = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=tuple(substitute(v, subs) for v in block.iter_values),
        reads=tuple(_sub_region(r, subs) for r in block.reads),
        writes=tuple(_sub_region(w, subs) for w in block.writes),
        alloc_buffers=block.alloc_buffers,
        annotations=dict(block.annotations),
        axis_map=block.axis_map,
    )
    tree.graph.nodes[block_nid]["data"] = new_block
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ISANode):
            new_bindings = {s: _sub_region(r, subs) for s, r in data.operand_bindings.items()}
            tree.graph.nodes[nid]["data"] = ISANode(op_cls=data.op_cls, operand_bindings=new_bindings, kwargs=dict(data.kwargs))


def _loopvar_to_dim(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, str]:
    """Map each ForNode loop_var in the block to the concrete dim it binds.

    A loop_var binds the iter_var whose iter_value affine contains it. Since
    iter_values are affine over a single dim's loops, the loop_var->dim map is
    each loop_var to the iter_var.axis of the iter_value mentioning it.
    """
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value).keys():
            if name is not None:
                out[name] = iv.axis
    """Fallback for loop_vars not yet in iter_values (freshly inserted by a split):
    parse the stem i_d{dim}_N -> d{dim}."""
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var not in out:
            out[data.loop_var] = _dim_from_loopvar(data.loop_var)
    return out


def _dim_from_loopvar(loop_var: str) -> str:
    """i_d1_0 / i_d1_0_0 -> d1. Strip the i_ prefix and trailing _<int> suffixes."""
    body = loop_var[2:] if loop_var.startswith("i_") else loop_var
    parts = body.split("_")
    return parts[0]


def _sub_region(region: BufferRegion, subs: dict[str, Expr]) -> BufferRegion:
    """Substitute renamed loop vars in both lo and width of every range."""
    return BufferRegion(
        tensor=region.tensor,
        ranges=tuple((substitute(lo, subs), substitute(w, subs)) for lo, w in region.ranges),
    )


__all__ = ["normalize_block"]
```

> **IMPLEMENTER:** `_rename_dense` renames by walking ForNodes outer→inner and counting per dim. The dim of a loop comes from `_loopvar_to_dim`. The tricky bit: iter_values are affine (`i_d1_0*t1 + i_d1_1`), so `to_affine(value).keys()` gives all loop_vars binding that dim — map each to `iv.axis`. Freshly-split loops not yet in iter_values fall back to parsing the stem. VERIFY both hand-built tests pass; if the affine/stem mapping mis-assigns a dim, STOP and report.

- [ ] **Step 4: Run the normalize tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_normalize.py -v`
Expected: PASS (dense rename + trip-1 drop).

- [ ] **Step 5: Full suite (standalone, not yet wired)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS (no regression; `_normalize` is unused so far).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/_normalize.py test/transforms/test_normalize.py
git commit -m "Add normalize_block: drop trip-1 ForNodes + dense per-dim loop-var rename

Shared post-transform pass restoring the two IR invariants (no trip-1, dense
i_d{dim}_{N} names). Rewrites iter_values + region lo's to the renamed vars.
Standalone; Split/Fuse wire it next."
```

---

## Task 3: Split routes through `normalize_block`; byte-exact k0→k1

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py`
- Test: `test/transforms/test_split.py` (byte-exact + dense-naming + insertion-renumber)

- [ ] **Step 1: Write the byte-exact k0→k1 test**

Add to `test/transforms/test_split.py`:

```python
def test_split_load_d1_matches_hand_k1_byteexact(tmp_path):
    """k0->k1: split the load's d1 tensorize 2048->(16,128). The rendered load loop must
    be exactly `for i_d1_0 in range(16): dma_copy(... i_d1_0*128 : +128)` — single dense
    loop, no i_d1_0_0, no trip-1 wrapper."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.codegen import render
    from nkigym.ir.tree import ISANode
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    load = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    lines = [l.strip() for l in render(new_ir).splitlines()]
    """The lhs_T load nest: a d0 loop, a d1 loop trip-16, then the dma_copy."""
    assert "for i_d1_0 in range(16):" in lines
    assert not any("i_d1_0_0" in l for l in lines), "double-suffix name leaked"
    assert not any("range(1)" in l for l in lines), "trip-1 wrapper leaked"
    load_line = next(l for l in lines if "dst=sbuf_lhs_T" in l)
    """Index must be i_d1_0*128 : +128 (no i_d1_0 trip-1 term, no i_d1_0_0)."""
    assert "i_d1_0 * 128" in load_line and "+ 128" in load_line, load_line
    assert "* 2048" not in load_line, load_line
```

- [ ] **Step 2: Run to verify fail**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_split.py::test_split_load_d1_matches_hand_k1_byteexact -v`
Expected: FAIL — current output has `i_d1_0 * 2048 + i_d1_0_0 * 128` (the `_tile_region` result) or a trip-1 wrapper.

- [ ] **Step 3: Rewrite Split to use `normalize_block`**

Both `_do_outer_trip` and `_do_tensorize` should: insert/replace ForNodes with the raw factors (names can be temporary/non-dense), set the tile width on regions, then call `normalize_block(ir.tree, block_nid)` which fixes names + drops trip-1 + rewrites bindings. Concretely:

- `_do_outer_trip`: keep the ForNode-replacement structure (split.py:108-133) but DROP the manual `_affine_split`/`substitute` iter_value rewrite (135-159) — `normalize_block` does it. The new ForNodes can keep temporary names (e.g. `f"{target.loop_var}__tmp{i}"`); normalize renames them dense. After the topology edit, call `normalize_block(ir.tree, block_nid)`.
- `_do_tensorize`: keep the loop-insertion + the leaf/block region WIDTH shrink (the `retile_region`/`narrow_region_axis` width part), but the lo/naming is handled by `normalize_block`. Simplest: insert the new ForNodes (trips = `factors[:-1]`), set the tile width = `factors[-1]` on the affected-axis regions (leaf + block, via the shipped `retile_region` for the width only, or directly), then `normalize_block`.

> **IMPLEMENTER — this is the crux.** The cleanest unification: after EITHER flavor edits the ForNode topology + region widths, the loop-var names and the region `lo` element-strides are ALL recomputed by `normalize_block`. So Split's job shrinks to: (a) get the right ForNodes in the tree with the right extents, (b) set the right tile WIDTH on the affected regions, (c) `normalize_block`. The `lo` math (`i_d1_0*128`) must be produced by `normalize_block`'s rewrite — meaning `_rename_dense` needs to set each region's `lo` to the affine over that dim's dense loops at their element strides, NOT just substitute names. **Extend `normalize_block`** (Task 2) so `_rename_dense` recomputes each affected region's `lo` as `Σ(dense_loopᵢ · stride_i)` where `stride_i = tensorize_size · prod(inner trip extents)`. Add a Task-2 test for this if not already covered. If this coupling makes Task 2 and Task 3 hard to separate, implement the lo-recompute in Task 2 and verify it here. STOP and report if the byte-exact index can't be produced.

Read both `_do_*` methods fully before editing. Delete `_shrink_region` if now unused; keep `_affine_split` only if still referenced.

- [ ] **Step 4: Run the byte-exact test + naming tests**

Add naming tests (outer-trip split for dense names + insertion-renumber):

```python
def test_split_trip_dense_names():
    """Split a trip-16 matmul d1 loop -> (2,8): the two loops are i_d1_0, i_d1_1 (dense)."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ForNode
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    d1 = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var == "i_d1_0" and ir.tree.data(n).extent == 16)
    new_ir = Split().apply(ir, SplitOption(target_nid=d1, factors=(2, 8)))
    names = [new_ir.tree.data(n).loop_var for n in new_ir.tree.preorder() if isinstance(new_ir.tree.data(n), ForNode) and new_ir.tree.data(n).loop_var.startswith("i_d1_")]
    assert "i_d1_0" in names and "i_d1_1" in names
    assert not any("_0_" in nm for nm in names), names
```

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_split.py -v`
Expected: PASS. Byte-exact k0→k1; dense names.

- [ ] **Step 5: Full suite + numerics**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS. The render-equivalence tests (which sim) confirm semantics.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/split.py nkigym/src/nkigym/transforms/_normalize.py test/transforms/test_split.py test/transforms/test_normalize.py
git commit -m "Split routes through normalize_block: dense names, no trip-1, byte-exact k0->k1

Both Split flavors now edit ForNode topology + region tile-widths, then call
normalize_block to assign dense i_d{dim}_{N} names, drop trip-1, and recompute
region lo's as the element-stride affine over the dense loops. Splitting the
load d1 2048->(16,128) renders exactly k1 (single i_d1_0 loop, i_d1_0*128)."
```

---

## Task 4: Fuse routes through `normalize_block`; Split↔Fuse round-trip

**Files:**
- Modify: `nkigym/src/nkigym/transforms/fuse.py`
- Test: `test/transforms/test_fuse.py` (dense merge name, absorb→trip-1-drop, round-trip)

- [ ] **Step 1: Write failing tests**

Add to `test/transforms/test_fuse.py`:

```python
def test_fuse_merge_trips_dense_name():
    """Fuse two same-dim trip loops -> one loop named densely (i_d1_0), not i_d1_fused."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ForNode
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    d1 = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var == "i_d1_0" and ir.tree.data(n).extent == 16)
    ir = Split().apply(ir, SplitOption(target_nid=d1, factors=(2, 8)))
    """Now d1 has i_d1_0(2), i_d1_1(8); fuse them back."""
    outer = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var == "i_d1_0" and ir.tree.data(n).extent == 2)
    inner = next(c for c in ir.tree.children(outer) if isinstance(ir.tree.data(c), ForNode))
    fused = Fuse().apply(ir, FuseOption(target_nids=(outer, inner), target_axis=None))
    names = [fused.tree.data(n).loop_var for n in fused.tree.preorder() if isinstance(fused.tree.data(n), ForNode) and fused.tree.data(n).loop_var.startswith("i_d1")]
    assert "i_d1_0" in names and not any("fused" in nm for nm in names), names


def test_split_then_fuse_round_trip_byteexact():
    """Split the load d1 2048->(16,128) then fuse back == the original trip-1-free k0 load
    (loopless d1, full 2048 width). Byte-exact."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.codegen import render
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    canonical_render = render(ir)
    load = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad")
    split_ir = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    new_load = next(n for n in split_ir.tree.preorder() if isinstance(split_ir.tree.data(n), ISANode) and split_ir.tree.data(n).op_cls.__name__ == "NKILoad")
    d1_loop = split_ir.tree.parent(new_load)
    assert isinstance(split_ir.tree.data(d1_loop), ForNode) and split_ir.tree.data(d1_loop).extent == 16
    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(d1_loop, new_load), target_axis="d1"))
    assert render(fused_ir) == canonical_render, "Split->Fuse did not round-trip to canonical"
```

- [ ] **Step 2: Run to verify fail**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_fuse.py -v`
Expected: FAIL — merge produces `i_d1_fused`; round-trip diverges.

- [ ] **Step 3: Rewrite Fuse to use `normalize_block`**

Mirror Task 3: both `_do_outer_trip` (merge trips) and `_do_tensorize` (absorb into tile) edit the ForNode topology + region widths, then call `normalize_block`. Delete `_same_loop_axis`/`_stem`/`_fused_loop_var`/`_widen_region_axis`. `Fuse.analyze` finds adjacent same-dim ForNodes via `block.axis_map` + `_loopvar_to_dim` (or the `_dim_from_loopvar` stem helper from `_normalize`) instead of `_same_loop_axis` stem comparison. Read `fuse.py` fully first.

> **IMPLEMENTER:** absorb-into-tile (`_do_tensorize`) widens the tile and removes the loop → the loop becomes trip-1 only if the whole dim collapses; more precisely Fuse REMOVES the fused ForNode(s) and widens the tile, then `normalize_block` densifies the rest. For the k1→k0 reverse (fuse the single d1 loop into the tile), after removing the loop the d1 axis is loopless and the tile is 2048 — `normalize_block` confirms no trip-1 remains and the render matches canonical. STOP and report if round-trip render != canonical.

- [ ] **Step 4: Run fuse tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_fuse.py -v`
Expected: PASS (dense merge name, round-trip byte-exact).

- [ ] **Step 5: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS. Paste real summary.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/fuse.py test/transforms/test_fuse.py
git commit -m "Fuse routes through normalize_block: dense names, Split<->Fuse round-trip

Both Fuse flavors edit topology + tile width then normalize_block. Deletes
_same_loop_axis/_stem/_fused_loop_var/_widen_region_axis (analyze finds
adjacent same-dim ForNodes via axis_map). Split then Fuse round-trips
byte-exact to canonical."
```

---

## Task 5: Absorb/remove `_tile_region`; final consolidation

**Files:**
- Delete or reduce: `nkigym/src/nkigym/transforms/_tile_region.py`
- Modify: `test/transforms/test_tile_region.py` (remove or repoint)

- [ ] **Step 1: Find remaining `_tile_region` references**

Run: `grep -rn "_tile_region\|narrow_region_axis\|widen_region_axis\|retile_region" nkigym/src/ test/ | grep -v __pycache__`
If Split/Fuse (Tasks 3-4) still import width helpers from `_tile_region`, decide: keep `_tile_region` as a thin width-only helper, OR move those into `_normalize`. The element-space `lo` math now lives in `normalize_block`; `_tile_region`'s `narrow`/`widen` may be reduced to width arithmetic or deleted if `normalize_block` subsumes them.

- [ ] **Step 2: Consolidate**

If `_tile_region` is fully unused after Tasks 3-4, `git rm` it and delete `test_tile_region.py`. If width helpers are still used, keep only those + their tests. Make the call based on the grep; do not leave dead code (repo learning: remove unused functions).

- [ ] **Step 3: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS. Paste real summary.

- [ ] **Step 4: Re-verify the ladder semantics end-to-end**

Run: `source ~/venvs/kernel-env/bin/activate && python kernel_transforms.py 2>&1 | tail -3`
Expected: 15/15 still pass (this file is the oracle; unchanged by our IR work, but confirm nothing regressed the sim path).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Consolidate _tile_region into normalize_block; remove dead tensorize-fix code

The element-space lo math now lives in normalize_block; _tile_region's
narrow/widen are [removed|reduced to width-only]. No dead code."
```

---

## Task 6: Update superseded docs

**Files:**
- Modify: `.claude/rules/learnings.md` (supersede "canonical KEEPS trip-1")
- Modify: `docs/superpowers/specs/2026-06-01-tensorize-split-fuse-fix-design.md` (mark superseded)

- [ ] **Step 1: Supersede the trip-1 learning**

In `.claude/rules/learnings.md`, the "Canonical KEEPS trip-1 ForNodes" line — update it to: canonical (and all IR) is now trip-1-free; a trip-1 axis is loopless (tensorize_size only); `normalize_block` enforces this after every transform. Note the supersession date.

- [ ] **Step 2: Mark the partial tensorize-fix spec superseded**

In `docs/superpowers/specs/2026-06-01-tensorize-split-fuse-fix-design.md`, add a top note: "SUPERSEDED by 2026-06-01-unified-split-loopvar-normalization-design.md — the region-arithmetic fix shipped but left structural defects (double-suffix names, trip-1 wrappers); the unified-normalization reshape replaces it."

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/learnings.md docs/superpowers/specs/2026-06-01-tensorize-split-fuse-fix-design.md
git commit -m "Docs: supersede trip-1 learning + partial tensorize-fix spec"
```

---

## Self-Review (completed)

**Spec coverage:** no-trip-1-canonical → Task 1; `normalize_block` (drop trip-1 + dense rename + lo recompute) → Task 2 (+ extended in Task 3 for lo math); unified Split → Task 3; unified Fuse + delete ad-hoc helpers + round-trip → Task 4; absorb `_tile_region` → Task 5; supersede learning/spec → Task 6. Byte-exact k0→k1 + Split↔Fuse round-trip are the gates. Deferred (4 compute_at-needing transitions) stated in spec, not in plan scope.

**Placeholder scan:** every code step has full code EXCEPT the Split/Fuse `_do_*` rewrites (Tasks 3-4 Step 3), which give the algorithm + the exact `normalize_block` contract rather than full method bodies — flagged as "the crux, read the method first, STOP if byte-exact fails." This is the acknowledged soft spot; bounded by byte-exact tests + STOP guards. The lo-recompute coupling between Task 2 and Task 3 is called out explicitly with a "implement in Task 2, verify in Task 3" instruction.

**Type consistency:** `normalize_block(tree, block_nid)` signature consistent across Tasks 2-4. `_dim_from_loopvar`/`_loopvar_to_dim`/`_sub_region` defined in Task 2, reused in Task 4. `SplitOption(target_nid, factors, target_axis)` / `FuseOption(target_nids, target_axis)` match shipped option shapes. `i_d{dim}_{N}` naming consistent throughout.

**Known soft spot for execution:** the lo-element-stride recompute in `normalize_block` (`_rename_dense` must set region lo to `Σ dense_loopᵢ·strideᵢ`, not just rename) is the highest-risk piece — Task 3 Step 3 calls it out and says implement-in-Task-2-verify-in-Task-3. The byte-exact k0→k1 test is the forcing function.
