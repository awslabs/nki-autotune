# ComputeAt / ReverseComputeAt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `ComputeAt` (sink a producer) and `ReverseComputeAt` (lift a consumer) code-motion transforms — full-coverage block moves under a target loop — plus the codegen buffer compaction (Part C) that makes a moved kernel render correctly. Together these reproduce the `kernel_transforms.py` move ladder (k1→k2, k3→k4, k6→k7, k9→k10 sink; k11→k14 hoist) and the PSUM hoist.

**Architecture:** A move is purely structural — collapse the moved block's loops that the target's enclosing loops fully cover, keep uncovered loops as a private inner nest, splice the block under the target. Both directions share one `_compute_at_impl(ir, ..., is_reverse)`. After every move, two derived passes run on the tree: `place_buffers` (declaration descends to new LCA) and `compact_shapes` (buffer shape shrinks to the region bounding box); index rebase is a render-time projection. Buffer geometry is therefore always consistent with the tree after each atom.

**Tech Stack:** Python 3.12, networkx, numpy, pytest. Activate `~/venvs/kernel-env`. Spec: `docs/superpowers/specs/2026-05-29-compute-at-design.md` (Parts B + C).

---

## Ground truth (verified against current `dev_1`)

- **Shipped:** Part A region overlap (`interval.py`, region-gated `dependency.py`), `place_buffers` (`buffer_placement.py`), `Buffer.physical_shape()` + `physical_dtype()` (`tree.py`), `ReverseComputeAt` **legality only** (`reverse_compute_at.py` — `_check_legality` done, `apply` raises `NotImplementedError`), its 4 legality tests (`test_reverse_compute_at.py`), SSA frontend (`f_matmul` fixture is SSA, memset synthesized in `canonical_build`).
- **Canonical tree** (`build_canonical_ir()`), nids verified:
  ```
  0 root
  1 load_lhsT block  → 2 For(i_d0_0,16) → 3 For(i_d1_0,1) → 4 NKILoad
  5 load_rhs block   → 6 For(i_d0_0,16) → 7 For(i_d2_0,1) → 8 NKILoad
  9 memset block     → 10 For(i_d1_0,16) → 11 For(i_d2_0,1) → 12 NKIMemset
  13 matmul block    → 14 For(i_d0_0,16) → 15 For(i_d1_0,16) → 16 For(i_d2_0,4) → 17 NKIMatmul
  18 copy block      → 19 For(i_d1_0,16) → 20 For(i_d2_0,1) → 21 NKITensorCopy
  22 store block     → 23 For(i_d1_0,16) → 24 For(i_d2_0,1) → 25 NKIStore
  ```
  (nids are recomputed per build; tests must locate blocks by op class, not hardcode nids — use the `_block_for_op` helper pattern already in `test_reverse_compute_at.py`.)
- **Reused helpers** (`transforms/_tree_ops.py`): `_replace_in_parent_children(tree, parent, old_children, new_children)` (sibling-order-preserving splice), `_block_local_descendants(tree, block_nid)` (iter_value-substitution scope, stops at sub-blocks). `_find_enclosing_block` lives in both `split.py` and `fuse.py` (duplicated); reuse one or lift to `_tree_ops`.
- **Affine machinery** (`interval.py`): `to_affine`, `_affine_range(coeffs, loop_extents)`, `_interval_for_axis`, `regions_disjoint`. `expr.py`: `Var`, `Const`, `Mul`, `substitute`, `to_affine`, `from_affine`, `format_expr`.

## Sequencing rationale

**Part C (compaction) lands FIRST.** A move with no compaction renders over-allocated and wrong-indexed (spec lines 436-444: `sbuf_lhs_T` stays `(128,16,2048)` re-declared inside the loop). So every move's render+numerics test depends on compaction existing. Order: compaction → shared move impl → ReverseComputeAt → ComputeAt → PSUM-hoist E2E. The compute_at transforms run on the **current single-`last_writer` `Dependency`** unchanged (writers-list is a separate later spec; compute_at's only multi-writer pattern, memset+matmul, is non-disjoint).

---

## Task 1: `compact_shapes` — bounding-box shape on the tree

> **SHIPPED (`71fd2fb`) with a correction:** the code below shows a
> tree-global `_loop_extents(tree)`, which is BUGGY — canonical IR reuses
> loop-var names with different extents across subtrees (`i_d1_0` ∈ {1,16},
> `i_d2_0` ∈ {1,4}), so a flat map mis-sizes buffers (ballooned
> `sbuf_lhs_T` to `(2048,32768)`). The shipped version instead scopes
> extents PER-LEAF: `_regions_touching` returns `(leaf_nid, region)` pairs
> and `_leaf_loop_extents(tree, leaf_nid)` reads the leaf's own ForNode
> ancestors. A regression test
> (`test_compact_shapes_uses_per_leaf_extents_not_global`) pins it. The
> public API (`compact_shapes(tree)`, `rebased_region(region, buf, tree)`)
> is unchanged, so Tasks 2-7 below are unaffected.

**Files:**
- Create: `nkigym/src/nkigym/codegen/compact.py`
- Test: `test/codegen/test_compact.py` (NEW)

`compact_shapes(tree)` recomputes each `Buffer.shape` (2D logical) as the bounding box of its access regions, measured relative to the buffer's declaration (LCA) block. Loop vars bound at-or-above the declaration are *anchors* (don't contribute to shape); loop vars bound below contribute. Idempotent. For canonical IR every buffer is declared at/near root with no anchors → shape unchanged → render byte-identical.

- [ ] **Step 1: Write the failing test (canonical = no-op)**

Create `test/codegen/test_compact.py`:

```python
"""Tests for codegen.compact — buffer shape bounding-box + index rebase."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen.compact import compact_shapes


def test_compact_shapes_canonical_is_noop():
    """On canonical IR (buffers at/near root, no anchor loops above declaration),
    compact_shapes leaves every logical Buffer.shape unchanged."""
    ir = build_canonical_ir()
    before = {b.name: b.shape for b in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    after = {b.name: b.shape for b in ir.all_buffers().values()}
    assert before == after


def test_compact_shapes_idempotent():
    """compact_shapes applied twice equals once."""
    ir = build_canonical_ir()
    compact_shapes(ir.tree)
    once = {b.name: b.shape for b in ir.all_buffers().values()}
    compact_shapes(ir.tree)
    twice = {b.name: b.shape for b in ir.all_buffers().values()}
    assert once == twice
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/codegen/test_compact.py -v`
Expected: FAIL — `nkigym.codegen.compact` does not exist (ImportError).

- [ ] **Step 3: Implement `compact_shapes` + the bbox helper**

Create `nkigym/src/nkigym/codegen/compact.py`:

```python
"""Buffer geometry compaction over a transformed schedule tree.

Two entry points:

* :func:`compact_shapes` — recompute each :class:`Buffer`'s logical shape
  as the bounding box of its access regions within its declaration (LCA)
  block, and write it back on the tree. Idempotent; materialized like
  :func:`nkigym.ir.buffer_placement.place_buffers`.
* :func:`rebased_region` — a read-time projection that subtracts the
  declaration block's anchor loop vars from a region's ``lo``, so a
  compacted buffer is indexed within its single live instance. Never
  written back (tree regions stay global-frame for ``Dependency``).
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.expr import Const, Expr, Var, substitute, to_affine
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree, PARTITION_DIM


def compact_shapes(tree: KernelTree) -> None:
    """Recompute and write back every Buffer's logical shape (bbox over its LCA scope)."""
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        if not block.alloc_buffers:
            continue
        anchors = _anchor_loop_vars(tree, block_nid)
        new_bufs = tuple(_compact_one(tree, buf, anchors) for buf in block.alloc_buffers)
        tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=new_bufs)


def _anchor_loop_vars(tree: KernelTree, decl_block_nid: int) -> set[str]:
    """Loop vars bound at or above the declaration block (ForNode ancestors)."""
    out: set[str] = set()
    for anc in tree.ancestors(decl_block_nid):
        data = tree.data(anc)
        if isinstance(data, ForNode):
            out.add(data.loop_var)
    return out


def _compact_one(tree: KernelTree, buf: Buffer, anchors: set[str]) -> Buffer:
    """Return a copy of ``buf`` whose logical shape is the bbox of its access regions.

    shared_hbm buffers keep their declared shape (params/outputs are never
    resized). For sbuf/psum, each logical axis extent is the max of
    ``lo + width`` over the interior-loop box, with anchor loop vars zeroed.
    """
    if buf.location == "shared_hbm":
        return buf
    regions = _regions_touching(tree, buf.name)
    if not regions:
        return buf
    extents = _loop_extents(tree)
    n_axes = len(buf.shape)
    new_shape = list(buf.shape)
    for axis in range(n_axes):
        widest = 0
        for region in regions:
            if axis >= len(region.ranges):
                continue
            lo, width = region.ranges[axis]
            span = _axis_span(lo, width, axis, buf.location, anchors, extents)
            widest = max(widest, span)
        new_shape[axis] = widest
    return replace(buf, shape=tuple(new_shape))


def _axis_span(lo: Expr, width: Expr, axis: int, location: str, anchors: set[str], extents: dict[str, int]) -> int:
    """Max value of ``lo + width`` over the interior-loop box, anchors zeroed.

    Axis 0 of sbuf/psum carries a bare partition-tile index with width 128;
    its compacted extent is reported in element space (num_tiles * 128).
    """
    assert isinstance(width, Const), f"region width must be Const; got {width!r}"
    """Zero the anchor loop vars in lo; the remaining (interior) vars range over their extents."""
    zeroed = substitute(lo, {a: Const(value=0) for a in anchors})
    coeffs = to_affine(zeroed)
    hi = coeffs.get(None, 0)
    for var, coeff in coeffs.items():
        if var is None:
            continue
        trips = extents.get(var, 1)
        if coeff > 0:
            hi += coeff * (trips - 1)
    is_partition = axis == 0 and location in ("sbuf", "psum") and width.value == PARTITION_DIM
    if is_partition:
        """lo is a bare tile index; element-space extent = (max_tile_index + 1) * 128."""
        return (hi + 1) * PARTITION_DIM
    return hi + width.value


def _regions_touching(tree: KernelTree, tensor: str) -> list[BufferRegion]:
    """Every operand BufferRegion across all ISA leaves that names ``tensor``."""
    out: list[BufferRegion] = []
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append(region)
    return out


def _loop_extents(tree: KernelTree) -> dict[str, int]:
    """Map every ForNode loop_var in the tree to its extent."""
    out: dict[str, int] = {}
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ForNode):
            out[data.loop_var] = data.extent
    return out


def rebased_region(region: BufferRegion, buf: Buffer, tree: KernelTree) -> BufferRegion:
    """Subtract the declaration block's anchor loop vars from each axis ``lo``.

    Params (shared_hbm, declared at root → no anchors) project to
    themselves. For a compacted sbuf/psum buffer declared inside loops, the
    enclosing loop vars are subtracted so the index addresses the single
    resident instance (e.g. ``[i_d0_0, (i_d1_0)*128 : +128]`` → ``[0, 0:128]``).
    """
    decl_block_nid = _declaring_block(tree, buf.name)
    if decl_block_nid is None:
        return region
    anchors = _anchor_loop_vars(tree, decl_block_nid)
    if not anchors:
        return region
    subs = {a: Const(value=0) for a in anchors}
    new_ranges = tuple((substitute(lo, subs), width) for lo, width in region.ranges)
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


def _declaring_block(tree: KernelTree, tensor: str) -> int | None:
    """Return the block nid whose alloc_buffers declares ``tensor``, or None (a param)."""
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buf in block.alloc_buffers:
            if buf.name == tensor:
                return nid
    return None


__all__ = ["compact_shapes", "rebased_region"]
```

- [ ] **Step 4: Run the canonical tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/codegen/test_compact.py -v`
Expected: PASS (canonical buffers have no anchors above their LCA → shape unchanged; idempotent).

- [ ] **Step 5: Run the full suite (no regression yet — compact_shapes is not wired into render)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS (compact.py is standalone; nothing calls it yet).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/compact.py test/codegen/test_compact.py
git commit -m "Add codegen.compact: compact_shapes (bbox shape) + rebased_region (index projection)

Standalone, not yet wired. compact_shapes recomputes each Buffer's logical
shape as the access-region bbox over its LCA scope (idempotent, canonical
no-op); rebased_region subtracts declaration-block anchor loops from a
region at read time."
```

---

## Task 2: Wire `rebased_region` into the renderer

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`render_buffer_region`, `_emit_isa_call`, `_emit_subtree`/`_emit_block` thread `tree`)
- Test: `test/codegen/test_compact.py`

The renderer must rebase region indices via `rebased_region`. `compact_shapes` is NOT yet called in any transform (Task 5/6 wire it); this task only makes the renderer rebase-aware, which is a no-op on canonical IR (no anchors).

- [ ] **Step 1: Write the failing test**

Add to `test/codegen/test_compact.py`:

```python
def test_rebased_region_canonical_unchanged():
    """On canonical IR, rebased_region is identity (no buffer declared under loops)."""
    from nkigym.codegen.compact import rebased_region

    ir = build_canonical_ir()
    for nid in ir.tree.preorder():
        from nkigym.ir.tree import ISANode

        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                buf = ir.buffer(region.tensor)
                assert rebased_region(region, buf, ir.tree).ranges == region.ranges
```

- [ ] **Step 2: Run to verify it passes already (rebased_region exists from Task 1)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/codegen/test_compact.py::test_rebased_region_canonical_unchanged -v`
Expected: PASS (canonical buffers declared at root → no anchors → identity). This test pins the invariant before we wire it.

- [ ] **Step 3: Thread `tree` into `render_buffer_region` and rebase**

In `nkigym/src/nkigym/codegen/body.py`, `render_buffer_region` currently is `render_buffer_region(region, buf)`. Change its callers to pass the tree and rebase first. Edit `_emit_isa_call` (which calls `render_buffer_region`) — it currently receives `ir`; it has `ir.tree`. Change the call:

```python
def _emit_isa_call(node: ISANode, ir: KernelIR) -> str:
    """Emit ``nisa.<NAME>(slot=<region>, ..., kwarg=value, ...)`` for one ISA leaf."""
    op_cls = node.op_cls
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if slot in node.operand_bindings:
            region = node.operand_bindings[slot]
            buf = ir.buffer(region.tensor)
            parts.append(f"{slot}={render_buffer_region(rebased_region(region, buf, ir.tree), buf)}")
    for k, v in node.kwargs.items():
        parts.append(f"{k}={v!r}")
    return f"nisa.{op_cls.NAME}({', '.join(parts)})"
```

Add the import at the top of `body.py`:
```python
from nkigym.codegen.compact import rebased_region
```
(`render_buffer_region`'s own signature stays `(region, buf)` — it now receives an already-rebased region. No change to its body.)

WATCH FOR CIRCULAR IMPORT: `compact.py` imports from `nkigym.ir.tree` only (not `codegen`), and `body.py` importing `compact` is one-directional. Confirm `python -c "import nkigym.codegen.body"` succeeds.

- [ ] **Step 4: Run codegen + full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/codegen/ test/transforms/test_render_equivalence.py -q && python -m pytest test/ -q`
Expected: PASS — canonical/Split/Fuse renders are byte-identical (rebase is identity with no anchors).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_compact.py
git commit -m "Renderer rebases region indices via compact.rebased_region

_emit_isa_call now projects each operand region through rebased_region
before formatting. Identity on canonical IR (no anchor loops); becomes
load-bearing once compute_at declares buffers under loops."
```

---

## Task 3: Wire `_emit_alloc` shape through compacted tree state + envelope consistency

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`_emit_alloc` already uses `physical_shape()` — verify no change needed)
- Modify: `nkigym/src/nkigym/ir/ir.py` (`_render_envelope_md` already uses `physical_shape()` — verify)
- Test: `test/codegen/test_compact.py`

`compact_shapes` writes the compacted *logical* shape onto `Buffer.shape`. `_emit_alloc` and `Buffer.label()` already read `physical_shape()` (which derives from `shape`), so once a transform calls `compact_shapes` the alloc line and PNG label follow automatically — **no emitter change needed for shape**. This task is a verification + a direct test that a manually-compacted buffer renders compacted.

- [ ] **Step 1: Write the test (manually compact a buffer, assert alloc line shrinks)**

Add to `test/codegen/test_compact.py`:

```python
def test_emit_alloc_follows_compacted_shape():
    """After compact_shapes writes a smaller logical shape, _emit_alloc emits it."""
    from dataclasses import replace

    from nkigym.codegen.body import _emit_alloc
    from nkigym.ir.tree import Buffer

    full = Buffer(name="sbuf_x", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    compacted = replace(full, shape=(128, 128))
    assert _emit_alloc(full) == "sbuf_x = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)"
    assert _emit_alloc(compacted) == "sbuf_x = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)"
```

- [ ] **Step 2: Run it**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/codegen/test_compact.py::test_emit_alloc_follows_compacted_shape -v`
Expected: PASS (no code change — `_emit_alloc` already routes through `physical_shape()`, which expands the logical `(128,128)` → `(128,1,128)`). If it FAILS, `_emit_alloc` was not reading `physical_shape()`; fix it to do so.

- [ ] **Step 3: Run full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test/codegen/test_compact.py
git commit -m "Test: _emit_alloc emits compacted shape via physical_shape (no emitter change)

Confirms the materialize-on-tree design: compact_shapes writes logical
shape, physical_shape expands it, _emit_alloc/Buffer.label follow with no
signature change."
```

---

## Task 4: Shared move mechanics — `_compute_at_impl`

**Files:**
- Create: `nkigym/src/nkigym/transforms/_code_motion.py`
- Test: `test/transforms/test_code_motion.py` (NEW)

The structural core both directions share: given a block to move, a target loop, and a direction flag, collapse the moved block's same-axis loops the target's enclosing loops fully cover, keep uncovered loops as a private inner nest, and splice the block under the target at the insertion index. This task builds + unit-tests the mechanics in isolation (legality is the caller's job, Tasks 5/6).

- [ ] **Step 1: Write the failing test (sink load_lhsT under matmul's d1 loop — k1→k2)**

Create `test/transforms/test_code_motion.py`:

```python
"""Unit tests for the shared _compute_at_impl move mechanics."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import copy

from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms._code_motion import _compute_at_impl


def _block_for_op(ir, op_name):
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(op_name)


def _loops_under(ir, block_nid):
    return [ir.tree.data(d).loop_var for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ForNode)]


def test_sink_load_lhsT_under_matmul_d1_collapses_both_loops():
    """ComputeAt-direction: sink load_lhsT (loops d0,d1) under matmul's d1 loop.
    The load's d0,d1 are fully covered by the matmul's enclosing d0,d1 → both
    collapse; the load block becomes loopless, spliced under matmul's d1 loop."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")  # first NKILoad = lhs_T
    matmul = _block_for_op(ir, "NKIMatmul")
    """matmul's d1 loop is the 2nd ForNode under the matmul block (d0 then d1)."""
    matmul_loops = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode)]
    d1_loop = matmul_loops[1]  # i_d1_0
    new_ir = copy.deepcopy(ir)
    _compute_at_impl(new_ir, block_nid=load, target_loop_nid=d1_loop, index=-2, is_reverse=False)
    """After: load block has no private loops (d0,d1 both covered)."""
    assert _loops_under(new_ir, load) == []
    """And the load block is now a descendant of the matmul's d1 loop."""
    assert load in new_ir.tree.descendants(d1_loop)
```

NOTE: nids are stable within a single `build_canonical_ir()` call (networkx integer ids), so capturing `load`/`matmul`/`d1_loop` before the move and querying them after is valid — `_compute_at_impl` mutates topology but does not renumber surviving nodes.

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_code_motion.py -v`
Expected: FAIL — `_code_motion` does not exist.

- [ ] **Step 3: Implement `_compute_at_impl`**

Create `nkigym/src/nkigym/transforms/_code_motion.py`:

```python
"""Shared structural move mechanics for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop: same-axis loops the
target's enclosing nest fully covers collapse (their iter_var binds to the
target's loop var instead); uncovered loops stay as the moved block's
private inner nest; the block is spliced under the target at ``index``.
Direction (sink producer vs lift consumer) only affects the caller's
legality check and which neighbor bounds the insertion gap — the
structural move is identical, so both share this function.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.expr import Var, substitute
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children


def _compute_at_impl(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Move ``block_nid`` under ``target_loop_nid`` in place (full-coverage only).

    Mutates ``ir.tree``. Caller has already checked legality. ``index`` is
    the insertion position among the target loop's body children (TVM
    convention: -1 last legal, -2 earliest legal, >=0 explicit). ``is_reverse``
    is accepted for symmetry but does not change the structural steps.
    """
    tree = ir.tree
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)

    target_enclosing = _enclosing_loop_vars_by_axis(tree, target_loop_nid)
    moved_loops = _block_loops_by_axis(tree, block_nid)

    covered_axis_to_target_var: dict[str, str] = {}
    for axis, (moved_loop_nid, moved_var) in moved_loops.items():
        if axis in target_enclosing:
            covered_axis_to_target_var[moved_var] = target_enclosing[axis]

    _collapse_covered_loops(tree, block_nid, covered_axis_to_target_var)
    _rebind_block(tree, block_nid, covered_axis_to_target_var)
    _splice_under_target(tree, block_nid, target_loop_nid, index)


def _enclosing_loop_vars_by_axis(tree: KernelTree, target_loop_nid: int) -> dict[str, str]:
    """Map concrete axis (from the enclosing block's iter_values) → loop_var, for
    every ForNode at or above ``target_loop_nid`` within its block."""
    block_nid = _enclosing_block(tree, target_loop_nid)
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    var_to_axis = _loop_var_to_axis(block)
    out: dict[str, str] = {}
    chain = [target_loop_nid, *tree.ancestors(target_loop_nid)]
    for nid in chain:
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in var_to_axis:
            out[var_to_axis[data.loop_var]] = data.loop_var
    return out


def _block_loops_by_axis(tree: KernelTree, block_nid: int) -> dict[str, tuple[int, str]]:
    """For the moved block, map concrete axis → (loop_nid, loop_var) for each ForNode it owns."""
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    var_to_axis = _loop_var_to_axis(block)
    out: dict[str, tuple[int, str]] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in var_to_axis:
            out[var_to_axis[data.loop_var]] = (nid, data.loop_var)
    return out


def _loop_var_to_axis(block: BlockNode) -> dict[str, str]:
    """Invert the block's iter_values: bare-Var loop_var name → iter_var axis."""
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        if isinstance(value, Var):
            out[value.name] = iv.axis
    return out


def _collapse_covered_loops(tree: KernelTree, block_nid: int, covered_vars: dict[str, str]) -> None:
    """Remove the moved block's ForNodes whose loop_var is in ``covered_vars``,
    re-linking each removed loop's children to its parent (preserving order)."""
    to_remove = [
        nid
        for nid in _block_local_descendants(tree, block_nid)
        if isinstance(tree.data(nid), ForNode) and tree.data(nid).loop_var in covered_vars
    ]
    for nid in to_remove:
        parent = tree.parent(nid)
        children = tree.children(nid)
        _replace_in_parent_children(tree, parent, [nid], children)
        tree.graph.remove_node(nid)


def _rebind_block(tree: KernelTree, block_nid: int, covered_vars: dict[str, str]) -> None:
    """Substitute each covered loop_var with the target's loop_var across the
    block's iter_values / reads / writes and its ISA leaf operand_bindings."""
    subs = {old: Var(name=new) for old, new in covered_vars.items()}
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    new_block = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=tuple(substitute(v, subs) for v in block.iter_values),
        reads=tuple(_sub_region(r, subs) for r in block.reads),
        writes=tuple(_sub_region(w, subs) for w in block.writes),
        alloc_buffers=block.alloc_buffers,
        annotations=dict(block.annotations),
    )
    tree.graph.nodes[block_nid]["data"] = new_block
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ISANode):
            new_bindings = {slot: _sub_region(r, subs) for slot, r in data.operand_bindings.items()}
            tree.graph.nodes[nid]["data"] = ISANode(
                op_cls=data.op_cls, operand_bindings=new_bindings, kwargs=dict(data.kwargs)
            )


def _sub_region(region: BufferRegion, subs: dict) -> BufferRegion:
    """Apply Var substitutions to both lo and width of every range."""
    return BufferRegion(
        tensor=region.tensor,
        ranges=tuple((substitute(lo, subs), substitute(width, subs)) for lo, width in region.ranges),
    )


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach the moved block from its current parent and insert it among the
    target loop's body children at ``index`` (TVM convention: -1 last, -2 first)."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None
    siblings = tree.children(old_parent)
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    tree.graph.remove_edge(old_parent, block_nid)

    target_children = tree.children(target_loop_nid)
    if index == -1:
        pos = len(target_children)
    elif index == -2:
        pos = 0
    else:
        pos = index
    new_order = target_children[:pos] + [block_nid] + target_children[pos:]
    for child in target_children:
        tree.graph.remove_edge(target_loop_nid, child)
    for child in new_order:
        tree.graph.add_edge(target_loop_nid, child)


def _enclosing_block(tree: KernelTree, nid: int) -> int:
    """Walk ancestors of ``nid`` until a BlockNode; return its nid."""
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            return anc
    raise ValueError(f"no enclosing BlockNode for {nid}")


__all__ = ["_compute_at_impl"]
```

- [ ] **Step 4: Run the move test**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_code_motion.py -v`
Expected: PASS — load block loopless and under the matmul's d1 loop.

- [ ] **Step 5: Run full suite (no regression — _code_motion not yet called by any transform)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py test/transforms/test_code_motion.py
git commit -m "Add shared _compute_at_impl move mechanics (collapse covered loops + splice)

Structural core for both compute_at directions: collapse the moved block's
same-axis loops the target's enclosing nest covers (rebind iter_vars to the
target's loop vars), keep uncovered loops, splice the block under the target
at index. Not yet wired into a transform."
```

---

## Task 5: `ReverseComputeAt.apply` — fill the stub

> **SHIPPED (`27e47ec`) with two notes:**
> 1. **Renderer gap fixed (out of original plan scope):** `_emit_subtree`
>    in `body.py` only handled `ForNode`/`ISANode` and raised on a
>    `BlockNode` nested under a `ForNode` — the exact post-move shape. The
>    implementer added the symmetric `BlockNode`→`_emit_block` delegation
>    (matches the learnings' "renderer generic over BlockNode/ForNode/
>    ISANode interleavings"). Genuine cross-task gap, correctly filled.
> 2. **The render test lifts under the WRONG loop:** it used
>    `_first_for_in_block(matmul)` = the OUTERMOST loop `i_d0_0` (= K), so
>    the drain renders inside K-accumulation (16× redundant, last-drain-
>    wins — numerically correct but NOT the PSUM hoist). `apply`/`analyze`
>    are correct (they moved where told). **Task 7's PSUM-hoist E2E MUST
>    target the (M,N) loops, not the outermost loop** — see Task 7 note.

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reverse_compute_at.py` (`apply`, add `analyze`)
- Test: `test/transforms/test_reverse_compute_at.py` (add apply-mechanics + render tests)

`reverse_compute_at.py` has shipped legality (`_check_legality`); only `apply` is `NotImplementedError`. Wire it to `_compute_at_impl` + the two derived passes, and add `analyze`.

- [ ] **Step 1: Write the failing test (lift tensor_copy under matmul's M loop, render + sim)**

Add to `test/transforms/test_reverse_compute_at.py`:

```python
def test_reverse_compute_at_lifts_tensor_copy_and_renders(tmp_path):
    """Lift tensor_copy under the matmul's M-loop, then render + fp32-sim the result."""
    import importlib.util

    import numpy as np

    from test.transforms._fixtures import INPUT_SPECS
    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    matmul = _block_for_op(ir, "NKIMatmul")
    m_loop = _first_for_in_block(ir, matmul)
    new_ir = ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=m_loop))

    """tensor_copy block is now under the matmul's M loop."""
    assert tc in new_ir.tree.descendants(m_loop)

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

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_reverse_compute_at.py::test_reverse_compute_at_lifts_tensor_copy_and_renders -v`
Expected: FAIL — `NotImplementedError` in `apply`.

- [ ] **Step 3: Implement `apply` + `analyze`**

In `reverse_compute_at.py`, replace the `apply` body and add `analyze`. Add imports:
```python
import copy
from nkigym.ir.dependency import Dependency
from nkigym.ir.buffer_placement import place_buffers
from nkigym.codegen.compact import compact_shapes
from nkigym.transforms._code_motion import _compute_at_impl
from nkigym.ir.tree import BlockNode, ISANode
```

```python
    def apply(self, ir: KernelIR, option: ReverseComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, lift the block, re-derive geometry, return new IR."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _compute_at_impl(
            new_ir,
            block_nid=option.block_nid,
            target_loop_nid=option.target_loop_nid,
            index=option.index,
            is_reverse=True,
        )
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[ReverseComputeAtOption]:
        """Enumerate (consumer block, producer-enclosing target loop) pairs passing legality."""
        options: list[ReverseComputeAtOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                opt = ReverseComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid)
                try:
                    self._check_legality(ir, opt)
                except TransformLegalityError:
                    continue
                options.append(opt)
        return options
```

- [ ] **Step 4: Run the lift+render test + the existing legality tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_reverse_compute_at.py -v`
Expected: PASS (4 shipped legality tests + the new render test).

- [ ] **Step 5: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/reverse_compute_at.py test/transforms/test_reverse_compute_at.py
git commit -m "Implement ReverseComputeAt.apply via shared _compute_at_impl + analyze

apply: deep-copy, move via _compute_at_impl, then place_buffers +
compact_shapes + rebuild Dependency. analyze enumerates legal
(consumer-block, target-loop) pairs. Lifts tensor_copy under the matmul
M-loop and renders to numerically-correct NKI."
```

---

## Task 6: `ComputeAt` — new transform (sink a producer)

**Files:**
- Create: `nkigym/src/nkigym/transforms/compute_at.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py` (export)
- Test: `test/transforms/test_compute_at.py` (NEW)

`ComputeAt` is the mirror: sink a producer under a consumer's loop. Same `_compute_at_impl`; legality is condition 5a (consumers under-target-or-later) + the output-block guard (condition 4). Reuse the structure of `reverse_compute_at.py`'s legality, flipping producers↔consumers and adding the output guard.

- [ ] **Step 1: Write the failing test (sink load_lhsT under matmul d1 — k1→k2 — render + sim)**

Create `test/transforms/test_compute_at.py`:

```python
"""Tests for nkigym.transforms.ComputeAt (producer-sink)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir, INPUT_SPECS

import importlib.util

import numpy as np
import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import ComputeAt, ComputeAtOption, TransformLegalityError


def _block_for_op(ir, op_name):
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(op_name)


def _matmul_d1_loop(ir, matmul):
    loops = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode)]
    return loops[1]  # d0, d1, ...


def test_compute_at_sinks_load_and_renders(tmp_path):
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")  # lhs_T
    matmul = _block_for_op(ir, "NKIMatmul")
    d1 = _matmul_d1_loop(ir, matmul)
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=d1))
    assert load in new_ir.tree.descendants(d1)
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


def test_compute_at_rejects_sinking_the_store():
    """Condition 4: the kernel's output store has no consumer; sinking it is illegal."""
    ir = build_canonical_ir()
    store = _block_for_op(ir, "NKIStore")
    matmul = _block_for_op(ir, "NKIMatmul")
    d1 = _matmul_d1_loop(ir, matmul)
    with pytest.raises(TransformLegalityError, match="output|return"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=store, target_loop_nid=d1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_compute_at.py -v`
Expected: FAIL — `ComputeAt` not importable.

- [ ] **Step 3: Implement `ComputeAt`**

Create `nkigym/src/nkigym/transforms/compute_at.py`:

```python
"""``ComputeAt`` transform — sink a producer block under a consumer's loop.

Mirror of :class:`ReverseComputeAt`. See
``nkigym/src/nkigym/transforms/compute_at_legality.md`` for the six
legality conditions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _compute_at_impl
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ComputeAtOption(TransformOption):
    """Sink producer ``block_nid`` under ``target_loop_nid`` (a loop of a consumer)."""

    block_nid: int
    target_loop_nid: int
    index: int = -1


class ComputeAt(Transform):
    """Sink a producer block under a consumer's loop."""

    def apply(self, ir: KernelIR, option: ComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, sink the block, re-derive geometry, return new IR."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _compute_at_impl(
            new_ir,
            block_nid=option.block_nid,
            target_loop_nid=option.target_loop_nid,
            index=option.index,
            is_reverse=False,
        )
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[ComputeAtOption]:
        """Enumerate (producer block, consumer-enclosing target loop) pairs passing legality."""
        options: list[ComputeAtOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                opt = ComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid)
                try:
                    self._check_legality(ir, opt)
                except TransformLegalityError:
                    continue
                options.append(opt)
        return options

    def _check_legality(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Six conditions; mirror of ReverseComputeAt with the output guard (cond 4)."""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"ComputeAt.target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError("ComputeAt requires target_loop_nid to be a ForNode")
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"ComputeAt.block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError("ComputeAt: target_loop_nid is a descendant of the moved block")
        self._check_not_output(ir, option)
        self._check_consumers_visited(ir, option)

    def _check_not_output(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Condition 4: a producer whose write is the kernel's return tensor has no consumer to sink under."""
        block = ir.tree.data(option.block_nid)
        for region in getattr(block, "writes", ()):
            if region.tensor == ir.return_name:
                raise TransformLegalityError(
                    f"ComputeAt: block {option.block_nid} writes the kernel output {ir.return_name!r}; "
                    f"a final-store producer has no consumer to sink under"
                )

    def _check_consumers_visited(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Condition 5a: every consumer of the moved producer is a descendant of the target loop,
        OR lives in a root-sibling whose pre-order index is AFTER the target's."""
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for consumer in ir.dependency.consumers(option.block_nid):
            if consumer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(consumer):
                continue
            consumer_root = self._root_sibling_of(ir, consumer)
            if consumer_root not in root_order:
                raise TransformLegalityError(f"ComputeAt: consumer block {consumer} is not under a root-sibling")
            if root_order.index(consumer_root) > target_index:
                continue
            raise TransformLegalityError(
                f"ComputeAt: consumer block {consumer} runs before the target loop "
                f"(root index {root_order.index(consumer_root)} <= target {target_index}); "
                f"not all consumers are visited after the sunk producer"
            )

    @staticmethod
    def _root_sibling_of(ir: KernelIR, nid: int) -> int:
        """Return the direct child of ``tree.root`` that is ``nid`` or an ancestor of it."""
        if nid in ir.tree.children(ir.tree.root):
            return nid
        for anc in ir.tree.ancestors(nid):
            if anc in ir.tree.children(ir.tree.root):
                return anc
        raise TransformLegalityError(f"node {nid} has no root-sibling ancestor")


__all__ = ["ComputeAt", "ComputeAtOption"]
```

- [ ] **Step 4: Export from `__init__.py`**

In `nkigym/src/nkigym/transforms/__init__.py`, add the import and `__all__` entries (alphabetical, matching existing style):
```python
from nkigym.transforms.compute_at import ComputeAt, ComputeAtOption
```
and add `"ComputeAt"`, `"ComputeAtOption"` to `__all__`.

- [ ] **Step 5: Run the ComputeAt tests + full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_compute_at.py -v && python -m pytest test/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/compute_at.py nkigym/src/nkigym/transforms/__init__.py test/transforms/test_compute_at.py
git commit -m "Add ComputeAt transform (sink producer) mirroring ReverseComputeAt

Shares _compute_at_impl; legality is condition 5a (consumers under-target
-or-later) plus the output-block guard (condition 4). Sinks load_lhsT under
the matmul d1-loop (k1->k2), renders to numerically-correct NKI."
```

---

## Task 6.5: Full-coverage legality check (spec Part B step 2)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` (add `check_full_coverage`)
- Modify: `nkigym/src/nkigym/transforms/compute_at.py` and `reverse_compute_at.py` (call it in `_check_legality`)
- Test: `test/transforms/test_compute_at.py`

The spec requires a move be legal ONLY when, for each axis the moved block shares with the target's enclosing loops, the target's loop-extent product equals the moved block's loop-extent product on that axis (full cover). `_compute_at_impl` collapses matching axes but does not *check* this; a partial-coverage move would silently mis-collapse. Add an explicit guard so partial coverage raises with Split-first guidance (matching the spec's composition note).

- [ ] **Step 1: Write the failing test (a partial-coverage move is rejected)**

Add to `test/transforms/test_compute_at.py`. A clean partial-coverage case: Split the matmul's d1 loop into (2, 8) so the matmul's enclosing d1 coverage is 8, then try to sink a block whose d1 loop is still trip-16 — coverage 8 ≠ 16 → reject.

```python
def test_compute_at_rejects_partial_coverage():
    """A move whose covered axis extents don't match the target's enclosing extents
    is rejected with Split-first guidance."""
    from nkigym.ir.tree import ForNode
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    matmul = _block_for_op(ir, "NKIMatmul")
    """Split the matmul's d1 loop (trip 16) into (2, 8) → enclosing d1 coverage becomes 8."""
    d1 = _matmul_d1_loop(ir, matmul)
    ir = Split().apply(ir, SplitOption(target_nid=d1, factors=(2, 8), target_axis=None))

    load = _block_for_op(ir, "NKILoad")  # lhs_T, still has a trip-16 d1 loop
    matmul = _block_for_op(ir, "NKIMatmul")
    """Inner d1 loop of the now-split matmul nest (trip 8)."""
    inner_d1 = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).extent == 8][0]
    with pytest.raises(TransformLegalityError, match="coverage|Split"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=inner_d1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_compute_at.py::test_compute_at_rejects_partial_coverage -v`
Expected: FAIL — no coverage check yet, so the move proceeds (or raises a different error).

- [ ] **Step 3: Add `check_full_coverage` to `_code_motion.py`**

Add this public helper (and export it):

```python
def check_full_coverage(tree: KernelTree, block_nid: int, target_loop_nid: int) -> str | None:
    """Return an error message if the move is NOT full-coverage, else None.

    For each concrete axis the moved block shares with the target's enclosing
    loops, the product of the target's enclosing loop extents on that axis must
    equal the product of the moved block's loop extents on that axis. Partial
    coverage means the collapse would drop iterations — reject (Split first).
    """
    target_enclosing = _enclosing_loop_vars_by_axis(tree, target_loop_nid)
    moved = _block_loops_by_axis(tree, block_nid)
    target_block = tree.data(_enclosing_block(tree, target_loop_nid))
    moved_block = tree.data(block_nid)
    assert isinstance(target_block, BlockNode) and isinstance(moved_block, BlockNode)
    msg: str | None = None
    for axis in set(target_enclosing) & set(moved):
        moved_extent = _axis_extent_product(tree, block_nid, axis)
        target_extent = _target_axis_extent_product(tree, target_loop_nid, axis)
        if moved_extent != target_extent:
            msg = (
                f"ComputeAt/ReverseComputeAt: partial coverage on axis {axis!r} "
                f"(moved extent {moved_extent} != target enclosing extent {target_extent}); "
                f"Split the moved block or the target loop to full coverage first"
            )
            break
    return msg


def _axis_extent_product(tree: KernelTree, block_nid: int, axis: str) -> int:
    """Product of the moved block's ForNode extents binding ``axis``."""
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    var_to_axis = _loop_var_to_axis(block)
    product = 1
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and var_to_axis.get(data.loop_var) == axis:
            product *= data.extent
    return product


def _target_axis_extent_product(tree: KernelTree, target_loop_nid: int, axis: str) -> int:
    """Product of the target's enclosing ForNode extents (at or above target) binding ``axis``."""
    block = tree.data(_enclosing_block(tree, target_loop_nid))
    assert isinstance(block, BlockNode)
    var_to_axis = _loop_var_to_axis(block)
    product = 1
    for nid in [target_loop_nid, *tree.ancestors(target_loop_nid)]:
        data = tree.data(nid)
        if isinstance(data, ForNode) and var_to_axis.get(data.loop_var) == axis:
            product *= data.extent
    return product
```
Add `"check_full_coverage"` to `_code_motion.py`'s `__all__`.

- [ ] **Step 4: Call it in both transforms' `_check_legality`**

In `compute_at.py` `_check_legality`, after the descendant check and before `_check_not_output`:
```python
        coverage_error = check_full_coverage(ir.tree, option.block_nid, option.target_loop_nid)
        if coverage_error is not None:
            raise TransformLegalityError(coverage_error)
```
(import `check_full_coverage` from `nkigym.transforms._code_motion`.) Add the identical call + import in `reverse_compute_at.py` `_check_legality`, after its descendant check.

- [ ] **Step 5: Run the partial-coverage test + both transforms' suites + full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_compute_at.py test/transforms/test_reverse_compute_at.py -v && python -m pytest test/ -q`
Expected: PASS (partial-coverage rejected; the canonical full-cover moves from Tasks 5-6 still pass).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py nkigym/src/nkigym/transforms/compute_at.py nkigym/src/nkigym/transforms/reverse_compute_at.py test/transforms/test_compute_at.py
git commit -m "Add full-coverage legality check to compute_at moves (spec Part B step 2)

check_full_coverage rejects a move whose covered-axis extents don't match
the target's enclosing extents (would drop iterations), with Split-first
guidance. Both directions call it in _check_legality."
```

---

## Task 7: PSUM-hoist end-to-end + example wiring

**Files:**
- Test: `test/transforms/test_compute_at.py` (add the trio E2E)
- Modify: `examples/matmul_lhsT_rhs.py` (add ComputeAt + ReverseComputeAt to the MDP transform list)

The motivating composite: lift tensor_copy + sink memset under the matmul's `(M,N)`, then assert `psum_prod` descended into the matmul block's `alloc_buffers` AND compacted to one tile, with correct numerics.

- [ ] **Step 1: Write the PSUM-hoist E2E test**

Add to `test/transforms/test_compute_at.py`:

```python
def test_psum_hoist_descends_and_compacts(tmp_path):
    """Lift tensor_copy and sink memset under the matmul's (M,N); assert psum_prod
    descends into the matmul block and compacts, and the kernel still computes correctly."""
    from nkigym.ir.tree import BlockNode
    from nkigym.transforms import ReverseComputeAt, ReverseComputeAtOption

    ir = build_canonical_ir()
    matmul = _block_for_op(ir, "NKIMatmul")
    m_loop = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode)][0]

    tc = _block_for_op(ir, "NKITensorCopy")
    ir = ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=m_loop))

    memset = _block_for_op(ir, "NKIMemset")
    matmul = _block_for_op(ir, "NKIMatmul")
    m_loop = [d for d in ir.tree.preorder(matmul) if isinstance(ir.tree.data(d), ForNode)][0]
    ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=memset, target_loop_nid=m_loop))

    """psum_prod now declared on a block at or below the matmul's M loop (descended from root)."""
    decl = None
    for nid in ir.tree.blocks():
        blk = ir.tree.data(nid)
        if any(b.name == "psum_prod" for b in blk.alloc_buffers):
            decl = nid
    assert decl is not None and decl != ir.tree.root, "psum_prod should descend from root"

    """Numerics still correct."""
    import importlib.util

    import numpy as np

    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32

    src = render(ir)
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

NOTE: if the PSUM-hoist requires the matmul's M to be the *outermost* shared loop and the tensor_copy's M-loop to fully cover it (full-coverage rule), this test uses the canonical matmul whose M-loop is trip-16 and tensor_copy's M-loop is also trip-16 → full cover, no Split needed. If a coverage mismatch surfaces (e.g. memset has only M,N while matmul has K,M,N), the memset's covered axes are M,N (K is the matmul's private inner loop, uncovered for memset which has no K) — confirm the move keeps the matmul's K loop as its private nest. If legality rejects, report which condition and we add a Split-first step.

- [ ] **Step 2: Run it**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/transforms/test_compute_at.py::test_psum_hoist_descends_and_compacts -v`
Expected: PASS. If a full-coverage or legality error surfaces, STOP and report — the spec's composition note (Split-first) may be needed; do not force.

- [ ] **Step 3: Add the transforms to the example's MDP list**

In `examples/matmul_lhsT_rhs.py`, the env is `KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder()])`. Add the two new transforms and their import:
```python
from nkigym.transforms import ComputeAt, Fuse, Reorder, ReverseComputeAt, Split
...
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
```

- [ ] **Step 4: Run the example end-to-end**

Run: `source ~/venvs/kernel-env/bin/activate && PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py 2>&1 | tail -5`
Expected: every rollout step prints `[numerics] PASS`. Random rollouts now include ComputeAt/ReverseComputeAt moves; all must stay numerically correct. (If a random move sequence hits an unhandled coverage case and raises, that's a real finding — report it.)

- [ ] **Step 5: Full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add test/transforms/test_compute_at.py examples/matmul_lhsT_rhs.py
git commit -m "PSUM-hoist E2E test + wire ComputeAt/ReverseComputeAt into the example MDP

Lift tensor_copy + sink memset under the matmul (M,N); assert psum_prod
descends from root into the matmul block and compacts, numerics hold.
Add both transforms to the random-rollout transform list."
```

---

## Task 8: Refresh the legality doc

**Files:**
- Modify: `nkigym/src/nkigym/transforms/compute_at_legality.md`

The spec (line 282-287) notes the legality doc's "What is a block?" section still describes the removed `init` field / bundled memset. Update to decomposed canonical (memset is a sibling block, no `init`, `tree.root` is the root BlockNode). The six conditions are unchanged.

- [ ] **Step 1: Read the current "What is a block?" section**

Run: `grep -n "init\|memset\|RootNode\|root" nkigym/src/nkigym/transforms/compute_at_legality.md | head -20`
Read the section that mentions `init` / bundled memset.

- [ ] **Step 2: Update the prose**

Edit the "What is a block?" section: remove references to a `BlockNode.init` field and to the memset being bundled into the matmul. State that the memset is a synthesized sibling block before the matmul (decomposed canonical), and `tree.root` is the root BlockNode holding kernel-lifetime buffers. Keep the six conditions and their examples unchanged (they're still accurate).

- [ ] **Step 3: Verify no code references broke (doc-only change)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS (markdown only).

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/transforms/compute_at_legality.md
git commit -m "Refresh compute_at legality doc: decomposed canonical (no init field)

memset is a synthesized sibling block before the matmul; tree.root is the
root BlockNode. The six legality conditions are unchanged."
```

---

## Self-Review (completed)

**Spec coverage:** Part C compaction → Tasks 1-3; shared move mechanics → Task 4; ReverseComputeAt.apply → Task 5; ComputeAt + 6 conditions → Task 6; PSUM-hoist E2E + example → Task 7; legality-doc refresh → Task 8. The four sink cases (k1→k2 etc.) are all `ComputeAt` under different targets — Task 6 covers k1→k2 directly; k3/k6/k9 are the same mechanism (analyze enumerates them, the E2E + random rollouts exercise them). The index-rebase / param-invariance spec tests fold into Tasks 2-3.

**Placeholder scan:** every code step shows full code; commands have expected output. The one conditional ("if coverage mismatch surfaces, report") is a genuine STOP-and-escalate instruction, not a placeholder.

**Type consistency:** `_compute_at_impl(ir, block_nid, target_loop_nid, index, is_reverse)` signature consistent across Tasks 4-6; `compact_shapes(tree)` / `rebased_region(region, buf, tree)` consistent across Tasks 1-3, 5-6; `ComputeAtOption`/`ReverseComputeAtOption` fields (`block_nid`, `target_loop_nid`, `index`) match the spec and the shipped stub.

**Full-coverage rule (spec Part B step 2):** `_compute_at_impl` (Task 4) collapses whatever axes match but does not itself *check* coverage; the explicit reject lives in **Task 6.5** (`check_full_coverage`, called by both transforms' `_check_legality`). Tasks 5-6 use canonical full-cover cases (trip-16 M-loops match) so they pass before 6.5 lands; 6.5 then guards partial coverage with Split-first guidance. Task 7's random rollouts may legitimately hit partial-coverage moves — with 6.5 in place those are cleanly rejected (legal-action enumeration in `analyze` filters them via `_check_legality`), not crashes.

**Sequencing note:** Task 6.5 is placed after Task 6 (both transforms exist to call the shared check) and before Task 7 (so the example's `analyze`-driven rollouts never propose a partial-coverage move). An implementer may fold 6.5's check into Task 6 if preferred — but keep it a distinct commit.
