# BufferCompaction Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the anonymous `place_buffers` + `compact_shapes` tail (and the render-time `rebased_region`) into a first-class per-buffer `BufferCompaction` transform materialized in the IR; decouple `CodeMotion`/`RFactor` to structural-only; drive the ladder to reproduce `manual_transforms.py` k0…k32 byte-exact.

**Architecture:** `BufferCompaction(tensor)` is a per-buffer atomic move — place the decl at its LCA, shrink its logical shape to the access bbox, and rewrite its access regions into the single-instance local frame, all **materialized on the tree**. The renderer then emits regions verbatim (no `rebased_region`) and anchors decls to their owning block (not the leaf-LCA). `CodeMotion`/`RFactor` do the structural edit + `Dependency` rebuild only.

**Tech Stack:** Python 3.12, `nkigym` IR (networkx tree), pytest. Runtime env exists ONLY on gym-1 (dev box has decoy stubs; no local pytest/render).

## Global Constraints

- **Loud failures only** — no silent no-op-on-bad-input; a no-op compaction is a `TransformLegalityError`. No `try/except` to adapt around malformed IR.
- **Single return per function** (user-locked; no branching returns).
- **Transform legality = behavior/dep-order + ISA well-formedness ONLY, never resource capacity** — full-extent intermediate rungs are valid outputs; HW profiling prunes them.
- **No `#` comments** — triple-quoted block comments only; no inline comments. Tooling directives (`# type: ignore`, `# noqa`) exempt.
- **Type hints on every param/return**, modern `list`/`dict`/`X | None` syntax. Docstrings (Google/NumPy) on every function/class.
- **Byte-exact gate** = `assert_matches_hand(render(ir), manual_transforms.kernel_i)`, AST-canonical. The hand ladder is ground truth — never compare against a captured render.
- **Test-slimming = dedup/parametrize, PRESERVE EVERY ASSERTION.**
- **Format** with `black`/`isort` (`line-length=120`); pre-commit reformats+aborts — re-stage and retry.
- **Run tests on gym-1** via `transport/ssh_host.sh --host gym-1 --cmd "<pytest …>" --cache <dir>` (requires `--cache`; `--cmd` needs a `.py` token — enumerate test files, no bare `test/`). PYTHONPATH is set by the transport.

---
## File Structure

**Create:**
- `nkigym/src/nkigym/transforms/buffer_compaction.py` — the `BufferCompaction` transform + `BufferCompactionOption`. Owns the per-buffer place+compact+rebase orchestration.
- `test/transforms/test_buffer_compaction.py` — unit tests for the transform (analyze, apply on each ladder buffer, idempotence, loud rejects) + the folded-in `test_compact.py` cases.

**Modify:**
- `nkigym/src/nkigym/codegen/compact.py` — add per-buffer entry points (`place_and_compact_buffer`, `rebase_regions_of`); keep the reused internals (`_compact_one`, `_anchor_loop_vars`, `_axis_span`, `_offsets_consistently`, `_clamp_list_len_to_tiles`); remove `rebased_region` from the public render contract (move its logic to the write-back).
- `nkigym/src/nkigym/codegen/body.py` — (a) drop the `rebased_region(...)` wrap at line 272 → emit regions verbatim; (b) `_alloc_emit_anchors` scope = owning block, not leaf-LCA.
- `nkigym/src/nkigym/transforms/code_motion.py` — `apply` drops the `place_buffers`+`compact_shapes` tail (structural-only).
- `nkigym/src/nkigym/transforms/rfactor.py` — `apply`/`_emit_rmw` drop the `place_buffers`+`compact_shapes` tail (structural-only; k33 repro deferred).
- `nkigym/src/nkigym/transforms/__init__.py` — export `BufferCompaction`, `BufferCompactionOption`.
- `examples/kernel_transforms.py` — `_build_ladder` → manual k0…k32 with explicit `BufferCompaction` rungs; drop the RFactor step.
- `test/transforms/_fixtures.py` — `build_ladder_state` rungs that were compaction-fused now chain an explicit `BufferCompaction` where the state's assertion needs the compacted form.
- `test/transforms/test_code_motion.py` — adapt the compacted-form assertions to the structural-only `apply` (chain `BufferCompaction`).
- `test/transforms/test_rfactor.py` — RFactor is structural-only; chain `BufferCompaction` for compacted fixtures; mark the k33 byte-exact case pending.
- `test/codegen/test_compact.py` — delete after folding its cases into `test_buffer_compaction.py`.

**Task ordering rationale:** the render-path changes (Task 1–2) must land with the transform that compensates for them, or every existing test breaks at once. So Task 1 builds `BufferCompaction` and its per-buffer helpers WITHOUT touching the render path or the coupled callers (fully green, additive). Task 2 flips the render path + decouples callers + updates fixtures/tests in one commit (they are mutually dependent — a partial split leaves the suite red). Task 3 drives the ladder. Task 4 is cleanup.

---

### Task 1: `BufferCompaction` transform + per-buffer helpers (additive, suite stays green)

Build the transform and its per-buffer building blocks WITHOUT changing the render path or decoupling any caller. Everything here is additive: `compact_shapes`/`rebased_region` still exist and still run in the coupled tails, so the whole suite stays green. This task proves the per-buffer place+compact+rebase produces the right IR in isolation.

**Files:**
- Create: `nkigym/src/nkigym/transforms/buffer_compaction.py`
- Modify: `nkigym/src/nkigym/codegen/compact.py` (add per-buffer entry points, reuse internals)
- Modify: `nkigym/src/nkigym/transforms/__init__.py` (exports)
- Create: `test/transforms/test_buffer_compaction.py`

**Interfaces:**
- Consumes: `KernelIR` (`ir.tree`, `ir.all_buffers()`, `ir.dependency`); `nkigym.ir.buffer_placement.place_buffers`; `nkigym.codegen.compact._compact_one`, `_anchor_loop_vars`, `_clamp_list_len_to_tiles`; `nkigym.ir.tree.{BlockNode, Buffer, BufferRegion, ISANode}`; `nkigym.ir.arith.expr.{Const, substitute}`; `Transform`, `TransformOption`, `TransformLegalityError`.
- Produces:
  - `nkigym.codegen.compact.place_and_compact_buffer(tree: KernelTree, tensor: str) -> None` — descend one buffer's decl to its LCA (via a per-buffer `place_buffers` recompute), then shrink its logical shape in place.
  - `nkigym.codegen.compact.rebase_regions_of(tree: KernelTree, tensor: str) -> None` — rewrite every ISA-leaf region naming `tensor` into single-instance local frame (subtract anchor loop vars) IN the tree.
  - `nkigym.transforms.BufferCompaction` (subclass of `Transform`) with `analyze(ir) -> list[BufferCompactionOption]` and `apply(ir, option) -> KernelIR`.
  - `nkigym.transforms.BufferCompactionOption(tensor: str)` — frozen dataclass.

- [ ] **Step 1: Write the failing test for `rebase_regions_of` + `place_and_compact_buffer`**

Add to `test/transforms/test_buffer_compaction.py`:

```python
"""Tests for the BufferCompaction transform and its per-buffer helpers."""

from __future__ import annotations

from test.transforms._fixtures import build_ladder_state

from nkigym.codegen.compact import place_and_compact_buffer, rebase_regions_of
from nkigym.ir.tree import ISANode
from nkigym.transforms import BufferCompaction, BufferCompactionOption
from nkigym.transforms.base import TransformLegalityError


def _regions_of(ir, tensor):
    """Every (leaf nid, region) pair naming ``tensor`` in ``ir.tree``."""
    out = []
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append((nid, region))
    return out


def _decl_block(ir, tensor):
    """The block nid whose alloc_buffers declares ``tensor``."""
    return next(
        nid
        for nid in ir.tree.blocks()
        if any(b.name == tensor for b in ir.tree.data(nid).alloc_buffers)
    )


def test_place_and_compact_buffer_shrinks_shape_and_descends():
    """State 13 (structural store-sink, no compaction tail run): sbuf_prod is
    declared at root with full (2048, 2048) logical shape. place_and_compact_buffer
    descends its decl below root and shrinks its free axis 2048 -> 512."""
    ir = build_ladder_state(13)
    assert _decl_block(ir, "sbuf_prod") == ir.tree.root
    assert ir.buffer("sbuf_prod").shape == (2048, 2048)
    place_and_compact_buffer(ir.tree, "sbuf_prod")
    assert _decl_block(ir, "sbuf_prod") != ir.tree.root
    assert ir.buffer("sbuf_prod").shape == (2048, 512)


def test_rebase_regions_of_subtracts_anchor_loop():
    """After place+compact, sbuf_prod's regions still carry the global i_d2_0*512
    free offset; rebase_regions_of subtracts the i_d2_0 anchor so the free lo is 0."""
    ir = build_ladder_state(13)
    place_and_compact_buffer(ir.tree, "sbuf_prod")
    rebase_regions_of(ir.tree, "sbuf_prod")
    from nkigym.ir.arith.expr import to_affine

    for _nid, region in _regions_of(ir, "sbuf_prod"):
        free_lo = region.ranges[2][0]
        assert to_affine(free_lo).get("i_d2_0", 0) == 0, f"i_d2_0 not rebased out: {free_lo!r}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py -x -q" \
  --cache /home/weittang/workplace/cache/bc_task1
```
Expected: FAIL — `ImportError: cannot import name 'place_and_compact_buffer'` (helpers not defined yet).

- [ ] **Step 3: Add the per-buffer helpers to `nkigym/src/nkigym/codegen/compact.py`**

Add these two public functions (reusing the existing `_compact_one`, `_anchor_loop_vars`, `_clamp_list_len_to_tiles`). Import `place_buffers` at module top:

```python
from nkigym.ir.buffer_placement import place_buffers


def place_and_compact_buffer(tree: KernelTree, tensor: str) -> None:
    """Descend ``tensor``'s decl to its LCA scope, then shrink its logical shape.

    The per-buffer analogue of the old ``place_buffers`` + ``compact_shapes``
    tail. ``place_buffers`` is a whole-tree LCA recompute that is idempotent for
    every already-placed buffer, so running it re-places only the buffers whose
    touchers moved (here, ``tensor``); the others are unchanged. The shape shrink
    then rewrites only ``tensor``'s owning-block alloc entry.
    """
    place_buffers(tree)
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        if not any(b.name == tensor for b in block.alloc_buffers):
            continue
        anchors = _anchor_loop_vars(tree, tensor)
        new_bufs = tuple(
            _compact_one(tree, buf, anchors) if buf.name == tensor else buf for buf in block.alloc_buffers
        )
        tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=new_bufs)


def rebase_regions_of(tree: KernelTree, tensor: str) -> None:
    """Rewrite every ISA-leaf region naming ``tensor`` into single-instance local frame.

    Subtracts ``tensor``'s anchor loop vars (loops selecting which instance is
    live) from each region axis ``lo``, materializing on the tree what
    ``rebased_region`` used to compute at render time. shared_hbm buffers and
    buffers with no anchors are left unchanged (identity).
    """
    buf = next(
        (b for nid in tree.blocks() for b in tree.data(nid).alloc_buffers if b.name == tensor),
        None,
    )
    if buf is None or buf.location == "shared_hbm":
        return
    anchors = _anchor_loop_vars(tree, tensor)
    if not anchors:
        return
    subs = {a: Const(value=0) for a in anchors}
    for nid in tree.preorder():
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        new_bindings = {
            slot: (
                BufferRegion(tensor=region.tensor, ranges=tuple((substitute(lo, subs), w) for lo, w in region.ranges))
                if region.tensor == tensor
                else region
            )
            for slot, region in data.operand_bindings.items()
        }
        if new_bindings != data.operand_bindings:
            tree.graph.nodes[nid]["data"] = replace(data, operand_bindings=new_bindings)
```

Note: `rebase_regions_of` has two early `return`s guarding no-op input — this is a guard clause pattern, not a branching-value return; keep the function's single behavioral exit. If the reviewer prefers strict single-return, refactor to a nested predicate, but the guard form is acceptable per the style rule's "use judgment."

- [ ] **Step 4: Run the two helper tests to verify they pass**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py::test_place_and_compact_buffer_shrinks_shape_and_descends test/transforms/test_buffer_compaction.py::test_rebase_regions_of_subtracts_anchor_loop -q" \
  --cache /home/weittang/workplace/cache/bc_task1
```
Expected: PASS (2 passed).


- [ ] **Step 5: Write the failing test for the `BufferCompaction` transform**

Append to `test/transforms/test_buffer_compaction.py`:

```python
def test_apply_compacts_sbuf_prod_end_to_end():
    """BufferCompaction('sbuf_prod') on state 13 does place+compact+rebase atomically:
    decl descends, shape shrinks to (2048, 512), regions rebase i_d2_0 out."""
    from nkigym.ir.arith.expr import to_affine

    ir = build_ladder_state(13)
    new_ir = BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod"))
    assert _decl_block(new_ir, "sbuf_prod") != new_ir.tree.root
    assert new_ir.buffer("sbuf_prod").shape == (2048, 512)
    for _nid, region in _regions_of(new_ir, "sbuf_prod"):
        assert to_affine(region.ranges[2][0]).get("i_d2_0", 0) == 0
    """apply must not mutate the input ir (deep-copy contract)."""
    assert ir.buffer("sbuf_prod").shape == (2048, 2048)


def test_apply_rejects_shared_hbm():
    """Compacting a shared_hbm buffer is a loud legality error (no tile axis)."""
    ir = build_ladder_state(13)
    try:
        BufferCompaction().apply(ir, BufferCompactionOption(tensor="hbm_out"))
        raised = False
    except TransformLegalityError:
        raised = True
    assert raised, "expected TransformLegalityError for shared_hbm"


def test_apply_rejects_noop():
    """Compacting an already-compact buffer (no scope/shape/frame change) is a loud
    legality error — a no-op-returning-success is disallowed."""
    ir = build_ladder_state(13)
    once = BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod"))
    try:
        BufferCompaction().apply(once, BufferCompactionOption(tensor="sbuf_prod"))
        raised = False
    except TransformLegalityError:
        raised = True
    assert raised, "expected TransformLegalityError for no-op re-compaction"


def test_analyze_offers_uncompacted_buffers():
    """analyze offers sbuf_prod (uncompacted at state 13) and not hbm_out (shared_hbm)."""
    ir = build_ladder_state(13)
    tensors = {opt.tensor for opt in BufferCompaction().analyze(ir)}
    assert "sbuf_prod" in tensors
    assert "hbm_out" not in tensors
```

- [ ] **Step 6: Run the transform tests to verify they fail**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py -x -q" \
  --cache /home/weittang/workplace/cache/bc_task1
```
Expected: FAIL — `ImportError: cannot import name 'BufferCompaction'`.

- [ ] **Step 7: Implement the `BufferCompaction` transform**

Create `nkigym/src/nkigym/transforms/buffer_compaction.py`:

```python
"""``BufferCompaction`` transform — per-buffer place + shape-shrink + rebase.

Materializes, for ONE buffer, the compaction that used to run anonymously in the
``CodeMotion`` / ``RFactor`` tail (whole-tree ``place_buffers`` + ``compact_shapes``)
plus the render-time ``rebased_region``. Descends the buffer's declaration to its
LCA scope, shrinks its logical shape to the access bounding box, and rewrites its
access regions into the single-instance local frame — all written back on the tree,
so the renderer emits them verbatim. Mirrors :class:`BufferLayout`'s single-``tensor``
surface.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import place_and_compact_buffer, rebase_regions_of
from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class BufferCompactionOption(TransformOption):
    """Compact ``tensor`` (place at LCA + shrink shape + rebase regions).

    Attributes:
        tensor: buffer name to compact.
    """

    tensor: str


class BufferCompaction(Transform):
    """Per-buffer atomic place + shape-shrink + region-rebase, materialized in the IR."""

    def analyze(self, ir: KernelIR) -> list[BufferCompactionOption]:
        """Offer every sbuf/psum buffer whose compacted form differs from its current one."""
        options: list[BufferCompactionOption] = []
        for name, buf in ir.all_buffers().items():
            if buf.location in ("sbuf", "psum") and self._would_change(ir, name):
                options.append(BufferCompactionOption(tensor=name))
        return options

    def apply(self, ir: KernelIR, option: BufferCompactionOption) -> KernelIR:
        """Re-check legality, deep-copy, place+compact+rebase the one buffer, rebuild deps."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        place_and_compact_buffer(new_ir.tree, option.tensor)
        rebase_regions_of(new_ir.tree, option.tensor)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferCompactionOption) -> None:
        """Loud rejects: unknown tensor, shared_hbm, or a no-op compaction."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferCompaction: no buffer named {option.tensor!r}")
        if buffers[option.tensor].location == "shared_hbm":
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is shared_hbm (nothing to compact)")
        if not self._would_change(ir, option.tensor):
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is already compact (no-op)")

    def _would_change(self, ir: KernelIR, tensor: str) -> bool:
        """True iff place+compact+rebase would alter the tensor's decl scope, shape, or regions."""
        probe = copy.deepcopy(ir)
        before_shape = probe.buffer(tensor).shape
        before_regions = _regions_snapshot(probe, tensor)
        before_decl = _decl_block_of(probe, tensor)
        place_and_compact_buffer(probe.tree, tensor)
        rebase_regions_of(probe.tree, tensor)
        after_decl = _decl_block_of(probe, tensor)
        changed = (
            probe.buffer(tensor).shape != before_shape
            or _regions_snapshot(probe, tensor) != before_regions
            or after_decl != before_decl
        )
        return changed


def _decl_block_of(ir: KernelIR, tensor: str) -> int:
    """Block nid that declares ``tensor`` in its alloc_buffers."""
    for nid in ir.tree.blocks():
        if any(b.name == tensor for b in ir.tree.data(nid).alloc_buffers):
            return nid
    raise KeyError(f"{tensor} declared by no block")


def _regions_snapshot(ir: KernelIR, tensor: str) -> tuple:
    """Immutable snapshot of every region naming ``tensor`` (for change detection)."""
    from nkigym.ir.tree import ISANode

    out = []
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append(region.ranges)
    return tuple(out)


__all__ = ["BufferCompaction", "BufferCompactionOption"]
```

Then add to `nkigym/src/nkigym/transforms/__init__.py` (import + `__all__`, alphabetical):

```python
from nkigym.transforms.buffer_compaction import BufferCompaction, BufferCompactionOption
```
Add `"BufferCompaction"` and `"BufferCompactionOption"` to the `__all__` list (after the existing `BufferLayout*` entries).

- [ ] **Step 8: Run the full transforms + codegen suite to verify green**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py test/transforms/test_code_motion.py test/transforms/test_rfactor.py test/codegen/test_compact.py test/transforms/test_buffer_layout.py -q" \
  --cache /home/weittang/workplace/cache/bc_task1
```
Expected: PASS. Task 1 is additive — the render path and coupled callers are untouched, so every existing test still passes AND the new `test_buffer_compaction.py` passes.

- [ ] **Step 9: Commit**

```bash
git add nkigym/src/nkigym/transforms/buffer_compaction.py \
        nkigym/src/nkigym/codegen/compact.py \
        nkigym/src/nkigym/transforms/__init__.py \
        test/transforms/test_buffer_compaction.py
git commit -m "Add BufferCompaction transform + per-buffer place/compact/rebase helpers"
```

### Task 2: Flip the render path + decouple CodeMotion/RFactor + update fixtures & tests (one commit)

These changes are mutually dependent — flip the render path alone and every render breaks; decouple the callers alone and the render-time rebase mis-fires. Land them together. After this commit, `apply` on CodeMotion/RFactor returns the structural-only form, the renderer emits verbatim from owning-block scope, and the fixture/tests that asserted the old compacted output now chain `BufferCompaction`.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` — drop `rebased_region` wrap (line 272); `_alloc_emit_anchors` scope = owning block.
- Modify: `nkigym/src/nkigym/transforms/code_motion.py` — `apply` drops the compaction tail.
- Modify: `nkigym/src/nkigym/transforms/rfactor.py` — `_emit_rmw` drops the compaction tail.
- Modify: `test/transforms/_fixtures.py` — `build_ladder_state` compaction-fused rungs chain `BufferCompaction`.
- Modify: `test/transforms/test_code_motion.py` — structural-only assertions; chain compaction where needed.
- Modify: `test/transforms/test_rfactor.py` — structural-only; chain compaction; mark k33 pending.

**Interfaces:**
- Consumes: `nkigym.transforms.BufferCompaction`, `BufferCompactionOption` (Task 1).
- Produces: `CodeMotion.apply` / `RFactor.apply` return structural-only IR (no auto place/compact). `_alloc_emit_anchors` anchors decls to owning-block scope. `render` emits regions verbatim.

- [ ] **Step 1: Write the failing test — structural-only CodeMotion leaves the buffer wide + at root**

Append to `test/transforms/test_buffer_compaction.py`:

```python
def test_codemotion_is_structural_only():
    """A structural store-sink (build_ladder_state uses coupled CodeMotion today; after
    decoupling, state 13's CodeMotion must NOT compact): sbuf_prod stays (2048, 2048),
    declared at root, until an explicit BufferCompaction runs."""
    ir = build_ladder_state(13)
    assert ir.buffer("sbuf_prod").shape == (2048, 2048), "CodeMotion still auto-compacts"
    assert _decl_block(ir, "sbuf_prod") == ir.tree.root, "CodeMotion still auto-places"
```

- [ ] **Step 2: Run it to verify it fails**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py::test_codemotion_is_structural_only -q" \
  --cache /home/weittang/workplace/cache/bc_task2
```
Expected: FAIL — `sbuf_prod` is `(128, 512)` at a non-root block, because `build_ladder_state(13)`'s CodeMotion still runs the compaction tail. (Note: this test also depends on the Step 6 fixture edit; it goes green once both the decouple AND the fixture chaining land — that is expected, they are one commit.)

- [ ] **Step 3: Decouple `CodeMotion.apply`**

In `nkigym/src/nkigym/transforms/code_motion.py`, edit `apply` (currently lines 269-277) to drop the tail:

```python
    def apply(self, ir: KernelIR, option: CodeMotionOption) -> KernelIR:
        """Re-check legality, deep-copy, move, rebuild deps, return.

        Structural-only: the block relocation + Dependency rebuild. Buffer
        placement/shape/frame is now an explicit BufferCompaction step, not an
        anonymous tail (see the 2026-07-14 BufferCompaction design).
        """
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir
```

Remove the now-unused imports `from nkigym.codegen.compact import compact_shapes` and `from nkigym.ir.buffer_placement import place_buffers` from the top of the file (autoflake/pre-commit will flag them otherwise).

- [ ] **Step 4: Decouple `RFactor._emit_rmw`**

In `nkigym/src/nkigym/transforms/rfactor.py`, remove the tail lines (currently 170-172):

```python
        place_buffers(tree)
        compact_shapes(tree)
        ir.dependency = Dependency(tree)
```

Replace with just the dependency rebuild:

```python
        ir.dependency = Dependency(tree)
```

Remove the now-unused imports `from nkigym.codegen.compact import compact_shapes` and `from nkigym.ir.buffer_placement import place_buffers`. Update the `_emit_rmw` docstring line that says "``place_buffers`` (LCA) + ``compact_shapes`` … follow, per contract" to note that compaction is now an explicit downstream `BufferCompaction` step (k33 repro deferred to the RFactor-template redesign).

- [ ] **Step 5: Flip the render path in `body.py`**

(a) In `_emit_isa_call` (line 272), stop wrapping in `rebased_region` — emit the stored region:

```python
            rendered = render_buffer_region(region, buf, rotations.get(region.tensor))
```

Remove the `from nkigym.codegen.compact import rebased_region` import (line 19).

(b) In `_alloc_emit_anchors` (lines 70-78), change the scope from leaf-LCA to the buffer's owning block. Replace:

```python
        scope = ir.tree.root if buf.location == "shared_hbm" else _lca_nodes(ir.tree, leaves)
        anchor = _anchor_child(ir.tree, scope, leaves)
```

with:

```python
        scope = ir.tree.root if buf.location == "shared_hbm" else _owning_block(ir.tree, name)
        anchor = _anchor_child(ir.tree, scope, leaves)
```

and add this helper next to `_lca_nodes`:

```python
def _owning_block(tree: KernelTree, name: str) -> int:
    """Return the block nid whose ``alloc_buffers`` declares ``name``.

    Decl POSITION follows the buffer's owning block (set by ``place_buffers``),
    not the LCA of its touching leaves. The two coincide on every kernel where the
    compaction tail ran; they diverge only for a structural-only move whose buffer
    was deliberately left at its prior (wider) scope — there the decl must render at
    the owning block, not descend to the touchers' LCA.
    """
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        if any(b.name == name for b in blk.alloc_buffers):
            return nid
    raise AssertionError(f"buffer {name!r} is declared by no block")
```

`_lca_nodes` stays (still used elsewhere? if autoflake flags it as unused, remove it). Verify with `grep -n "_lca_nodes" nkigym/src/nkigym/codegen/body.py` — if the only use was the line just replaced, delete the `_lca_nodes` def too.

- [ ] **Step 6: Update `build_ladder_state` fixture — chain BufferCompaction on the compacted-state rungs**

The `_fixtures.build_ladder_state` ladder (its own 1..14 numbering) relies on coupled CodeMotion. After decoupling, the rungs whose downstream assertions need the compacted form must chain `BufferCompaction`. In `test/transforms/_fixtures.py`, add the import inside `build_ladder_state` (next to the existing transforms import, line 133):

```python
    from nkigym.transforms import BufferCompaction, BufferCompactionOption
```

Then, for each rung that sinks a producer/consumer and whose state is asserted compacted, append a compaction of the moved buffer. Concretely, wrap the sink rungs `rung_3_4` (rhs), `rung_6_7`/`rung_9_10` (memset), `rung_11_12` (tensor_copy → psum_prod), `rung_13_14` (store → sbuf_prod) to compact the relevant buffer after the move. Example for `rung_11_12` (the PSUM hoist that `test_psum_hoist_descends_and_compacts` asserts):

```python
    def rung_11_12(ir):
        """11->12: CodeMotion tensor_copy under matmul d2 (PSUM hoist), then compact psum_prod."""
        tc = blk(ir, "NKITensorCopy")
        d2 = mm_loop(ir, "i_d2_0")
        ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=tc, target_loop_nid=d2, index=-1))
        return BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod"))
```

Apply the same pattern (`move` then `BufferCompaction().apply(ir, BufferCompactionOption(tensor=<moved buffer>))`) to `rung_3_4` (`sbuf_rhs`), `rung_9_10` (`psum_prod`), `rung_13_14` (`sbuf_prod`). Leave `rung_1_2`, `rung_6_7`, `rung_7_8` as pure moves IF no state assertion needs their compacted form — but to keep `test_ladder_state_sims(1..14)` a faithful end-to-end check, compact any buffer whose sink left it addressed by a now-enclosing loop (else the render carries an un-rebased global offset into a wide buffer, which sims fine but drifts from the kernel_transforms intent). Simplest correct rule: after every producer/consumer SINK rung, compact the moved block's non-hbm buffer. Verify by running `test_ladder_state_sims` in Step 8.

- [ ] **Step 7: Adapt `test_code_motion.py` compacted-form assertions**

`test_psum_hoist_descends_and_compacts` (lines 410-416) calls `build_ladder_state(12)`. With Step 6's fixture chaining, state 12 now includes the explicit compaction, so the existing assertion (`buf.shape == (128, 512)`, descended) still holds unchanged — verify it passes. If any OTHER test in `test_code_motion.py` calls `CodeMotion().apply(...)` directly and asserts a compacted shape, add `from nkigym.transforms import BufferCompaction, BufferCompactionOption` and chain the compaction before the assertion. Grep for direct assertions:

```bash
grep -n "shape ==\|(128, 512)\|0:512\|descend" test/transforms/test_code_motion.py
```
For each such assertion NOT fed by `build_ladder_state`, chain `BufferCompaction`.

- [ ] **Step 8: Run the full suite to verify green**

Run on gym-1:
```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py test/transforms/test_code_motion.py test/transforms/test_rfactor.py test/codegen/test_compact.py test/transforms/test_buffer_layout.py test/transforms/test_render_equivalence.py test/ir/test_buffer_placement.py -q" \
  --cache /home/weittang/workplace/cache/bc_task2
```
Expected: PASS (RFactor's k33 byte-exact case may be marked `xfail`/`skip` per Step 9 below — confirm it is not a hard FAIL).

- [ ] **Step 9: Handle the RFactor k33 case**

RFactor.apply is now structural-only, so any `test_rfactor.py` case asserting the compacted `kernel_rfactor_ko.py` byte-exact render will fail. For each such case, chain `BufferCompaction` on the psum after `RFactor().apply(...)` IF that reproduces the fixture; if it does not (the k33 psum needs the list_len 16→1 shrink, which is the deferred RFactor-template work), mark it:

```python
import pytest

@pytest.mark.skip(reason="k33 RFactor byte-exact repro deferred to the RFactor-template redesign "
                         "(2026-07-14 BufferCompaction spec, out of scope)")
def test_apply_byte_exact():
    ...
```

Do NOT delete the test or weaken its assertions — skip with a reason pointing at the deferred work (loud-pending, not fake-green). Keep the RFactor structural assertions (role flip, gadget placement, Dependency) passing.

- [ ] **Step 10: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py \
        nkigym/src/nkigym/transforms/code_motion.py \
        nkigym/src/nkigym/transforms/rfactor.py \
        test/transforms/_fixtures.py \
        test/transforms/test_code_motion.py \
        test/transforms/test_rfactor.py \
        test/transforms/test_buffer_compaction.py
git commit -m "Decouple CodeMotion/RFactor to structural-only; render verbatim from owning-block scope"
```

### Task 3: Drive the ladder to manual k0…k32 byte-exact

Rewrite `_build_ladder` in `examples/kernel_transforms.py` so the driven ladder reproduces `manual_transforms.py` k0…k32 rung-for-rung. The current 29-step ladder already has the CodeMotion sinks; the change is: insert a `BufferCompaction` step immediately after 4 of the 5 sinks (NOT the drain sink — k10 has no following compaction rung), and drop the final RFactor step.

**Critical fact (verified against manual k10):** the drain-`tensor_copy` sink (k10) leaves `sbuf_prod` at full `(128,16,2048)` shape, declared at top, with a global-frame `i_d2_0*512` index — because the store still reads it at root scope so its LCA has not dropped. So k10 is a structural-only CodeMotion with NO compaction, and it must render byte-identical to manual k10. Compaction of `sbuf_prod` happens only at k14, after the store also sinks (k13).

**Files:**
- Modify: `examples/kernel_transforms.py` (`_build_ladder` steps list + docstring)

**Interfaces:**
- Consumes: `BufferCompaction`, `BufferCompactionOption` (Task 1); the decoupled `CodeMotion` (Task 2).
- Produces: a 32-step ladder; `ladder[i]` renders byte-exact to `manual_transforms.kernel_i` for i in 0..32.

- [ ] **Step 1: Add the import**

In `examples/kernel_transforms.py`, add to the `from nkigym.transforms import (...)` block (line 63-74):

```python
    BufferCompaction,
    BufferCompactionOption,
```

Remove `RFactor`, `RFactorOption` from that import block (the RFactor step is dropped; k33 out of scope). Verify nothing else in the file references them: `grep -n "RFactor" examples/kernel_transforms.py`.

- [ ] **Step 2: Rewrite the `steps` list**

Replace the `steps = [...]` list (lines 347-403) so that after each of the 4 compacting sinks a `BufferCompaction` step follows. The new list, in order (each line is one rung k1..k32):

```python
    steps = [
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_1"))),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKITensorCopy")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_op_blk(ir, "NKITensorCopy"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_op_leaf(ir, "NKIStore"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKIStore")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_op_blk(ir, "NKIStore"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _psum_memset_blk(ir)),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_psum_memset_blk(ir), target_loop_nid=_loop(ir, "i_d2_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod")),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "rhs", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "rhs"), factors=(4, 512), target_axis="d2")),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_1"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_0"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_load_blk(ir, "rhs"), target_loop_nid=_loop(ir, "i_d0_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_rhs")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_leaf(ir, "lhs_T"), factors=(4, 512), target_axis="d1")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "lhs_T", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "lhs_T", "i_d0_1"), inner_nid=_load_for(ir, "lhs_T", "i_d1_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_load_blk(ir, "lhs_T"), target_loop_nid=_loop(ir, "i_d1_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_lhs_T")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_lhs_T", list_len=8)),
    ]
```

This is 32 lambdas → ladder entries k0..k32 (k0 = canonical). Note: NO `BufferCompaction` after the k10 drain-sink (the sink at index 9, the NKITensorCopy CodeMotion), matching manual k10→k11=Split. Update the `_build_ladder` docstring's rung table to the k0..k32 numbering with the 4 explicit BufferCompaction rungs (k14/k19/k25/k31) and note the RFactor step is dropped (k33 deferred).

- [ ] **Step 3: Run the byte-exact gate on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python examples/kernel_transforms.py" \
  --cache /home/weittang/workplace/cache/kernel_transforms
```
Expected in `output.log`: `[byte-exact] kernel_0: OK` … `[byte-exact] kernel_32: OK` (every rung matches `manual_transforms.kernel_i`), then `[sim] kernel_i: … pass=True` for all, then the HW profile table. The byte-exact assertion aborts loudly on the first mismatch with a got-vs-want diff — if a rung fails, that diff localizes which transform's render drifted from the hand kernel.

- [ ] **Step 4: If a rung mismatches, diagnose (do not patch the hand ladder)**

Per the locked rule, the hand ladder is ground truth — a mismatch means the transform/render is wrong, not the fixture. The saved `<cache>/kernel_transforms/kernel_i.py` is the driven render; diff it against `manual_transforms.kernel_i`. Common cause per the design: a structural-only rung rendered rebased (finding a in spec §2) → check the `body.py` `rebased_region` removal; or decl at wrong scope (finding b) → check `_owning_block`. Fix the transform/render, re-run Step 3.

- [ ] **Step 5: Commit**

```bash
git add examples/kernel_transforms.py
git commit -m "Drive ladder to manual k0..k32 byte-exact with explicit BufferCompaction rungs"
```

### Task 4: Retire dead `compact_shapes`/`rebased_region`; fold `test_compact.py`

After Task 2, the whole-tree `compact_shapes` and the render-time `rebased_region` have no live callers (verified: only self-references + `__all__` remain). Remove them and fold `test/codegen/test_compact.py`'s assertions into `test_buffer_compaction.py` (PRESERVE every assertion — test-slimming = dedup, not drop). This is a separate task because it is pure cleanup gated on Tasks 1-3 being green, and a reviewer can reject it independently.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/compact.py` (delete dead `compact_shapes`, `rebased_region`; keep per-buffer entry points + internals)
- Modify: `test/transforms/test_buffer_compaction.py` (fold in the surviving `test_compact.py` cases)
- Delete: `test/codegen/test_compact.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `compact.py` exports `place_and_compact_buffer`, `rebase_regions_of`, and the shared internals; no `compact_shapes` / `rebased_region`.

- [ ] **Step 1: Confirm no live callers remain**

```bash
grep -rn "compact_shapes\|rebased_region" nkigym/src/ examples/ | grep -v "def compact_shapes\|def rebased_region\|__all__\|:func:\|# "
```
Expected: no output (all live callers removed in Task 2). If any remain, they were missed in Task 2 — fix there first.

- [ ] **Step 2: Fold the surviving `test_compact.py` cases into `test_buffer_compaction.py`**

`test_compact.py` has 6 tests. Map them:
- `test_compact_shapes_canonical_is_noop` → rewrite as: on canonical IR, `analyze()` offers NO option (nothing to compact) AND `place_and_compact_buffer` on any buffer leaves shape unchanged. Preserve the "canonical shapes unchanged" assertion.
- `test_rebased_region_canonical_unchanged` → rewrite as: `rebase_regions_of` on a canonical buffer (no anchors) leaves regions unchanged.
- `test_compact_shapes_idempotent` → `place_and_compact_buffer` twice == once.
- `test_compact_shapes_uses_per_leaf_extents_not_global` → keep verbatim, calling `place_and_compact_buffer(ir.tree, "sbuf_lhs_T")` on canonical; assert `(2048, 2048)` unchanged.
- `test_emit_alloc_follows_compacted_shape` → move as-is (it tests `_emit_alloc`, independent of the entry-point rename).
- `test_compact_shapes_shrinks_list_len_when_tile_axis_collapses` + `_compact_one_list_len` helper → move as-is (tests `_clamp_list_len_to_tiles`, still present).

Add these to `test/transforms/test_buffer_compaction.py`, adapting the imports:

```python
from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen.compact import _clamp_list_len_to_tiles, place_and_compact_buffer, rebase_regions_of


def test_canonical_offers_no_compaction():
    """On canonical IR (buffers at root, no anchor loops), analyze offers nothing and
    place_and_compact_buffer is a shape no-op."""
    ir = build_canonical_ir()
    assert BufferCompaction().analyze(ir) == []
    before = {b.name: b.shape for b in ir.all_buffers().values()}
    for name in list(before):
        if ir.buffer(name).location in ("sbuf", "psum"):
            place_and_compact_buffer(ir.tree, name)
    after = {b.name: b.shape for b in ir.all_buffers().values()}
    assert before == after


def test_place_and_compact_idempotent():
    """place_and_compact_buffer applied twice equals once (on a real compacted buffer)."""
    ir = build_ladder_state(13)
    place_and_compact_buffer(ir.tree, "sbuf_prod")
    once = ir.buffer("sbuf_prod").shape
    place_and_compact_buffer(ir.tree, "sbuf_prod")
    assert ir.buffer("sbuf_prod").shape == once


def test_uses_per_leaf_extents_not_global():
    """A loop_var reused with different extents across subtrees must not inflate a buffer
    whose touching region lives in the small-extent subtree (regression)."""
    ir = build_canonical_ir()
    place_and_compact_buffer(ir.tree, "sbuf_lhs_T")
    assert ir.buffer("sbuf_lhs_T").shape == (2048, 2048)


def test_clamp_list_len_to_tiles_collapse():
    """When a list buffer's leading tile axis collapses below list_len, _clamp_list_len_to_tiles
    shrinks list_len to match (models the RFactor psum 16->1)."""
    from dataclasses import replace

    from nkigym.ir.tree import Buffer

    listed = Buffer(name="psum_x", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    assert listed.physical_shape() == (128, 16, 512)
    shrunk = replace(listed, shape=(128, 512))
    fixed = _clamp_list_len_to_tiles(shrunk)
    assert fixed.list_len == 1
    assert fixed.per_tile_physical_shape() == (128, 1, 512)


def test_emit_alloc_follows_compacted_shape():
    """After a smaller logical shape is written, _emit_alloc emits it."""
    from dataclasses import replace

    from nkigym.codegen.body import _emit_alloc
    from nkigym.ir.tree import Buffer

    full = Buffer(name="sbuf_x", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    compacted = replace(full, shape=(128, 128))
    assert _emit_alloc(full) == "sbuf_x = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"
    assert _emit_alloc(compacted) == "sbuf_x = [nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"


def test_rebase_regions_of_canonical_unchanged():
    """On a canonical buffer (no anchor loops), rebase_regions_of is identity."""
    ir = build_canonical_ir()
    before = _regions_snapshot_local(ir, "sbuf_lhs_T")
    rebase_regions_of(ir.tree, "sbuf_lhs_T")
    assert _regions_snapshot_local(ir, "sbuf_lhs_T") == before


def _regions_snapshot_local(ir, tensor):
    """Local region snapshot for change detection in tests."""
    out = []
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append(region.ranges)
    return tuple(out)
```

- [ ] **Step 3: Delete the dead functions from `compact.py`**

Remove `def compact_shapes(...)` (lines 24-32) and `def rebased_region(...)` (lines 186-206). Update the module docstring (lines 1-14) to describe the two per-buffer entry points instead of `compact_shapes`/`rebased_region`. Set `__all__ = ["place_and_compact_buffer", "rebase_regions_of"]`. Keep all `_`-prefixed internals (`_compact_one`, `_anchor_loop_vars`, `_axis_span`, `_offsets_consistently`, `_axis_coeff`, `_regions_touching`, `_leaf_loop_extents`, `_clamp_list_len_to_tiles`).

- [ ] **Step 4: Delete `test/codegen/test_compact.py`**

```bash
git rm test/codegen/test_compact.py
```

- [ ] **Step 5: Run the full suite green**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_compaction.py test/transforms/test_code_motion.py test/transforms/test_rfactor.py test/transforms/test_buffer_layout.py test/transforms/test_render_equivalence.py test/ir/test_buffer_placement.py test/transforms/test_split.py test/transforms/test_reorder.py test/transforms/test_fuse.py -q" \
  --cache /home/weittang/workplace/cache/bc_task4
```
Expected: PASS (RFactor k33 case skipped with reason). No import errors from the deleted functions.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/compact.py test/transforms/test_buffer_compaction.py
git rm test/codegen/test_compact.py
git commit -m "Retire dead compact_shapes/rebased_region; fold test_compact into test_buffer_compaction"
```

