# ComputeAt / ReverseComputeAt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ComputeAt` (sink producer under consumer loop) and `ReverseComputeAt` (lift consumer under producer loop) transforms, preceded by precise affine region-overlap in the dependency graph so they are trustworthy on tiled IR.

**Architecture:** First extract LCA buffer placement into a reusable `place_buffers(tree)` so compute_at can re-run it (automatic buffer descent). Then replace the `_bufferregion_overlaps` stub with affine integer-set intersection and gate dependency edges on it. Then build the two transforms on TVM's six legality conditions, full-coverage-only: covered loops collapse into the target's shared scope, uncovered dims stay private; `apply` re-runs `place_buffers` so allocations descend automatically. PSUM-hoist is the end-to-end gate.

**Tech Stack:** Python 3.12, networkx, pytest, numpy, NKI CPU simulator. Hand-rolled affine `Expr` AST (already shipped) — no new external deps.

**Spec:** `docs/superpowers/specs/2026-05-29-compute-at-design.md`.

**Legality reference:** `nkigym/src/nkigym/transforms/compute_at_legality.md` (six conditions, worked examples).

**Environment:** `source ~/venvs/kernel-env/bin/activate`; run from repo root `/home/ubuntu/nki-autotune`; tests use `PYTHONPATH=nkigym/src` where they import via `simulate_fp32`. Current HEAD `814afcf`, all tests green (`pytest test/ --ignore=test/ops` → 97 passed), matmul example → 24 PASS.

---

## File Structure

New files:
- `nkigym/src/nkigym/ir/buffer_placement.py` — `place_buffers(tree)`: LCA-of-users placement, extracted from `canonical_build`.
- `nkigym/src/nkigym/ir/interval.py` — `AffineInterval`, `intervals_disjoint`, `regions_disjoint`.
- `nkigym/src/nkigym/transforms/compute_at.py` — `ComputeAt`, `ComputeAtOption`.
- `nkigym/src/nkigym/transforms/reverse_compute_at.py` — `ReverseComputeAt`, `ReverseComputeAtOption`.
- `test/ir/test_buffer_placement.py`, `test/ir/test_interval.py`, `test/transforms/test_compute_at.py`, `test/transforms/test_reverse_compute_at.py` — tests.

Modified:
- `nkigym/src/nkigym/ir/canonical_build.py` — call `place_buffers` instead of inline LCA.
- `nkigym/src/nkigym/ir/dependency.py` — region-gated hazard edges; `_BlockInfo` keeps per-tensor regions + enclosing-loop extents.
- `nkigym/src/nkigym/transforms/__init__.py` — export the two transforms + options.
- `nkigym/src/nkigym/transforms/compute_at_legality.md` — refresh "What is a block?" for decomposed canonical (no `init`).
- `examples/matmul_lhsT_rhs.py` — add the two transforms to the MDP list.

---

## Phase outline

1. Extract `place_buffers` (pure refactor) — Task 1. **[planned below]**
2. Affine region overlap — Tasks 2–4. **[planned below]**
3. Legality doc refresh — Task 5. *[planned after Phase 2 lands]*
4. ComputeAt — Tasks 6–8. *[planned after Phase 2 lands]*
5. ReverseComputeAt — Task 9. *[planned after Phase 2 lands]*
6. PSUM-hoist E2E + example wiring — Task 10. *[planned after Phase 2 lands]*

> **Plan status:** Tasks 1–4 (the region-overlap prerequisite) are fully
> specified with complete code below and ready to execute. Tasks 5–10
> (the compute_at transforms) are deferred: their `apply` mechanics
> (full-coverage loop collapse, binding rewrite, splice) are intricate
> enough that they should be written against the real post-Phase-2 code
> rather than speculatively. The spec
> (`docs/superpowers/specs/2026-05-29-compute-at-design.md`, Part B) is
> the authority for what they must do; this plan's Tasks 5–10 will be
> appended once Phase 2 is green.

---

## Phase 1 — Extract `place_buffers`

### Task 1: Extract LCA placement into `buffer_placement.place_buffers`

**Files:**
- Create: `nkigym/src/nkigym/ir/buffer_placement.py`
- Modify: `nkigym/src/nkigym/ir/canonical_build.py`
- Test: `test/ir/test_buffer_placement.py`

`canonical_build` currently inlines the LCA placement (the `touchers` dict, the `_lca` helper, the `placement` loop, the `replace(blk, alloc_buffers=...)` write). Move this into a standalone `place_buffers(tree)` that recomputes placement from scratch over the current tree topology, so compute_at can call it post-move. It must be idempotent: clear every block's `alloc_buffers`, recompute each buffer's LCA over its touchers, re-attach.

- [ ] **Step 1: Write the failing test**

Create `test/ir/test_buffer_placement.py`:

```python
"""Tests for nkigym.ir.buffer_placement.place_buffers."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.tree import BlockNode, ISANode


def _block_for_op(ir, op_name):
    from nkigym.ir.tree import ISANode as _ISA

    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            dd = ir.tree.data(d)
            if isinstance(dd, _ISA) and dd.op_cls.__name__ == op_name:
                return nid
    raise AssertionError(f"no block for {op_name}")


def test_place_buffers_canonical_matches_build():
    """place_buffers on the canonical tree reproduces the build-time placement:
    multi-toucher buffers at root, store-only hbm_out at the store block."""
    ir = build_canonical_ir()
    root = ir.tree.root
    store_nid = _block_for_op(ir, "NKIStore")

    """Capture placement, clear it, re-run, assert identical."""
    before = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
              for nid in ir.tree.blocks() if isinstance(ir.tree.data(nid), BlockNode)}
    place_buffers(ir.tree)
    after = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
             for nid in ir.tree.blocks() if isinstance(ir.tree.data(nid), BlockNode)}
    assert before == after, f"place_buffers not idempotent: {before} != {after}"

    """hbm_out lives on the store block; the four others at root."""
    assert "hbm_out" in after[store_nid]
    root_names = set(after[root])
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod"} <= root_names
    assert "hbm_out" not in root_names


def test_place_buffers_is_idempotent_when_called_twice():
    """Calling place_buffers twice yields the same placement as once."""
    ir = build_canonical_ir()
    place_buffers(ir.tree)
    once = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
            for nid in ir.tree.blocks()}
    place_buffers(ir.tree)
    twice = {nid: tuple(b.name for b in ir.tree.data(nid).alloc_buffers)
             for nid in ir.tree.blocks()}
    assert once == twice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && PYTHONPATH=nkigym/src pytest test/ir/test_buffer_placement.py -v`
Expected: FAIL with `ModuleNotFoundError: nkigym.ir.buffer_placement`.

- [ ] **Step 3: Create `buffer_placement.py`**

The placement needs the set of `Buffer` objects to place. After the initial build, those Buffers already live in the tree's `alloc_buffers`; `place_buffers` collects them from every block, clears them, recomputes. Create `nkigym/src/nkigym/ir/buffer_placement.py`:

```python
"""LCA-of-users buffer placement over a :class:`KernelTree`.

Each :class:`Buffer` is declared on the lowest-common-ancestor
:class:`BlockNode` of every block that reads or writes it — the
shallowest block whose scope dominates all uses (lifetime-correct).

:func:`place_buffers` is a pure recompute: it gathers every Buffer
currently declared anywhere in the tree, clears all ``alloc_buffers``,
recomputes each Buffer's LCA over its touchers, and re-attaches. It is
idempotent and safe to call after a transform has moved blocks (e.g.
``compute_at``), so allocations descend automatically when producer and
consumers become co-located.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.tree import BlockNode, Buffer, KernelTree


def place_buffers(tree: KernelTree) -> None:
    """Recompute and apply LCA-of-users placement in place.

    Gathers all Buffers from existing ``alloc_buffers``, clears them,
    then re-attaches each to the LCA block of its touchers. Buffers in
    a stable iteration order (first-seen) so a block's ``alloc_buffers``
    list is deterministic.
    """
    buffers = _gather_buffers(tree)
    _clear_alloc_buffers(tree)
    touchers = _touchers_by_tensor(tree)
    placement: dict[int, list[Buffer]] = {}
    for name, buf in buffers.items():
        touch = touchers.get(name)
        lca = tree.root if not touch else _lca(tree, touch)
        placement.setdefault(lca, []).append(buf)
    for block_nid, bufs in placement.items():
        blk = tree.data(block_nid)
        assert isinstance(blk, BlockNode)
        tree.graph.nodes[block_nid]["data"] = replace(blk, alloc_buffers=tuple(bufs))


def _gather_buffers(tree: KernelTree) -> dict[str, Buffer]:
    """Collect every Buffer currently declared in any block, keyed by name, first-seen order."""
    out: dict[str, Buffer] = {}
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        for buf in blk.alloc_buffers:
            if buf.name not in out:
                out[buf.name] = buf
    return out


def _clear_alloc_buffers(tree: KernelTree) -> None:
    """Set every block's ``alloc_buffers`` to the empty tuple."""
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        if blk.alloc_buffers:
            tree.graph.nodes[nid]["data"] = replace(blk, alloc_buffers=())


def _touchers_by_tensor(tree: KernelTree) -> dict[str, set[int]]:
    """Map each buffer name to the set of block nids that read or write it."""
    touchers: dict[str, set[int]] = {}
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        for region in (*blk.reads, *blk.writes):
            touchers.setdefault(region.tensor, set()).add(nid)
    return touchers


def _lca(tree: KernelTree, nids: set[int]) -> int:
    """Lowest common ancestor of ``nids`` (deepest common ancestor).

    For a single-element set, returns that element.
    """
    if len(nids) == 1:
        return next(iter(nids))
    ancestor_sets: list[set[int]] = []
    for nid in nids:
        anc = set(tree.ancestors(nid))
        anc.add(nid)
        ancestor_sets.append(anc)
    common = ancestor_sets[0].intersection(*ancestor_sets[1:])
    lca_nid = tree.root
    max_depth = -1
    for candidate in common:
        depth = len(tree.ancestors(candidate))
        if depth > max_depth:
            max_depth = depth
            lca_nid = candidate
    return lca_nid


__all__ = ["place_buffers"]
```

Note: `_touchers_by_tensor` keys on the block's declared `reads`/`writes` BufferRegions. Parameters (e.g. `lhs_T`) appear in reads but have no Buffer (they're in `ir.param_buffers`), so `_gather_buffers` never yields them and they're skipped — params are never placed. Good.

- [ ] **Step 4: Rewrite `canonical_build` to call `place_buffers`**

In `nkigym/src/nkigym/ir/canonical_build.py`, replace the inline placement. The builder must first attach each Buffer somewhere so `place_buffers` can gather them — simplest: attach all Buffers to the root block, then call `place_buffers` to redistribute. Replace the body of `build_canonical_blocknode_tree` (the toucher/placement/replace section, lines ~44-75) with:

```python
def build_canonical_blocknode_tree(analysis: "_AnalysisResult") -> KernelTree:
    """Build the canonical :class:`BlockNode`-rooted tree.

    Tree.root is already an empty BlockNode from KernelTree.__init__.
    Build leaf blocks under it, seed all Buffers on the root, then run
    LCA placement to distribute them to their lifetime-dominating blocks.
    """
    from dataclasses import replace

    from nkigym.ir.buffer_placement import place_buffers

    tree = KernelTree()
    op_records = list(analysis.ops)
    buffers_by_name = _collect_buffers(analysis.tensors, analysis.param_names)
    compute_records = [rec for rec in op_records if rec.op_cls is not NKIAlloc]
    for rec in compute_records:
        _build_subblock(tree, tree.root, rec, analysis)
    """Seed every Buffer on the root block, then let place_buffers redistribute by LCA."""
    root_blk = tree.data(tree.root)
    tree.graph.nodes[tree.root]["data"] = replace(root_blk, alloc_buffers=tuple(buffers_by_name.values()))
    place_buffers(tree)
    return tree
```

Then DELETE the now-unused `_lca` function from `canonical_build.py` (it moved to `buffer_placement.py`). Keep `_collect_buffers`, `_build_subblock`, and the region helpers. Remove the `from dataclasses import replace` at module top if it's now only used in the function (it's imported locally above); check and remove the duplicate top-level import if present. Update the module docstring's "Buffer placement" paragraph to say placement is delegated to `buffer_placement.place_buffers`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_buffer_placement.py test/ir/ test/codegen/ -q`
Expected: all pass. Canonical output unchanged.

- [ ] **Step 6: Confirm rendered canonical is byte-identical**

Run:
```bash
PYTHONPATH=nkigym/src python -c "from test.transforms._fixtures import build_canonical_ir; from nkigym.codegen import render; print(render(build_canonical_ir()))" > /tmp/after_place.py
git stash; PYTHONPATH=nkigym/src python -c "from test.transforms._fixtures import build_canonical_ir; from nkigym.codegen import render; print(render(build_canonical_ir()))" > /tmp/before_place.py 2>/dev/null || true; git stash pop
diff /tmp/before_place.py /tmp/after_place.py && echo "IDENTICAL"
```
Expected: `IDENTICAL` (pure refactor). If the stash/pop is awkward mid-edit, instead compare against a known-good render captured before starting — the key assertion is no rendered diff.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/ir/buffer_placement.py nkigym/src/nkigym/ir/canonical_build.py test/ir/test_buffer_placement.py
git commit -m "Extract LCA buffer placement into reusable place_buffers(tree)"
```

---

## Phase 2 — Affine region overlap

### Task 2: `AffineInterval` + `intervals_disjoint`

**Files:**
- Create: `nkigym/src/nkigym/ir/interval.py`
- Test: `test/ir/test_interval.py`

An `AffineInterval` is the integer range one `BufferRegion` axis covers:
a half-open `[base, base + width)` where `base` is affine in loop-var
symbols. Two intervals are *provably disjoint* iff the integer range of
`(a.base - b.base)` over the loop-var box cannot land in the open
overlap window `(-b.width, a.width)`.

- [ ] **Step 1: Write the failing test**

Create `test/ir/test_interval.py`:

```python
"""Tests for nkigym.ir.interval affine interval disjointness."""

from __future__ import annotations

from nkigym.ir.interval import AffineInterval, intervals_disjoint


def _iv(coeffs, width):
    return AffineInterval(coeffs=coeffs, width=width)


def test_same_var_same_tile_overlaps():
    """i_m*128 vs i_m*128, width 128 each -> difference 0 in (-128,128) -> overlap."""
    a = _iv({"i_m": 128}, 128)
    b = _iv({"i_m": 128}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16})


def test_same_var_adjacent_tiles_disjoint():
    """i_m*128 vs i_m*128+128 -> difference -128, not in (-128,128) -> disjoint."""
    a = _iv({"i_m": 128, None: 0}, 128)
    b = _iv({"i_m": 128, None: 128}, 128)
    assert intervals_disjoint(a, b, {"i_m": 16})


def test_same_var_gap_disjoint():
    """offset 256 apart, width 128 -> disjoint."""
    a = _iv({"i_m": 128, None: 0}, 128)
    b = _iv({"i_m": 128, None: 256}, 128)
    assert intervals_disjoint(a, b, {"i_m": 16})


def test_different_vars_overlap_sound_fallback():
    """i_m*128 vs i_n*128 (independent vars) -> difference ranges across 0 -> overlap."""
    a = _iv({"i_m": 128}, 128)
    b = _iv({"i_n": 128}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16, "i_n": 16})


def test_negative_coefficient_range():
    """Reversed-iteration base (negative coeff) still bounded correctly.
    a.base = -i_m*128 (ranges [-1920, 0]); b.base = 0. diff = -i_m*128 in [-1920,0].
    Overlap window (-128, 128) intersects [-1920,0] at [-127,0] -> overlap."""
    a = _iv({"i_m": -128}, 128)
    b = _iv({None: 0}, 128)
    assert not intervals_disjoint(a, b, {"i_m": 16})


def test_constant_only_intervals():
    """Two constant intervals far apart -> disjoint; touching -> disjoint (half-open); overlapping -> overlap."""
    assert intervals_disjoint(_iv({None: 0}, 128), _iv({None: 128}, 128), {})
    assert not intervals_disjoint(_iv({None: 0}, 128), _iv({None: 64}, 128), {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_interval.py -v`
Expected: FAIL with `ModuleNotFoundError: nkigym.ir.interval`.

- [ ] **Step 3: Implement `interval.py`**

Create `nkigym/src/nkigym/ir/interval.py`:

```python
"""Affine integer-interval disjointness for buffer-region overlap analysis.

An :class:`AffineInterval` is a half-open integer range
``[base, base + width)`` where ``base`` is an affine combination of
loop-var symbols (``coeffs`` in :func:`nkigym.ir.expr.to_affine` form)
and ``width`` is a constant. Two intervals on the same axis are
*provably disjoint* iff the integer range of ``a.base - b.base`` over
the loop-var box (each var in ``[0, extent)``) cannot fall in the open
overlap window ``(-b.width, a.width)``.

Soundness: when the difference range straddles the window (e.g. two
independent loop vars), we conservatively report *not disjoint* — never
a false "disjoint", so dependency edges are never dropped incorrectly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class AffineInterval:
    """Half-open integer interval ``[base, base + width)``.

    Attributes:
        coeffs: affine coefficients of ``base`` (``{var: coeff, ..., None: const}``).
        width: constant tile width.
    """

    coeffs: dict[str | None, int]
    width: int


def intervals_disjoint(a: AffineInterval, b: AffineInterval, loop_extents: dict[str, int]) -> bool:
    """Return True iff ``a`` and ``b`` are disjoint for EVERY assignment of the loop vars.

    ``loop_extents`` maps each loop-var name to its trip count (the var
    ranges over ``[0, extent)``). Overlap requires
    ``-b.width < (a.base - b.base) < a.width``. We compute the integer
    range of ``a.base - b.base`` over the box and check it cannot intersect
    that open window.
    """
    diff = _sub(a.coeffs, b.coeffs)
    lo, hi = _affine_range(diff, loop_extents)
    """Open overlap window: (-b.width, a.width). Overlap iff lo < a.width and hi > -b.width."""
    overlaps = lo < a.width and hi > -b.width
    return not overlaps


def _sub(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise ``a - b``."""
    out = dict(a)
    for var, coeff in b.items():
        out[var] = out.get(var, 0) - coeff
    return out


def _affine_range(coeffs: dict[str | None, int], loop_extents: dict[str, int]) -> tuple[int, int]:
    """Integer range ``[lo, hi]`` of an affine expression over the loop-var box.

    Each var ``v`` ranges over ``[0, extent_v - 1]``. A var with no known
    extent is treated as unbounded → returns a window-spanning range so the
    caller conservatively reports overlap.
    """
    lo = coeffs.get(None, 0)
    hi = coeffs.get(None, 0)
    for var, coeff in coeffs.items():
        if var is None:
            continue
        if var not in loop_extents:
            """Unknown extent — unbounded; force overlap by returning a maximal range."""
            return (-(1 << 62), (1 << 62))
        span = coeff * (loop_extents[var] - 1)
        if span >= 0:
            hi += span
        else:
            lo += span
    return (lo, hi)


__all__ = ["AffineInterval", "intervals_disjoint"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_interval.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/interval.py test/ir/test_interval.py
git commit -m "Add AffineInterval + intervals_disjoint"
```

### Task 3: `regions_disjoint` over BufferRegions (with partition-axis normalisation)

**Files:**
- Modify: `nkigym/src/nkigym/ir/interval.py`
- Test: `test/ir/test_interval.py`

Two `BufferRegion`s on the same tensor are disjoint iff ANY axis is
provably disjoint (box intersection empty). The interval for each axis
is built from the range `(lo, width_const)`. The SBUF/PSUM **partition
axis** (axis 0) uses bare-tile-index encoding: `lo` is the bare tile
index Var and `width` is 128, meaning the element-space interval is
`[lo*128, lo*128 + 128)`. Normalise it to element space before
comparison so all axes share a coordinate system.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_interval.py`:

```python
def test_regions_disjoint_multi_axis_one_disjoint():
    """Disjoint on axis 1 (different constant tiles), same on axis 0 -> regions disjoint."""
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import Buffer, BufferRegion
    from nkigym.ir.interval import regions_disjoint

    buf = Buffer(name="t", shape=(2048, 2048), dtype="float32", location="shared_hbm")
    a = BufferRegion(tensor="t", ranges=(
        (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
        (Const(value=0), Const(value=512)),
    ))
    b = BufferRegion(tensor="t", ranges=(
        (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
        (Const(value=512), Const(value=512)),
    ))
    assert regions_disjoint(a, b, buf, buf, {"i": 16})


def test_regions_overlap_all_axes():
    """Same indices on every axis -> overlap."""
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import Buffer, BufferRegion
    from nkigym.ir.interval import regions_disjoint

    buf = Buffer(name="t", shape=(2048, 2048), dtype="float32", location="shared_hbm")
    r = BufferRegion(tensor="t", ranges=(
        (Mul(left=Var(name="i"), right=Const(value=128)), Const(value=128)),
        (Const(value=0), Const(value=512)),
    ))
    assert not regions_disjoint(r, r, buf, buf, {"i": 16})


def test_regions_partition_axis_normalised():
    """SBUF partition axis 0 uses bare tile index; different tiles must be disjoint.
    a: psum[i_m, ...]  b: psum[i_m', ...] would overlap (independent vars), but
    a: psum tile index Var i_m vs constant-offset is the canonical case. Here test
    that bare-index axis-0 with the SAME var but the renderer's bare-Var encoding
    is normalised: tile i_m and tile i_m are same -> overlap; tile i_m vs i_m via
    different vars -> overlap (sound). The normalisation must not crash and must
    treat width-128 partition tiles in element space."""
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import Buffer, BufferRegion
    from nkigym.ir.interval import regions_disjoint

    buf = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    """Axis 0 is bare-Var partition index (i_m), axis 1 is element offset."""
    a = BufferRegion(tensor="p", ranges=(
        (Var(name="i_m"), Const(value=128)),
        (Const(value=0), Const(value=512)),
    ))
    b = BufferRegion(tensor="p", ranges=(
        (Var(name="i_m"), Const(value=128)),
        (Const(value=512), Const(value=512)),
    ))
    """Same partition tile, disjoint free-axis tiles -> disjoint overall."""
    assert regions_disjoint(a, b, buf, buf, {"i_m": 16})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_interval.py -v -k regions`
Expected: FAIL with `ImportError: cannot import name 'regions_disjoint'`.

- [ ] **Step 3: Implement `regions_disjoint` + interval builder**

Append to `nkigym/src/nkigym/ir/interval.py` (add the imports at top):

```python
from nkigym.ir.expr import Const, to_affine
from nkigym.ir.tree import Buffer, BufferRegion

_PARTITION_DIM = 128


def regions_disjoint(
    a: BufferRegion, b: BufferRegion, buf_a: Buffer, buf_b: Buffer, loop_extents: dict[str, int]
) -> bool:
    """Return True iff two same-tensor regions are provably disjoint.

    Disjoint iff ANY axis is provably disjoint (box intersection empty).
    Each axis range ``(lo, width_const)`` becomes an :class:`AffineInterval`;
    the SBUF/PSUM partition axis (axis 0) is normalised from bare tile-index
    to element space.
    """
    if len(a.ranges) != len(b.ranges):
        return False
    disjoint_on_some_axis = False
    for axis_index, (range_a, range_b) in enumerate(zip(a.ranges, b.ranges)):
        iv_a = _interval_for_axis(range_a, axis_index, buf_a)
        iv_b = _interval_for_axis(range_b, axis_index, buf_b)
        if intervals_disjoint(iv_a, iv_b, loop_extents):
            disjoint_on_some_axis = True
            break
    return disjoint_on_some_axis


def _interval_for_axis(axis_range: tuple, axis_index: int, buf: Buffer) -> AffineInterval:
    """Build an element-space :class:`AffineInterval` from one ``(lo, width_const)`` range.

    Axis 0 of SBUF/PSUM buffers carries a bare tile-index ``lo`` and
    width 128; convert to element space (multiply base coeffs by 128,
    width = 128) so every axis shares a coordinate system.
    """
    lo_expr, width_expr = axis_range
    if not isinstance(width_expr, Const):
        raise ValueError(f"region width must be Const; got {width_expr!r}")
    base = to_affine(lo_expr)
    width = width_expr.value
    is_partition = axis_index == 0 and buf.location in ("sbuf", "psum") and width == _PARTITION_DIM
    if is_partition:
        """bare tile index -> element space: base *= 128, width stays 128."""
        base = {var: coeff * _PARTITION_DIM for var, coeff in base.items()}
    return AffineInterval(coeffs=base, width=width)
```

Update `__all__` to add `regions_disjoint`.

Note on the partition normalisation: for a non-partition axis, `lo` is already in element space (`i*128`), and `_interval_for_axis` leaves it. For the partition axis, `lo` is the bare tile index (`i_m`), so multiplying base by 128 gives `[i_m*128, i_m*128+128)` — element space. Both branches end up element-space, so cross-axis comparison is consistent.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_interval.py -v`
Expected: all pass (6 from Task 2 + 3 here).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/interval.py test/ir/test_interval.py
git commit -m "Add regions_disjoint with partition-axis normalisation"
```

### Task 4: Gate dependency hazard edges on region overlap

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py`

`_BlockInfo` currently keeps only tensor-name sets. Extend it to keep the
per-tensor `BufferRegion`s and the block's enclosing-loop extents, then
gate each candidate hazard edge on `not regions_disjoint(...)`.

- [ ] **Step 1: Write the failing regression test**

Append to `test/ir/test_dependency.py`:

```python
def test_disjoint_tile_writes_have_no_edge():
    """Two hand-built blocks writing disjoint tiles of one buffer get NO dependency edge."""
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
    from nkigym.ops.base import AxisRole
    from nkigym.ops.memset import NKIMemset

    tree = KernelTree()
    """Two sibling blocks under root, each writing a distinct CONSTANT tile of 'buf'."""
    def add_writer(offset):
        blk = BlockNode(
            iter_vars=(IterVar(axis="d0", dom=(0, 256), role=AxisRole.PARALLEL),),
            iter_values=(Var(name="i"),),
            reads=(),
            writes=(BufferRegion(tensor="buf", ranges=((Const(value=offset), Const(value=128)),)),),
        )
        nid = tree.add_node(blk, parent=tree.root)
        f = tree.add_node(ForNode(loop_var="i", extent=1), parent=nid)
        tree.add_node(ISANode(op_cls=NKIMemset, operand_bindings={
            "dst": BufferRegion(tensor="buf", ranges=((Const(value=offset), Const(value=128)),))}, kwargs={"value": 0.0}), parent=f)
        return nid

    a = add_writer(0)
    b = add_writer(128)
    dep = Dependency(tree)
    assert not dep.must_precede(a, b)
    assert not dep.must_precede(b, a)


def test_overlapping_tile_writes_have_edge():
    """Two blocks writing the SAME tile get a WAW edge."""
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
    from nkigym.ops.base import AxisRole
    from nkigym.ops.memset import NKIMemset

    tree = KernelTree()
    def add_writer():
        blk = BlockNode(
            iter_vars=(IterVar(axis="d0", dom=(0, 256), role=AxisRole.PARALLEL),),
            iter_values=(Var(name="i"),),
            reads=(),
            writes=(BufferRegion(tensor="buf", ranges=((Const(value=0), Const(value=128)),)),),
        )
        nid = tree.add_node(blk, parent=tree.root)
        f = tree.add_node(ForNode(loop_var="i", extent=1), parent=nid)
        tree.add_node(ISANode(op_cls=NKIMemset, operand_bindings={
            "dst": BufferRegion(tensor="buf", ranges=((Const(value=0), Const(value=128)),))}, kwargs={"value": 0.0}), parent=f)
        return nid

    a = add_writer()
    b = add_writer()
    dep = Dependency(tree)
    assert dep.must_precede(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_dependency.py -v -k "disjoint_tile or overlapping_tile"`
Expected: `test_disjoint_tile_writes_have_no_edge` FAILS (current code emits a spurious edge on tensor-name match); `test_overlapping_tile` passes.

- [ ] **Step 3: Extend `_BlockInfo` and gate edges**

In `nkigym/src/nkigym/ir/dependency.py`:

Add import at top: `from nkigym.ir.interval import regions_disjoint` and `from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, KernelTree`.

Replace the `_BlockInfo` dataclass to carry regions + buffers + extents:

```python
@dataclass(frozen=True)
class _BlockInfo:
    """Cached read/write regions, the buffers they touch, and enclosing-loop extents."""

    name: str
    reads: frozenset[str]
    writes: frozenset[str]
    read_regions: tuple[BufferRegion, ...]
    write_regions: tuple[BufferRegion, ...]
    extents: dict[str, int]
    buffers: dict[str, Buffer]
```

`_summarise` must now also collect each block's enclosing-loop extents
(walk descendant ForNodes within the block — but extents are per loop_var,
and the block's own loops live between it and its ISA leaf) and resolve
each region's tensor to its `Buffer`. Add a `tree` + `ir`-buffers handle.
Since `Dependency.__init__` only gets the `tree`, resolve buffers by
walking `tree.blocks()` `alloc_buffers` once into a name→Buffer map, plus
param buffers are absent (params have no Buffer — treat a region whose
tensor has no Buffer as "full tensor, always overlaps", i.e. skip the
disjoint test for it and keep the conservative edge).

Rewrite `_summarise` to take the per-name buffer map and the block's
loop extents:

```python
    def _build(self, tree: KernelTree) -> None:
        buffers = self._buffer_map(tree)
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if not block.iter_vars and not block.reads and not block.writes:
                continue
            info = self._summarise(nid, block, tree, buffers)
            self.graph.add_node(nid, info=info)
            self.blocks.append(nid)
            for name in info.reads | info.writes:
                self.touches_by_tensor.setdefault(name, []).append(nid)
            self._record_hazards(nid, info, last_writer, prior_readers)
            for name in info.writes:
                last_writer[name] = nid
                prior_readers.pop(name, None)
            for name in info.reads - info.writes:
                prior_readers.setdefault(name, []).append(nid)

    @staticmethod
    def _buffer_map(tree: KernelTree) -> dict[str, Buffer]:
        out: dict[str, Buffer] = {}
        for nid in tree.blocks():
            blk = tree.data(nid)
            assert isinstance(blk, BlockNode)
            for buf in blk.alloc_buffers:
                out[buf.name] = buf
        return out

    def _summarise(self, nid: int, block: BlockNode, tree: KernelTree, buffers: dict[str, Buffer]) -> _BlockInfo:
        extents: dict[str, int] = {}
        for d in tree.descendants(nid):
            dd = tree.data(d)
            if isinstance(dd, ForNode):
                extents[dd.loop_var] = dd.extent
        reads = {r.tensor for r in block.reads}
        writes = {w.tensor for w in block.writes}
        return _BlockInfo(
            name=_block_name(block),
            reads=frozenset(reads),
            writes=frozenset(writes),
            read_regions=tuple(block.reads),
            write_regions=tuple(block.writes),
            extents=extents,
            buffers=buffers,
        )
```

Then gate the hazards. `_record_hazards` and `_try_edge` must check
region overlap before inserting an edge. Change `_record_hazards` to pass
the relevant region pair:

```python
    def _record_hazards(
        self, nid: int, info: _BlockInfo, last_writer: dict[str, int], prior_readers: dict[str, list[int]]
    ) -> None:
        for name in info.reads:
            self._try_edge(last_writer.get(name), nid, "RAW", name)
        for name in info.writes:
            self._try_edge(last_writer.get(name), nid, "WAW", name)
            for prior_r in prior_readers.get(name, ()):
                self._try_edge(prior_r, nid, "WAR", name)

    def _regions_for(self, nid: int, tensor: str, kind: str) -> tuple[BufferRegion, ...]:
        """Regions of ``tensor`` touched by block ``nid`` on the read or write side."""
        info = self.graph.nodes[nid]["info"]
        side = info.write_regions if kind == "write" else info.read_regions
        return tuple(r for r in side if r.tensor == tensor)

    def _try_edge(self, producer: int | None, consumer: int, kind: str, tensor: str) -> None:
        if producer is None or producer == consumer:
            return
        if self._provably_disjoint(producer, consumer, tensor, kind):
            return
        if self.graph.has_edge(producer, consumer):
            current = self.graph.edges[producer, consumer]["kind"]
            if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                return
        self.graph.add_edge(producer, consumer, kind=kind)

    def _provably_disjoint(self, producer: int, consumer: int, tensor: str, kind: str) -> bool:
        """True iff every producer-region/consumer-region pair on ``tensor`` is disjoint.

        RAW: producer writes, consumer reads. WAW: both write. WAR: producer
        reads, consumer writes. If the tensor has no Buffer (a kernel param),
        treat as full-tensor → never disjoint (keep the edge).
        """
        pinfo = self.graph.nodes[producer]["info"]
        cinfo = self.graph.nodes[consumer]["info"]
        if tensor not in pinfo.buffers:
            return False
        buf = pinfo.buffers[tensor]
        prod_side = "write" if kind in ("RAW", "WAW") else "read"
        cons_side = "read" if kind == "RAW" else "write"
        prod_regions = self._regions_for(producer, tensor, prod_side)
        cons_regions = self._regions_for(consumer, tensor, cons_side)
        extents = {**pinfo.extents, **cinfo.extents}
        for pr in prod_regions:
            for cr in cons_regions:
                if not regions_disjoint(pr, cr, buf, buf, extents):
                    return False
        return True
```

Caveat to note in a comment: the two blocks may bind the same tile to
*different* loop_var names (each block has its own `i_d0_0`). When they
share a name, `intervals_disjoint` treats them as the same iteration
(correct for nested/shared loops); when names differ, it conservatively
reports overlap. For the canonical (sibling) case both blocks use the
same loop_var names but are independent — that yields conservative
overlap, preserving today's edges. Disjointness only kicks in for
*constant*-offset tiles (no shared var), which is the safe case.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=nkigym/src pytest test/ir/test_dependency.py -v`
Expected: all pass, including the two new ones.

- [ ] **Step 5: Full regression**

Run: `PYTHONPATH=nkigym/src pytest test/ --ignore=test/ops -q`
Expected: all pass — canonical dependency edges unchanged (full-tensor regions never disjoint), only the synthetic disjoint-tile test exercises the new path.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Gate dependency hazard edges on affine region overlap"
```

---

