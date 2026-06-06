# Software Pipeline (Tier B) Implementation Plan

> **STATUS (post-completion):** Increment 1 SHIPPED (`dev_1`); Increment 2
> (skew renderer) DEFERRED. This is the as-written execution record — some
> references below to `examples/handwritten_pipeline_matmul.py` (the throwaway
> validation/oracle-source driver) are stale: that file was removed after
> completion. The shipped oracle is `kernel_15` in `kernel_transforms.py`; the
> HW result is reproduced by `examples/tune_matmul_lhsT_rhs.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SoftwarePipeline` transform that double-buffers a loop's
accumulator by assigning its child blocks to pipeline stages, lifting the
matmul kernel from ~59% to ~84% MFU (HW-validated target).

**Architecture:** Faithful port of TVM's `InjectSoftwarePipeline`. The
persisted knob is a per-child-block `(stages, order)` annotation on the
pipelined loop's parent `BlockNode.annotations`; per-buffer version count
(`use_stage − def_stage + 1`) is materialized on `Buffer.versions` so
`physical_shape()` stays the single source of truth; the
prologue/skewed-body/epilogue structure and the `% versions` index rotation
are manifested at render time. Tier B = stage assignment only (identity
order); a linear matmul chain has no independent siblings to reorder.

**Tech Stack:** Python 3.12, the nkigym IR (`nx.DiGraph` schedule tree,
sidecar `Dependency` DAG, affine `Expr` arith), pytest, CPU fp32 simulator,
byte-exact AST oracle (`_ladder_compare.assert_matches_hand`), Trn2 profiling
via `transport/kaizen.sh`.

**Spec:** `docs/superpowers/specs/2026-06-06-software-pipeline-tier-b-design.md`

---

## File Structure

**New files:**
- `nkigym/src/nkigym/transforms/software_pipeline.py` — `SoftwarePipeline`
  transform + `SoftwarePipelineOption`. The transform's own legality (TVM's
  two graph rules) + version derivation + annotation write.
- `test/transforms/test_software_pipeline.py` — unit tests (analyze,
  legality, apply, byte-exact, sim).

**Modified files:**
- `nkigym/src/nkigym/ir/tree.py` — `Buffer.versions: int = 1` field;
  `physical_shape()` multiplies the tile dim by `versions`.
- `nkigym/src/nkigym/codegen/body.py` — `render_buffer_region` injects the
  `(loop_var % versions)` tile-axis rotation; `_emit_subtree` emits
  prologue/steady-body/epilogue for a pipeline-annotated ForNode.
- `nkigym/src/nkigym/transforms/__init__.py` — export `SoftwarePipeline`,
  `SoftwarePipelineOption`.
- `kernel_transforms.py` — add `kernel_15` (the validated `rotate_only`
  MVP kernel, Increment 1 oracle) and `kernel_16` (the validated `skewed`
  kernel, Increment 2 oracle). Both bodies come from the HW-validated
  `examples/handwritten_pipeline_matmul.py` builders.
- `examples/tune_matmul_lhsT_rhs.py` — append the pipeline atom to `TRACE`.
- `test/ir/test_node_labels.py` — `Buffer.versions` physical_shape units.

**Two increments (each ships a HW-validated milestone):**

- **Increment 1 — MVP rotation (Tasks 1-5):** `Buffer.versions` + index
  rotation + a `SoftwarePipeline` transform that grows the accumulator and
  rotates the index on the *existing monolithic loop* (no structural skew).
  Renders to `kernel_15` (`rotate_only`), HW-validated at 82.9% MFU. The
  compiler auto-pipelines the loop once the WAR is gone.
- **Increment 2 — skew renderer (Tasks 6-9):** add the
  prologue/skewed-body/epilogue emitter (per-stage `loop_var := loop_var − s`
  substitution through all of a block's regions, literal-bound lead-in/out).
  Renders to `kernel_16` (`skewed`), the faithful TVM port, HW-validated at
  83.7%. End state.

The transform's `apply` is the same in both increments (derive versions,
write annotation); only the *renderer's response to the annotation* changes
— monolithic-with-rotation in Inc 1, 3-phase in Inc 2. The annotation schema
is identical, so Inc 1 options remain valid; Inc 2 only widens what the
renderer does with them.

**Annotation schema** (stored in `BlockNode.annotations`, keyed on the
pipelined loop's nid so a block can host more than one pipelined loop):
```python
block.annotations["software_pipeline"] = {
    "loop_nid": int,            # the pipelined ForNode
    "stages": tuple[int, ...],  # stage per child block, child order
    "order": tuple[int, ...],   # emission order per child block
}
```

**Key verified facts (from code, not assumed):**
- `Mod(left, right)` Expr node exists (`arith/expr.py:61`); `right` must be a
  non-zero `Const` for affinity. `i_d1_0 % 2` → `Mod(Var("i_d1_0"), Const(2))`,
  formats as `i_d1_0 % 2`. Affine-safe (divisor is `Const(2)`).
- `render_buffer_region(region, buf) -> str` (`body.py:96`): for sbuf/psum,
  axis-0 emits `0:128` + the tile-axis `lo` via `format_expr`. The rotation
  is added to that tile-axis `lo`.
- `_emit_subtree(ir, nid, depth, code) -> None` (`body.py:51`): ForNode
  branch at `body.py:60-63` emits one `for`; this is where the 3-phase
  branch goes.
- `Dependency.must_precede(producer, consumer) -> bool` (`dependency.py:87`),
  keyed on ISA-leaf nids; `_resolve` maps a block nid → its leaf. A loop's
  child blocks: `[c for c in ir.tree.children(loop_nid) if isinstance(ir.tree.data(c), BlockNode)]`;
  a block's leaves: `[d for d in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(d), ISANode)]`.
- Transform shape (`reorder.py`): `apply` = `_check_legality` →
  `copy.deepcopy(ir)` → mutate → `new_ir.dependency = Dependency(new_ir.tree)`.
- Oracle: `assert_matches_hand(render(ir), KT.kernel_N)` (`_ladder_compare.py:237`)
  AST-canonicalizes both and asserts equal; `KT.kernel_N` is a `@nki.jit`
  function in `kernel_transforms.py` (currently `kernel_0..kernel_14`).
- Tuned state: `build_ladder_state(n)` (`_fixtures.py:126`) replays rungs;
  but the example's `TRACE` is the canonical-nid path. Tests discover the
  M-loop + child blocks by preorder/op-type match (nids are reassigned after
  Reorder/ComputeAt).

---

### Task 1: `Buffer.versions` field grows the tile dim

The version count materialized on the Buffer so `physical_shape()` (the
single source of truth) reflects the grown allocation. `versions=1` must be
byte-identical to today.

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (the `Buffer` dataclass ~line 111,
  `physical_shape` ~line 130)
- Test: `test/ir/test_node_labels.py` (already imports `Buffer`)

- [ ] **Step 1: Write the failing test**

Add to `test/ir/test_node_labels.py`:

```python
def test_buffer_versions_default_unchanged():
    """versions defaults to 1 and leaves physical_shape identical to today."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    assert buf.versions == 1
    assert buf.physical_shape() == (128, 1, 2048)


def test_buffer_versions_grows_tile_dim():
    """versions=2 doubles the tile (middle) dim for sbuf/psum."""
    psum = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    assert psum.physical_shape() == (128, 2, 2048)
    sbuf = Buffer(name="sbuf_x", shape=(256, 512), dtype="bfloat16", location="sbuf", versions=2)
    assert sbuf.physical_shape() == (128, 4, 512)


def test_buffer_versions_hbm_unaffected():
    """shared_hbm keeps its 2D logical shape regardless of versions."""
    hbm = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", versions=2)
    assert hbm.physical_shape() == (2048, 2048)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_node_labels.py -k versions -v`
Expected: FAIL — `Buffer.__init__() got an unexpected keyword argument 'versions'`.

- [ ] **Step 3: Add the field and grow the tile dim**

In `nkigym/src/nkigym/ir/tree.py`, add to the `Buffer` dataclass (after the
`location: str` field, ~line 128):

```python
    location: str
    versions: int = 1
    """Pipeline buffer-version count. 1 = single instance (renders
    byte-identically to today). >1 multiplies the tile (middle) dim of
    physical_shape so the renderer's ``loop_var % versions`` rotation
    addresses distinct slots. Set by SoftwarePipeline (use_stage − def_stage
    + 1); left 1 everywhere else."""
```

Then in `physical_shape`, change the sbuf/psum return (the last line, ~line
148) from:

```python
        return (PARTITION_DIM, leading // PARTITION_DIM, free)
```

to:

```python
        return (PARTITION_DIM, (leading // PARTITION_DIM) * self.versions, free)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_node_labels.py -k versions -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full IR + codegen suite (regression — versions=1 unchanged)**

Run: `pytest test/ir/ test/codegen/ -q`
Expected: all PASS — no existing test constructs a Buffer with `versions>1`,
so every physical_shape stays identical.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "feat(ir): Buffer.versions grows tile dim for pipeline multi-buffering"
```

---

### Task 2: Renderer injects the `loop_var % versions` tile-axis rotation

When a buffer has `versions > 1`, every access inside the pipelined loop must
index slot `(loop_var % versions)`. The rotation is added to the tile-axis
`lo` term. This task adds a pure helper and wires it into
`render_buffer_region`; the 3-phase loop emission is Task 3.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`render_buffer_region` ~line 96)
- Test: `test/codegen/test_body.py`

- [ ] **Step 1: Write the failing test**

Add to `test/codegen/test_body.py` (imports `Buffer`, `BufferRegion`,
`render_buffer_region`, `Const`, `Var`, `Mod` as needed).

> **CRITICAL — regions are 2-range, NOT 3-range.** An SBUF/PSUM
> `BufferRegion` carries **one range per LOGICAL operand axis** (verified:
> `canonical_build._build_region` iterates `present_axes`; existing
> `test_render_psum_3d_region_partition_axis_split` uses a 2-tuple region).
> The renderer SPLITS axis-0 into `0:128` + the tile-coord — the physical 3D
> `(128, num_p_tiles, F)` shape is the ALLOCATION shape, not the region range
> count. So the rotation attaches to axis-0's tile-coord. A 3-range region
> would wrongly render a 4-component slice. Use 2 ranges:

```python
def test_render_region_rotation_applied():
    """A versions>1 psum buffer rotates the tile-axis (axis-0) index by loop_var % versions."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum", versions=2)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048))),
    )
    out = render_buffer_region(region, buf, rotation=Mod(left=Var(name="i_d1_0"), right=Const(value=2)))
    assert out == "psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048]"


def test_render_region_no_rotation_when_versions_one():
    """versions=1 (rotation=None) renders byte-identically to today."""
    buf = Buffer(name="psum_prod", shape=(128, 2048), dtype="float32", location="psum")
    region = BufferRegion(
        tensor="psum_prod",
        ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=2048))),
    )
    assert render_buffer_region(region, buf, rotation=None) == "psum_prod[0:128, 0, 0:0 + 2048]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_body.py -k rotation -v`
Expected: FAIL — `render_buffer_region() got an unexpected keyword argument 'rotation'`.

- [ ] **Step 3: Add the `rotation` parameter**

In `nkigym/src/nkigym/codegen/body.py`, change `render_buffer_region`'s
signature and the tile-axis emission. The current axis-0 branch appends
`format_expr(lo)` as the tile index (line 104); add the rotation to that
`lo`. New version:

```python
def render_buffer_region(region: BufferRegion, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor.

    ``rotation`` (when not None) is an affine ``Expr`` added to the
    SBUF/PSUM tile-axis index — the ``loop_var % versions`` term that
    selects a pipeline buffer version. Ignored for ``shared_hbm`` (no tile
    axis) and when None (single-version buffers, byte-identical to before).
    """
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            parts.append(f"0:{PARTITION_DIM}")
            parts.append(_format_tile_index(lo, rotation))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}[{', '.join(parts)}]"
```

Add the helper (and the reason it cannot use `format_expr` on the combined
expression):

```python
def _format_tile_index(lo: Expr, rotation: Expr | None) -> str:
    """Render the SBUF/PSUM tile-axis index, optionally + a version rotation.

    ``format_expr`` normalises through ``to_affine``, which RAISES
    ``NonAffineError`` on ``Mod(Var, Const)`` (a version rotation like
    ``i_d1_0 % 2``) — the modulo of a variable is not affine. So the rotation
    is rendered with the non-normalising ``_format_raw`` and combined with the
    (affine) ``lo`` here, dropping ``lo`` when it is the rebased ``Const(0)``.
    """
    if rotation is None:
        return format_expr(lo)
    rot_str = _format_raw(rotation)
    if isinstance(lo, Const) and lo.value == 0:
        return rot_str
    return f"{format_expr(lo)} + {rot_str}"
```

**Imports** at the top of `body.py`: add `Expr` and `_format_raw` from
`nkigym.ir.arith.expr` (`Const`, `format_expr` already imported; `Add` is NOT
needed — the separate-format approach avoids building an `Add` node). FIRST
read the import block and add only what is missing.

> **Why not `Add(lo, rotation)` + `format_expr`?** `format_expr(Add(Const(0),
> Mod(Var, Const)))` routes through `to_affine`, whose `Mod` branch raises
> `NonAffineError` when the modulo's left operand is a `Var` (verified
> `arith/expr.py:166-167`). The tile rotation is intrinsically non-affine, so
> it must bypass affine normalisation. `_format_raw` (`expr.py:335`) renders
> `Mod` directly as `left % right`. This same `_format_tile_index` helper is
> reused by Tasks 3/6/7.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/codegen/test_body.py -k rotation -v`
Expected: PASS (2 tests). `_format_tile_index(Const(0), Mod(Var("i_d1_0"),
Const(2)))` → `"i_d1_0 % 2"` (lo==0 dropped); `rotation=None` →
`format_expr(Const(0))` → `"0"` (byte-identical to today).

- [ ] **Step 5: Update the existing caller**

`_emit_isa_call` (~line 90) calls `render_buffer_region(rebased_region(...), buf)`.
The new `rotation` param defaults to `None`, so this caller is unchanged and
keeps rendering single-version buffers identically. Verify:

Run: `pytest test/codegen/ -q`
Expected: all PASS (rotation defaults to None everywhere).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "feat(codegen): render_buffer_region optional pipeline version rotation"
```

---

> **Design note (render-time rotation, not region-baking).** TVM's
> `PipelineBodyRewriter` bakes `floormod(loop_var, versions)*offset` into each
> buffer access. We instead compute the rotation at render time and leave the
> IR `BufferRegion`s clean. Rationale: Increment 2's skew substitution
> (`loop_var := loop_var − s`, with the *same* region emitted at multiple
> literal prologue/epilogue iterations) is *inherently* a render-time concern —
> it cannot be baked into a single region. Keeping the rotation render-time too
> means one mechanism, not two, and avoids the region-mutation drift the
> learnings flag against (`rebased_region` / `compact`).

### Task 3: Renderer threads rotation context from the pipeline annotation (Increment 1)

The renderer reads the pipeline annotation on a loop's parent block, and for
every buffer with `versions > 1` accessed inside that loop, injects
`(loop_var % versions) * num_p_tiles` as the tile-axis rotation. Increment 1
emits the loop **monolithically** (no skew) — this renders the validated
`rotate_only` kernel.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`_emit_block`, `_emit_subtree`,
  `_emit_isa_call` — thread a `rotations` context dict)
- Test: `test/codegen/test_body.py`

- [ ] **Step 1: Write the failing test**

The cleanest unit test renders the tuned IR after the transform is applied —
but the transform doesn't exist until Task 4. So this task's test is a
**focused renderer test** using a hand-built annotation, deferring the
end-to-end byte-exact gate to Task 5. First create a shared test-helper
module `test/transforms/_pipeline_fixtures.py` so both this codegen test and
Task 4's transform tests import the SAME helpers (DRY):

```python
"""Shared fixtures: tuned-matmul IR + M-loop/child discovery for pipeline tests."""

from __future__ import annotations

from nkigym.environment import KernelMDP
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import ComputeAt, Fuse, Reorder, ReverseComputeAt, Split
from examples.tune_matmul_lhsT_rhs import INPUT_SPECS, TRACE, f_nkigym


def tuned_ir():
    """Replay the example TRACE to the tuned (M>N>K, sunk memset+drain) state."""
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()
    for action in TRACE:
        state = env.step(state, action)
    return state


def m_loop_and_children(ir):
    """The i_d1_0 ForNode enclosing the matmul leaf, and ITS DIRECT CHILDREN
    in order — the stageable units. CRITICAL: these are ALL direct children
    (BlockNode OR ForNode), NOT BlockNode-filtered. The tuned M loop body is
    ``[memset-block, matmul-loopnest(ForNode), drain-block]`` = 3 units; the
    matmul is a nested loop subtree, not a sibling block. Matches TVM's
    one-stage-per-SeqStmt-child model."""
    mm_leaf = next(n for n in ir.tree.preorder()
                   if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul")
    m_loop = next(a for a in ir.tree.ancestors(mm_leaf)
                  if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d1_0")
    children = list(ir.tree.children(m_loop))
    return m_loop, children


def parent_block_of(ir, loop_nid):
    """Nearest enclosing BlockNode of a loop (for writing the annotation in tests)."""
    result = ir.tree.root
    for anc in reversed(ir.tree.ancestors(loop_nid)):
        if isinstance(ir.tree.data(anc), BlockNode):
            result = anc
            break
    return result
```

> **NOTE:** `reversed(...)` assumes `ancestors` is farthest-first (root →
> leaf). If it is nearest-first, drop the `reversed`. Verify against
> `tree.py:306` and the working `_parent_block` in Task 4 (they must agree).

Then add to `test/codegen/test_body.py`:

```python
from test.transforms._pipeline_fixtures import tuned_ir, m_loop_and_children, parent_block_of


def test_emit_pipeline_annotation_rotates_monolithic_loop():
    """A loop whose parent block carries a software_pipeline annotation emits
    a monolithic loop with every versions>1 access rotated by loop_var % versions."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)  # frozen; Task 4 does this properly
    parent = parent_block_of(ir, m_loop)
    ir.tree.data(parent).annotations["software_pipeline"] = {
        "loop_nid": m_loop, "stages": (0, 0, 1), "order": (0, 1, 2),
    }
    src = render(ir)
    assert "psum_prod = nl.ndarray((128, 2, 2048)" in src
    assert "psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048]" in src
    assert "for i_d1_0 in range(16):" in src  # monolithic, no prologue
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_body.py -k pipeline_annotation -v`
Expected: FAIL — the rotation is not applied (no rotation context threaded),
so the access renders as `psum_prod[0:128, 0, 0:0 + 2048]`.

- [ ] **Step 3: Thread the rotation context**

In `nkigym/src/nkigym/codegen/body.py`:

(a) Add a helper that builds the rotation `Expr` for a buffer under a loop:

```python
def _version_rotation(buf: Buffer, loop_var: str) -> Expr | None:
    """Return the tile-axis version rotation for a multi-version buffer, or
    None for single-version. ``num_p_tiles`` (the per-version tile span) is the
    middle physical dim divided by versions. When ``num_p_tiles == 1`` the
    rotation is the bare ``loop_var % versions`` (NO ``* 1`` — the validated
    kernel renders ``i_d1_0 % 2``, not ``i_d1_0 % 2 * 1``); only a >1 span
    wraps in ``Mul(..., Const(num_p_tiles))``."""
    if buf.versions <= 1:
        return None
    mod = Mod(left=Var(name=loop_var), right=Const(value=buf.versions))
    num_p_tiles = buf.physical_shape()[1] // buf.versions
    return mod if num_p_tiles == 1 else Mul(left=mod, right=Const(value=num_p_tiles))
```

> **NOTE:** `_format_raw` renders `Mul(Mod(...), Const(k))` as
> `loop_var % versions * k` — correct precedence for `k > 1`. The bare-`Mod`
> branch avoids the `* 1` that would break the byte-exact gate against the
> validated kernel (`num_p_tiles == 1` for the full-width matmul accumulator).

(b) Thread a `rotations: dict[str, Expr]` (tensor name → rotation Expr) param,
defaulting to `{}`, through `_emit_block`, `_emit_subtree`, and `_emit_isa_call`.
When `_emit_subtree` hits a ForNode that is the `loop_nid` of a
`software_pipeline` annotation on any enclosing block, it extends `rotations`
for every versions>1 buffer touched in the subtree, keyed by tensor name,
then recurses (monolithically for Increment 1). `_emit_isa_call` passes
`rotations.get(region.tensor)` as the `rotation` arg to `render_buffer_region`.

The annotation lookup: a loop is pipelined iff some block in the tree has
`annotations.get("software_pipeline", {}).get("loop_nid") == nid`. Build a
`{loop_nid: annotation}` map once at `emit_body` entry and pass it down.

(c) Add imports to `body.py`: `from nkigym.ir.arith.expr import Const, Mod, Mul, Var`
(add only those not already imported).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/codegen/test_body.py -k pipeline_annotation -v`
Expected: PASS — monolithic loop, rotated psum access, grown alloc.

- [ ] **Step 5: Regression — un-annotated kernels unchanged**

Run: `pytest test/codegen/ test/transforms/ -q`
Expected: all PASS — with no annotation, `rotations` stays `{}` and every
access renders identically.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "feat(codegen): thread pipeline rotation context for monolithic loops"
```

---

### Task 4: `SoftwarePipeline` transform — analyze, legality, apply

The transform itself: enumerate non-decreasing stage labelings, validate
TVM's two graph rules, and on apply derive per-buffer versions + write the
annotation. `apply` is identical across both increments — only the renderer's
response differs.

**Files:**
- Create: `nkigym/src/nkigym/transforms/software_pipeline.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`
- Test: `test/transforms/test_software_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `test/transforms/test_software_pipeline.py`:

```python
"""Tests for nkigym.transforms.SoftwarePipeline (Tier B)."""

from __future__ import annotations

import pytest

from nkigym.transforms import SoftwarePipeline, SoftwarePipelineOption, TransformLegalityError
from test.transforms._pipeline_fixtures import tuned_ir, m_loop_and_children


def test_analyze_enumerates_nondecreasing_labelings():
    """The tuned M loop yields exactly the contiguous non-decreasing stage labelings."""
    ir = tuned_ir()
    opts = SoftwarePipeline().analyze(ir)
    m_loop, children = m_loop_and_children(ir)
    stage_sets = {o.stages for o in opts if o.loop_nid == m_loop}
    assert (0, 0, 1) in stage_sets
    assert (0, 1, 1) in stage_sets
    assert (0, 1, 2) in stage_sets
    assert (0, 0, 0) not in stage_sets  # single-stage no-op filtered
    assert all(max(s) <= len(children) - 1 for s in stage_sets)  # derived bound


def test_apply_derives_versions_and_annotates():
    """apply((0,0,1)) sets psum versions=2 and writes the annotation; tree unchanged."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    n_nodes_before = ir.tree.graph.number_of_nodes()
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert new_ir.buffer("psum_prod").versions == 2
    assert new_ir.tree.graph.number_of_nodes() == n_nodes_before  # structural no-op
    anns = [new_ir.tree.data(nid).annotations.get("software_pipeline") for nid in new_ir.tree.blocks()]
    assert any(a and a["stages"] == (0, 0, 1) for a in anns)


def test_apply_rejects_consumer_before_producer_stage():
    """A stage assignment putting a consumer earlier than its producer raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(1, 0, 1), order=(0, 1, 2)))


def test_apply_rejects_duplicate_order():
    """An order array that is not a permutation raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 1)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/transforms/test_software_pipeline.py -v`
Expected: FAIL at import — `cannot import name 'SoftwarePipeline'`.

- [ ] **Step 3: Implement the transform**

Create `nkigym/src/nkigym/transforms/software_pipeline.py`:

```python
"""``SoftwarePipeline`` transform — assign a loop's child blocks to pipeline
stages, deriving per-buffer version counts (Tier B: stage only, identity order).

Faithful port of TVM ``InjectSoftwarePipeline`` (``src/s_tir/transform/
inject_software_pipeline.cc``). ``apply`` derives versions and writes an
annotation; the prologue/skewed-body/epilogue + ``% versions`` rotation are
manifested by the renderer."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class SoftwarePipelineOption(TransformOption):
    """Pipeline ``loop_nid``'s child blocks across stages.

    Attributes:
        loop_nid: the ForNode whose child blocks are staged.
        stages: stage index per child block, in child order. Full
            assignment (one entry per child); non-decreasing along the
            dependency chain (Tier B).
        order: emission order per child block. Tier B: identity.
    """

    loop_nid: int
    stages: tuple[int, ...]
    order: tuple[int, ...]


class SoftwarePipeline(Transform):
    """Stage-driven accumulator multi-buffer (Tier B)."""

    def analyze(self, ir: KernelIR) -> list[SoftwarePipelineOption]:
        """Enumerate non-decreasing stage labelings for every pipelineable loop.

        Stageable units = ALL direct children of the loop (BlockNode OR
        ForNode subtree), one stage per unit — TVM's per-SeqStmt-child model.
        A loop is pipelineable iff it has >= 2 direct children and is not
        already pipelined."""
        options: list[SoftwarePipelineOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            children = list(ir.tree.children(nid))
            if len(children) < 2 or self._already_pipelined(ir, nid):
                continue
            n = len(children)
            order = tuple(range(n))
            for stages in self._nondecreasing_labelings(n):
                opt = SoftwarePipelineOption(loop_nid=nid, stages=stages, order=order)
                if self._is_legal(ir, opt, children):
                    options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: SoftwarePipelineOption) -> KernelIR:
        """Re-check legality, deep-copy, derive versions, write annotation."""
        children = list(ir.tree.children(option.loop_nid))
        self._check_legality(ir, option, children)
        new_ir = copy.deepcopy(ir)
        new_children = list(new_ir.tree.children(option.loop_nid))
        self._apply_versions(new_ir, option, new_children)
        parent = self._parent_block(new_ir, option.loop_nid)
        new_ir.tree.data(parent).annotations["software_pipeline"] = {
            "loop_nid": option.loop_nid, "stages": option.stages, "order": option.order,
        }
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _nondecreasing_labelings(self, n: int) -> list[tuple[int, ...]]:
        """All contiguous non-decreasing stage labelings of n blocks with
        stage 0 present and no gap (max_stage <= n-1), excluding all-zero."""
        out: list[tuple[int, ...]] = []
        for combo in itertools.product(range(n), repeat=n):
            if combo[0] != 0:
                continue
            if any(combo[i + 1] < combo[i] for i in range(n - 1)):
                continue
            if any(combo[i + 1] - combo[i] > 1 for i in range(n - 1)):
                continue  # no empty stage
            if max(combo) == 0:
                continue  # single-stage no-op
            out.append(combo)
        return out

    def _unit_leaves(self, ir: KernelIR, unit_nid: int) -> list[int]:
        """ISA-leaf nids inside a stageable unit (a direct loop child). Works
        for both a BlockNode child and a ForNode-nest child (e.g. the matmul
        loop nest). Includes ``unit_nid`` itself if it is an ISA leaf."""
        candidates = [unit_nid, *ir.tree.descendants(unit_nid)]
        return [d for d in candidates if isinstance(ir.tree.data(d), ISANode)]

    def _is_legal(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> bool:
        """TVM's two graph rules + order permutation. True iff legal."""
        result = True
        if len(option.stages) != len(children) or len(option.order) != len(children):
            result = False
        elif sorted(option.order) != list(range(len(children))):
            result = False  # not a permutation
        else:
            stage_of = {children[i]: option.stages[i] for i in range(len(children))}
            order_of = {children[i]: option.order[i] for i in range(len(children))}
            for src_b in children:
                for dst_b in children:
                    if src_b is dst_b:
                        continue
                    dep = any(ir.dependency.must_precede(ls, ld)
                              for ls in self._unit_leaves(ir, src_b)
                              for ld in self._unit_leaves(ir, dst_b))
                    if not dep:
                        continue
                    if stage_of[src_b] > stage_of[dst_b]:
                        result = False
                    elif stage_of[src_b] == stage_of[dst_b] and order_of[src_b] >= order_of[dst_b]:
                        result = False
        return result

    def _check_legality(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Raise TransformLegalityError if illegal."""
        if not self._is_legal(ir, option, children):
            raise TransformLegalityError(f"illegal software-pipeline option {option}")

    def _apply_versions(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Set Buffer.versions = use_stage - def_stage + 1 for each buffer the
        staged units touch. Reads/writes per ISA leaf come from
        ``ir.dependency.info(leaf)`` (frozensets of tensor names) — NOT
        ``block.writes``/``block.reads``, because a ForNode-nest unit (the
        matmul) has no enclosing BlockNode carrying regions; the regions live
        on the ISA leaf. ``def`` = min stage among units writing a buffer,
        ``use`` = max stage among units reading it."""
        stage_of = {children[i]: option.stages[i] for i in range(len(children))}
        defs: dict[str, int] = {}
        uses: dict[str, int] = {}
        for unit_nid in children:
            st = stage_of[unit_nid]
            reads: set[str] = set()
            writes: set[str] = set()
            for leaf in self._unit_leaves(ir, unit_nid):
                info = ir.dependency.info(leaf)
                reads |= set(info.reads)
                writes |= set(info.writes)
            for name in writes:
                defs[name] = min(defs.get(name, st), st)
            for name in reads:
                uses[name] = max(uses.get(name, st), st)
        for name in set(defs) & set(uses):
            versions = uses[name] - defs[name] + 1
            if versions > 1:
                self._set_versions(ir, name, versions)

    def _set_versions(self, ir: KernelIR, name: str, versions: int) -> None:
        """Replace the buffer's owning alloc with a versions-updated copy."""
        from dataclasses import replace
        for nid in ir.tree.blocks():
            block = ir.tree.data(nid)
            assert isinstance(block, BlockNode)
            new_allocs = tuple(replace(b, versions=versions) if b.name == name else b for b in block.alloc_buffers)
            if new_allocs != block.alloc_buffers:
                ir.tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=new_allocs)

    def _parent_block(self, ir: KernelIR, loop_nid: int) -> int:
        """Nearest enclosing BlockNode of the loop."""
        cur = loop_nid
        result = ir.tree.root
        found = False
        for anc in ir.tree.ancestors(loop_nid):
            if isinstance(ir.tree.data(anc), BlockNode) and not found:
                result = anc
                found = True
        return result

    def _already_pipelined(self, ir: KernelIR, loop_nid: int) -> bool:
        """True if some block already annotates this loop as pipelined."""
        return any(ir.tree.data(nid).annotations.get("software_pipeline", {}).get("loop_nid") == loop_nid
                   for nid in ir.tree.blocks())
```

> **NOTE for implementer:** `_parent_block` and `_set_versions` use
> single-return-friendly loops per the codebase style (build a result var).
> Verify `ir.tree.ancestors` returns nearest-first or farthest-first and
> adjust `_parent_block` accordingly — pick the *nearest* enclosing block.
> If `ancestors` is farthest-first, take the last match instead of guarding
> with `found`.

- [ ] **Step 4: Export the transform**

In `nkigym/src/nkigym/transforms/__init__.py`, add `SoftwarePipeline` and
`SoftwarePipelineOption` to the imports and `__all__` (match the existing
export pattern for `Reorder`/`ReorderOption`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest test/transforms/test_software_pipeline.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/software_pipeline.py nkigym/src/nkigym/transforms/__init__.py test/transforms/test_software_pipeline.py
git commit -m "feat(transforms): SoftwarePipeline analyze + legality + version derivation"
```

---

### Task 5: Increment 1 byte-exact + sim gate against `kernel_15`

Add the validated `rotate_only` kernel as `kernel_15` and gate
`render(apply(tuned, (0,0,1)))` against it byte-exact, plus fp32 sim.

**Files:**
- Modify: `kernel_transforms.py` (add `kernel_15`)
- Test: `test/transforms/test_software_pipeline.py` (byte-exact + sim)

- [ ] **Step 1: Add `kernel_15` and the shared test helpers**

Append to `kernel_transforms.py` (after `kernel_14`, before `kernel_partial`),
the exact body of `build_rotate_only()` from
`examples/handwritten_pipeline_matmul.py` (HW-validated 82.9%):

```python
@nki.jit
def kernel_15(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(src=lhs_T[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0:0 + 2048])
    for i_d0_0 in range(16):
        nisa.dma_copy(src=rhs[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0:0 + 2048])
    psum_prod = nl.ndarray((128, 2, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048], value=0.0)
        for i_d2_0 in range(4):
            for i_d0_0 in range(16):
                nisa.nc_matmul(stationary=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128], moving=sbuf_rhs[0:128, i_d0_0, i_d2_0 * 512:i_d2_0 * 512 + 512], dst=psum_prod[0:128, i_d1_0 % 2, i_d2_0 * 512:i_d2_0 * 512 + 512])
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0:0 + 2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(src=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], dst=hbm_out[i_d1_0 * 128:i_d1_0 * 128 + 128, 0:0 + 2048])
    return hbm_out
```

> **VERIFY before trusting:** this body must equal `build_rotate_only()`'s
> output. Run `python -c "from examples.handwritten_pipeline_matmul import build_rotate_only; print(build_rotate_only())"`
> and diff against the function body above (modulo the `@nki.jit`/`def` lines).

Add the shared helpers to `test/transforms/test_software_pipeline.py` (used
by Task 3's renderer test too — keep them in this one module and import):

```python
import kernel_transforms as KT
from test.transforms._ladder_compare import assert_matches_hand
from nkigym.codegen import render
```

- [ ] **Step 2: Write the failing byte-exact + sim test**

Add to `test/transforms/test_software_pipeline.py`:

```python
def test_increment1_matches_kernel_15_byte_exact():
    """render(apply(tuned, (0,0,1))) reproduces the validated rotate_only kernel."""
    ir = tuned_ir()
    m_loop, _ = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert_matches_hand(render(new_ir), KT.kernel_15)
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest test/transforms/test_software_pipeline.py -k kernel_15 -v`
Expected: FAIL — the rendered source differs (rotation context wired in
Task 3 must now flow through the real annotation path).

- [ ] **Step 4: Make it pass**

This is the integration point of Tasks 1-4. If it fails, the likely gaps are:
(a) the rotation context isn't keyed correctly in `_emit_subtree`, or (b)
`num_p_tiles` math. Fix in `body.py` until the byte-exact oracle passes. Do
NOT weaken the oracle.

Run: `pytest test/transforms/test_software_pipeline.py -k kernel_15 -v`
Expected: PASS.

- [ ] **Step 5: Sim-validate on CPU (numeric correctness)**

The byte-exact gate proves structure; add a fp32 sim check that the kernel
computes `lhs_T.T @ rhs`. Reuse the example's `_sim_check` pattern — add a
test that renders, writes to a temp file, imports, and runs `simulate_fp32`
against the numpy golden (copy the helper from
`examples/tune_matmul_lhsT_rhs.py:136-152`, `atol=rtol=5e-3`).

Run: `pytest test/transforms/test_software_pipeline.py -k sim -v`
Expected: PASS.

- [ ] **Step 6: Full suite + commit**

Run: `pytest test/ -q`
Expected: all PASS.

```bash
git add kernel_transforms.py test/transforms/test_software_pipeline.py
git commit -m "feat: SoftwarePipeline Increment 1 (rotate-only) byte-exact + sim vs kernel_15"
```

- [ ] **Step 7: HW validation gate (Increment 1 milestone)**

Append the pipeline atom to `examples/tune_matmul_lhsT_rhs.py`'s `TRACE`
(after discovering the M-loop nid in that script's deterministic post-TRACE
state — add a 5th atom `(SoftwarePipeline(), SoftwarePipelineOption(...))`),
then profile on Trn2:

```bash
transport/kaizen.sh --name default --cmd "python examples/tune_matmul_lhsT_rhs.py" --cache /home/weittang/cache/tune_matmul
```

Expected: tuned kernel MFU ≈ 82-83% (vs 59% before), `tensor_engine_active_time_percent`
≈ 87-88% (read from `tuned/profile_summary.json`). **Milestone: Increment 1
ships the validated MVP win.**

---

## Increment 2 — Skew renderer (the faithful TVM port)

Increment 1 emits the pipelined loop monolithically and lets neuronx-cc
overlap it. Increment 2 makes the renderer emit the explicit
prologue / skewed-body / epilogue — the faithful `InjectSoftwarePipeline`
structure — rendering the validated `skewed` kernel (`kernel_16`, 83.7%).

**The core mechanism — per-stage iteration substitution.** A block at stage
`s` logically runs `s` iterations behind the loop counter. In the steady body
`for i in range(max_stage, extent)`, a stage-`s` block emits with every
occurrence of the loop var substituted `loop_var := loop_var - s`. In the
prologue/epilogue it emits with the loop var bound to a literal integer. This
substitution must flow through **all** of the block's regions (both the
versioned psum index AND the `sbuf_prod[i-1]` write), via
`nkigym.ir.arith.expr.substitute`.

### Task 6: Stage-substitution emitter for a single block at an iteration

A pure helper: render one child block's leaves with the loop var replaced by
a given `Expr` (a `Var - s` for the body, or a `Const` literal for
prologue/epilogue), and the version rotation recomputed on the substituted
loop var.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py`
- Test: `test/codegen/test_body.py`

- [ ] **Step 1: Write the failing tests**

The emitter must handle BOTH a flat block (drain: one ISA leaf) AND a
ForNode-nest unit (the matmul: inner `i_d2_0`/`i_d0_0` loops must still emit
their `for` lines, with only the PIPELINED `i_d1_0` substituted). Two tests:

```python
def test_emit_unit_at_iteration_flat_block_substitutes_loop_var():
    """The drain block at iteration (i_d1_0 - 1) rewrites BOTH its psum read
    index (with version rotation) and its sbuf write index by the substitution."""
    ir = tuned_ir()
    m_loop, children = m_loop_and_children(ir)
    drain_unit = children[-1]  # tensor_copy block is the last direct child
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    code: list[str] = []
    _emit_unit_at_iteration(ir, drain_unit, depth=1, code=code, loop_var="i_d1_0",
                            iteration=Sub(left=Var(name="i_d1_0"), right=Const(value=1)),
                            versions={"psum_prod": 2})
    body = "\n".join(code)
    assert "psum_prod[0:128, (i_d1_0 - 1) % 2, 0:0 + 2048]" in body
    assert "sbuf_prod[0:128, i_d1_0 - 1, 0:0 + 2048]" in body


def test_emit_unit_at_iteration_nested_loops_preserved():
    """The matmul-nest unit at iteration i_d1_0 (stage 0) keeps its inner
    d2/d0 for-loops and rotates psum; only i_d1_0 is substituted (here identity)."""
    ir = tuned_ir()
    m_loop, children = m_loop_and_children(ir)
    matmul_unit = children[1]  # the ForNode nest (i_d2_0 -> i_d0_0 -> matmul)
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    code: list[str] = []
    _emit_unit_at_iteration(ir, matmul_unit, depth=1, code=code, loop_var="i_d1_0",
                            iteration=Var(name="i_d1_0"), versions={"psum_prod": 2})
    body = "\n".join(code)
    assert "for i_d2_0 in range(4):" in body
    assert "for i_d0_0 in range(16):" in body
    assert "psum_prod[0:128, i_d1_0 % 2, i_d2_0 * 512:i_d2_0 * 512 + 512]" in body
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest test/codegen/test_body.py -k at_iteration -v`
Expected: FAIL — `_emit_unit_at_iteration` not defined.

- [ ] **Step 3: Implement the recursive substitution emitter**

It mirrors `_emit_subtree` but (a) substitutes `subs = {loop_var: iteration}`
into every region's `lo`, (b) applies the version rotation per buffer, (c)
**recurses through inner ForNode/BlockNode children** so a nested matmul loop
nest still emits its `for` lines. Only the pipelined `loop_var` is substituted;
inner loop vars (`i_d2_0`, `i_d0_0`) are untouched.

```python
def _emit_unit_at_iteration(ir, nid, depth, code, loop_var, iteration, versions):
    """Emit a stageable unit (a direct loop child: a BlockNode, a ForNode
    nest, or a bare ISA leaf) with the pipelined ``loop_var`` replaced by
    ``iteration`` (``Var - s`` for the steady body; a ``Const`` for
    prologue/epilogue) in every region, and the per-buffer version rotation
    recomputed on ``iteration``. Recurses through inner ForNodes/BlockNodes;
    inner loop vars are left intact.

    ``versions``: ``{tensor_name: version_count}`` for the buffers this
    pipeline grew (>1). Single-version buffers get no rotation.
    """
    subs = {loop_var: iteration}
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        code.append(_INDENT * depth + f"for {node.loop_var} in range({node.extent}):")
        for child in ir.tree.children(nid):
            _emit_unit_at_iteration(ir, child, depth + 1, code, loop_var, iteration, versions)
    elif isinstance(node, BlockNode):
        for child in ir.tree.children(nid):
            _emit_unit_at_iteration(ir, child, depth, code, loop_var, iteration, versions)
    elif isinstance(node, ISANode):
        parts: list[str] = []
        for slot in node.op_cls.OPERAND_AXES:
            if slot in node.operand_bindings:
                region = node.operand_bindings[slot]
                buf = ir.buffer(region.tensor)
                based = rebased_region(region, buf, ir.tree)
                shifted = BufferRegion(
                    tensor=based.tensor,
                    ranges=tuple((substitute(lo, subs), width) for lo, width in based.ranges),
                )
                rot = _rotation_on(buf, iteration, versions)
                parts.append(f"{slot}={render_buffer_region(shifted, buf, rotation=rot)}")
        for k, v in node.kwargs.items():
            parts.append(f"{k}={v!r}")
        code.append(_INDENT * depth + f"nisa.{node.op_cls.NAME}({', '.join(parts)})")
    else:
        raise TypeError(f"unexpected unit node type {type(node).__name__}")
```

`_rotation_on(buf, iteration, versions)` builds the rotation for `iteration`
(the iteration-parameterised twin of Task 3's `_version_rotation`). It takes
the `versions` dict so it only rotates buffers this pipeline grew, and uses
`iteration` (not a bare loop_var) as the modulo's left operand:

```python
def _rotation_on(buf: Buffer, iteration: Expr, versions: dict[str, int]) -> Expr | None:
    """Version rotation for ``buf`` at ``iteration``, or None. Bare
    ``iteration % v`` when num_p_tiles == 1 (no ``* 1``); else
    ``Mul(iteration % v, Const(num_p_tiles))``. Only buffers in ``versions``
    (the pipeline-grown ones) rotate."""
    v = versions.get(buf.name)
    if v is None or v <= 1:
        result = None
    else:
        mod = Mod(left=iteration, right=Const(value=v))
        num_p_tiles = buf.physical_shape()[1] // v
        result = mod if num_p_tiles == 1 else Mul(left=mod, right=Const(value=num_p_tiles))
    return result
```

> **DRY with Task 3.** Task 3's `_version_rotation(buf, loop_var)` and this
> `_rotation_on(buf, iteration, versions)` share the bare-Mod/Mul logic.
> Refactor Task 3's helper to call
> `_rotation_on(buf, Var(name=loop_var), {buf.name: buf.versions})` so there
> is ONE rotation builder. (Task 3 keyed rotation off `buf.versions` directly;
> this version keys off the explicit `versions` dict, which the 3-phase loop
> passes. Both must yield bare `Mod` when num_p_tiles == 1.) NOTE: Task 3's
> monolithic path is REPLACED in Task 7 anyway, so the cleanest move is to
> have Task 7 delete the old monolithic rotation-threading and route
> everything through `_emit_unit_at_iteration`. For Task 6, just ADD the new
> helpers without breaking Task 3's still-live monolithic path (keep both
> `_version_rotation` and `_rotation_on`; unify in Task 7).

> **NOTE:** `_ordered_leaves(ir, block_nid)` must yield the block's ISA leaves
> in source order, descending through any inner ForNodes (the matmul block
> has a d2/d0 nest). If the block has inner loops, this helper must emit those
> `for` lines too — reuse `_emit_subtree` with the `subs`/`versions` threaded.
> For the matmul block (which carries the d2,d0 inner nest) the emitter must
> recurse; for memset/drain (single leaf) it is flat. Implement
> `_emit_block_at_iteration` to delegate to a subs-aware `_emit_subtree`
> variant rather than only handling flat leaves. **This is the task's real
> work — the test above covers the flat drain; add a second assertion for the
> matmul block's nested emission.**

Add the matmul-block assertion:

```python
    # matmul block at iteration i_d1_0 (stage 0): inner d2/d0 nest preserved, psum rotated
    code2: list[str] = []
    _emit_block_at_iteration(ir, children[1], depth=1, code=code2,
                             loop_var="i_d1_0", iteration=Var(name="i_d1_0"),
                             versions={"psum_prod": 2})
    body2 = "\n".join(code2)
    assert "for i_d2_0 in range(4):" in body2
    assert "psum_prod[0:128, i_d1_0 % 2, i_d2_0 * 512:i_d2_0 * 512 + 512]" in body2
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest test/codegen/test_body.py -k at_iteration -v`
Expected: PASS — both flat (drain) and nested (matmul) emission.

- [ ] **Step 5: Imports**

Add to `body.py`: `from nkigym.ir.arith.expr import Sub` and ensure
`substitute`, `rebased_region`, `BufferRegion` are imported.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "feat(codegen): per-stage loop_var substitution emitter for pipeline blocks"
```

---

### Task 7: 3-phase loop emission (prologue / steady body / epilogue)

Wire the substitution emitter into `_emit_subtree`: a pipeline-annotated
ForNode emits the three phases instead of one `for` (replacing Increment 1's
monolithic-with-rotation path).

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`_emit_subtree` ForNode branch)
- Test: `test/codegen/test_body.py`

- [ ] **Step 1: Write the failing test**

```python
def test_emit_pipeline_three_phase_structure():
    """A (0,0,1) pipeline emits: prologue (stage-0 blocks at i=0), steady
    body for i in range(1, 16) with stage-1 drain shifted -1, epilogue drain at 15."""
    ir = tuned_ir()
    m_loop, children = m_loop_and_children(ir)
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    parent = parent_block_of(ir, m_loop)
    ir.tree.data(parent).annotations["software_pipeline"] = {
        "loop_nid": m_loop, "stages": (0, 0, 1), "order": (0, 1, 2),
    }
    src = render(ir)
    # prologue: memset+matmul at bank 0, NO drain
    assert "nisa.memset(dst=psum_prod[0:128, 0, 0:0 + 2048]" in src
    # steady body starts at 1
    assert "for i_d1_0 in range(1, 16):" in src
    # drain shifted by -1 inside the body
    assert "sbuf_prod[0:128, i_d1_0 - 1, 0:0 + 2048]" in src
    # epilogue: final drain at literal 15
    assert "sbuf_prod[0:128, 15, 0:0 + 2048]" in src
    # NO trip-1 for-loop emitted (prologue/epilogue are straight-line)
    assert "range(0, 1)" not in src and "range(15, 16)" not in src
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest test/codegen/test_body.py -k three_phase -v`
Expected: FAIL — currently emits the monolithic loop (Increment 1 path).

- [ ] **Step 3: Implement 3-phase emission**

In `_emit_subtree`, replace the monolithic-pipeline branch (from Task 3) with
the 3-phase emitter. When a ForNode is the `loop_nid` of an annotation:

```python
def _emit_pipeline_loop(ir, loop_nid, annotation, depth, code):
    """Emit prologue + skewed steady body + epilogue for a pipelined loop."""
    for_node = ir.tree.data(loop_nid)
    loop_var = for_node.loop_var
    extent = for_node.extent
    children = [c for c in ir.tree.children(loop_nid) if isinstance(ir.tree.data(c), BlockNode)]
    stage_of = {children[i]: annotation["stages"][i] for i in range(len(children))}
    max_stage = max(annotation["stages"])
    versions = _pipeline_versions(ir, children)  # {tensor: version_count}

    # Prologue: lead-in p in [0, max_stage), straight-line (NO trip-1 for).
    for p in range(max_stage):
        for block_nid in children:
            s = stage_of[block_nid]
            if s <= p:
                _emit_block_at_iteration(ir, block_nid, depth, code, loop_var,
                                         Const(value=p - s), versions)
    # Steady body: for loop_var in range(max_stage, extent).
    code.append(_INDENT * depth + f"for {loop_var} in range({max_stage}, {extent}):")
    for block_nid in children:
        s = stage_of[block_nid]
        iteration = Var(name=loop_var) if s == 0 else Sub(left=Var(name=loop_var), right=Const(value=s))
        _emit_block_at_iteration(ir, block_nid, depth + 1, code, loop_var, iteration, versions)
    # Epilogue: lead-out e in [0, max_stage), straight-line.
    for e in range(max_stage):
        for block_nid in children:
            s = stage_of[block_nid]
            if s > e:
                _emit_block_at_iteration(ir, block_nid, depth, code, loop_var,
                                         Const(value=extent + e - s), versions)
```

`_pipeline_versions(ir, children)` returns `{tensor: buf.versions}` for every
versions>1 buffer the children touch (read `ir.buffer(name).versions`).

In `_emit_subtree`'s ForNode branch, dispatch: if `nid` is an annotated
`loop_nid`, call `_emit_pipeline_loop` (and stop — do NOT also emit the
monolithic loop); else the existing single-loop path. **Remove the
Increment-1 monolithic-rotation special case** — the annotation now always
means 3-phase. (The rotation context helper from Task 3 is subsumed by
`_emit_block_at_iteration`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest test/codegen/test_body.py -k three_phase -v`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `pytest test/codegen/ -q`
Expected: PASS. (Increment-1's Task-3 renderer test will now render 3-phase;
update its assertions to expect the prologue, or delete it in favor of the
Task-7 test — note this in the commit.)

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "feat(codegen): 3-phase prologue/skewed-body/epilogue pipeline emission"
```

---

### Task 8: Increment 2 byte-exact + sim gate against `kernel_16`

**Files:**
- Modify: `kernel_transforms.py` (add `kernel_16`)
- Test: `test/transforms/test_software_pipeline.py`

- [ ] **Step 1: Add `kernel_16` (the validated `skewed` body)**

Append to `kernel_transforms.py` the exact body of `build_skewed()` from
`examples/handwritten_pipeline_matmul.py`. **VERIFY:** run
`python -c "from examples.handwritten_pipeline_matmul import build_skewed; print(build_skewed())"`
and transcribe verbatim as `@nki.jit def kernel_16(...)`. The body is the
prologue-memset+matmul / `for i_d1_0 in range(1, 16)` skewed body / epilogue
drain shown in the spec Section 5.

- [ ] **Step 2: Write the failing byte-exact test**

```python
def test_increment2_matches_kernel_16_byte_exact():
    """render(apply(tuned, (0,0,1))) reproduces the validated skewed kernel."""
    ir = tuned_ir()
    m_loop, _ = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert_matches_hand(render(new_ir), KT.kernel_16)
```

> **NOTE:** the Increment-1 `kernel_15` byte-exact test (Task 5) now FAILS —
> the same `apply` renders the skewed kernel, not rotate_only. This is
> expected: Increment 2 changes the renderer. Update the Task-5 test to either
> (a) assert against `kernel_16` (merge with this test), or (b) delete it.
> Document the change in the commit. `kernel_15` stays in the file as the
> profiled MVP reference but is no longer a render target.

- [ ] **Step 3: Run to verify it fails, then make it pass**

Run: `pytest test/transforms/test_software_pipeline.py -k kernel_16 -v`
Expected: FAIL first (transcription/emitter gaps), then PASS after fixing
`body.py`. Do NOT weaken the oracle.

- [ ] **Step 4: Sim-validate**

Run: `pytest test/transforms/test_software_pipeline.py -k sim -v`
Expected: PASS — the skewed kernel computes `lhs_T.T @ rhs` (sim already
proved this in the experiment; this re-proves the *transformed* render).

- [ ] **Step 5: Full suite + commit**

Run: `pytest test/ -q`
Expected: all PASS.

```bash
git add kernel_transforms.py test/transforms/test_software_pipeline.py
git commit -m "feat: SoftwarePipeline Increment 2 (skew) byte-exact + sim vs kernel_16"
```

---

### Task 9: Increment 2 HW validation + example integration

**Files:**
- Modify: `examples/tune_matmul_lhsT_rhs.py` (the TRACE 5th atom from Task 5
  Step 7 now renders the skewed kernel — no code change, just re-profile)

- [ ] **Step 1: Re-profile on Trn2**

```bash
transport/kaizen.sh --name default --cmd "python examples/tune_matmul_lhsT_rhs.py" --cache /home/weittang/cache/tune_matmul
```

Expected: tuned MFU ≈ 83-84%, `tensor_engine_active_time_percent` ≈ 88%
(read from `/home/weittang/cache/tune_matmul/.../tuned/profile_summary.json`).
Matches the experiment's `skewed` measurement (83.7% / 88.1%).

- [ ] **Step 2: Record the result**

Update `examples/tune_matmul_lhsT_rhs.py`'s module docstring measured-table
with the new tuned number, and add a one-line learning to
`.claude/rules/learnings.md` noting the transform shipped and its measured MFU.

- [ ] **Step 3: Commit**

```bash
git add examples/tune_matmul_lhsT_rhs.py .claude/rules/learnings.md
git commit -m "docs: SoftwarePipeline shipped — matmul 59->84% MFU, HW-validated"
```

---
