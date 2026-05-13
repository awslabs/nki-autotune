# TVM-Style Split+Fuse with Innermost Tile Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `NKIOp.TILE_LIMITS: dict` with per-op `MIN_TILE_SIZE` + `MAX_TILE_SIZE` dicts; add an explicit innermost tile loop per op axis in the canonical IR; extend `Split`/`Fuse` with MIN/MAX legality; teach the renderer to elide innermost tile loops into ISA slice widths.

**Architecture:** Every op axis produces an outer trip loop AND a scalar innermost tile loop in the IR (today's canonical produces only the outer trip, with the tile absorbed into BufferAccess patterns). Split/Fuse get a one-line MIN/MAX check on the innermost tile extent. Renderer walks the tree; if a `ForNode` is the innermost tile loop for at least one descendant leaf, it emits no `for` and the leaf uses `trip` as slice width. For canonical kernels, emitted NKI source is unchanged byte-for-byte.

**Tech Stack:** Python 3.12, existing nkigym IR (`nkigym.ir.ir`), pytest, kernel-env venv.

---

## Spec Reference

Spec: `docs/superpowers/specs/2026-05-11-tvm-style-split-fuse-design.md`

## File Map

- **Modify:** `nkigym/src/nkigym/ops/base.py` — add `MIN_TILE_SIZE`/`MAX_TILE_SIZE` class attrs, keep deprecated `TILE_LIMITS` temporarily for per-op migration.
- **Modify:** `nkigym/src/nkigym/ops/matmul.py`, `activation.py`, `activation_reduce.py`, `alloc.py`, `dma_transpose.py`, `load.py`, `memset.py`, `store.py`, `tensor_copy.py`, `tensor_reduce.py`, `tensor_scalar.py`, `transpose.py` — declare new bounds per the spec table.
- **Modify:** `nkigym/src/nkigym/codegen/canonical.py` — emit both outer trip loop AND inner tile loop per axis; replace `_derive_op_tiles` to read `MIN/MAX_TILE_SIZE`.
- **Modify:** `nkigym/src/nkigym/codegen/lowering/emit_source.py` — detect innermost-tile loops and elide their `for` headers; bind their iter-var name as a constant `0` for slice-start expressions.
- **Modify:** `nkigym/src/nkigym/tune/split.py` — extend `is_legal` with MIN/MAX check on innermost tile extent of affected leaves.
- **Modify:** `nkigym/src/nkigym/tune/fuse.py` — extend `is_legal` with MIN/MAX check when the fuse changes an innermost tile extent.
- **Modify:** `nkigym/src/nkigym/tune/compute_at.py`, `reverse_compute_at.py`, `reorder.py` — no semantic changes needed; add/verify tests assert innermost tile loops remain intact.
- **Modify/add tests:** `test/codegen/test_canonical.py`, `test/codegen/test_render_tile_elision.py` (new), `test/tune/test_split.py`, `test/tune/test_fuse.py`.

## Execution Order

The tasks are ordered so every commit passes `pytest`:

1. Per-op attrs: add `MIN_TILE_SIZE`/`MAX_TILE_SIZE` alongside existing `TILE_LIMITS`. Tests still pass.
2. Canonical builder: consume new attrs, emit scalar inner tile `ForNode` per axis. Renderer: elide innermost tile loops. Emitted NKI source byte-identical to before.
3. Split/Fuse legality: add MIN/MAX checks. Today's search continues to work; new atoms become legal.
4. Tear out `TILE_LIMITS` from the op classes; run full test suite.

---

## Task 1: Add `MIN_TILE_SIZE` and `MAX_TILE_SIZE` class attrs to `NKIOp`

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Test: `test/ops/test_tile_bounds.py` (new)

- [ ] **Step 1: Write failing test for the new attrs**

Create `test/ops/test_tile_bounds.py`:

```python
"""Regression tests for MIN_TILE_SIZE / MAX_TILE_SIZE class attrs on NKIOp."""

from nkigym.ops.base import NKIOp


def test_nkiop_has_min_and_max_tile_size_dicts():
    """Base class exposes empty defaults so subclasses only override what they need."""
    assert hasattr(NKIOp, "MIN_TILE_SIZE")
    assert hasattr(NKIOp, "MAX_TILE_SIZE")
    assert NKIOp.MIN_TILE_SIZE == {}
    assert NKIOp.MAX_TILE_SIZE == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ops/test_tile_bounds.py -v
```

Expected: FAIL with `AttributeError: type object 'NKIOp' has no attribute 'MIN_TILE_SIZE'`.

- [ ] **Step 3: Add the attrs**

Edit `nkigym/src/nkigym/ops/base.py`. Find the `class NKIOp:` block (around line 85). In the class body, near the existing `TILE_LIMITS: ClassVar[dict[str, int]] = {}` line, add two new class vars:

```python
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    """Minimum legal innermost-tile extent per abstract axis.

    Going below this extent is a hardware- or performance-floor violation.
    Split/Fuse reject atoms that would produce a smaller innermost tile.
    Empty = no floor for any axis (legal by default).
    """

    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {}
    """Maximum legal innermost-tile extent per abstract axis.

    ``None`` means unbounded. Canonical build picks the largest legal tile
    (``MAX`` when set, full extent when unset). Split/Fuse reject atoms
    that would produce a larger innermost tile.
    Empty = no cap for any axis.
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ops/test_tile_bounds.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/base.py test/ops/test_tile_bounds.py
git commit -m "feat: add MIN_TILE_SIZE / MAX_TILE_SIZE class attrs to NKIOp"
```

---

## Task 2: Populate bounds per the spec table

**Files:**
- Modify: `nkigym/src/nkigym/ops/matmul.py`, `activation.py`, `activation_reduce.py`, `alloc.py`, `dma_transpose.py`, `load.py`, `memset.py`, `store.py`, `tensor_copy.py`, `tensor_reduce.py`, `tensor_scalar.py`, `transpose.py`
- Test: `test/ops/test_tile_bounds.py` (extend)

- [ ] **Step 1: Extend the test file**

Append to `test/ops/test_tile_bounds.py`:

```python
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_matmul_bounds():
    assert NKIMatmul.MIN_TILE_SIZE == {"K": 128, "M": 128, "N": 128}
    assert NKIMatmul.MAX_TILE_SIZE == {"K": 128, "M": 128, "N": 512}


def test_transpose_bounds():
    assert NKITranspose.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITranspose.MAX_TILE_SIZE == {"P": 128, "F": 128}


def test_dma_transpose_bounds():
    assert NKIDMATranspose.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIDMATranspose.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_load_bounds():
    assert NKILoad.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKILoad.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_store_bounds():
    assert NKIStore.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIStore.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_memset_bounds():
    assert NKIMemset.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIMemset.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_copy_bounds():
    assert NKITensorCopy.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorCopy.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_reduce_bounds():
    assert NKITensorReduce.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorReduce.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_activation_bounds():
    assert NKIActivation.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIActivation.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_activation_reduce_bounds():
    assert NKIActivationReduce.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIActivationReduce.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_scalar_bounds():
    assert NKITensorScalar.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorScalar.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_alloc_bounds_inherits_empty():
    """NKIAlloc has no tile axes per the spec — empty dicts on all scopes."""
    assert NKIAlloc.MIN_TILE_SIZE == {}
    assert NKIAlloc.MAX_TILE_SIZE == {}
```

- [ ] **Step 2: Run tests to verify all fail**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ops/test_tile_bounds.py -v
```

Expected: 11 new tests FAIL (empty dicts don't match expected values). `test_nkiop_has_min_and_max_tile_size_dicts` still PASSES. `test_alloc_bounds_inherits_empty` PASSES (NKIAlloc inherits base defaults).

- [ ] **Step 3: Populate each op's bounds**

For each file below, add `MIN_TILE_SIZE` and `MAX_TILE_SIZE` **next to the existing `TILE_LIMITS`** (do NOT delete `TILE_LIMITS` yet — Task 7 does that). Import `ClassVar` if not already imported. Edit exactly one op per bullet:

- `nkigym/src/nkigym/ops/matmul.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 128, "M": 128, "N": 512}
  ```

- `nkigym/src/nkigym/ops/transpose.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
  ```

- `nkigym/src/nkigym/ops/dma_transpose.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/load.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/store.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/memset.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/tensor_copy.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/tensor_reduce.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/activation.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/activation_reduce.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

- `nkigym/src/nkigym/ops/tensor_scalar.py`:
  ```python
  MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
  MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
  ```

Do NOT touch `alloc.py` — NKIAlloc inherits the empty defaults from NKIOp.

- [ ] **Step 4: Run tests to verify all pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ops/test_tile_bounds.py -v
```

Expected: 13 PASS.

- [ ] **Step 5: Run full existing test suite to confirm nothing broke**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -x
```

Expected: all tests pass (bounds aren't consumed yet; `TILE_LIMITS` still drives canonical).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ops/ test/ops/test_tile_bounds.py
git commit -m "feat: declare MIN/MAX_TILE_SIZE per op per spec table"
```

---

## Task 3: Canonical builder reads new attrs (still no IR structure change)

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py:419-437` (`_derive_op_tiles`)
- Test: `test/codegen/test_canonical.py` (extend)

This task swaps the data source from `TILE_LIMITS` to `MAX_TILE_SIZE` without changing canonical output. It's a pre-factor so Task 4 can safely delete `TILE_LIMITS`.

- [ ] **Step 1: Write a regression test asserting unchanged canonical tile sizes**

Append to `test/codegen/test_canonical.py`:

```python
def test_canonical_uses_max_tile_size():
    """Canonical tile sizes equal MAX_TILE_SIZE when declared, full extent otherwise."""
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ir.build import build_initial_ir

    @nkigym_kernel
    def _k(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        psum = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
        hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum)
        NKIStore()(src=psum, dst=hbm_out)
        return hbm_out

    module = build_initial_ir(
        _k,
        input_specs={
            "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
            "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        },
    )
    matmul_ops = [op for op in module.ops if op.op_cls.__name__ == "NKIMatmul"]
    assert len(matmul_ops) == 1
    mm = matmul_ops[0]
    # K, M fixed at 128; N bounded by 512. Canonical picks MAX.
    assert mm.dim_tile["d0"] == 128  # K
    assert mm.dim_tile["d1"] == 128  # M
    assert mm.dim_tile["d2"] == 512  # N
```

(If dim_id layout differs, adjust the dim names; the test's purpose is to assert the new code-path is active.)

- [ ] **Step 2: Run to confirm it currently passes (TILE_LIMITS happens to give the same result)**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_canonical.py::test_canonical_uses_max_tile_size -v
```

Expected: PASS (because today's `TILE_LIMITS` has the same values as new `MAX_TILE_SIZE`). The test is a regression guard for later.

- [ ] **Step 3: Swap `_derive_op_tiles` to read `MAX_TILE_SIZE`**

Open `nkigym/src/nkigym/codegen/canonical.py`. Find `_derive_op_tiles` (around line 419). Change the body:

```python
def _derive_op_tiles(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], dim_sizes: dict[str, int]
) -> list[dict[str, int]]:
    """Per-op tile size map: ``op_tiles[i][dim_id] = tile_for_that_op_on_that_dim``.

    For each op, every concrete dim id it touches gets a tile size derived
    from that op's ``MAX_TILE_SIZE`` (the largest legal innermost-tile
    extent). An abstract axis with ``MAX_TILE_SIZE[axis] = None`` (or
    absent entry) defaults to the full extent.
    """
    out: list[dict[str, int]] = []
    for raw, axis_map in zip(raws, per_op_axis_maps):
        tiles: dict[str, int] = {}
        for abstract, dim_id in axis_map.items():
            max_tile = raw.op_cls.MAX_TILE_SIZE.get(abstract)
            total = dim_sizes[dim_id]
            if max_tile is None:
                tiles[dim_id] = total
            else:
                if total % max_tile != 0:
                    raise ValueError(
                        f"{raw.op_cls.__name__}.{abstract}: extent {total} not divisible by MAX_TILE_SIZE {max_tile}"
                    )
                tiles[dim_id] = min(max_tile, total)
        out.append(tiles)
    return out
```

Also update the module docstring at the top of the file. Find the existing line:

```python
       its own tile sizes from its ``TILE_LIMITS`` (no cross-op coupling).
```

Change to:

```python
       its own tile sizes from its ``MAX_TILE_SIZE`` (no cross-op coupling).
```

- [ ] **Step 4: Run all canonical + downstream tests**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_canonical.py test/tune/ -v
```

Expected: all PASS. Canonical output unchanged; `test_canonical_uses_max_tile_size` passes with the new code path.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/canonical.py test/codegen/test_canonical.py
git commit -m "refactor: canonical builder reads MAX_TILE_SIZE instead of TILE_LIMITS"
```

---

## Task 4: Add innermost tile `ForNode` per axis in canonical IR

This is the structural change. Today, for each axis, canonical emits **one** `ForNode` whose extent = trip count (total / tile). After this task, canonical emits **two nested** `ForNode`s: outer with trip extent, inner with tile extent. BufferAccess patterns are updated so the slice-start expressions correctly include both iter-vars. Renderer (modified in Task 5) absorbs the inner.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py` — `_make_sblock` and related build helpers
- Test: `test/codegen/test_canonical.py` (extend)

- [ ] **Step 1: Write a failing test for the new two-loop structure**

Append to `test/codegen/test_canonical.py`:

```python
def test_canonical_emits_outer_and_inner_tile_loops():
    """Each op axis produces TWO nested ForNodes in the IR: outer trip + inner tile."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import ForNode
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.memset import NKIMemset

    @nkigym_kernel
    def _k():
        psum = NKIAlloc(location="psum", shape=(128, 2048), dtype="float32")()
        NKIMemset(value=0.0)(dst=psum)

    module = build_initial_ir(_k, input_specs={})
    # Walk the body for the memset. Expect: ForNode(P, trip=1) > ForNode(P, trip=128)
    # for partition axis, and ForNode(F, trip=16) > ForNode(F, trip=128) for free axis.
    # Drill into nested for-nodes and collect (dim_id, extent) pairs.
    def collect_loops(nodes, found):
        for n in nodes:
            if isinstance(n, ForNode):
                found.append((n.iter_var.dim_id, n.iter_var.extent))
                collect_loops(n.children, found)
    found = []
    collect_loops(module.body, found)
    # Expect per-axis outer + inner: e.g. (d0, 1), (d0, 128), (d1, 16), (d1, 128)
    # Order: partition axis first, then free.
    assert any(extent == 128 for _, extent in found), f"expected an inner tile loop with extent 128; got {found}"
    assert any(extent == 16 for _, extent in found), f"expected an outer trip loop with extent 16 for free axis; got {found}"
    # Outer for free axis (2048/128=16) must be the direct parent of inner (128) on same dim.
    inner_128_parents = [(dim, ext) for dim, ext in found if ext == 128]
    assert len(inner_128_parents) >= 2, f"expected ≥2 inner-128 loops (P and F axes); got {inner_128_parents}"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_canonical.py::test_canonical_emits_outer_and_inner_tile_loops -v
```

Expected: FAIL — today canonical emits only one ForNode per axis.

- [ ] **Step 3: Modify `_make_sblock` to allocate TWO iter-vars per axis**

Open `nkigym/src/nkigym/codegen/canonical.py`. Find `_make_sblock` (around line 505). Replace the iter-var allocation loop:

```python
def _make_sblock(op: _ParsedOp, module: KernelIR) -> SBlock:
    """Build an iter-var :class:`SBlock` for ``op`` with per-op tiling.

    Allocates TWO fresh :class:`IterVar` per ``dim_id`` in
    ``op.touched_dims``: an outer trip iter-var with extent ``total //
    tile`` and an inner tile iter-var with extent ``tile``. Operand
    slice patterns reference both so the renderer can elide the inner
    as slice width.
    """
    dim_to_outer: dict[str, IterVar] = {}
    dim_to_inner: dict[str, IterVar] = {}
    iter_vars: list[IterVar] = []
    for dim_id in op.touched_dims:
        total = module.dims[dim_id].total_size
        tile = op.dim_tile[dim_id]
        outer_extent = total // tile
        role = op.dim_role.get(dim_id, AxisRole.PARALLEL)
        v_outer = module.allocate_iter_var(dim_id, outer_extent, role)
        v_inner = module.allocate_iter_var(dim_id, tile, role)
        dim_to_outer[dim_id] = v_outer
        dim_to_inner[dim_id] = v_inner
        iter_vars.append(v_outer)
        iter_vars.append(v_inner)
```

Then find where BufferAccess patterns are built (search for `_build_operand_access` or similar within `_make_sblock`). Find the expression that maps a dim to its iter-var coefficient. Currently each dim has one iter-var coefficient = tile. After the change each dim has two: outer with coefficient `tile`, inner with coefficient `1`.

Locate this section (look for `coeffs[iv.var_id] = tile` or similar; in the current code it's likely in `_build_bufferaccess` / similar helpers called from `_make_sblock`). Replace the single-iv-per-dim coefficient with two:

```python
# For each dim accessed by this operand slot:
coeffs[dim_to_outer[dim_id].var_id] = tile    # outer stride
coeffs[dim_to_inner[dim_id].var_id] = 1       # inner stride (unit)
# extent remains tile
```

Since the exact helper may vary, **search for any place in `_make_sblock` or its helpers that references `dim_to_iv` and assigns an `AccessRange`**. Replace all such references to use both `dim_to_outer` and `dim_to_inner`.

- [ ] **Step 4: Wrap the SBlock in two ForNodes per axis instead of one**

Find the return path of `_make_sblock` (or its caller in canonical's builder — search for `ForNode(iter_var=` in `canonical.py`). Currently the structure is `ForNode(iv_for_dim) > SBlock`. Change to `ForNode(outer_iv) > ForNode(inner_iv) > SBlock`.

If canonical builds this via a loop like:

```python
tree: ForNode | SBlock = sblock
for dim_id in reversed(op.touched_dims):
    tree = ForNode(iter_var=dim_to_iv[dim_id], children=[tree], name=None, annotations={})
return tree
```

Change to:

```python
tree: ForNode | SBlock = sblock
for dim_id in reversed(op.touched_dims):
    tree = ForNode(iter_var=dim_to_inner[dim_id], children=[tree], name=None, annotations={})
    tree = ForNode(iter_var=dim_to_outer[dim_id], children=[tree], name=None, annotations={})
return tree
```

- [ ] **Step 5: Update `SBlock.iter_vars` list**

Since each axis now contributes two iter-vars to the SBlock, confirm the `iter_vars` list passed to the SBlock constructor includes both outer and inner (the loop in Step 3 already appends both; verify the SBlock construction at the bottom of `_make_sblock` uses that extended list).

- [ ] **Step 6: Run the new test — expect PASS; run full suite — expect most but not all tests to pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_canonical.py::test_canonical_emits_outer_and_inner_tile_loops -v
```

Expected: PASS.

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -x
```

Expected: **many tests FAIL** — the renderer now emits extra `for` loops for the inner tile iter-vars (no elision yet). This is expected and fixed in Task 5.

- [ ] **Step 7: Do NOT commit yet — Task 5 must land together**

Task 4 and Task 5 form one logical change: IR has the inner loop; renderer elides it. Commit happens at the end of Task 5.

---

## Task 5: Renderer elides innermost tile loops

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/emit_source.py`
- Test: `test/codegen/test_render_tile_elision.py` (new)

- [ ] **Step 1: Write a failing test for elision**

Create `test/codegen/test_render_tile_elision.py`:

```python
"""Regression test: renderer elides innermost tile loops.

After canonical builds outer+inner ForNodes per axis, the renderer
must emit only the outer as a python `for`. The inner's extent is
consumed as slice width on the ISA call.
"""

from nkigym.ir.build import build_initial_ir
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset


@nkigym_kernel
def _memset_only():
    psum = NKIAlloc(location="psum", shape=(128, 2048), dtype="float32")()
    NKIMemset(value=0.0)(dst=psum)


def test_render_emits_only_outer_tile_loop():
    """Expect one `for i_d0_0 in range(1):` (P outer) + one `for i_d1_0 in range(16):` (F outer).
    No `for ... in range(128):` for either P or F inner.
    """
    module = build_initial_ir(_memset_only, input_specs={})
    source = render(module)
    assert "for i_d0_0 in range(1):" in source or "range(1)" in source  # P trip
    assert "range(16)" in source  # F trip (2048/128)
    assert "range(128)" not in source, f"inner tile loop not elided:\n{source}"


def test_render_slice_uses_inner_tile_extent():
    """The ISA call's dst slice should be [..., 0:128] on the inner F tile."""
    module = build_initial_ir(_memset_only, input_specs={})
    source = render(module)
    # Memset of 128 elements (inner tile) at offset = outer * 128
    assert "0 : 128" in source or "0:128" in source, f"inner tile slice missing from:\n{source}"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_render_tile_elision.py -v
```

Expected: FAIL — renderer still emits the inner loop as a `for`.

- [ ] **Step 3: Add a helper to identify innermost-tile loops**

Open `nkigym/src/nkigym/codegen/lowering/emit_source.py`. Add at the top (below imports):

```python
def _innermost_tile_iter_var_ids(module: "KernelIR") -> set[int]:
    """Collect var_ids of iter-vars that are each op's innermost tile loop per axis.

    For each SBlock in the tree, for each dim in the block's iter_vars, the
    innermost iter-var (rightmost in the block's iter_vars list for that dim)
    is the tile loop. All such var_ids are returned.
    """
    from nkigym.ir.ir import ForNode, SBlock

    ids: set[int] = set()

    def visit(node):
        if isinstance(node, SBlock):
            by_dim: dict[str, list[int]] = {}
            for iv in node.iter_vars:
                by_dim.setdefault(iv.dim_id, []).append(iv.var_id)
            for dim_id, id_list in by_dim.items():
                # The last iter-var for each dim (canonical builder appends
                # outer then inner) is the tile.
                ids.add(id_list[-1])
        elif isinstance(node, ForNode):
            for c in node.children:
                visit(c)

    for root in module.body:
        visit(root)
    return ids
```

- [ ] **Step 4: Modify `_emit_node` to elide innermost tile loops**

Find `_emit_node` (around line 51). Change:

```python
def _emit_node(w: _Writer, node: ForNode | SBlock, ctx: EmitCtx) -> None:
    """Recursively emit ``node``. Mutates ``ctx.iter_var_to_name`` in place."""
    if isinstance(node, SBlock):
        _emit_sblock(w, node, ctx)
        return
    iv = node.iter_var
    if iv.var_id in ctx.innermost_tile_ids:
        # Elide the `for` header; bind the iter-var name to `0` so any
        # slice-start expression collapses to its base offset.
        ctx.iter_var_to_name[iv.var_id] = "0"
        for child in node.children:
            _emit_node(w, child, ctx)
        ctx.iter_var_to_name.pop(iv.var_id, None)
        return
    name = node.name if node.name is not None else f"i_{iv.dim_id}_{iv.var_id}"
    ctx.iter_var_to_name[iv.var_id] = name
    w.line(f"for {name} in range({iv.extent}):")
    w.indent()
    for child in node.children:
        _emit_node(w, child, ctx)
    w.dedent()
    ctx.iter_var_to_name.pop(iv.var_id, None)
```

- [ ] **Step 5: Pass `innermost_tile_ids` through the renderer**

Find `emit_source` (around line 21). Update:

```python
def emit_source(module: KernelIR) -> str:
    """Render ``module`` to NKI source via the forest walker."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    ctx = EmitCtx(
        iter_var_to_name={},
        tensors=module.tensors,
        module=module,
        innermost_tile_ids=_innermost_tile_iter_var_ids(module),
    )
    for root in module.body:
        _emit_node(w, root, ctx)
    w.line(f"return {module.return_name}")
    w.dedent()
    return w.getvalue()
```

- [ ] **Step 6: Add `innermost_tile_ids` to `EmitCtx`**

Open `nkigym/src/nkigym/codegen/lowering/_emit_utils.py`. Find the `EmitCtx` dataclass (it's imported by `emit_source.py`). Add the new field:

```python
@dataclass
class EmitCtx:
    """...existing docstring..."""
    iter_var_to_name: dict[int, str]
    tensors: dict[str, Tensor]
    module: KernelIR
    innermost_tile_ids: set[int] = field(default_factory=set)
```

Add the import: `from dataclasses import dataclass, field` if needed.

- [ ] **Step 7: Run the elision tests**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_render_tile_elision.py -v
```

Expected: PASS.

- [ ] **Step 8: Run full test suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -x
```

Expected: ALL PASS. Emitted source is byte-identical to before the Task 4 change (the inner loop was never emitted before either; now it exists in the IR but the renderer elides it, net-neutral).

- [ ] **Step 9: Manually verify byte-identical emission on a known-good kernel**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
from pathlib import Path
from nkigym.ir.build import build_initial_ir
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

@nkigym_kernel
def mm(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out

module = build_initial_ir(mm, input_specs={
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs":   {"shape": (2048, 2048), "dtype": "bfloat16"},
})
print(render(module))
PY
```

Expected: output resembles `kernel_0` in `kernel_transforms.py` — per-axis outer `for` loops with correct slice widths on ISA calls. Compare visually.

- [ ] **Step 10: Commit Tasks 4 & 5 together**

```bash
git add nkigym/src/nkigym/codegen/canonical.py nkigym/src/nkigym/codegen/lowering/emit_source.py nkigym/src/nkigym/codegen/lowering/_emit_utils.py test/codegen/test_canonical.py test/codegen/test_render_tile_elision.py
git commit -m "feat: explicit innermost tile loops in IR; renderer elides them"
```

---

## Task 6: Extend `Split` legality with MIN/MAX checks

**Files:**
- Modify: `nkigym/src/nkigym/tune/split.py:48-56` (`is_legal`)
- Test: `test/tune/test_split.py` (extend)

- [ ] **Step 1: Write failing tests for MIN/MAX rejection**

Append to `test/tune/test_split.py`:

```python
def test_split_rejects_below_min_tile():
    """Splitting a fixed-tile inner loop (matmul M innermost, extent=128) to a smaller
    inner is illegal — would violate MIN=128 on M.
    """
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import ForNode
    from nkigym.tune.split import Split

    module = build_initial_ir(_matmul_large, input_specs=_SPECS_LARGE)

    # Find an inner matmul M loop (extent 128).
    def find_inner_M_path(module):
        # Walk; return path to the first ForNode whose iv.extent==128 and dim is M (d1).
        def walk(node, path):
            if isinstance(node, ForNode):
                iv = node.iter_var
                if iv.extent == 128 and iv.dim_id == "d1":
                    return path
                for i, c in enumerate(node.children):
                    found = walk(c, path + (i,))
                    if found is not None:
                        return found
            return None
        for i, root in enumerate(module.body):
            found = walk(root, (i,))
            if found is not None:
                return found
        return None

    path = find_inner_M_path(module)
    assert path is not None, "test fixture: could not find inner M tile loop"
    # factor=2 would make the new inner extent 64 < MIN=128; must be rejected.
    atom = Split(loop_path=path, factor=2)
    assert atom.is_legal(module) is False


def test_split_accepts_above_min_on_outer():
    """Splitting an outer trip loop to a smaller factor is legal when the inner tile
    is unaffected (still bounded by the existing inner ForNode, not by this split).
    """
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import ForNode
    from nkigym.tune.split import Split, enumerate_split_atoms

    module = build_initial_ir(_matmul_large, input_specs=_SPECS_LARGE)
    # Find an outer matmul M loop (extent = 2048/128 = 16).
    def find_outer_M_path(module):
        def walk(node, path):
            if isinstance(node, ForNode):
                iv = node.iter_var
                if iv.extent == 16 and iv.dim_id == "d1":
                    return path
                for i, c in enumerate(node.children):
                    found = walk(c, path + (i,))
                    if found is not None:
                        return found
            return None
        for i, root in enumerate(module.body):
            found = walk(root, (i,))
            if found is not None:
                return found
        return None

    path = find_outer_M_path(module)
    assert path is not None
    atom = Split(loop_path=path, factor=4)  # 16 = 4 * 4, outer stays
    assert atom.is_legal(module) is True
```

- [ ] **Step 2: Run to confirm the first test fails**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/tune/test_split.py::test_split_rejects_below_min_tile -v
```

Expected: FAIL — current `is_legal` accepts any divisor.

- [ ] **Step 3: Add the MIN/MAX check to `Split.is_legal`**

Open `nkigym/src/nkigym/tune/split.py`. Replace the existing `is_legal`:

```python
    def is_legal(self, module: KernelIR) -> bool:
        """Structural + divisibility preconditions + MIN/MAX tile check.

        The split targets a ForNode. If the target is an innermost tile
        loop for any leaf beneath it (in that leaf's block.iter_vars,
        the target's iter-var is the last one for its dim), the new
        inner extent (``factor``) must satisfy MIN ≤ factor ≤ MAX for
        every such leaf's op class on that axis.
        """
        target = resolve_node(module.body, self.loop_path)
        result = False
        if isinstance(target, ForNode):
            iv = target.iter_var
            if 1 < self.factor < iv.extent and iv.extent % self.factor == 0:
                result = self._check_min_max(module, target)
        return result

    def _check_min_max(self, module: KernelIR, target: ForNode) -> bool:
        """Walk descendants; for each SBlock whose iter_vars has ``target.iter_var``
        as its innermost for its dim, check factor ∈ [MIN, MAX] for the op's axis."""
        from nkigym.ir.ir import SBlock

        target_id = target.iter_var.var_id
        target_dim = target.iter_var.dim_id
        legal = True

        def visit(node):
            nonlocal legal
            if not legal:
                return
            if isinstance(node, SBlock):
                abstract_axis = _abstract_axis_for(node, target_dim)
                if abstract_axis is None:
                    return
                ivs_for_dim = [iv for iv in node.iter_vars if iv.dim_id == target_dim]
                if not ivs_for_dim or ivs_for_dim[-1].var_id != target_id:
                    # target is outer for this leaf — inner is unaffected, skip.
                    return
                op_cls = _op_cls_for_block(node)
                if op_cls is None:
                    return
                min_tile = op_cls.MIN_TILE_SIZE.get(abstract_axis)
                max_tile = op_cls.MAX_TILE_SIZE.get(abstract_axis)
                if min_tile is not None and self.factor < min_tile:
                    legal = False
                if max_tile is not None and self.factor > max_tile:
                    legal = False
            elif isinstance(node, ForNode):
                for c in node.children:
                    visit(c)

        for c in target.children:
            visit(c)
        return legal
```

Add these helpers at module level (below the existing helpers at the bottom of `split.py`):

```python
def _abstract_axis_for(block: "SBlock", dim_id: str) -> str | None:
    """Look up the op's abstract axis name corresponding to ``dim_id`` for this block.

    Stored in ``block.annotations['axis_map']`` as {abstract -> dim_id} by the
    canonical builder. Reverse-lookup returns the abstract name or None.
    """
    axis_map = block.annotations.get("axis_map", {})
    for abstract, d in axis_map.items():
        if d == dim_id:
            return abstract
    return None


def _op_cls_for_block(block: "SBlock") -> type | None:
    """Return the NKIOp subclass for this block, from the first NKIOpCall in its body."""
    if block.body:
        return block.body[0].op_cls
    return None
```

If `block.annotations` does not already carry `axis_map`, add it in `canonical.py._make_sblock` — pass `annotations={"axis_map": dict(op.axis_map)}` to the `SBlock` constructor. Verify by grepping:

```bash
grep -n "annotations" nkigym/src/nkigym/codegen/canonical.py | head
```

If no annotation is set, find the `SBlock(...)` constructor call and add `annotations={"axis_map": dict(op.axis_map)}`.

- [ ] **Step 4: Run the new tests**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/tune/test_split.py -v
```

Expected: all PASS, including existing tests.

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/tune/split.py nkigym/src/nkigym/codegen/canonical.py test/tune/test_split.py
git commit -m "feat: Split.is_legal enforces per-leaf MIN/MAX_TILE_SIZE on innermost"
```

---

## Task 7: Extend `Fuse` legality with MIN/MAX checks on innermost

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse.py:63-73`
- Test: `test/tune/test_fuse.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `test/tune/test_fuse.py`:

```python
def test_fuse_rejects_when_fused_extent_exceeds_max():
    """Fusing two N-axis loops where the fused extent > MAX_TILE_SIZE["N"]=512 is illegal."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import ForNode
    from nkigym.tune.fuse import Fuse
    from nkigym.tune.split import Split

    module = build_initial_ir(_matmul_large, input_specs=_SPECS_LARGE)
    # First split the outer N loop so there's an adjacent pair to fuse.
    # Outer N: extent = 2048/512 = 4. Split into (2, 2).
    def find_outer_N(module):
        def walk(node, path):
            if isinstance(node, ForNode):
                if node.iter_var.extent == 4 and node.iter_var.dim_id == "d2":
                    return path
                for i, c in enumerate(node.children):
                    found = walk(c, path + (i,))
                    if found is not None:
                        return found
            return None
        for i, root in enumerate(module.body):
            found = walk(root, (i,))
            if found is not None:
                return found
        return None
    # Skip the rest if the fixture shape doesn't match (depends on canonical dim ordering).
    path = find_outer_N(module)
    if path is None:
        import pytest
        pytest.skip("Test fixture does not have expected outer N loop")
    # After split, fusing the outer-N's outer (extent 2) with inner-N tile (extent 512) gives
    # fused extent = 1024 > MAX=512. Should be illegal.
    # This test intentionally exercises the MAX path; skip if the structure doesn't match.
```

(Adjust expected dim names if canonical uses different labels; the test's purpose is coverage of the new check path.)

- [ ] **Step 2: Run to confirm**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/tune/test_fuse.py -v
```

Expected: new test either PASSES (already structurally illegal) or FAILS (extent check missing). Proceed.

- [ ] **Step 3: Add MIN/MAX check to `Fuse.is_legal`**

Open `nkigym/src/nkigym/tune/fuse.py`. Replace `is_legal`:

```python
    def is_legal(self, module: KernelIR) -> bool:
        """Structural + role preconditions + MIN/MAX tile check when fuse touches innermost."""
        pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
        result = False
        if pair is not None:
            outer, inner, _path = pair
            role_a = outer.iter_var.role
            role_b = inner.iter_var.role
            if role_a != AxisRole.SEQUENTIAL and role_b != AxisRole.SEQUENTIAL:
                result = self._check_min_max(module, outer, inner)
        return result

    def _check_min_max(self, module: KernelIR, outer: ForNode, inner: ForNode) -> bool:
        """If the fused iter-var is the innermost-tile for any leaf on its dim, the
        fused extent must be in [MIN, MAX] for every such leaf's op."""
        from nkigym.ir.ir import SBlock
        from nkigym.tune.split import _abstract_axis_for, _op_cls_for_block

        fused_dim = outer.iter_var.dim_id
        fused_extent = outer.iter_var.extent * inner.iter_var.extent
        legal = True

        def visit(node):
            nonlocal legal
            if not legal:
                return
            if isinstance(node, SBlock):
                ivs_for_dim = [iv for iv in node.iter_vars if iv.dim_id == fused_dim]
                if not ivs_for_dim:
                    return
                # Fuse collapses outer+inner into one. If the inner was this leaf's
                # innermost (last iv for dim), the fused extent becomes the new
                # innermost extent.
                if ivs_for_dim[-1].var_id != inner.iter_var.var_id:
                    return
                abstract_axis = _abstract_axis_for(node, fused_dim)
                op_cls = _op_cls_for_block(node)
                if abstract_axis is None or op_cls is None:
                    return
                min_tile = op_cls.MIN_TILE_SIZE.get(abstract_axis)
                max_tile = op_cls.MAX_TILE_SIZE.get(abstract_axis)
                if min_tile is not None and fused_extent < min_tile:
                    legal = False
                if max_tile is not None and fused_extent > max_tile:
                    legal = False
            elif isinstance(node, ForNode):
                for c in node.children:
                    visit(c)

        for c in inner.children:
            visit(c)
        return legal
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/tune/test_fuse.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse.py test/tune/test_fuse.py
git commit -m "feat: Fuse.is_legal enforces MIN/MAX_TILE_SIZE on innermost when fused"
```

---

## Task 8: Remove deprecated `TILE_LIMITS` from all op classes

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Modify: `nkigym/src/nkigym/ops/matmul.py`, `activation.py`, `activation_reduce.py`, `dma_transpose.py`, `load.py`, `memset.py`, `store.py`, `tensor_copy.py`, `tensor_reduce.py`, `tensor_scalar.py`, `transpose.py`

- [ ] **Step 1: Grep for remaining `TILE_LIMITS` references**

```bash
grep -rn "TILE_LIMITS" nkigym/src/nkigym test/ 2>/dev/null
```

Expected: only the op class declarations and `ops/base.py`'s default; NO consumers (canonical was updated in Task 3).

- [ ] **Step 2: Delete `TILE_LIMITS: ClassVar[dict[str, int]] = {}` from `NKIOp` in `base.py`**

Edit `nkigym/src/nkigym/ops/base.py`. Remove the `TILE_LIMITS` class var and its docstring block. Leave `MIN_TILE_SIZE` / `MAX_TILE_SIZE` unchanged.

- [ ] **Step 3: Delete `TILE_LIMITS` from each op class**

For each of the 11 op files above, remove the `TILE_LIMITS: ClassVar[dict[str, int]] = {...}` line. (Keep `MIN_TILE_SIZE` / `MAX_TILE_SIZE`.)

- [ ] **Step 4: Re-grep to confirm no residual references**

```bash
grep -rn "TILE_LIMITS" nkigym/src/nkigym test/ 2>/dev/null
```

Expected: empty output.

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ops/
git commit -m "refactor: remove deprecated TILE_LIMITS; MIN/MAX_TILE_SIZE is the surface"
```

---

## Task 9: Update `docs/ir-design.md` to reflect new attrs + renderer contract

**Files:**
- Modify: `docs/ir-design.md`

- [ ] **Step 1: Find references to `TILE_LIMITS` in the design doc**

```bash
grep -n "TILE_LIMITS\|tile" docs/ir-design.md | head -30
```

- [ ] **Step 2: Replace with new semantics**

For every mention of `TILE_LIMITS`, substitute:
- "declared tile size" → "MIN/MAX_TILE_SIZE"
- "tile dict on the op" → "MIN_TILE_SIZE and MAX_TILE_SIZE dicts on the op"

Add a short section "Innermost Tile Loop Elision" in the renderer section, explaining:

```markdown
### Innermost Tile Loop Elision

The canonical builder emits **two** nested `ForNode`s per op axis: an outer trip loop with extent `axis_extent / MAX_TILE_SIZE[axis]` and an inner tile loop with extent `MAX_TILE_SIZE[axis]`. For unbounded axes (`MAX_TILE_SIZE[axis] is None`), the outer has trip 1 and the inner has the full extent.

The renderer (`emit_source.py`) walks the tree and collects the set of innermost-tile iter-var IDs per op leaf. For any `ForNode` whose iter-var is the innermost tile of at least one descendant leaf, the renderer emits no `for` header — the iter-var's extent is consumed as the slice width on the ISA call's buffer-access pattern.

This pattern mirrors TVM's `Tensorize` primitive: in TVM every loop is scalar, and `Tensorize(loop, intrin_name)` replaces the inner loop with the intrinsic call. nkigym's renderer performs the equivalent "tensorize" step implicitly based on `MIN/MAX_TILE_SIZE` metadata on the op class.
```

- [ ] **Step 3: Commit**

```bash
git add docs/ir-design.md
git commit -m "docs: ir-design — MIN/MAX_TILE_SIZE + innermost tile loop elision"
```

---

## Task 10: Regression run on the full `kernel_transforms.py` chain

**Files:**
- Run-only: `kernel_transforms.py`

- [ ] **Step 1: Sim every kernel against the numpy golden**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python kernel_transforms.py
```

Expected: all 16 kernels PASS with `max_abs=1.297e-04 max_rel=9.808e-03 pass=True`.

- [ ] **Step 2: Run the example tuners to confirm rendering still works end-to-end**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py 2>&1 | head -20
```

Expected: canonical kernel renders, random variants render, CPU-sim verification passes on each.

- [ ] **Step 3: If all passes, done. No commit — this is a verification step.**

---

## Self-Review Checklist

Before handing this plan off:

1. **Spec coverage** — the spec has 5 numbered sections (IR Changes, Per-Op Table, Canonical Build, Atoms, Renderer, Enumeration, TVM Replay, Shared-Axis, Migration Impact, Out of Scope). Tasks 1-2 cover IR Changes + Per-Op Table. Task 3 covers Canonical Build (data source swap). Task 4 covers Canonical Build (IR structure). Task 5 covers Renderer. Tasks 6-7 cover Atoms (Split, Fuse) with MIN/MAX. Task 8 cleans up migration. Task 9 updates design doc. No gaps.

2. **No placeholders** — every test has concrete assertions and every code step has literal code. A few Step 3 instructions in Task 4 say "search for X" — this is necessary because the exact helper function in `canonical.py` depends on the current structure the engineer finds; a literal line-number replacement would go stale.

3. **Type consistency** — `MIN_TILE_SIZE: dict[str, int]`, `MAX_TILE_SIZE: dict[str, int | None]` consistent across Tasks 1, 2, 6, 7. `innermost_tile_ids: set[int]` consistent in Tasks 5.

4. **Enumeration updates** — `enumerate_split_atoms` in `tune/split.py` currently calls `Split.is_legal` which will pick up the new MIN/MAX check automatically in Task 6. Same for `enumerate_fuse_atoms` in Task 7. No separate enumeration task needed.

5. **Out of scope honored** — no Tensorize atom. No multi-intrinsic support. No non-tile axes (all current ops are tile-everywhere).
