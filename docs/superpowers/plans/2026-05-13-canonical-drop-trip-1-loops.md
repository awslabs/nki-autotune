# Canonical Drops Trip-1 Outer Loops — Implementation Plan (Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonical emits one ForNode for unbounded axes (trip = full extent) instead of two (trip-1 outer + full-extent inner). Bounded axes continue to emit outer trip + inner tile (unchanged). Renderer elision unchanged.

**Architecture:** `_make_sblock` allocates two iter-vars per bounded axis (today) or one iter-var per unbounded axis (new). `_build_buffer_access` drops the outer entry for unbounded axes. Tree wrap emits one ForNode for unbounded, two for bounded. Renderer's elision of innermost-per-axis tile still applies; for unbounded axes the only loop is the tile, still elided.

**Tech Stack:** Python 3.12, existing nkigym IR, pytest.

---

## Spec Reference

`docs/superpowers/specs/2026-05-13-canonical-drop-trip-1-loops-design.md`

## File Map

**Production:**
- `nkigym/src/nkigym/codegen/canonical.py` — `_make_sblock`: for bounded axes, unchanged (outer+inner). For unbounded axes (tile == total), allocate ONE iter-var. `_build_buffer_access`: single-entry coeff for unbounded, two-entry for bounded. Tree wrap: one ForNode per unbounded axis.

**Renderer is unchanged.** Do NOT modify `emit_source.py` or `_emit_utils.py`.

**Tests:**
- `test/codegen/test_canonical.py` — update assertions on iter-var counts for SBlocks with unbounded axes.
- `test/tune/test_axis_identity.py` — rewrite helpers to either (a) target single d1 iter-var on lhs_T load, or (b) construct same-axis pair via Split on matmul M (bounded). Three tests total.
- `test/tune/test_fuse.py`, `test/tune/test_fuse_eager_rewrite.py` — analogous fixture rewrites.
- `test/tune/test_split.py`, `test/tune/test_reorder.py`, `test/tune/test_compute_at.py`, `test/tune/test_reverse_compute_at.py` — audit.
- `test/codegen/test_rfactor_rmw.py`, `test/codegen/test_rfactor_slot.py`, `test/codegen/test_first_class_buffers.py`, `test/codegen/test_place_buffers.py`, `test/codegen/test_batch.py` — audit.

**Reference:**
- `kernel_transforms.py` — mechanical rewrite of 16 kernels to drop `for i_dN_0 in range(1):` wrappers.

**Example:**
- `examples/matmul_lhsT_rhs.py` — step 1 switches from Fuse+Split to `ComputeAt(lhs_T_load, matmul_d1_outer)`.

## Execution Order

1. Task 1: write failing test targeting the unbounded-axis single-ForNode state.
2. Task 2: modify canonical.py to emit 1 iter-var for unbounded axes, 2 for bounded.
3. Task 3: migrate tests.
4. Task 4: rewrite `kernel_transforms.py` to drop trip-1 wrappers.
5. Task 5: update `examples/matmul_lhsT_rhs.py` + full regression + atomic commit.

---

## Task 1: Failing canonical-structure test

**Files:**
- Create: `/home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_unbounded_axis.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_unbounded_axis.py`:

```python
"""Canonical emits one ForNode per unbounded axis (trip = full extent)
and two ForNodes per bounded axis (outer trip + inner tile)."""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_sblock(module, op_name, tensor_substr=None):
    def walk(node, ancestors):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == op_name:
            if tensor_substr is None:
                return node, ancestors
            for _slot, ba in node.writes.items():
                if tensor_substr in ba.tensor_name:
                    return node, ancestors
        if isinstance(node, ForNode):
            for c in node.children:
                r = walk(c, ancestors + [node.iter_var])
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None:
            return r
    raise AssertionError(f"no {op_name} SBlock found")


def test_nkiload_unbounded_F_has_one_iter_var():
    """lhs_T load: bounded P (trip=16, tile=128) + unbounded F (one loop, trip=2048)."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKILoad", tensor_substr="a")
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    """d0 is bounded (P, MAX=128): two ForNodes, extents 16 and 128."""
    assert len(by_axis[d0]) == 2
    assert [iv.extent for iv in by_axis[d0]] == [16, 128]
    """d1 is unbounded (F, MAX=None): one ForNode with extent = full axis = 2048."""
    assert len(by_axis[d1]) == 1
    assert by_axis[d1][0].extent == 2048


def test_matmul_all_bounded_axes_have_two_iter_vars():
    """Matmul K/M/N are all bounded (128/128/512)."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKIMatmul")
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    assert [iv.extent for iv in by_axis[d0]] == [16, 128]
    assert [iv.extent for iv in by_axis[d1]] == [16, 128]
    assert [iv.extent for iv in by_axis[d3]] == [4, 512]


def test_nkimemset_unbounded_F_has_one_iter_var():
    """NKIMemset: bounded P (16, 128) + unbounded F (one loop extent=2048)."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKIMemset")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    assert [iv.extent for iv in by_axis[d1]] == [16, 128]
    assert len(by_axis[d3]) == 1 and by_axis[d3][0].extent == 2048
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_unbounded_axis.py -v 2>&1 | tail -20
```

Expected: `test_nkiload_unbounded_F_has_one_iter_var` and `test_nkimemset_unbounded_F_has_one_iter_var` FAIL (today the unbounded axis emits 2 iter-vars with extents [1, 2048]). `test_matmul_all_bounded_axes_have_two_iter_vars` PASSES (bounded axes unchanged).

- [ ] **Step 3: Do not commit yet**

---

## Task 2: Modify canonical builder

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/canonical.py`

- [ ] **Step 1: Update iter-var allocation in `_make_sblock`**

Open `canonical.py`. Find `_make_sblock` (around line 512). The current allocation loop is:

```python
dim_to_outer: dict[str, IterVar] = {}
dim_to_inner: dict[str, IterVar] = {}
iter_vars: list[IterVar] = []
for axis_id in op.touched_axes:
    total = module.axes[axis_id].total_size
    tile = op.axis_tile[axis_id]
    outer_extent = total // tile
    role = op.axis_role.get(axis_id, AxisRole.PARALLEL)
    v_outer = module.allocate_iter_var(axis_id, outer_extent, role)
    v_inner = module.allocate_iter_var(axis_id, tile, role)
    dim_to_outer[axis_id] = v_outer
    dim_to_inner[axis_id] = v_inner
    iter_vars.append(v_outer)
    iter_vars.append(v_inner)
```

Change to conditional allocation:

```python
dim_to_outer: dict[int, IterVar] = {}
dim_to_inner: dict[int, IterVar] = {}
iter_vars: list[IterVar] = []
for axis_id in op.touched_axes:
    total = module.axes[axis_id].total_size
    tile = op.axis_tile[axis_id]
    role = op.axis_role.get(axis_id, AxisRole.PARALLEL)
    if tile < total:
        """Bounded axis: outer trip + inner tile (two ForNodes)."""
        v_outer = module.allocate_iter_var(axis_id, total // tile, role)
        v_inner = module.allocate_iter_var(axis_id, tile, role)
        dim_to_outer[axis_id] = v_outer
        dim_to_inner[axis_id] = v_inner
        iter_vars.append(v_outer)
        iter_vars.append(v_inner)
    else:
        """Unbounded axis (tile == total): single tile ForNode for the whole axis.

        The single iter-var plays the role of 'inner tile' for BufferAccess
        coefficient purposes; no outer trip loop exists.
        """
        v_tile = module.allocate_iter_var(axis_id, total, role)
        dim_to_inner[axis_id] = v_tile
        iter_vars.append(v_tile)
```

Note: `dim_to_outer` has no entry for unbounded axes (used by `_build_buffer_access` to skip emitting the outer coefficient). `dim_to_inner` always has an entry — it's the innermost-per-axis iter-var, and the renderer's elision logic treats it as the tile.

- [ ] **Step 2: Update `_build_buffer_access`**

Same file. Find `_build_buffer_access` (around line 564). The current per-axis coefficient logic emits both outer and inner:

```python
for i, dim_id_str in enumerate(tensor.dim_ids):
    axis_id = module.axis_id_by_name(dim_id_str)
    if axis_id in dim_to_outer:
        tile = op.axis_tile[axis_id]
        coeffs[dim_to_outer[axis_id].var_id] = tile
        coeffs[dim_to_inner[axis_id].var_id] = 1
        used_iv_ids.append(dim_to_outer[axis_id].var_id)
        used_iv_ids.append(dim_to_inner[axis_id].var_id)
        extent = tile
    else:
        # fallback for tensors whose producer is untouched by op
        ...
```

Change to handle unbounded axes (which are in `dim_to_inner` but not `dim_to_outer`):

```python
for i, dim_id_str in enumerate(tensor.dim_ids):
    axis_id = module.axis_id_by_name(dim_id_str)
    total = module.axes[axis_id].total_size
    if axis_id in dim_to_inner:
        inner_iv = dim_to_inner[axis_id]
        if axis_id in dim_to_outer:
            """Bounded: outer*tile + inner coefficient entries, extent = tile."""
            tile = op.axis_tile[axis_id]
            per_dim_coeffs = {dim_to_outer[axis_id].var_id: tile, inner_iv.var_id: 1}
            used_iv_ids.append(dim_to_outer[axis_id].var_id)
            used_iv_ids.append(inner_iv.var_id)
            extent = tile
        else:
            """Unbounded: single inner coefficient, extent = full axis."""
            per_dim_coeffs = {inner_iv.var_id: 1}
            used_iv_ids.append(inner_iv.var_id)
            extent = total
        pattern.append(AccessRange.make(per_dim_coeffs, const_offset=0, extent=extent))
    else:
        # fallback for tensors whose producer is untouched by op (unchanged logic)
        ...
```

Adapt the fallback branch to the same new shape — if the axis isn't touched by the op, the coeff is empty and extent = axis total. (This path only fires for auxiliary tensors.)

- [ ] **Step 3: Update tree wrap**

Still in `_make_sblock`. The tree-building section after the SBlock is constructed currently wraps with both outer and inner ForNodes for every axis. It needs to wrap only with ForNodes that exist:

```python
tree: ForNode | SBlock = sblock
for axis_id in reversed(op.touched_axes):
    """Inner (tile) ForNode always exists."""
    tree = ForNode(iter_var=dim_to_inner[axis_id], children=[tree], name=None, annotations={})
    """Outer trip ForNode exists only for bounded axes."""
    if axis_id in dim_to_outer:
        tree = ForNode(iter_var=dim_to_outer[axis_id], children=[tree], name=None, annotations={})
```

- [ ] **Step 4: Run the Task 1 tests**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_unbounded_axis.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Smoke test canonical render + CPU-sim**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
import numpy as np
import nki
from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.verify import _rewrite_to_fp32

@nkigym_kernel
def mm(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a); NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s); NKIStore()(src=s, dst=h)
    return h

m = build_canonical_module(mm, input_specs={"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}})
src = render(m)
print(src)
assert "range(1)" not in src, "trip-1 loops STILL present"
sim = _rewrite_to_fp32(src)
ns = {}; exec(sim, ns)
rng = np.random.default_rng(0)
lhs = rng.standard_normal((2048, 2048)).astype(np.float32)
rhs_ = rng.standard_normal((2048, 2048)).astype(np.float32)
actual = np.asarray(nki.simulate(ns["mm"])(lhs, rhs_))
expected = lhs.T @ rhs_
ok = np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
print(f"pass={ok} max_abs={float(np.abs(actual - expected).max()):.3e}")
PY
```

Expected: no `range(1)` in rendered source; `pass=True max_abs=1.297e-04`.

- [ ] **Step 6: Do not commit yet**

---

## Task 3: Test migration

**Files (audit each, modify as needed):**
- `test/codegen/test_canonical.py`
- `test/tune/test_axis_identity.py`
- `test/tune/test_fuse.py`
- `test/tune/test_fuse_eager_rewrite.py`
- `test/tune/test_split.py`
- `test/tune/test_reorder.py`
- `test/tune/test_compute_at.py`
- `test/tune/test_reverse_compute_at.py`
- `test/codegen/test_rfactor_rmw.py`
- `test/codegen/test_rfactor_slot.py`
- `test/codegen/test_first_class_buffers.py`
- `test/codegen/test_place_buffers.py`
- `test/codegen/test_batch.py`

### Step 1: Run full suite, inventory failures

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py --tb=line 2>&1 | tail -40
```

### Step 2: Apply fixes per pattern

**Pattern A (iter-var count assertions change for unbounded axes):**
- Tests asserting "4 iter-vars per SBlock" for 2D ops (P bounded + F unbounded) become "3 iter-vars" (P outer 16 + P inner 128 + F tile 2048).
- Matmul assertions unchanged (all 3 axes bounded; still 6 iter-vars).
- Transpose assertions unchanged (both P and F bounded).

**Pattern B (fixture relied on trip-1 unbounded-axis outer):**
Tests using `_find_lhs_t_load_d1_pair` or analogous helpers on unbounded axes no longer find two iter-vars. Two options:
- **B.1 (simpler):** switch the test to use the single tile iter-var on the unbounded axis. Can't do Fuse+Split on it (no same-axis pair exists there).
- **B.2 (keep atom interaction):** construct a same-axis pair explicitly on a bounded axis. E.g., matmul M canonical `(16, 128)`; `Split(outer_iv, factor=4)` → `(4, 4, 128)`. Now there's a `(4, 4)` same-axis pair on d1 to Fuse.

Use B.2 for tests that specifically exercise Fuse+Split composition (test_axis_identity, test_fuse_eager_rewrite). Use B.1 for tests that just need a sample iter-var (most test_fuse tests).

### Step 3: Iteratively fix each test file

For each failing test file, run it individually, diagnose, fix, re-run. Move to the next when it passes.

### Step 4: Verify full suite passes

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS (+ 2 skipped + xfailed from prior work).

### Step 5: Verify `test_batch.py` only has the pre-existing failure

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_batch.py --tb=line 2>&1 | tail -5
```

Expected: 10 pass, 1 pre-existing fail.

### Step 6: Do not commit yet

---

## Task 4: Update `kernel_transforms.py` reference

**Files:**
- Modify: `/home/ubuntu/nki-autotune/kernel_transforms.py`

The 16 hand-written kernels have `for i_dN_0 in range(1):` wrappers for unbounded axes. These matched the old renderer output. Mechanical rewrite:

### Step 1: Remove all trip-1 wrappers

For each occurrence of:
```python
for i_d<axis>_0 in range(1):
    <body referencing i_d<axis>_0>
```

Replace with the body, substituting `i_d<axis>_0` with `0` in slice expressions like `(i_d<axis>_0) * <extent> : (i_d<axis>_0) * <extent> + <extent>` → `0 : <extent>`.

### Step 2: Verify all 15 kernels still CPU-sim pass

Write `/tmp/regress_all.py`:

```python
import sys, pathlib
sys.path.insert(0, "/home/ubuntu/nki-autotune/nkigym/src")
import numpy as np
import nki
from nkigym.tune.verify import _rewrite_to_fp32

src = pathlib.Path("/home/ubuntu/nki-autotune/kernel_transforms.py").read_text()
sim = _rewrite_to_fp32(src)
ns = {}
exec(sim, ns)

K = M = N = 2048
rng = np.random.default_rng(0)
lhs_T = rng.standard_normal((K, M)).astype(np.float32)
rhs = rng.standard_normal((K, N)).astype(np.float32)
expected = lhs_T.T @ rhs

names = sorted([n for n in ns if n.startswith("kernel_")], key=lambda s: int(s.split("_")[1]))
for name in names:
    actual = np.asarray(nki.simulate(ns[name])(lhs_T, rhs))
    ok = np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"{name}: pass={ok}")
```

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /tmp/regress_all.py 2>&1 | tail -20
```

Expected: 15 `pass=True`.

### Step 3: Do not commit yet

---

## Task 5: Update example + regression + atomic commit

**Files:**
- Modify: `/home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py`

### Step 1: Switch step 1 from Fuse+Split to ComputeAt

Replace the Fuse+Split sequence with `ComputeAt(lhs_T_load, matmul_d1_outer)`. New structure:

```python
if __name__ == "__main__":
    BUILD_SPECS = {"lhs_T": {"shape": (K, M), "dtype": "bfloat16"}, "rhs": {"shape": (K, N), "dtype": "bfloat16"}}
    VERIFY_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs")

    module = build_canonical_module(matmul_lhsT_rhs_nkigym, input_specs=BUILD_SPECS)
    save_step(module, 0, "canonical")

    """Step 1: move lhs_T load inside matmul's d1 outer loop (trip 16).
    This tiles the load at matmul's M-axis granularity in one atom."""
    load_path = _find_sblock_path(module, "NKILoad", tensor="lhs_T_sbuf")
    matmul_d1_outer_path = _find_matmul_d1_outer_path(module)
    module = ComputeAt(block_path=load_path, loop_path=matmul_d1_outer_path).apply(module)
    save_step(module, 1, "compute_at_load_d1")
```

Add helpers `_find_sblock_path` and `_find_matmul_d1_outer_path` at module level. Remove `Fuse`/`Split` imports and the old `find_lhs_T_load_d1_pair` / `find_lhs_T_load_d1_loop_path` helpers. Add `from nkigym.tune.compute_at import ComputeAt`.

### Step 2: Run the driver

```bash
rm -rf /home/ubuntu/cache/matmul_lhsT_rhs
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py 2>&1 | tail -20
```

Expected: 2 steps print each with `cpu-sim: PASS`. Artifacts:
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_0_canonical/{ir.txt,kernel.py}`
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_1_compute_at_load_d1/{ir.txt,kernel.py}`

Inspect step_1 kernel.py — the lhs_T load should now sit under matmul's `for i_d1_0 in range(16):` loop.

### Step 3: Regression

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /tmp/regress_all.py 2>&1 | tail -20
```

Expected: 15 `pass=True`.

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS (+ 2 skipped + xfailed).

### Step 4: Confirm no trip-1 stragglers

```bash
cd /home/ubuntu/nki-autotune
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
from nkigym.codegen.canonical import build_canonical_module
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
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a); NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s); NKIStore()(src=s, dst=h)
    return h

m = build_canonical_module(mm, input_specs={"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}})
src = render(m)
assert "range(1)" not in src, f"trip-1 loop still present:\n{src}"
print("No trip-1 loops in canonical render.")
PY
```

### Step 5: Atomic commit

```bash
cd /home/ubuntu/nki-autotune
git add -A
git commit -m "refactor: canonical drops trip-1 outer loops for unbounded-MAX axes

For unbounded axes (MAX_TILE_SIZE=None), canonical now emits ONE ForNode
with trip = full extent, instead of (outer_trip_1 × inner_tile_full).
Bounded axes unchanged: still (outer_trip × inner_tile). Renderer
unchanged: innermost-per-axis still elided into the ISA slice.

Collapses the k0 → k1 atom chain: where Fuse+Split was the only path
through canonical pathology, ComputeAt(load, matmul_d1_outer) is now
the natural one-atom sequence.

Production:
- canonical._make_sblock allocates 1 iter-var for unbounded axes (was 2),
  2 for bounded (unchanged).
- canonical._build_buffer_access emits one coeff entry for unbounded
  axes, two for bounded.
- Tree wrap: one ForNode per unbounded axis.
- Renderer untouched.

Tests: updated iter-var count assertions and fixtures that relied on
unbounded-axis trip-1 outer; same-axis Fuse fixtures reconstructed via
Split on matmul M (bounded axis) for genuine (outer, inner) pairs.

Reference: kernel_transforms.py 16 kernels rewritten to drop
for i_dN_0 in range(1): wrappers; all 15 CPU-sim pass.

Example: examples/matmul_lhsT_rhs.py step 1 switched to
ComputeAt(lhs_T_load, matmul_d1_outer) — one atom reaches the k1
structural target.

Regression: 15/15 kernel_transforms.py kernels pass; full test suite
passes (modulo pre-existing test_batch failure).

Spec: docs/superpowers/specs/2026-05-13-canonical-drop-trip-1-loops-design.md
Plan: docs/superpowers/plans/2026-05-13-canonical-drop-trip-1-loops.md"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Unbounded axes emit 1 iter-var → Task 2 Step 1.
   - Bounded axes emit 2 iter-vars (unchanged) → Task 2 Step 1.
   - `_build_buffer_access` handles the 1-iter-var case → Task 2 Step 2.
   - Tree wrap emits 1 ForNode for unbounded → Task 2 Step 3.
   - Renderer untouched → explicit "Do NOT modify renderer" in File Map.
   - kernel_transforms.py rewritten → Task 4.
   - Example driver simplified → Task 5 Step 1.
   - Test migration → Task 3.

2. **Placeholder scan:** no TBDs or "similar to Task N" references. Code blocks are complete.

3. **Type consistency:** `dim_to_outer: dict[int, IterVar]` and `dim_to_inner: dict[int, IterVar]` consistent. `axis_id: int` keys.

4. **Ordering:** Test first → production change → test migration → reference cleanup → example + commit. Each task has a clear gate.
