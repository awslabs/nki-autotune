# Canonical Drops Trip-1 Outer Loops — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonical builder emits one ForNode per bounded op axis (trip = extent / MAX) and **zero** ForNodes for unbounded axes (ISA call covers full extent). Remove renderer's innermost-tile elision — no elided loops exist anymore.

**Architecture:** `_make_sblock` allocates one iter-var per bounded axis, zero iter-vars per unbounded axis. `BufferAccess.iter_var_coeffs` has one entry per bounded axis; unbounded axes have no entry and `AccessRange.extent = full_axis_extent`. Renderer emits every ForNode as a Python `for` (no elision).

**Tech Stack:** Python 3.12, existing nkigym IR, pytest.

---

## Spec Reference

`docs/superpowers/specs/2026-05-13-canonical-drop-trip-1-loops-design.md`

## File Map

**Production:**
- `nkigym/src/nkigym/codegen/canonical.py` — `_make_sblock` allocates one iter-var per bounded axis, zero per unbounded. `_build_buffer_access` drops outer-inner pair logic; single iter-var per bounded axis with `coeff = tile`.
- `nkigym/src/nkigym/codegen/lowering/emit_source.py` — delete `_innermost_tile_iter_var_ids`; `_emit_node` drops the elision branch.
- `nkigym/src/nkigym/codegen/lowering/_emit_utils.py` — remove `EmitCtx.innermost_tile_ids` field; remove elision branches in `_emit_affine_start` / `_emit_index_expr`.
- `kernel_transforms.py` — mechanical rewrite: drop `for i_dN_0 in range(1):` wrappers (all 16 kernels).

**Tests:**
- `test/codegen/test_canonical.py` — update assertions on iter-var counts per SBlock.
- `test/codegen/test_render_tile_elision.py` — delete (renderer no longer elides).
- `test/tune/test_axis_identity.py` — rewrite fixtures to construct `(outer, inner)` pair via Split on a bounded axis (matmul M), since unbounded-axis canonical no longer has trip-1 outer.
- `test/tune/test_fuse.py`, `test_fuse_eager_rewrite.py` — similar fixture rewrites.
- `test/tune/test_split.py`, `test_reorder.py`, `test_compute_at.py`, `test_reverse_compute_at.py` — audit tests that locate specific ForNodes by extent and update if they used trip-1 markers; most should be fine since they target bounded-axis loops.
- `test/codegen/test_rfactor_rmw.py`, `test_rfactor_slot.py` — audit.
- `test/codegen/test_batch.py` — audit.
- `test/codegen/test_first_class_buffers.py` — audit.
- `test/codegen/test_place_buffers.py` — audit.
- `examples/matmul_lhsT_rhs.py` — update to demonstrate the new cleaner atom chain: canonical → `ComputeAt(lhs_T_load, matmul_d1_outer)` as step 1.

## Execution Order

One atomic production change (Tasks 1-3 are one commit), then test migration (Task 4), then reference kernel rewrite (Task 5), then regression + example driver update (Task 6).

---

## Task 1: Failing canonical-structure test

**Files:**
- Create: `test/codegen/test_canonical_one_loop_per_axis.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_axis.py`:

```python
"""Canonical emits one ForNode per bounded op axis, zero per unbounded axis."""

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


def _find_sblock(module, op_name):
    def walk(node, ancestors):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == op_name:
            return ancestors
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


def test_nkiload_has_no_loop_for_unbounded_F_axis():
    """NKILoad's F axis is unbounded (MAX_TILE_SIZE.F=None) -> no ForNode."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    ancestors = _find_sblock(module, "NKILoad")
    """Ancestors contain only P (d0) loop; no F (d1) loop."""
    d0 = module.axis_id_by_name("d0")
    axis_ids = {iv.axis_id for iv in ancestors}
    assert d0 in axis_ids, f"expected d0 ancestor, got {[(module.axes[a].name, a) for a in axis_ids]}"
    """Exactly one loop per bounded axis — P here with extent 16 (2048/128)."""
    d0_ivs = [iv for iv in ancestors if iv.axis_id == d0]
    assert len(d0_ivs) == 1
    assert d0_ivs[0].extent == 16


def test_matmul_has_one_loop_per_bounded_axis():
    """Matmul K/M/N are bounded (128/128/512) -> one ForNode each."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    ancestors = _find_sblock(module, "NKIMatmul")
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    assert len(by_axis[d0]) == 1 and by_axis[d0][0].extent == 16
    assert len(by_axis[d1]) == 1 and by_axis[d1][0].extent == 16
    assert len(by_axis[d3]) == 1 and by_axis[d3][0].extent == 4


def test_nkimemset_f_axis_has_no_loop():
    """NKIMemset's F axis is unbounded -> no ForNode."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    ancestors = _find_sblock(module, "NKIMemset")
    d3 = module.axis_id_by_name("d3")
    d3_ivs = [iv for iv in ancestors if iv.axis_id == d3]
    assert len(d3_ivs) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_axis.py -v 2>&1 | tail -15
```

Expected: all 3 tests FAIL. Today canonical emits 2 ForNodes per axis regardless of MAX_TILE_SIZE.

- [ ] **Step 3: Do not commit yet**

---

## Task 2: Modify canonical builder

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/canonical.py`

- [ ] **Step 1: Update `_make_sblock` iter-var allocation**

Open `canonical.py`. Find `_make_sblock` (around line 512). Replace the iter-var allocation loop:

Current (outer + inner per axis):
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

New (one iter-var per bounded axis, zero per unbounded):
```python
axis_to_iv: dict[int, IterVar] = {}
iter_vars: list[IterVar] = []
for axis_id in op.touched_axes:
    total = module.axes[axis_id].total_size
    tile = op.axis_tile[axis_id]
    if tile < total:
        """Bounded: one loop with trip = total / tile."""
        trip = total // tile
        role = op.axis_role.get(axis_id, AxisRole.PARALLEL)
        iv = module.allocate_iter_var(axis_id, trip, role)
        axis_to_iv[axis_id] = iv
        iter_vars.append(iv)
    else:
        """Unbounded (tile == total): no loop. Access pattern has
        no iter-var entry for this axis and extent = total."""
        pass
```

- [ ] **Step 2: Update `_build_buffer_access` signature + body**

Same file. Find `_build_buffer_access` (around line 564). Replace the signature and coefficient-building logic:

Current (takes `dim_to_outer` + `dim_to_inner`):
```python
def _build_buffer_access(
    tensor: Tensor,
    op: _ParsedOp,
    dim_to_outer: dict[int, IterVar],
    dim_to_inner: dict[int, IterVar],
    module: KernelModule,
) -> BufferAccess:
    ...
    for i, dim_id_str in enumerate(tensor.dim_ids):
        axis_id = module.axis_id_by_name(dim_id_str)
        if axis_id in dim_to_outer:
            tile = op.axis_tile[axis_id]
            coeffs[dim_to_outer[axis_id].var_id] = tile
            coeffs[dim_to_inner[axis_id].var_id] = 1
            extent = tile
        ...
```

New (takes `axis_to_iv`, handles unbounded axes):
```python
def _build_buffer_access(
    tensor: Tensor,
    op: _ParsedOp,
    axis_to_iv: dict[int, IterVar],
    module: KernelModule,
) -> BufferAccess:
    coeffs: dict[int, int] = {}
    pattern: list[AccessRange] = []
    used_iv_ids: list[int] = []
    for i, dim_id_str in enumerate(tensor.dim_ids):
        axis_id = module.axis_id_by_name(dim_id_str)
        axis_total = module.axes[axis_id].total_size
        if axis_id in axis_to_iv:
            """Bounded axis: loop exists, coefficient = tile, extent = tile."""
            tile = op.axis_tile[axis_id]
            iv = axis_to_iv[axis_id]
            per_dim_coeffs = {iv.var_id: tile}
            used_iv_ids.append(iv.var_id)
            extent = tile
        else:
            """Unbounded axis: no loop, no iter-var entry, extent = full axis."""
            per_dim_coeffs = {}
            extent = axis_total
        pattern.append(AccessRange.make(per_dim_coeffs, const_offset=0, extent=extent))
    return BufferAccess(tensor_name=tensor.name, iter_var_ids=tuple(used_iv_ids), pattern=tuple(pattern))
```

- [ ] **Step 3: Update call sites of `_build_buffer_access` in `_make_sblock`**

Search `_make_sblock` for `_build_buffer_access(` calls. Replace `dim_to_outer=dim_to_outer, dim_to_inner=dim_to_inner` kwargs with `axis_to_iv=axis_to_iv`.

- [ ] **Step 4: Update tree-building to match reduced loop count**

Find the tree-building section in `_make_sblock` (after the SBlock is constructed). The current code wraps the SBlock in `for outer in reversed(op.touched_axes): ForNode(outer) > ForNode(inner) > ...`. Change to wrap only with iter-vars that exist:

```python
tree: ForNode | SBlock = sblock
for axis_id in reversed(op.touched_axes):
    if axis_id in axis_to_iv:
        tree = ForNode(iter_var=axis_to_iv[axis_id], children=[tree], name=None, annotations={})
"""Unbounded axes contribute no ForNode wrapper."""
```

- [ ] **Step 5: Run the Task 1 tests and existing canonical tests**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_canonical_one_loop_per_axis.py /home/ubuntu/nki-autotune/test/codegen/test_canonical.py -v 2>&1 | tail -25
```

Expected: Task 1 tests PASS. Existing `test_canonical.py` tests may fail (they assert 2-iter-vars-per-axis). That's expected; Task 4 migrates them.

- [ ] **Step 6: Do not commit yet**

---

## Task 3: Remove renderer elision

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/emit_source.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/_emit_utils.py`
- Delete: `/home/ubuntu/nki-autotune/test/codegen/test_render_tile_elision.py`

- [ ] **Step 1: Remove `_innermost_tile_iter_var_ids` + elision branch from `emit_source.py`**

Delete the function `_innermost_tile_iter_var_ids` (around line 40-62) entirely.

In `emit_source` (around line 21), remove the `innermost_tile_ids=_innermost_tile_iter_var_ids(module)` kwarg from the `EmitCtx` constructor:

Before:
```python
ctx = EmitCtx(
    iter_var_to_name={},
    tensors=module.tensors,
    module=module,
    innermost_tile_ids=_innermost_tile_iter_var_ids(module),
)
```

After:
```python
ctx = EmitCtx(
    iter_var_to_name={},
    tensors=module.tensors,
    module=module,
)
```

In `_emit_node` (around line 89), remove the elision branch:

Before:
```python
if iv.var_id in ctx.innermost_tile_ids:
    ctx.iter_var_to_name[iv.var_id] = "0"
    for child in node.children:
        _emit_node(w, child, ctx)
    ctx.iter_var_to_name.pop(iv.var_id, None)
    return
```

Delete that block — every ForNode now emits a `for` header.

- [ ] **Step 2: Remove `innermost_tile_ids` from `EmitCtx`**

Open `_emit_utils.py`. Find `EmitCtx` (around line 60). Remove the `innermost_tile_ids: set[int] = field(default_factory=set)` field and its docstring entry.

Remove elision branches in `_emit_affine_start` and `_emit_index_expr`:

Before:
```python
if iv_id in ctx.innermost_tile_ids:
    continue
```

Delete both occurrences.

- [ ] **Step 3: Delete the tile-elision test file**

```bash
rm /home/ubuntu/nki-autotune/test/codegen/test_render_tile_elision.py
```

- [ ] **Step 4: Verify production imports don't break**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python -c "
from nkigym.codegen.render import render
from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.lowering._emit_utils import EmitCtx
print('imports: ok')
"
```

Expected: `imports: ok`.

- [ ] **Step 5: Smoke-test render on the repro kernel**

```bash
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

module = build_canonical_module(mm, input_specs={"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}})
source = render(module)
print(source)
"range(1)" in source and print(">>> TRIP-1 LOOP STILL PRESENT <<<")
PY
```

Expected: rendered source contains no `range(1)` loops. Only bounded-axis loops appear (trip 16, 16, 4 for matmul; trip 16 for loads/store/tensor_copy on d0 or d1).

- [ ] **Step 6: Do not commit yet**

---

## Task 4: Test migration

**Files:**
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_canonical.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_axis_identity.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_fuse.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_split.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_reorder.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_compute_at.py`
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_reverse_compute_at.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_rfactor_rmw.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_rfactor_slot.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_batch.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_first_class_buffers.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_place_buffers.py`

- [ ] **Step 1: Run full suite, inventory failures**

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py --tb=line 2>&1 | tail -40
```

Record the failing test names and line numbers.

- [ ] **Step 2: Categorize failures**

For each failing test, determine its category:

**Category A (fixture relied on trip-1 loop):** Test looks up a ForNode by `iv.extent == 1` or iterates through 2 iter-vars per axis. These need fixture rewrite to a bounded-axis target.

**Category B (legitimate assertion change):** Test asserts "2 iter-vars per axis" or "outer and inner ForNode". These need assertion update to the new "1 iter-var per bounded axis, 0 per unbounded" reality.

**Category C (broken by production change in unexpected way):** Surface an actual regression in production code. If you hit this, STOP and report BLOCKED with details.

- [ ] **Step 3: Fix Category A tests using matmul M bounded axis**

Example rewrite for `test/tune/test_axis_identity.py::_find_lhs_t_load_d1_pair`:

Before: finds `(outer_d1_trip1, inner_d1_trip2048)` above lhs_T load.
After: finds matmul's d1 ForNode (trip=16, bounded), then calls `Split(loop_path=matmul_d1_path, factor=4)` to produce an `(outer_4, inner_4)` same-axis pair. Then the Fuse+Split tests proceed as before on this constructed pair.

Apply this pattern to any test that previously relied on unbounded-axis trip-1 wrappers.

- [ ] **Step 4: Fix Category B tests**

Update assertions. E.g. in `test_canonical.py`:

Before:
```python
matmul_ops = ...
assert len(matmul_ops[0].iter_vars) == 6  # 2 per axis: outer+inner for K, M, N
```

After:
```python
matmul_ops = ...
assert len(matmul_ops[0].iter_vars) == 3  # 1 per bounded axis: K, M, N (all bounded)
```

- [ ] **Step 5: Run full suite iteratively**

After each file's fixes, re-run the tests for that file. When all tests pass:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS (+ 2 skipped + N xfailed). The xfail count may change; document the new baseline.

- [ ] **Step 6: Confirm `test_batch.py` only has the pre-existing failure**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/codegen/test_batch.py --tb=line 2>&1 | tail -5
```

Expected: only `test_enumerate_pool_includes_annotate_buffer_degree_variants` fails (pre-existing).

- [ ] **Step 7: Do not commit yet**

---

## Task 5: Update `kernel_transforms.py` reference

The 16 hand-written kernels currently have `for i_dN_0 in range(1):` wrappers. These were a cosmetic match for the old renderer output. After the refactor, the rendered source has no such wrappers; the reference needs the same cleanup.

**Files:**
- Modify: `/home/ubuntu/nki-autotune/kernel_transforms.py`

- [ ] **Step 1: Mechanical rewrite**

For each `for i_d<axis>_0 in range(1):` line followed by a body that uses `i_d<axis>_0 * <extent>` in slice expressions, replace with the body directly, substituting `i_d<axis>_0` with `0`.

Concretely, each `range(1)` wrapper will be deleted, and slice expressions like `(i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048` simplify to `0:2048`.

There are ~20 such occurrences across the 16 kernels. Apply consistently.

- [ ] **Step 2: Run `kernel_transforms.py` to verify all 15 kernels still pass CPU-sim**

Write a small script `/tmp/regress_all.py` (reuse from prior work):

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

Expected: 15 kernels, all `pass=True`.

- [ ] **Step 3: Do not commit yet**

---

## Task 6: Update the example driver + regression + commit

**Files:**
- Modify: `/home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py`

- [ ] **Step 1: Switch the example from Fuse+Split to ComputeAt**

Replace step 1's Fuse+Split sequence with `ComputeAt(lhs_T_load, matmul_d1_outer)`. New driver flow:

```python
if __name__ == "__main__":
    BUILD_SPECS = {"lhs_T": {"shape": (K, M), "dtype": "bfloat16"}, "rhs": {"shape": (K, N), "dtype": "bfloat16"}}
    VERIFY_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs")

    module = build_canonical_module(matmul_lhsT_rhs_nkigym, input_specs=BUILD_SPECS)
    save_step(module, 0, "canonical")

    """Step 1: move lhs_T load inside matmul's d1 outer loop. This is the
    natural way to express 'tile the producer at the consumer's granularity'
    without any Fuse+Split gymnastics."""
    load_path = find_sblock_path(module, "NKILoad", tensor="lhs_T_sbuf")
    matmul_d1_loop_path = find_loop_path(module, "NKIMatmul", axis_name="d1")
    module = ComputeAt(block_path=load_path, loop_path=matmul_d1_loop_path).apply(module)
    save_step(module, 1, "compute_at_load_d1")
```

Add import: `from nkigym.tune.compute_at import ComputeAt`. Remove unused `Fuse` and `Split` imports.

Add helper functions `find_sblock_path` and `find_loop_path` (simple tree walks); the existing `find_lhs_T_load_d1_pair` helper is no longer needed — delete it.

- [ ] **Step 2: Run the driver**

```bash
rm -rf /home/ubuntu/cache/matmul_lhsT_rhs
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py 2>&1 | tail -20
```

Expected: 2 steps print, each `cpu-sim: PASS`. Artifacts exist at:
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_0_canonical/{ir.txt,kernel.py}`
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_1_compute_at_load_d1/{ir.txt,kernel.py}`

Inspect step 1's kernel.py — the lhs_T load should sit inside matmul's d1 outer loop.

- [ ] **Step 3: Regression on kernel_transforms.py**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /tmp/regress_all.py 2>&1 | tail -20
```

Expected: 15 `pass=True`.

- [ ] **Step 4: Full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ --ignore=test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS (+ 2 skipped + N xfailed).

- [ ] **Step 5: Check no trip-1 stragglers in production**

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
assert "range(1)" not in src, f"trip-1 loop still in canonical render:\n{src}"
print("No trip-1 loops in canonical render.")
PY
```

Expected: "No trip-1 loops in canonical render."

- [ ] **Step 6: Commit atomic refactor**

```bash
cd /home/ubuntu/nki-autotune
git add -A
git commit -m "refactor: canonical drops trip-1 outer loops for unbounded-MAX axes

Each op axis emits one ForNode if MAX_TILE_SIZE is bounded (trip =
extent/MAX), zero ForNodes if unbounded (ISA call covers full extent).
Renderer loses innermost-tile elision — no elided loops left.

Collapses k0 → k1 from Fuse+Split to one ComputeAt(load, matmul_d1_outer)
call. Matches hand-written atom idiom; eliminates canonical pathology
that required Fuse gymnastics.

Production changes:
- canonical.py _make_sblock allocates one iter-var per bounded axis
  (none per unbounded); _build_buffer_access takes axis_to_iv (was
  dim_to_outer/dim_to_inner); tree wrap only creates ForNodes for axes
  that have iter-vars.
- emit_source.py deletes _innermost_tile_iter_var_ids; _emit_node drops
  the elision branch; every ForNode emits as a python for.
- _emit_utils.py removes EmitCtx.innermost_tile_ids and the elision
  branches in _emit_affine_start / _emit_index_expr.

Tests:
- test/codegen/test_canonical_one_loop_per_axis.py (new) — regression.
- test/codegen/test_render_tile_elision.py deleted (premise gone).
- Fixtures across test/tune and test/codegen updated to construct the
  (outer, inner) same-axis pair explicitly via Split on a bounded axis
  (e.g. matmul K/M/N) rather than relying on unbounded-axis trip-1
  wrappers.

Reference:
- kernel_transforms.py's 16 kernels rewritten to drop for i_dN_0 in
  range(1): wrappers. All 15 CPU-sim cases pass.

Example:
- examples/matmul_lhsT_rhs.py step 1 switches from Fuse+Split to
  ComputeAt(lhs_T_load, matmul_d1_outer).

Regression: 15/15 kernel_transforms.py kernels pass CPU-sim;
full unit test suite passes (modulo pre-existing test_batch failure).

Spec: docs/superpowers/specs/2026-05-13-canonical-drop-trip-1-loops-design.md
Plan: docs/superpowers/plans/2026-05-13-canonical-drop-trip-1-loops.md"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Canonical emits one ForNode per bounded axis → Task 1 + Task 2 Step 1.
   - Canonical emits zero ForNodes for unbounded axes → Task 1 + Task 2 Step 1.
   - Renderer loses elision → Task 3.
   - BufferAccess has no entry for unbounded axes → Task 2 Step 2.
   - AccessRange.extent for unbounded = full axis extent → Task 2 Step 2.
   - Tests migrated → Task 4.
   - kernel_transforms.py updated → Task 5.
   - Example driver simplified → Task 6 Step 1.

2. **Placeholder scan:** Step 5 of Task 5 says "~20 such occurrences across the 16 kernels" — not a TODO, just a mechanical count estimate. All code blocks are complete.

3. **Type consistency:** `axis_to_iv: dict[int, IterVar]` consistent with Task 2 Step 2 signature. `op.touched_axes`, `op.axis_tile`, `op.axis_role` consistent with existing `_ParsedOp` dataclass fields.

4. **Task ordering:** Tasks 1-3 are one commit's worth of production change (can be done sequentially by a single agent). Task 4 migrates tests. Task 5 updates the reference kernels. Task 6 wraps up with the example + regression + single commit.
