# SSA Ops Backend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `f_nkigym` frontend from destination-passing style (DPS) into single-static-assignment (SSA): each op allocates and returns its own output; `NKIAlloc`, `dst=`, and authored `NKIMemset` disappear from kernel source.

**Architecture:** Ops gain `OUTPUT_LOCATION` and `_run` methods that allocate-and-return. The trace (`dimension_analysis`) synthesizes each op's output `_Sym` from the unified axis-map, naming it from the assignment LHS. `canonical_build` synthesizes the PSUM-zeroing memset from `RMW_OPERANDS`. Buffer dtype is one logical value (propagated from the first kernel input); PSUM's fp32 is a render-time physical override (`Buffer.physical_dtype()`, paralleling the shipped `physical_shape()`).

**Tech Stack:** Python 3.12, numpy, networkx, pytest. Activate `~/venvs/kernel-env`. Subprocess tests need `PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src`.

**Reference spec:** `docs/superpowers/specs/2026-05-31-ssa-ops-backend-refactor-design.md`

---

## Spec corrections folded into this plan

Two points in the spec are imprecise; this plan uses the corrected forms:

1. **Primary output slot.** Spec line 124 says "non-`INPUT_OPERANDS`, non-`RMW_OPERANDS`" — but matmul's `dst` is RMW, so that rule yields no output for matmul. **Corrected rule:** a slot is an *output* iff it is not in `INPUT_OPERANDS` (so matmul `dst` and activation_reduce `dst`+`reduce_res` count). The *primary* returned slot is `reduce_res` if the op declares it, else `dst`. RMW-ness only affects the reads/writes split in `canonical_build`, never output synthesis.

2. **PSUM dtype.** Spec Decision 4 says PSUM "allocated fp32 but carries its logical dtype forward." This plan implements that as: every `Buffer.dtype` is the *logical* dtype (uniformly propagated from the first kernel input); a new `Buffer.physical_dtype()` returns `"float32"` when `location == "psum"` else `self.dtype`, and `_emit_alloc` uses it — symmetric with the already-shipped `physical_shape()`.

---

## File Structure

**Ops backend** — each op keeps IR metadata (`NAME`, `OPERAND_AXES`, `INPUT_OPERANDS`, `RMW_OPERANDS`, `AXIS_ROLES`, `MIN/MAX_TILE_SIZE`) and `_check_roles`; `_run` rewritten to allocate-and-return; gains `OUTPUT_LOCATION`.
- `ops/base.py` — add `OUTPUT_LOCATION` ClassVar; `_run` contract no longer takes `dst`.
- `ops/load.py`, `store.py`, `tensor_copy.py`, `matmul.py`, `transpose.py`, `dma_transpose.py`, `activation.py`, `tensor_reduce.py`, `tensor_scalar.py`, `activation_reduce.py` — rewrite `_run`; add `OUTPUT_LOCATION`.
- `ops/memset.py` — **kept** as IR-internal carrier; `_run` unchanged (DPS-style, only ever synthesized).
- `ops/alloc.py` — **deleted**.

**Trace / IR build**
- `ir/dimension_analysis.py` — SSA name collector, output synthesis, dtype propagation, seed param dtype from specs, drop `NKIAlloc` branch + `_infer_param_dtypes`. Specs type → `dict[str, tuple[tuple[int, ...], str]]`.
- `ir/canonical_build.py` — synthesized memset; drop `NKIAlloc` filter.
- `ir/ir.py` — `build_initial_ir` specs type; param_buffers dtype from specs.
- `ir/tree.py` — add `Buffer.physical_dtype()`; `ISANode.label()` → `nisa.<NAME>`.
- `ir/tree_visualize.py` — drop `NKIAlloc` special-case + `"alloc"` CSS bucket.

**Codegen**
- `codegen/body.py` — `_emit_alloc` uses `buf.physical_dtype()`.

**Environment / example / sim**
- `environment/mdp.py` — specs type hint.
- `examples/matmul_lhsT_rhs.py` — SSA `f_nkigym`; `INPUT_SPECS` with dtype.
- `synthesis/numpy_to_nkigym.py` — rewrite `_SYSTEM_PROMPT` for SSA.

**Tests** — migrate fixtures to SSA + `(shape, dtype)` specs; update label/ISA-name assertions.

### Sequencing note (read before starting)

Groups A–C are **green-preserving prerequisites** (each commit leaves the full suite passing). Group D is the **SSA cutover** — ops `_run`, the trace, `canonical_build`, and every fixture flip together; the full suite is transiently red across D1–D4 and goes green at **D4**. Groups E–F finalize. Do not claim suite-green between D1 and D4; the gate is D4's full-suite run.

---

## Task A1: `(shape, dtype)` input_specs — analyze_dimensions

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py:83-104`
- Test: `test/ir/test_ir_extensions.py`

This is a green-preserving prerequisite: specs carry dtype, param dtype seeds from the spec (replacing `_infer_param_dtypes`), DPS allocs still work.

- [ ] **Step 1: Write the failing test**

Add to `test/ir/test_ir_extensions.py`:

```python
def test_param_dtype_seeded_from_spec():
    """analyze_dimensions reads param dtype from the (shape, dtype) spec, not from buffer inference."""
    from nkigym.ir.dimension_analysis import analyze_dimensions
    from test.transforms._fixtures import f_matmul

    analysis = analyze_dimensions(f_matmul, {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")})
    assert analysis.tensors["lhs_T"].dtype == "bfloat16"
    assert analysis.tensors["rhs"].dtype == "bfloat16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_ir_extensions.py::test_param_dtype_seeded_from_spec -v`
Expected: FAIL — `analyze_dimensions` still expects `{name: shape}`, raises on tuple unpack or KeyError.

- [ ] **Step 3: Change the signature + param seeding**

In `dimension_analysis.py`, change `analyze_dimensions` signature and param sentinel construction (lines 83-104). Replace:

```python
def analyze_dimensions(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> _AnalysisResult:
    """Trace ``func`` against sentinel inputs and run cross-op dim unification.

    Args:
        func: An ``@nkigym_kernel``-decorated callable to analyse.
        input_specs: ``{param_name: shape}`` for every positional parameter.
    """
    unwrapped = inspect.unwrap(func)
    param_names = list(inspect.signature(unwrapped).parameters)
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    state = _TraceState(alloc_names=_collect_alloc_names(unwrapped))
    for name in param_names:
        sym = _Sym(tuple(input_specs[name]), name)
        sym.location = "shared_hbm"
        state.sentinels[name] = sym

    _run_trace(unwrapped, [state.sentinels[n] for n in param_names], state)
    _canonicalize_dim_names(state)
    _infer_param_dtypes(state, param_names)
```

with:

```python
def analyze_dimensions(
    func: Callable[..., Any], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> _AnalysisResult:
    """Trace ``func`` against sentinel inputs and run cross-op dim unification.

    Args:
        func: An ``@nkigym_kernel``-decorated callable to analyse.
        input_specs: ``{param_name: (shape, dtype)}`` for every positional parameter.
    """
    unwrapped = inspect.unwrap(func)
    param_names = list(inspect.signature(unwrapped).parameters)
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    state = _TraceState(alloc_names=_collect_alloc_names(unwrapped))
    for name in param_names:
        shape, dtype = input_specs[name]
        sym = _Sym(tuple(shape), name)
        sym.location = "shared_hbm"
        sym.dtype = dtype
        state.sentinels[name] = sym

    _run_trace(unwrapped, [state.sentinels[n] for n in param_names], state)
    _canonicalize_dim_names(state)
```

(Drops the `_infer_param_dtypes(state, param_names)` call — param dtype now comes from the spec.)

- [ ] **Step 4: Delete `_infer_param_dtypes`**

Remove the entire `_infer_param_dtypes` function (`dimension_analysis.py:219-244`). It is now dead.

- [ ] **Step 5: Run the new test**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_ir_extensions.py::test_param_dtype_seeded_from_spec -v`
Expected: PASS

- [ ] **Step 6: Update all callers + fixtures to the new spec shape (keep suite green)**

Update every `input_specs`/`INPUT_SPECS` to `{name: (shape, dtype)}` and `build_initial_ir`/`analyze_dimensions` call sites. Files (search `grep -rln "INPUT_SPECS\|input_specs\|analyze_dimensions\|build_initial_ir" test/ nkigym/ examples/ | grep -v __pycache__`):

- `nkigym/src/nkigym/ir/ir.py:119` — change `build_initial_ir` signature to `input_specs: dict[str, tuple[tuple[int, ...], str]]`; in `param_buffers`, replace `shape=tuple(analysis.tensors[name].shape)` line's surrounding dict comprehension to read dtype from `analysis.tensors[name].dtype` (already does). The param_buffers comprehension already pulls dtype from `analysis.tensors` — no change needed there beyond the signature type.
- `nkigym/src/nkigym/environment/mdp.py:30` — type hint `input_specs: dict[str, tuple[tuple[int, ...], str]]`.
- `test/transforms/_fixtures.py:19` — `INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}`.
- `test/environment/_fixtures.py`, `test/codegen/test_render.py`, `test/codegen/test_header.py`, `test/environment/test_mdp.py`, `test/ops/test_tile_bounds.py` — update any `INPUT_SPECS`/inline specs to `(shape, dtype)` form (dtype `"bfloat16"` for the matmul fixtures, matching the DPS allocs).
- `examples/matmul_lhsT_rhs.py:38` — `INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}`.

- [ ] **Step 7: Run the full suite**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS (all green — DPS kernels still trace; only param-dtype source changed).

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/ir/dimension_analysis.py nkigym/src/nkigym/ir/ir.py nkigym/src/nkigym/environment/mdp.py test/ examples/matmul_lhsT_rhs.py
git commit -m "Carry dtype in input_specs; seed param dtype from spec

input_specs becomes {name: (shape, dtype)}; param dtype reads from the
spec directly, replacing _infer_param_dtypes (deleted). Green-preserving
prerequisite for the SSA cutover."
```

---

## Task A2: `TensorDims` docstring + `Buffer.physical_dtype()`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (Buffer class)
- Modify: `nkigym/src/nkigym/codegen/body.py:65-71` (`_emit_alloc`)
- Test: `test/ir/test_node_labels.py`

- [ ] **Step 1: Write the failing test**

Add to `test/ir/test_node_labels.py`:

```python
def test_buffer_physical_dtype_overrides_psum_to_fp32():
    """physical_dtype() returns fp32 for psum (HW accumulation), logical dtype otherwise."""
    from nkigym.ir.tree import Buffer

    psum = Buffer(name="p", shape=(256, 512), dtype="bfloat16", location="psum")
    sbuf = Buffer(name="s", shape=(256, 512), dtype="bfloat16", location="sbuf")
    hbm = Buffer(name="h", shape=(256, 512), dtype="bfloat16", location="shared_hbm")
    assert psum.physical_dtype() == "float32"
    assert sbuf.physical_dtype() == "bfloat16"
    assert hbm.physical_dtype() == "bfloat16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_node_labels.py::test_buffer_physical_dtype_overrides_psum_to_fp32 -v`
Expected: FAIL — `Buffer` has no `physical_dtype`.

- [ ] **Step 3: Add `physical_dtype()` to `Buffer`**

In `tree.py`, add directly after `Buffer.physical_shape()`:

```python
    def physical_dtype(self) -> str:
        """Return the dtype ``nl.ndarray`` actually allocates for this buffer.

        ``psum`` buffers are always allocated ``float32`` — the matmul HW
        accumulates at fp32 regardless of the logical dtype the value
        carries. Every other location uses the logical :attr:`dtype`. This
        is the physical override paired with :meth:`physical_shape`.
        """
        if self.location == "psum":
            return "float32"
        return self.dtype
```

- [ ] **Step 4: Route `_emit_alloc` through it**

In `body.py`, change `_emit_alloc`'s last line. Replace:

```python
    shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.dtype}, buffer=nl.{buf.location})"
```

with:

```python
    shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"
```

- [ ] **Step 5: Run tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_node_labels.py test/codegen/ -q`
Expected: PASS. (Today psum buffers already carry `dtype="float32"` from the DPS alloc, so `physical_dtype()` returns the same fp32 → render byte-identical. The override only diverges once SSA makes psum's logical dtype bf16.)

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/codegen/body.py test/ir/test_node_labels.py
git commit -m "Add Buffer.physical_dtype: psum allocates fp32, logical dtype elsewhere

Parallels physical_shape. Render-byte-identical today (psum already
fp32); diverges once SSA makes psum logical dtype bf16."
```

---

## Task B1: `ISANode.label()` → `nisa.<NAME>`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (`ISANode.label`)
- Test: `test/ir/test_node_labels.py:55-77`

- [ ] **Step 1: Update the failing tests**

In `test/ir/test_node_labels.py`, change the two ISANode-label assertions:

`test_isanode_label_includes_op_bindings_and_kwargs` — replace `assert text.startswith("NKIMemset")` with:

```python
    assert text.startswith("nisa.memset")
```

`test_isanode_label_omits_empty_kwargs_and_bindings` — replace the assertion with:

```python
    assert ISANode(op_cls=NKIMatmul, operand_bindings={}, kwargs={}).label() == "nisa.nc_matmul"
```

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_node_labels.py::test_isanode_label_omits_empty_kwargs_and_bindings -v`
Expected: FAIL — label currently emits `NKIMatmul`.

- [ ] **Step 3: Change `ISANode.label()`**

In `tree.py`, in `ISANode.label`, replace the first list line:

```python
        lines: list[str] = [self.op_cls.__name__]
```

with:

```python
        lines: list[str] = [f"nisa.{self.op_cls.NAME}"]
```

- [ ] **Step 4: Run tests**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_node_labels.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "ISANode.label shows nisa.<NAME> instead of gym class name"
```

---

## Task D1: Ops backend — `_run` allocate-and-return + `OUTPUT_LOCATION`

> **Cutover begins.** From here through D4 the full suite is transiently red; D4 is the green gate. Commit per task to checkpoint, but verify only the task's own isolated tests until D4.

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Modify: all op files except `memset.py`, `alloc.py`
- Delete: `nkigym/src/nkigym/ops/alloc.py`
- Test: `test/ops/test_direct_numpy_call.py`

### Background: the rewrite pattern

DPS `_run` took a `dst` kwarg and wrote into it. SSA `_run` allocates its output, computes, returns it. The decorator (`NKIOp.__call__`) tags the return with `_output_role`. Inputs are still validated by `_check_roles`.

- [ ] **Step 1: Add `OUTPUT_LOCATION` to base**

In `base.py`, after the `OUTPUT_ROLE` ClassVar (line 183), add:

```python
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    """Physical residency of this op's synthesized output buffer:
    ``"sbuf"`` / ``"psum"`` / ``"shared_hbm"``. Read by the trace to set
    the output ``Buffer.location``. Distinct from ``OUTPUT_ROLE`` (the
    role-lattice lineage tag); the two coincide for every op except
    ``NKIStore`` (location ``"shared_hbm"``, role ``"stored"``)."""
```

- [ ] **Step 2: Write the SSA direct-call test**

Rewrite `test/ops/test_direct_numpy_call.py` to the SSA form. Replace the `_lhsT_matmul` kernel and its imports:

```python
import numpy as np
import pytest

from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 128, 128, 128


@nkigym_kernel
def _lhsT_matmul(lhs_T, rhs):
    lhs_T_sbuf = NKILoad()(src=lhs_T)
    rhs_sbuf = NKILoad()(src=rhs)
    psum_acc = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    sbuf_prod = NKITensorCopy()(src=psum_acc)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def test_direct_call_computes_matmul():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    expected = lhs_T.T @ rhs
    assert np.allclose(np.asarray(actual), expected, atol=1e-4, rtol=1e-4)


def test_direct_call_returns_hbm_role():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    out = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    assert getattr(out, "role", None) == "stored"


def test_load_rejects_non_param_src():
    """NKILoad's _check_roles fires when src is not an HBM param."""
    rng = np.random.default_rng(0)
    sbuf = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    with pytest.raises(TypeError, match="NKILoad.*expects HBM param"):
        NKILoad()(src=sbuf)


def test_store_rejects_non_sbuf_src():
    """NKIStore's _check_roles fires when src is PSUM instead of SBUF."""
    rng = np.random.default_rng(0)
    lhs = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    rhs = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    psum = NKIMatmul()(stationary=lhs, moving=rhs)
    with pytest.raises(TypeError, match="NKIStore.*expects sbuf"):
        NKIStore()(src=psum)
```

(Note: `test_direct_call_returns_hbm_role` now asserts role `"stored"` — the SSA store returns its own output, role `"stored"`, which the decorator accepts. The `bad_return` test moves to Task D-fixtures; drop it here.)

- [ ] **Step 3: Rewrite `NKILoad._run` + add `OUTPUT_LOCATION`**

`ops/load.py` — replace `_run` and add the ClassVar after `OUTPUT_ROLE`:

```python
    OUTPUT_ROLE: ClassVar[str] = "sbuf"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be HBM-resident (``param``)."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "param":
            raise TypeError(f"NKILoad(src=<role={role}>) expects HBM param; did you forget to load?")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate an SBUF copy of ``src`` and return it."""
        src: np.ndarray = kwargs["src"]
        return np.array(src)
```

- [ ] **Step 4: Rewrite `NKIMatmul._run` + `OUTPUT_LOCATION`**

`ops/matmul.py` — `OUTPUT_LOCATION` already implied by `OUTPUT_ROLE = "psum"`; add the ClassVar and rewrite `_run`:

```python
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
```

```python
    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``stationary.T @ moving`` (fp32)."""
        stationary: np.ndarray = kwargs["stationary"]
        moving: np.ndarray = kwargs["moving"]
        return (stationary.astype(np.float32).T @ moving.astype(np.float32))
```

- [ ] **Step 5: Rewrite `NKITensorCopy._run` + `OUTPUT_LOCATION`**

`ops/tensor_copy.py` — add ClassVar after `MAX_TILE_SIZE` and rewrite `_run`:

```python
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be SBUF- or PSUM-resident (drain pattern allows PSUM src)."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorCopy(src=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate an SBUF copy of ``src`` and return it."""
        src = kwargs["src"]
        return np.array(src)
```

- [ ] **Step 6: Rewrite `NKIStore._run` + `OUTPUT_LOCATION`**

`ops/store.py` — add ClassVar and rewrite `_run`:

```python
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be SBUF-resident."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIStore(src=<role={role}>) expects sbuf; did you forget to stage through SBUF?")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate an HBM copy of ``src`` and return it."""
        src: np.ndarray = kwargs["src"]
        return np.array(src)
```

- [ ] **Step 7: Rewrite the remaining ops' `_run` + `OUTPUT_LOCATION`**

Apply the same allocate-and-return pattern to each remaining op. Per-op specifics (all add `OUTPUT_LOCATION` after their existing ClassVars; primary output computed from inputs):

- `ops/transpose.py` (`NKITranspose`): `OUTPUT_LOCATION = "psum"`. `_run`: `src = kwargs["src"]; return np.array(src).T`.
- `ops/dma_transpose.py` (`NKIDMATranspose`): `OUTPUT_LOCATION = "sbuf"`. `_run`: `src = kwargs["src"]; return np.array(src).T`.
- `ops/activation.py` (`NKIActivation`): `OUTPUT_LOCATION = "sbuf"`. `_run`: compute `op(data * scale + bias)` (reuse the existing activation math currently writing into `dst`), allocate the result array, return it.
- `ops/tensor_reduce.py` (`NKITensorReduce`): `OUTPUT_LOCATION = "sbuf"`. `_run`: `result = _REDUCE_FNS[op](data, axis=axis); return np.asarray(result)`.
- `ops/tensor_scalar.py` (`NKITensorScalar`): `OUTPUT_LOCATION = "sbuf"`. `_run`: `broadcast = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0; return _OPS[op_name](data, broadcast)`.
- `ops/activation_reduce.py` (`NKIActivationReduce`): `OUTPUT_LOCATION = "sbuf"`. `_run`: keep the kwargs allow-list guard; compute the activated tile and the reduction; **return the reduction vector** (`reduce_res`, the primary output) — drop the `dst`/`reduce_res` writes. (Its synthesized scratch `dst` buffer is handled by the trace, Task D2.)

For each: delete the `dst=`/`reduce_res=` writes; the function now returns the computed array.

- [ ] **Step 8: Delete `ops/alloc.py`**

```bash
git rm nkigym/src/nkigym/ops/alloc.py
```

- [ ] **Step 9: Run the ops direct-call tests (isolated green)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ops/test_direct_numpy_call.py -v`
Expected: PASS. (Other suites red until D4 — expected.)

- [ ] **Step 10: Commit**

```bash
git add nkigym/src/nkigym/ops/ test/ops/test_direct_numpy_call.py
git commit -m "SSA ops backend: _run allocates-and-returns; add OUTPUT_LOCATION; delete NKIAlloc

Cutover step 1 of 4. Full suite red until canonical build + trace + fixtures
land (Task D4)."
```

---

## Task D2: Trace — SSA name collector + output synthesis

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

The matmul fixture must move to SSA first for this test to import. Update `test/transforms/_fixtures.py`'s `f_matmul` to SSA now (the canonical IR build won't work until D3, but `analyze_dimensions` will after this task):

```python
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` — load, matmul, drain, store (SSA)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def build_canonical_ir() -> KernelIR:
    """Build the canonical :class:`KernelIR` for the matmul fixture."""
    return build_initial_ir(f_matmul, INPUT_SPECS)
```

Add to `test/ir/test_ir_extensions.py`:

```python
def test_trace_synthesizes_ssa_outputs():
    """The SSA trace names each op's output from its assignment LHS and infers dims/dtype/location."""
    from nkigym.ir.dimension_analysis import analyze_dimensions
    from test.transforms._fixtures import f_matmul, INPUT_SPECS

    analysis = analyze_dimensions(f_matmul, INPUT_SPECS)
    t = analysis.tensors
    assert set(t) == {"lhs_T", "rhs", "sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"}
    assert t["psum_prod"].location == "psum"
    assert t["psum_prod"].dtype == "bfloat16"  # logical dtype propagated; physical fp32 applied at render
    assert t["sbuf_prod"].location == "sbuf"
    assert t["sbuf_prod"].dtype == "bfloat16"
    assert t["hbm_out"].location == "shared_hbm"
    """matmul output dims are (M, N) = the rhs/lhs non-K extents."""
    assert analysis.dim_sizes[t["psum_prod"].dim_ids[0]] == 2048
    assert analysis.dim_sizes[t["psum_prod"].dim_ids[1]] == 2048
```

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_ir_extensions.py::test_trace_synthesizes_ssa_outputs -v`
Expected: FAIL — hook still returns `merged.get("dst")` (None in SSA); no output sentinels synthesized.

- [ ] **Step 3: Replace the SSA name collector**

In `dimension_analysis.py`, replace `_collect_alloc_names` and `_is_alloc_call` (lines 291-313) with an SSA-call collector:

```python
def _collect_ssa_names(func: Callable[..., Any]) -> Iterator[str]:
    """Yield the LHS of every ``var = NKIOp()(...)`` assignment in source order.

    Each straight-line op call is assigned to a name; the trace consumes
    these in execution (= source) order to name synthesized outputs.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    for stmt in func_def.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and _is_op_call(stmt.value):
            yield target.id


def _is_op_call(node: ast.expr) -> bool:
    """Return True if ``node`` is an ``NKIXxx(...)(...)`` double-call (op invocation)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and isinstance(node.func.func, ast.Name)
        and node.func.func.id.startswith("NKI")
    )
```

Update the `_TraceState` constructor reference and `analyze_dimensions`: rename the field `alloc_names` → `ssa_names`. In `_TraceState.__init__` change the param and attribute name; in `analyze_dimensions` change `state = _TraceState(alloc_names=_collect_alloc_names(unwrapped))` to `state = _TraceState(ssa_names=_collect_ssa_names(unwrapped))`.

- [ ] **Step 4: Rewrite the hook for output synthesis**

Replace `_make_hook` (lines 176-193) with:

```python
def _make_hook(state: _TraceState) -> Callable[..., Any]:
    """Build a replacement for :meth:`NKIOp.__call__` that records into ``state`` and synthesizes outputs."""

    def hook(op: NKIOp, **kwargs: Any) -> Any:
        merged = {**getattr(op, "_init_kwargs", {}), **kwargs}
        cls = type(op)
        input_syms = _trace_compute_op(cls, merged, state)
        name = next(state.ssa_names)
        return _synthesize_outputs(cls, name, merged, input_syms, state)

    return hook
```

Change `_trace_compute_op` to **return** the ordered list of input `_Sym`s (it currently returns None). At the end of `_trace_compute_op`, after appending the `_OpRecord`, add:

```python
    input_syms = [kwargs[slot] for slot in cls.OPERAND_AXES if slot in cls.INPUT_OPERANDS and isinstance(kwargs.get(slot), _Sym)]
    return input_syms
```

and change its signature return type annotation to `-> list["_Sym"]`.

- [ ] **Step 5: Add `_synthesize_outputs`**

Add after `_trace_compute_op`:

```python
def _synthesize_outputs(
    cls: type[NKIOp], name: str, kwargs: dict[str, Any], input_syms: list["_Sym"], state: _TraceState
) -> "_Sym":
    """Create the output sentinel(s) for an op call; return the primary (assigned) one.

    Output slots = ``OPERAND_AXES`` keys not in ``INPUT_OPERANDS``. The
    primary slot (gets ``name``, returned to thread the SSA chain) is
    ``reduce_res`` if declared, else ``dst``. Any secondary output slot
    (e.g. activation_reduce's scratch ``dst``) gets ``f"{name}_scratch"``.

    Output dims come from the op's unified axis-map (recomputed locally
    from the input sentinels); dtype propagates from the first input's
    logical dtype; location is ``cls.OUTPUT_LOCATION``.
    """
    axis_map = _output_axis_map(cls, input_syms)
    output_slots = [slot for slot in cls.OPERAND_AXES if slot not in cls.INPUT_OPERANDS]
    primary_slot = "reduce_res" if "reduce_res" in cls.OPERAND_AXES else "dst"
    logical_dtype = input_syms[0].dtype if input_syms else None
    primary_sym: "_Sym" | None = None
    for slot in output_slots:
        slot_name = name if slot == primary_slot else f"{name}_scratch"
        axes = cls.OPERAND_AXES[slot]
        dim_ids = [axis_map[a] for a in axes]
        shape = tuple(state.dim_sizes[d] for d in dim_ids)
        sym = _Sym(shape, slot_name)
        sym.dim_ids = list(dim_ids)
        sym.location = cls.OUTPUT_LOCATION
        sym.dtype = logical_dtype
        state.sentinels[slot_name] = sym
        if slot == primary_slot:
            primary_sym = sym
    if primary_sym is None:
        raise ValueError(f"{cls.__name__}: no primary output slot {primary_slot!r} in OPERAND_AXES")
    return primary_sym


def _output_axis_map(cls: type[NKIOp], input_syms: list["_Sym"]) -> dict[str, str]:
    """Map each abstract axis appearing on inputs to its (unified) concrete dim id.

    After ``_trace_compute_op`` unifies input dims, each input ``_Sym``
    carries concrete ``dim_ids`` aligned with its ``OPERAND_AXES`` slot;
    invert that to abstract → concrete so output slots resolve.
    """
    axis_map: dict[str, str] = {}
    by_name = {s.source_name: s for s in input_syms}
    for slot, axes in cls.OPERAND_AXES.items():
        if slot not in cls.INPUT_OPERANDS:
            continue
        sym = next((s for s in input_syms if s is by_name.get(s.source_name) and slot in cls.OPERAND_AXES), None)
    """Re-walk inputs in slot order, aligning axes with each input sym's dim_ids."""
    axis_map = {}
    input_iter = iter(input_syms)
    for slot, axes in cls.OPERAND_AXES.items():
        if slot not in cls.INPUT_OPERANDS:
            continue
        sym = next(input_iter, None)
        if sym is None:
            continue
        for abstract, concrete in zip(axes, sym.dim_ids):
            if concrete is not None:
                axis_map[abstract] = concrete
    return axis_map
```

> **Implementer note:** `_output_axis_map`'s first loop is exploratory scaffolding — delete it; keep only the clean second `input_iter` walk. (Shown above as the body after the `"""Re-walk..."""` docstring line.) The final function is the `input_iter` version.

Clean final form of `_output_axis_map`:

```python
def _output_axis_map(cls: type[NKIOp], input_syms: list["_Sym"]) -> dict[str, str]:
    """Map each abstract axis on the op's inputs to its unified concrete dim id."""
    axis_map: dict[str, str] = {}
    input_iter = iter(input_syms)
    for slot, axes in cls.OPERAND_AXES.items():
        if slot not in cls.INPUT_OPERANDS:
            continue
        sym = next(input_iter, None)
        if sym is None:
            continue
        for abstract, concrete in zip(axes, sym.dim_ids):
            if concrete is not None:
                axis_map[abstract] = concrete
    return axis_map
```

(Use this clean form; ignore the scaffolded version.)

- [ ] **Step 6: Run the synthesis test**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_ir_extensions.py::test_trace_synthesizes_ssa_outputs -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/ir/dimension_analysis.py test/transforms/_fixtures.py test/ir/test_ir_extensions.py
git commit -m "SSA trace: collect output names from assignment LHS, synthesize output syms

Cutover step 2 of 4. Output dims from unified axis-map, dtype propagated
from first input, location from OUTPUT_LOCATION."
```

---

## Task D3: `canonical_build` — synthesize the memset

**Files:**
- Modify: `nkigym/src/nkigym/ir/canonical_build.py`
- Test: `test/ir/test_dependency.py`

- [ ] **Step 1: Write the failing test**

Add to `test/ir/test_dependency.py`:

```python
def test_canonical_synthesizes_memset_for_matmul():
    """A matmul (RMW dst) gets a synthesized memset sibling block zeroing its PSUM region."""
    from nkigym.ir.tree import ISANode
    from nkigym.ops.memset import NKIMemset
    from test.transforms._fixtures import build_canonical_ir

    ir = build_canonical_ir()
    memset_leaves = [
        d for nid in ir.tree.blocks() for d in ir.tree.descendants(nid)
        if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMemset
    ]
    assert len(memset_leaves) == 1, "exactly one synthesized memset for the matmul"
    memset = ir.tree.data(memset_leaves[0])
    assert memset.operand_bindings["dst"].tensor == "psum_prod"
    assert memset.kwargs == {"value": 0.0}
```

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_dependency.py::test_canonical_synthesizes_memset_for_matmul -v`
Expected: FAIL — no memset record exists in the SSA trace (no authored `NKIMemset`).

- [ ] **Step 3: Drop the `NKIAlloc` filter + import**

In `canonical_build.py`, remove `from nkigym.ops.alloc import NKIAlloc` (line 23). Replace the filter (lines 41-44):

```python
    compute_records = [rec for rec in op_records if rec.op_cls is not NKIAlloc]
    for rec in compute_records:
        _build_subblock(tree, tree.root, rec, analysis)
```

with:

```python
    for rec in op_records:
        if rec.op_cls.RMW_OPERANDS:
            _build_memset_subblock(tree, tree.root, rec, analysis)
        _build_subblock(tree, tree.root, rec, analysis)
```

- [ ] **Step 4: Add `_build_memset_subblock`**

Add after `_build_subblock`:

```python
def _build_memset_subblock(tree: KernelTree, parent_nid: int, rec: "_OpRecord", analysis: "_AnalysisResult") -> int:
    """Synthesize a memset sibling block zeroing the RMW (accumulator) operand of ``rec``.

    Emitted immediately before the RMW op's own block, mirroring the
    decomposed-canonical form (memset is a sibling, not a nested init).
    The dependency edge falls out by sibling pre-order: memset writes the
    PSUM region, the matmul RMW-reads+writes it (WAW/RAW after memset).
    """
    rmw_slot = next(iter(rec.op_cls.RMW_OPERANDS))
    rmw_axes = rec.op_cls.OPERAND_AXES[rmw_slot]
    memset_rec = _OpRecord(
        op_cls=NKIMemset,
        operand_names={"dst": rec.operand_names[rmw_slot]},
        axis_map={a: rec.axis_map[a] for a in rmw_axes if a in rec.axis_map},
        kwargs={"value": 0.0},
    )
    return _build_subblock(tree, parent_nid, memset_rec, analysis)
```

Add the imports at the top of `canonical_build.py`:

```python
from nkigym.ir.dimension_analysis import _OpRecord
from nkigym.ops.memset import NKIMemset
```

(`_OpRecord` is currently only under `TYPE_CHECKING`; promote it to a real import since we now construct one. Remove it from the `TYPE_CHECKING` block to avoid a duplicate.)

> **Axis-map note:** `NKIMemset.OPERAND_AXES["dst"] = ("P", "F")`, but the matmul's RMW `dst` axes are `("M", "N")`. The synthesized memset must use the matmul's *concrete* dims for those positions. Since `_build_subblock` keys off `rec.axis_map` (abstract → concrete) and `rec.op_cls.OPERAND_AXES`, build the memset record with `op_cls=NKIMemset` but an `axis_map` whose keys are memset's abstract axes (`P`, `F`) mapped to the matmul dst's concrete dims positionally. Replace the `axis_map` line above with the positional alignment:

```python
    memset_concrete = [rec.axis_map[a] for a in rmw_axes if a in rec.axis_map]
    memset_axis_map = {abstract: concrete for abstract, concrete in zip(NKIMemset.OPERAND_AXES["dst"], memset_concrete)}
    memset_rec = _OpRecord(
        op_cls=NKIMemset,
        operand_names={"dst": rec.operand_names[rmw_slot]},
        axis_map=memset_axis_map,
        kwargs={"value": 0.0},
    )
```

- [ ] **Step 5: Run the memset test**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ir/test_dependency.py::test_canonical_synthesizes_memset_for_matmul -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/canonical_build.py test/ir/test_dependency.py
git commit -m "canonical_build: synthesize memset sibling from RMW_OPERANDS

Cutover step 3 of 4. PSUM-zero block built from the matmul's dst region;
no longer sourced from an authored NKIMemset."
```

---

## Task D4: Display + fixtures cutover — full suite green

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree_visualize.py`
- Modify: `examples/matmul_lhsT_rhs.py`
- Modify: remaining test fixtures (`test/environment/_fixtures.py`, `test/ops/test_tile_bounds.py`, `test/ops/test_nkigym_kernel_tag.py`, `test/ir/test_ir_extensions.py`)

This task removes the last `NKIAlloc` references and brings the whole suite green.

- [ ] **Step 1: Drop `NKIAlloc` from `tree_visualize`**

In `tree_visualize.py`: remove `from nkigym.ops.alloc import NKIAlloc` (line 15); remove the `ClassStyle(name="alloc", ...)` entry from `_FLOWCHART_STYLES`. In `_tree_node_decl`, replace the ISANode branch (lines 60-61):

```python
    elif isinstance(data, ISANode):
        decl, class_name = f'{node_id}["{text}"]', "alloc" if data.op_cls is NKIAlloc else "leaf"
```

with:

```python
    elif isinstance(data, ISANode):
        decl, class_name = f'{node_id}["{text}"]', "leaf"
```

- [ ] **Step 2: Convert the example to SSA**

In `examples/matmul_lhsT_rhs.py`: remove the `NKIAlloc` and `NKIMemset` imports. Replace `f_nkigym` with the SSA form and `INPUT_SPECS`:

```python
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """Cached output of ``compile_numpy_to_nkigym(f_numpy, ...)``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out
```

- [ ] **Step 3: Convert remaining fixtures + drop stale alloc tests**

- `test/ops/test_nkigym_kernel_tag.py` and `test/ops/test_tile_bounds.py`: rewrite any `NKIAlloc`-based kernels to SSA; **drop** the alloc-tile-bounds test (`NKIAlloc` is deleted) — note the deletion in the commit message.
- `test/environment/_fixtures.py`: SSA kernel + `(shape, dtype)` specs.
- `test/ir/test_ir_extensions.py`: drop any `NKIAlloc`-record assertions (e.g. tests asserting an alloc `_OpRecord` exists); the trace emits no alloc records now.
- The `bad_return` lineage test (moved out of `test_direct_numpy_call.py` in D1): re-add in SSA form to `test/ops/test_nkigym_kernel_tag.py`:

```python
def test_nkigym_kernel_rejects_non_stored_non_hbm_return():
    """The decorator rejects kernels returning an sbuf-roled array."""
    import numpy as np
    import pytest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad

    @nkigym_kernel
    def bad_return(x):
        return NKILoad()(src=x)

    with pytest.raises(TypeError, match="returned role='sbuf'"):
        bad_return(np.ones((16, 16), dtype=np.float32))
```

- [ ] **Step 4: Run the FULL suite (the green gate)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q`
Expected: PASS — full suite green. If any test still references `NKIAlloc`/`dst=`/`{name: shape}`, fix it now.

- [ ] **Step 5: Run the example end-to-end**

Run: `source ~/venvs/kernel-env/bin/activate && PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py`
Expected: every rollout step prints `[numerics] PASS (atol=0.005, rtol=0.005)`. Confirm the dumped `step_0/kernel.py` allocates `psum_prod` as fp32 and `sbuf_lhs_T`/`sbuf_prod`/`hbm_out` as bf16; the memset appears as a sibling block before the matmul.

- [ ] **Step 6: Commit**

```bash
git add nkigym/ test/ examples/
git commit -m "SSA cutover complete: drop NKIAlloc from viz, convert example + fixtures

Cutover step 4 of 4 — full suite green. Removed the alloc-tile-bounds
test (NKIAlloc deleted)."
```

---

## Task E1: Synthesis prompt — SSA form

**Files:**
- Modify: `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py`
- Test: `test/` (whichever asserts on prompt content, if any — else manual)

- [ ] **Step 1: Inspect the current prompt + any test**

Run: `source ~/venvs/kernel-env/bin/activate && grep -rln "_SYSTEM_PROMPT\|NKIAlloc\|compile_numpy_to_nkigym" test/ | grep -v __pycache__`
Read `numpy_to_nkigym.py`'s `_SYSTEM_PROMPT` and any cheat-sheet/example/translation-procedure that mentions `NKIAlloc`, `dst=`, or `NKIMemset`.

- [ ] **Step 2: Rewrite the prompt for SSA**

In `_SYSTEM_PROMPT`: remove `NKIAlloc`, `dst=`, and authored-`NKIMemset` mentions; replace the worked example with the SSA matmul (the D4 `f_nkigym` body); update the op cheat-sheet rows to drop the output operand from each signature (`NKILoad()(src=...)` returns the SBUF buffer, etc.); state that ops allocate-and-return and that `psum` accumulators are zeroed automatically.

- [ ] **Step 3: Run the synthesis tests (if any)**

Run: `source ~/venvs/kernel-env/bin/activate && python -m pytest test/ -q -k "synth or numpy_to_nkigym"`
Expected: PASS (or "no tests ran" — then verify the prompt manually reads correctly for SSA).

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/synthesis/numpy_to_nkigym.py test/
git commit -m "Synthesis prompt: SSA op form (no NKIAlloc/dst/authored memset)"
```

---

## Task F1: Final verification + spec status update

**Files:**
- Modify: `docs/superpowers/specs/2026-05-31-ssa-ops-backend-refactor-design.md` (add a "Status: SHIPPED" note)

- [ ] **Step 1: Full suite + example, clean run**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/ -q
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
```
Expected: full suite green; every example rollout prints `[numerics] PASS`.

- [ ] **Step 2: Grep for stragglers**

Run: `grep -rn "NKIAlloc\|_infer_param_dtypes\|alloc_names\|_collect_alloc_names" nkigym/src/ test/ examples/ | grep -v __pycache__`
Expected: no matches (every reference removed). Fix any straggler.

- [ ] **Step 3: Add a status note to the spec**

At the top of `docs/superpowers/specs/2026-05-31-ssa-ops-backend-refactor-design.md`, add:

```markdown
## Status: SHIPPED (2026-05-31)

Implemented per `docs/superpowers/plans/2026-05-31-ssa-ops-backend-refactor.md`.
Two corrections applied during implementation: (1) output slot = non-INPUT
(matmul dst is RMW but still the output); primary = reduce_res else dst.
(2) PSUM fp32 is a render-time `Buffer.physical_dtype()` override; logical
dtype propagates uniformly.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-31-ssa-ops-backend-refactor-design.md
git commit -m "Mark SSA ops backend refactor SHIPPED"
```

---

## Self-Review checklist (completed)

- **Spec coverage:** ops backend (D1) ✓; numpy simulator = rewritten `_run`s (D1) ✓; SSA frontend / NKIAlloc deletion (D1) ✓; memset synthesis (D3) ✓; dtype rule (A2 physical_dtype + D2 propagation) ✓; `(shape, dtype)` specs (A1) ✓; `nisa.<NAME>` label (B1) ✓; tree_visualize alloc-bucket drop (D4) ✓; codegen verify (A2 + D4 e2e) ✓; prompt (E1) ✓. `compile_numpy_to_nkigym` named deliverable = the prompt rewrite (E1); its output form is the D4 example. Gadget functions: out of scope per spec.
- **Placeholder scan:** none — every step shows exact code. The one scaffolded helper (`_output_axis_map` first draft) is explicitly flagged with its clean final form.
- **Type consistency:** `input_specs: dict[str, tuple[tuple[int, ...], str]]` used uniformly (A1, ir.py, mdp.py); `OUTPUT_LOCATION` ClassVar consistent across ops; `_synthesize_outputs`/`_output_axis_map`/`_collect_ssa_names`/`ssa_names` names consistent between D2 definitions and call sites.
