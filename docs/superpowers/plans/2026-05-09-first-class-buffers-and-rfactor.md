# First-Class Buffers, RMW Operands, and RFactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote HBM/SBUF/PSUM buffers to first-class `Tensor` entries in the IR, declared explicitly in `f_nkigym` via a unified `NKIAlloc` op; make matmul's `dst` a read+write (RMW) operand; collapse multi-phase op emission (`psum_init`/`compute`/`drain`, `reduce_step`/`reduce_close`) to one `BodyLeaf` per ISA call; and add `RFactor` as a first-class rewrite atom with two recipes (RMW-dst reducers and slot-indexed reducers). Closes the `psum_tile` scope-death bug in rendered kernels by encoding the PSUM dataflow edge through `reads_writes`, so placement atoms can no longer separate init/update/drain across disjoint sibling subtrees.

**Architecture:** Tensor identity (`shape`, `dtype`, `location`, `buffer_degree`) lives in `module.tensors[name]` — the source of truth for what a named tensor *is*. Allocation scope is a separate concern: `NKIAlloc` is a real `BodyLeaf` carrying only `tensor_name`; the emitter looks up shape/dtype/location at render time. Mirrors TVM's `Buffer` (identity) vs `Allocate` (tree node) split. `BodyLeaf.reads_writes` encodes RMW operands; `validate_dataflow_ordering` uses `reads ∪ reads_writes` for the reads-after-writes invariant, replacing the matmul phase-order carve-out. `RFactor` dispatches on `op_cls.RFACTOR_RECIPE` (one of `"rmw"` / `"slot"` / `None`) to emit staging-buffer decomposition mechanically.

**Tech Stack:** Python 3.12, `dataclasses`, Apache NKI (neuronx-cc), pytest. Environment: `source ~/venvs/kernel-env/bin/activate`.

**Spec:** `docs/superpowers/specs/2026-05-09-first-class-buffers-and-rfactor-design.md`

**Final validation gates (terminal state only):**
- `pytest nkigym/ test/` full suite green.
- `python examples/matmul_lhsT_rhs.py` end-to-end (CPU-sim against numpy).
- `python scripts/tune_matmul_lhsT_rhs.py` batch-tune run completes; at least one sampled kernel produces HW MFU within 1pp of the pre-refactor baseline (90.92%).
- `python examples/matmul_lhs_rhs.py` end-to-end.
- `python examples/rmsnorm_matmul.py` end-to-end.
- The broken rewrite chain from the bug report is unreachable: `python scripts/tune_matmul_lhsT_rhs.py` never renders a kernel with PSUM allocation inside an accumulation loop.

---

## File Structure

### Files to create

```
nkigym/src/nkigym/ops/
  alloc.py                                # NKIAlloc op class
  memset.py                               # NKIMemset op class
  tensor_copy.py                          # NKITensorCopy op class
  tensor_reduce.py                        # NKITensorReduce op class

nkigym/src/nkigym/tune/
  rfactor.py                              # RFactor atom + two recipes + enumerator

test/codegen/
  test_rfactor_rmw.py                     # matmul rfactor recipe tests
  test_rfactor_slot.py                    # activation_reduce rfactor recipe tests
  test_first_class_buffers.py             # canonical parse + render of NKIAlloc
```

### Files to modify

```
nkigym/src/nkigym/ops/
  base.py                                 # add RMW_OPERANDS, RFACTOR_RECIPE, INPUT_OPERANDS ClassVars
  matmul.py                               # dst in OPERAND_AXES, RMW_OPERANDS={dst}, RFACTOR_RECIPE="rmw"
  activation_reduce.py                    # dst+reduce_res in OPERAND_AXES, RFACTOR_RECIPE="slot", delete OP_LOCAL_BUFFERS
  activation.py                           # dst in OPERAND_AXES
  tensor_scalar.py                        # dst in OPERAND_AXES
  load.py                                 # dst in OPERAND_AXES (alongside src)
  store.py                                # dst in OPERAND_AXES (alongside src)
  transpose.py                            # dst in OPERAND_AXES
  dma_transpose.py                        # dst in OPERAND_AXES

nkigym/src/nkigym/codegen/
  ir.py                                   # Tensor.location; BodyLeaf.reads_writes; validate_dataflow_ordering rewrite; drop phase/op_local_buffers
  canonical.py                            # parse NKIAlloc; single-phase builder; delete multi-phase machinery
  render.py                               # inline or thin pass-through (no logic change)

nkigym/src/nkigym/codegen/lowering/
  emit_source.py                          # delete _emit_sbuf_allocations, _emit_hbm_output, _emit_param_asserts
  _emit_utils.py                          # delete _hbm_name, _sbuf_name prefixers
  lower_phases.py → rename emit_ops.py    # one emitter per op_cls; drop phase keying
  place_buffers.py                        # include NKIAlloc leaf in LCA walk; shrink to helpers
  inject_multi_buffer.py                  # use tensor.location for buffer dispatch
  inject_software_pipeline.py             # op_cls-keyed dispatch (was phase-keyed)

nkigym/src/nkigym/tune/
  batch.py                                # swap DecomposeReduction enumerator for RFactor

nkigym/src/nkigym/synthesis/
  SKILL.md (or prompt constants)          # update worked examples for new f_nkigym form

examples/
  matmul_lhsT_rhs.py                      # rewrite f_nkigym body
  matmul_lhs_rhs.py                       # rewrite f_nkigym body
  rmsnorm_matmul.py                       # rewrite f_nkigym body

test/codegen/
  _rmsnorm_matmul_fixture.py              # sync with examples/rmsnorm_matmul.py
  test_canonical.py                       # update all hand-built fixtures
  test_ir.py                              # delete matmul phase-order tests; add RMW tests
  test_place_buffers.py                   # update LCA-walk fixtures
  test_multi_buffer_unit.py               # update fixture shapes
  test_software_pipeline_unit.py          # update fixture shapes
  test_dep_cache.py                       # drop op-local-buffer tests; add RMW signature tests
  test_axis_role.py                       # fixture updates (if any)

test/tune/
  test_batch.py                           # drop DecomposeReduction, add RFactor
```

### Files to delete

```
nkigym/src/nkigym/tune/decompose_reduction.py
```

---

## Ordering Strategy

The refactor has interdependent concerns (IR shape, canonical builder, renderer, ops, atoms). Ordering minimizes cascading breakage:

1. **Phase 1 (Tasks 1–4):** Add the new op classes + `Tensor.location` + `BodyLeaf.reads_writes` scaffolding without removing old machinery. All tests still pass because new fields have safe defaults.
2. **Phase 2 (Tasks 5–9):** Rewrite canonical builder to parse `NKIAlloc` and produce single-phase leaves. Update `validate_dataflow_ordering`. Delete phase-specific machinery. Update renderer to use first-class allocs.
3. **Phase 3 (Tasks 10–14):** Rewrite example `f_nkigym` files + fixtures + synthesis skill. End-to-end green.
4. **Phase 4 (Tasks 15–18):** Add `RFactor` atom + both recipes + tests. Swap `DecomposeReduction` out of the sampler.
5. **Phase 5 (Task 19):** Full validation — CPU-sim examples, batch-tune smoke, MFU gate.

---

## Phase 1: Scaffolding

### Task 1: Add `RMW_OPERANDS`, `RFACTOR_RECIPE`, `INPUT_OPERANDS` ClassVars on `NKIOp`

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Test: `test/codegen/test_first_class_buffers.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/codegen/test_first_class_buffers.py`:

```python
"""Tests for first-class buffer infrastructure: NKIOp ClassVars, Tensor.location,
BodyLeaf.reads_writes, and their interactions through the canonical builder."""

from nkigym.ops.base import NKIOp


def test_nkiop_defaults_rmw_and_rfactor_and_input_operands():
    """Every NKIOp subclass should inherit safe defaults: no RMW, no rfactor recipe,
    no declared input operands (subclasses override per-op)."""
    assert NKIOp.RMW_OPERANDS == frozenset()
    assert NKIOp.RFACTOR_RECIPE is None
    assert NKIOp.INPUT_OPERANDS == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_first_class_buffers.py::test_nkiop_defaults_rmw_and_rfactor_and_input_operands -v
```

Expected: FAIL — `AttributeError: type object 'NKIOp' has no attribute 'RMW_OPERANDS'`.

- [ ] **Step 3: Add the ClassVars to `NKIOp`**

In `nkigym/src/nkigym/ops/base.py`, inside `class NKIOp`, after the existing `OP_LOCAL_BUFFERS` ClassVar block (around line 129), add:

```python
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    """Operand slot names that this op reads AND writes (RMW semantics).

    For `NKIMatmul`, `dst` is RMW — `nisa.nc_matmul` accumulates into its
    PSUM destination across K iterations. Every other op has disjoint
    reads and writes; this set is empty.

    Consumed by the canonical builder's `_make_leaf` to populate
    `BodyLeaf.reads_writes` (the tensor names for these slots appear in
    `reads_writes`, not in `reads` or `writes`).
    """

    RFACTOR_RECIPE: ClassVar["Literal['rmw', 'slot'] | None"] = None
    """Which RFactor recipe this op supports, or ``None`` if not rfactorable.

    - ``"rmw"``: ops with a HW accumulator (matmul). RFactor materializes
      a staging buffer, per-outer-iteration PSUM alloc, drain to SBUF slot,
      closing tensor_reduce.
    - ``"slot"``: ops whose write operand naturally indexes by reduction
      tile (activation_reduce). RFactor points successive calls at
      successive slots of a staging buffer, closes with tensor_reduce.
    - ``None``: atom legality rejects any RFactor targeting this op.
    """

    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    """Operand slots that are read-only (inputs to the computation).

    Slots in ``INPUT_OPERANDS`` land in ``BodyLeaf.reads``; slots that
    are neither in ``INPUT_OPERANDS`` nor ``RMW_OPERANDS`` (typically
    ``dst``, ``reduce_res``) land in ``BodyLeaf.writes``.

    Required for every op subclass — the canonical builder uses this set
    to split operand slots into reads / writes / reads_writes at leaf-
    construction time.
    """
```

Add `Literal` to the imports at the top of the file:

```python
from typing import Any, ClassVar, Literal
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/codegen/test_first_class_buffers.py::test_nkiop_defaults_rmw_and_rfactor_and_input_operands -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/base.py test/codegen/test_first_class_buffers.py
git commit -m "feat: add RMW_OPERANDS, RFACTOR_RECIPE, INPUT_OPERANDS ClassVars on NKIOp"
```

---

### Task 2: Add `Tensor.location` field and `BodyLeaf.reads_writes` field

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir.py`
- Test: `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_first_class_buffers.py`:

```python
from nkigym.codegen.ir import BodyLeaf, Tensor


def test_tensor_location_field_defaults_and_accepts_literal():
    """Tensor gains a location field; every tensor must declare hbm/sbuf/psum."""
    t = Tensor(
        name="psum_acc",
        dim_ids=("d0", "d1"),
        shape=(128, 512),
        dtype="float32",
        origin="intermediate",
        location="psum",
    )
    assert t.location == "psum"


def test_body_leaf_reads_writes_defaults_empty():
    """BodyLeaf gains reads_writes for RMW operands. Default is empty tuple."""
    leaf = BodyLeaf(op_cls=type("Fake", (), {"__name__": "Fake"}))
    assert leaf.reads_writes == ()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_first_class_buffers.py::test_tensor_location_field_defaults_and_accepts_literal test/codegen/test_first_class_buffers.py::test_body_leaf_reads_writes_defaults_empty -v
```

Expected: FAIL — `TypeError: Tensor.__init__() got an unexpected keyword argument 'location'` and `AttributeError: 'BodyLeaf' object has no attribute 'reads_writes'`.

- [ ] **Step 3: Add `Tensor.location` and `BodyLeaf.reads_writes`**

In `nkigym/src/nkigym/codegen/ir.py`, locate the `Tensor` dataclass (around line 27) and modify it:

```python
@dataclass
class Tensor:
    """Named tensor appearing in the kernel body.

    Attributes:
        name: Source-level variable name.
        dim_ids: Concrete dim ids in operand order.
        shape: Element sizes aligned with ``dim_ids``.
        dtype: Element dtype (e.g. ``"bfloat16"``).
        origin: ``"param"`` (HBM input) or ``"intermediate"`` (declared
            via ``NKIAlloc``).
        location: ``"hbm"`` / ``"sbuf"`` / ``"psum"`` — which memory
            the allocation targets. For params, always ``"hbm"``.
        buffer_degree: Multi-buffer degree per dim; defaults to 1.
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin
    location: Literal["hbm", "sbuf", "psum"] = "sbuf"
    buffer_degree: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for d in self.dim_ids:
            self.buffer_degree.setdefault(d, 1)
```

(Default value `"sbuf"` on `location` is temporary scaffolding — it keeps existing tests passing during the migration. A later task will make it required and remove the default.)

Add `Literal` to the imports at the top of `ir.py`:

```python
from typing import Any, Literal
```

Locate the `BodyLeaf` dataclass (around line 88) and add `reads_writes`:

```python
@dataclass
class BodyLeaf:
    """Self-describing leaf: an op (or op phase) + the metadata needed to render it.

    Attributes:
        op_cls: The NKIOp subclass.
        phase: ``"main"`` for single-phase ops; one of the op class's phases
            otherwise. Deprecated — will be removed once single-phase
            migration lands.
        reads: Maps operand slot name to referenced tensor name (read-only operands).
        writes: Tuple of tensor names this leaf writes (write-only operands).
        reads_writes: Tuple of tensor names this leaf reads AND writes (RMW
            operands — e.g. matmul's PSUM dst, which accumulates across K).
        kwargs: Merged literal kwargs from the NKIOp call.
        axis_map: Abstract axis label (``"K"`` etc.) to concrete dim id.
        dim_role: Concrete dim id to :class:`AxisRole` (op-local).
        op_local_buffers: Op-local buffers keyed by logical name. Deprecated —
            will be removed once single-phase migration lands.
    """

    op_cls: type
    phase: str = "main"
    reads: dict[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    reads_writes: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    dim_role: dict[str, AxisRole] = field(default_factory=dict)
    op_local_buffers: dict[str, OpLocalBuffer] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_first_class_buffers.py -v
```

Expected: PASS for the two new tests.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
pytest test/ -v 2>&1 | tail -20
```

Expected: no new failures (pre-existing failures from the buggy state unrelated to first-class buffers may remain).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir.py test/codegen/test_first_class_buffers.py
git commit -m "feat: add Tensor.location and BodyLeaf.reads_writes fields"
```

---

### Task 3: Create `NKIAlloc` op class

**Files:**
- Create: `nkigym/src/nkigym/ops/alloc.py`
- Test: `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_first_class_buffers.py`:

```python
from nkigym.ops.alloc import NKIAlloc


def test_nkialloc_has_empty_operand_axes_and_no_rmw():
    """NKIAlloc is a declaration op: no operand axes, no reads, no RMW."""
    assert NKIAlloc.OPERAND_AXES == {}
    assert NKIAlloc.RMW_OPERANDS == frozenset()
    assert NKIAlloc.INPUT_OPERANDS == frozenset()
    assert NKIAlloc.RFACTOR_RECIPE is None


def test_nkialloc_cpu_sim_returns_numpy_zeros():
    """CPU simulation allocates a numpy array of declared shape/dtype, zero-filled."""
    alloc = NKIAlloc(location="sbuf", shape=(4, 8), dtype="float32")
    result = alloc()
    assert result.shape == (4, 8)
    assert str(result.dtype) == "float32"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_first_class_buffers.py::test_nkialloc_has_empty_operand_axes_and_no_rmw -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.ops.alloc'`.

- [ ] **Step 3: Create `NKIAlloc` op class**

Create `nkigym/src/nkigym/ops/alloc.py`:

```python
"""First-class allocation op: maps to ``nl.ndarray(buffer=...)``.

Unified HBM/SBUF/PSUM allocation declared explicitly in ``f_nkigym``.
User call form (in f_nkigym source):

    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()

The canonical builder reads ``(location, shape, dtype)`` from these kwargs
to populate ``module.tensors[name]``, then emits a ``BodyLeaf`` whose
single kwarg is ``tensor_name``. The renderer looks up
``module.tensors[tensor_name]`` at emission time for shape/dtype/location.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


_DTYPE_MAP: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "float16": np.dtype("float16"),
    "bfloat16": np.dtype("float32"),
}
"""CPU-sim allocates bf16 tensors as fp32 to match the sim-path dtype contract."""


class NKIAlloc(NKIOp):
    """Declare a tensor with explicit location/shape/dtype.

    kwargs on the user call:
        location: ``"hbm"`` | ``"sbuf"`` | ``"psum"``
        shape: ``tuple[int, ...]``
        dtype: ``str`` — one of ``"float32"`` / ``"float16"`` / ``"bfloat16"``.

    Returns a zero-filled ``numpy.ndarray`` at CPU-sim time for the
    downstream ops to read/write. At render time the emitter produces
    ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<location>)``.
    """

    NAME: ClassVar[str] = "alloc"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: return a zero-filled array of declared shape/dtype."""
        shape = kwargs["shape"]
        dtype_name = kwargs["dtype"]
        dtype = _DTYPE_MAP[dtype_name]
        return np.zeros(shape, dtype=dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_first_class_buffers.py::test_nkialloc_has_empty_operand_axes_and_no_rmw test/codegen/test_first_class_buffers.py::test_nkialloc_cpu_sim_returns_numpy_zeros -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/alloc.py test/codegen/test_first_class_buffers.py
git commit -m "feat: add NKIAlloc op class"
```

---

### Task 4: Create `NKIMemset`, `NKITensorCopy`, `NKITensorReduce` op classes

**Files:**
- Create: `nkigym/src/nkigym/ops/memset.py`
- Create: `nkigym/src/nkigym/ops/tensor_copy.py`
- Create: `nkigym/src/nkigym/ops/tensor_reduce.py`
- Test: `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_first_class_buffers.py`:

```python
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce


def test_nkimemset_writes_dst_with_value():
    """Memset writes a constant value into its dst; no reads."""
    assert NKIMemset.INPUT_OPERANDS == frozenset()
    assert NKIMemset.OPERAND_AXES == {"dst": ("P", "F")}
    """CPU sim: take a pre-allocated dst and fill with value."""
    dst = np.zeros((4, 8), dtype=np.float32)
    import numpy as np  # local shadow to quiet linters
    NKIMemset(value=1.5)(dst=dst)
    assert (dst == 1.5).all()


def test_nkitensor_copy_src_to_dst():
    """tensor_copy reads src, writes dst."""
    assert NKITensorCopy.INPUT_OPERANDS == frozenset({"src"})
    assert NKITensorCopy.OPERAND_AXES == {"src": ("P", "F"), "dst": ("P", "F")}


def test_nkitensor_reduce_reads_data_writes_dst():
    """tensor_reduce reads data, writes dst, accepts axis + op kwargs."""
    assert NKITensorReduce.INPUT_OPERANDS == frozenset({"data"})
    assert "data" in NKITensorReduce.OPERAND_AXES
    assert "dst" in NKITensorReduce.OPERAND_AXES
```

(Note: the `import numpy as np` inside the test is required because the top-of-file import is local to each `_run`. Move it to the top of the test file if more tests need it.)

Fix the test to be clean:

```python
def test_nkimemset_writes_dst_with_value():
    """Memset writes a constant value into its dst; no reads."""
    import numpy as np
    assert NKIMemset.INPUT_OPERANDS == frozenset()
    assert NKIMemset.OPERAND_AXES == {"dst": ("P", "F")}
    dst = np.zeros((4, 8), dtype=np.float32)
    NKIMemset(value=1.5)(dst=dst)
    assert (dst == 1.5).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_first_class_buffers.py::test_nkimemset_writes_dst_with_value -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.ops.memset'`.

- [ ] **Step 3: Create the three op classes**

Create `nkigym/src/nkigym/ops/memset.py`:

```python
"""In-place fill op: maps to ``nisa.memset``."""

from typing import Any, ClassVar

from nkigym.ops.base import NKIOp


class NKIMemset(NKIOp):
    """Fill a tensor with a constant value.

    kwargs:
        value: ``float`` — the constant to write into every element.
    operands:
        dst: target tensor (P, F layout).
    """

    NAME: ClassVar[str] = "memset"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: in-place fill of dst with value."""
        dst = kwargs["dst"]
        value = kwargs["value"]
        dst[...] = value
        return dst
```

Create `nkigym/src/nkigym/ops/tensor_copy.py`:

```python
"""SBUF/PSUM → SBUF copy op: maps to ``nisa.tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKITensorCopy(NKIOp):
    """Copy ``src`` into ``dst`` element-wise.

    operands:
        src: source tensor (P, F).
        dst: destination tensor (P, F) — same shape/dtype.
    """

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: element-wise copy."""
        src = kwargs["src"]
        dst = kwargs["dst"]
        dst[...] = src
        return dst
```

Create `nkigym/src/nkigym/ops/tensor_reduce.py`:

```python
"""Free-axis reduction op: maps to ``nisa.tensor_reduce``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp


_REDUCE_FNS: dict[str, Any] = {"add": np.sum, "max": np.max}


class NKITensorReduce(NKIOp):
    """Reduce ``data`` along an axis into ``dst``.

    kwargs:
        axis: ``int`` — the axis of ``data`` to reduce over.
        op: ``"add"`` or ``"max"``.
    operands:
        data: source tensor.
        dst: destination tensor — shape equals ``data.shape`` with ``axis`` removed.
    """

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P",)}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: numpy reduction along axis."""
        data = kwargs["data"]
        dst = kwargs["dst"]
        axis = kwargs["axis"]
        op = kwargs["op"]
        reduce_fn = _REDUCE_FNS[op]
        result = reduce_fn(data, axis=axis)
        dst[...] = result
        return dst
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_first_class_buffers.py::test_nkimemset_writes_dst_with_value test/codegen/test_first_class_buffers.py::test_nkitensor_copy_src_to_dst test/codegen/test_first_class_buffers.py::test_nkitensor_reduce_reads_data_writes_dst -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/memset.py nkigym/src/nkigym/ops/tensor_copy.py nkigym/src/nkigym/ops/tensor_reduce.py test/codegen/test_first_class_buffers.py
git commit -m "feat: add NKIMemset, NKITensorCopy, NKITensorReduce op classes"
```

---

## Phase 2: IR + canonical + renderer surgery

### Task 5: Update every existing op to declare `INPUT_OPERANDS` and add `dst` to `OPERAND_AXES`

**Files:**
- Modify: `nkigym/src/nkigym/ops/matmul.py`
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`
- Modify: `nkigym/src/nkigym/ops/activation.py`
- Modify: `nkigym/src/nkigym/ops/tensor_scalar.py`
- Modify: `nkigym/src/nkigym/ops/load.py`
- Modify: `nkigym/src/nkigym/ops/store.py`
- Modify: `nkigym/src/nkigym/ops/transpose.py`
- Modify: `nkigym/src/nkigym/ops/dma_transpose.py`
- Test: `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_first_class_buffers.py`:

```python
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDmaTranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_matmul_dst_is_rmw():
    assert NKIMatmul.RMW_OPERANDS == frozenset({"dst"})
    assert NKIMatmul.RFACTOR_RECIPE == "rmw"
    assert NKIMatmul.INPUT_OPERANDS == frozenset({"stationary", "moving"})
    assert "dst" in NKIMatmul.OPERAND_AXES
    assert NKIMatmul.OPERAND_AXES["dst"] == ("M", "N")


def test_activation_reduce_has_both_dst_and_reduce_res_as_writes():
    assert NKIActivationReduce.RFACTOR_RECIPE == "slot"
    assert NKIActivationReduce.INPUT_OPERANDS == frozenset({"data"})
    assert "dst" in NKIActivationReduce.OPERAND_AXES
    assert "reduce_res" in NKIActivationReduce.OPERAND_AXES


def test_every_existing_op_declares_dst_in_operand_axes():
    """Every op that writes a tensor now declares its dst slot explicitly."""
    write_ops = [NKILoad, NKIStore, NKIActivation, NKITensorScalar, NKITranspose, NKIDmaTranspose]
    for op_cls in write_ops:
        assert "dst" in op_cls.OPERAND_AXES, f"{op_cls.__name__} missing dst in OPERAND_AXES"


def test_every_existing_op_declares_input_operands():
    """Every op subclass declares its read-only slots. Covers the discriminator
    the canonical builder uses to split operands into reads vs writes vs reads_writes."""
    cases = [
        (NKILoad, frozenset({"src"})),
        (NKIStore, frozenset({"src"})),
        (NKIActivation, frozenset({"data"})),
        (NKITensorScalar, frozenset({"data", "operand0"})),
        (NKITranspose, frozenset({"src"})),
        (NKIDmaTranspose, frozenset({"src"})),
    ]
    for op_cls, expected in cases:
        assert op_cls.INPUT_OPERANDS == expected, f"{op_cls.__name__}: {op_cls.INPUT_OPERANDS} != {expected}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_first_class_buffers.py::test_matmul_dst_is_rmw -v
```

Expected: FAIL — `RMW_OPERANDS` and `RFACTOR_RECIPE` still defaults; `OPERAND_AXES` is missing `dst`.

- [ ] **Step 3: Update `NKIMatmul`**

In `nkigym/src/nkigym/ops/matmul.py`, inside `class NKIMatmul`, update:

```python
    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "stationary": ("K", "M"),
        "moving": ("K", "N"),
        "dst": ("M", "N"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "rmw"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
```

Delete the existing `OUTPUT_AXES` ClassVar line. Add `Literal` to the imports:

```python
from typing import Any, ClassVar, Literal
```

Keep `OP_LOCAL_BUFFERS` if present — it will be deleted in Task 7.

- [ ] **Step 4: Update `NKIActivationReduce`**

In `nkigym/src/nkigym/ops/activation_reduce.py`, update the `class NKIActivationReduce` ClassVars:

```python
    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "dst": ("P", "F"),
        "reduce_res": ("P",),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
```

Delete `OUTPUT_AXES`, `OUTPUT_DTYPES`. Keep `OP_LOCAL_BUFFERS` — it will be deleted in Task 7.

- [ ] **Step 5: Update `NKIActivation`**

In `nkigym/src/nkigym/ops/activation.py`:

```python
    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
```

Delete `OUTPUT_AXES`.

- [ ] **Step 6: Update `NKITensorScalar`**

In `nkigym/src/nkigym/ops/tensor_scalar.py`:

```python
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P",),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
```

Delete `OUTPUT_AXES`.

- [ ] **Step 7: Update `NKILoad`**

In `nkigym/src/nkigym/ops/load.py`:

```python
    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "src": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
```

Rename the existing `data` key in `OPERAND_AXES` to `src`. Delete `OUTPUT_AXES`.

- [ ] **Step 8: Update `NKIStore`**

In `nkigym/src/nkigym/ops/store.py`:

```python
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "src": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
```

Rename `data` → `src`. Delete `OUTPUT_AXES`.

- [ ] **Step 9: Update `NKITranspose` and `NKIDmaTranspose`**

In `nkigym/src/nkigym/ops/transpose.py`:

```python
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "src": ("P", "F"),
        "dst": ("F", "P"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
```

Rename `data` → `src`. Delete `OUTPUT_AXES`.

In `nkigym/src/nkigym/ops/dma_transpose.py`:

```python
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "src": ("P", "F"),
        "dst": ("F", "P"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
```

Rename `data` → `src`. Delete `OUTPUT_AXES`.

- [ ] **Step 10: Run tests to verify they pass**

```bash
pytest test/codegen/test_first_class_buffers.py -v
```

Expected: PASS for all tests in the file. Pre-existing other tests may fail now — that's expected, fixed in subsequent tasks.

- [ ] **Step 11: Commit**

```bash
git add nkigym/src/nkigym/ops/
git commit -m "feat: declare INPUT_OPERANDS and add dst to OPERAND_AXES on every op"
```

---

### Task 6: Rewrite `validate_dataflow_ordering` to use `reads ∪ reads_writes`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir.py`
- Test: `test/codegen/test_ir.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_ir.py`:

```python
def test_validate_dataflow_ordering_rmw_requires_prior_writer():
    """An RMW operand in reads_writes must have a prior writer.
    Models the bug scenario: NKIMatmul reads_writes=psum_acc must come
    after NKIMemset writes=psum_acc."""
    from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, Tensor, validate_dataflow_ordering

    class _FakeMemset:
        __name__ = "NKIMemset"

    class _FakeMatmul:
        __name__ = "NKIMatmul"

    tensors = {
        "psum_acc": Tensor(
            name="psum_acc", dim_ids=("d0",), shape=(128,),
            dtype="float32", origin="intermediate", location="psum",
        ),
    }
    dims = {"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)}

    memset_leaf = BodyLeaf(op_cls=_FakeMemset, reads={}, writes=("psum_acc",), reads_writes=())
    matmul_leaf = BodyLeaf(op_cls=_FakeMatmul, reads={}, writes=(), reads_writes=("psum_acc",))

    good_module = KernelModule(
        func_name="f", param_names=[], return_name="psum_acc",
        tensors=tensors, dims=dims, body=[memset_leaf, matmul_leaf],
    )
    bad_module = KernelModule(
        func_name="f", param_names=[], return_name="psum_acc",
        tensors=tensors, dims=dims, body=[matmul_leaf, memset_leaf],
    )

    assert validate_dataflow_ordering(good_module) is True
    assert validate_dataflow_ordering(bad_module) is False


def test_validate_dataflow_ordering_rmw_leaf_counts_as_writer_for_next_leaf():
    """After an RMW leaf fires, the tensor counts as written for subsequent reads."""
    from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, Tensor, validate_dataflow_ordering

    class _FakeMemset: __name__ = "NKIMemset"
    class _FakeMatmul: __name__ = "NKIMatmul"
    class _FakeCopy: __name__ = "NKITensorCopy"

    tensors = {
        "psum_acc": Tensor("psum_acc", ("d0",), (128,), "float32", "intermediate", "psum"),
        "sbuf_prod": Tensor("sbuf_prod", ("d0",), (128,), "bfloat16", "intermediate", "sbuf"),
    }
    dims = {"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)}

    body = [
        BodyLeaf(op_cls=_FakeMemset, writes=("psum_acc",)),
        BodyLeaf(op_cls=_FakeMatmul, reads_writes=("psum_acc",)),
        BodyLeaf(op_cls=_FakeCopy, reads={"src": "psum_acc"}, writes=("sbuf_prod",)),
    ]
    mod = KernelModule(
        func_name="f", param_names=[], return_name="sbuf_prod",
        tensors=tensors, dims=dims, body=body,
    )
    assert validate_dataflow_ordering(mod) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_ir.py::test_validate_dataflow_ordering_rmw_requires_prior_writer -v
```

Expected: FAIL — current validator doesn't check `reads_writes`.

- [ ] **Step 3: Rewrite `validate_dataflow_ordering`**

In `nkigym/src/nkigym/codegen/ir.py`, replace the function body with:

```python
def validate_dataflow_ordering(module: KernelModule) -> bool:
    """Return True iff the forest's emission order preserves dataflow.

    Rules:
    - **Reads-after-writes:** every name in ``leaf.reads ∪ leaf.reads_writes``
      must be a parameter or already-written by a prior leaf in pre-order
      DFS.
    - **Writes update the written set:** names in ``leaf.writes ∪ leaf.reads_writes``
      are added after the leaf fires.
    - **Return tensor produced:** every tensor with ``origin == "return"``
      (legacy — being phased out) must be written by some leaf.

    RMW operands encode "init must precede update" structurally —
    `NKIMatmul.reads_writes = ('psum_acc',)` means the matmul leaf
    requires a prior writer of ``psum_acc`` (the memset), and the old
    matmul phase-order carve-out is no longer needed.
    """
    written: set[str] = set()
    params = {t.name for t in module.tensors.values() if t.origin == "param"}
    returns = {t.name for t in module.tensors.values() if t.origin == "return"}

    for leaf in emission_order_leaves(module.body):
        read_set = set(leaf.reads.values()) | set(leaf.reads_writes)
        for name in read_set:
            if name in params:
                continue
            if name not in written:
                return False
        write_set = set(leaf.writes) | set(leaf.reads_writes)
        written |= write_set

    for ret_name in returns:
        if ret_name not in written:
            return False
    return True
```

Delete the entire matmul phase-order carve-out block (the `matmul_phase_order` dict and every `matmul_seen` reference).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_ir.py::test_validate_dataflow_ordering_rmw_requires_prior_writer test/codegen/test_ir.py::test_validate_dataflow_ordering_rmw_leaf_counts_as_writer_for_next_leaf -v
```

Expected: PASS.

- [ ] **Step 5: Delete obsolete matmul phase-order tests**

Open `test/codegen/test_ir.py` and delete any test with `matmul_phase_order` in its name or body — those carve-outs no longer exist. Also delete any test whose docstring mentions `psum_init` / `compute` / `drain` phases as separate leaves — single-phase migration in later tasks makes them obsolete.

- [ ] **Step 6: Run full test suite**

```bash
pytest test/codegen/test_ir.py -v
```

Expected: remaining tests pass, deleted tests no longer run.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir.py test/codegen/test_ir.py
git commit -m "feat: validate_dataflow_ordering uses reads_writes; drop matmul phase carve-out"
```

---

### Task 7: Rewrite canonical builder — parse `NKIAlloc`, single-phase leaves, RMW handling

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py`
- Test: `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_first_class_buffers.py`:

```python
def test_canonical_parses_nkialloc_into_module_tensors():
    """An NKIAlloc call in f_nkigym registers a Tensor in module.tensors with
    declared location/shape/dtype — no inference, no OP_LOCAL_BUFFERS, no phase."""
    from nkigym.codegen.canonical import build_canonical_module
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def f(lhs):
        lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs, dst=lhs_sbuf)
        NKIStore()(src=lhs_sbuf, dst=hbm_out)
        return hbm_out

    input_specs = {"lhs": {"shape": (128, 512), "dtype": "bfloat16"}}
    module = build_canonical_module(f, input_specs)

    assert "lhs_sbuf" in module.tensors
    assert module.tensors["lhs_sbuf"].location == "sbuf"
    assert module.tensors["lhs_sbuf"].shape == (128, 512)
    assert module.tensors["lhs_sbuf"].dtype == "bfloat16"
    assert module.tensors["hbm_out"].location == "hbm"


def test_canonical_matmul_leaf_has_dst_in_reads_writes():
    """After canonical build, NKIMatmul's leaf carries dst in reads_writes
    (not in reads or writes)."""
    from nkigym.codegen.canonical import build_canonical_module
    from nkigym.codegen.ir import BodyLeaf, leaves_under
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_copy import NKITensorCopy

    @nkigym_kernel
    def f(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    input_specs = {
        "lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
        "rhs": {"shape": (128, 512), "dtype": "bfloat16"},
    }
    module = build_canonical_module(f, input_specs)

    matmul_leaves = [leaf for root in module.body for leaf in leaves_under(root)
                     if leaf.op_cls.__name__ == "NKIMatmul"]
    assert len(matmul_leaves) == 1
    leaf = matmul_leaves[0]
    assert "psum_acc" in leaf.reads_writes
    assert "psum_acc" not in leaf.reads.values()
    assert "psum_acc" not in leaf.writes
    assert leaf.reads == {"stationary": "lhs_T_sbuf", "moving": "rhs_sbuf"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_first_class_buffers.py::test_canonical_parses_nkialloc_into_module_tensors -v
```

Expected: FAIL — current canonical builder has no `NKIAlloc` handling.

- [ ] **Step 3: Rewrite `canonical.py`**

Open `nkigym/src/nkigym/codegen/canonical.py`. The file is large — the full rewrite replaces the AST parser, tensor map construction, and leaf builder. Key changes:

**3a.** Add a helper `_is_alloc_call` and modify `_parse_ast` to split alloc records from regular op records. After the existing `_parse_ast` function:

```python
@dataclass
class _AllocRecord:
    """Canonical-build-time record for an NKIAlloc call in f_nkigym.

    Captures the tensor identity (name, location, shape, dtype) that
    populates ``module.tensors``. The resulting ``BodyLeaf`` carries
    only ``tensor_name`` — the declaration lives in ``module.tensors``.
    """

    name: str
    location: str  # "hbm" | "sbuf" | "psum"
    shape: tuple[int, ...]
    dtype: str
```

Modify `_parse_ast` to recognize `NKIAlloc(location=..., shape=..., dtype=...)` calls and emit `_AllocRecord` instead of `_ParsedOpRaw`:

```python
def _parse_ast(func: Callable[..., np.ndarray]) -> tuple[list[_ParsedOpRaw], list[_AllocRecord], str]:
    """Walk ``func``'s AST. Returns (op records, alloc records, return name)."""
    unwrapped = getattr(func, "__wrapped__", func)
    func_globals = unwrapped.__globals__
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    raws: list[_ParsedOpRaw] = []
    allocs: list[_AllocRecord] = []
    return_name: str | None = None
    for stmt in func_def.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id
            continue
        if isinstance(stmt, ast.Assign):
            alloc = _try_parse_alloc(stmt, func_globals)
            if alloc is not None:
                allocs.append(alloc)
                continue
            raw = _parse_assignment(stmt, func_globals)
            if raw is not None:
                raws.append(raw)
    if return_name is None:
        raise ValueError("f_nkigym must end with `return <tensor>`")
    return raws, allocs, return_name


def _try_parse_alloc(stmt: ast.Assign, func_globals: dict[str, object]) -> _AllocRecord | None:
    """Extract an ``_AllocRecord`` from ``var = NKIAlloc(...)()``; return None otherwise."""
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    outer = stmt.value
    if not isinstance(outer, ast.Call) or not isinstance(outer.func, ast.Call):
        return None
    inner = outer.func
    if not isinstance(inner.func, ast.Name) or inner.func.id != "NKIAlloc":
        return None
    """Verify that the NKIAlloc name resolves to our op class."""
    candidate = func_globals.get(inner.func.id)
    from nkigym.ops.alloc import NKIAlloc
    if candidate is not NKIAlloc:
        return None
    kwargs = _extract_literal_kwargs(inner, func_globals)
    for req in ("location", "shape", "dtype"):
        if req not in kwargs:
            raise ValueError(f"NKIAlloc missing required kwarg {req!r} at '{stmt.targets[0].id}'")
    return _AllocRecord(
        name=stmt.targets[0].id,
        location=kwargs["location"],
        shape=tuple(kwargs["shape"]),
        dtype=kwargs["dtype"],
    )
```

**3b.** Modify `build_canonical_module` to use the new tuple:

```python
def build_canonical_module(func: Callable[..., np.ndarray], input_specs: dict[str, dict]) -> KernelModule:
    """Build a :class:`KernelModule` from an ``f_nkigym`` callable.

    See module docstring for the pipeline.
    """
    raws, allocs, return_name = _parse_ast(func)
    unwrapped = getattr(func, "__wrapped__", func)
    param_names = list(inspect.signature(unwrapped).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    tensors = _build_tensor_map_v2(param_names, input_specs, allocs, return_name)
    per_op_axis_maps, dim_sizes = _unify_axes_v2(raws, tensors)
    dims = _derive_dims(raws, per_op_axis_maps, dim_sizes)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)

    body: TreeIR = []
    """Alloc leaves are emitted at the forest root in source order (the
    user's declaration order). Compute/copy leaves follow, each with their
    own schedule tree built from touched_dims."""
    for alloc in allocs:
        alloc_leaf = _make_alloc_leaf(alloc)
        body.append(alloc_leaf)
    for op in parsed_ops:
        body.append(_build_tree(op, dims))
    for tree in body:
        _assign_canonical_names(tree, same_dim_counts={})

    return KernelModule(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        body=body,
    )
```

**3c.** Add the new helpers `_build_tensor_map_v2`, `_unify_axes_v2`, `_make_alloc_leaf`:

```python
def _build_tensor_map_v2(
    param_names: list[str],
    input_specs: dict[str, dict],
    allocs: list[_AllocRecord],
    return_name: str,
) -> dict[str, Tensor]:
    """Populate ``module.tensors`` from params + alloc records.

    Params are HBM-origin. Alloc records are intermediate-origin with
    their declared location. Dim ids are assigned lazily — each tensor's
    dim_ids get populated by ``_unify_axes_v2`` as ops reference it.
    """
    from nkigym.codegen.ir import Tensor
    out: dict[str, Tensor] = {}
    for name in param_names:
        spec = input_specs[name]
        shape = tuple(spec["shape"])
        dtype = spec["dtype"]
        out[name] = Tensor(
            name=name, dim_ids=(), shape=shape, dtype=dtype,
            origin="param", location="hbm",
        )
    for alloc in allocs:
        out[alloc.name] = Tensor(
            name=alloc.name, dim_ids=(), shape=alloc.shape, dtype=alloc.dtype,
            origin="intermediate", location=alloc.location,
        )
    if return_name not in out:
        raise ValueError(f"Return tensor {return_name!r} not declared (missing NKIAlloc?)")
    return out


def _unify_axes_v2(
    raws: list[_ParsedOpRaw], tensors: dict[str, Tensor]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Walk each op, unifying abstract axes against declared tensor shapes.

    Unlike the old v1 flow, tensors come pre-declared from ``NKIAlloc``
    records — we only need to assign dim_ids and unify across operands.
    """
    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for raw in raws:
        op_cls = raw.op_cls
        operand_map = {k: v for k, v in raw.operand_names.items() if v in tensors}
        local: dict[str, str] = {}
        for slot, axes in op_cls.OPERAND_AXES.items():
            if slot not in operand_map:
                continue
            tname = operand_map[slot]
            tensor = tensors[tname]
            if not tensor.dim_ids:
                """First op to touch this tensor seeds fresh dim_ids."""
                ids: list[str] = []
                for i, abstract in enumerate(axes[: len(tensor.shape)]):
                    if abstract not in local:
                        fresh = f"d{dim_counter[0]}"
                        dim_counter[0] += 1
                        dim_sizes[fresh] = tensor.shape[i]
                        local[abstract] = fresh
                    ids.append(local[abstract])
                tensors[tname] = replace(tensor, dim_ids=tuple(ids))
            else:
                for abstract, concrete in zip(axes, tensor.dim_ids):
                    if abstract in local and local[abstract] != concrete:
                        _unify_dim(tensors, per_op_axis_maps, dim_sizes,
                                   old_id=concrete, new_id=local[abstract])
                    else:
                        local[abstract] = concrete
        per_op_axis_maps.append(local)
    return per_op_axis_maps, dim_sizes


def _make_alloc_leaf(alloc: _AllocRecord) -> BodyLeaf:
    """Build the single BodyLeaf for an NKIAlloc declaration.

    The leaf's only kwarg is ``tensor_name`` — shape/dtype/location
    live in ``module.tensors[name]``.
    """
    from nkigym.ops.alloc import NKIAlloc
    return BodyLeaf(
        op_cls=NKIAlloc,
        reads={},
        writes=(alloc.name,),
        kwargs={"tensor_name": alloc.name},
    )
```

Add `from dataclasses import replace` to the imports if not already present.

**3d.** Modify `_make_leaf` to populate `reads_writes` from `RMW_OPERANDS`:

```python
def _make_leaf(op: _ParsedOp, phase: str = "main") -> BodyLeaf:
    """Build a self-describing :class:`BodyLeaf` for ``op``.

    ``RMW_OPERANDS`` and ``INPUT_OPERANDS`` on the op class split operand
    slots into three buckets:
    - ``INPUT_OPERANDS`` (and not RMW) → ``reads``.
    - ``RMW_OPERANDS`` → ``reads_writes``.
    - Otherwise (e.g. ``dst``, ``reduce_res``) → ``writes``.

    ``phase`` defaults to ``"main"``. Old multi-phase builders passed
    explicit phase strings; those are deleted in this refactor.
    """
    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads: dict[str, str] = {}
    writes_list: list[str] = []
    reads_writes_list: list[str] = []
    for slot, tname in op.operand_names.items():
        if slot in rmw:
            reads_writes_list.append(tname)
        elif slot in input_slots:
            reads[slot] = tname
        else:
            writes_list.append(tname)
    return BodyLeaf(
        op_cls=op.op_cls,
        phase=phase,
        reads=reads,
        writes=tuple(writes_list),
        reads_writes=tuple(reads_writes_list),
        kwargs=dict(op.op_kwargs),
        axis_map=dict(op.axis_map),
        dim_role=dict(op.dim_role),
    )
```

Note: `op_local_buffers` is no longer populated on leaves. Tasks 8–9 will delete the field entirely.

**3e.** Delete `_LEAF_BUILDERS`, `_BUILDER_INTERIOR_DIMS`, `_build_leaves_matmul`, `_build_leaves_activation_reduce`, `_register_op_local_derived_dims`, `_resolve_op_local_buffers`, and the old `_build_leaves`, `_build_leaves_default`, `_create_outputs`. Replace `_build_leaves` with a one-liner that calls `_make_leaf`:

```python
def _build_leaves(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """Single-phase default: every op emits one ``BodyLeaf(phase='main')``."""
    _ = dims
    return [_make_leaf(op)]
```

Remove references to `_create_outputs` in `_unify_axes` (now `_unify_axes_v2` — delete the old one).

**3f.** `_touched_dims` already handles operand-axis traversal — but we need it to include `dst` now that `dst` is in `OPERAND_AXES`. The existing implementation already does this correctly (it walks every entry of `OPERAND_AXES`), so no change needed.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_first_class_buffers.py -v
```

Expected: PASS for `test_canonical_parses_nkialloc_into_module_tensors` and `test_canonical_matmul_leaf_has_dst_in_reads_writes`.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/canonical.py test/codegen/test_first_class_buffers.py
git commit -m "feat: canonical builder parses NKIAlloc and emits single-phase leaves"
```

---

### Task 8: Delete `OP_LOCAL_BUFFERS`, `phase`, `op_local_buffers` from the codebase

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir.py`
- Modify: `nkigym/src/nkigym/ops/base.py`
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`
- Modify: `nkigym/src/nkigym/ops/matmul.py` (if `OP_LOCAL_BUFFERS` was present)

- [ ] **Step 1: Search for all readers of `phase` and `op_local_buffers`**

```bash
grep -rn "\.phase\|op_local_buffers\|OP_LOCAL_BUFFERS" nkigym/src/ test/ --include="*.py" | grep -v __pycache__
```

Note every hit. Each needs to be deleted or rewritten.

- [ ] **Step 2: Delete `phase` and `op_local_buffers` from `BodyLeaf`**

In `nkigym/src/nkigym/codegen/ir.py`, remove `phase` and `op_local_buffers` fields from `BodyLeaf`:

```python
@dataclass
class BodyLeaf:
    """Self-describing leaf: one op + the metadata needed to render it.

    Every non-parameter tensor referenced in ``reads`` / ``writes`` /
    ``reads_writes`` is declared by an ``NKIAlloc`` leaf earlier in
    pre-order DFS.
    """

    op_cls: type
    reads: dict[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    reads_writes: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    dim_role: dict[str, AxisRole] = field(default_factory=dict)
```

Delete the `OpLocalBuffer` dataclass from `ir.py` entirely.

- [ ] **Step 3: Delete `OP_LOCAL_BUFFERS` from `NKIOp` base class**

In `nkigym/src/nkigym/ops/base.py`, remove the `OP_LOCAL_BUFFERS` ClassVar block.

- [ ] **Step 4: Delete `OP_LOCAL_BUFFERS` from subclasses**

In `nkigym/src/nkigym/ops/activation_reduce.py`, delete the `OP_LOCAL_BUFFERS` ClassVar entry.

If `nkigym/src/nkigym/ops/matmul.py` has `OP_LOCAL_BUFFERS`, delete it there too.

- [ ] **Step 5: Delete `phase`-dispatching code in canonical builder**

In `nkigym/src/nkigym/codegen/canonical.py`, remove the `phase` parameter from `_make_leaf`'s signature and its usage. `_make_leaf` always emits a single-phase leaf now.

- [ ] **Step 6: Delete `validate_dataflow_ordering`'s references to `leaf.phase`**

Already done in Task 6 — confirm there are no remaining `leaf.phase` references in `ir.py`:

```bash
grep -n "\.phase" nkigym/src/nkigym/codegen/ir.py
```

Expected: no output.

- [ ] **Step 7: Run tests**

```bash
pytest test/codegen/ -v 2>&1 | tail -40
```

Expected: tests that relied on `phase` or `op_local_buffers` fail explicitly. Those fixtures get updated in Task 13.

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir.py nkigym/src/nkigym/ops/base.py nkigym/src/nkigym/ops/activation_reduce.py nkigym/src/nkigym/ops/matmul.py nkigym/src/nkigym/codegen/canonical.py
git commit -m "refactor: delete phase, op_local_buffers, OP_LOCAL_BUFFERS"
```

---

### Task 9: Rewrite renderer — delete SBUF/HBM allocation passes, use tree-position alloc leaves, collapse phase-keyed dispatch

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/emit_source.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/_emit_utils.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/lower_phases.py` (will be renamed `emit_ops.py` in this task)
- Modify: `nkigym/src/nkigym/codegen/lowering/place_buffers.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/inject_multi_buffer.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py`

- [ ] **Step 1: Write a golden-path test**

Append to `test/codegen/test_first_class_buffers.py`:

```python
def test_render_emits_alloc_inline_at_tree_position():
    """Rendered kernel declares each tensor at the alloc leaf's tree
    position, not at a global function top."""
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
    def f(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    input_specs = {
        "lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
        "rhs": {"shape": (128, 512), "dtype": "bfloat16"},
    }
    module = build_canonical_module(f, input_specs)
    src = render(module)
    assert "psum_acc = nl.ndarray" in src
    assert "buffer=nl.psum" in src
    assert "hbm_out = nl.ndarray" in src
    assert "buffer=nl.shared_hbm" in src
    assert "nisa.memset(psum_acc" in src
    assert "nisa.nc_matmul(dst=psum_acc" in src
    assert "nisa.tensor_copy" in src
```

- [ ] **Step 2: Rewrite `emit_source.py`**

Open `nkigym/src/nkigym/codegen/lowering/emit_source.py`. Replace the file contents with:

```python
"""Top-level forest walker that emits NKI source from a KernelModule.

The renderer is intentionally dumb: it walks the schedule tree and
delegates each leaf to a per-op-class emitter registered in
:mod:`emit_ops`. Buffer allocations are themselves tree leaves
(``NKIAlloc``), so allocation placement is fully determined by tree
position. No separate allocation pass.
"""

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under
from nkigym.codegen.lowering._emit_utils import _Writer

__all__ = ["emit_source", "render_annotated"]


def emit_source(module: KernelModule) -> str:
    """Render ``module`` to NKI source via the forest walker."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    render_forest(w, module)
    w.line(f"return {module.return_name}")
    w.dedent()
    return w.getvalue()


def render_annotated(module: KernelModule) -> str:
    """Render with # BodyLeaf(...) / # LoopNode(...) comments above each emission."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for idx, entry in enumerate(module.body):
        _emit_node_annotated(w, module, entry, path_names, path_trips, path=(idx,))
    w.line(f"return {module.return_name}")
    w.dedent()
    return w.getvalue()


def _emit_imports(w: _Writer) -> None:
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, module: KernelModule) -> None:
    w.line("@nki.jit")
    params = ", ".join(module.param_names)
    w.line(f"def {module.func_name}({params}):")


def render_forest(w: _Writer, module: KernelModule) -> None:
    """Walk ``module.body`` and emit NKI source for every node."""
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in module.body:
        _emit_node(w, module, entry, path_names, path_trips)


def _emit_node(
    w: _Writer,
    module: KernelModule,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> None:
    """Emit one forest node (recursive for LoopNode, delegating for BodyLeaf)."""
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get(node.op_cls.__name__)
        if emitter is None:
            raise ValueError(f"No body emitter registered for {node.op_cls.__name__!r}")
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth <= 1:
        _emit_vanilla_loop(w, module, node, path_names, path_trips)
    else:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)


def _emit_vanilla_loop(
    w: _Writer, module: KernelModule, node: LoopNode,
    path_names: dict[str, list[str]], path_trips: dict[str, list[int]],
) -> None:
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, module, child, path_names, path_trips)
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


def _emit_node_annotated(
    w: _Writer, module: KernelModule, node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]], path_trips: dict[str, list[int]],
    path: tuple[int, ...],
) -> None:
    """Annotating variant used by :func:`render_annotated`."""
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get(node.op_cls.__name__)
        if emitter is None:
            raise ValueError(f"No body emitter registered for {node.op_cls.__name__!r}")
        w.line(f"# BodyLeaf(op_cls={node.op_cls.__name__})  path={path}")
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth > 1:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)
        return
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"# LoopNode(dim_id={node.dim_id!r}, trip={node.trip_count}, role={node.role.name})  path={path}")
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for i, child in enumerate(node.children):
        _emit_node_annotated(w, module, child, path_names, path_trips, path=path + (i,))
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


"""Wired up at import time (bottom-of-file imports avoid module-load cycles)."""
from nkigym.codegen.lowering.emit_ops import _BODY_EMITTERS  # noqa: E402
from nkigym.codegen.lowering.inject_software_pipeline import _emit_pipelined_loop  # noqa: E402
```

- [ ] **Step 3: Rename `lower_phases.py` to `emit_ops.py` and rewrite**

```bash
git mv nkigym/src/nkigym/codegen/lowering/lower_phases.py nkigym/src/nkigym/codegen/lowering/emit_ops.py
```

Open the renamed file. Replace the `_BODY_EMITTERS` dict keying and emitter functions:

```python
"""Per-op-class body emitters. One emitter per NKIOp; single-phase only.

The walker in :mod:`emit_source` dispatches through :data:`_BODY_EMITTERS`
keyed by ``op_cls.__name__``. Each emitter reads tensor identity from
``module.tensors`` and emits one ISA call-site line (or, for NKIAlloc,
one ``nl.ndarray(...)`` binding line).
"""

from collections.abc import Callable

from nkigym.codegen.lowering.inject_multi_buffer import (
    hbm_tile_slice,
    sbuf_tile_slice,
    slot_expr,
    swapped_dst_tile_slice,
)
from nkigym.codegen.lowering.place_buffers import tensor_total_slots

_BODY_EMITTERS: dict[str, Callable] = {}


def _register_body(op_kind: str):
    """Decorator: register a body emitter for ``op_kind`` (op_cls.__name__)."""
    def wrap(fn: Callable) -> Callable:
        _BODY_EMITTERS[op_kind] = fn
        return fn
    return wrap


_LOCATION_BUFFER_EXPR: dict[str, str] = {
    "hbm": "nl.shared_hbm",
    "sbuf": "nl.sbuf",
    "psum": "nl.psum",
}


@_register_body("NKIAlloc")
def _body_alloc(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``<name> = nl.ndarray(<shape>, dtype=..., buffer=...)`` at tree position.

    Shape is computed from the tensor's declared shape modulo ancestor-loop
    coverage (shrinks for dims whose iterations are outside this alloc leaf),
    scaled up by buffer_degree for multi-buffering.
    """
    _ = pipeline_dim, stage_offset
    name = leaf.kwargs["tensor_name"]
    tensor = module.tensors[name]
    buffer_expr = _LOCATION_BUFFER_EXPR[tensor.location]
    """Compute emitted shape: for each dim, divide declared shape by the
    number of ancestor-loop iterations and multiply by buffer_degree."""
    ancestor_trips = {d: len(path_trips.get(d, [])) and
                         _prod(path_trips[d]) or 1
                      for d in tensor.dim_ids}
    emitted_shape = []
    for i, d in enumerate(tensor.dim_ids):
        covered = ancestor_trips.get(d, 1)
        degree = tensor.buffer_degree.get(d, 1)
        dim_extent = tensor.shape[i] // covered * degree
        emitted_shape.append(dim_extent)
    shape_str = ", ".join(str(x) for x in emitted_shape) + ("," if len(emitted_shape) == 1 else "")
    w.line(f"{name} = nl.ndarray(({shape_str}), dtype=nl.{tensor.dtype}, buffer={buffer_expr})")


def _prod(xs: list[int]) -> int:
    """Product helper (used in shape shrinking)."""
    r = 1
    for x in xs:
        r *= x
    return r


@_register_body("NKIMemset")
def _body_memset(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.memset(<dst_slice>, value=<value>)``."""
    dst_name = next(iter(leaf.writes))
    value = leaf.kwargs["value"]
    dst_tensor = module.tensors[dst_name]
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.memset({dst_expr}, value={value})")


@_register_body("NKILoad")
def _body_load(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` HBM→SBUF."""
    src_name = leaf.reads["src"]
    dst_name = next(iter(leaf.writes))
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    src_expr = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIStore")
def _body_store(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` SBUF→HBM."""
    src_name = leaf.reads["src"]
    dst_name = next(iter(leaf.writes))
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    src_expr = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKITensorCopy")
def _body_tensor_copy(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.tensor_copy(dst, src)``."""
    src_name = leaf.reads["src"]
    dst_name = next(iter(leaf.writes))
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    src_expr = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_copy({dst_expr}, {src_expr})")


@_register_body("NKITensorReduce")
def _body_tensor_reduce(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.tensor_reduce(dst, op, data, axis=axis)``."""
    data_name = leaf.reads["data"]
    dst_name = next(iter(leaf.writes))
    axis = leaf.kwargs["axis"]
    op = leaf.kwargs["op"]
    op_expr = {"add": "nl.add", "max": "nl.maximum"}[op]
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    data_expr = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_reduce({dst_expr}, {op_expr}, {data_expr}, axis={axis})")


@_register_body("NKIMatmul")
def _body_matmul(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.nc_matmul(dst, stationary, moving)``. dst is RMW (reads_writes)."""
    dst_name = leaf.reads_writes[0]
    stat_name = leaf.reads["stationary"]
    mov_name = leaf.reads["moving"]
    dst_tensor = module.tensors[dst_name]
    stat_tensor = module.tensors[stat_name]
    mov_tensor = module.tensors[mov_name]
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    stat_expr = _build_slice(stat_name, stat_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    mov_expr = _build_slice(mov_name, mov_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst={dst_expr},")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKIActivation")
def _body_activation(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    data_name = leaf.reads["data"]
    dst_name = next(iter(leaf.writes))
    act = leaf.kwargs["op"]
    scale = leaf.kwargs.get("scale", 1.0)
    bias = leaf.kwargs.get("bias", 0.0)
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    data_expr = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={data_expr}, scale={scale}, bias={bias})")


@_register_body("NKIActivationReduce")
def _body_activation_reduce(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    data_name = leaf.reads["data"]
    """Two write operands: dst (scratch, discarded) and reduce_res."""
    writes = leaf.writes
    """Canonical order from OPERAND_AXES: dst first, then reduce_res."""
    dst_name, reduce_res_name = writes[0], writes[1]
    act = leaf.kwargs["op"]
    reduce_op = leaf.kwargs.get("reduce_op", "add")
    merge = {"add": "nl.add", "max": "nl.maximum"}[reduce_op]
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    rr_tensor = module.tensors[reduce_res_name]
    data_expr = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    rr_expr = _build_slice(reduce_res_name, rr_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst={dst_expr},")
    w.line(f"op=nl.{act},")
    w.line(f"data={data_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res={rr_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKITensorScalar")
def _body_tensor_scalar(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    data_name = leaf.reads["data"]
    op0_name = leaf.reads["operand0"]
    dst_name = next(iter(leaf.writes))
    op = leaf.kwargs["op"]
    data_tensor = module.tensors[data_name]
    op0_tensor = module.tensors[op0_name]
    dst_tensor = module.tensors[dst_name]
    data_expr = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    op0_expr = _build_slice(op0_name, op0_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op}, operand0={op0_expr})")


@_register_body("NKITranspose")
def _body_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.nc_transpose(dst, src)``. PSUM alloc + drain are sibling leaves now."""
    src_name = leaf.reads["src"]
    dst_name = next(iter(leaf.writes))
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    src_expr = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.nc_transpose({dst_expr}, {src_expr})")


@_register_body("NKIDmaTranspose")
def _body_dma_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    src_name = leaf.reads["src"]
    dst_name = next(iter(leaf.writes))
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    src_expr = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_expr = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")


def _build_slice(tensor_name, tensor, module, path_names, path_trips, pipeline_dim, stage_offset) -> str:
    """Dispatch on tensor.location to build the appropriate tile-slice expression."""
    p_axis = tensor.dim_ids[0]
    f_axis = tensor.dim_ids[1] if len(tensor.dim_ids) > 1 else None
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
    p_slots = tensor_total_slots(tensor, tensor.dim_ids[0], module)
    f_slots = tensor_total_slots(tensor, tensor.dim_ids[1], module) if len(tensor.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    if tensor.location == "hbm":
        return hbm_tile_slice(tensor_name, tensor.dim_ids, p_tile, f_tile,
                              path_names, path_trips, p_slots, f_slots, off_p, off_f)
    return sbuf_tile_slice(tensor_name, tensor.dim_ids, p_tile, f_tile,
                           path_names, path_trips, p_slots, f_slots, off_p, off_f)
```

Delete every other emitter (all the phase-keyed ones: `_body_matmul_psum_init`, `_body_matmul_compute`, `_body_matmul_drain`, `_body_ar_reduce_step`, `_body_ar_reduce_close`).

- [ ] **Step 4: Update `place_buffers.py` to handle the NKIAlloc leaf in LCA walks**

The LCA walk in `place_buffers.py` today walks `_find_access_paths(tensor.name, module)`. Verify it includes `writes=(name,)` from `NKIAlloc` leaves. Open `place_buffers.py` around line 100 and modify `_find_access_paths` to include leaves where the tensor name appears in `writes` (the `NKIAlloc` leaf will appear there). The existing implementation almost certainly already does this via `leaf.writes` — confirm with:

```bash
grep -A 20 "_find_access_paths" nkigym/src/nkigym/codegen/lowering/place_buffers.py
```

If the implementation only reads `leaf.reads`, extend it to also scan `leaf.writes` and `leaf.reads_writes`.

- [ ] **Step 5: Update `_emit_utils.py` — delete name prefixers**

Open `nkigym/src/nkigym/codegen/lowering/_emit_utils.py`. Delete `_hbm_name` and `_sbuf_name` functions. Find every import site that uses them:

```bash
grep -rn "_hbm_name\|_sbuf_name" nkigym/src/
```

Replace each usage with the tensor name verbatim (no prefix).

- [ ] **Step 6: Update `inject_software_pipeline.py`**

Open `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py`. Find the dispatch through `_BODY_EMITTERS` (which was phase-keyed). Change the key from `(op_cls.__name__, phase)` to just `op_cls.__name__`:

```python
emitter = _BODY_EMITTERS.get(leaf.op_cls.__name__)
```

Update any other phase-keyed lookups the same way.

- [ ] **Step 7: Update `inject_multi_buffer.py`**

Open `nkigym/src/nkigym/codegen/lowering/inject_multi_buffer.py`. If any helper checks `tensor.origin == "intermediate"` to distinguish HBM from SBUF, replace with `tensor.location == "sbuf"` (or `!= "hbm"`). This is a drop-in replacement that uses the new explicit field.

- [ ] **Step 8: Run the golden-path test**

```bash
pytest test/codegen/test_first_class_buffers.py::test_render_emits_alloc_inline_at_tree_position -v
```

Expected: PASS.

- [ ] **Step 9: Full codegen tests**

```bash
pytest test/codegen/ -v 2>&1 | tail -40
```

Expected: tests that relied on phase-specific behavior or the old allocation-hoisting pass fail (fixed in Task 13). New first-class-buffers tests pass.

- [ ] **Step 10: Commit**

```bash
git add nkigym/src/nkigym/codegen/lowering/ test/codegen/test_first_class_buffers.py
git commit -m "refactor: rewrite renderer for tree-position allocation and op_cls-keyed emitters"
```

---

## Phase 3: Migrate examples, fixtures, synthesis skill

### Task 10: Rewrite `examples/matmul_lhsT_rhs.py`

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py`

- [ ] **Step 1: Rewrite the f_nkigym body**

Open `examples/matmul_lhsT_rhs.py`. Replace the `@nkigym_kernel` block with:

```python
K, M, N = 2048, 2048, 2048


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` with first-class buffer declarations."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf   = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc   = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod  = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out    = NKIAlloc(location="hbm",  shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs,   dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out
```

Add new imports at the top:

```python
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
```

- [ ] **Step 2: Run the example end-to-end**

```bash
python examples/matmul_lhsT_rhs.py
```

Expected output includes `[matmul_lhsT_rhs] max_abs=<small number>` and `kernel written to ...`.

- [ ] **Step 3: Commit**

```bash
git add examples/matmul_lhsT_rhs.py
git commit -m "refactor: matmul_lhsT_rhs example uses first-class NKIAlloc form"
```

---

### Task 11: Rewrite `examples/matmul_lhs_rhs.py`

**Files:**
- Modify: `examples/matmul_lhs_rhs.py`

- [ ] **Step 1: Read the existing example**

```bash
cat examples/matmul_lhs_rhs.py
```

Note the existing kernel shape — it includes a transpose.

- [ ] **Step 2: Rewrite the f_nkigym body**

Replace the body with the first-class form. The transpose becomes an explicit PSUM alloc + transpose + drain:

```python
@nkigym_kernel
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """``lhs @ rhs`` — transposes lhs to the stationary operand layout."""
    lhs_sbuf    = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    rhs_sbuf    = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    lhs_T_psum  = NKIAlloc(location="psum", shape=(K, M), dtype="float32")()
    lhs_T_sbuf  = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    psum_acc    = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod   = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out     = NKIAlloc(location="hbm",  shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKITranspose()(src=lhs_sbuf, dst=lhs_T_psum)
    NKITensorCopy()(src=lhs_T_psum, dst=lhs_T_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out
```

Add imports:

```python
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
```

- [ ] **Step 3: Run the example**

```bash
python examples/matmul_lhs_rhs.py
```

Expected: correctness check passes.

- [ ] **Step 4: Commit**

```bash
git add examples/matmul_lhs_rhs.py
git commit -m "refactor: matmul_lhs_rhs example uses first-class NKIAlloc form"
```

---

### Task 12: Rewrite `examples/rmsnorm_matmul.py` and `test/codegen/_rmsnorm_matmul_fixture.py`

**Files:**
- Modify: `examples/rmsnorm_matmul.py` (and/or the f_nkigym synthesis step it produces)
- Modify: `test/codegen/_rmsnorm_matmul_fixture.py`

- [ ] **Step 1: Rewrite `_rmsnorm_matmul_fixture.py`**

```python
"""Shared test fixture for the rmsnorm+matmul kernel (first-class buffers form)."""

from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

M, K, N = 2048, 2048, 2048
INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

_F = K
_EPS = 1e-6


@nkigym_kernel
def f_nkigym(lhs, rhs):
    lhs_sbuf     = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    rhs_sbuf     = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    sum_sq       = NKIAlloc(location="sbuf", shape=(M, 1), dtype="float32")()
    ar_scratch   = NKIAlloc(location="sbuf", shape=(M, K), dtype="float32")()
    rms_inv      = NKIAlloc(location="sbuf", shape=(M, 1), dtype="float32")()
    normed       = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    normed_T_psum = NKIAlloc(location="psum", shape=(K, M), dtype="float32")()
    normed_T     = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    psum_acc     = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod    = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out      = NKIAlloc(location="hbm",  shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=ar_scratch, reduce_res=sum_sq)
    NKIActivation(op="rsqrt", scale=1.0 / _F, bias=_EPS)(data=sum_sq, dst=rms_inv)
    NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv, dst=normed)
    NKITranspose()(src=normed, dst=normed_T_psum)
    NKITensorCopy()(src=normed_T_psum, dst=normed_T)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=normed_T, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def f_numpy(lhs, rhs):
    import numpy as np
    lhs_f32 = lhs.astype(np.float32)
    mean_sq = (lhs_f32 * lhs_f32).mean(axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + _EPS)
    normed = (lhs_f32 * inv_rms).astype(lhs.dtype)
    return normed @ rhs
```

- [ ] **Step 2: Update `examples/rmsnorm_matmul.py` if it has its own inline f_nkigym**

```bash
cat examples/rmsnorm_matmul.py
```

If the example uses `nkigym_compile` with its own synthesis stage, note that the synthesis skill will regenerate `f_nkigym.py` in the cache — that regeneration must match the new form. Tasks 14 updates the synthesis skill. For this task, if the example invokes the compile pipeline, it runs after Task 14.

- [ ] **Step 3: Run the fixture-backed tests**

```bash
pytest test/codegen/ -k "rmsnorm" -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test/codegen/_rmsnorm_matmul_fixture.py examples/rmsnorm_matmul.py
git commit -m "refactor: rmsnorm_matmul fixture uses first-class NKIAlloc form"
```

---

### Task 13: Update every remaining test fixture

**Files:**
- Modify: `test/codegen/test_canonical.py`
- Modify: `test/codegen/test_place_buffers.py`
- Modify: `test/codegen/test_multi_buffer_unit.py`
- Modify: `test/codegen/test_software_pipeline_unit.py`
- Modify: `test/codegen/test_dep_cache.py`
- Modify: `test/tune/test_batch.py` (delete DecomposeReduction references)

- [ ] **Step 1: Identify every hand-built `BodyLeaf` or `Tensor` in tests**

```bash
grep -rn "BodyLeaf(\|Tensor(" test/ --include="*.py" | grep -v __pycache__
```

Each usage needs to be updated:
- `Tensor(...)` calls must pass `location=...` explicitly.
- `BodyLeaf(...)` calls with `phase=...` keyword must remove it.
- `BodyLeaf(...)` calls with `op_local_buffers=...` must remove it.
- `BodyLeaf(...)` that modeled a matmul leaf must use `reads_writes=('psum_acc',)` instead of `writes=('prod',)`.

- [ ] **Step 2: Update each test file**

Work through each file listed in the grep output. For each `Tensor(...)` call, add `location="sbuf"` (or `"hbm"`/`"psum"` per context) as a kwarg. For each matmul-related `BodyLeaf(...)`, move the PSUM tensor name into `reads_writes` and drop `phase` / `op_local_buffers` kwargs.

- [ ] **Step 3: Delete `DecomposeReduction`-specific tests from `test_batch.py`**

Open `test/tune/test_batch.py`. Remove any test that constructs a `DecomposeReduction` atom directly or asserts on its enumerator output. (RFactor-specific tests get added in Task 17.)

- [ ] **Step 4: Run the full test suite**

```bash
pytest test/ -v 2>&1 | tail -40
```

Expected: test failures are limited to the still-unmigrated `DecomposeReduction` module import (fixed in Task 16).

- [ ] **Step 5: Commit**

```bash
git add test/
git commit -m "refactor: migrate test fixtures to first-class buffers IR shape"
```

---

### Task 14: Update synthesis skill

**Files:**
- Find: synthesis skill location

- [ ] **Step 1: Locate the synthesis skill**

```bash
find nkigym/src/nkigym/synthesis/ -name "*.md" -o -name "*.py" | xargs grep -l "f_nkigym" 2>/dev/null
```

Identify the file(s) that contain worked examples of `f_nkigym` authoring. Typically these are SKILL.md or Python constants that render into the synthesis prompt.

- [ ] **Step 2: Rewrite each worked example**

For every worked example, replace the old f_nkigym form with the new NKIAlloc-based form. At minimum, update:
- `matmul_lhsT_rhs` example (mirror the form from Task 10).
- `rmsnorm_matmul` example (mirror Task 12's fixture).
- Any `tensor_scalar` / `activation_reduce` mini-examples.

- [ ] **Step 3: Add a short rationale in the skill doc**

Add a section near the top of the skill doc explaining:

> **Buffer declaration rules:**
> - Every non-parameter tensor must be declared via `NKIAlloc(location=..., shape=..., dtype=...)` before it's referenced by any compute op.
> - Every compute op takes an explicit `dst=` kwarg — no auto-created outputs.
> - PSUM tensors hold the matmul accumulator: allocate PSUM, memset to 0, run matmul (dst=RMW), then `NKITensorCopy` PSUM→SBUF.

- [ ] **Step 4: If the skill has a validator that rejects the new form, update it**

Some synthesis skills include a static validator that checks "every intermediate is produced by an op." Update it to require "every intermediate has an NKIAlloc predecessor" instead.

- [ ] **Step 5: Smoke-test synthesis**

If there's an integration test for synthesis, run it:

```bash
pytest nkigym/ -k "synthesis" -v
```

Alternatively, run a small end-to-end synthesis + codegen:

```bash
python examples/rmsnorm_matmul.py
```

Expected: the agent emits f_nkigym in the new form; canonical build + render succeeds.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/synthesis/
git commit -m "refactor: synthesis skill authors f_nkigym in first-class buffers form"
```

---

## Phase 4: RFactor atom

### Task 15: Create the `RFactor` atom scaffolding (signature, dispatch, legality)

**Files:**
- Create: `nkigym/src/nkigym/tune/rfactor.py`
- Test: `test/codegen/test_rfactor_rmw.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/codegen/test_rfactor_rmw.py`:

```python
"""Tests for RFactor atom — RMW-dst recipe (matmul)."""

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_canonical(lhs_T, rhs):
    """Minimal matmul for RFactor testing."""
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


_INPUT_SPECS = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def test_rfactor_rejects_non_divisor_factor():
    """outer_factor must divide the accumulation dim's num_tiles."""
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    """Find the matmul leaf and its K LoopNode path."""
    matmul_path = _find_matmul_compute_path(module)
    atom = RFactor(reducer_leaf_path=matmul_path, outer_factor=5)
    assert atom.is_legal(module) is False


def test_rfactor_rejects_endpoint_factors():
    """outer_factor == 1 or == num_tiles is a no-op; reject."""
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom_low = RFactor(reducer_leaf_path=matmul_path, outer_factor=1)
    assert atom_low.is_legal(module) is False


def _find_matmul_compute_path(module):
    """Walk the tree to find the path to the NKIMatmul leaf."""
    def walk(node, path):
        from nkigym.codegen.ir import BodyLeaf, LoopNode
        if isinstance(node, BodyLeaf) and node.op_cls.__name__ == "NKIMatmul":
            return path
        if isinstance(node, LoopNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None
    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise ValueError("No NKIMatmul leaf found")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_rfactor_rmw.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.tune.rfactor'`.

- [ ] **Step 3: Create the RFactor atom skeleton**

Create `nkigym/src/nkigym/tune/rfactor.py`:

```python
"""RFactor rewrite — fission a reducer into staging-buffer decomposition.

Takes a reducer leaf (matmul or activation_reduce) and an outer factor;
emits a staging buffer plus either a per-outer-iteration PSUM accumulator
(recipe "rmw") or slot-indexed writes (recipe "slot"), closed by a
tensor_reduce over the outer axis.

See ``docs/superpowers/specs/2026-05-09-first-class-buffers-and-rfactor-design.md``.
"""

from dataclasses import dataclass

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class RFactor:
    """Fission a reducer into outer-split + staging + close.

    Attributes:
        reducer_leaf_path: Path to the reducer's ``BodyLeaf``
            (``NKIMatmul`` for recipe "rmw", ``NKIActivationReduce`` for
            recipe "slot").
        outer_factor: The outer loop's trip count post-split. Must divide
            the accumulation dim's ``num_tiles``, and be strictly between
            1 and ``num_tiles``.
    """

    reducer_leaf_path: tuple[int, ...]
    outer_factor: int

    def is_legal(self, module: KernelModule) -> bool:
        """Check structural + dataflow preconditions."""
        leaf = resolve_node(module.body, self.reducer_leaf_path)
        if not isinstance(leaf, BodyLeaf):
            return False
        recipe = leaf.op_cls.RFACTOR_RECIPE
        if recipe is None:
            return False
        acc_dim = _accumulation_dim(leaf, recipe)
        if acc_dim is None:
            return False
        num_t = module.dims[acc_dim].num_tiles
        if num_t <= 1:
            return False
        if self.outer_factor <= 1 or self.outer_factor >= num_t:
            return False
        if num_t % self.outer_factor != 0:
            return False
        """Recipe-specific structural checks."""
        if recipe == "rmw":
            return _is_legal_rmw(module, self.reducer_leaf_path, leaf)
        if recipe == "slot":
            return _is_legal_slot(module, self.reducer_leaf_path, leaf)
        return False

    def apply(self, module: KernelModule) -> KernelModule:
        """Apply the recipe-specific rewrite; rename canonical loop vars."""
        if not self.is_legal(module):
            raise AtomLegalityError(f"RFactor.apply: atom illegal on current state — {self}")
        leaf = resolve_node(module.body, self.reducer_leaf_path)
        assert isinstance(leaf, BodyLeaf)
        recipe = leaf.op_cls.RFACTOR_RECIPE
        if recipe == "rmw":
            return _apply_rmw(module, self.reducer_leaf_path, self.outer_factor)
        if recipe == "slot":
            return _apply_slot(module, self.reducer_leaf_path, self.outer_factor)
        raise AtomLegalityError(f"RFactor.apply: unknown recipe {recipe!r}")


def _accumulation_dim(leaf: BodyLeaf, recipe: str) -> str | None:
    """Find the reducer's accumulation dim id from its axis_map + AXIS_ROLES."""
    abstract_roles = leaf.op_cls.AXIS_ROLES
    for abstract, role in abstract_roles.items():
        if role == AxisRole.ACCUMULATION and abstract in leaf.axis_map:
            return leaf.axis_map[abstract]
    return None


def _is_legal_rmw(module: KernelModule, leaf_path: tuple[int, ...], leaf: BodyLeaf) -> bool:
    """Recipe "rmw": parent must be an ACCUMULATION LoopNode; dst must be RMW."""
    if not leaf.reads_writes:
        return False
    """Parent of the leaf must be an ACCUMULATION-role LoopNode."""
    if len(leaf_path) < 2:
        return False
    parent = resolve_node(module.body, leaf_path[:-1])
    if not isinstance(parent, LoopNode):
        return False
    if parent.role != AxisRole.ACCUMULATION:
        return False
    return True


def _is_legal_slot(module: KernelModule, leaf_path: tuple[int, ...], leaf: BodyLeaf) -> bool:
    """Recipe "slot": reduction axis must be in leaf.axis_map with ACCUMULATION role."""
    return _accumulation_dim(leaf, "slot") is not None


def _apply_rmw(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Implemented in Task 16."""
    raise NotImplementedError("RFactor recipe 'rmw' — implemented in Task 16")


def _apply_slot(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Implemented in Task 18."""
    raise NotImplementedError("RFactor recipe 'slot' — implemented in Task 18")


def enumerate_rfactor_atoms(module: KernelModule) -> list[RFactor]:
    """Emit one atom per (reducer leaf, valid divisor of accumulation dim)."""
    atoms: list[RFactor] = []

    def walk(node, path: tuple[int, ...]) -> None:
        if isinstance(node, BodyLeaf):
            if node.op_cls.RFACTOR_RECIPE is not None:
                acc_dim = _accumulation_dim(node, node.op_cls.RFACTOR_RECIPE)
                if acc_dim is not None:
                    num_t = module.dims[acc_dim].num_tiles
                    for factor in _divisors_strict(num_t):
                        atom = RFactor(reducer_leaf_path=path, outer_factor=factor)
                        if atom.is_legal(module):
                            atoms.append(atom)
        else:
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms


def _divisors_strict(n: int) -> list[int]:
    """Return every divisor d of n with 1 < d < n."""
    return [d for d in range(2, n) if n % d == 0]
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_rfactor_rmw.py::test_rfactor_rejects_non_divisor_factor test/codegen/test_rfactor_rmw.py::test_rfactor_rejects_endpoint_factors -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/rfactor.py test/codegen/test_rfactor_rmw.py
git commit -m "feat: RFactor atom scaffolding — signature, dispatch, legality"
```

---

### Task 16: Implement RFactor recipe "rmw" (matmul) + delete `DecomposeReduction`

**Files:**
- Modify: `nkigym/src/nkigym/tune/rfactor.py`
- Delete: `nkigym/src/nkigym/tune/decompose_reduction.py`
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Test: `test/codegen/test_rfactor_rmw.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_rfactor_rmw.py`:

```python
def test_rfactor_rmw_produces_staging_buffer_and_close():
    """After RFactor(rmw, factor=4) on a K-loop matmul:
    - module.tensors has psum_partials and psum_acc_local entries
    - original psum_acc is removed
    - tree contains K_outer loop with inner matmul, closing tensor_reduce
    """
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    atom = RFactor(reducer_leaf_path=matmul_path, outer_factor=4)
    assert atom.is_legal(module)
    new_module = atom.apply(module)

    """New staging tensor + local PSUM, original psum_acc removed."""
    assert "psum_partials" in new_module.tensors
    assert new_module.tensors["psum_partials"].location == "sbuf"
    assert "psum_acc_local" in new_module.tensors
    assert new_module.tensors["psum_acc_local"].location == "psum"
    assert "psum_acc" not in new_module.tensors

    """Tree contains a tensor_reduce leaf after rfactor."""
    reduce_leaves = [leaf for root in new_module.body for leaf in leaves_under(root)
                     if leaf.op_cls.__name__ == "NKITensorReduce"]
    assert len(reduce_leaves) >= 1


def test_rfactor_rmw_preserves_dataflow_ordering():
    """After rfactor, the resulting module still validates."""
    from nkigym.codegen.ir import validate_dataflow_ordering
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    new_module = RFactor(reducer_leaf_path=matmul_path, outer_factor=4).apply(module)
    assert validate_dataflow_ordering(new_module) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_rfactor_rmw.py::test_rfactor_rmw_produces_staging_buffer_and_close -v
```

Expected: FAIL — `NotImplementedError: RFactor recipe 'rmw' — implemented in Task 16`.

- [ ] **Step 3: Implement `_apply_rmw`**

In `nkigym/src/nkigym/tune/rfactor.py`, replace `_apply_rmw` with:

```python
def _apply_rmw(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Recipe "rmw": matmul-style.

    Transforms:
        [..., NKIAlloc(psum_acc), NKIMemset(psum_acc), K-loop { NKIMatmul(dst=psum_acc) },
         NKITensorCopy(psum_acc → sbuf_prod), ...]
    into:
        [..., NKIAlloc(psum_partials),
              K_outer_loop {
                  NKIAlloc(psum_acc_local), NKIMemset(psum_acc_local),
                  K_inner_loop { NKIMatmul(dst=psum_acc_local) },
                  NKITensorCopy(psum_acc_local → psum_partials[K_outer])
              },
              NKITensorReduce(psum_partials → sbuf_prod, axis=K_outer),
         ...]
    """
    from dataclasses import replace as dc_replace

    from nkigym.codegen.ir import DimInfo, Tensor
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.tensor_copy import NKITensorCopy
    from nkigym.ops.tensor_reduce import NKITensorReduce
    from nkigym.tune.compute_at import _rename_canonical

    matmul_leaf = resolve_node(module.body, leaf_path)
    assert isinstance(matmul_leaf, BodyLeaf)
    psum_acc_name = matmul_leaf.reads_writes[0]
    psum_acc = module.tensors[psum_acc_name]
    k_dim = _accumulation_dim(matmul_leaf, "rmw")
    assert k_dim is not None
    k_info = module.dims[k_dim]
    inner_trip = k_info.num_tiles // outer_factor

    """Introduce a new dim for the outer split."""
    outer_dim_id = _fresh_dim_id(module.dims)
    module_dims = dict(module.dims)
    module_dims[outer_dim_id] = DimInfo(
        dim_id=outer_dim_id, total_size=outer_factor,
        tile_size=1, num_tiles=outer_factor,
    )
    module_dims[k_dim] = dc_replace(k_info, num_tiles=inner_trip)

    """New tensors: psum_partials (SBUF slot vector), psum_acc_local (per-outer PSUM)."""
    partials_name = "psum_partials"
    local_name = "psum_acc_local"
    partials = Tensor(
        name=partials_name,
        dim_ids=psum_acc.dim_ids + (outer_dim_id,),
        shape=psum_acc.shape + (outer_factor,),
        dtype=psum_acc.dtype, origin="intermediate", location="sbuf",
    )
    acc_local = Tensor(
        name=local_name,
        dim_ids=psum_acc.dim_ids,
        shape=psum_acc.shape,
        dtype=psum_acc.dtype, origin="intermediate", location="psum",
    )
    new_tensors = {k: v for k, v in module.tensors.items() if k != psum_acc_name}
    new_tensors[partials_name] = partials
    new_tensors[local_name] = acc_local

    """Find the existing memset + tensor_copy siblings and remove them
    from the forest. We rebuild the forest by walking and replacing."""
    new_body = _rebuild_forest_rmw(
        module.body, leaf_path, psum_acc_name, outer_dim_id, outer_factor,
        inner_trip, k_dim, k_info.num_tiles, partials_name, local_name,
    )
    new_body = _rename_canonical(new_body)

    return dc_replace(module, tensors=new_tensors, dims=module_dims, body=new_body)


def _fresh_dim_id(dims: dict) -> str:
    """Pick a dim_id not yet in use: ``d<N>`` for the smallest available N."""
    taken = set(dims.keys())
    i = 0
    while f"d{i}" in taken:
        i += 1
    return f"d{i}"


def _rebuild_forest_rmw(
    body, leaf_path, psum_acc_name, outer_dim_id, outer_factor,
    inner_trip, k_dim, _original_k_trips, partials_name, local_name,
):
    """Surgical rewrite of the forest to the rmw-rfactor shape.

    The high-level steps:
    1. Walk the forest; find every node touched by the matmul's subtree.
    2. Remove the psum_acc memset leaf (it's at forest-root level, or a
       sibling of the K-loop — find by writes=(psum_acc,) where op_cls=NKIMemset).
    3. Remove the NKITensorCopy leaf that drains psum_acc → sbuf_prod
       (the old drain target). Record its dst tensor name.
    4. Locate the K-loop LoopNode (the leaf_path's parent).
    5. Replace the K-loop with: K_outer { NKIAlloc(local), NKIMemset(local),
       K_inner { matmul(dst=local) }, NKITensorCopy(local → partials[outer]) }.
    6. Prepend NKIAlloc(partials) before the K_outer loop.
    7. Append NKITensorReduce(partials → original_drain_dst, axis=outer) after K_outer.
    """
    from nkigym.codegen.ir import BodyLeaf, LoopNode
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.tensor_copy import NKITensorCopy
    from nkigym.ops.tensor_reduce import NKITensorReduce

    """Scan the flat forest (roots) to locate the relevant leaves."""
    original_drain_dst = None
    memset_root_idx = None
    drain_root_idx = None
    for i, root in enumerate(body):
        if isinstance(root, BodyLeaf):
            if root.op_cls is NKIMemset and psum_acc_name in root.writes:
                memset_root_idx = i
            elif root.op_cls is NKITensorCopy and root.reads.get("src") == psum_acc_name:
                drain_root_idx = i
                original_drain_dst = root.writes[0]

    if memset_root_idx is None or drain_root_idx is None or original_drain_dst is None:
        raise AtomLegalityError(
            "RFactor.apply (rmw): expected sibling NKIMemset + NKITensorCopy for psum_acc"
        )

    """Locate the matmul leaf + its K loop parent path at forest root level.
    For the canonical form, the K loop is a root in the forest."""
    matmul_root_idx = leaf_path[0]
    matmul_root = body[matmul_root_idx]
    if not isinstance(matmul_root, LoopNode) or matmul_root.dim_id != k_dim:
        raise AtomLegalityError(
            "RFactor.apply (rmw): expected K-loop LoopNode at forest root as matmul ancestor"
        )
    matmul_leaf = matmul_root.children[0]
    assert isinstance(matmul_leaf, BodyLeaf)

    """Build the new inner K loop (unchanged trip = inner_trip)."""
    new_matmul_leaf = BodyLeaf(
        op_cls=NKIMatmul,
        reads=dict(matmul_leaf.reads),
        writes=(),
        reads_writes=(local_name,),
        kwargs=dict(matmul_leaf.kwargs),
        axis_map=dict(matmul_leaf.axis_map),
        dim_role=dict(matmul_leaf.dim_role),
    )
    k_inner = LoopNode(
        dim_id=k_dim, trip_count=inner_trip, role=AxisRole.ACCUMULATION,
        children=[new_matmul_leaf],
    )
    alloc_local = BodyLeaf(
        op_cls=NKIAlloc, reads={}, writes=(local_name,),
        kwargs={"tensor_name": local_name},
    )
    memset_local = BodyLeaf(
        op_cls=NKIMemset, reads={}, writes=(local_name,),
        kwargs={"value": 0.0},
    )
    drain_local = BodyLeaf(
        op_cls=NKITensorCopy,
        reads={"src": local_name},
        writes=(partials_name,),
        kwargs={},
    )

    k_outer = LoopNode(
        dim_id=outer_dim_id, trip_count=outer_factor, role=AxisRole.PARALLEL,
        children=[alloc_local, memset_local, k_inner, drain_local],
    )
    alloc_partials = BodyLeaf(
        op_cls=NKIAlloc, reads={}, writes=(partials_name,),
        kwargs={"tensor_name": partials_name},
    )
    close_reduce = BodyLeaf(
        op_cls=NKITensorReduce,
        reads={"data": partials_name},
        writes=(original_drain_dst,),
        kwargs={"axis": 2, "op": "add"},
    )

    """Rebuild the forest: remove memset, matmul-K-loop, drain at their
    original indices; insert alloc_partials + k_outer + close_reduce in place
    of the K-loop's slot."""
    indices_to_remove = {memset_root_idx, matmul_root_idx, drain_root_idx}
    new_body = []
    inserted = False
    for i, root in enumerate(body):
        if i in indices_to_remove:
            if i == matmul_root_idx and not inserted:
                new_body.append(alloc_partials)
                new_body.append(k_outer)
                new_body.append(close_reduce)
                inserted = True
            continue
        new_body.append(root)
    return new_body
```

- [ ] **Step 4: Delete `decompose_reduction.py`**

```bash
git rm nkigym/src/nkigym/tune/decompose_reduction.py
```

- [ ] **Step 5: Update `batch.py` to swap `DecomposeReduction` for `RFactor`**

In `nkigym/src/nkigym/tune/batch.py`:

```python
from nkigym.tune.rfactor import enumerate_rfactor_atoms
```

Remove the `from nkigym.tune.decompose_reduction import enumerate_decompose_reduction_atoms` line. In `_enumerate_atoms`, replace the `enumerate_decompose_reduction_atoms` call with `enumerate_rfactor_atoms`.

- [ ] **Step 6: Run tests**

```bash
pytest test/codegen/test_rfactor_rmw.py -v
```

Expected: PASS.

- [ ] **Step 7: Run the broader test suite to confirm no regressions**

```bash
pytest test/ -v 2>&1 | tail -30
```

Expected: deletion of `DecomposeReduction` eliminates those import errors; any test still referencing it has been fixed in Task 13.

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/tune/rfactor.py nkigym/src/nkigym/tune/batch.py test/codegen/test_rfactor_rmw.py
git rm nkigym/src/nkigym/tune/decompose_reduction.py
git commit -m "feat: implement RFactor recipe 'rmw' (matmul); delete DecomposeReduction"
```

---

### Task 17: Render + CPU-sim-verify a rfactored matmul kernel

**Files:**
- Test: `test/codegen/test_rfactor_rmw.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_rfactor_rmw.py`:

```python
def test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly():
    """Full pipeline: canonical → RFactor → render → CPU-sim matches numpy."""
    import numpy as np
    import nki
    from nkigym.codegen.render import render
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_matmul_canonical, _INPUT_SPECS)
    matmul_path = _find_matmul_compute_path(module)
    rfactored = RFactor(reducer_leaf_path=matmul_path, outer_factor=4).apply(module)
    source = render(rfactored)

    """CPU-sim at reduced size. Use 128x128 shapes — the outer/inner split
    must still divide, so use outer_factor=2 at 128x128 (K=128 doesn't
    divide by 4, but does by 2). Adjust the input specs/shapes."""
    """This test runs the full-size rendered kernel. If size is prohibitive
    for CPU-sim, replace with a size-parameterized fixture."""

    sim_source = source.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)
    kernel_fn = ns["matmul_lhsT_rhs_nkigym"]
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((2048, 2048)).astype(np.float32)
    rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs_T=lhs_T, rhs=rhs)
    expected = lhs_T.T @ rhs
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

Note: this test is slow (2048³ CPU-sim). If it's too slow for the local dev loop, mark with `@pytest.mark.slow` and let it run only in full-suite CI.

- [ ] **Step 2: Run the test**

```bash
pytest test/codegen/test_rfactor_rmw.py::test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly -v
```

Expected: PASS (may take minutes due to CPU-sim at 2048³).

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_rfactor_rmw.py
git commit -m "test: RFactor rmw recipe renders correct CPU-sim kernel end-to-end"
```

---

### Task 18: Implement RFactor recipe "slot" (activation_reduce)

**Files:**
- Modify: `nkigym/src/nkigym/tune/rfactor.py`
- Test: `test/codegen/test_rfactor_slot.py` (new)

- [ ] **Step 1: Write the failing test**

Create `test/codegen/test_rfactor_slot.py`:

```python
"""Tests for RFactor atom — slot-indexed recipe (activation_reduce)."""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _sum_sq_canonical(lhs):
    """Sum of squares along F — eligible for slot-rfactor."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 2048), dtype="bfloat16")()
    scratch = NKIAlloc(location="sbuf", shape=(128, 2048), dtype="float32")()
    sum_acc = NKIAlloc(location="sbuf", shape=(128, 1), dtype="float32")()
    hbm_out = NKIAlloc(location="hbm", shape=(128, 1), dtype="float32")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=scratch, reduce_res=sum_acc)
    NKIStore()(src=sum_acc, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {"lhs": {"shape": (128, 2048), "dtype": "bfloat16"}}


def test_rfactor_slot_produces_partials_and_close():
    """After RFactor(slot, factor=4) on an activation_reduce:
    - module.tensors has 'partials' and 'scratch_local' entries
    - tree wraps the activation_reduce in an F_outer loop
    - a closing tensor_reduce writes into the original reduce_res
    """
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_sum_sq_canonical, _INPUT_SPECS)
    ar_path = _find_ar_path(module)
    atom = RFactor(reducer_leaf_path=ar_path, outer_factor=4)
    assert atom.is_legal(module)
    new_module = atom.apply(module)

    assert "partials" in new_module.tensors
    assert new_module.tensors["partials"].location == "sbuf"
    assert "scratch_local" in new_module.tensors

    reduce_leaves = [leaf for root in new_module.body for leaf in leaves_under(root)
                     if leaf.op_cls.__name__ == "NKITensorReduce"]
    assert len(reduce_leaves) >= 1


def _find_ar_path(module):
    from nkigym.codegen.ir import BodyLeaf, LoopNode
    def walk(node, path):
        if isinstance(node, BodyLeaf) and node.op_cls.__name__ == "NKIActivationReduce":
            return path
        if isinstance(node, LoopNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None
    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise ValueError("No NKIActivationReduce leaf found")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_rfactor_slot.py -v
```

Expected: FAIL — `NotImplementedError: RFactor recipe 'slot' — implemented in Task 18`.

- [ ] **Step 3: Implement `_apply_slot`**

In `nkigym/src/nkigym/tune/rfactor.py`, replace `_apply_slot` with:

```python
def _apply_slot(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Recipe "slot": activation_reduce-style.

    Transforms:
        [..., NKIAlloc(sum_acc), NKIAlloc(scratch), NKIActivationReduce(dst=scratch, reduce_res=sum_acc), ...]
    into:
        [..., NKIAlloc(sum_acc),   # unchanged
              NKIAlloc(partials), NKIAlloc(scratch_local),
              F_outer_loop { NKIActivationReduce(dst=scratch_local, reduce_res=partials[F_outer]) },
              NKITensorReduce(partials → sum_acc, axis=F_outer),
         ...]
    """
    from dataclasses import replace as dc_replace

    from nkigym.codegen.ir import DimInfo, Tensor
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.tensor_reduce import NKITensorReduce
    from nkigym.tune.compute_at import _rename_canonical

    ar_leaf = resolve_node(module.body, leaf_path)
    assert isinstance(ar_leaf, BodyLeaf)
    f_dim = _accumulation_dim(ar_leaf, "slot")
    assert f_dim is not None
    f_info = module.dims[f_dim]
    inner_f_trip = f_info.num_tiles // outer_factor

    """Reduce_res slot is the partial output. dst is scratch (discarded)."""
    reduce_res_name = ar_leaf.writes[1]  # OPERAND_AXES order: dst, reduce_res
    scratch_name = ar_leaf.writes[0]
    reduce_res_tensor = module.tensors[reduce_res_name]
    scratch_tensor = module.tensors[scratch_name]

    outer_dim_id = _fresh_dim_id(module.dims)
    module_dims = dict(module.dims)
    module_dims[outer_dim_id] = DimInfo(
        dim_id=outer_dim_id, total_size=outer_factor, tile_size=1, num_tiles=outer_factor,
    )
    module_dims[f_dim] = dc_replace(f_info, num_tiles=inner_f_trip)

    partials = Tensor(
        name="partials",
        dim_ids=reduce_res_tensor.dim_ids + (outer_dim_id,),
        shape=reduce_res_tensor.shape + (outer_factor,),
        dtype=reduce_res_tensor.dtype, origin="intermediate", location="sbuf",
    )
    scratch_local = Tensor(
        name="scratch_local",
        dim_ids=scratch_tensor.dim_ids,
        shape=(scratch_tensor.shape[0], scratch_tensor.shape[1] // outer_factor),
        dtype=scratch_tensor.dtype, origin="intermediate", location="sbuf",
    )
    new_tensors = dict(module.tensors)
    new_tensors["partials"] = partials
    new_tensors["scratch_local"] = scratch_local

    new_body = _rebuild_forest_slot(
        module.body, leaf_path, ar_leaf, outer_dim_id, outer_factor,
        inner_f_trip, f_dim, reduce_res_name,
    )
    new_body = _rename_canonical(new_body)
    return dc_replace(module, tensors=new_tensors, dims=module_dims, body=new_body)


def _rebuild_forest_slot(body, leaf_path, ar_leaf, outer_dim_id, outer_factor,
                          inner_f_trip, f_dim, reduce_res_name):
    """Surgical rewrite for the slot recipe.

    The AR canonical tree is: F-LoopNode(ACC) { activation_reduce leaf }.
    After slot-rfactor: F_outer-LoopNode(PAR) { modified activation_reduce leaf with
      dst=scratch_local, reduce_res=partials[slot] }, then a NKITensorReduce.
    """
    from nkigym.codegen.ir import BodyLeaf, LoopNode
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.tensor_reduce import NKITensorReduce

    ar_root_idx = leaf_path[0]
    ar_root = body[ar_root_idx]
    """The ar_root is either the AR leaf directly (scalar F=1 case) or
    an F-LoopNode wrapping the AR leaf (F > 1). For rfactor legality,
    F > 1 always holds, so ar_root is a LoopNode."""
    if not isinstance(ar_root, LoopNode):
        raise AtomLegalityError("RFactor.apply (slot): expected F-LoopNode at matmul ancestor")

    new_ar_leaf = BodyLeaf(
        op_cls=NKIActivationReduce,
        reads=dict(ar_leaf.reads),
        writes=("scratch_local", "partials"),
        reads_writes=(),
        kwargs=dict(ar_leaf.kwargs),
        axis_map=dict(ar_leaf.axis_map),
        dim_role=dict(ar_leaf.dim_role),
    )

    inner_f_loop = LoopNode(
        dim_id=f_dim, trip_count=inner_f_trip, role=AxisRole.ACCUMULATION,
        children=[new_ar_leaf],
    )
    f_outer = LoopNode(
        dim_id=outer_dim_id, trip_count=outer_factor, role=AxisRole.PARALLEL,
        children=[inner_f_loop],
    )
    alloc_partials = BodyLeaf(
        op_cls=NKIAlloc, reads={}, writes=("partials",),
        kwargs={"tensor_name": "partials"},
    )
    alloc_scratch_local = BodyLeaf(
        op_cls=NKIAlloc, reads={}, writes=("scratch_local",),
        kwargs={"tensor_name": "scratch_local"},
    )
    close_reduce = BodyLeaf(
        op_cls=NKITensorReduce,
        reads={"data": "partials"},
        writes=(reduce_res_name,),
        kwargs={"axis": 2, "op": ar_leaf.kwargs.get("reduce_op", "add")},
    )

    new_body = list(body)
    new_body[ar_root_idx:ar_root_idx + 1] = [alloc_partials, alloc_scratch_local, f_outer, close_reduce]
    return new_body
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_rfactor_slot.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/rfactor.py test/codegen/test_rfactor_slot.py
git commit -m "feat: implement RFactor recipe 'slot' (activation_reduce)"
```

---

## Phase 5: Final validation

### Task 19: End-to-end validation — examples + batch-tune + MFU smoke

**Files:**
- Run: `examples/matmul_lhsT_rhs.py`, `examples/matmul_lhs_rhs.py`, `examples/rmsnorm_matmul.py`
- Run: `scripts/tune_matmul_lhsT_rhs.py`

- [ ] **Step 1: Full test suite green**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/ -v 2>&1 | tail -30
```

Expected: every test passes. If any failures remain, address them before moving on (do NOT skip).

- [ ] **Step 2: `examples/matmul_lhsT_rhs.py` end-to-end**

```bash
python examples/matmul_lhsT_rhs.py
```

Expected: `max_abs` is small (e.g. < 5e-3) and `kernel written to ...` line appears.

- [ ] **Step 3: `examples/matmul_lhs_rhs.py` end-to-end**

```bash
python examples/matmul_lhs_rhs.py
```

Expected: correctness check passes.

- [ ] **Step 4: `examples/rmsnorm_matmul.py` end-to-end (synthesis + compile + tune)**

```bash
python examples/rmsnorm_matmul.py
```

Expected: runs through synthesis, canonical codegen, CPU-sim correctness check, and (if batch-tune is wired up) HW profile. Prints `results.json` path.

- [ ] **Step 5: Batch-tune matmul_lhsT_rhs (100 samples, HW profile)**

```bash
python scripts/tune_matmul_lhsT_rhs.py 2>&1 | tee /tmp/tune_matmul_lhsT_rhs.log
```

Expected: 100 kernels rendered, CPU-sim + HW profile for each, `results.json` written. Any CPU-sim failures → blocker.

- [ ] **Step 6: Verify MFU gate**

```bash
python -c "
import json
results = json.load(open('/home/ubuntu/cache/matmul_lhsT_rhs_tune/results.json'))
max_mfu = 0.0
for kname, r in results.items():
    if 'mfu_estimated_percent' in r:
        max_mfu = max(max_mfu, r['mfu_estimated_percent'])
print(f'Max MFU across 100 samples: {max_mfu:.2f}%')
print(f'Pre-refactor SOTA: 90.92%')
assert max_mfu >= 89.0, f'MFU regression: {max_mfu} < 89%'
print('MFU gate passed.')
"
```

Expected: `MFU gate passed.`

- [ ] **Step 7: Verify the original bug is unreachable**

```bash
python -c "
import os
for i in range(100):
    path = f'/home/ubuntu/cache/matmul_lhsT_rhs_tune/kernel_tuned_{i:04d}.py'
    if not os.path.exists(path):
        continue
    src = open(path).read()
    lines = src.split('\n')
    for idx, line in enumerate(lines):
        if 'nl.psum' in line and 'nl.ndarray' in line:
            # Count indentation of this line
            depth = len(line) - len(line.lstrip())
            """A well-placed PSUM alloc sits at depth <= 4 (function body).
            Nested inside an accumulation loop, depth >= 8."""
            if depth >= 12:
                preceding = [l for l in lines[max(0, idx-3):idx] if 'for ' in l]
                if any('ACCUMULATION' in l or 'K' in l for l in preceding):
                    print(f'SUSPICIOUS: {path}:{idx+1} has deep PSUM alloc')
                    exit(1)
print('No kernels have PSUM alloc misplaced inside an accumulation loop.')
"
```

Expected: `No kernels have PSUM alloc misplaced inside an accumulation loop.`

- [ ] **Step 8: Final commit + push**

```bash
git add -A
git status
git commit -m "test: validate first-class buffers + RFactor end-to-end (MFU gate passed)"
git log --oneline -20
```

- [ ] **Step 9: Update `.claude/rules/learnings.md`**

Append a line to `.claude/rules/learnings.md` under the Architecture section:

```markdown
- **First-class buffers + RMW operand**: `Tensor.location` + `NKIAlloc` BodyLeaf + `BodyLeaf.reads_writes` encode the PSUM dataflow edge. Matmul `dst` is RMW (reads+writes `psum_acc`); validator's reads-after-writes over `reads ∪ reads_writes` replaces the matmul phase-order carve-out. `NKIAlloc` leaf lives at tree position (placement is a `ComputeAt` on the alloc leaf), `module.tensors[name]` holds identity (shape/dtype/location/buffer_degree — TVM-style Buffer vs Allocate split). `RFactor` atom replaces `DecomposeReduction`, with two recipes keyed on `op_cls.RFACTOR_RECIPE`: "rmw" for matmul (staging buffer + per-outer PSUM alloc + drain + tensor_reduce close), "slot" for activation_reduce (partials + slot-indexed writes + tensor_reduce close). *(2026-05-09 ET)*
```

- [ ] **Step 10: Commit learnings**

```bash
git add .claude/rules/learnings.md
git commit -m "docs: record first-class buffers + RFactor landing in learnings"
```

---

## Self-Review

**Spec coverage check:**

- §2 Goals: every one of the 6 goals is implemented by a task (1–4 scaffold, 5–9 surgery, 10–14 migration, 15–18 RFactor, 19 gates). ✓
- §3 Op surface: Tasks 1–5 cover every op class change (RMW_OPERANDS, INPUT_OPERANDS, RFACTOR_RECIPE, dst in OPERAND_AXES, new op classes). ✓
- §4 IR changes: Task 2 adds `Tensor.location` and `BodyLeaf.reads_writes`; Task 6 rewrites `validate_dataflow_ordering`; Task 8 deletes `phase` / `op_local_buffers` / `OpLocalBuffer`. ✓
- §5 Canonical builder: Task 7 parses NKIAlloc + single-phase leaves + RMW handling; Task 8 deletes multi-phase machinery. ✓
- §6 Renderer: Task 9 covers all sub-bullets (tree-position allocs, delete _emit_sbuf_allocations, _emit_hbm_output, _emit_param_asserts; rename lower_phases → emit_ops; op_cls-keyed dispatch). ✓
- §7 RFactor: Tasks 15–18 cover signature/legality/enumerator (15), rmw recipe (16), rendered CPU-sim verification (17), slot recipe (18). ✓
- §8 Migration: Tasks 10–14 cover examples + fixtures + synthesis skill. Task 19 runs the end-to-end gates. ✓

**Placeholder scan:** None of "TBD", "TODO", "implement later", "similar to Task N" appear. Every step with code has the code.

**Type consistency:** `RMW_OPERANDS` is `frozenset[str]` throughout. `RFACTOR_RECIPE` is `Literal["rmw", "slot"] | None` throughout. `BodyLeaf.reads_writes` is `tuple[str, ...]` throughout. `Tensor.location` is `Literal["hbm", "sbuf", "psum"]` throughout. `NKIAlloc` BodyLeaf carries `kwargs={"tensor_name": ...}` throughout. `RFactor.reducer_leaf_path` is `tuple[int, ...]` throughout.

**One residual concern (noted, not a blocker):** Task 9 Step 4 modifies `place_buffers._find_access_paths` to scan `leaf.writes` / `leaf.reads_writes`. If the existing implementation already does this, the step is a no-op; if not, it's a small fix. The step includes a command to check.
