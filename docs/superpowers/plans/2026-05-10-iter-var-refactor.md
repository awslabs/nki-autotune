# TVM-style Iter-Var IR Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nkigym's path-based `LoopNode`/`BodyLeaf` IR with a TVM-style iter-var identity model (`ForNode`/`SBlock`/`IterVar`) so `ComputeAt`/`ReverseComputeAt` preserve consumer loop chains and every reordering is an explicit atom.

**Architecture:** Big-bang replacement on an isolated `iter-var-refactor` branch. 4 phases: data model → canonical+render parity → atoms → MFU acceptance. The 106-test suite is the regression harness; MFU parity on three tuned workloads is the acceptance gate. Tuning pauses through Phase C.

**Tech Stack:** Python 3.12, dataclasses, NKI (`nki.simulate`, `nki.jit`), `pytest`. Development venv at `~/venvs/kernel-env/bin/activate`. PYTHONPATH includes `/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/nki-autotune/autotune/src`.

**Spec:** `docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`

---

## Conventions

Every task's tests go under `test/codegen/` or `test/tune/` matching the module it covers. All commands assume the kernel-env venv is active:

```bash
source ~/venvs/kernel-env/bin/activate
export PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/nki-autotune/autotune/src:$PYTHONPATH
```

Test runner: `pytest -x test/<path> -v`. `-x` stops on first failure.

Commit message format: `refactor: <short summary>` for code changes; `test: <short summary>` for test-only commits; `docs: <short summary>` for documentation. Co-authored trailer per project convention.

---

## File Structure (post-refactor)

```
nkigym/src/nkigym/
├── codegen/
│   ├── ir.py                              # IterVar, ForNode, SBlock, NKIOpCall, BufferAccess, AccessRange, KernelIR, resolve_node, replace_at_path, blocks_under, validate_dataflow_ordering
│   ├── canonical.py                       # build_initial_ir (AST parse → iter-var-based IR)
│   ├── dep_cache.py                       # DepCache keyed on SBlockId; _classify_edge folds reads_writes
│   ├── render.py                          # render entry point
│   └── lowering/
│       ├── place_buffers.py               # N-D LCA walk over iter-var ancestors
│       ├── inject_annotations/
│       │   ├── __init__.py                # Dispatcher; walks tree, invokes per-key sub-passes
│       │   ├── buffer_degree.py           # "buffer_degree" key — widens Tensor.buffer_degree, slot-modulo
│       │   └── software_pipeline.py       # "software_pipeline_depth" key — prologue/body/epilogue
│       ├── emit_ops.py                    # Per-op_cls ISA call emitters
│       ├── emit_source.py                 # Forest walker
│       └── _emit_utils.py                 # Slice emission, naming helpers
├── tune/
│   ├── __init__.py                        # KernelRewrite protocol, AtomLegalityError
│   ├── split.py                           # Split (iter-var-aware)
│   ├── reorder.py                         # Reorder (n-ary, iter-var-keyed)
│   ├── fuse.py                            # Fuse (renderer emits //, %)
│   ├── compute_at.py                      # ComputeAt (prefix-match + role promotion)
│   ├── reverse_compute_at.py              # ReverseComputeAt (dual)
│   ├── rfactor.py                         # RFactor (rmw + slot, 3D staging works)
│   ├── annotate.py                        # Unified Annotate atom + per-key validators
│   ├── batch.py                           # enumerate_pool; now enumerates 7 atoms
│   └── verify.py                          # CPU-sim verification helper (unchanged)
```

Deleted in this refactor:
- `nkigym/src/nkigym/tune/hoist_invariant.py` (redundant with ComputeAt)
- `nkigym/src/nkigym/tune/multi_buffer.py` (consolidated into Annotate)
- `nkigym/src/nkigym/tune/software_pipeline.py` (consolidated into Annotate)
- `nkigym/src/nkigym/codegen/lowering/inject_multi_buffer.py`
- `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py`
- `test/tune/test_hoist_invariant.py`

---

## Phase A — Data Model (Week 1)

### Task 1: Create isolated worktree

- [ ] **Step 1: Invoke using-git-worktrees skill**

Use `superpowers:using-git-worktrees` to create a worktree named `iter-var-refactor` cut from `dev_1`. This provides isolation from the current workspace.

Expected: worktree path printed; subsequent tasks operate inside the worktree.

- [ ] **Step 2: Set working directory + verify branch**

```bash
cd <worktree-path>
git branch --show-current
```

Expected output: `iter-var-refactor`.

- [ ] **Step 3: Verify test baseline**

```bash
source ~/venvs/kernel-env/bin/activate
export PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/nki-autotune/autotune/src:$PYTHONPATH
pytest test/ -x --collect-only -q 2>&1 | tail -5
```

Expected: baseline test count reported (document the exact number — ~106 tests per learnings).

- [ ] **Step 4: Commit worktree setup note**

Nothing to commit yet; proceed to Task 2.

---

### Task 2: Define `IterVar` dataclass

**Files:**
- Create: `nkigym/src/nkigym/codegen/ir_v2.py` (temporary sibling; replaces `ir.py` in Task 10)
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for IterVar (TVM-style iter-var identity)."""

import pytest
from nkigym.codegen.ir_v2 import IterVar
from nkigym.ops.base import AxisRole


def test_iter_var_is_frozen() -> None:
    """IterVar must be immutable — atoms retire and replace, never mutate."""
    iv = IterVar(var_id=0, dim_id="d0", extent=4, role=AxisRole.PARALLEL)
    with pytest.raises(AttributeError):
        iv.extent = 8  # type: ignore[misc]


def test_iter_var_equality_by_fields() -> None:
    """Two IterVars with the same fields compare equal."""
    a = IterVar(var_id=0, dim_id="d0", extent=4, role=AxisRole.PARALLEL)
    b = IterVar(var_id=0, dim_id="d0", extent=4, role=AxisRole.PARALLEL)
    assert a == b
    assert hash(a) == hash(b)


def test_iter_var_distinct_ids() -> None:
    """Same dim, different var_ids → unequal (distinct iter vars)."""
    a = IterVar(var_id=0, dim_id="d0", extent=4, role=AxisRole.PARALLEL)
    b = IterVar(var_id=1, dim_id="d0", extent=4, role=AxisRole.PARALLEL)
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: `ModuleNotFoundError: nkigym.codegen.ir_v2`.

- [ ] **Step 3: Implement IterVar**

```python
"""Iter-var-identity IR for the nkigym tune stage.

Transitional module: lives alongside ``ir.py`` through Phase A + B;
replaces ``ir.py`` in Task 10. TVM-style dataclasses
(IterVar, ForNode, SBlock, NKIOpCall, BufferAccess, AccessRange) replace
today's path-based LoopNode/BodyLeaf. See
``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md``.
"""

from dataclasses import dataclass

from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class IterVar:
    """Stable identity for a loop iteration variable.

    Created by the canonical builder and every Split / Fuse. Never
    mutated — atoms retire iter vars and emit fresh ones.

    Attributes:
        var_id: Monotonic unique id per module.
        dim_id: Concrete dim this iter var traverses.
        extent: Trip count (# tiles).
        role: PARALLEL / SEQUENTIAL / ACCUMULATION.
    """

    var_id: int
    dim_id: str
    extent: int
    role: AxisRole
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: add IterVar dataclass for iter-var-identity IR"
```

---

### Task 3: Define `AccessRange` and `BufferAccess`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing tests**

Append to `test/codegen/test_ir_v2.py`:

```python
from nkigym.codegen.ir_v2 import AccessRange, BufferAccess


def test_access_range_immutable() -> None:
    """AccessRange must be frozen."""
    ar = AccessRange(iter_var_coeffs={0: 1}, const_offset=0, extent=128)
    with pytest.raises(AttributeError):
        ar.const_offset = 4  # type: ignore[misc]


def test_access_range_simple_1to1() -> None:
    """coeffs={iv: 1}, offset=0, extent=tile_size encodes direct indexing."""
    ar = AccessRange(iter_var_coeffs={7: 1}, const_offset=0, extent=128)
    assert ar.iter_var_coeffs == {7: 1}
    assert ar.extent == 128


def test_access_range_split_rewrite() -> None:
    """After Split(factor=2), an iter var coefficient becomes inner_extent."""
    ar = AccessRange(iter_var_coeffs={0: 2, 1: 1}, const_offset=0, extent=128)
    assert ar.iter_var_coeffs[0] == 2  # outer iter var × inner_extent
    assert ar.iter_var_coeffs[1] == 1  # inner iter var × 1


def test_buffer_access_immutable() -> None:
    """BufferAccess is frozen."""
    ba = BufferAccess(
        tensor_name="x",
        iter_var_ids=(0,),
        pattern=(AccessRange(iter_var_coeffs={0: 1}, const_offset=0, extent=128),),
    )
    with pytest.raises(AttributeError):
        ba.tensor_name = "y"  # type: ignore[misc]


def test_buffer_access_uses_tuple_for_patterns() -> None:
    """pattern field must be a tuple (frozen dataclass requires hashable fields)."""
    p = (
        AccessRange(iter_var_coeffs={0: 1}, const_offset=0, extent=128),
        AccessRange(iter_var_coeffs={1: 1}, const_offset=0, extent=2048),
    )
    ba = BufferAccess(tensor_name="t", iter_var_ids=(0, 1), pattern=p)
    assert len(ba.pattern) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: `ImportError: cannot import name 'AccessRange'`.

- [ ] **Step 3: Implement AccessRange + BufferAccess**

Append to `ir_v2.py`:

```python
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class AccessRange:
    """Affine access for one buffer dim: sum(coeff * iv) + const_offset.

    Attributes:
        iter_var_coeffs: Frozen mapping iter_var_id → coefficient.
            Stored as a tuple of (id, coeff) pairs internally to stay
            hashable; expose as dict via ``coeffs`` property.
        const_offset: Constant offset added to the affine form.
        extent: Per-iteration extent (the tile size along this dim).
    """

    iter_var_coeffs: Mapping[int, int]
    const_offset: int
    extent: int

    def __post_init__(self) -> None:
        """Convert dict to tuple-of-pairs for hashability."""
        if isinstance(self.iter_var_coeffs, dict):
            object.__setattr__(
                self, "iter_var_coeffs", tuple(sorted(self.iter_var_coeffs.items()))
            )

    @property
    def coeffs(self) -> dict[int, int]:
        """Return coefficients as a dict (for ergonomic access)."""
        if isinstance(self.iter_var_coeffs, tuple):
            return dict(self.iter_var_coeffs)
        return dict(self.iter_var_coeffs)


@dataclass(frozen=True)
class BufferAccess:
    """Which region of a tensor a block reads or writes.

    TVM BufferRegion analog. The renderer consumes ``pattern`` to emit
    slice expressions; cache_read / cache_write atoms (future) consume
    it to infer staging buffer shapes.

    Attributes:
        tensor_name: Name of the tensor in ``module.tensors``.
        iter_var_ids: Tuple of iter_var ids that index this buffer.
        pattern: One AccessRange per tensor dim, in tensor-order.
    """

    tensor_name: str
    iter_var_ids: tuple[int, ...]
    pattern: tuple[AccessRange, ...]
```

Wait — `Mapping` isn't hashable. Use `tuple[tuple[int, int], ...]` as the stored form:

```python
@dataclass(frozen=True)
class AccessRange:
    """Affine access for one buffer dim: sum(coeff * iv) + const_offset.

    iter_var_coeffs is stored as a sorted tuple of (id, coeff) pairs for
    hashability; construct via the ``make`` classmethod if you have a dict.
    """

    iter_var_coeffs: tuple[tuple[int, int], ...]
    const_offset: int
    extent: int

    @classmethod
    def make(cls, coeffs: dict[int, int], const_offset: int, extent: int) -> "AccessRange":
        """Construct from a dict of coefficients; normalizes ordering."""
        return cls(
            iter_var_coeffs=tuple(sorted(coeffs.items())),
            const_offset=const_offset,
            extent=extent,
        )

    @property
    def coeffs(self) -> dict[int, int]:
        """Return coefficients as a dict."""
        return dict(self.iter_var_coeffs)
```

Update tests to construct via `AccessRange.make(...)`.

- [ ] **Step 4: Update test file to use AccessRange.make**

Replace every `AccessRange(iter_var_coeffs={...}, ...)` in `test/codegen/test_ir_v2.py` with `AccessRange.make(coeffs={...}, ...)`.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: add AccessRange + BufferAccess dataclasses"
```

---

### Task 4: Define `NKIOpCall` and `SBlock`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing tests**

Append to `test/codegen/test_ir_v2.py`:

```python
from nkigym.codegen.ir_v2 import NKIOpCall, SBlock
from nkigym.ops.matmul import NKIMatmul


def test_nki_op_call_immutable() -> None:
    """NKIOpCall is frozen."""
    call = NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})
    with pytest.raises(AttributeError):
        call.op_cls = object  # type: ignore[misc]


def test_sblock_single_leaf_canonical() -> None:
    """Canonical build emits single-NKIOpCall SBlocks."""
    call = NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={"K": "d0", "M": "d1", "N": "d3"},
                     dim_role={"d0": AxisRole.ACCUMULATION, "d1": AxisRole.PARALLEL,
                               "d3": AxisRole.PARALLEL})
    lhs_access = BufferAccess(tensor_name="lhs_T_sbuf", iter_var_ids=(0, 1),
                              pattern=(AccessRange.make({0: 1}, 0, 128),
                                       AccessRange.make({1: 1}, 0, 128)))
    block = SBlock(
        iter_vars=[IterVar(0, "d0", 4, AxisRole.ACCUMULATION)],
        reads={"stationary": lhs_access},
        writes={},
        reads_writes={},
        body=[call],
    )
    assert len(block.body) == 1
    assert block.iter_vars[0].var_id == 0


def test_sblock_supports_multi_leaf_body() -> None:
    """SBlock.body is a list; future fused-block atoms will emit len > 1."""
    calls = [
        NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={}),
        NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={}),
    ]
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=calls)
    assert len(block.body) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: ImportError for `NKIOpCall` / `SBlock`.

- [ ] **Step 3: Implement NKIOpCall + SBlock**

Append to `ir_v2.py`:

```python
from dataclasses import field


@dataclass(frozen=True)
class NKIOpCall:
    """One ISA call inside an SBlock.body.

    Attributes:
        op_cls: The NKIOp subclass.
        kwargs: Op-level kwargs (e.g. value=0.0 for memset, op="square" for
            activation_reduce).
        axis_map: Abstract axis → concrete dim id.
        dim_role: Concrete dim → AxisRole. Op-local; same dim can have
            different roles across ops in the module.
    """

    op_cls: type
    kwargs: dict[str, Any]
    axis_map: dict[str, str]
    dim_role: dict[str, AxisRole]


@dataclass
class SBlock:
    """Atomic (or fused) compute block.

    Multi-leaf blocks are supported in the data model; canonical builder
    always emits single-leaf. Fusion atoms (future) produce len > 1.

    Attributes:
        iter_vars: Block-local iter vars, canonical order (output-axis
            dims first, then reduction dims).
        reads: slot_name → BufferAccess (read-only operands).
        writes: slot_name → BufferAccess (write-only operands).
        reads_writes: slot_name → BufferAccess (RMW operands).
        body: Ordered list of NKIOpCalls.
        annotations: Keyed annotations consumed by lowering passes.
    """

    iter_vars: list[IterVar]
    reads: dict[str, "BufferAccess"]
    writes: dict[str, "BufferAccess"]
    reads_writes: dict[str, "BufferAccess"]
    body: list[NKIOpCall]
    annotations: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: add NKIOpCall + SBlock dataclasses"
```

---

### Task 5: Define `ForNode`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing tests**

```python
from nkigym.codegen.ir_v2 import ForNode


def test_for_node_binds_iter_var_by_reference() -> None:
    """ForNode stores an IterVar; multiple ForNodes can bind distinct iter
    vars on the same dim."""
    iv_outer = IterVar(0, "d0", 2, AxisRole.PARALLEL)
    iv_inner = IterVar(1, "d0", 2, AxisRole.PARALLEL)
    outer = ForNode(iter_var=iv_outer, children=[])
    inner = ForNode(iter_var=iv_inner, children=[])
    outer.children.append(inner)
    assert outer.iter_var.var_id == 0
    assert outer.children[0].iter_var.var_id == 1


def test_for_node_annotations_default_empty() -> None:
    """annotations default to empty dict."""
    iv = IterVar(0, "d0", 4, AxisRole.PARALLEL)
    fn = ForNode(iter_var=iv, children=[])
    assert fn.annotations == {}


def test_for_node_supports_sblock_child() -> None:
    """ForNode.children can hold SBlocks and nested ForNodes."""
    iv = IterVar(0, "d0", 4, AxisRole.PARALLEL)
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    fn = ForNode(iter_var=iv, children=[block])
    assert isinstance(fn.children[0], SBlock)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: `ImportError: cannot import name 'ForNode'`.

- [ ] **Step 3: Implement ForNode**

Append to `ir_v2.py`:

```python
@dataclass
class ForNode:
    """A for-loop in the schedule tree. Binds one IterVar by reference.

    ``annotations`` holds keyed annotations (e.g. "software_pipeline_depth")
    consumed by lowering passes.

    Attributes:
        iter_var: The IterVar bound to this loop.
        children: Nested ForNodes and SBlocks.
        name: Canonical rendered name (i_<dim>_<ordinal>); assigned by
            the canonicalize pass.
        annotations: Keyed annotations.
    """

    iter_var: IterVar
    children: "list[ForNode | SBlock]"
    name: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: add ForNode dataclass"
```

---

### Task 6: Define `KernelIR` v2 + `TreeIR` type alias

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing test**

```python
from nkigym.codegen.ir_v2 import KernelIR, TreeIR
from nkigym.ir.ir import Tensor, DimInfo  # reuse existing
from nkigym.ops.base import TensorOrigin


def test_kernel_ir_minimal() -> None:
    """KernelIR carries signature + tensors + dims + body + iter_var_counter."""
    tensors = {
        "x": Tensor(name="x", dim_ids=("d0",), shape=(128,), dtype="float32",
                    origin="param", location="hbm", buffer_degree={}),
    }
    dims = {"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)}
    m = KernelIR(
        func_name="f",
        param_names=["x"],
        return_name="x",
        tensors=tensors,
        dims=dims,
        iter_var_counter=0,
        body=[],
    )
    assert m.iter_var_counter == 0
    assert m.body == []


def test_kernel_ir_allocates_monotonic_iter_var_ids() -> None:
    """allocate_iter_var bumps the counter and returns a fresh IterVar."""
    m = KernelIR(func_name="f", param_names=[], return_name="",
                     tensors={}, dims={}, iter_var_counter=0, body=[])
    iv1 = m.allocate_iter_var("d0", extent=4, role=AxisRole.PARALLEL)
    iv2 = m.allocate_iter_var("d0", extent=4, role=AxisRole.PARALLEL)
    assert iv1.var_id == 0
    assert iv2.var_id == 1
    assert m.iter_var_counter == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement KernelIR**

Append to `ir_v2.py`:

```python
from nkigym.ir.ir import DepCache, DimInfo, Tensor  # reuse unchanged types


TreeIR = list["ForNode | SBlock"]


@dataclass
class KernelIR:
    """Envelope IR. Minimal delta from v1 — adds iter_var_counter, retypes body.

    Attributes:
        func_name: Emitted kernel name.
        param_names: Signature order.
        return_name: Tensor name of the return.
        tensors: All named tensors, keyed by name.
        dims: All dims, keyed by dim_id.
        iter_var_counter: Monotonic counter for IterVar.var_id allocation.
        body: Schedule tree — list of ForNode / SBlock roots.
        dep: Per-scope dependency cache (lazy).
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    iter_var_counter: int = 0
    body: TreeIR = field(default_factory=list)
    dep: DepCache = field(default_factory=lambda: DepCache(scopes={}))

    def allocate_iter_var(self, dim_id: str, extent: int, role: AxisRole) -> IterVar:
        """Allocate a fresh IterVar with a monotonic unique id.

        Never reuses retired var_ids.
        """
        iv = IterVar(var_id=self.iter_var_counter, dim_id=dim_id, extent=extent, role=role)
        self.iter_var_counter += 1
        return iv
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: add KernelIR v2 with iter_var allocator"
```

---

### Task 7: Tree utilities — `resolve_node`, `replace_at_path`, `blocks_under`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing tests**

```python
from nkigym.codegen.ir_v2 import blocks_under, resolve_node, replace_at_path


def _mk_mod_with_single_block() -> KernelIR:
    """Build a minimal module: one ForNode → SBlock tree."""
    m = KernelIR(func_name="f", param_names=[], return_name="",
                     tensors={}, dims={}, iter_var_counter=0, body=[])
    iv = m.allocate_iter_var("d0", 4, AxisRole.PARALLEL)
    block = SBlock(iter_vars=[iv], reads={}, writes={}, reads_writes={}, body=[])
    root = ForNode(iter_var=iv, children=[block])
    m.body.append(root)
    return m


def test_resolve_node_root() -> None:
    """Path (0,) returns the first root."""
    m = _mk_mod_with_single_block()
    node = resolve_node(m.body, (0,))
    assert isinstance(node, ForNode)


def test_resolve_node_nested_block() -> None:
    """Path (0, 0) returns the SBlock under the root ForNode."""
    m = _mk_mod_with_single_block()
    node = resolve_node(m.body, (0, 0))
    assert isinstance(node, SBlock)


def test_resolve_node_invalid_path() -> None:
    """Out-of-range path returns None."""
    m = _mk_mod_with_single_block()
    assert resolve_node(m.body, (5,)) is None


def test_blocks_under_returns_descendant_sblocks() -> None:
    """blocks_under walks the subtree yielding every SBlock."""
    m = _mk_mod_with_single_block()
    root = m.body[0]
    assert isinstance(root, ForNode)
    blocks = list(blocks_under(root))
    assert len(blocks) == 1
    assert isinstance(blocks[0], SBlock)


def test_replace_at_path_swaps_subtree() -> None:
    """replace_at_path returns a new body with the node at path replaced."""
    m = _mk_mod_with_single_block()
    new_block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={},
                       body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={},
                                        dim_role={})])
    new_body = replace_at_path(m.body, (0, 0), new_block)
    new_root = new_body[0]
    assert isinstance(new_root, ForNode)
    assert new_root.children[0] is new_block
    # Original untouched
    assert m.body[0].children[0] is not new_block  # type: ignore[union-attr]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: ImportError.

- [ ] **Step 3: Implement utilities**

Append to `ir_v2.py`:

```python
from collections.abc import Iterator


def resolve_node(forest: TreeIR, path: tuple[int, ...]) -> "ForNode | SBlock | None":
    """Walk ``path`` from the forest root; return node or None on invalid path.

    Args:
        forest: Top-level list of trees.
        path: Tuple of child indices from forest root down to target.

    Returns:
        Node at ``path``, or ``None`` if the path is empty, out of range,
        or attempts to descend through an SBlock.
    """
    if not path:
        return None
    siblings: list[ForNode | SBlock] = list(forest)
    node: ForNode | SBlock | None = None
    for idx in path:
        if idx < 0 or idx >= len(siblings):
            return None
        node = siblings[idx]
        if isinstance(node, SBlock):
            siblings = []
        else:
            siblings = node.children
    return node


def replace_at_path(
    forest: TreeIR, path: tuple[int, ...], replacement: "ForNode | SBlock"
) -> TreeIR:
    """Return a new forest with ``replacement`` placed at ``path``.

    Ancestors along ``path`` are rebuilt; untouched subtrees pass by reference.
    """
    if not path:
        raise ValueError("replace_at_path: path must be non-empty")
    if len(path) == 1:
        idx = path[0]
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    if not isinstance(parent, ForNode):
        raise ValueError("replace_at_path: non-ForNode ancestor")
    new_children = replace_at_path(parent.children, rest, replacement)
    new_parent = ForNode(
        iter_var=parent.iter_var,
        children=new_children,
        name=parent.name,
        annotations=dict(parent.annotations),
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def blocks_under(node: "ForNode | SBlock") -> Iterator[SBlock]:
    """Yield every SBlock in ``node``'s subtree (includes ``node`` if SBlock)."""
    if isinstance(node, SBlock):
        yield node
        return
    for child in node.children:
        yield from blocks_under(child)
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: tree utilities for iter-var IR (resolve/replace/walk)"
```

---

### Task 8: Port `validate_dataflow_ordering` to v2 IR

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir_v2.py`
- Test: `test/codegen/test_ir_v2.py`

- [ ] **Step 1: Append failing tests**

```python
from nkigym.codegen.ir_v2 import validate_dataflow_ordering
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad


def _mk_access(name: str, iv_id: int) -> BufferAccess:
    """Simple 1:1 access helper."""
    return BufferAccess(
        tensor_name=name,
        iter_var_ids=(iv_id,),
        pattern=(AccessRange.make({iv_id: 1}, 0, 128),),
    )


def test_validate_rejects_read_before_alloc() -> None:
    """Non-alloc block reading an un-alloc'd tensor must fail validation."""
    tensors = {
        "x": Tensor("x", ("d0",), (128,), "float32", "intermediate", "sbuf", {}),
    }
    dims = {"d0": DimInfo("d0", 128, 128, 1)}
    m = KernelIR("f", [], "x", tensors, dims, 0,
                     body=[SBlock(iter_vars=[], reads={"src": _mk_access("x", 0)},
                                  writes={}, reads_writes={}, body=[])])
    assert not validate_dataflow_ordering(m)


def test_validate_accepts_alloc_then_write_then_read() -> None:
    """Canonical order: alloc → write → read is legal."""
    tensors = {
        "x": Tensor("x", ("d0",), (128,), "float32", "intermediate", "sbuf", {}),
    }
    dims = {"d0": DimInfo("d0", 128, 128, 1)}
    alloc_call = NKIOpCall(op_cls=NKIAlloc, kwargs={"tensor_name": "x",
                                                     "location": "sbuf",
                                                     "shape": (128,),
                                                     "dtype": "float32"},
                            axis_map={}, dim_role={})
    alloc_block = SBlock(iter_vars=[], reads={}, writes={"output": _mk_access("x", 0)},
                          reads_writes={}, body=[alloc_call])
    # Simplified "writer" block (not a real NKILoad, just structural)
    writer_call = NKIOpCall(op_cls=NKILoad, kwargs={}, axis_map={}, dim_role={})
    writer = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x", 0)},
                     reads_writes={}, body=[writer_call])
    m = KernelIR("f", [], "x", tensors, dims, 0, body=[alloc_block, writer])
    assert validate_dataflow_ordering(m)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: ImportError.

- [ ] **Step 3: Implement validator**

Append to `ir_v2.py`:

```python
def validate_dataflow_ordering(module: KernelIR) -> bool:
    """Enforce 5 dataflow legality rules in pre-order DFS of the forest.

    Rules:
    1. Alloc precedes use — tensor name cannot appear in any block's
       reads/writes/reads_writes before its NKIAlloc block.
    2. Non-alloc blocks' operand names must be allocated earlier (or be
       params).
    3. Reads after writes — every read name must be written by some
       prior block (or be a param).
    4. RMW finalization — for tensor T with any RMW writer, every
       non-RMW reader of T must come after the LAST RMW write.
    5. Return produced — every tensor with origin == "return" must be
       in the written set by end of walk.
    """
    allocated: set[str] = set(module.param_names)
    written: set[str] = set(module.param_names)
    rmw_count: dict[str, int] = {}
    rmw_seen: dict[str, int] = {}

    # First pass: count RMW writes per tensor (needed for rule 4).
    def count_rmw(node: ForNode | SBlock) -> None:
        if isinstance(node, SBlock):
            for name in node.reads_writes:
                tname = node.reads_writes[name].tensor_name
                rmw_count[tname] = rmw_count.get(tname, 0) + 1
            return
        for c in node.children:
            count_rmw(c)

    for root in module.body:
        count_rmw(root)

    def walk(node: ForNode | SBlock) -> bool:
        """Recurse through ``node``; mutates allocated/written/rmw_seen."""
        if isinstance(node, SBlock):
            return _check_block(node, allocated, written, rmw_count, rmw_seen, module)
        for c in node.children:
            if not walk(c):
                return False
        return True

    for root in module.body:
        if not walk(root):
            return False

    # Rule 5: every return tensor must be written.
    for tname, tensor in module.tensors.items():
        if tensor.origin == "return" and tname not in written:
            return False
    return True


def _check_block(
    block: SBlock,
    allocated: set[str],
    written: set[str],
    rmw_count: dict[str, int],
    rmw_seen: dict[str, int],
    module: KernelIR,
) -> bool:
    """Check one SBlock against all 5 dataflow rules."""
    is_alloc = len(block.body) == 1 and block.body[0].op_cls.__name__ == "NKIAlloc"
    if is_alloc:
        call = block.body[0]
        tname = call.kwargs["tensor_name"]
        allocated.add(tname)
        written.add(tname)
        return True

    touched = (
        {a.tensor_name for a in block.reads.values()}
        | {a.tensor_name for a in block.writes.values()}
        | {a.tensor_name for a in block.reads_writes.values()}
    )
    for tname in touched:
        if tname not in allocated:
            return False

    read_set = {a.tensor_name for a in block.reads.values()} | {
        a.tensor_name for a in block.reads_writes.values()
    }
    for tname in read_set:
        if tname not in written:
            return False

    # Rule 4: non-RMW read of T after LAST RMW write.
    for name, access in block.reads.items():
        _ = name
        tname = access.tensor_name
        if rmw_count.get(tname, 0) > 0 and rmw_seen.get(tname, 0) < rmw_count[tname]:
            return False

    for access in block.writes.values():
        written.add(access.tensor_name)
    for access in block.reads_writes.values():
        written.add(access.tensor_name)
        rmw_seen[access.tensor_name] = rmw_seen.get(access.tensor_name, 0) + 1
    return True
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_ir_v2.py -x -v
```

Expected: all pass (including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir_v2.py test/codegen/test_ir_v2.py
git commit -m "refactor: port validate_dataflow_ordering to iter-var IR"
```

---

### Task 9: Port `DepCache` to v2 IR — fix RMW classifier blind spot

**Files:**
- Create: `nkigym/src/nkigym/codegen/dep_cache_v2.py`
- Test: `test/codegen/test_dep_cache_v2.py`

- [ ] **Step 1: Write failing test**

```python
"""Unit tests for v2 DepCache — classifier folds reads_writes correctly."""

import pytest
from nkigym.codegen.dep_cache_v2 import (
    DepKind,
    LeafId,
    _classify_edge,
    rebuild_scope,
)
from nkigym.codegen.ir_v2 import AccessRange, BufferAccess, SBlock, NKIOpCall
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy


def _mk_access(name: str) -> BufferAccess:
    return BufferAccess(tensor_name=name, iter_var_ids=(),
                        pattern=(AccessRange.make({}, 0, 128),))


def test_classify_edge_raw_rmw_into_reads() -> None:
    """A block that RMWs T → another block that reads T is RAW (strongest)."""
    matmul = SBlock(iter_vars=[], reads={}, writes={},
                     reads_writes={"dst": _mk_access("psum_acc")},
                     body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={},
                                      dim_role={})])
    tcopy = SBlock(iter_vars=[], reads={"src": _mk_access("psum_acc")},
                    writes={}, reads_writes={},
                    body=[NKIOpCall(op_cls=NKITensorCopy, kwargs={},
                                     axis_map={}, dim_role={})])
    assert _classify_edge(matmul, tcopy) == DepKind.RAW


def test_classify_edge_waw_memset_into_matmul_rmw() -> None:
    """memset T then matmul RMW T is WAW."""
    memset = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("psum_acc")},
                     reads_writes={},
                     body=[NKIOpCall(op_cls=NKIMemset, kwargs={"value": 0.0},
                                      axis_map={}, dim_role={})])
    matmul = SBlock(iter_vars=[], reads={}, writes={},
                     reads_writes={"dst": _mk_access("psum_acc")},
                     body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={},
                                      dim_role={})])
    """memset WRITES psum_acc; matmul's reads_writes counts as a write, too.
    writes-writes = WAW precedence (RAW beats WAW but memset doesn't *read*
    psum_acc, and matmul's reads_writes is treated as both read and write, so
    writes ∩ reads_writes → RAW. Verify that behaviour explicitly."""
    assert _classify_edge(memset, matmul) == DepKind.RAW


def test_classify_edge_disjoint_returns_none() -> None:
    """Blocks touching disjoint tensors have no edge."""
    a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")},
                reads_writes={}, body=[])
    b = SBlock(iter_vars=[], reads={"src": _mk_access("y")}, writes={},
                reads_writes={}, body=[])
    assert _classify_edge(a, b) is None


def test_rebuild_scope_populates_buffer_writers() -> None:
    """Every alloc/write block appears in buffer_writers in path order."""
    a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")},
                reads_writes={}, body=[])
    b = SBlock(iter_vars=[], reads={"src": _mk_access("x")},
                writes={"dst": _mk_access("y")}, reads_writes={}, body=[])
    scope = rebuild_scope([a, b])
    assert "x" in scope.buffer_writers
    assert "y" in scope.buffer_writers
    assert len(scope.buffer_writers["x"]) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_dep_cache_v2.py -x -v
```

Expected: `ModuleNotFoundError: nkigym.codegen.dep_cache_v2`.

- [ ] **Step 3: Implement dep_cache_v2**

```python
"""V2 DepCache keyed on SBlockId instead of LeafId.

Fixes the RMW-blind spot: ``_classify_edge`` unions ``reads_writes`` into
both the reads side and the writes side of the intersection, so RMW
operands contribute to RAW / WAR / WAW edges correctly.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md``.
"""

from dataclasses import dataclass, field
from enum import Enum

from nkigym.codegen.ir_v2 import ForNode, SBlock


class DepKind(Enum):
    """Dependency edge classification."""

    RAW = 0
    WAR = 1
    WAW = 2
    OPAQUE = 3


@dataclass(frozen=True)
class LeafId:
    """Structural identifier for an SBlock — path from forest root."""

    path: tuple[int, ...]


@dataclass(frozen=True)
class ScopeId:
    """Structural identifier for a scope root.

    Empty tuple = forest root is the scope.
    """

    path: tuple[int, ...]


@dataclass(frozen=True)
class Dependency:
    """One directed dep edge between two blocks."""

    src: LeafId
    dst: LeafId
    kind: DepKind


@dataclass
class SBlockScope:
    """Per-scope dep graph."""

    src2deps: dict[LeafId, list[Dependency]]
    dst2deps: dict[LeafId, list[Dependency]]
    buffer_writers: dict[str, list[LeafId]]
    signature: int = 0


@dataclass
class DepCache:
    """Per-scope dep cache; lazy-rebuilds on signature mismatch."""

    scopes: dict[ScopeId, SBlockScope] = field(default_factory=dict)

    def for_scope(
        self, scope_id: ScopeId, children: "list[ForNode | SBlock]"
    ) -> SBlockScope:
        """Return scope's dep graph; rebuild lazily on signature mismatch."""
        current_sig = hash(tuple(subtree_signature(c) for c in children))
        cached = self.scopes.get(scope_id)
        if cached is not None and cached.signature == current_sig:
            return cached
        fresh = rebuild_scope(children)
        self.scopes[scope_id] = fresh
        return fresh


def subtree_signature(node: "ForNode | SBlock") -> int:
    """Structural hash of a subtree.

    Folds iter-var ids, buffer access patterns, block roles, for-node
    annotations, pipeline depth. Detects all atom-driven changes so the
    DepCache rebuilds when iter vars, patterns, or tree structure change.
    """
    if isinstance(node, SBlock):
        reads_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern)
                                 for k, v in node.reads.items()))
        writes_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern)
                                  for k, v in node.writes.items()))
        rmw_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern)
                               for k, v in node.reads_writes.items()))
        body_key = tuple((c.op_cls.__name__, tuple(sorted(c.kwargs.items()))
                          if all(isinstance(v, (int, str, float, tuple))
                                 for v in c.kwargs.values()) else ())
                         for c in node.body)
        iv_key = tuple((iv.var_id, iv.dim_id, iv.extent, iv.role.value)
                       for iv in node.iter_vars)
        return hash(("block", iv_key, reads_key, writes_key, rmw_key, body_key))
    iv = node.iter_var
    return hash((
        "for",
        iv.var_id, iv.dim_id, iv.extent, iv.role.value,
        tuple(sorted(node.annotations.items()) if all(
            isinstance(v, (int, str, float, tuple)) for v in node.annotations.values()
        ) else ()),
        tuple(subtree_signature(c) for c in node.children),
    ))


def rebuild_scope(children: "list[ForNode | SBlock]") -> SBlockScope:
    """Build an SBlockScope for a scope's top-level children.

    Walks every descendant SBlock, classifies pair-wise edges, returns dep
    graph + buffer_writers index.
    """
    blocks: list[tuple[LeafId, SBlock]] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, SBlock):
            blocks.append((LeafId(path), node))
            return
        for i, c in enumerate(node.children):
            walk(c, path + (i,))

    for i, c in enumerate(children):
        walk(c, (i,))

    src2deps: dict[LeafId, list[Dependency]] = {}
    dst2deps: dict[LeafId, list[Dependency]] = {}
    buffer_writers: dict[str, list[LeafId]] = {}

    for i, (src_id, src_block) in enumerate(blocks):
        for dst_id, dst_block in blocks[i + 1 :]:
            kind = _classify_edge(src_block, dst_block)
            if kind is not None:
                dep = Dependency(src=src_id, dst=dst_id, kind=kind)
                src2deps.setdefault(src_id, []).append(dep)
                dst2deps.setdefault(dst_id, []).append(dep)
        for access in src_block.writes.values():
            buffer_writers.setdefault(access.tensor_name, []).append(src_id)
        for access in src_block.reads_writes.values():
            buffer_writers.setdefault(access.tensor_name, []).append(src_id)

    sig = hash(tuple(subtree_signature(c) for c in children))
    return SBlockScope(src2deps=src2deps, dst2deps=dst2deps,
                       buffer_writers=buffer_writers, signature=sig)


def _classify_edge(src: SBlock, dst: SBlock) -> DepKind | None:
    """Classify the strongest edge from ``src`` to ``dst``.

    Unions ``reads_writes`` into both the reads and writes sides —
    fixes v1's RMW-blind spot. Precedence: RAW > WAW > WAR.
    """
    src_reads = {a.tensor_name for a in src.reads.values()} | {
        a.tensor_name for a in src.reads_writes.values()
    }
    src_writes = {a.tensor_name for a in src.writes.values()} | {
        a.tensor_name for a in src.reads_writes.values()
    }
    dst_reads = {a.tensor_name for a in dst.reads.values()} | {
        a.tensor_name for a in dst.reads_writes.values()
    }
    dst_writes = {a.tensor_name for a in dst.writes.values()} | {
        a.tensor_name for a in dst.reads_writes.values()
    }
    if src_writes & dst_reads:
        return DepKind.RAW
    if src_writes & dst_writes:
        return DepKind.WAW
    if src_reads & dst_writes:
        return DepKind.WAR
    return None
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_dep_cache_v2.py -x -v
```

Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/dep_cache_v2.py test/codegen/test_dep_cache_v2.py
git commit -m "refactor: port DepCache to iter-var IR; fix RMW classifier"
```

---

### Task 10: Replace v1 IR with v2 — rename files, update imports

**Files:**
- Delete: `nkigym/src/nkigym/codegen/ir.py`
- Delete: `nkigym/src/nkigym/codegen/dep_cache.py`
- Rename: `ir_v2.py` → `ir.py`
- Rename: `dep_cache_v2.py` → `dep_cache.py`
- Rename: `test_ir_v2.py` → `test_ir.py`
- Rename: `test_dep_cache_v2.py` → `test_dep_cache.py`

- [ ] **Step 1: Rename test files first**

```bash
git mv test/codegen/test_ir.py test/codegen/test_ir_v1.py.bak
git mv test/codegen/test_ir_v2.py test/codegen/test_ir.py
git mv test/codegen/test_dep_cache.py test/codegen/test_dep_cache_v1.py.bak
git mv test/codegen/test_dep_cache_v2.py test/codegen/test_dep_cache.py
```

- [ ] **Step 2: Update internal imports in new test files**

Replace every `nkigym.codegen.ir_v2` with `nkigym.ir.ir` and `nkigym.codegen.dep_cache_v2` with `nkigym.ir.dep_cache` in the two new test files:

```bash
sed -i 's/nkigym.codegen.ir_v2/nkigym.ir.ir/g' test/codegen/test_ir.py
sed -i 's/nkigym.codegen.dep_cache_v2/nkigym.ir.dep_cache/g' test/codegen/test_dep_cache.py
```

- [ ] **Step 3: Rename module files + update internal cross-references**

```bash
git mv nkigym/src/nkigym/codegen/ir.py nkigym/src/nkigym/codegen/ir_v1.py.bak
git mv nkigym/src/nkigym/codegen/ir_v2.py nkigym/src/nkigym/codegen/ir.py
git mv nkigym/src/nkigym/codegen/dep_cache.py nkigym/src/nkigym/codegen/dep_cache_v1.py.bak
git mv nkigym/src/nkigym/codegen/dep_cache_v2.py nkigym/src/nkigym/codegen/dep_cache.py

# Inside the new ir.py, update the "reuse from ir" import:
sed -i 's|from nkigym.ir.ir import DepCache|# from nkigym.ir.dep_cache import DepCache (below)|' nkigym/src/nkigym/codegen/ir.py
# Instead, DepCache import moves to dep_cache.py; fix ir.py
python3 -c "
import re
p = 'nkigym/src/nkigym/codegen/ir.py'
s = open(p).read()
s = s.replace(
    'from nkigym.ir.ir import DepCache, DimInfo, Tensor',
    'from nkigym.ir.dep_cache import DepCache'
)
open(p, 'w').write(s)
"
```

This will fail because `DimInfo` and `Tensor` came from the old ir.py. Inline them into the new ir.py:

- [ ] **Step 4: Inline `Tensor` and `DimInfo` into ir.py**

Read the old ir_v1.py.bak and copy `Tensor` and `DimInfo` dataclasses into ir.py. Near the top of ir.py, after the existing imports, insert:

```python
from typing import Literal


@dataclass(frozen=True)
class DimInfo:
    """Full extent + per-tile extent for one concrete dim."""

    dim_id: str
    total_size: int
    tile_size: int
    num_tiles: int


@dataclass
class Tensor:
    """Tensor identity: shape, dtype, location, origin, multi-buffer degree.

    Buffer / Allocate separation (TVM style): this is the tensor's
    identity; NKIAlloc is the allocation scope.
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: Literal["param", "intermediate", "return"]
    location: Literal["hbm", "sbuf", "psum"]
    buffer_degree: dict[str, int] = field(default_factory=dict)
```

Remove the old `DepCache` reuse import from the new ir.py (it's in dep_cache.py; ir.py just references it via forward reference).

- [ ] **Step 5: Update KernelIR's DepCache field to use dep_cache module**

Near the top of ir.py:

```python
from nkigym.ir.dep_cache import DepCache
```

And update the `dep` field default:

```python
dep: DepCache = field(default_factory=DepCache)
```

- [ ] **Step 6: Run tests**

```bash
pytest test/codegen/test_ir.py test/codegen/test_dep_cache.py -x -v
```

Expected: all pass.

- [ ] **Step 7: Verify the v1 backup files aren't imported anywhere**

```bash
grep -rn "ir_v1\|dep_cache_v1\|ir_v2\|dep_cache_v2" nkigym/src/ test/ 2>/dev/null | grep -v ".bak" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 8: Commit (backup .bak files stay for reference during Phase B)**

```bash
git add -A
git commit -m "refactor: replace v1 IR + DepCache with iter-var versions"
```

---

## Phase B — Canonical + Render Parity (Week 2)

### Task 11: Port canonical builder — allocate iter vars per block

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py` (large rewrite)
- Test: `test/codegen/test_canonical.py` (adapt existing tests)

This is the largest single task. The canonical builder has to:

1. Parse `f_nkigym` AST (reuse existing AST logic — unchanged).
2. Build `Tensor` map from `NKIAlloc` records (unchanged).
3. Derive `DimInfo` from op `TILE_LIMITS` (unchanged).
4. Per op: allocate one fresh `IterVar` per touched dim; build `BufferAccess` per operand slot; wrap in `SBlock`; nest under one `ForNode` per iter var in canonical order.

- [ ] **Step 1: Read existing canonical.py**

```bash
wc -l nkigym/src/nkigym/codegen/canonical.py
```

Expected: ~586 lines.

- [ ] **Step 2: Identify AST-parse logic to preserve (unchanged)**

Functions to keep verbatim:
- `_parse_ast`
- `_try_parse_alloc`
- `_build_tensor_map` (minor: origin strings stay; field signature matches new Tensor class)
- `_resolve_dim_ids`, `_build_dim_info`, `_touched_dims`, `_resolve_dim_role`

Functions to rewrite:
- `_make_leaf` → `_make_sblock` (builds SBlock with iter vars + BufferAccess)
- `_make_alloc_leaf` → `_make_alloc_sblock` (alloc block, no iter vars)
- `build_initial_ir` — top-level orchestrator

- [ ] **Step 3: Write failing test for single-op canonical build**

Add to `test/codegen/test_canonical.py`:

```python
def test_canonical_emits_sblock_per_op() -> None:
    """Canonical form: one SBlock per NKIOp call; each block has per-op iter vars."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import SBlock, ForNode
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": {"shape": (512, 256), "dtype": "bfloat16"},
        "rhs": {"shape": (512, 1024), "dtype": "bfloat16"},
    }
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs)
    blocks = []
    def collect(n):
        if isinstance(n, SBlock):
            blocks.append(n)
        else:
            for c in n.children:
                collect(c)
    for root in module.body:
        collect(root)
    assert len(blocks) == 11  # 5 allocs + 6 compute: load, load, memset, matmul, copy, store


def test_canonical_iter_vars_per_block_distinct() -> None:
    """Two blocks iterating d0 start with distinct iter var ids."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.ir.ir import SBlock
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": {"shape": (512, 256), "dtype": "bfloat16"},
        "rhs": {"shape": (512, 1024), "dtype": "bfloat16"},
    }
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs)
    # Collect all SBlocks and their d0 iter vars
    d0_ids = set()
    def collect(n):
        if isinstance(n, SBlock):
            for iv in n.iter_vars:
                if iv.dim_id == "d0":
                    d0_ids.add(iv.var_id)
        else:
            for c in n.children:
                collect(c)
    for root in module.body:
        collect(root)
    # Two loads + matmul all touch d0 → at least 3 distinct ids expected
    assert len(d0_ids) >= 3
```

- [ ] **Step 4: Run to verify failure**

```bash
pytest test/codegen/test_canonical.py::test_canonical_emits_sblock_per_op -x -v
```

Expected: either import-error or assertion-error depending on v2 canonical.py state.

- [ ] **Step 5: Rewrite `_make_sblock`**

Replace `_make_leaf` function (around line 488-517 in current canonical.py) with:

```python
def _make_sblock(op: _ParsedOp, module: KernelIR) -> SBlock:
    """Build an SBlock for one parsed op.

    - Allocates fresh IterVars on ``module`` for each dim in
      ``op.touched_dims``.
    - Builds BufferAccess per operand slot using ``op.op_cls.OPERAND_AXES``
      and ``op.axis_map``.
    - Splits operands into reads / writes / reads_writes buckets.
    """
    iter_vars: list[IterVar] = []
    dim_to_iv: dict[str, IterVar] = {}
    for dim_id in op.touched_dims:
        role = op.dim_role[dim_id]
        extent = module.dims[dim_id].num_tiles
        iv = module.allocate_iter_var(dim_id, extent, role)
        iter_vars.append(iv)
        dim_to_iv[dim_id] = iv

    reads: dict[str, BufferAccess] = {}
    writes: dict[str, BufferAccess] = {}
    rmw: dict[str, BufferAccess] = {}

    for slot, abstract_axes in op.op_cls.OPERAND_AXES.items():
        tname = op.operand_names.get(slot)
        if tname is None:
            continue
        iv_ids: list[int] = []
        pattern_entries: list[AccessRange] = []
        for abstract in abstract_axes:
            dim_id = op.axis_map.get(abstract)
            if dim_id is None:
                continue
            iv = dim_to_iv.get(dim_id)
            if iv is None:
                continue
            iv_ids.append(iv.var_id)
            tile_size = module.dims[dim_id].tile_size
            pattern_entries.append(
                AccessRange.make({iv.var_id: 1}, 0, tile_size)
            )
        access = BufferAccess(
            tensor_name=tname,
            iter_var_ids=tuple(iv_ids),
            pattern=tuple(pattern_entries),
        )
        if slot in op.op_cls.RMW_OPERANDS:
            rmw[slot] = access
        elif slot in op.op_cls.INPUT_OPERANDS:
            reads[slot] = access
        else:
            writes[slot] = access

    call = NKIOpCall(
        op_cls=op.op_cls,
        kwargs=dict(op.op_kwargs),
        axis_map=dict(op.axis_map),
        dim_role=dict(op.dim_role),
    )
    return SBlock(
        iter_vars=iter_vars,
        reads=reads,
        writes=writes,
        reads_writes=rmw,
        body=[call],
    )
```

- [ ] **Step 6: Rewrite `_make_alloc_sblock`**

Replace `_make_alloc_leaf` with:

```python
def _make_alloc_sblock(alloc: _AllocRecord) -> SBlock:
    """Build the single-NKIOpCall SBlock for an NKIAlloc declaration.

    Alloc blocks have no iter vars; they're root-level markers for
    tensor storage allocation. The NKIOpCall kwargs carry location,
    shape, dtype, tensor_name.
    """
    from nkigym.ops.alloc import NKIAlloc

    call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={
            "tensor_name": alloc.name,
            "location": alloc.location,
            "shape": alloc.shape,
            "dtype": alloc.dtype,
        },
        axis_map={},
        dim_role={},
    )
    # Alloc writes its own tensor — single BufferAccess with empty iter_var_ids
    output_access = BufferAccess(
        tensor_name=alloc.name,
        iter_var_ids=(),
        pattern=(),
    )
    return SBlock(
        iter_vars=[],
        reads={},
        writes={"output": output_access},
        reads_writes={},
        body=[call],
    )
```

- [ ] **Step 7: Rewrite `build_initial_ir`**

Replace the top-level function:

```python
def build_initial_ir(
    f: Callable, input_specs: dict[str, dict]
) -> KernelIR:
    """Parse ``f``'s AST and build a canonical iter-var-based KernelIR.

    Steps:
    1. AST-parse to extract op records + alloc records + return name.
    2. Build tensor map, dim info.
    3. Create empty module with iter_var_counter=0.
    4. For each alloc: emit root-level alloc SBlock.
    5. For each op: emit SBlock wrapped in ForNodes (one per iter var).
    6. Append wrapped subtrees to module.body.
    """
    func = f if not getattr(f, "__nkigym_kernel__", False) else f.__wrapped__
    raws, allocs, return_name = _parse_ast(func)

    if return_name is None:
        raise ValueError("build_initial_ir: no return statement")

    param_names = list(input_specs.keys())
    tensors = _build_tensor_map(param_names, input_specs, allocs, return_name)
    dim_sizes = _resolve_dim_ids(raws, tensors)
    per_op_axis_maps = _resolve_per_op_axis_maps(raws, tensors)
    dims = _build_dim_info(raws, per_op_axis_maps, dim_sizes)
    parsed = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)

    module = KernelIR(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        iter_var_counter=0,
        body=[],
    )

    for alloc in allocs:
        module.body.append(_make_alloc_sblock(alloc))

    for op in parsed:
        block = _make_sblock(op, module)
        tree = _wrap_block_in_fornodes(block)
        module.body.append(tree)

    return module


def _wrap_block_in_fornodes(block: SBlock) -> ForNode | SBlock:
    """Wrap ``block`` in one ForNode per iter var, in canonical order.

    Outermost-first. Returns the block unwrapped if it has no iter vars
    (e.g. alloc blocks).
    """
    if not block.iter_vars:
        return block
    node: ForNode | SBlock = block
    for iv in reversed(block.iter_vars):
        node = ForNode(iter_var=iv, children=[node])
    return node
```

- [ ] **Step 8: Fix imports in canonical.py top**

```python
from nkigym.ir.ir import (
    AccessRange,
    BufferAccess,
    DimInfo,
    ForNode,
    IterVar,
    KernelIR,
    NKIOpCall,
    SBlock,
    Tensor,
)
```

Remove old imports of `BodyLeaf`, `LoopNode`, `TreeIR`.

- [ ] **Step 9: Run canonical tests**

```bash
pytest test/codegen/test_canonical.py -x -v
```

Expected: new tests pass. Some pre-existing tests in `test_canonical.py` may fail — these expected `BodyLeaf` / `LoopNode` fields and need updating in Task 12.

Document the failures in the commit message.

- [ ] **Step 10: Commit**

```bash
git add nkigym/src/nkigym/codegen/canonical.py test/codegen/test_canonical.py
git commit -m "refactor: port canonical builder to iter-var IR (pre-existing canonical tests may need updates in Task 12)"
```

---

### Task 12: Update pre-existing canonical tests to new IR

**Files:**
- Modify: `test/codegen/test_canonical.py`, `test/codegen/test_axis_role.py`, `test/codegen/test_ir.py` (extend), `test/codegen/test_first_class_buffers.py`

- [ ] **Step 1: Run all canonical-related tests to identify failures**

```bash
pytest test/codegen/test_canonical.py test/codegen/test_axis_role.py test/codegen/test_first_class_buffers.py -x --no-header 2>&1 | tail -50
```

Expected: specific failures listing references to `BodyLeaf`, `LoopNode`, `reads.values()` (old format was `dict[str, str]`, new is `dict[str, BufferAccess]`).

- [ ] **Step 2: Search for old-type references in test files**

```bash
grep -rn "BodyLeaf\|LoopNode\|\.reads\.values\(\)" test/ 2>/dev/null | head -40
```

Document every file + line.

- [ ] **Step 3: Per-file fixup (manual)**

For each test file in the grep output, update references:
- `BodyLeaf` → `SBlock`
- `LoopNode` → `ForNode`
- `leaf.reads.values()` → `[a.tensor_name for a in block.reads.values()]`
- `leaf.writes` → `[a.tensor_name for a in block.writes.values()]` (tuple → list of names)
- `leaf.axis_map` → `block.body[0].axis_map` (axis_map moved to NKIOpCall)
- `leaf.dim_role` → `block.body[0].dim_role`
- `loop.trip_count` → `for_node.iter_var.extent`
- `loop.role` → `for_node.iter_var.role`
- `loop.dim_id` → `for_node.iter_var.dim_id`
- `loop.reduce_op` → **removed** (was stored on LoopNode; not carried forward — reduce semantics live in iter var role + tensor operation)

For tests that checked `.name` on LoopNode: `for_node.name` still works (kept).

- [ ] **Step 4: Run tests per-file**

Iterate until each test file passes:

```bash
pytest test/codegen/test_canonical.py -x -v
pytest test/codegen/test_axis_role.py -x -v
pytest test/codegen/test_first_class_buffers.py -x -v
```

- [ ] **Step 5: Commit**

```bash
git add test/
git commit -m "test: update canonical / axis-role / first-class-buffers tests for iter-var IR"
```

---

### Task 13: Port `place_buffers` to iter-var-aware LCA walk (N-D)

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/place_buffers.py`
- Test: `test/codegen/test_place_buffers.py`

- [ ] **Step 1: Write failing tests for N-D buffer shape**

Add to `test/codegen/test_place_buffers.py`:

```python
def test_place_buffers_emits_trivial_dims_explicitly() -> None:
    """Q5 decision: trivial dims (num_slots=1) stay explicit in buffer shape."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.place_buffers import place_buffers
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
        "rhs": {"shape": (128, 512), "dtype": "bfloat16"},
    }
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs)
    place_buffers(module)
    # In this config: K=128, M=128, N=512 → all single-tile
    # lhs_T_sbuf shape = (P_tile, num_P_tiles=1, F_tile*num_F=128) = (128, 1, 128)
    lhs_T_sbuf = module.tensors["lhs_T_sbuf"]
    assert lhs_T_sbuf.shape == (128, 1, 128)


def test_place_buffers_nd_sbuf_shape_for_multi_tile() -> None:
    """Multi-tile config: shape = (P_tile, num_P_tiles, F_tile * num_F_tiles)."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.place_buffers import place_buffers
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs)
    place_buffers(module)
    # K=2048, tile=128 → num_K_tiles=16; M=2048, tile=128 → num_M_tiles=16
    # lhs_T_sbuf has dims (K, M); canonical SBUF shape = (P, num_K_tiles, num_M_tiles * M_tile)
    lhs_T_sbuf = module.tensors["lhs_T_sbuf"]
    assert lhs_T_sbuf.shape == (128, 16, 2048)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_place_buffers.py -x -v
```

Expected: either import/reference errors or shape mismatch.

- [ ] **Step 3: Rewrite place_buffers for iter-var IR**

Replace `place_buffers.py` content:

```python
"""Buffer placement: compute tensor SBUF/PSUM shapes from iter-var LCA walk.

Per-tensor shape derivation:
- Walk the tree; for each SBlock referencing this tensor, collect enclosing
  ForNode iter vars (by dim_id).
- LCA(tensor) = deepest ForNode in tree that encloses all accesses to tensor.
- required_tiles(tensor, dim) = num_tiles(dim) / product_of_trips_above_LCA.
- total_slots(tensor, dim) = required_tiles * buffer_degree.
- Shape: (P_tile, *[total_slots[d] for d in non_partition_dims], F_tile * num_F_tiles).

N-D unified path: trivial dims (total_slots == 1) stay explicit.
"""

from nkigym.ir.ir import (
    ForNode,
    IterVar,
    KernelIR,
    SBlock,
    Tensor,
    blocks_under,
)


def place_buffers(module: KernelIR) -> None:
    """Mutates ``module.tensors`` to set each intermediate tensor's final shape.

    Param / return HBM tensors are untouched — their shapes come from
    input_specs. Intermediate SBUF / PSUM tensors get their shape
    recomputed from LCA.
    """
    for name, tensor in module.tensors.items():
        if tensor.origin == "param":
            continue
        if tensor.location == "hbm":
            continue
        _place_one_tensor(module, tensor)


def _place_one_tensor(module: KernelIR, tensor: Tensor) -> None:
    """Derive N-D SBUF/PSUM shape for one tensor."""
    accesses = _find_accesses_to_tensor(module, tensor.name)
    if not accesses:
        return  # unused tensor; leave shape as declared

    required_tiles = _required_tiles_per_dim(module, tensor.name, accesses)

    # Determine partition (P) dim and free (F) dim from tensor.dim_ids.
    # Convention: first dim = P axis, last dim = F axis.
    if len(tensor.dim_ids) < 2:
        return  # 1-D tensor handling: leave as declared

    p_dim = tensor.dim_ids[0]
    f_dim = tensor.dim_ids[-1]
    middle_dims = tensor.dim_ids[1:-1]  # empty for 2D tensors

    p_tile = module.dims[p_dim].tile_size
    f_tile = module.dims[f_dim].tile_size
    num_f_tiles = required_tiles.get(f_dim, 1) * tensor.buffer_degree.get(f_dim, 1)

    shape_parts = [p_tile]
    # Middle slot dims (between P and F)
    for d in middle_dims:
        slots = required_tiles.get(d, 1) * tensor.buffer_degree.get(d, 1)
        shape_parts.append(slots)
    # P-slot dim (num_P_tiles)
    num_p_slots = required_tiles.get(p_dim, 1) * tensor.buffer_degree.get(p_dim, 1)
    # Canonical SBUF shape: (P_tile, num_P_slots, ..., F)
    # Reorder: partition tile first, then slot dims (including P-slots), then F
    shape_parts = [p_tile, num_p_slots] + [
        required_tiles.get(d, 1) * tensor.buffer_degree.get(d, 1) for d in middle_dims
    ] + [f_tile * num_f_tiles]

    tensor.shape = tuple(shape_parts)


def _find_accesses_to_tensor(
    module: KernelIR, tensor_name: str
) -> list[tuple[SBlock, tuple[IterVar, ...]]]:
    """Collect (block, enclosing_iter_vars) for every block that reads or
    writes the named tensor.

    Enclosing_iter_vars = ancestor ForNode iter vars of block, forest-root-down.
    """
    results: list[tuple[SBlock, tuple[IterVar, ...]]] = []

    def walk(node: ForNode | SBlock, ancestors: tuple[IterVar, ...]) -> None:
        if isinstance(node, SBlock):
            touched = (
                {a.tensor_name for a in node.reads.values()}
                | {a.tensor_name for a in node.writes.values()}
                | {a.tensor_name for a in node.reads_writes.values()}
            )
            if tensor_name in touched:
                results.append((node, ancestors))
            return
        new_ancestors = ancestors + (node.iter_var,)
        for c in node.children:
            walk(c, new_ancestors)

    for root in module.body:
        walk(root, ())
    return results


def _required_tiles_per_dim(
    module: KernelIR,
    tensor_name: str,
    accesses: list[tuple[SBlock, tuple[IterVar, ...]]],
) -> dict[str, int]:
    """For each tensor dim, compute required tile count based on LCA depth.

    required_tiles[d] = num_tiles[d] / gcd_across_accesses(coverage_at_LCA[d]).
    Simplification: take the MIN coverage across accesses — each access
    needs enough slots to index its own LCA-relative range.
    """
    # Compute per-access coverage: product of enclosing ForNode extents per dim.
    coverages: dict[str, list[int]] = {}
    for block, ancestors in accesses:
        cover: dict[str, int] = {}
        for iv in ancestors:
            cover[iv.dim_id] = cover.get(iv.dim_id, 1) * iv.extent
        for d, c in cover.items():
            coverages.setdefault(d, []).append(c)

    result: dict[str, int] = {}
    for dim_id, dim in module.dims.items():
        if dim_id not in coverages:
            result[dim_id] = dim.num_tiles  # cross-nest tensor: full extent
            continue
        min_cov = min(coverages[dim_id])
        if min_cov > 0:
            result[dim_id] = max(1, dim.num_tiles // min_cov)
        else:
            result[dim_id] = dim.num_tiles
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_place_buffers.py -x -v
```

Expected: new tests pass; old tests may fail if they expected old shape computations. Document and update in Task 14 if needed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/lowering/place_buffers.py test/codegen/test_place_buffers.py
git commit -m "refactor: N-D buffer placement via iter-var LCA walk"
```

---

### Task 14: Port emit_ops + emit_source — render SBlock bodies

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/emit_ops.py` (large — ~11 per-op_cls emitters)
- Modify: `nkigym/src/nkigym/codegen/lowering/emit_source.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/_emit_utils.py`
- Test: `test/codegen/test_canonical.py` (add render parity tests)

**Note:** Given the size (~11 emitters × method signatures change), break this into sub-tasks — one per op_cls emitter. To keep the plan readable, outline the general process and show one emitter in detail; the pattern repeats.

- [ ] **Step 1: Write failing "end-to-end canonical render" tests**

Add to `test/codegen/test_canonical.py`:

```python
def test_canonical_render_matmul_lhsT_rhs_produces_valid_python() -> None:
    """Canonical render must parse as valid Python and contain expected ISA calls."""
    import ast
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.render import render
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs)
    source = render(module)
    # Parse as Python
    ast.parse(source)
    # Contains expected ISA calls
    assert "nisa.dma_copy" in source
    assert "nisa.nc_matmul" in source
    assert "nisa.memset" in source
    assert "nl.ndarray" in source


def test_canonical_render_cpu_sim_matches_numpy() -> None:
    """Canonical-render output CPU-sims equal to numpy reference."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.render import render
    from nkigym.tune.verify import _draw_fp32_inputs, _verify
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym

    input_specs = {
        "lhs_T": ((2048, 2048), "bfloat16"),
        "rhs": ((2048, 2048), "bfloat16"),
    }
    canon_specs = {k: {"shape": s, "dtype": d} for k, (s, d) in input_specs.items()}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, canon_specs)
    source = render(module)
    _verify(source, matmul_lhsT_rhs_nkigym, input_specs)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/codegen/test_canonical.py::test_canonical_render_matmul_lhsT_rhs_produces_valid_python -x -v
```

Expected: renderer fails because it still expects LoopNode/BodyLeaf.

- [ ] **Step 3: Rewrite `_emit_utils.py`**

Key helper: slice expression from `BufferAccess.pattern`.

```python
"""Rendering helpers for the iter-var IR.

Slice emission: consumes ``BufferAccess.pattern`` to produce an affine
indexing expression like ``buf[0:128, (i_d0_0 * 4 + i_d0_1), ...]``.
Trivial dims (extent=1) stay explicit.
"""

from nkigym.ir.ir import AccessRange, BufferAccess, IterVar


def emit_slice(
    buf_name: str,
    access: BufferAccess,
    iter_var_to_name: dict[int, str],
) -> str:
    """Emit ``buf_name[slice_expr_per_dim]`` for the given access pattern.

    Args:
        buf_name: Rendered buffer variable name.
        access: BufferAccess with pattern per tensor dim.
        iter_var_to_name: iter_var_id → rendered name (e.g. "i_d0_0").

    Returns:
        Python expression string.
    """
    slice_parts: list[str] = []
    for i, ar in enumerate(access.pattern):
        start = _emit_affine_start(ar, iter_var_to_name)
        end = f"{start} + {ar.extent}"
        slice_parts.append(f"{start} : {end}")
    return f"{buf_name}[{', '.join(slice_parts)}]"


def _emit_affine_start(
    ar: AccessRange, iter_var_to_name: dict[int, str]
) -> str:
    """Emit the starting index of an AccessRange as a Python expression.

    Form: sum(coeff * name) + const_offset. Canonical ordering: positive
    coeffs first, then const. Collapses trivially to "0" when no coeffs
    and offset=0.
    """
    terms: list[str] = []
    for iv_id, coeff in sorted(ar.iter_var_coeffs):
        name = iter_var_to_name[iv_id]
        if coeff == 1:
            terms.append(name)
        else:
            terms.append(f"({name} * {coeff})")
    if ar.const_offset != 0:
        terms.append(str(ar.const_offset))
    if not terms:
        return "0"
    return " + ".join(terms)
```

- [ ] **Step 4: Rewrite per-op_cls emitters**

Replace `emit_ops.py`. Structure:

```python
"""Per-op_cls ISA call emitters for the iter-var IR.

Each emitter takes an NKIOpCall plus its enclosing SBlock (for operand
buffer access maps) plus an EmitCtx (enclosing iter var name map).
Returns one source line.
"""

from dataclasses import dataclass

from nkigym.ir.ir import BufferAccess, IterVar, NKIOpCall, SBlock
from nkigym.codegen._emit_utils import emit_slice


@dataclass
class EmitCtx:
    """Per-emission context: current iter var name mapping."""

    iter_var_to_name: dict[int, str]


def emit_op_call(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Dispatch to the per-op_cls emitter by class name."""
    emitter = _EMITTERS.get(call.op_cls.__name__)
    if emitter is None:
        raise NotImplementedError(f"No emitter for op class {call.op_cls.__name__}")
    return emitter(call, block, ctx)


def _emit_NKIAlloc(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nl.ndarray declaration for the alloc block."""
    tensor_name = call.kwargs["tensor_name"]
    shape = call.kwargs["shape"]
    dtype = call.kwargs["dtype"]
    location = call.kwargs["location"]
    # Dtype translation
    _dt_map = {"float32": "nl.float32", "float16": "nl.float16",
                "bfloat16": "nl.bfloat16"}
    dt_expr = _dt_map.get(dtype, f"nl.{dtype}")
    _loc_map = {"hbm": "nl.shared_hbm", "sbuf": "nl.sbuf", "psum": "nl.psum"}
    return f"{tensor_name} = nl.ndarray({shape!r}, dtype={dt_expr}, buffer={_loc_map[location]})"


def _emit_NKILoad(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nisa.dma_copy HBM → SBUF."""
    src_access = block.reads["src"]
    dst_access = block.writes["dst"]
    src_expr = emit_slice(src_access.tensor_name, src_access, ctx.iter_var_to_name)
    dst_expr = emit_slice(dst_access.tensor_name, dst_access, ctx.iter_var_to_name)
    return f"nisa.dma_copy(dst={dst_expr}, src={src_expr})"


def _emit_NKIStore(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nisa.dma_copy SBUF → HBM."""
    src_access = block.reads["src"]
    dst_access = block.writes["dst"]
    src_expr = emit_slice(src_access.tensor_name, src_access, ctx.iter_var_to_name)
    dst_expr = emit_slice(dst_access.tensor_name, dst_access, ctx.iter_var_to_name)
    return f"nisa.dma_copy(dst={dst_expr}, src={src_expr})"


def _emit_NKIMemset(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nisa.memset."""
    dst_access = block.writes["dst"]
    dst_expr = emit_slice(dst_access.tensor_name, dst_access, ctx.iter_var_to_name)
    value = call.kwargs.get("value", 0.0)
    return f"nisa.memset({dst_expr}, value={value})"


def _emit_NKIMatmul(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nisa.nc_matmul."""
    stationary = block.reads["stationary"]
    moving = block.reads["moving"]
    dst = block.reads_writes["dst"]
    s_expr = emit_slice(stationary.tensor_name, stationary, ctx.iter_var_to_name)
    m_expr = emit_slice(moving.tensor_name, moving, ctx.iter_var_to_name)
    d_expr = emit_slice(dst.tensor_name, dst, ctx.iter_var_to_name)
    return (f"nisa.nc_matmul(dst={d_expr}, stationary={s_expr}, moving={m_expr})")


def _emit_NKITensorCopy(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str:
    """Emit nisa.tensor_copy."""
    src = block.reads["src"]
    dst = block.writes["dst"]
    s_expr = emit_slice(src.tensor_name, src, ctx.iter_var_to_name)
    d_expr = emit_slice(dst.tensor_name, dst, ctx.iter_var_to_name)
    return f"nisa.tensor_copy({d_expr}, {s_expr})"


# Additional emitters for NKIActivation, NKIActivationReduce, NKITensorScalar,
# NKITensorReduce, NKIDMATranspose, NKITranspose follow the same pattern.
# Each one reads its specific slot names from block.reads / writes /
# reads_writes; emits via emit_slice; returns one source line.


_EMITTERS = {
    "NKIAlloc": _emit_NKIAlloc,
    "NKILoad": _emit_NKILoad,
    "NKIStore": _emit_NKIStore,
    "NKIMemset": _emit_NKIMemset,
    "NKIMatmul": _emit_NKIMatmul,
    "NKITensorCopy": _emit_NKITensorCopy,
    # Add remaining as implemented
}
```

**Important**: Before running tests, port the remaining 5 emitters (`NKIActivation`, `NKIActivationReduce`, `NKITensorScalar`, `NKITensorReduce`, `NKIDMATranspose`, `NKITranspose`). Pattern: inspect each class's `INPUT_OPERANDS` / `RMW_OPERANDS`; consume the corresponding slots off `block.reads` / `block.writes` / `block.reads_writes`; construct the `nisa.*` call with kwargs carried in `call.kwargs`.

- [ ] **Step 5: Rewrite emit_source.py**

```python
"""Forest walker: open/close for-loop headers; delegate SBlock bodies.

Produces the final NKI source string with full imports + @nki.jit
wrapper + function body.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock
from nkigym.codegen.emit_ops import EmitCtx, emit_op_call


HEADER = """import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def {func_name}({param_list}):
"""


def render(module: KernelIR) -> str:
    """Render ``module`` as NKI source.

    Pipeline steps assumed already run (place_buffers, annotate injection).
    Emits:
    1. Imports + @nki.jit decorator + function signature.
    2. Alloc-block declarations at indent 1.
    3. For-loop nests + SBlock bodies at indent 1+.
    4. ``return <return_name>`` at indent 1.
    """
    lines: list[str] = []
    param_list = ", ".join(module.param_names)
    lines.append(HEADER.format(func_name=module.func_name, param_list=param_list).rstrip("\n"))
    ctx = EmitCtx(iter_var_to_name={})
    for root in module.body:
        _emit_node(root, ctx, indent=1, lines=lines)
    lines.append(f"    return {module.return_name}")
    return "\n".join(lines) + "\n"


def _emit_node(
    node: ForNode | SBlock,
    ctx: EmitCtx,
    indent: int,
    lines: list[str],
) -> None:
    """Recursively emit ``node`` at ``indent``; mutates ``lines`` + ``ctx``."""
    pad = "    " * indent
    if isinstance(node, ForNode):
        iv = node.iter_var
        name = node.name or f"i_{iv.dim_id}_{iv.var_id}"
        ctx.iter_var_to_name[iv.var_id] = name
        lines.append(f"{pad}for {name} in range({iv.extent}):")
        for child in node.children:
            _emit_node(child, ctx, indent + 1, lines)
        # Leave iter_var_to_name entry in place (canonical renaming reuses)
    else:
        # SBlock: emit each NKIOpCall in order
        for call in node.body:
            line = emit_op_call(call, node, ctx)
            lines.append(f"{pad}{line}")
```

- [ ] **Step 6: Rewrite render.py**

```python
"""Top-level render entry: orchestrates passes then emits source."""

from nkigym.ir.ir import KernelIR
from nkigym.codegen.emit_source import render as _emit_source
from nkigym.codegen.place_buffers import place_buffers
from nkigym.codegen.canonicalize_names import canonicalize_iter_var_names


def render(module: KernelIR) -> str:
    """Run passes and emit NKI source.

    Passes: place_buffers → inject_annotations → canonicalize_names → emit_source.
    """
    place_buffers(module)
    # inject_annotations lands in Task 17
    canonicalize_iter_var_names(module)
    return _emit_source(module)
```

- [ ] **Step 7: Write canonicalize_iter_var_names**

Create `nkigym/src/nkigym/codegen/lowering/canonicalize_names.py`:

```python
"""Canonical iter-var naming: i_<dim>_<ordinal>.

Per-tree-position ordinals — two sibling ForNodes on the same dim get
distinct ordinals; nested ForNodes on the same dim (e.g. Split pair)
get increasing ordinals top-down.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def canonicalize_iter_var_names(module: KernelIR) -> None:
    """Assign ``ForNode.name = "i_<dim>_<ordinal>"`` across the tree."""
    for root in module.body:
        _walk(root, counts={})


def _walk(node: ForNode | SBlock, counts: dict[str, int]) -> None:
    if isinstance(node, SBlock):
        return
    dim = node.iter_var.dim_id
    k = counts.get(dim, 0)
    node.name = f"i_{dim}_{k}"
    counts[dim] = k + 1
    for child in node.children:
        _walk(child, counts)
    counts[dim] = k
```

- [ ] **Step 8: Run tests**

```bash
pytest test/codegen/test_canonical.py -x -v
```

Expected: both new tests (valid Python, CPU-sim match) pass.

- [ ] **Step 9: Commit**

```bash
git add nkigym/src/nkigym/codegen/lowering/ nkigym/src/nkigym/codegen/render.py test/codegen/test_canonical.py
git commit -m "refactor: port emit_ops + emit_source + canonicalize to iter-var IR"
```

---

### Task 15: Delete inject_multi_buffer + inject_software_pipeline scaffolding

**Files:**
- Delete: `nkigym/src/nkigym/codegen/lowering/inject_multi_buffer.py`
- Delete: `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py`
- Create: `nkigym/src/nkigym/codegen/lowering/inject_annotations/__init__.py` (stub — sub-passes land in Task 17)

- [ ] **Step 1: Delete + create stub**

```bash
git rm nkigym/src/nkigym/codegen/lowering/inject_multi_buffer.py
git rm nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py
mkdir -p nkigym/src/nkigym/codegen/lowering/inject_annotations
cat > nkigym/src/nkigym/codegen/lowering/inject_annotations/__init__.py <<'EOF'
"""Dispatcher for annotation-keyed lowering passes.

Walks the module tree; for each ForNode/SBlock with annotations, invokes
per-key sub-passes. Sub-passes land in followup tasks.
"""

from nkigym.ir.ir import KernelIR


def inject_annotations(module: KernelIR) -> None:
    """Dispatch per-key annotation passes. No-op until sub-passes land."""
    _ = module
EOF
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove old multi_buffer/software_pipeline injection; stub annotations package"
```

---

### Task 16: Render parity sweep on three example kernels

**Files:**
- Test: `test/codegen/test_canonical.py`

- [ ] **Step 1: Add canonical-render-and-CPU-sim tests for all three example kernels**

```python
def test_canonical_render_matmul_lhs_rhs_cpu_sim() -> None:
    from examples.matmul_lhs_rhs import matmul_lhs_rhs_nkigym
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.render import render
    from nkigym.tune.verify import _verify
    input_specs = {"lhs": ((2048, 2048), "bfloat16"),
                    "rhs": ((2048, 2048), "bfloat16")}
    canon_specs = {k: {"shape": s, "dtype": d} for k, (s, d) in input_specs.items()}
    module = build_initial_ir(matmul_lhs_rhs_nkigym, canon_specs)
    source = render(module)
    _verify(source, matmul_lhs_rhs_nkigym, input_specs)


def test_canonical_render_rmsnorm_matmul_cpu_sim() -> None:
    from examples.rmsnorm_matmul import rmsnorm_matmul_nkigym
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.render import render
    from nkigym.tune.verify import _verify
    input_specs = {
        "x": ((2048, 2048), "bfloat16"),
        "gamma": ((2048,), "bfloat16"),
        "weight": ((2048, 2048), "bfloat16"),
    }
    canon_specs = {k: {"shape": s, "dtype": d} for k, (s, d) in input_specs.items()}
    module = build_initial_ir(rmsnorm_matmul_nkigym, canon_specs)
    source = render(module)
    _verify(source, rmsnorm_matmul_nkigym, input_specs)
```

- [ ] **Step 2: Run tests**

```bash
pytest test/codegen/test_canonical.py -x -v
```

Expected: all three example kernels render + CPU-sim.

- [ ] **Step 3: Fix any failures**

Common failure modes:
- Missing emitter for a specific NKIOp (port per Task 14 pattern).
- Operand slot name mismatch (inspect op class's `INPUT_OPERANDS` / `RMW_OPERANDS` and adjust emitter).
- Access pattern mismatch on non-matmul ops (activation_reduce, tensor_reduce — these have reduce semantics; inspect their OPERAND_AXES).

- [ ] **Step 4: Commit**

```bash
git add test/codegen/test_canonical.py nkigym/src/nkigym/codegen/lowering/emit_ops.py
git commit -m "test: render parity on all three example kernels; fix per-op emitters"
```

---

## Phase C — Atoms (Week 3)

### Task 17: Annotate atom — unified dispatcher

**Files:**
- Create: `nkigym/src/nkigym/tune/annotate.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/inject_annotations/__init__.py`
- Create: `nkigym/src/nkigym/codegen/lowering/inject_annotations/buffer_degree.py`
- Create: `nkigym/src/nkigym/codegen/lowering/inject_annotations/software_pipeline.py`
- Test: `test/tune/test_annotate.py`

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for the Annotate atom and per-key validators."""

import pytest
from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode, SBlock
from nkigym.tune import AtomLegalityError
from nkigym.tune.annotate import Annotate


def test_annotate_rejects_unknown_key() -> None:
    """Unknown keys raise AtomLegalityError on apply."""
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (128, 128), "dtype": "bfloat16"},
              "rhs": {"shape": (128, 128), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    atom = Annotate(target_path=(0,), key="bogus_key", value=42)
    assert not atom.is_legal(module)


def test_annotate_buffer_degree_on_sblock() -> None:
    """buffer_degree annotation sets Tensor.buffer_degree at apply."""
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    # Find a root-level alloc SBlock for lhs_T_sbuf (first one in body).
    target_path = None
    for i, root in enumerate(module.body):
        if isinstance(root, SBlock) and any(
            c.op_cls.__name__ == "NKIAlloc" and c.kwargs.get("tensor_name") == "lhs_T_sbuf"
            for c in root.body
        ):
            target_path = (i,)
            break
    assert target_path is not None
    atom = Annotate(target_path=target_path, key="buffer_degree",
                     value={"lhs_T_sbuf": 2})
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    block = new_module.body[target_path[0]]
    assert isinstance(block, SBlock)
    assert block.annotations["buffer_degree"] == {"lhs_T_sbuf": 2}
```

- [ ] **Step 2: Implement Annotate**

```python
"""Unified Annotate atom — consolidates MultiBuffer + SoftwarePipeline.

Key-dispatched legality + apply. Each registered key has its own
validator; unknown keys raise AtomLegalityError.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir.ir import ForNode, KernelIR, SBlock, resolve_node
from nkigym.tune import AtomLegalityError

_KEY_VALIDATORS: dict[str, Callable[[Any, ForNode | SBlock, KernelIR], bool]] = {}


def register_annotation_key(
    key: str, validator: Callable[[Any, ForNode | SBlock, KernelIR], bool]
) -> None:
    """Register a key + its legality validator."""
    _KEY_VALIDATORS[key] = validator


def _validate_buffer_degree(value: Any, target: ForNode | SBlock, module: KernelIR) -> bool:
    """buffer_degree requires: target is SBlock; value is dict[str, int]; every
    tensor named exists + not a param."""
    if not isinstance(target, SBlock):
        return False
    if not isinstance(value, dict):
        return False
    for tname, degree in value.items():
        if not isinstance(tname, str) or not isinstance(degree, int) or degree < 1:
            return False
        if tname not in module.tensors:
            return False
        if module.tensors[tname].origin == "param":
            return False
    return True


def _validate_software_pipeline_depth(
    value: Any, target: ForNode | SBlock, module: KernelIR
) -> bool:
    """software_pipeline_depth requires: target is ForNode; value is int >= 1."""
    _ = module
    if not isinstance(target, ForNode):
        return False
    if not isinstance(value, int) or value < 1:
        return False
    return True


register_annotation_key("buffer_degree", _validate_buffer_degree)
register_annotation_key("software_pipeline_depth", _validate_software_pipeline_depth)


@dataclass(frozen=True)
class Annotate:
    """Attach a keyed annotation to a ForNode or SBlock.

    Attributes:
        target_path: Path to target in module.body.
        key: Registered annotation key.
        value: Serializable value matching the key's contract.
    """

    target_path: tuple[int, ...]
    key: str
    value: Any

    def is_legal(self, module: KernelIR) -> bool:
        target = resolve_node(module.body, self.target_path)
        if target is None:
            return False
        validator = _KEY_VALIDATORS.get(self.key)
        if validator is None:
            return False
        return validator(self.value, target, module)

    def apply(self, module: KernelIR) -> KernelIR:
        if not self.is_legal(module):
            raise AtomLegalityError(f"Annotate.apply: illegal {self!r}")
        target = resolve_node(module.body, self.target_path)
        assert target is not None
        new_annotations = dict(target.annotations)
        new_annotations[self.key] = self.value
        if isinstance(target, SBlock):
            # SBlock is mutable; rebuild via replace-at-path
            new_target = SBlock(
                iter_vars=target.iter_vars,
                reads=target.reads,
                writes=target.writes,
                reads_writes=target.reads_writes,
                body=target.body,
                annotations=new_annotations,
            )
        else:
            new_target = ForNode(
                iter_var=target.iter_var,
                children=target.children,
                name=target.name,
                annotations=new_annotations,
            )
        from nkigym.ir.ir import replace_at_path
        new_body = replace_at_path(module.body, self.target_path, new_target)
        return replace(module, body=new_body)


def enumerate_annotate_atoms(module: KernelIR) -> list[Annotate]:
    """Emit every legal Annotate instance across keys."""
    atoms: list[Annotate] = []
    atoms.extend(_enumerate_buffer_degree(module))
    atoms.extend(_enumerate_software_pipeline_depth(module))
    return atoms


def _enumerate_buffer_degree(module: KernelIR) -> list[Annotate]:
    """For every alloc SBlock, emit degree ∈ {2, 3, 4} on each of its tensors."""
    atoms: list[Annotate] = []
    for i, root in enumerate(module.body):
        if isinstance(root, SBlock) and any(
            c.op_cls.__name__ == "NKIAlloc" for c in root.body
        ):
            for call in root.body:
                if call.op_cls.__name__ != "NKIAlloc":
                    continue
                tname = call.kwargs["tensor_name"]
                if module.tensors[tname].origin == "param":
                    continue
                for degree in [2, 3, 4]:
                    atom = Annotate(target_path=(i,), key="buffer_degree",
                                     value={tname: degree})
                    if atom.is_legal(module):
                        atoms.append(atom)
    return atoms


def _enumerate_software_pipeline_depth(module: KernelIR) -> list[Annotate]:
    """For every ForNode, emit depth ∈ {2, 3}."""
    atoms: list[Annotate] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, ForNode):
            for depth in [2, 3]:
                atom = Annotate(target_path=path, key="software_pipeline_depth",
                                 value=depth)
                if atom.is_legal(module):
                    atoms.append(atom)
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
```

- [ ] **Step 3: Run tests**

```bash
pytest test/tune/test_annotate.py -x -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/annotate.py test/tune/test_annotate.py
git commit -m "refactor: unified Annotate atom with per-key dispatch"
```

---

### Task 18: Implement buffer_degree + software_pipeline sub-passes

**Files:**
- Create: `nkigym/src/nkigym/codegen/lowering/inject_annotations/buffer_degree.py`
- Create: `nkigym/src/nkigym/codegen/lowering/inject_annotations/software_pipeline.py`
- Modify: `nkigym/src/nkigym/codegen/lowering/inject_annotations/__init__.py`

Scope: port the body of the old `inject_multi_buffer.py` and `inject_software_pipeline.py` to read annotations off blocks/for_nodes instead of calling the respective atoms' `apply` logic.

- [ ] **Step 1: Port buffer_degree sub-pass**

`buffer_degree.py`:

```python
"""Sub-pass for the "buffer_degree" annotation key.

Reads buffer_degree annotations off alloc SBlocks; widens
``Tensor.buffer_degree[dim]`` per annotation; emits slot-modulo index
expressions in BufferAccess.pattern for subsequent accesses of widened
tensors.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def apply_buffer_degree(module: KernelIR) -> None:
    """Consume buffer_degree annotations; mutate module.tensors + AccessRanges."""
    for root in module.body:
        _walk(root, module)


def _walk(node: ForNode | SBlock, module: KernelIR) -> None:
    if isinstance(node, SBlock):
        bd = node.annotations.get("buffer_degree")
        if bd is not None:
            for tname, degree in bd.items():
                tensor = module.tensors[tname]
                # Widen along the P-slot dim; place_buffers reads this
                if tensor.dim_ids:
                    p_dim = tensor.dim_ids[0]
                    existing = tensor.buffer_degree.get(p_dim, 1)
                    tensor.buffer_degree[p_dim] = max(existing, degree)
        return
    for c in node.children:
        _walk(c, module)
```

- [ ] **Step 2: Port software_pipeline sub-pass**

`software_pipeline.py` — for now, just validate and mark ForNodes; renderer handles prologue/body/epilogue emission in emit_source (future extension; currently a no-op because SoftwarePipeline is deferred per brainstorm).

```python
"""Sub-pass for the "software_pipeline_depth" annotation key.

Currently validates + records; emission support (prologue / body /
epilogue) is deferred to the Bug B fix followup. Canonical + Annotate
without this sub-pass performs the same as depth=1.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def apply_software_pipeline(module: KernelIR) -> None:
    """Validate all software_pipeline_depth annotations; no-op emission."""
    for root in module.body:
        _walk(root)


def _walk(node: ForNode | SBlock) -> None:
    if isinstance(node, ForNode):
        depth = node.annotations.get("software_pipeline_depth", 1)
        if depth > 1:
            """TODO(Bug B followup): emit prologue/body/epilogue. For now
            depth > 1 is treated as a no-op — validates legality but emits
            the loop normally. Stamped here so a future atom that consumes
            this annotation has a clear insertion point."""
            pass
        for c in node.children:
            _walk(c)
    elif isinstance(node, SBlock):
        return
```

- [ ] **Step 3: Wire up dispatcher**

Replace `inject_annotations/__init__.py`:

```python
"""Dispatcher — calls per-key sub-passes in registered order."""

from nkigym.ir.ir import KernelIR
from nkigym.codegen.buffer_degree import apply_buffer_degree
from nkigym.codegen.software_pipeline import (
    apply_software_pipeline,
)


def inject_annotations(module: KernelIR) -> None:
    """Run each annotation key's sub-pass. Order matters: buffer_degree
    before software_pipeline because SW pipeline prologue/body/epilogue
    consumes buffer_degree-widened slots."""
    apply_buffer_degree(module)
    apply_software_pipeline(module)
```

- [ ] **Step 4: Update render.py**

```python
from nkigym.codegen.inject_annotations import inject_annotations


def render(module: KernelIR) -> str:
    place_buffers(module)
    inject_annotations(module)
    canonicalize_iter_var_names(module)
    return _emit_source(module)
```

- [ ] **Step 5: Write end-to-end test with buffer_degree annotation**

```python
def test_annotate_buffer_degree_widens_shape() -> None:
    """buffer_degree=2 annotation doubles the P-slot dim in emitted shape."""
    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.render import render
    from nkigym.tune.annotate import Annotate
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    # Find the lhs_T_sbuf alloc block index
    target_path = None
    from nkigym.ir.ir import SBlock
    for i, root in enumerate(module.body):
        if isinstance(root, SBlock) and any(
            c.op_cls.__name__ == "NKIAlloc" and c.kwargs.get("tensor_name") == "lhs_T_sbuf"
            for c in root.body
        ):
            target_path = (i,)
            break
    assert target_path is not None
    atom = Annotate(target_path=target_path, key="buffer_degree", value={"lhs_T_sbuf": 2})
    new_module = atom.apply(module)
    source = render(new_module)
    # Expect shape (128, 32, 2048) — double the num_P_tiles
    assert "(128, 32, 2048)" in source
```

- [ ] **Step 6: Run tests**

```bash
pytest test/tune/test_annotate.py test/codegen/test_canonical.py -x -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: implement buffer_degree + software_pipeline annotation sub-passes"
```

---

### Task 19: Port Split atom

**Files:**
- Modify: `nkigym/src/nkigym/tune/split.py`
- Test: `test/tune/test_split.py`

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for Split on the iter-var IR."""

import pytest
from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.tune import AtomLegalityError
from nkigym.tune.split import Split


def _find_first_fornode_path(module, predicate):
    """Helper: return path to the first ForNode matching predicate."""
    def walk(node, path):
        if isinstance(node, ForNode) and predicate(node):
            return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None
    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    return None


def test_split_divisor_factor_succeeds():
    """Split(target, factor=2) on a trip=4 loop produces trip-2 × trip-2."""
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    # Find matmul's d0 ACC loop (outermost of matmul nest; 16 num_tiles)
    target = _find_first_fornode_path(
        module, lambda n: n.iter_var.dim_id == "d0" and n.iter_var.extent == 16
    )
    assert target is not None
    atom = Split(loop_path=target, factor=4)
    assert atom.is_legal(module)
    new_mod = atom.apply(module)
    # After Split: outer extent=4, inner extent=4
    outer = new_mod.body[target[0]] if len(target) == 1 else None
    if outer is None:
        # descend
        node = new_mod.body[target[0]]
        for idx in target[1:]:
            node = node.children[idx]
        outer = node
    assert isinstance(outer, ForNode)
    assert outer.iter_var.extent == 4
    inner = outer.children[0]
    assert isinstance(inner, ForNode)
    assert inner.iter_var.extent == 4


def test_split_non_divisor_rejects():
    """Split(factor=3) on trip=4 raises AtomLegalityError."""
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    target = _find_first_fornode_path(
        module, lambda n: n.iter_var.dim_id == "d0" and n.iter_var.extent == 16
    )
    atom = Split(loop_path=target, factor=3)
    assert not atom.is_legal(module)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_split_rewrites_buffer_access_patterns():
    """BufferAccess patterns referencing the split iter var rewrite to
    iter_outer * inner_extent + iter_inner."""
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    target = _find_first_fornode_path(
        module, lambda n: n.iter_var.dim_id == "d0" and n.iter_var.extent == 16
    )
    atom = Split(loop_path=target, factor=4)
    new_mod = atom.apply(module)
    source = render(new_mod)
    # Verify split loop headers exist
    assert "for i_d0_0 in range(4):" in source
    assert "for i_d0_1 in range(4):" in source
    # Verify split composite indexing in matmul slice
    # e.g. lhs_T_sbuf[..., (i_d0_0 * 4 + i_d0_1), ...]
    assert ("* 4" in source) or ("* 4 +" in source)
```

- [ ] **Step 2: Implement Split**

```python
"""Split atom — partition a loop into outer × inner via factor.

Operates on iter-var IR: retires the target ForNode's iter var; emits
two new iter vars (v_outer, v_inner); rewrites all BufferAccess.pattern
references from v to (v_outer * factor + v_inner).
"""

from dataclasses import dataclass, replace

from nkigym.ir.ir import (
    AccessRange,
    BufferAccess,
    ForNode,
    IterVar,
    KernelIR,
    SBlock,
    resolve_node,
    replace_at_path,
)
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class Split:
    """Split a loop's iter var into outer/inner pair.

    Attributes:
        loop_path: Path to the target ForNode in module.body.
        factor: Inner extent; must divide target extent.
    """

    loop_path: tuple[int, ...]
    factor: int

    def is_legal(self, module: KernelIR) -> bool:
        target = resolve_node(module.body, self.loop_path)
        if not isinstance(target, ForNode):
            return False
        iv = target.iter_var
        if self.factor < 2 or self.factor >= iv.extent:
            return False
        return iv.extent % self.factor == 0

    def apply(self, module: KernelIR) -> KernelIR:
        if not self.is_legal(module):
            raise AtomLegalityError(f"Split.apply: illegal {self!r}")
        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, ForNode)
        old_iv = target.iter_var
        outer_extent = old_iv.extent // self.factor
        inner_extent = self.factor
        v_outer = module.allocate_iter_var(old_iv.dim_id, outer_extent, old_iv.role)
        v_inner = module.allocate_iter_var(old_iv.dim_id, inner_extent, old_iv.role)

        # Rewrite all BufferAccess patterns referencing old_iv.var_id
        new_body = _rewrite_patterns(module.body, old_iv.var_id, v_outer.var_id,
                                       v_inner.var_id, inner_extent)

        # Replace target ForNode with nested pair
        new_inner = ForNode(iter_var=v_inner, children=target.children,
                             name=None, annotations=dict(target.annotations))
        new_outer = ForNode(iter_var=v_outer, children=[new_inner], name=None)
        new_body = replace_at_path(new_body, self.loop_path, new_outer)

        # Update SBlock iter_vars lists
        new_body = _update_sblock_iter_vars(new_body, old_iv.var_id, v_outer, v_inner)

        return replace(module, body=new_body)


def _rewrite_patterns(body, old_id, outer_id, inner_id, inner_extent):
    """Rewrite every BufferAccess.pattern referencing old_id to
    (outer_id * inner_extent + inner_id)."""
    def rewrite_block(block: SBlock) -> SBlock:
        def rewrite_access(acc: BufferAccess) -> BufferAccess:
            if old_id not in acc.iter_var_ids:
                return acc
            new_ids = tuple(
                (inner_id if i == old_id else i) for i in acc.iter_var_ids
            )
            if outer_id not in new_ids:
                new_ids = new_ids + (outer_id,)
            new_pattern = []
            for ar in acc.pattern:
                coeffs = ar.coeffs
                if old_id in coeffs:
                    old_c = coeffs.pop(old_id)
                    coeffs[outer_id] = old_c * inner_extent
                    coeffs[inner_id] = old_c
                new_pattern.append(AccessRange.make(coeffs, ar.const_offset, ar.extent))
            return BufferAccess(tensor_name=acc.tensor_name,
                                  iter_var_ids=new_ids, pattern=tuple(new_pattern))
        return SBlock(
            iter_vars=block.iter_vars,
            reads={k: rewrite_access(v) for k, v in block.reads.items()},
            writes={k: rewrite_access(v) for k, v in block.writes.items()},
            reads_writes={k: rewrite_access(v) for k, v in block.reads_writes.items()},
            body=block.body,
            annotations=dict(block.annotations),
        )
    def rewrite_node(node):
        if isinstance(node, SBlock):
            return rewrite_block(node)
        return ForNode(
            iter_var=node.iter_var,
            children=[rewrite_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )
    return [rewrite_node(n) for n in body]


def _update_sblock_iter_vars(body, old_id, v_outer, v_inner):
    """Update every SBlock.iter_vars list: replace old_id entry with v_outer
    and v_inner (in that order)."""
    def update_block(block: SBlock) -> SBlock:
        new_ivs: list[IterVar] = []
        replaced = False
        for iv in block.iter_vars:
            if iv.var_id == old_id:
                new_ivs.append(v_outer)
                new_ivs.append(v_inner)
                replaced = True
            else:
                new_ivs.append(iv)
        if not replaced:
            return block
        return SBlock(
            iter_vars=new_ivs,
            reads=block.reads,
            writes=block.writes,
            reads_writes=block.reads_writes,
            body=block.body,
            annotations=dict(block.annotations),
        )
    def update_node(node):
        if isinstance(node, SBlock):
            return update_block(node)
        return ForNode(
            iter_var=node.iter_var,
            children=[update_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )
    return [update_node(n) for n in body]


def enumerate_split_atoms(module: KernelIR) -> list[Split]:
    """Emit Split(target, factor) for every ForNode × divisor factor."""
    atoms: list[Split] = []

    def walk(node, path):
        if isinstance(node, ForNode):
            extent = node.iter_var.extent
            for factor in range(2, extent):
                if extent % factor == 0:
                    atom = Split(loop_path=path, factor=factor)
                    if atom.is_legal(module):
                        atoms.append(atom)
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
```

- [ ] **Step 3: Run tests**

```bash
pytest test/tune/test_split.py -x -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/split.py test/tune/test_split.py
git commit -m "refactor: port Split atom to iter-var IR"
```

---

### Task 20: Port Reorder atom (n-ary, iter-var-keyed)

**Files:**
- Modify: `nkigym/src/nkigym/tune/reorder.py`
- Test: `test/tune/test_reorder.py`

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for Reorder atom on iter-var IR."""

import pytest
from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode
from nkigym.tune.reorder import Reorder
from nkigym.ops.base import AxisRole


def _collect_fornodes(module):
    """Walk module.body; return list of (path, ForNode) tuples."""
    results = []
    def walk(node, path):
        if isinstance(node, ForNode):
            results.append((path, node))
            for i, c in enumerate(node.children):
                walk(c, path + (i,))
    for i, root in enumerate(module.body):
        walk(root, (i,))
    return results


def test_reorder_adjacent_par_par_commutes():
    """Adjacent PAR × PAR swap is legal — matmul's d1, d3 loops (both PAR)."""
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    # Find matmul subtree: d0 ACC outer, d1 PAR, d3 PAR, SBlock
    # Reorder d1 and d3 at their iter-var positions.
    fns = _collect_fornodes(module)
    d1 = next((p, n) for p, n in fns if n.iter_var.dim_id == "d1"
              and n.iter_var.role == AxisRole.PARALLEL)
    d3 = next((p, n) for p, n in fns if n.iter_var.dim_id == "d3"
              and n.iter_var.role == AxisRole.PARALLEL)
    atom = Reorder(iter_var_ids=(d3[1].iter_var.var_id, d1[1].iter_var.var_id))
    assert atom.is_legal(module)


def test_reorder_adjacent_par_acc_rejects_when_par_writes():
    """PAR × ACC swap is illegal when PAR's dim indexes a write region of
    the subtree."""
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    fns = _collect_fornodes(module)
    # matmul's d0 is ACC; its d1 is PAR. d1 indexes psum_acc's M axis (write).
    # Reorder(d1, d0_acc) would require subtree-purity w.r.t. d1 → expected fail.
    d0_acc = next((p, n) for p, n in fns if n.iter_var.dim_id == "d0"
                   and n.iter_var.role == AxisRole.ACCUMULATION)
    d1 = next((p, n) for p, n in fns if n.iter_var.dim_id == "d1"
              and n.iter_var.role == AxisRole.PARALLEL)
    atom = Reorder(iter_var_ids=(d1[1].iter_var.var_id, d0_acc[1].iter_var.var_id))
    assert not atom.is_legal(module)


def test_reorder_round_trip_restores_canonical():
    """Applying Reorder twice (with same args) restores the original tree."""
    from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym
    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
              "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, specs)
    fns = _collect_fornodes(module)
    d1 = next((p, n) for p, n in fns if n.iter_var.dim_id == "d1"
              and n.iter_var.role == AxisRole.PARALLEL)
    d3 = next((p, n) for p, n in fns if n.iter_var.dim_id == "d3"
              and n.iter_var.role == AxisRole.PARALLEL)
    atom_fwd = Reorder(iter_var_ids=(d3[1].iter_var.var_id, d1[1].iter_var.var_id))
    once = atom_fwd.apply(module)
    # After reorder, iter-var IDs are stable — reorder back
    atom_back = Reorder(iter_var_ids=(d1[1].iter_var.var_id, d3[1].iter_var.var_id))
    twice = atom_back.apply(once)
    # Verify the canonical order returns: d1 above d3 in matmul subtree.
    fns2 = _collect_fornodes(twice)
    d1_2 = next((p, n) for p, n in fns2 if n.iter_var.var_id == d1[1].iter_var.var_id)
    d3_2 = next((p, n) for p, n in fns2 if n.iter_var.var_id == d3[1].iter_var.var_id)
    assert len(d1_2[0]) < len(d3_2[0])  # d1 is outer
```

- [ ] **Step 2: Implement Reorder**

Port the existing `Reorder` atom's role-commutation legality (from `reorder.py`). Key changes:

- Atom takes `iter_var_ids: tuple[int, ...]` (n-ary) instead of `outer_path` + `inner_path` pair.
- Legality: iter vars form a consecutive chain in the current tree (rebuilt lookup via walk).
- Apply: reshape the ForNode chain to match the given iter-var order top-to-bottom.
- Role-commute check: reuse `_roles_commute` logic from old impl; inspect iter var roles.

See old `nkigym/src/nkigym/tune/reorder.py` for the role-commutation logic; port verbatim adapting to iter-var-based access.

- [ ] **Step 3: Run tests**

```bash
pytest test/tune/test_reorder.py -x -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/reorder.py test/tune/test_reorder.py
git commit -m "refactor: port Reorder atom to n-ary iter-var-keyed API"
```

---

### Task 21: Port Fuse atom (renderer-supported mod/div)

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse.py`
- Test: `test/tune/test_fuse.py`

- [ ] **Step 1: Write failing tests**

```python
def test_fuse_two_nested_par_loops():
    """Fuse(outer, inner) where both PAR produces single loop with merged extent."""
    # Build a module with Loop d0 trip=4 → Loop d3 trip=2; apply Fuse; expect
    # one ForNode with extent=8 and synthetic dim_id "d0_x_d3".
    ...


def test_fuse_registers_synthetic_dim_in_module_dims():
    """Fuse adds a DimInfo entry for the synthetic dim."""
    # After Fuse: "d0_x_d3" in module.dims.
    ...


def test_fuse_renderer_emits_div_and_mod():
    """Rendered code contains // and % for fused-dim decomposition."""
    # Rendered output has "fused_var // inner_ext" and "fused_var % inner_ext".
    ...
```

- [ ] **Step 2: Implement Fuse**

Pattern matching Task 19's Split but in reverse:
- Retire `v_outer`, `v_inner`; allocate `v_fused` with `extent = v_outer.extent * v_inner.extent`, `dim_id = f"{v_outer.dim_id}_x_{v_inner.dim_id}"`, role = `max(v_outer.role, v_inner.role)` in the PAR⊂SEQ⊂ACC lattice.
- Register `DimInfo(dim_id=synthetic, total_size=outer.extent * inner.extent * tile, tile_size=min(outer.tile, inner.tile), num_tiles=v_fused.extent)`.
- Replace nested `ForNode(v_outer) → ForNode(v_inner)` with single `ForNode(v_fused)`.
- Rewrite all `BufferAccess.pattern`: `v_outer coeff X, v_inner coeff Y` → `v_fused coeffs {combined}`. For the renderer to emit `//` and `%`, the pattern needs a flag or additional metadata.

Extension needed: `AccessRange` gets an optional `fuse_decomp: tuple[int, int, int] | None` field: `(fused_iv_id, outer_coeff, inner_ext)`. When set, renderer emits `(fused_iv // inner_ext * outer_coeff) + (fused_iv % inner_ext)`. Add a helper function in `_emit_utils.py` that dispatches on fuse_decomp.

- [ ] **Step 3: Update `_emit_utils.py` to emit div/mod**

Add conditional logic in `_emit_affine_start` that checks for fuse_decomp annotation and emits `//` / `%` accordingly.

- [ ] **Step 4: Run tests**

```bash
pytest test/tune/test_fuse.py -x -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse.py nkigym/src/nkigym/codegen/lowering/_emit_utils.py test/tune/test_fuse.py
git commit -m "refactor: port Fuse atom + renderer div/mod support for synthetic dims"
```

---

### Task 22: Port ComputeAt atom (prefix-match + role promotion)

**Files:**
- Modify: `nkigym/src/nkigym/tune/compute_at.py`
- Test: `test/tune/test_compute_at.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_at_prefix_match_required():
    """Target chain must be a prefix of block's iter_vars (matched by dim_id)."""
    # Build module; attempt ComputeAt where target's dims don't match
    # block's leading iter-var order; assert AtomLegalityError.
    ...


def test_compute_at_preserves_consumer_inner_chain():
    """ComputeAt preserves block's inner loops (not regenerated)."""
    # Split matmul's d1; apply ComputeAt(matmul, target); verify matmul's
    # split d1/outer d1/inner loops are preserved in the result.
    ...


def test_compute_at_promotes_role_par_plus_acc_to_acc():
    """PAR + ACC = ACC role promotion on matched ancestor."""
    # Place block with ACC iter var on dim d0 under target with PAR iter var
    # on dim d0; verify target ForNode's iter_var.role becomes ACCUMULATION.
    ...


def test_compute_at_rejects_acc_to_par_demotion():
    """ACC + PAR demotion is illegal."""
    # Place block with PAR iter var on dim d0 under target with ACC iter var;
    # assert rejection.
    ...


def test_compute_at_merges_iter_vars_on_matched_dims():
    """Block's iter var on dim d0 retires; pattern references rebind to target's iter var."""
    # Verify BufferAccess.iter_var_ids contains target's iter var id, not
    # block's old iter var id, for the merged dim.
    ...
```

- [ ] **Step 2: Implement ComputeAt**

Structure:
- Legality:
  - Target is ForNode, not ancestor of block.
  - Target's subtree contains a consumer of block's writes.
  - Target's ancestor chain iter-var dims form a prefix of block's iter_vars dims.
  - For each matched dim pair, `max(target_role, block_role)` ≥ current target role (no demote).
- Apply:
  - Compute merged iter-var mapping: `block_iv_id → target_iv_id` for each matched dim.
  - Walk block + its subtree; retire merged iter vars; rewrite `BufferAccess.iter_var_ids` and `AccessRange.iter_var_coeffs` (replace block_iv_id with target_iv_id; adjust coefficients using block's original pattern).
  - Role promotion: for each matched (target_iv, block_iv) pair where block_iv.role > target_iv.role in the PAR ⊂ SEQ ⊂ ACC lattice, retire the target_iv; allocate a fresh iter var with the stronger role (same dim_id, same extent); rewrite the target ForNode's iter_var binding and all BufferAccess.pattern references from old target_iv_id to new iter_var_id.
  - Build new subtree: remaining (unmatched) block iter vars become inner ForNodes in canonical order, innermost is the block.
  - Remove block from its current location; append subtree under target.
  - Canonical rename.

- [ ] **Step 3: Run tests**

```bash
pytest test/tune/test_compute_at.py -x -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/compute_at.py test/tune/test_compute_at.py
git commit -m "refactor: port ComputeAt with prefix-match + role promotion"
```

---

### Task 23: Port ReverseComputeAt (dual of ComputeAt)

**Files:**
- Modify: `nkigym/src/nkigym/tune/reverse_compute_at.py`
- Test: `test/tune/test_reverse_compute_at.py`

Mirror Task 22's structure, dual semantics: target's subtree must contain a producer of block's reads (instead of consumer of writes).

- [ ] **Step 1: Write failing tests**

Mirror Task 22's tests with "consumer" / "producer" roles swapped.

- [ ] **Step 2: Implement ReverseComputeAt**

Reuse `_merge_iter_vars` / `_promote_roles` / `_regenerate_subtree` helpers from `compute_at.py` via shared imports. Change only the legality's producer-vs-consumer direction.

- [ ] **Step 3: Run tests**

```bash
pytest test/tune/test_reverse_compute_at.py -x -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/reverse_compute_at.py test/tune/test_reverse_compute_at.py
git commit -m "refactor: port ReverseComputeAt with same semantics as ComputeAt"
```

---

### Task 24: Port RFactor atom

**Files:**
- Modify: `nkigym/src/nkigym/tune/rfactor.py`
- Test: `test/codegen/test_rfactor_rmw.py`, `test/codegen/test_rfactor_slot.py`

- [ ] **Step 1: Remove xfail markers**

In `test/codegen/test_rfactor_rmw.py`, remove `@pytest.mark.xfail` from
`test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly`.

- [ ] **Step 2: Port RFactor implementation**

Reuse the existing structural logic for rmw + slot recipes; adapt:
- Read target's `SBlock` / iter vars instead of `BodyLeaf` / `dim_role`.
- Build 3 new SBlocks (init, update, drain) using the iter-var allocator.
- Construct staging tensor's `BufferAccess.pattern` with 3-D access
  `(d_M, d_N, d_outer)` — now emittable because renderer handles N-D.

- [ ] **Step 3: Run tests**

```bash
pytest test/codegen/test_rfactor_rmw.py test/codegen/test_rfactor_slot.py -x -v
```

Expected: pass, including previously xfailed test.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/rfactor.py test/codegen/
git commit -m "refactor: port RFactor; re-enable 3D staging buffer test"
```

---

### Task 25: Remove deleted atoms + update batch sampler

**Files:**
- Delete: `nkigym/src/nkigym/tune/hoist_invariant.py`
- Delete: `nkigym/src/nkigym/tune/multi_buffer.py`
- Delete: `nkigym/src/nkigym/tune/software_pipeline.py`
- Delete: `test/tune/test_hoist_invariant.py`
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Modify: `nkigym/src/nkigym/tune/__init__.py`

- [ ] **Step 1: Delete files**

```bash
git rm nkigym/src/nkigym/tune/hoist_invariant.py
git rm nkigym/src/nkigym/tune/multi_buffer.py
git rm nkigym/src/nkigym/tune/software_pipeline.py
git rm test/tune/test_hoist_invariant.py
```

- [ ] **Step 2: Update batch.py**

```python
"""Frontier-expansion sampler — 7 atoms."""

import random
import warnings

from nkigym.ir.dep_cache import subtree_signature
from nkigym.ir.ir import KernelIR, validate_dataflow_ordering
from nkigym.tune import AtomLegalityError, KernelRewrite
from nkigym.tune.annotate import enumerate_annotate_atoms
from nkigym.tune.compute_at import enumerate_compute_at_atoms
from nkigym.tune.fuse import enumerate_fuse_atoms
from nkigym.tune.reorder import enumerate_reorder_atoms
from nkigym.tune.reverse_compute_at import enumerate_reverse_compute_at_atoms
from nkigym.tune.rfactor import enumerate_rfactor_atoms
from nkigym.tune.split import enumerate_split_atoms


def _enumerate_atoms(module: KernelIR) -> list[KernelRewrite]:
    """7 atoms — Split, Reorder, Fuse, ComputeAt, ReverseComputeAt, RFactor, Annotate."""
    atoms: list[KernelRewrite] = []
    atoms.extend(enumerate_split_atoms(module))
    atoms.extend(enumerate_reorder_atoms(module))
    atoms.extend(enumerate_fuse_atoms(module))
    atoms.extend(enumerate_compute_at_atoms(module))
    atoms.extend(enumerate_reverse_compute_at_atoms(module))
    atoms.extend(enumerate_rfactor_atoms(module))
    atoms.extend(enumerate_annotate_atoms(module))
    return atoms


# hash_state, enumerate_pool, sample_pool unchanged from v1
```

Port `hash_state`, `enumerate_pool`, `sample_pool` from the current `batch.py`; update references from `BodyLeaf`/`LoopNode` to `SBlock`/`ForNode`.

- [ ] **Step 3: Run sampler test**

```bash
pytest test/codegen/test_batch.py -x -v
```

Expected: pass. Frontier produces >0 atoms for a canonical matmul module.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: 7-atom sampler — remove HoistInvariant/MultiBuffer/SoftwarePipeline"
```

---

## Phase D — MFU Acceptance (Week 4)

### Task 26: End-to-end matmul_lhsT_rhs tuning run

**Files:**
- No code changes; executes `examples/matmul_lhsT_rhs.py`.

- [ ] **Step 1: Clean cache**

```bash
rm -rf /home/ubuntu/cache/matmul_lhsT_rhs_tune
pkill -9 -u $USER python compile walrus || true  # clear residuals
```

- [ ] **Step 2: Run tuner**

```bash
source ~/venvs/kernel-env/bin/activate
export PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/nki-autotune/autotune/src:$PYTHONPATH
python -u examples/matmul_lhsT_rhs.py 2>&1 | tee /tmp/matmul_lhsT_rhs_mfu_gate.log
```

Expected: all 100 kernels render + CPU-sim; profile results written.

- [ ] **Step 3: Extract best MFU**

```bash
python3 <<'EOF'
import json
from pathlib import Path
data = json.loads(Path("/home/ubuntu/cache/matmul_lhsT_rhs_tune/results.json").read_text())
mfus = []
for k in data["kernels"]:
    s = k.get("profiler_summary") or {}
    m = s.get("mfu_estimated_percent")
    if m is not None:
        mfus.append(m * 100)
mfus.sort(reverse=True)
print(f"num succeeded: {len(mfus)}")
print(f"top 5: {mfus[:5]}")
print(f"best: {mfus[0] if mfus else 'N/A'}")
EOF
```

- [ ] **Step 4: Verify acceptance bar**

Best MFU must be ≥ 90.00%. If lower, investigate — diagnose the gap before
merging. Likely causes: Annotate buffer_degree enumeration not hitting
optimal values; Split enumerator factor range too narrow; Reorder not
commuting in cases where it should.

- [ ] **Step 5: Document in the plan's progress tracker**

Append the actual achieved MFU to this plan document as a note at the end.

- [ ] **Step 6: Commit only if bar met**

```bash
# After confirming ≥90% MFU achieved:
echo "Best MFU on matmul_lhsT_rhs: <X.XX%>" >> docs/superpowers/plans/2026-05-10-iter-var-refactor.md
git add docs/superpowers/plans/2026-05-10-iter-var-refactor.md
git commit -m "gate: matmul_lhsT_rhs MFU acceptance — <X.XX%>"
```

---

### Task 27: End-to-end matmul_lhs_rhs tuning run

Repeat Task 26's pattern for `matmul_lhs_rhs`. Acceptance bar: ≥84.00%.

- [ ] **Step 1: Run tuner**

```bash
rm -rf /home/ubuntu/cache/matmul_lhs_rhs_tune
pkill -9 -u $USER python compile walrus || true
python -u examples/matmul_lhs_rhs.py 2>&1 | tee /tmp/matmul_lhs_rhs_mfu_gate.log
```

- [ ] **Step 2: Extract + verify + commit**

Follow the same pattern as Task 26's steps 3-6.

---

### Task 28: End-to-end rmsnorm_matmul tuning run

Repeat for `rmsnorm_matmul`. Acceptance bar: ≥79.00%.

- [ ] **Step 1: Run tuner**

```bash
rm -rf /home/ubuntu/cache/rmsnorm_matmul_tune
pkill -9 -u $USER python compile walrus || true
python -u examples/rmsnorm_matmul.py 2>&1 | tee /tmp/rmsnorm_matmul_mfu_gate.log
```

- [ ] **Step 2: Extract + verify + commit**

Same pattern.

---

### Task 29: Update `docs/ir-design.md` to new IR

**Files:**
- Modify: `docs/ir-design.md`
- Modify: diagram mmd/png files

- [ ] **Step 1: Rewrite §2 (Envelope) field table to reflect new structs**

Replace `LoopNode` / `BodyLeaf` field references with `ForNode` / `SBlock` /
`IterVar` / `NKIOpCall` / `BufferAccess` / `AccessRange` entries.

- [ ] **Step 2: Update §5 (Atom taxonomy) to 7 atoms**

Remove `HoistInvariant` row; collapse `MultiBuffer` + `SoftwarePipeline`
rows into a single `Annotate` row. Update diagram `atom-taxonomy.mmd`.

- [ ] **Step 3: Update §7 (Rendering pipeline)**

Rename passes: `inject_annotations` replaces `inject_multi_buffer` +
`inject_software_pipeline`. Mention iter-var-keyed canonicalization.

- [ ] **Step 4: Update §9 (File map)**

Reflect final file structure.

- [ ] **Step 5: Regenerate diagrams**

```bash
cd docs/diagrams
for f in *.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.png" -s 4 -p pipeline-config.json
done
```

- [ ] **Step 6: Commit**

```bash
git add docs/
git commit -m "docs: rewrite ir-design.md + diagrams for iter-var IR (7 atoms)"
```

---

### Task 30: Remove .bak files + update CLAUDE.md

**Files:**
- Delete: `nkigym/src/nkigym/codegen/ir_v1.py.bak`
- Delete: `nkigym/src/nkigym/codegen/dep_cache_v1.py.bak`
- Delete: `test/codegen/test_ir_v1.py.bak`
- Delete: `test/codegen/test_dep_cache_v1.py.bak`

- [ ] **Step 1: Delete backup files**

```bash
git rm nkigym/src/nkigym/codegen/ir_v1.py.bak
git rm nkigym/src/nkigym/codegen/dep_cache_v1.py.bak
git rm test/codegen/test_ir_v1.py.bak
git rm test/codegen/test_dep_cache_v1.py.bak
```

- [ ] **Step 2: Update CLAUDE.md learnings (optional)**

If any learning entries reference `BodyLeaf` / `LoopNode`, update terminology.
Leave historical context entries unchanged — they describe past state.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove transitional .bak files after refactor"
```

---

### Task 31: Merge iter-var-refactor branch to dev_1

**Files:**
- No file changes; git operation only.

- [ ] **Step 1: Verify all tests pass**

```bash
source ~/venvs/kernel-env/bin/activate
export PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/nki-autotune/autotune/src:$PYTHONPATH
pytest test/ -x --no-header
```

Expected: 106+ passed.

- [ ] **Step 2: Verify MFU gates committed**

```bash
git log --oneline | grep "^gate:"
```

Expected: three commits — matmul_lhsT_rhs, matmul_lhs_rhs, rmsnorm_matmul gates.

- [ ] **Step 3: Switch to dev_1 + merge**

```bash
cd <main-worktree>
git checkout dev_1
git merge --no-ff iter-var-refactor -m "Merge iter-var-refactor: TVM-style iter-var IR

Replaces path-based LoopNode/BodyLeaf with iter-var identity
(SBlock/ForNode/IterVar). 7 atoms (HoistInvariant dropped;
MultiBuffer/SoftwarePipeline consolidated into Annotate). MFU parity
achieved on matmul_lhsT_rhs, matmul_lhs_rhs, rmsnorm_matmul."
```

- [ ] **Step 4: Run full suite on dev_1**

```bash
pytest test/ -x --no-header
```

Expected: pass.

- [ ] **Step 5: (DO NOT push without explicit user approval)**

Stop here. Report to user; await push authorization per project's git safety protocol.

---

## Final Notes

**After Task 31:** Followup specs unblocked by this refactor:
- Bug B (`SoftwarePipeline` pipeline-skew widening) — now straightforward via annotation key.
- Bug #1 (`DecomposeReduction` PSUM scope) — spec + plan separately. Split `RFactor` into `rfactor` + `decompose_reduction`.
- `cache_read` / `cache_write` — principled staging buffer insertion as atoms.
- Full TVM software pipeline (asymmetric `stage[]` + `order[]`) — as new annotation keys.

**Rollback plan if any acceptance gate fails:**
1. `git checkout dev_1` (original).
2. Diagnose the specific failure on `iter-var-refactor` branch.
3. If the gap is fixable, iterate on the specific task; re-run MFU gate.
4. If fundamentally blocked, document the blocker, halt, and consult the user before abandoning.
