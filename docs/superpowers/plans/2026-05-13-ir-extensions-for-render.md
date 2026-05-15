# IR Extensions for `render()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `KernelIR` so it carries every piece of information a mechanical `render(ir) -> str` pass needs (tensor `location` / `dtype`, per-op call kwargs, kernel return name) while flattening the envelope so no information is duplicated or stale.

**Architecture:** Five incremental TDD tasks add fields to `ISANode` and `TensorDims` and parse the return name, leaving the existing public API intact. A sixth task flattens `DimensionAnalysis` into `KernelIR`, privatises `OpAxes` as `_OpRecord`, and drops `KernelTree.dim_sizes` in one coordinated change. A seventh task cleans up package exports.

**Tech Stack:** Python 3.12, `dataclasses`, `ast`, `networkx`, `pytest`. Test via `pytest test/ir/ -v` from the repo root with the kernel venv active.

**Reference spec:** `docs/superpowers/specs/2026-05-13-ir-extensions-for-render-design.md`

---

## File Structure

**Created:**
- `test/ir/__init__.py` — makes `test/ir/` a package for pytest discovery.
- `test/ir/test_ir_extensions.py` — end-to-end tests against a single fixture kernel (matmul `lhs_T.T @ rhs` with `K=M=N=2048`). Grows across tasks; the fixture kernel is defined at the top of the file.

**Modified:**
- `nkigym/src/nkigym/ir/dimension_analysis.py` — tracer extensions (alloc kwargs, compute-op kwargs, return-name parse), `OpAxes` → `_OpRecord` rename, `DimensionAnalysis` dataclass removal.
- `nkigym/src/nkigym/ir/tree.py` — `ISANode.kwargs` + `ISANode.axis_map` fields, `_attach_op_subtree` signature, `KernelTree.__init__` / `KernelTree.dim_sizes` removal, `__all__` cleanup.
- `nkigym/src/nkigym/ir/ir.py` — `KernelIR` flat fields, `build_initial_ir` pipeline.
- `nkigym/src/nkigym/ir/__init__.py` — package export list.

**Unchanged:**
- `nkigym/src/nkigym/ir/dependency.py` — `Dependency` only touches `KernelTree`.
- `nkigym/src/nkigym/ir/_mermaid.py` — Mermaid helpers.
- `nkigym/src/nkigym/ops/*` — op surface is untouched.
- `examples/matmul_lhsT_rhs.py` — only uses `build_initial_ir`, which keeps its public contract.

---

## Environment setup

Before starting, every task assumes:

```bash
cd /home/ubuntu/nki-autotune
source ~/venvs/kernel-env/bin/activate
```

Run the full IR suite after every task:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

The existing `test/ops/` suite should pass unchanged after each task:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

---

## Task 1: Bootstrap test directory and capture per-op call kwargs on `ISANode`

**Files:**
- Create: `test/ir/__init__.py`
- Create: `test/ir/test_ir_extensions.py`
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py` (add `kwargs` to `OpAxes`, capture in `_trace_compute_op`)
- Modify: `nkigym/src/nkigym/ir/tree.py` (add `kwargs` to `ISANode`, thread through `_attach_op_subtree`)

---

- [ ] **Step 1.1: Create the empty test package init**

Create `test/ir/__init__.py`:

```python
```

(Empty file; presence makes `test/ir/` a package so pytest can collect siblings that share a fixture.)

- [ ] **Step 1.2: Write the fixture kernel + failing test for memset kwargs**

Create `test/ir/test_ir_extensions.py`:

```python
"""Tests for the extended IR fields (location, dtype, kwargs, axis_map, return_name)."""

from nkigym.ir import ISANode, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
_INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def _matmul_fixture(lhs_T, rhs):
    """``lhs_T.T @ rhs`` fixture shared across tests in this file."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def _isa_leaves(ir) -> list[ISANode]:
    """Return every :class:`ISANode` payload in pre-order."""
    return [ir.tree.data(nid) for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def _leaves_by_op(ir, op_name: str) -> list[ISANode]:
    """Filter ISA leaves by their op class name."""
    return [leaf for leaf in _isa_leaves(ir) if leaf.op_cls.__name__ == op_name]


def test_memset_leaf_carries_value_kwarg():
    """NKIMemset(value=0.0) call-site kwargs are captured onto ISANode.kwargs."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    memsets = _leaves_by_op(ir, "NKIMemset")
    assert len(memsets) == 1
    assert memsets[0].kwargs == {"value": 0.0}


def test_non_config_leaves_have_empty_kwargs():
    """Load/Store/Matmul/TensorCopy take no non-operand kwargs in this fixture."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for op_name in ("NKILoad", "NKIStore", "NKIMatmul", "NKITensorCopy"):
        for leaf in _leaves_by_op(ir, op_name):
            assert leaf.kwargs == {}, f"{op_name} leaf kwargs={leaf.kwargs}"


def test_alloc_leaves_have_empty_kwargs():
    """NKIAlloc leaves never carry kwargs — alloc params live on TensorDims."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        assert leaf.kwargs == {}
```

- [ ] **Step 1.3: Run the new tests to verify they fail**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/test_ir_extensions.py -v
```

Expected: all three `kwargs` tests fail with `AttributeError: 'ISANode' object has no attribute 'kwargs'`.

- [ ] **Step 1.4: Add `kwargs` field to `OpAxes`**

Edit `nkigym/src/nkigym/ir/dimension_analysis.py`. Change the `OpAxes` dataclass (currently around lines 32-44):

```python
@dataclass
class OpAxes:
    """Per-op dim-unification result.

    Attributes:
        op_cls: The NKIOp subclass.
        operand_names: ``slot → tensor_name`` for every operand in the call.
        axis_map: ``abstract_axis → concrete_dim`` (e.g. ``{"K": "d0"}``).
        kwargs: Non-operand call kwargs (e.g. ``{"value": 0.0}``).
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    axis_map: dict[str, str]
    kwargs: dict[str, Any]
```

- [ ] **Step 1.5: Populate `OpAxes.kwargs` from `_trace_compute_op`**

In `nkigym/src/nkigym/ir/dimension_analysis.py`, edit `_trace_compute_op` (currently ends around line 198). Replace its final `state.op_records.append(...)` line with:

```python
    op_kwargs = {k: v for k, v in kwargs.items() if k not in cls.OPERAND_AXES}
    state.op_records.append(
        OpAxes(op_cls=cls, operand_names=operand_names, axis_map=local, kwargs=op_kwargs)
    )
```

- [ ] **Step 1.6: Add `kwargs` field to `ISANode`**

Edit `nkigym/src/nkigym/ir/tree.py`. Change the `ISANode` dataclass (currently lines 72-92):

```python
@dataclass(frozen=True, kw_only=True)
class ISANode:
    """NKI-instruction payload.

    Attributes:
        op_cls: The :class:`NKIOp` subclass (e.g. ``NKILoad``).
        reads: Tensor names from slots in ``op_cls.INPUT_OPERANDS``.
        writes: Tensor names from slots that are neither input nor RMW.
        rmw: Tensor names from slots in ``op_cls.RMW_OPERANDS``.
        tensorize_sizes: Per-axis tile width (``dim → tile_size``)
            lowered onto the ISA call's slice width. Keys mirror the
            ``dim`` of the enclosing :class:`ForNode` chain; for
            :class:`NKIAlloc` leaves (no surrounding loops) this is
            empty.
        kwargs: Non-operand call kwargs captured from the tracer
            (e.g. ``{"value": 0.0}`` for ``NKIMemset``,
            ``{"op": "rsqrt", "scale": 1.0}`` for ``NKIActivation``).
            Empty for :class:`NKIAlloc` leaves.
    """

    op_cls: type[NKIOp]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    rmw: tuple[str, ...] = ()
    tensorize_sizes: dict[str, int] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
```

Add `from typing import Any` near the top of the file if it's not already present.

- [ ] **Step 1.7: Propagate `kwargs` in `_attach_op_subtree`**

Still in `tree.py`, update `_attach_op_subtree` (currently lines 236-262). Change the final `tree.add_node(ISANode(...))` call to:

```python
    tree.add_node(
        ISANode(
            op_cls=op.op_cls,
            reads=tuple(reads),
            writes=tuple(writes),
            rmw=tuple(rmw),
            tensorize_sizes=tensorize_sizes,
            kwargs=dict(op.kwargs),
        ),
        parent=parent,
    )
```

Wrapping `op.kwargs` in `dict(...)` defensively copies so the frozen `ISANode` doesn't share mutable state with the tracer.

- [ ] **Step 1.8: Run the IR tests to verify they pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/test_ir_extensions.py -v
```

Expected: all three `kwargs` tests pass.

- [ ] **Step 1.9: Run the whole test suite to confirm no regression**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 1.10: Commit**

```bash
git add test/ir/__init__.py test/ir/test_ir_extensions.py \
        nkigym/src/nkigym/ir/dimension_analysis.py \
        nkigym/src/nkigym/ir/tree.py
git commit -m "$(cat <<'EOF'
Capture per-op call kwargs on ISANode

Add kwargs field to OpAxes and ISANode; the tracer now records every
non-operand kwarg (value=, op=, scale=, ...) so render() can emit the
corresponding nisa.<NAME>(..., kwarg=value) call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Thread `axis_map` onto `ISANode`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (add `axis_map` field to `ISANode`, thread from `OpAxes.axis_map`)
- Modify: `test/ir/test_ir_extensions.py` (new test)

---

- [ ] **Step 2.1: Add failing test for axis_map on compute leaves**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_matmul_leaf_carries_axis_map():
    """NKIMatmul leaf axis_map resolves every abstract axis to a concrete dim."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    matmuls = _leaves_by_op(ir, "NKIMatmul")
    assert len(matmuls) == 1
    axis_map = matmuls[0].axis_map
    assert set(axis_map) == {"K", "M", "N"}
    assert all(isinstance(v, str) and v.startswith("d") for v in axis_map.values())


def test_alloc_leaves_have_empty_axis_map():
    """NKIAlloc leaves carry no axis_map — their per-axis tiles live on tensorize_sizes."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        assert leaf.axis_map == {}
```

- [ ] **Step 2.2: Run to verify the new tests fail**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/test_ir_extensions.py -v
```

Expected: the two new tests fail with `AttributeError: 'ISANode' object has no attribute 'axis_map'`.

- [ ] **Step 2.3: Add `axis_map` field to `ISANode`**

In `nkigym/src/nkigym/ir/tree.py`, extend the `ISANode` dataclass (from Task 1):

```python
@dataclass(frozen=True, kw_only=True)
class ISANode:
    """NKI-instruction payload.

    Attributes:
        op_cls: The :class:`NKIOp` subclass (e.g. ``NKILoad``).
        reads: Tensor names from slots in ``op_cls.INPUT_OPERANDS``.
        writes: Tensor names from slots that are neither input nor RMW.
        rmw: Tensor names from slots in ``op_cls.RMW_OPERANDS``.
        tensorize_sizes: Per-axis tile width (``dim → tile_size``)
            lowered onto the ISA call's slice width. Keys mirror the
            ``dim`` of the enclosing :class:`ForNode` chain; for
            :class:`NKIAlloc` leaves (no surrounding loops) this is
            empty.
        axis_map: ``abstract_axis → concrete_dim`` (e.g. ``{"K": "d0"}``).
            Render consults this alongside ``op_cls.OPERAND_AXES`` to
            resolve each slot's axis labels to concrete dim ids.
            Empty for :class:`NKIAlloc` leaves.
        kwargs: Non-operand call kwargs captured from the tracer
            (e.g. ``{"value": 0.0}`` for ``NKIMemset``,
            ``{"op": "rsqrt", "scale": 1.0}`` for ``NKIActivation``).
            Empty for :class:`NKIAlloc` leaves.
    """

    op_cls: type[NKIOp]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    rmw: tuple[str, ...] = ()
    tensorize_sizes: dict[str, int] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2.4: Thread `axis_map` through `_attach_op_subtree`**

Still in `tree.py`, update the final `ISANode(...)` construction in `_attach_op_subtree`:

```python
    tree.add_node(
        ISANode(
            op_cls=op.op_cls,
            reads=tuple(reads),
            writes=tuple(writes),
            rmw=tuple(rmw),
            tensorize_sizes=tensorize_sizes,
            axis_map=dict(op.axis_map),
            kwargs=dict(op.kwargs),
        ),
        parent=parent,
    )
```

- [ ] **Step 2.5: Verify tests pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: all five IR tests pass.

- [ ] **Step 2.6: Run the full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 2.7: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py
git commit -m "$(cat <<'EOF'
Thread axis_map onto ISANode

Render() needs abstract-to-concrete dim lookup per op leaf to
synthesise slice expressions from op_cls.OPERAND_AXES. Migrate
axis_map from OpAxes onto ISANode so the tree is self-describing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Capture tensor `location` and `dtype` on `TensorDims`

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py` (`TensorDims` fields, `_Sym` fields, hook capture, materialisation)
- Modify: `test/ir/test_ir_extensions.py` (new tests)

---

- [ ] **Step 3.1: Add failing tests for tensor location and dtype**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_intermediate_tensor_location_and_dtype():
    """NKIAlloc kwargs flow through to TensorDims.location / .dtype."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    expected = {
        "sbuf_lhs_T": ("sbuf", "bfloat16"),
        "sbuf_rhs": ("sbuf", "bfloat16"),
        "psum_acc": ("psum", "float32"),
        "sbuf_prod": ("sbuf", "bfloat16"),
        "hbm_out": ("hbm", "bfloat16"),
    }
    for name, (loc, dt) in expected.items():
        t = ir.analysis.tensors[name]
        assert t.location == loc, f"{name}.location={t.location!r}"
        assert t.dtype == dt, f"{name}.dtype={t.dtype!r}"


def test_param_tensor_location_and_dtype_are_none():
    """Kernel params arrive via the function signature; they have no alloc."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for name in ("lhs_T", "rhs"):
        t = ir.analysis.tensors[name]
        assert t.location is None
        assert t.dtype is None
```

(These tests reference `ir.analysis.tensors` — the current shape. Task 5 flips them to `ir.tensors`; do not rename them here.)

- [ ] **Step 3.2: Run to verify new tests fail**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: new tests fail with `AttributeError: 'TensorDims' object has no attribute 'location'`.

- [ ] **Step 3.3: Add `location` and `dtype` to `TensorDims`**

Edit `nkigym/src/nkigym/ir/dimension_analysis.py`. Change the `TensorDims` dataclass (currently lines 17-30):

```python
@dataclass
class TensorDims:
    """Per-tensor dim-unification result.

    Attributes:
        name: Source-level variable name.
        shape: Per-dim extents, aligned with ``dim_ids``.
        dim_ids: Concrete dim names (``d0``, ``d1`` ...).
        location: ``"hbm"`` / ``"sbuf"`` / ``"psum"`` for tensors
            declared via :class:`NKIAlloc`; ``None`` for kernel params.
        dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"`` for
            tensors declared via :class:`NKIAlloc`; ``None`` for
            kernel params.
    """

    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]
    location: str | None
    dtype: str | None
```

- [ ] **Step 3.4: Stash alloc kwargs on `_Sym`**

Still in `dimension_analysis.py`, update the `_Sym` class (currently lines 119-127):

```python
class _Sym:
    """Symbolic tensor: shape + mutable ``dim_ids`` + source name + alloc kwargs."""

    __slots__ = ("shape", "dim_ids", "source_name", "location", "dtype")

    def __init__(self, shape: tuple[int, ...], source_name: str) -> None:
        self.shape: tuple[int, ...] = shape
        self.dim_ids: list[str | None] = [None] * len(shape)
        self.source_name: str = source_name
        self.location: str | None = None
        self.dtype: str | None = None
```

- [ ] **Step 3.5: Capture `location` and `dtype` in the trace hook**

In `dimension_analysis.py`, update the `NKIAlloc` branch of `_make_hook` (currently lines 165-173):

```python
    def hook(op: NKIOp, **kwargs: Any) -> Any:
        merged = {**getattr(op, "_init_kwargs", {}), **kwargs}
        cls = type(op)
        if cls is NKIAlloc:
            name = next(state.alloc_names)
            sym = _Sym(tuple(merged["shape"]), name)
            sym.location = merged["location"]
            sym.dtype = merged["dtype"]
            state.sentinels[name] = sym
            return sym
        _trace_compute_op(cls, merged, state)
        return merged.get("dst")
```

- [ ] **Step 3.6: Thread `location` and `dtype` into `TensorDims`**

In `dimension_analysis.py`, update the materialisation loop in `analyze_dimensions` (currently lines 103-110):

```python
    tensors: dict[str, TensorDims] = {}
    for sym in state.sentinels.values():
        if any(d is None for d in sym.dim_ids):
            raise ValueError(f"Tensor {sym.source_name!r} has un-unified dims: {sym.dim_ids}")
        tensors[sym.source_name] = TensorDims(
            name=sym.source_name,
            shape=sym.shape,
            dim_ids=tuple(d for d in sym.dim_ids if d is not None),
            location=sym.location,
            dtype=sym.dtype,
        )
```

- [ ] **Step 3.7: Verify tests pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: all seven tests pass.

- [ ] **Step 3.8: Run the full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 3.9: Commit**

```bash
git add nkigym/src/nkigym/ir/dimension_analysis.py test/ir/test_ir_extensions.py
git commit -m "$(cat <<'EOF'
Capture tensor location and dtype on TensorDims

NKIAlloc declares (location, shape, dtype); the tracer used to keep
only shape. Render needs location to pick nl.sbuf/nl.psum/nl.shared_hbm
and dtype to pick nl.<dtype> for the emitted nl.ndarray.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Parse the kernel return tensor name

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py` (add `return_name` to `DimensionAnalysis`, parse helper)
- Modify: `test/ir/test_ir_extensions.py` (positive + negative tests)

---

- [ ] **Step 4.1: Add failing tests for return-name parsing**

Append to `test/ir/test_ir_extensions.py`:

```python
import pytest

from nkigym.ir.dimension_analysis import analyze_dimensions


def test_return_name_is_parsed():
    """analyse_dimensions captures the top-level ``return <Name>`` identifier."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert ir.analysis.return_name == "hbm_out"


def test_missing_return_raises():
    """A kernel that does not end with ``return <Name>`` fails analysis."""

    @nkigym_kernel
    def no_return(x):
        sbuf_x = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        NKILoad()(src=x, dst=sbuf_x)

    with pytest.raises(ValueError, match="return"):
        analyze_dimensions(no_return, {"x": (128, 128)})
```

- [ ] **Step 4.2: Run to verify they fail**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: `test_return_name_is_parsed` fails with `AttributeError: 'DimensionAnalysis' object has no attribute 'return_name'`; `test_missing_return_raises` may fail for the same reason or because no validation happens yet.

- [ ] **Step 4.3: Add the AST helper**

In `nkigym/src/nkigym/ir/dimension_analysis.py`, add a helper near the other AST helpers (alongside `_collect_alloc_names` / `_is_alloc_call`):

```python
def _parse_return_name(func: Callable[..., Any]) -> str:
    """Return the identifier named in the kernel's single top-level ``return`` statement.

    Raises ``ValueError`` if the function has no ``return`` statement, has
    more than one, or returns an expression that is not a single ``Name``.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    returns = [stmt for stmt in func_def.body if isinstance(stmt, ast.Return)]
    if len(returns) == 0:
        raise ValueError(f"{func.__name__}: no top-level return statement")
    if len(returns) > 1:
        raise ValueError(f"{func.__name__}: expected a single top-level return, found {len(returns)}")
    value = returns[0].value
    if not isinstance(value, ast.Name):
        raise ValueError(f"{func.__name__}: return value must be a bare Name, got {type(value).__name__}")
    return value.id
```

- [ ] **Step 4.4: Add `return_name` to `DimensionAnalysis` and populate it**

Change the `DimensionAnalysis` dataclass (currently lines 47-80):

```python
@dataclass
class DimensionAnalysis:
    """Output of :func:`analyze_dimensions`.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent`` (e.g. ``{"d0": 2048}``).
        tensors: All named tensors, keyed by name.
        ops: Compute ops in source order.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    ops: list[OpAxes]

    def __repr__(self) -> str:
        """Multi-line dump: signature + dims + tensors + op calls."""
        lines: list[str] = [
            f"DimensionAnalysis {self.func_name}({', '.join(self.param_names)}) -> {self.return_name}",
            "dims:",
        ]
        for name, size in self.dim_sizes.items():
            lines.append(f"  {name} size={size}")
        lines.append("tensors:")
        for t in self.tensors.values():
            shape_parts = [f"{d}={sz}" for d, sz in zip(t.dim_ids, t.shape)]
            shape_str = f"[{', '.join(shape_parts)}]" if shape_parts else "[]"
            lines.append(f"  {t.name}: shape={shape_str}")
        lines.append("ops:")
        for op in self.ops:
            operands = ",".join(f"{slot}={name}" for slot, name in op.operand_names.items())
            axes = ",".join(f"{k}={v}" for k, v in op.axis_map.items())
            lines.append(f"  {op.op_cls.__name__}({operands}) axes={{{axes}}}")
        return "\n".join(lines)
```

- [ ] **Step 4.5: Wire `return_name` into `analyze_dimensions`**

At the end of `analyze_dimensions`, update the `DimensionAnalysis(...)` construction (currently lines 110-116):

```python
    return DimensionAnalysis(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=_parse_return_name(unwrapped),
        dim_sizes=state.dim_sizes,
        tensors=tensors,
        ops=state.op_records,
    )
```

- [ ] **Step 4.6: Verify all IR tests pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: both new tests pass.

- [ ] **Step 4.7: Run the full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 4.8: Commit**

```bash
git add nkigym/src/nkigym/ir/dimension_analysis.py test/ir/test_ir_extensions.py
git commit -m "$(cat <<'EOF'
Parse the kernel's return tensor name

Render() needs to emit 'return <name>' as the final line of the
generated function. Add _parse_return_name AST helper and stamp the
result onto DimensionAnalysis.return_name. Rejects kernels with no
return, multiple returns, or non-Name return expressions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Flatten `DimensionAnalysis` into `KernelIR`

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py` (delete `DimensionAnalysis` dataclass; rename `OpAxes` → `_OpRecord`; introduce `_AnalysisResult`)
- Modify: `nkigym/src/nkigym/ir/tree.py` (drop `KernelTree.dim_sizes`; update `build_initial_tree` and `_attach_op_subtree` signatures; update `__all__`)
- Modify: `nkigym/src/nkigym/ir/ir.py` (flat `KernelIR` fields; rewrite `build_initial_ir`)
- Modify: `test/ir/test_ir_extensions.py` (flip every `ir.analysis.<x>` to `ir.<x>`; add coverage for envelope flatness and `KernelTree.dim_sizes` removal)

---

- [ ] **Step 5.1: Update tests to the target envelope shape (they should fail first)**

Replace every `ir.analysis.tensors[...]` and `ir.analysis.<field>` in `test/ir/test_ir_extensions.py` with the flat form. Concretely, change:

```python
t = ir.analysis.tensors[name]
```

to:

```python
t = ir.tensors[name]
```

And change:

```python
assert ir.analysis.return_name == "hbm_out"
```

to:

```python
assert ir.return_name == "hbm_out"
```

Then append these new tests:

```python
def test_envelope_fields_are_flat():
    """KernelIR exposes signature + dim_sizes + tensors directly — no .analysis wrapper."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert ir.func_name == "_matmul_fixture"
    assert ir.param_names == ["lhs_T", "rhs"]
    assert ir.return_name == "hbm_out"
    assert set(ir.tensors) == {
        "lhs_T", "rhs", "sbuf_lhs_T", "sbuf_rhs", "psum_acc", "sbuf_prod", "hbm_out",
    }
    assert set(ir.dim_sizes.values()) == {K, M, N}
    assert not hasattr(ir, "analysis")
    assert not hasattr(ir, "ops")


def test_tree_has_no_dim_sizes_attribute():
    """KernelTree is a pure schedule tree — dim_sizes lives on KernelIR."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert not hasattr(ir.tree, "dim_sizes")
```

- [ ] **Step 5.2: Run to confirm the tests fail in the right way**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: assertions against `ir.<field>` fail with `AttributeError: 'KernelIR' object has no attribute 'func_name'`; the `not hasattr(ir.tree, "dim_sizes")` assertion fails because the attribute is still present.

- [ ] **Step 5.3: Rewrite `dimension_analysis.py`**

Replace the public `DimensionAnalysis`/`OpAxes` pair with a private `_AnalysisResult` / `_OpRecord` pair. Full replacement (top of file — keep the existing imports and `_Sym`/`_TraceState`/`_unify`/`_canonicalize_dim_names`/`_apply_rename`/`_collect_alloc_names`/`_is_alloc_call`/`_parse_return_name` helpers exactly as they are, modifying only the dataclasses and the public function):

```python
"""Dim unification for an ``f_nkigym`` callable via symbolic tracing.

Entry point: :func:`analyze_dimensions`. Returns a private
:class:`_AnalysisResult` consumed by :func:`build_initial_ir`.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp


@dataclass
class TensorDims:
    """Per-tensor dim-unification result.

    Attributes:
        name: Source-level variable name.
        shape: Per-dim extents, aligned with ``dim_ids``.
        dim_ids: Concrete dim names (``d0``, ``d1`` ...).
        location: ``"hbm"`` / ``"sbuf"`` / ``"psum"`` for tensors
            declared via :class:`NKIAlloc`; ``None`` for kernel params.
        dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"`` for
            tensors declared via :class:`NKIAlloc`; ``None`` for
            kernel params.
    """

    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]
    location: str | None
    dtype: str | None


@dataclass
class _OpRecord:
    """Per-op tracer record. Private — consumed by ``build_initial_tree``.

    Attributes:
        op_cls: The NKIOp subclass.
        operand_names: ``slot → tensor_name`` for every operand in the call.
        axis_map: ``abstract_axis → concrete_dim``.
        kwargs: Non-operand call kwargs.
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    axis_map: dict[str, str]
    kwargs: dict[str, Any]


@dataclass
class _AnalysisResult:
    """Private hand-off from ``analyze_dimensions`` to ``build_initial_ir``.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent``.
        tensors: All named tensors, keyed by name.
        ops: Compute ops in source order.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    ops: list[_OpRecord]
```

Then update `analyze_dimensions` to return `_AnalysisResult`:

```python
def analyze_dimensions(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> _AnalysisResult:
    """Trace ``func`` against sentinel inputs and run cross-op dim unification."""
    unwrapped = inspect.unwrap(func)
    param_names = list(inspect.signature(unwrapped).parameters)
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    state = _TraceState(alloc_names=_collect_alloc_names(unwrapped))
    for name in param_names:
        state.sentinels[name] = _Sym(tuple(input_specs[name]), name)

    _run_trace(unwrapped, [state.sentinels[n] for n in param_names], state)
    _canonicalize_dim_names(state)

    tensors: dict[str, TensorDims] = {}
    for sym in state.sentinels.values():
        if any(d is None for d in sym.dim_ids):
            raise ValueError(f"Tensor {sym.source_name!r} has un-unified dims: {sym.dim_ids}")
        tensors[sym.source_name] = TensorDims(
            name=sym.source_name,
            shape=sym.shape,
            dim_ids=tuple(d for d in sym.dim_ids if d is not None),
            location=sym.location,
            dtype=sym.dtype,
        )
    return _AnalysisResult(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=_parse_return_name(unwrapped),
        dim_sizes=state.dim_sizes,
        tensors=tensors,
        ops=state.op_records,
    )
```

Update `_TraceState.op_records` type annotation and `_trace_compute_op`'s final append to use `_OpRecord`:

```python
class _TraceState:
    """Mutable state threaded through the hook during tracing."""

    def __init__(self, alloc_names: Iterator[str]) -> None:
        self.sentinels: dict[str, _Sym] = {}
        self.dim_sizes: dict[str, int] = {}
        self.op_records: list[_OpRecord] = []
        self.alloc_names = alloc_names
        self.next_dim = 0
    ...
```

```python
def _trace_compute_op(cls: type[NKIOp], kwargs: dict[str, Any], state: _TraceState) -> None:
    """Unify a compute op's operands and record an :class:`_OpRecord` entry."""
    local: dict[str, str] = {}
    operand_names: dict[str, str] = {}
    for slot, axes in cls.OPERAND_AXES.items():
        sym = kwargs.get(slot)
        if not isinstance(sym, _Sym):
            continue
        operand_names[slot] = sym.source_name
        for i, abstract in enumerate(axes[: len(sym.shape)]):
            existing = sym.dim_ids[i]
            if existing is None:
                if abstract not in local:
                    local[abstract] = state.fresh_dim(sym.shape[i])
                sym.dim_ids[i] = local[abstract]
            elif abstract in local and local[abstract] != existing:
                _unify(existing, local[abstract], state, local)
            else:
                local[abstract] = existing
    op_kwargs = {k: v for k, v in kwargs.items() if k not in cls.OPERAND_AXES}
    state.op_records.append(
        _OpRecord(op_cls=cls, operand_names=operand_names, axis_map=local, kwargs=op_kwargs)
    )
```

Update `_apply_rename` to iterate `_OpRecord`s (it already does — the field name is `axis_map` on both; just verify):

```python
def _apply_rename(state: _TraceState, remap: dict[str, str]) -> None:
    """Substitute every dim id in sentinels and op records via ``remap``."""
    for sym in state.sentinels.values():
        sym.dim_ids = [remap.get(d, d) if d is not None else None for d in sym.dim_ids]
    for rec in state.op_records:
        for abstract in rec.axis_map:
            rec.axis_map[abstract] = remap.get(rec.axis_map[abstract], rec.axis_map[abstract])
```

- [ ] **Step 5.4: Rewrite `tree.py` to drop `dim_sizes` and accept `_AnalysisResult`**

Edit `nkigym/src/nkigym/ir/tree.py`.

Update the import line near the top (currently line 39):

```python
from nkigym.ir.dimension_analysis import TensorDims, _AnalysisResult, _OpRecord
```

Update `KernelTree.__init__` (currently lines 112-117) to drop the `dim_sizes` parameter:

```python
    def __init__(self) -> None:
        """Initialise an empty tree."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self.root: int = self.add_node(RootNode())
```

Also drop the `dim_sizes` attribute from the class docstring (lines 98-110) — replace the docstring with:

```python
class KernelTree:
    """Schedule tree stored as an ``nx.DiGraph`` of integer node ids.

    Edges point parent → child. Child order is the networkx
    successor order (insertion order on ``DiGraph``), which matches
    source order because children are added sequentially.

    Attributes:
        graph: The underlying ``nx.DiGraph``. Node payloads live at
            ``graph.nodes[nid]["data"]``.
        root: Node id of the forest root (a :class:`RootNode`).
    """
```

Update `build_initial_tree` (currently lines 185-214) to consume `_AnalysisResult` directly:

```python
def build_initial_tree(analysis: _AnalysisResult) -> KernelTree:
    """Build the canonical schedule tree from an :class:`_AnalysisResult`.

    Alloc leaves (every :class:`NKIAlloc`) sit as direct children of the
    root in declaration order — allocation is a whole-tensor statement,
    so no surrounding trip loop is emitted. Each alloc still carries
    ``tensorize_sizes`` keyed by its concrete dim ids, derived from the
    tensor's shape and :attr:`NKIAlloc.MAX_TILE_SIZE` so downstream atoms
    see a uniform per-leaf tile map. Each compute op gets its own
    per-axis loop nest; loops appear outermost-to-innermost in the op's
    ``axis_map`` iteration order (which mirrors ``OPERAND_AXES``).
    """
    tree = KernelTree()
    param_names = set(analysis.param_names)
    for name, tensor in analysis.tensors.items():
        if name in param_names:
            continue
        tree.add_node(
            ISANode(op_cls=NKIAlloc, writes=(name,), tensorize_sizes=_alloc_tensorize_sizes(tensor)),
            parent=tree.root,
        )
    for op in analysis.ops:
        _attach_op_subtree(tree, op, analysis.dim_sizes)
    return tree
```

Update `_attach_op_subtree` (currently lines 236-262) to take `dim_sizes` explicitly:

```python
def _attach_op_subtree(tree: KernelTree, op: _OpRecord, dim_sizes: dict[str, int]) -> None:
    """Attach one compute-op loop nest under ``tree.root``."""
    reads: list[str] = []
    writes: list[str] = []
    rmw: list[str] = []
    for slot, tensor_name in op.operand_names.items():
        if slot in op.op_cls.INPUT_OPERANDS:
            reads.append(tensor_name)
        elif slot in op.op_cls.RMW_OPERANDS:
            rmw.append(tensor_name)
        else:
            writes.append(tensor_name)
    parent = tree.root
    tensorize_sizes = {}
    for abstract, concrete in op.axis_map.items():
        extent = dim_sizes[concrete]
        max_tile = op.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        role = op.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
        parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile, loop_type=role), parent=parent)
        tensorize_sizes[concrete] = tile
    tree.add_node(
        ISANode(
            op_cls=op.op_cls,
            reads=tuple(reads),
            writes=tuple(writes),
            rmw=tuple(rmw),
            tensorize_sizes=tensorize_sizes,
            axis_map=dict(op.axis_map),
            kwargs=dict(op.kwargs),
        ),
        parent=parent,
    )
```

Update the module `__all__` (currently line 301) to drop `DimensionAnalysis`:

```python
__all__ = ["ForNode", "ISANode", "KernelTree", "NodeData", "RootNode", "build_initial_tree"]
```

- [ ] **Step 5.5: Rewrite `ir.py` with flat `KernelIR` fields**

Replace `nkigym/src/nkigym/ir/ir.py` entirely:

```python
"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` is the single envelope. It carries the kernel
signature, return-tensor identity, per-dim sizes, tensor table,
canonical schedule tree, and the producer-consumer dependency graph
that rewrite atoms consult.

:func:`build_initial_ir` runs dim unification, tree construction, and
dependency graph construction, then flattens the analysis output onto
a :class:`KernelIR` instance. :meth:`KernelIR.dump` writes the tree
and dependency diagrams side-by-side into a cache directory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import TensorDims, analyze_dimensions
from nkigym.ir.tree import KernelTree, build_initial_tree


@dataclass
class KernelIR:
    """Envelope holding signature, tensor table, schedule tree, and dependency graph.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent``.
        tensors: All named tensors, keyed by name.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    tree: KernelTree
    dependency: Dependency

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``tree.*`` and ``dependency.*`` diagrams into ``cache_dir``."""
        self.tree.dump(cache_dir)
        self.dependency.dump(cache_dir)


def build_initial_ir(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> KernelIR:
    """Run dim analysis, build the schedule tree, derive the dependency graph, flatten.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: shape}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_initial_tree(analysis)
    dependency = Dependency(tree)
    return KernelIR(
        func_name=analysis.func_name,
        param_names=analysis.param_names,
        return_name=analysis.return_name,
        dim_sizes=analysis.dim_sizes,
        tensors=analysis.tensors,
        tree=tree,
        dependency=dependency,
    )


__all__ = ["KernelIR", "build_initial_ir"]
```

- [ ] **Step 5.6: Update the IR package `__init__.py` to stay consistent**

Edit `nkigym/src/nkigym/ir/__init__.py` to drop dead re-exports — the full exports cleanup happens in Task 6, but `DimensionAnalysis` and `analyze_dimensions` no longer exist, so remove them now to keep the package importable:

```python
"""Dim unification analysis + canonical schedule tree for an ``f_nkigym`` callable."""

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import TensorDims
from nkigym.ir.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ForNode, ISANode, KernelTree, NodeData, RootNode, build_initial_tree

__all__ = [
    "Dependency",
    "ForNode",
    "ISANode",
    "KernelIR",
    "KernelTree",
    "NodeData",
    "RootNode",
    "TensorDims",
    "build_initial_ir",
    "build_initial_tree",
]
```

- [ ] **Step 5.7: Verify all IR tests pass**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: every test passes, including the two new flatness / tree-has-no-dim_sizes tests.

- [ ] **Step 5.8: Run the full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 5.9: Verify the example script still runs**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
```

Expected: prints `[numerics] PASS (...)`, dumps tree + dependency PNGs into `/home/ubuntu/cache/matmul_lhsT_rhs/`.

- [ ] **Step 5.10: Commit**

```bash
git add nkigym/src/nkigym/ir/dimension_analysis.py \
        nkigym/src/nkigym/ir/tree.py \
        nkigym/src/nkigym/ir/ir.py \
        nkigym/src/nkigym/ir/__init__.py \
        test/ir/test_ir_extensions.py
git commit -m "$(cat <<'EOF'
Flatten DimensionAnalysis into KernelIR

KernelIR now carries func_name / param_names / return_name / dim_sizes /
tensors directly; the analysis wrapper is gone. OpAxes is privatised as
_OpRecord (tracer-local). KernelTree no longer stores dim_sizes —
build_initial_tree reads it from the private _AnalysisResult and passes
it down to _attach_op_subtree.

Single canonical home per field: KernelIR owns signature + per-dim
metadata; KernelTree is a pure schedule tree; ISANode holds every
per-op field (kwargs, axis_map, tensorize_sizes, reads/writes/rmw).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Tighten package exports and delete dead re-exports

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (`TensorDims` import is unused at module level once dataclass fields are typed — verify and, if unused, remove)
- Modify: `test/ir/test_ir_extensions.py` (negative import tests)

Task 5 already pruned the package-level `__init__.py`. This task adds explicit negative-import tests so the surface stays tight, and removes the now-dead `DimensionAnalysis` reference in `tree.py`'s `__all__` history if any remains.

---

- [ ] **Step 6.1: Add failing negative-import tests**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_removed_symbols_are_not_exported_from_package():
    """DimensionAnalysis / OpAxes / analyze_dimensions must not be package-level imports."""
    import nkigym.ir as ir_pkg

    for removed in ("DimensionAnalysis", "OpAxes", "analyze_dimensions"):
        assert not hasattr(ir_pkg, removed), f"nkigym.ir.{removed} should have been removed"


def test_op_record_is_private_in_dimension_analysis_module():
    """_OpRecord is the tracer-local replacement for the old OpAxes — it must not be re-exported."""
    import nkigym.ir.dimension_analysis as dm

    assert not hasattr(dm, "OpAxes")
    assert hasattr(dm, "_OpRecord")
```

- [ ] **Step 6.2: Run to verify — they should already pass after Task 5**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ir/ -v
```

Expected: new tests pass.

- [ ] **Step 6.3: Scan for leftover references to deleted symbols**

```bash
grep -rn "DimensionAnalysis\|OpAxes\|analyze_dimensions" nkigym/ test/ examples/ 2>/dev/null | grep -v __pycache__
```

Expected: the only hits are inside `nkigym/src/nkigym/ir/dimension_analysis.py` itself (the `analyze_dimensions` function definition). No stale imports anywhere else.

If a stale reference shows up (e.g. an unused import in `tree.py`), delete it with a single-line edit.

- [ ] **Step 6.4: Run the full suite one more time**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest test/ -v
```

Expected: every test passes.

- [ ] **Step 6.5: Commit**

```bash
git add test/ir/test_ir_extensions.py
git commit -m "$(cat <<'EOF'
Lock in the slimmed-down nkigym.ir surface with negative tests

DimensionAnalysis / OpAxes / analyze_dimensions are gone at the
package level after the flatten; _OpRecord is tracer-local. Tests pin
the public surface so future refactors can't accidentally re-expose
them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**

- Change 1 (`KernelIR` absorbs `DimensionAnalysis`) → Task 5.
- Change 2 (`TensorDims.location` / `.dtype`) → Task 3.
- Change 3 (`ISANode.kwargs` + `axis_map`) → Tasks 1 and 2.
- Change 4 (`OpAxes` → `_OpRecord`, tracer-local) → Task 5.
- Change 5 (drop `KernelTree.dim_sizes`) → Task 5.
- Tracer / build-pipeline plumbing rows → Tasks 1 (op kwargs), 3 (alloc kwargs, materialisation), 4 (return-name parse), 5 (pipeline return + tree builder signature + envelope assembly).
- Public surface changes → `__init__.py` pruned in Task 5.6, negative tests in Task 6.1.
- Every test in the spec's Testing section has a matching `- [ ]` step above (envelope flattening, tensor fields, per-leaf kwargs + axis_map, export surface, negative return-statement case).

**Placeholder scan:** every step has exact file paths, exact commands, exact expected output, and complete code blocks. No "TBD" / "add appropriate validation" / "similar to Task N".

**Type consistency:** `_OpRecord` fields (`op_cls`, `operand_names`, `axis_map`, `kwargs`) match across Task 5's `dimension_analysis.py` rewrite, Task 5's `tree.py` `_attach_op_subtree` signature, and the Task 1 / Task 2 `ISANode` field list. `KernelIR` fields in Task 5's `ir.py` rewrite match the test assertions in Task 5.1 (`func_name`, `param_names`, `return_name`, `dim_sizes`, `tensors`, `tree`, `dependency`).
