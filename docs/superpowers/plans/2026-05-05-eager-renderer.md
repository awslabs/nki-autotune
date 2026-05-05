# Eager Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `nkigym/kernel_ir/` + knob-driven `nkigym/codegen/render.py` with a two-module eager renderer that turns any validated `f_nkigym` callable into an `@nki.jit` NKI kernel where each `NKIOp` owns an independent loop nest over only the dims it touches.

**Architecture:** `nkigym/codegen/graph.py` (AST → `OpGraph`) + `nkigym/codegen/render.py` (`OpGraph` → source string). Single public entry: `render_eager(func, input_specs)`. SBUF intermediates are full-extent 3D `(p_tile, num_p_tiles, num_f_tiles*f_tile)` allocations hoisted to function top. HBM appears only at `NKILoad` (kernel inputs) and `NKIStore` (kernel output). No knobs, no rewrites, no fusion — strictly the math DAG written out tile-by-tile.

**Tech Stack:** Python 3.10+, `nki`, `nki.isa`, `nki.language`, `ast`, `pytest`. Runtime environment: `source ~/venvs/kernel-env/bin/activate`.

---

## Reference map (read these before starting)

- Design spec: `docs/superpowers/specs/2026-05-05-eager-renderer-design.md`
- Target example: `examples/rmsnorm_matmul.py` + `examples/rmsnorm_matmul.md` (the kernel walkthrough we want the renderer to approximate — eager version will have one nest per op, not the fused two-pass version)
- NKIOp base + decorator: `nkigym/src/nkigym/ops/base.py`
- Op definitions to support: `nkigym/src/nkigym/ops/{load,store,matmul,transpose,dma_transpose,activation,activation_reduce,tensor_scalar}.py`
- Source of dim-unification logic to port: `nkigym/src/nkigym/kernel_ir/build.py:_build_axis_map` + `_unify_dim` + `_resolve_dimensions` + `_create_outputs`
- Source of AST walk to port: `nkigym/src/nkigym/kernel_ir/parse.py:find_ops`
- CPU sim entry point: `autotune/src/autotune/runner/benchmark.py:simulate_one` — calls `nki.simulate(kernel_func)(**kwargs)` after rewriting every `dtype=nl.<name>` to `nl.float32`. Tests can import `nki.simulate` directly.
- Project conventions: `/home/ubuntu/.claude/rules/code_style.md` (docstrings, triple-quoted block comments instead of `#`, 120-char lines, Google/NumPy docstring style, modern type hints).
- Pytest config: root `pyproject.toml` has `pythonpath = ["nkigym/src"]`; tests go anywhere pytest auto-discovers.

## File structure

**New files (create):**
- `nkigym/src/nkigym/codegen/graph.py` — `OpGraph`, `ParsedOp`, `Tensor`, `DimInfo` dataclasses + `parse_and_resolve(func, input_specs) -> OpGraph`. ~200 lines.
- `nkigym/src/nkigym/codegen/render.py` — `render(op_graph) -> str`. ~450 lines with per-op emitters.
- `test/codegen/test_graph.py` — dim unification + parse coverage.
- `test/codegen/test_render.py` — golden-string checks per op kind + end-to-end smoke tests via `nki.simulate`.
- `test/codegen/__init__.py` — empty package marker.
- `test/__init__.py` — empty package marker.

**Rewrite:**
- `nkigym/src/nkigym/codegen/__init__.py` — exports `render_eager` only.

**Delete:**
- `nkigym/src/nkigym/kernel_ir/` — entire subtree.
- `nkigym/src/nkigym/codegen/gadgets.py`.
- Existing `nkigym/src/nkigym/codegen/render.py` (fully replaced).
- `nkigym/src/nkigym/search/` — entire subtree (every import resolves to deleted `kernel_ir` / `render_ir`; caller sites are in doomed examples).
- `nkigym/src/nkigym/ops/online_flash_attention.py`, `nkigym/src/nkigym/ops/online_matmul.py` (unused after rewrites are removed).
- `examples/double_matmul.py`, `examples/attention.py`, `examples/attention_online_fused.py`, `examples/rmsnorm_matmul_online.py`, `examples/rmsnorm_matmul_online_fused.py` (rely on knob-driven IR or online fusion; out of scope).
- `examples/matmul_lhsT_rhs.md`, `examples/matmul_lhs_rhs.md`, `examples/rmsnorm_matmul.md` (walkthroughs describe the deleted knob-IR).

**Port:**
- `examples/rmsnorm_matmul.py` — extend to render + CPU-sim.
- `examples/matmul_lhsT_rhs.py` — switch to `render_eager` + CPU-sim.
- `examples/matmul_lhs_rhs.py` — switch to `render_eager` + CPU-sim.

---

## Task 0: Create worktree + scaffolding

**Files:**
- Create: `test/__init__.py`
- Create: `test/codegen/__init__.py`

- [ ] **Step 1: Verify venv + working tree**

Run: `source ~/venvs/kernel-env/bin/activate && python -c "import nki, nki.simulate; print('ok')"`
Expected: `ok`.

Run: `git status` — expected to be on branch `dev_1` with only the in-flight `examples/*.py` modifications from the base workspace.

- [ ] **Step 2: Create empty test package markers**

```bash
mkdir -p test/codegen
```

Create `test/__init__.py`:

```python
```

Create `test/codegen/__init__.py`:

```python
```

- [ ] **Step 3: Commit scaffolding**

```bash
git add test/__init__.py test/codegen/__init__.py
git commit -m "Scaffold test/codegen package for eager-renderer tests"
```

---

## Task 1: Define OpGraph dataclasses

**Files:**
- Create: `nkigym/src/nkigym/codegen/graph.py`
- Create: `test/codegen/test_graph.py`

- [ ] **Step 1: Write failing test for dataclass shape**

Create `test/codegen/test_graph.py`:

```python
"""Tests for OpGraph dataclasses and dim resolution."""

from nkigym.codegen.graph import DimInfo, OpGraph, ParsedOp, Tensor


def test_tensor_fields() -> None:
    """Tensor carries name, dim_ids, shape, dtype, origin."""
    t = Tensor(name="lhs", dim_ids=("d0", "d1"), shape=(2048, 2048), dtype="bfloat16", origin="param")
    assert t.name == "lhs"
    assert t.dim_ids == ("d0", "d1")
    assert t.shape == (2048, 2048)
    assert t.dtype == "bfloat16"
    assert t.origin == "param"


def test_dim_info_fields() -> None:
    """DimInfo carries dim_id, total_size, tile_size, num_tiles."""
    d = DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)
    assert d.num_tiles == 16


def test_op_graph_empty_per_op_attrs() -> None:
    """OpGraph initialises per_op_attrs as an empty dict."""
    g = OpGraph(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={},
        dims={},
        ops=[],
        per_op_attrs={},
    )
    assert g.per_op_attrs == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest test/codegen/test_graph.py -v`
Expected: `ModuleNotFoundError: No module named 'nkigym.codegen.graph'`.

- [ ] **Step 3: Create `graph.py` with the dataclasses**

Write `nkigym/src/nkigym/codegen/graph.py`:

```python
"""``parse_and_resolve``: turn an ``f_nkigym`` callable into an :class:`OpGraph`.

Pipeline:
    1. AST-parse the math function to an ordered list of parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size + tile size from ``input_specs`` and
       per-op ``TILE_LIMITS``.
    4. Tag each tensor's ``origin`` (``param`` / ``intermediate`` / ``return``).

The resulting :class:`OpGraph` is read-only: for any
``(func, input_specs)`` exactly one graph exists. There are no tunable
knobs; the renderer lowers this graph mechanically.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

TensorOrigin = Literal["param", "intermediate", "return"]


@dataclass
class Tensor:
    """Named tensor appearing in the ``f_nkigym`` body.

    Attributes:
        name: Source-level variable name (e.g. ``"lhs"`` or ``"rms_inv"``).
        dim_ids: Concrete dim ids in operand order (e.g. ``("d0", "d1")``).
        shape: Element sizes aligned with ``dim_ids``.
        dtype: Element dtype (e.g. ``"bfloat16"``, ``"float32"``).
        origin: Lineage role — ``"param"`` (HBM kernel input),
            ``"intermediate"`` (SBUF handoff), or ``"return"`` (final
            op output).
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin


@dataclass
class DimInfo:
    """Concrete dimension metadata derived from ops + input specs."""

    dim_id: str
    total_size: int
    tile_size: int
    num_tiles: int


@dataclass
class ParsedOp:
    """One ``NKIOp()(...)`` call captured from the ``f_nkigym`` body.

    Attributes:
        idx: 0-indexed position in the math function.
        op_cls: The NKIOp subclass (e.g. ``NKIMatmul``).
        operand_names: Maps operand slot name (``"data"``, ``"stationary"``
            etc.) to the local variable name the call references.
        op_kwargs: Literal keyword arguments merged from constructor
            and call site (e.g. ``{"op": "square", "scale": 1/2048}``).
        output_names: Names the assignment target binds.
        axis_map: Abstract axis label (``"K"``, ``"M"`` ...) to concrete
            dim id.
        touched_dims: Every dim id this op's operands or outputs mention,
            in canonical loop-nest order: partition-axis dim first, then
            free-axis dims, then any reducing dim.
    """

    idx: int
    op_cls: type
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]


@dataclass
class OpGraph:
    """Read-only resolved view of an ``f_nkigym`` function.

    Attributes:
        func_name: Function name (lands on the emitted kernel).
        param_names: Kernel parameters in signature order.
        return_name: Tensor name of the return value (the ``NKIStore``
            output).
        tensors: All named tensors, keyed by name.
        dims: All dims, keyed by dim id.
        ops: Parsed ops in source order.
        per_op_attrs: Per-op annotation side-table keyed by
            ``ParsedOp.idx``. Empty by default — reserved for future
            passes like ``propagate_compute_skip``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    ops: list[ParsedOp]
    per_op_attrs: dict[int, dict[str, Any]] = field(default_factory=dict)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest test/codegen/test_graph.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "Add OpGraph dataclasses for the eager renderer"
```

---

## Task 2: Port the AST walk (parse step)

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Modify: `test/codegen/test_graph.py`

- [ ] **Step 1: Add failing AST walk test**

Append to `test/codegen/test_graph.py`:

```python
from nkigym.codegen.graph import _ParsedOpRaw, _parse_ast


def _matmul_func() -> Any:
    """Inline-defined nkigym function for AST tests."""
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.transpose import NKITranspose

    @nkigym_kernel
    def f_nkigym(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        lhs_T = NKITranspose()(data=lhs_sbuf)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    return f_nkigym


def test_parse_ast_captures_ops_in_source_order() -> None:
    """_parse_ast returns one entry per NKIOp call, in source order."""
    raws, return_name = _parse_ast(_matmul_func())
    kinds = [r.op_cls.__name__ for r in raws]
    assert kinds == ["NKILoad", "NKILoad", "NKITranspose", "NKIMatmul", "NKIStore"]
    assert return_name == "out"


def test_parse_ast_captures_operand_names() -> None:
    """Name-valued kwargs become operand_names entries."""
    raws, _ = _parse_ast(_matmul_func())
    assert raws[0].operand_names == {"data": "lhs"}
    assert raws[3].operand_names == {"stationary": "lhs_T", "moving": "rhs_sbuf"}
    assert raws[3].output_names == ["prod"]


def test_parse_ast_captures_literal_kwargs() -> None:
    """Constructor + call-site literal kwargs merge into op_kwargs."""
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    EPS = 1e-6

    @nkigym_kernel
    def f_nkigym(lhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 2048, bias=EPS)(
            data=lhs_sbuf
        )
        out = NKIStore()(data=rms_inv)
        return out

    raws, _ = _parse_ast(f_nkigym)
    kwargs = raws[1].op_kwargs
    assert kwargs["op"] == "square"
    assert kwargs["reduce_op"] == "add"
    assert kwargs["post_op"] == "rsqrt"
    assert kwargs["scale"] == 1 / 2048
    assert kwargs["bias"] == EPS
```

Add the `typing.Any` import at the top of the test module if missing:

```python
from typing import Any
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `pytest test/codegen/test_graph.py -v`
Expected: ImportError on `_ParsedOpRaw` / `_parse_ast`.

- [ ] **Step 3: Implement the AST walk**

Append to `nkigym/src/nkigym/codegen/graph.py` (after the existing dataclasses):

```python
import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np

from nkigym.ops.base import NKIOp


@dataclass
class _ParsedOpRaw:
    """Raw AST-parsed op record — pre dim unification.

    Attributes:
        op_cls: Resolved NKIOp subclass.
        operand_names: Name-valued kwargs from the outer call
            (tensor operands).
        op_kwargs: Merged literal kwargs from inner ``OpClass(...)``
            constructor and outer call.
        output_names: Names bound by the assignment target.
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]


def _parse_ast(func: Callable[..., np.ndarray]) -> tuple[list[_ParsedOpRaw], str]:
    """Walk ``func``'s AST to extract ordered ``NKIOp`` calls + return name.

    Args:
        func: An ``f_nkigym`` math function decorated with
            ``@nkigym_kernel``. Body must be straight-line
            ``var = NKIOp()(...)`` assignments terminated by a
            ``return var`` statement.

    Returns:
        ``(raws, return_name)`` — one ``_ParsedOpRaw`` per NKIOp call,
        in source order, plus the name of the returned tensor.

    Raises:
        ValueError: The source lacks a ``return`` or binds a
            non-``Name`` target.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    raws: list[_ParsedOpRaw] = []
    return_name: str | None = None
    for stmt in func_def.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id
            continue
        if isinstance(stmt, ast.Assign):
            raw = _parse_assignment(stmt, func.__globals__)
            if raw is not None:
                raws.append(raw)
    if return_name is None:
        raise ValueError("f_nkigym must end with `return <tensor>`")
    return raws, return_name


def _parse_assignment(stmt: ast.Assign, func_globals: dict[str, object]) -> _ParsedOpRaw | None:
    """Convert one ``var = OpClass(...)(kw=...)`` statement into a ``_ParsedOpRaw``.

    Returns ``None`` for assignments that are not NKIOp double-calls.
    """
    result: _ParsedOpRaw | None = None
    op_cls = _resolve_op_class(stmt.value, func_globals) if len(stmt.targets) == 1 else None
    if op_cls is not None:
        output_names = _extract_output_names(stmt.targets[0])
        if output_names:
            if len(output_names) != len(op_cls.OUTPUT_AXES):
                raise ValueError(
                    f"Op {op_cls.NAME}: {len(output_names)} outputs assigned but "
                    f"OUTPUT_AXES has {len(op_cls.OUTPUT_AXES)} entries"
                )
            outer_call = stmt.value
            assert isinstance(outer_call, ast.Call)
            operand_names = _extract_name_kwargs(outer_call)
            outer_kwargs = _extract_literal_kwargs(outer_call, func_globals)
            inner_call = outer_call.func
            assert isinstance(inner_call, ast.Call)
            merged = {**_extract_literal_kwargs(inner_call, func_globals), **outer_kwargs}
            result = _ParsedOpRaw(
                op_cls=op_cls, operand_names=operand_names, op_kwargs=merged, output_names=output_names
            )
    return result


def _resolve_op_class(node: ast.expr, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Return the NKIOp subclass a double-call expression references, or ``None``."""
    result: type[NKIOp] | None = None
    is_double = isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name)
    if is_double:
        inner_call = node.func
        assert isinstance(inner_call, ast.Call)
        name_node = inner_call.func
        assert isinstance(name_node, ast.Name)
        candidate = func_globals.get(name_node.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            result = candidate
    return result


def _extract_name_kwargs(call: ast.Call) -> dict[str, str]:
    """Return ``{kwarg_name: local_variable_name}`` for every Name-valued kwarg."""
    return {kw.arg: kw.value.id for kw in call.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)}


def _extract_output_names(target: ast.expr) -> list[str]:
    """Extract the bound names from an assignment target."""
    if isinstance(target, ast.Name):
        names = [target.id]
    elif isinstance(target, ast.Tuple):
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
    else:
        names = []
    return names


def _extract_literal_kwargs(call: ast.Call, func_globals: dict[str, object]) -> dict[str, Any]:
    """Return ``{kwarg_name: python_literal}`` for every literal-valued kwarg.

    Handles plain ``ast.Constant`` values, BinOp/UnaryOp trees over
    constants (``1/2048``, ``-eps``), and Name references that resolve
    to numeric constants in ``func_globals`` (e.g. module-level ``EPS``).
    """
    out: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        ok, value = _literal_value(kw.value, func_globals)
        if ok:
            out[kw.arg] = value
    return out


def _literal_value(node: ast.expr, func_globals: dict[str, object]) -> tuple[bool, Any]:
    """Try to evaluate ``node`` as a Python literal. Returns ``(ok, value)``."""
    result: tuple[bool, Any] = (False, None)
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name):
        value = func_globals.get(node.id)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = (True, value)
    if isinstance(node, ast.UnaryOp):
        ok, inner = _literal_value(node.operand, func_globals)
        if ok and isinstance(node.op, ast.USub):
            result = (True, -inner)
        elif ok and isinstance(node.op, ast.UAdd):
            result = (True, +inner)
    if isinstance(node, ast.BinOp):
        ok_l, lhs = _literal_value(node.left, func_globals)
        ok_r, rhs = _literal_value(node.right, func_globals)
        if ok_l and ok_r:
            if isinstance(node.op, ast.Add):
                result = (True, lhs + rhs)
            elif isinstance(node.op, ast.Sub):
                result = (True, lhs - rhs)
            elif isinstance(node.op, ast.Mult):
                result = (True, lhs * rhs)
            elif isinstance(node.op, ast.Div):
                result = (True, lhs / rhs)
    return result
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest test/codegen/test_graph.py -v`
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "Port AST walk for eager-renderer op discovery"
```

---

## Task 3: Dim unification + tile-size derivation

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Modify: `test/codegen/test_graph.py`

- [ ] **Step 1: Add failing unification tests**

Append to `test/codegen/test_graph.py`:

```python
def test_parse_and_resolve_simple_matmul() -> None:
    """lhs_T (K, M) @ moving (K, N) → output (M, N) unifies correctly."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.param_names == ["lhs_T", "rhs"]
    assert g.return_name == "out"
    """K-axis of lhs_T and rhs must unify (both first dim)."""
    assert g.tensors["lhs_T"].dim_ids[0] == g.tensors["rhs"].dim_ids[0]
    assert g.tensors["prod"].dim_ids == (g.tensors["lhs_T"].dim_ids[1], g.tensors["rhs"].dim_ids[1])
    """Matmul tile limits produce K=128, M=128, N=512 after min() with sizes."""
    k_dim = g.tensors["lhs_T"].dim_ids[0]
    m_dim = g.tensors["lhs_T"].dim_ids[1]
    n_dim = g.tensors["rhs"].dim_ids[1]
    assert g.dims[k_dim].tile_size == 128
    assert g.dims[m_dim].tile_size == 128
    assert g.dims[n_dim].tile_size == 512
    assert g.dims[k_dim].num_tiles == 16
    assert g.dims[n_dim].num_tiles == 4


def test_parse_and_resolve_tensor_origins() -> None:
    """Params tag 'param', intermediates 'intermediate', last store output 'return'."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.tensors["lhs_T"].origin == "param"
    assert g.tensors["rhs"].origin == "param"
    assert g.tensors["lhs_T_sbuf"].origin == "intermediate"
    assert g.tensors["prod"].origin == "intermediate"
    assert g.tensors["out"].origin == "return"


def test_parse_and_resolve_activation_reduce_output_dtype() -> None:
    """NKIActivationReduce's output (P,) is pinned to float32."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def kernel(lhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rms = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
        out = NKIStore()(data=rms)
        return out

    specs = {"lhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.tensors["rms"].dtype == "float32"
    assert g.tensors["rms"].dim_ids == g.tensors["lhs"].dim_ids[:1]


def test_parse_and_resolve_touched_dims_ordering() -> None:
    """touched_dims lists partition dim first, then free, then reducing."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    matmul_op = next(op for op in g.ops if op.op_cls.__name__ == "NKIMatmul")
    """Matmul partition = M (from stationary's second axis), frees = N, reducing = K."""
    k_dim = g.tensors["lhs_T_sbuf"].dim_ids[0]
    m_dim = g.tensors["lhs_T_sbuf"].dim_ids[1]
    n_dim = g.tensors["rhs_sbuf"].dim_ids[1]
    assert matmul_op.touched_dims == (m_dim, n_dim, k_dim)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest test/codegen/test_graph.py -v`
Expected: `ImportError: cannot import name 'parse_and_resolve'`.

- [ ] **Step 3: Implement `parse_and_resolve`**

Append to `nkigym/src/nkigym/codegen/graph.py`:

```python
def parse_and_resolve(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> OpGraph:
    """AST-parse ``func`` and resolve dims, tensors, tile sizes.

    Args:
        func: A math function decorated with ``@nkigym_kernel`` whose
            body is straight-line ``NKIOp()(...)`` assignments followed
            by ``return <tensor>``.
        input_specs: ``{param_name: (shape, dtype)}`` for every function
            parameter.

    Returns:
        A fully resolved :class:`OpGraph`.
    """
    raws, return_name = _parse_ast(func)
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    tensors_scratch: dict[str, _ScratchTensor] = {}
    for name in param_names:
        shape, dtype = input_specs[name]
        tensors_scratch[name] = _ScratchTensor(name=name, shape=shape, dtype=dtype)

    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for raw in raws:
        op_cls = raw.op_cls
        operand_map = {k: v for k, v in raw.operand_names.items() if v in tensors_scratch}
        axis_map = _build_axis_map(op_cls, operand_map, tensors_scratch, dim_counter, per_op_axis_maps, dim_sizes)
        per_op_axis_maps.append(axis_map)
        _create_outputs(op_cls, operand_map, raw.output_names, axis_map, tensors_scratch, dim_sizes)

    if return_name not in tensors_scratch:
        raise ValueError(f"Return tensor {return_name!r} not produced by any op")

    dims = _resolve_dimensions(raws, per_op_axis_maps, dim_sizes)
    tensors = _finalize_tensors(tensors_scratch, param_names, return_name, raws)
    ops = _build_parsed_ops(raws, per_op_axis_maps, tensors)

    return OpGraph(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        ops=ops,
        per_op_attrs={},
    )


@dataclass
class _ScratchTensor:
    """Mutable dim-inference record used during resolution."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dim_ids: list[str] = field(default_factory=list)


def _build_axis_map(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    tensors: dict[str, _ScratchTensor],
    dim_counter: list[int],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
) -> dict[str, str]:
    """Zip abstract axis labels against each operand's concrete dim ids."""
    local: dict[str, str] = {}
    for slot, axes in op_cls.OPERAND_AXES.items():
        if slot not in operand_map:
            continue
        tensor = tensors[operand_map[slot]]
        if tensor.dim_ids:
            for abstract, concrete in zip(axes, tensor.dim_ids):
                if abstract in local and local[abstract] != concrete:
                    _unify_dim(tensors, per_op_axis_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
                else:
                    local[abstract] = concrete
        else:
            for i, abstract in enumerate(axes[: len(tensor.shape)]):
                if abstract not in local:
                    fresh = f"d{dim_counter[0]}"
                    dim_counter[0] += 1
                    dim_sizes[fresh] = tensor.shape[i]
                    local[abstract] = fresh
            tensor.dim_ids = [local[a] for a in axes[: len(tensor.shape)]]
    return local


def _unify_dim(
    tensors: dict[str, _ScratchTensor],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    old_id: str,
    new_id: str,
) -> None:
    """Rename ``old_id`` to ``new_id`` everywhere; raise on size conflict."""
    old_size = dim_sizes.get(old_id)
    new_size = dim_sizes.get(new_id)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(f"Cannot unify {old_id} (size {old_size}) with {new_id} (size {new_size})")
    if old_id in dim_sizes:
        dim_sizes.setdefault(new_id, dim_sizes.pop(old_id))
    for tensor in tensors.values():
        tensor.dim_ids = [new_id if d == old_id else d for d in tensor.dim_ids]
    for axis_map in per_op_axis_maps:
        for ax in axis_map:
            if axis_map[ax] == old_id:
                axis_map[ax] = new_id


def _create_outputs(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    output_names: list[str],
    local: dict[str, str],
    tensors: dict[str, _ScratchTensor],
    dim_sizes: dict[str, int],
) -> None:
    """Register output tensors with concrete dim ids derived from ``OUTPUT_AXES``."""
    first_slot = next(iter(op_cls.OPERAND_AXES))
    has_first = first_slot in operand_map and operand_map[first_slot] in tensors
    default_dtype = (
        tensors[operand_map[first_slot]].dtype
        if has_first
        else next(t.dtype for t in tensors.values() if t.dtype)
    )
    for oname, (out_slot, output_axes) in zip(output_names, op_cls.OUTPUT_AXES.items()):
        dim_ids = [local[a] for a in output_axes if a in local]
        shape = tuple(dim_sizes[d] for d in dim_ids)
        dtype = op_cls.OUTPUT_DTYPES.get(out_slot, default_dtype)
        tensors[oname] = _ScratchTensor(name=oname, shape=shape, dtype=dtype, dim_ids=dim_ids)


def _resolve_dimensions(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], dim_sizes: dict[str, int]
) -> dict[str, DimInfo]:
    """Derive per-dim tile size = min of op TILE_LIMITS touching the dim."""
    per_dim_tile: dict[str, int] = {}
    for raw, axis_map in zip(raws, per_op_axis_maps):
        for abstract, limit in raw.op_cls.TILE_LIMITS.items():
            if abstract not in axis_map:
                continue
            dim_id = axis_map[abstract]
            tile = min(limit, dim_sizes[dim_id])
            per_dim_tile[dim_id] = min(per_dim_tile.get(dim_id, tile), tile)
    for d in dim_sizes:
        if d not in per_dim_tile:
            raise ValueError(f"Dim {d!r} has no op-declared tile size")
    return {
        d: DimInfo(dim_id=d, total_size=dim_sizes[d], tile_size=per_dim_tile[d], num_tiles=dim_sizes[d] // per_dim_tile[d])
        for d in dim_sizes
    }


def _finalize_tensors(
    scratch: dict[str, _ScratchTensor], param_names: list[str], return_name: str, raws: list[_ParsedOpRaw]
) -> dict[str, Tensor]:
    """Convert scratch tensors to read-only ``Tensor``s, tagging origin."""
    store_inputs: set[str] = set()
    for raw in raws:
        if raw.op_cls.__name__ == "NKIStore":
            store_inputs.update(raw.operand_names.values())
    out: dict[str, Tensor] = {}
    for name, st in scratch.items():
        if name in param_names:
            origin: TensorOrigin = "param"
        elif name == return_name:
            origin = "return"
        else:
            origin = "intermediate"
        out[name] = Tensor(
            name=name, dim_ids=tuple(st.dim_ids), shape=tuple(st.shape), dtype=st.dtype, origin=origin
        )
    _ = store_inputs
    return out


def _build_parsed_ops(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], tensors: dict[str, Tensor]
) -> list[ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims``."""
    ops: list[ParsedOp] = []
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        ops.append(
            ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
            )
        )
    return ops


def _touched_dims(raw: _ParsedOpRaw, axis_map: dict[str, str], tensors: dict[str, Tensor]) -> tuple[str, ...]:
    """Canonical order: partition-axis dim first, then free dims, then reducing dims."""
    op_cls = raw.op_cls
    first_out_slot = next(iter(op_cls.OUTPUT_AXES))
    out_axes = op_cls.OUTPUT_AXES[first_out_slot]
    ordered: list[str] = []
    for abstract in out_axes:
        if abstract in axis_map and axis_map[abstract] not in ordered:
            ordered.append(axis_map[abstract])
    blocking = [axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map]
    for d in blocking:
        if d not in ordered:
            ordered.append(d)
    for slot, axes in op_cls.OPERAND_AXES.items():
        tname = raw.operand_names.get(slot)
        if tname is None or tname not in tensors:
            continue
        for abstract in axes:
            if abstract in axis_map and axis_map[abstract] not in ordered:
                ordered.append(axis_map[abstract])
    return tuple(ordered)
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest test/codegen/test_graph.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "Resolve dims, tile sizes, and tensor origins for OpGraph"
```

---

## Task 4: `render.py` scaffolding + writer + kernel header

**Files:**
- Create: `nkigym/src/nkigym/codegen/render.py` (replaces old)
- Create: `test/codegen/test_render.py`

- [ ] **Step 1: Rewrite `render.py` down to a stub**

Open `nkigym/src/nkigym/codegen/render.py` and replace its contents with:

```python
"""``render``: lower an :class:`OpGraph` to NKI source.

Each NKIOp becomes one independent loop nest over only the dims it
touches. SBUF intermediates are hoisted to function top as full-extent
``(p_tile, num_p_tiles, num_f_tiles * f_tile)`` allocations. HBM lives
on kernel inputs (consumed by ``NKILoad``) and the kernel's return
tensor (written by ``NKIStore``).
"""

from nkigym.codegen.graph import OpGraph


class _Writer:
    """Line-based writer with indentation tracking."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        """Open a nested block — subsequent ``line`` calls indent one level deeper."""
        self._depth += 1

    def dedent(self) -> None:
        """Close a nested block."""
        self._depth -= 1

    def line(self, text: str = "") -> None:
        """Append a source line at the current indent."""
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        """Return the accumulated source with a trailing newline."""
        return "\n".join(self._lines) + "\n"


def render(op_graph: OpGraph) -> str:
    """Render ``op_graph`` to NKI kernel source."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph)
    for op in op_graph.ops:
        _emit_op(w, op_graph, op)
    w.line(f"return hbm_{op_graph.return_name}")
    w.dedent()
    return w.getvalue()


def _emit_imports(w: _Writer) -> None:
    """Emit the standard import header."""
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``@nki.jit`` + ``def <func>(<params>):``."""
    w.line("@nki.jit")
    params = ", ".join(op_graph.param_names)
    w.line(f"def {op_graph.func_name}({params}):")


def _emit_param_asserts(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``assert <param>.shape == (...)`` for every kernel input."""
    for name in op_graph.param_names:
        shape = op_graph.tensors[name].shape
        w.line(f"assert {name}.shape == {shape}")


def _emit_hbm_output(w: _Writer, op_graph: OpGraph) -> None:
    """Allocate the HBM output tensor (``hbm_<return>``)."""
    ret = op_graph.tensors[op_graph.return_name]
    w.line(
        f"hbm_{ret.name} = nl.ndarray({ret.shape}, dtype=nl.{ret.dtype}, buffer=nl.shared_hbm)"
    )


def _emit_sbuf_allocations(w: _Writer, op_graph: OpGraph) -> None:
    """Allocate every SBUF intermediate + the SBUF mirror of the return."""
    for name, tensor in op_graph.tensors.items():
        if tensor.origin == "param":
            continue
        shape = _sbuf_shape(tensor, op_graph)
        w.line(f"sbuf_{name} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    w.line()


def _sbuf_shape(tensor, op_graph: OpGraph) -> tuple[int, int, int]:
    """Compute 3D SBUF shape ``(p_tile, num_p_tiles, num_f_tiles*f_tile)``.

    1D tensors collapse the free axis to a single element.
    """
    if not tensor.dim_ids:
        raise ValueError(f"Tensor {tensor.name!r} has no dims")
    p_axis = tensor.dim_ids[0]
    p = op_graph.dims[p_axis]
    if len(tensor.dim_ids) == 1:
        return (p.tile_size, p.num_tiles, 1)
    f_axis = tensor.dim_ids[1]
    f = op_graph.dims[f_axis]
    return (p.tile_size, p.num_tiles, f.num_tiles * f.tile_size)


def _emit_op(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit one op's independent loop nest. Stub — dispatched in Task 5."""
    raise NotImplementedError(f"op emitter not implemented for {op.op_cls.__name__}")
```

- [ ] **Step 2: Create a header-only test**

Create `test/codegen/test_render.py`:

```python
"""Tests for the eager renderer."""

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _matmul_lhsT_rhs(lhs_T, rhs):
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_SPECS = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}


def test_render_emits_header_and_allocations() -> None:
    """Kernel header lists imports, decorator, signature, asserts, HBM output, SBUF allocs."""
    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    try:
        src = render(g)
    except NotImplementedError:
        """Op emitters aren't implemented yet; validate scaffold on a
        fixture that only exercises the header path by stubbing the op
        loop. For now, verify header pieces by inspecting getvalue()
        before NotImplementedError fires — assemble manually."""
        src = _header_only(g)
    assert "import nki" in src
    assert "@nki.jit" in src
    assert "def _matmul_lhsT_rhs(lhs_T, rhs):" in src
    assert "assert lhs_T.shape == (2048, 2048)" in src
    assert "assert rhs.shape == (2048, 2048)" in src
    assert "hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)" in src
    assert "sbuf_lhs_T_sbuf = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_rhs_sbuf = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_out = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src


def _header_only(g) -> str:
    """Run the renderer's header path without op emission (Task-4 scope)."""
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _Writer,
    )

    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, g)
    w.indent()
    _emit_param_asserts(w, g)
    _emit_hbm_output(w, g)
    _emit_sbuf_allocations(w, g)
    return w.getvalue()
```

- [ ] **Step 3: Run the test and verify pass**

Run: `pytest test/codegen/test_render.py -v`
Expected: 1 passed. `NotImplementedError` triggers the fallback path.

- [ ] **Step 4: Update the `codegen` package entry point**

Replace the contents of `nkigym/src/nkigym/codegen/__init__.py`:

```python
"""Eager renderer — ``f_nkigym`` → NKI source."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.render import render


def render_eager(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> str:
    """Lower a decorated ``f_nkigym`` callable to NKI kernel source.

    Args:
        func: Math function decorated with ``@nkigym_kernel``.
        input_specs: ``{param: (shape, dtype)}`` for every parameter.

    Returns:
        NKI source string containing the ``@nki.jit`` kernel.
    """
    return render(parse_and_resolve(func, input_specs))


__all__ = ["render_eager"]
```

Run `pytest test/codegen/ -v` — should still be green.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/__init__.py nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add eager renderer scaffolding (header + SBUF allocations)"
```

---

## Task 5: Slice helper + per-op dispatch table

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing slice-helper test**

Append to `test/codegen/test_render.py`:

```python
def test_sbuf_tile_slice_2d() -> None:
    """Per-tile slice for a 2D SBUF tensor uses i_block_<p> and i_block_<f>."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == "sbuf_lhs[0:128, i_block_d0, i_block_d1 * 128 : i_block_d1 * 128 + 128]"


def test_sbuf_tile_slice_1d() -> None:
    """Per-tile slice for a 1D SBUF tensor (e.g. activation_reduce output) uses 0:1 on free."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_rms", ("d0",), p_tile=128, f_tile=1)
    assert slice_expr == "sbuf_rms[0:128, i_block_d0, 0:1]"


def test_hbm_tile_slice() -> None:
    """HBM tile slice uses p_tile and f_tile offsets."""
    from nkigym.codegen.render import _hbm_tile_slice

    slice_expr = _hbm_tile_slice("lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == "lhs[i_block_d0 * 128 : i_block_d0 * 128 + 128, i_block_d1 * 128 : i_block_d1 * 128 + 128]"
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest test/codegen/test_render.py -v`
Expected: ImportError on `_sbuf_tile_slice`.

- [ ] **Step 3: Implement the slice helpers + dispatch scaffolding**

In `nkigym/src/nkigym/codegen/render.py`, replace the stub `_emit_op` body and append the helpers:

```python
def _sbuf_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the ``sbuf_<name>[...]`` slice for one per-tile access.

    Args:
        name: Full buffer name (caller passes ``sbuf_<tensor>``).
        dim_ids: Tensor dim ids in operand order.
        p_tile: Partition-axis tile size.
        f_tile: Free-axis tile size (pass ``1`` for 1D tensors).

    Returns:
        A Python slice expression referencing the open-loop variables
        ``i_block_<p>`` (+ ``i_block_<f>`` for 2D tensors).
    """
    p_axis = dim_ids[0]
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, i_block_{p_axis}, 0:1]"
    f_axis = dim_ids[1]
    return (
        f"{name}[0:{p_tile}, i_block_{p_axis}, "
        f"i_block_{f_axis} * {f_tile} : i_block_{f_axis} * {f_tile} + {f_tile}]"
    )


def _hbm_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the HBM slice ``name[p_range, f_range]`` for one per-tile access."""
    p_axis = dim_ids[0]
    if len(dim_ids) == 1:
        return f"{name}[i_block_{p_axis} * {p_tile} : i_block_{p_axis} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    return (
        f"{name}[i_block_{p_axis} * {p_tile} : i_block_{p_axis} * {p_tile} + {p_tile}, "
        f"i_block_{f_axis} * {f_tile} : i_block_{f_axis} * {f_tile} + {f_tile}]"
    )


def _emit_op(w: _Writer, op_graph: OpGraph, op) -> None:
    """Dispatch to the per-op-kind emitter."""
    emitter = _EMITTERS.get(op.op_cls.__name__)
    if emitter is None:
        raise ValueError(f"No emitter for op kind {op.op_cls.__name__!r}")
    emitter(w, op_graph, op)


_EMITTERS: dict = {}
"""Populated in Task 6 by ``_register_emitter`` (one entry per supported op kind)."""


def _register_emitter(kind: str):
    """Decorator: register an emit function for ``op_cls.__name__``."""

    def wrap(fn):
        _EMITTERS[kind] = fn
        return fn

    return wrap


def _open_block_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open ``for i_block_<d> in range(num_tiles):`` for each dim; return depth opened."""
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
    return len(dims)


def _close_loops(w: _Writer, depth: int) -> None:
    """Dedent ``depth`` times to close previously-opened loops."""
    for _ in range(depth):
        w.dedent()


def _op_header_comment(op) -> str:
    """Return the header docstring emitted at the top of an op's nest."""
    operands = ", ".join(f"{k}={v}" for k, v in op.operand_names.items())
    outputs = ", ".join(op.output_names)
    return f'"""Op {op.idx}: nisa.{op.op_cls.NAME} — {operands} -> {outputs}"""'
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest test/codegen/test_render.py -v`
Expected: Slice-helper tests pass. The `test_render_emits_header_and_allocations` test still runs the fallback path because no emitters are registered yet.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add slice helpers + per-op dispatch table to eager renderer"
```

---

## Task 6a: NKILoad + NKIStore emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Add failing emitter test**

Append to `test/codegen/test_render.py`:

```python
def test_render_load_store_kernel() -> None:
    """A bare NKILoad + NKIStore kernel renders and simulates correctly."""
    import numpy as np
    import nki

    @nkigym_kernel
    def passthrough(x):
        xs = NKILoad()(data=x)
        out = NKIStore()(data=xs)
        return out

    specs = {"x": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(passthrough, specs)
    src = render(g)
    assert "nisa.dma_copy(" in src
    assert "for i_block_" in src

    """CPU-sim the rendered kernel at fp32 and compare to the numpy identity."""
    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["passthrough"]
    rng = np.random.default_rng(0)
    x_in = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x_in)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x_in, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest test/codegen/test_render.py::test_render_load_store_kernel -v`
Expected: `ValueError: No emitter for op kind 'NKILoad'` or `NKIStore`.

- [ ] **Step 3: Implement both emitters**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
@_register_emitter("NKILoad")
def _emit_load(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA load nest: HBM parameter → SBUF intermediate."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_loops(w, op_graph, src_tensor.dim_ids)
    dst_expr = _sbuf_tile_slice(f"sbuf_{dst_name}", dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _hbm_tile_slice(src_name, src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)


@_register_emitter("NKIStore")
def _emit_store(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA store nest: SBUF producer → HBM return tensor."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_loops(w, op_graph, dst_tensor.dim_ids)
    dst_expr = _hbm_tile_slice(f"hbm_{dst_name}", dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(f"sbuf_{src_name}", src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)
```

- [ ] **Step 4: Run tests — passthrough kernel should now render + simulate**

Run: `pytest test/codegen/test_render.py -v`
Expected: `test_render_load_store_kernel` passes. Other tests still green.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add NKILoad + NKIStore emitters"
```

---

## Task 6b: NKIMatmul emitter

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing matmul test**

Append to `test/codegen/test_render.py`:

```python
def test_render_matmul_lhsT_rhs() -> None:
    """Full lhs_T @ rhs kernel renders and simulates to the numpy reference."""
    import numpy as np
    import nki

    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    src = render(g)
    assert "nisa.nc_matmul(" in src
    assert "nl.psum" in src
    assert "nisa.tensor_copy(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_matmul_lhsT_rhs"]

    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((2048, 2048)).astype(np.float32)
    rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel)(lhs_T=lhs_T, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs_T.T @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest test/codegen/test_render.py::test_render_matmul_lhsT_rhs -v`
Expected: `ValueError: No emitter for op kind 'NKIMatmul'`.

- [ ] **Step 3: Implement matmul emitter**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
@_register_emitter("NKIMatmul")
def _emit_matmul(w: _Writer, op_graph: OpGraph, op) -> None:
    """Matmul nest: open M/N blocks → allocate PSUM → accumulate over K → drain."""
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    out_name = op.output_names[0]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size
    num_m = op_graph.dims[m_dim].num_tiles
    num_n = op_graph.dims[n_dim].num_tiles
    num_k = op_graph.dims[k_dim].num_tiles

    w.line()
    w.line(_op_header_comment(op))
    w.line(f"for i_block_{m_dim} in range({num_m}):")
    w.indent()
    w.line(f"for i_block_{n_dim} in range({num_n}):")
    w.indent()
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")
    w.line(f"for i_block_{k_dim} in range({num_k}):")
    w.indent()
    stat_expr = _sbuf_tile_slice(f"sbuf_{stat_name}", stat.dim_ids, p_tile_K, p_tile_M)
    mov_expr = _sbuf_tile_slice(f"sbuf_{mov_name}", mov.dim_ids, p_tile_K, f_tile_N)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")
    w.dedent()
    out_expr = _sbuf_tile_slice(f"sbuf_{out_name}", out.dim_ids, p_tile_M, f_tile_N)
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")
    w.dedent()
    w.dedent()
```

- [ ] **Step 4: Verify**

Run: `pytest test/codegen/test_render.py::test_render_matmul_lhsT_rhs -v`
Expected: PASS.

Then `pytest test/codegen/ -v` — all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add NKIMatmul emitter (PSUM accumulator + tensor_copy drain)"
```

---

## Task 6c: NKITranspose + NKIDMATranspose emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing transpose test**

Append to `test/codegen/test_render.py`:

```python
def test_render_transpose() -> None:
    """Stand-alone NKITranspose kernel renders and simulates correctly."""
    import numpy as np
    import nki
    from nkigym.ops.transpose import NKITranspose

    @nkigym_kernel
    def tr(x):
        xs = NKILoad()(data=x)
        y = NKITranspose()(data=xs)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(tr, specs)
    src = render(g)
    assert "nisa.nc_transpose(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["tr"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x.T, atol=1e-5)


def test_render_dma_transpose() -> None:
    """NKIDMATranspose emits nisa.dma_transpose."""
    import numpy as np
    import nki
    from nkigym.ops.dma_transpose import NKIDMATranspose

    @nkigym_kernel
    def tr(x):
        xs = NKILoad()(data=x)
        y = NKIDMATranspose()(data=xs)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(tr, specs)
    src = render(g)
    assert "nisa.dma_transpose(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["tr"]
    x = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x.T, atol=1e-5)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest test/codegen/test_render.py::test_render_transpose -v`
Expected: `ValueError: No emitter for op kind 'NKITranspose'`.

- [ ] **Step 3: Implement both emitters**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
@_register_emitter("NKITranspose")
def _emit_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """Tensor-Engine transpose via PSUM staging."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    w.line(f"for i_block_{src_p_axis} in range({op_graph.dims[src_p_axis].num_tiles}):")
    w.indent()
    w.line(f"for i_block_{src_f_axis} in range({op_graph.dims[src_f_axis].num_tiles}):")
    w.indent()
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_expr = _sbuf_tile_slice(f"sbuf_{src_name}", src.dim_ids, p_tile, f_tile)
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    """Output tensor has swapped dim order: (F, P) — slice with the same open loops."""
    dst_expr = (
        f"sbuf_{dst_name}[0:{p_tile}, i_block_{src_f_axis}, "
        f"i_block_{src_p_axis} * {p_tile} : i_block_{src_p_axis} * {p_tile} + {p_tile}]"
    )
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")
    w.dedent()
    w.dedent()


@_register_emitter("NKIDMATranspose")
def _emit_dma_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """DMA-engine transpose — one ``dma_transpose`` per (P, F) tile."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    w.line(f"for i_block_{src_p_axis} in range({op_graph.dims[src_p_axis].num_tiles}):")
    w.indent()
    w.line(f"for i_block_{src_f_axis} in range({op_graph.dims[src_f_axis].num_tiles}):")
    w.indent()
    src_expr = _sbuf_tile_slice(f"sbuf_{src_name}", src.dim_ids, p_tile, f_tile)
    dst_expr = (
        f"sbuf_{dst_name}[0:{p_tile}, i_block_{src_f_axis}, "
        f"i_block_{src_p_axis} * {p_tile} : i_block_{src_p_axis} * {p_tile} + {p_tile}]"
    )
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")
    w.dedent()
    w.dedent()
```

- [ ] **Step 4: Run tests**

Run: `pytest test/codegen/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add NKITranspose and NKIDMATranspose emitters"
```

---

## Task 6d: NKITensorScalar + NKIActivation emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing test**

Append to `test/codegen/test_render.py`:

```python
def test_render_activation() -> None:
    """NKIActivation emits nisa.activation with scale/bias plumbed through."""
    import numpy as np
    import nki
    from nkigym.ops.activation import NKIActivation

    @nkigym_kernel
    def act(x):
        xs = NKILoad()(data=x)
        y = NKIActivation(op="tanh", scale=0.5, bias=0.0)(data=xs)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(act, specs)
    src = render(g)
    assert "nisa.activation(" in src
    assert "scale=0.5" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["act"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, np.tanh(x * 0.5), atol=1e-4)


def test_render_tensor_scalar() -> None:
    """NKITensorScalar emits nisa.tensor_scalar with per-row operand0."""
    import numpy as np
    import nki
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.tensor_scalar import NKITensorScalar

    @nkigym_kernel
    def scale(x):
        xs = NKILoad()(data=x)
        rms = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 128, bias=1e-6)(data=xs)
        y = NKITensorScalar(op="multiply")(data=xs, operand0=rms)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(scale, specs)
    src = render(g)
    assert "nisa.tensor_scalar(" in src
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/codegen/test_render.py::test_render_activation -v`
Expected: `ValueError: No emitter for op kind 'NKIActivation'`.

- [ ] **Step 3: Implement both emitters**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
@_register_emitter("NKIActivation")
def _emit_activation(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.activation(dst, op, data, scale, bias)`` per tile."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1] if len(src.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    act = op.op_kwargs["op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_loops(w, op_graph, src.dim_ids)
    dst_expr = _sbuf_tile_slice(f"sbuf_{dst_name}", dst.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(f"sbuf_{src_name}", src.dim_ids, p_tile, f_tile)
    w.line(
        f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})"
    )
    _close_loops(w, depth)


@_register_emitter("NKITensorScalar")
def _emit_tensor_scalar(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.tensor_scalar(dst, data, op0, operand0)`` per tile."""
    data_name = op.operand_names["data"]
    op0_name = op.operand_names["operand0"]
    dst_name = op.output_names[0]
    data = op_graph.tensors[data_name]
    op0 = op_graph.tensors[op0_name]
    dst = op_graph.tensors[dst_name]
    p_axis = data.dim_ids[0]
    f_axis = data.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    op_name = op.op_kwargs["op"]

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_loops(w, op_graph, data.dim_ids)
    dst_expr = _sbuf_tile_slice(f"sbuf_{dst_name}", dst.dim_ids, p_tile, f_tile)
    data_expr = _sbuf_tile_slice(f"sbuf_{data_name}", data.dim_ids, p_tile, f_tile)
    op0_expr = _sbuf_tile_slice(f"sbuf_{op0_name}", op0.dim_ids, p_tile, 1)
    w.line(
        f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})"
    )
    _close_loops(w, depth)
```

- [ ] **Step 4: Run tests and verify**

Run: `pytest test/codegen/test_render.py -v`
Expected: PASS (note `test_render_tensor_scalar` can't simulate yet — `NKIActivationReduce` still missing — but it renders).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add NKIActivation and NKITensorScalar emitters"
```

---

## Task 6e: NKIActivationReduce emitter

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing simulate-to-numpy rmsnorm test**

Append to `test/codegen/test_render.py`:

```python
def test_render_activation_reduce_rmsnorm() -> None:
    """activation_reduce with post_op=rsqrt emits memset + accumulate loop + post_op."""
    import numpy as np
    import nki
    from nkigym.ops.activation_reduce import NKIActivationReduce

    @nkigym_kernel
    def rms(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 128, bias=1e-6)(data=xs)
        out = NKIStore()(data=m)
        return out

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(rms, specs)
    src = render(g)
    assert "nisa.memset(" in src
    assert "nisa.activation_reduce(" in src
    assert "nisa.activation(" in src
    assert "op=nl.rsqrt" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["rms"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = 1.0 / np.sqrt(np.mean(x * x, axis=1) + 1e-6)
    assert np.allclose(actual.reshape(-1), expected, atol=5e-4, rtol=5e-4)
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/codegen/test_render.py::test_render_activation_reduce_rmsnorm -v`
Expected: `ValueError: No emitter for op kind 'NKIActivationReduce'`.

- [ ] **Step 3: Implement emitter**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
_REDUCE_IDENTITY: dict[str, float] = {"add": 0.0, "max": float("-inf")}
_REDUCE_MERGE_OP: dict[str, str] = {"add": "nl.add", "max": "nl.maximum"}


@_register_emitter("NKIActivationReduce")
def _emit_activation_reduce(w: _Writer, op_graph: OpGraph, op) -> None:
    """Activation + free-axis reduce with optional ``post_op`` on the closed reduction."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    num_p = op_graph.dims[p_axis].num_tiles
    num_f = op_graph.dims[f_axis].num_tiles
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    post_op = op.op_kwargs.get("post_op")
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    identity = _REDUCE_IDENTITY[reduce_op]
    merge = _REDUCE_MERGE_OP[reduce_op]

    w.line()
    w.line(_op_header_comment(op))
    w.line(f"for i_block_{p_axis} in range({num_p}):")
    w.indent()
    dst_slot = f"sbuf_{dst_name}[0:{p_tile}, i_block_{p_axis}, 0:1]"
    w.line(f"nisa.memset({dst_slot}, value={identity})")
    w.line(f"for i_block_{f_axis} in range({num_f}):")
    w.indent()
    w.line(f"tmp_red = nl.ndarray(({p_tile}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    w.line(f"scratch = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    src_expr = _sbuf_tile_slice(f"sbuf_{src_name}", src.dim_ids, p_tile, f_tile)
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst=scratch[0:{p_tile}, 0:{f_tile}],")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res=tmp_red[0:{p_tile}, 0:1],")
    w.dedent()
    w.line(")")
    w.line(
        f"nisa.tensor_tensor({dst_slot}, {dst_slot}, tmp_red[0:{p_tile}, 0:1], op={merge})"
    )
    w.dedent()
    if post_op is not None:
        w.line(
            f"nisa.activation(dst={dst_slot}, op=nl.{post_op}, data={dst_slot}, scale={scale}, bias={bias})"
        )
    w.dedent()
```

- [ ] **Step 4: Run tests**

Run: `pytest test/codegen/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add NKIActivationReduce emitter with memset + merge loop"
```

---

## Task 7: End-to-end `rmsnorm_matmul` CPU sim

**Files:**
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Failing end-to-end rmsnorm+matmul test**

Append to `test/codegen/test_render.py`:

```python
def test_render_rmsnorm_matmul_end_to_end() -> None:
    """The full rmsnorm+matmul DAG renders and matches the numpy reference."""
    import numpy as np
    import nki
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose

    EPS = 1e-6

    @nkigym_kernel
    def rmsnorm_matmul(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        rms_inv = NKIActivationReduce(
            op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=EPS
        )(data=lhs_sbuf)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(rmsnorm_matmul, specs)
    src = render(g)

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["rmsnorm_matmul"]

    rng = np.random.default_rng(42)
    lhs = rng.standard_normal((128, 256)).astype(np.float32)
    rhs = rng.standard_normal((256, 512)).astype(np.float32)
    actual = nki.simulate(kernel)(lhs=lhs, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    m = np.mean(lhs * lhs, axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + EPS)
    expected = (lhs * rms_inv) @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run the test**

Run: `pytest test/codegen/test_render.py::test_render_rmsnorm_matmul_end_to_end -v`
Expected: PASS (all required emitters are in place from Task 6a–6e).

If it fails, inspect the rendered source by adding a temporary `print(src)` in the test and re-running — most likely a slice-expression or dim-ordering bug in one of the emitters.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_render.py
git commit -m "Add end-to-end rmsnorm+matmul CPU-sim test"
```

---

## Task 8: Wire `examples/rmsnorm_matmul.py`

**Files:**
- Modify: `examples/rmsnorm_matmul.py`

- [ ] **Step 1: Understand what's there**

Read `examples/rmsnorm_matmul.py`. It currently calls `compile_numpy_to_nkigym`
and writes the source to `cache_dir / "f_nkigym.py"`. We extend it to also
render the kernel and sanity-check it.

- [ ] **Step 2: Replace file**

Overwrite `examples/rmsnorm_matmul.py`:

```python
"""Compile ``rmsnorm(lhs) @ rhs`` from numpy to nkigym, then render the eager kernel.

Pipeline:
    1. ``compile_numpy_to_nkigym`` synthesises an ``f_nkigym`` math
       function from the numpy reference.
    2. The ``f_nkigym`` source is written to ``cache_dir / "f_nkigym.py"``
       and loaded back as a module.
    3. ``render_eager`` lowers the callable to NKI source; the kernel is
       written to ``cache_dir / "kernel.py"``.
    4. The kernel is CPU-simulated via ``nki.simulate`` and numerically
       compared to the numpy reference.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen import render_eager
from nkigym.synthesis import compile_numpy_to_nkigym


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` golden."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = lhs * rms_inv
    return normed @ rhs


def _load_module_from_source(path: Path, mod_name: str) -> object:
    """Load ``path`` as a Python module under name ``mod_name``."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not spec_from_file_location for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    cache_dir = Path("/home/ubuntu/cache/rmsnorm_matmul_compile")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    nkigym_source = compile_numpy_to_nkigym(rmsnorm_matmul_numpy, INPUT_SPECS)
    (cache_dir / "f_nkigym.py").write_text(nkigym_source)

    nkigym_module = _load_module_from_source(cache_dir / "f_nkigym.py", "_f_nkigym_mod")
    f_nkigym = getattr(nkigym_module, "f_nkigym")

    kernel_source = render_eager(f_nkigym, INPUT_SPECS)
    (cache_dir / "kernel.py").write_text(kernel_source)

    """CPU-simulate by executing the rendered source at fp32."""
    sim_source = kernel_source.replace("nl.bfloat16", "nl.float32")
    sim_ns: dict = {}
    exec(sim_source, sim_ns)
    kernel_fn = sim_ns["f_nkigym"]
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((M, K)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs=lhs, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = rmsnorm_matmul_numpy(lhs, rhs)
    max_abs = float(np.abs(actual - expected).max())
    max_rel = float((np.abs(actual - expected) / (np.abs(expected) + 1e-5)).max())
    print(f"[rmsnorm_matmul] max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}")
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3), "CPU-sim mismatch vs numpy golden"
    print(f"[rmsnorm_matmul] kernel written to {cache_dir / 'kernel.py'}")
```

- [ ] **Step 3: Run the example**

Run: `source ~/venvs/kernel-env/bin/activate && python examples/rmsnorm_matmul.py`
Expected: synthesis logs stream; the last two lines print `max_abs_diff=...  max_rel_diff=...` and a kernel-written path. Exit code 0.

If the synthesis loop fails (anthropic API / credits), skip this step and commit anyway — the smoke test in Task 7 exercises the same path with a hand-written `f_nkigym`.

- [ ] **Step 4: Commit**

```bash
git add examples/rmsnorm_matmul.py
git commit -m "Render eager kernel from synthesised rmsnorm+matmul and CPU-sim"
```

---

## Task 9: Port `matmul_lhsT_rhs.py` and `matmul_lhs_rhs.py` to `render_eager`

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py`
- Modify: `examples/matmul_lhs_rhs.py`

- [ ] **Step 1: Rewrite `matmul_lhsT_rhs.py`**

Overwrite `examples/matmul_lhsT_rhs.py`:

```python
"""Render the eager ``lhs_T @ rhs`` kernel and CPU-sim it against numpy.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen import render_eager
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` as an nkigym op DAG."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs_eager")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    source = render_eager(matmul_lhsT_rhs_nkigym, INPUT_SPECS)
    (CACHE_ROOT / "kernel.py").write_text(source)

    sim_source = source.replace("nl.bfloat16", "nl.float32")
    sim_ns: dict = {}
    exec(sim_source, sim_ns)
    kernel_fn = sim_ns["matmul_lhsT_rhs_nkigym"]
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs_T=lhs_T, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs_T.T @ rhs
    print(f"[matmul_lhsT_rhs] max_abs={float(np.abs(actual - expected).max()):.3e}")
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"[matmul_lhsT_rhs] kernel written to {CACHE_ROOT / 'kernel.py'}")
```

- [ ] **Step 2: Rewrite `matmul_lhs_rhs.py`**

Overwrite `examples/matmul_lhs_rhs.py`:

```python
"""Render the eager ``lhs @ rhs`` kernel (with inline Transpose) and CPU-sim.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen import render_eager
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.transpose import NKITranspose


@nkigym_kernel
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """``lhs @ rhs`` via a Transpose staging pass."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    lhs_T = NKITranspose()(data=lhs_sbuf)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs_eager")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    source = render_eager(matmul_lhs_rhs_nkigym, INPUT_SPECS)
    (CACHE_ROOT / "kernel.py").write_text(source)

    sim_source = source.replace("nl.bfloat16", "nl.float32")
    sim_ns: dict = {}
    exec(sim_source, sim_ns)
    kernel_fn = sim_ns["matmul_lhs_rhs_nkigym"]
    rng = np.random.default_rng(1)
    lhs = rng.standard_normal((M, K)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs=lhs, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs @ rhs
    print(f"[matmul_lhs_rhs] max_abs={float(np.abs(actual - expected).max()):.3e}")
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"[matmul_lhs_rhs] kernel written to {CACHE_ROOT / 'kernel.py'}")
```

- [ ] **Step 3: Run both**

```bash
source ~/venvs/kernel-env/bin/activate
python examples/matmul_lhsT_rhs.py
python examples/matmul_lhs_rhs.py
```
Expected: both print `max_abs=...` within tolerance and the kernel file path.

- [ ] **Step 4: Commit**

```bash
git add examples/matmul_lhsT_rhs.py examples/matmul_lhs_rhs.py
git commit -m "Port matmul examples to render_eager with CPU-sim checks"
```

---

## Task 10: Delete `kernel_ir/`, old gadgets, search, online ops, doomed examples

**Files:**
- Delete: `nkigym/src/nkigym/kernel_ir/` (entire tree)
- Delete: `nkigym/src/nkigym/codegen/gadgets.py`
- Delete: `nkigym/src/nkigym/search/` (entire tree)
- Delete: `nkigym/src/nkigym/ops/online_flash_attention.py`
- Delete: `nkigym/src/nkigym/ops/online_matmul.py`
- Delete: `examples/double_matmul.py`
- Delete: `examples/attention.py`
- Delete: `examples/attention_online_fused.py`
- Delete: `examples/rmsnorm_matmul_online.py`
- Delete: `examples/rmsnorm_matmul_online_fused.py`
- Delete: `examples/matmul_lhsT_rhs.md`
- Delete: `examples/matmul_lhs_rhs.md`
- Delete: `examples/rmsnorm_matmul.md`

- [ ] **Step 1: Confirm nothing outside `nkigym/` imports these**

Run:
```bash
grep -rn 'nkigym.kernel_ir\|nkigym.codegen.gadgets\|nkigym.search\|nkigym.ops.online' \
  nkigym/src/nkigym examples autotune 2>/dev/null
```
Expected: hits only in the files being deleted. If any other file references them, fix that file to use `render_eager` or mark the dependency for a follow-up task.

- [ ] **Step 2: Run the delete**

```bash
git rm -r nkigym/src/nkigym/kernel_ir nkigym/src/nkigym/search
git rm nkigym/src/nkigym/codegen/gadgets.py
git rm nkigym/src/nkigym/ops/online_flash_attention.py nkigym/src/nkigym/ops/online_matmul.py
git rm examples/double_matmul.py examples/attention.py examples/attention_online_fused.py \
       examples/rmsnorm_matmul_online.py examples/rmsnorm_matmul_online_fused.py
git rm examples/matmul_lhsT_rhs.md examples/matmul_lhs_rhs.md examples/rmsnorm_matmul.md
```

- [ ] **Step 3: Verify the build is still clean**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/ -v
python examples/matmul_lhsT_rhs.py
python examples/matmul_lhs_rhs.py
```
Expected: tests pass; examples print their `max_abs` lines and exit 0.

- [ ] **Step 4: Commit**

```bash
git commit -m "Delete kernel_ir/, search/, online ops, and deprecated examples"
```

---

## Task 11: Update learnings + final sweep

**Files:**
- Modify: `.claude/rules/learnings.md`

- [ ] **Step 1: Append a learnings entry**

Open `/home/ubuntu/nki-autotune/.claude/rules/learnings.md` and add one bullet
under the appropriate section summarising the new pipeline:

```markdown
- **Eager renderer**: `render_eager(f_nkigym, input_specs)` = `render(parse_and_resolve(func, specs))`. Each NKIOp becomes an independent loop nest over its own `touched_dims`; SBUF intermediates are 3D `(p_tile, num_p_tiles, num_f_tiles*f_tile)` hoisted to function top. HBM only at NKILoad/NKIStore. No knobs. Extension seam: `OpGraph.per_op_attrs[idx]` for future `propagate_compute_skip`-style passes. *(2026-05-05)*
```

Update the `*Last updated:*` footer to `2026-05-05`.

- [ ] **Step 2: Confirm the final state**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/ -v
python examples/matmul_lhsT_rhs.py
python examples/matmul_lhs_rhs.py
```
All green; assertions hold.

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/learnings.md
git commit -m "Record eager-renderer learnings"
```

---

## Self-review checklist

- Spec §2 two-module pipeline → Task 1–3 build `graph.py`, Task 4–6 build `render.py`.
- Spec §3 dataclasses → Task 1.
- Spec §4 parse + dim resolution → Tasks 2 and 3.
- Spec §5 kernel scaffolding + per-op dispatch → Tasks 4, 5, 6a–6e.
- Spec §6 verification → Task 7 smoke test + Task 8 example.
- Spec §7 extension hook (`per_op_attrs`) → Task 1 includes the field with default `{}`.
- Spec §8 file changes → Tasks 8–10 perform every listed change.

All tasks include full code, explicit file paths, and expected pytest/shell outputs. No placeholders or "similar to" pointers. Method names stay consistent: `parse_and_resolve`, `render`, `render_eager`, `_sbuf_tile_slice`, `_hbm_tile_slice`, `_emit_<name>`, `_register_emitter`, `_Writer`.
