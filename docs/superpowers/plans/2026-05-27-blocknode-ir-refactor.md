# BlockNode IR Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `RootNode | ForNode | ISANode` IR with a TVM-aligned `BlockNode`-based IR that owns iter_vars, iter_values, declared reads/writes, alloc_buffers, and an optional reduction init sub-block.

**Architecture:** Add new payload types (`BlockNode`, `IterVar`, `BufferRegion`, `Buffer`) and a hand-rolled affine `Expr` AST alongside the existing types. Migrate the canonical builder to emit the new tree. Migrate the renderer next, gated by the existing 2048³ matmul golden and numerics test. Migrate the dependency graph, then Split / Fuse / Reorder one transform per commit. Finally drop the legacy types and `NKIAlloc` op. Compute_at lands in a follow-up spec on this substrate.

**Tech Stack:** Python 3.12, networkx (DAG backend), pytest, numpy, NKI CPU simulator. Hand-rolled AST + normaliser; no new external deps.

**Spec:** `docs/superpowers/specs/2026-05-27-blocknode-ir-refactor-design.md`.

**Environment:** Activate the kernel venv before running anything: `source ~/venvs/kernel-env/bin/activate`. All commands below assume the venv is active and the working directory is the repo root.

---

## File Structure

New files:
- `nkigym/src/nkigym/ir/expr.py` — affine `Expr` AST (`Const`, `Var`, `Add`, `Mul`, `FloorDiv`, `Mod`), normaliser, substitution, pretty-printer.
- `test/ir/test_expr.py` — `Expr` tests.

Modified files (one or more times across the plan):
- `nkigym/src/nkigym/ir/tree.py` — add `BlockNode`, `IterVar`, `BufferRegion`, `Buffer`; change `ForNode` payload to `(loop_var, extent)`; change `ISANode` to `(op_cls, operand_bindings, kwargs)`; add `KernelTree.blocks()` helper.
- `nkigym/src/nkigym/ir/__init__.py` — export new types.
- `nkigym/src/nkigym/ir/dimension_analysis.py` — emit new IR; pattern-detect memset+rmw → init; buffer scope analysis.
- `nkigym/src/nkigym/ir/ir.py` — drop `dim_sizes` and `tensors`; add `all_buffers` / `buffer` / `axis_extent` helpers.
- `nkigym/src/nkigym/ir/dependency.py` — block-keyed graph; per-region overlap.
- `nkigym/src/nkigym/ir/tree_visualize.py` — render block boxes; init dashed edge.
- `nkigym/src/nkigym/codegen/render.py`, `body.py`, `header.py` — block-driven emission; alloc from `alloc_buffers`; init emission; `BufferRegion` → slice.
- `nkigym/src/nkigym/transforms/split.py`, `fuse.py`, `reorder.py`, `_tree_ops.py`, `__init__.py` — operate on iter_values + extents.
- `nkigym/src/nkigym/ops/alloc.py` — REMOVED.
- `nkigym/src/nkigym/ops/__init__.py` — drop `NKIAlloc` export.
- `examples/matmul_lhsT_rhs.py` — drop `NKIAlloc(...)()` lines from `f_nkigym` source.
- `nkigym/src/nkigym/transforms/compute_at_legality.md` — update "What is a block?" to point at `BlockNode`.
- `test/ir/test_ir_extensions.py` — rewrite for new payloads.
- `test/ir/test_role_of.py` — `role_of` source moves from `op_cls.AXIS_ROLES` to `BlockNode.iter_vars[i].role`; tests update.
- `test/codegen/test_*.py`, `test/transforms/test_*.py`, `test/transforms/_fixtures.py`, `test/transforms/_seq_fixture.py` — rewrite fixtures and assertions.

---

## Phase outline (one phase = one or more tasks)

1. **Expr AST + tests** — Tasks 1–4.
2. **New payload types alongside old** — Tasks 5–7.
3. **Canonical builder migrates to new IR** — Tasks 8–11.
4. **Renderer migrates** — Tasks 12–15.
5. **Dependency graph migrates** — Tasks 16–17.
6. **Split migrates** — Task 18.
7. **Fuse migrates** — Task 19.
8. **Reorder migrates** — Task 20.
9. **Drop legacy types and NKIAlloc** — Tasks 21–23.
10. **Update legality doc + final E2E gate** — Tasks 24–25.

---

## Phase 1 — Expr AST

Hand-rolled affine integer AST with a normaliser. Sufficient for every binding (iter_value as a function of loop_vars) and every `BufferRegion` range our transforms emit.

### Task 1: Add `Expr` payload classes

**Files:**
- Create: `nkigym/src/nkigym/ir/expr.py`
- Test: `test/ir/test_expr.py`

- [ ] **Step 1: Write the failing test for constructors and structural equality**

```python
"""Tests for nkigym.ir.expr."""

from __future__ import annotations

import pytest

from nkigym.ir.expr import Add, Const, FloorDiv, Mod, Mul, NonAffineError, Var, from_affine, substitute, to_affine


def test_const_and_var_construction():
    """Const and Var are frozen dataclasses with structural equality."""
    assert Const(value=3) == Const(value=3)
    assert Const(value=3) != Const(value=4)
    assert Var(name="i") == Var(name="i")
    assert Var(name="i") != Var(name="j")


def test_compound_expression_construction():
    """Compound expressions compose Add/Mul/FloorDiv/Mod recursively."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert isinstance(expr, Add)
    assert isinstance(expr.left, Mul)
    assert expr.right == Var(name="j")


def test_expr_is_hashable():
    """Frozen dataclasses are hashable; equal exprs hash equal."""
    e1 = Add(left=Var(name="i"), right=Const(value=1))
    e2 = Add(left=Var(name="i"), right=Const(value=1))
    assert hash(e1) == hash(e2)
    assert {e1: 1, e2: 2} == {e1: 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_expr.py -v`
Expected: FAIL with `ImportError: No module named 'nkigym.ir.expr'`.

- [ ] **Step 3: Implement the Expr payload classes**

Create `nkigym/src/nkigym/ir/expr.py`:

```python
"""Affine integer Expression AST for iter_values and BufferRegion ranges.

Supports affine combinations of integer-typed Vars: Const, Var, Add,
Mul, FloorDiv, Mod. Sufficient for every binding and region range our
canonical builder and transforms emit.

Non-affine inputs (Var * Var, Mod / FloorDiv with non-Const divisor)
raise :class:`NonAffineError` from :func:`to_affine`.
"""

from __future__ import annotations

from dataclasses import dataclass


class NonAffineError(ValueError):
    """Raised when ``to_affine`` encounters a pattern that is not affine in Vars."""


@dataclass(frozen=True, kw_only=True)
class Const:
    """Integer literal."""

    value: int


@dataclass(frozen=True, kw_only=True)
class Var:
    """Symbolic variable identified by ``name``."""

    name: str


@dataclass(frozen=True, kw_only=True)
class Add:
    """Binary addition."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Mul:
    """Binary multiplication. At most one operand may contain a Var (affinity)."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class FloorDiv:
    """Floor division. ``right`` must reduce to a non-zero ``Const`` for affinity."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Mod:
    """Modulo. ``right`` must reduce to a non-zero ``Const`` for affinity."""

    left: "Expr"
    right: "Expr"


Expr = Const | Var | Add | Mul | FloorDiv | Mod


def to_affine(expr: Expr) -> dict[str | None, int]:
    """Collapse ``expr`` to canonical affine form ``c0 + c1*v1 + c2*v2 + ...``.

    The returned dict maps variable names to integer coefficients; the
    constant term lives under key ``None``. Raises :class:`NonAffineError`
    on patterns we don't support.
    """
    raise NotImplementedError


def from_affine(coeffs: dict[str | None, int]) -> Expr:
    """Inverse of :func:`to_affine`."""
    raise NotImplementedError


def substitute(expr: Expr, subs: dict[str, Expr]) -> Expr:
    """Replace each ``Var(name)`` in ``expr`` by ``subs[name]`` recursively."""
    raise NotImplementedError


__all__ = [
    "Add",
    "Const",
    "Expr",
    "FloorDiv",
    "Mod",
    "Mul",
    "NonAffineError",
    "Var",
    "from_affine",
    "substitute",
    "to_affine",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_expr.py::test_const_and_var_construction test/ir/test_expr.py::test_compound_expression_construction test/ir/test_expr.py::test_expr_is_hashable -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/expr.py test/ir/test_expr.py
git commit -m "Add Expr payload classes (Const, Var, Add, Mul, FloorDiv, Mod)"
```

### Task 2: Implement `to_affine` / `from_affine` round-trip

**Files:**
- Modify: `nkigym/src/nkigym/ir/expr.py`
- Test: `test/ir/test_expr.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_expr.py`:

```python
def test_to_affine_const_only():
    """A bare Const collapses to {None: value}."""
    assert to_affine(Const(value=7)) == {None: 7}


def test_to_affine_var_only():
    """A bare Var collapses to {name: 1}."""
    assert to_affine(Var(name="i")) == {"i": 1}


def test_to_affine_linear_combination():
    """i*8 + j collapses to {'i': 8, 'j': 1}."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert to_affine(expr) == {"i": 8, "j": 1}


def test_to_affine_zero_coefficients_dropped():
    """0*i + 5 collapses to {None: 5} (zero coefficients are pruned)."""
    expr = Add(left=Mul(left=Const(value=0), right=Var(name="i")), right=Const(value=5))
    assert to_affine(expr) == {None: 5}


def test_to_affine_rejects_var_times_var():
    """Var * Var is non-affine."""
    with pytest.raises(NonAffineError):
        to_affine(Mul(left=Var(name="i"), right=Var(name="j")))


def test_to_affine_rejects_var_in_mod_divisor():
    """Mod with a non-Const divisor is non-affine."""
    with pytest.raises(NonAffineError):
        to_affine(Mod(left=Var(name="i"), right=Var(name="j")))


def test_from_affine_round_trip():
    """from_affine(to_affine(e)) is structurally equal to a canonical form of e."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    coeffs = to_affine(expr)
    rebuilt = from_affine(coeffs)
    assert to_affine(rebuilt) == coeffs


def test_from_affine_constant_only():
    """from_affine({None: 5}) is Const(5)."""
    assert from_affine({None: 5}) == Const(value=5)


def test_from_affine_single_var():
    """from_affine({'i': 1}) is Var(i); from_affine({'i': 3}) is Mul(Var(i), Const(3))."""
    assert from_affine({"i": 1}) == Var(name="i")
    assert from_affine({"i": 3}) == Mul(left=Var(name="i"), right=Const(value=3))


def test_from_affine_empty_is_zero():
    """from_affine({}) == Const(0) (empty sum)."""
    assert from_affine({}) == Const(value=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_expr.py -v -k "to_affine or from_affine"`
Expected: 9 failed with `NotImplementedError`.

- [ ] **Step 3: Implement `to_affine` and `from_affine`**

Replace the two `raise NotImplementedError` stubs in `nkigym/src/nkigym/ir/expr.py`:

```python
def to_affine(expr: Expr) -> dict[str | None, int]:
    """Collapse ``expr`` to canonical affine form ``c0 + c1*v1 + c2*v2 + ...``.

    The returned dict maps variable names to integer coefficients; the
    constant term lives under key ``None``. Zero coefficients are
    pruned. Raises :class:`NonAffineError` on patterns we don't
    support (Var * Var, FloorDiv / Mod with a non-constant divisor).
    """
    coeffs = _accumulate(expr)
    return {k: v for k, v in coeffs.items() if v != 0}


def _accumulate(expr: Expr) -> dict[str | None, int]:
    """Recurse into ``expr`` and return its raw affine coefficients.

    Internal helper for :func:`to_affine` that does not prune zeroes
    so callers can detect ``0`` outputs (e.g. for ``from_affine``
    round-trip). Raises :class:`NonAffineError` on non-affine patterns.
    """
    if isinstance(expr, Const):
        return {None: expr.value}
    if isinstance(expr, Var):
        return {expr.name: 1}
    if isinstance(expr, Add):
        return _add(_accumulate(expr.left), _accumulate(expr.right))
    if isinstance(expr, Mul):
        return _mul(_accumulate(expr.left), _accumulate(expr.right))
    if isinstance(expr, FloorDiv):
        right = _accumulate(expr.right)
        if set(right) - {None}:
            raise NonAffineError(f"FloorDiv divisor is not constant: {expr.right}")
        divisor = right.get(None, 0)
        if divisor == 0:
            raise NonAffineError("FloorDiv by zero")
        left = _accumulate(expr.left)
        for var, coeff in left.items():
            if coeff % divisor != 0:
                raise NonAffineError(
                    f"FloorDiv coefficient {coeff} of {var} not divisible by {divisor}"
                )
        return {var: coeff // divisor for var, coeff in left.items()}
    if isinstance(expr, Mod):
        right = _accumulate(expr.right)
        if set(right) - {None}:
            raise NonAffineError(f"Mod divisor is not constant: {expr.right}")
        divisor = right.get(None, 0)
        if divisor == 0:
            raise NonAffineError("Mod by zero")
        left = _accumulate(expr.left)
        if set(left) - {None}:
            raise NonAffineError(f"Mod left side is not constant: {expr.left}")
        return {None: left.get(None, 0) % divisor}
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")


def _add(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise sum of two affine coefficient maps."""
    out = dict(a)
    for var, coeff in b.items():
        out[var] = out.get(var, 0) + coeff
    return out


def _mul(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise product. At most one operand may contain a non-None key."""
    a_vars = set(a) - {None}
    b_vars = set(b) - {None}
    if a_vars and b_vars:
        raise NonAffineError(f"Var * Var: {sorted(a_vars)} times {sorted(b_vars)}")
    if not a_vars:
        scale = a.get(None, 0)
        return {var: coeff * scale for var, coeff in b.items()}
    scale = b.get(None, 0)
    return {var: coeff * scale for var, coeff in a.items()}


def from_affine(coeffs: dict[str | None, int]) -> Expr:
    """Inverse of :func:`to_affine`. Returns a canonical-form Expr.

    Variables are emitted in sorted name order so equal coefficient
    maps produce structurally equal Exprs.
    """
    terms: list[Expr] = []
    var_names = sorted(name for name in coeffs if name is not None)
    for name in var_names:
        coeff = coeffs[name]
        if coeff == 0:
            continue
        if coeff == 1:
            terms.append(Var(name=name))
        else:
            terms.append(Mul(left=Var(name=name), right=Const(value=coeff)))
    constant = coeffs.get(None, 0)
    if constant != 0 or not terms:
        terms.append(Const(value=constant))
    result = terms[0]
    for term in terms[1:]:
        result = Add(left=result, right=term)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_expr.py -v`
Expected: 12 passed (3 from Task 1 + 9 here).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/expr.py test/ir/test_expr.py
git commit -m "Add to_affine and from_affine for the Expr AST"
```

### Task 3: Implement `substitute`

**Files:**
- Modify: `nkigym/src/nkigym/ir/expr.py`
- Test: `test/ir/test_expr.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_expr.py`:

```python
def test_substitute_simple_var():
    """substitute({'i': Const(7)}) into Var('i') returns Const(7)."""
    assert substitute(Var(name="i"), {"i": Const(value=7)}) == Const(value=7)


def test_substitute_passes_through_other_vars():
    """substitute leaves non-substituted Vars alone."""
    expr = Add(left=Var(name="i"), right=Var(name="j"))
    result = substitute(expr, {"i": Const(value=7)})
    assert result == Add(left=Const(value=7), right=Var(name="j"))


def test_substitute_into_compound_expression():
    """Substituting i with i_outer*8 + i_inner inside i*128 + j gives the expected affine form."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=128)), right=Var(name="j"))
    sub = Add(left=Mul(left=Var(name="i_outer"), right=Const(value=8)), right=Var(name="i_inner"))
    result = substitute(expr, {"i": sub})
    """The result, when normalised, should equal i_outer*1024 + i_inner*128 + j."""
    expected_coeffs = {"i_outer": 1024, "i_inner": 128, "j": 1}
    assert to_affine(result) == expected_coeffs


def test_substitute_unaffected_passes_through():
    """substitute on a Const returns the same Const."""
    assert substitute(Const(value=5), {"i": Const(value=7)}) == Const(value=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_expr.py -v -k "substitute"`
Expected: 4 failed with `NotImplementedError`.

- [ ] **Step 3: Implement `substitute`**

Replace the `substitute` stub in `nkigym/src/nkigym/ir/expr.py`:

```python
def substitute(expr: Expr, subs: dict[str, Expr]) -> Expr:
    """Replace each ``Var(name)`` in ``expr`` by ``subs[name]`` recursively.

    Variables not present in ``subs`` are left unchanged. The returned
    expression is not normalised; pipe through ``from_affine(to_affine(...))``
    if a canonical form is needed.
    """
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Var):
        return subs.get(expr.name, expr)
    if isinstance(expr, Add):
        return Add(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, Mul):
        return Mul(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, FloorDiv):
        return FloorDiv(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, Mod):
        return Mod(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_expr.py -v`
Expected: 16 passed (12 from prior tasks + 4 here).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/expr.py test/ir/test_expr.py
git commit -m "Add Expr.substitute"
```

### Task 4: Pretty-printer for the renderer

**Files:**
- Modify: `nkigym/src/nkigym/ir/expr.py`
- Test: `test/ir/test_expr.py`

The renderer needs to format `Expr` as a Python source string. The pretty-printer must round-trip through `to_affine` for canonical bindings (e.g. `i*8 + j` rather than `j + i*8`) so golden tests are stable.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_expr.py`:

```python
def test_format_const():
    """Const(5) formats as '5'."""
    from nkigym.ir.expr import format_expr

    assert format_expr(Const(value=5)) == "5"


def test_format_var():
    """Var('i') formats as 'i'."""
    from nkigym.ir.expr import format_expr

    assert format_expr(Var(name="i")) == "i"


def test_format_affine_combination():
    """An affine combination formats with terms in sorted-name order, then constant."""
    from nkigym.ir.expr import format_expr

    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert format_expr(expr) == "i * 8 + j"


def test_format_negative_constant():
    """Negative constants surface inline (not normalised away)."""
    from nkigym.ir.expr import format_expr

    expr = Add(left=Var(name="i"), right=Const(value=-3))
    assert format_expr(expr) == "i + -3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_expr.py -v -k "format"`
Expected: 4 failed with `ImportError: cannot import name 'format_expr'`.

- [ ] **Step 3: Implement `format_expr`**

Append to `nkigym/src/nkigym/ir/expr.py`:

```python
def format_expr(expr: Expr) -> str:
    """Pretty-print ``expr`` as Python source.

    Normalises through :func:`to_affine` / :func:`from_affine` first
    so bindings render in a deterministic, sorted-name canonical form.
    Variables come first in sorted name order; the constant term
    trails (omitted if zero or no other terms exist).
    """
    canonical = from_affine(to_affine(expr))
    return _format_raw(canonical)


def _format_raw(expr: Expr) -> str:
    """Format an Expr without prior normalisation. Internal helper."""
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Add):
        return f"{_format_raw(expr.left)} + {_format_raw(expr.right)}"
    if isinstance(expr, Mul):
        return f"{_format_raw(expr.left)} * {_format_raw(expr.right)}"
    if isinstance(expr, FloorDiv):
        return f"{_format_raw(expr.left)} // {_format_raw(expr.right)}"
    if isinstance(expr, Mod):
        return f"{_format_raw(expr.left)} % {_format_raw(expr.right)}"
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")
```

Update the `__all__` list at the bottom of the file:

```python
__all__ = [
    "Add",
    "Const",
    "Expr",
    "FloorDiv",
    "Mod",
    "Mul",
    "NonAffineError",
    "Var",
    "format_expr",
    "from_affine",
    "substitute",
    "to_affine",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_expr.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/expr.py test/ir/test_expr.py
git commit -m "Add format_expr pretty-printer for Expr"
```

---

## Phase 2 — New payload types alongside old

We add `BlockNode`, `IterVar`, `BufferRegion`, and `Buffer` as new payload classes inside `nkigym/ir/tree.py`. The `KernelTree` class is untouched in this phase — only the payload union grows. A new `KernelTree.blocks()` helper is added. Legacy `ForNode(dim, trip)` and `ISANode(reads, writes, rmw, axis_map, tensorize_sizes, ...)` remain in place; the canonical builder still emits them. We change `ForNode` and `ISANode` payload shape in Phase 3 (canonical builder cutover) so the renderer and transforms see the new fields atomically.

### Task 5: Add `IterVar`, `BufferRegion`, `Buffer` payload classes

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_itervar_constructor_and_equality():
    """IterVar is a frozen dataclass with structural equality."""
    from nkigym.ir.tree import IterVar
    from nkigym.ops.base import AxisRole

    a = IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    b = IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    c = IterVar(axis="M", dom=(0, 2048), role=AxisRole.ACCUMULATION)
    assert a == b
    assert a != c


def test_buffer_constructor_and_equality():
    """Buffer is a frozen dataclass with structural equality."""
    from nkigym.ir.tree import Buffer

    a = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    b = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    assert a == b


def test_bufferregion_constructor_and_equality():
    """BufferRegion is a frozen dataclass with structural equality on tensor + ranges."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion

    a = BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))
    b = BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py -v -k "itervar or buffer or bufferregion"`
Expected: 3 failed with `ImportError: cannot import name 'IterVar'`.

- [ ] **Step 3: Implement the payloads**

Append to `nkigym/src/nkigym/ir/tree.py` *after* the existing `ISANode` definition (do not remove the existing payloads):

```python
from nkigym.ir.expr import Expr


@dataclass(frozen=True, kw_only=True)
class IterVar:
    """Per-block iteration variable.

    Attributes:
        axis: abstract axis name (``"M"``, ``"K"``, ``"P"``, ...).
        dom: half-open extent ``(lo, hi)``.
        role: ``PARALLEL`` (TVM ``kDataPar``) / ``ACCUMULATION``
            (``kCommReduce``) / ``SEQUENTIAL`` (``kOrdered``).
    """

    axis: str
    dom: tuple[int, int]
    role: AxisRole


@dataclass(frozen=True, kw_only=True)
class Buffer:
    """Buffer declaration on an enclosing :class:`BlockNode`.

    Replaces the standalone :class:`NKIAlloc` ISA leaf. The lifetime is
    bounded by the declaring block.

    Attributes:
        name: tensor name.
        shape: per-axis extent.
        dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"``.
        location: ``"shared_hbm"`` / ``"sbuf"`` / ``"psum"``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str


@dataclass(frozen=True, kw_only=True)
class BufferRegion:
    """Affine half-open region of a buffer, expressed in iter_var ``Var``s.

    Attributes:
        tensor: tensor name (key into the kernel's buffers).
        ranges: one ``(lo, hi)`` pair per axis, in iter_var-Var space.
            For a single-element access, ``hi`` is ``lo + 1``; for a
            tile, ``hi`` is ``lo + tile_size``.
    """

    tensor: str
    ranges: tuple[tuple[Expr, Expr], ...]
```

Update the existing module docstring at the top of `tree.py` to mention the new payload types:

```python
"""Canonical schedule tree for an ``f_nkigym`` kernel, backed by ``networkx``.

The tree is stored as an ``nx.DiGraph`` where every node is a stable
integer id and the payload lives at ``graph.nodes[id]["data"]``. Payload
dataclasses discriminate the node kind:

* :class:`RootNode` — dummy root of the forest.
* :class:`BlockNode` — TVM-style schedulable unit owning iter_vars,
  declared reads / writes, and optional ``init`` / ``alloc_buffers``.
* :class:`ForNode` — a loop binding to (part of) a block iter_var.
* :class:`ISANode` — a single NKI instruction.

:class:`IterVar`, :class:`BufferRegion`, and :class:`Buffer` are
sub-payloads carried on :class:`BlockNode` and :class:`ISANode`.

:class:`KernelTree` wraps the graph with a small traversal surface
(``children``, ``parent``, ``ancestors``, ``descendants``, ``leaves``,
``preorder``, ``blocks``) so downstream atoms don't have to touch
``networkx`` directly. :func:`build_initial_tree` walks an
``@nkigym_kernel`` callable via :func:`nkigym.ir.dimension_analysis.analyze_dimensions`.
Visualization helpers live in :mod:`nkigym.ir.tree_visualize`.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py -v -k "itervar or buffer or bufferregion"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py
git commit -m "Add IterVar, BufferRegion, Buffer payload classes"
```

### Task 6: Add `BlockNode` payload

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_blocknode_constructor_minimal():
    """A BlockNode with no iter_vars is the legal empty case (e.g. the synthetic root block)."""
    from nkigym.ir.tree import BlockNode

    node = BlockNode(
        iter_vars=(),
        iter_values=(),
        reads=(),
        writes=(),
        alloc_buffers=(),
        init=None,
    )
    assert node.iter_vars == ()
    assert node.alloc_buffers == ()
    assert node.init is None
    assert node.annotations == {}


def test_blocknode_constructor_full():
    """A BlockNode with iter_vars and a region carries every field."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, IterVar
    from nkigym.ops.base import AxisRole

    block = BlockNode(
        iter_vars=(
            IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL),
            IterVar(axis="N", dom=(0, 2048), role=AxisRole.PARALLEL),
        ),
        iter_values=(Var(name="i_M"), Var(name="i_N")),
        reads=(),
        writes=(BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),)),),
        alloc_buffers=(Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum"),),
        init=None,
    )
    assert len(block.iter_vars) == 2
    assert len(block.alloc_buffers) == 1


def test_blocknode_default_annotations_is_fresh_dict():
    """Two default-constructed BlockNodes don't share an annotations dict."""
    from nkigym.ir.tree import BlockNode

    a = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    b = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    a.annotations["k"] = 1
    assert "k" not in b.annotations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py -v -k "blocknode"`
Expected: 3 failed with `ImportError: cannot import name 'BlockNode'`.

- [ ] **Step 3: Implement `BlockNode`**

Append to `nkigym/src/nkigym/ir/tree.py` after the `BufferRegion` definition added in Task 5:

```python
@dataclass(frozen=True, kw_only=True)
class BlockNode:
    """TVM-style block — schedulable unit aligned with ``tir.SBlockNode``.

    Attributes:
        iter_vars: per-axis iter_vars owned by this block.
        iter_values: one Expr per iter_var (in iter_vars order) mapping
            surrounding ``ForNode.loop_var`` symbols to iter_var values.
        reads: declared read regions in iter_var space.
        writes: declared write regions in iter_var space.
        alloc_buffers: buffers whose lifetime is bounded by this block.
        init: nid of the optional reduction-init sub-block. ``None`` for
            non-reduction blocks. The init block has a *prefix* of this
            block's iter_vars (spatial-only).
        annotations: free-form per-block metadata.
    """

    iter_vars: tuple[IterVar, ...]
    iter_values: tuple[Expr, ...]
    reads: tuple[BufferRegion, ...]
    writes: tuple[BufferRegion, ...]
    alloc_buffers: tuple[Buffer, ...] = ()
    init: int | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py -v -k "blocknode"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py
git commit -m "Add BlockNode payload"
```

### Task 7: Add `KernelTree.blocks()` helper and export new types

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py`
- Modify: `nkigym/src/nkigym/ir/__init__.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_kerneltree_blocks_helper_yields_blocknode_nids():
    """blocks() walks the tree in pre-order and yields nids whose payload is a BlockNode."""
    from nkigym.ir.tree import BlockNode, KernelTree

    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    nid_a = tree.add_node(block, parent=tree.root)
    nid_b = tree.add_node(block, parent=tree.root)

    out = list(tree.blocks())
    assert out == [nid_a, nid_b]


def test_kerneltree_blocks_skips_non_block_payloads():
    """blocks() ignores RootNode, ForNode, ISANode."""
    from nkigym.ir.tree import BlockNode, ForNode, KernelTree

    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    block_nid = tree.add_node(block, parent=tree.root)
    tree.add_node(ForNode(dim="d0", trip=2), parent=block_nid)

    out = list(tree.blocks())
    assert out == [block_nid]


def test_ir_module_exports_new_payloads():
    """The ir package re-exports the new payload classes."""
    from nkigym.ir import BlockNode, Buffer, BufferRegion, IterVar  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py -v -k "blocks or exports"`
Expected: 3 failed (no `blocks` method; missing exports).

- [ ] **Step 3: Implement `blocks()` and update exports**

Add the helper method to `KernelTree` in `nkigym/src/nkigym/ir/tree.py`, immediately after the existing `leaves` method:

```python
    def blocks(self, nid: int | None = None) -> Iterator[int]:
        """Yield ``BlockNode``-bearing nids in pre-order DFS from ``nid``.

        Convenience for transforms that walk blocks rather than ISA leaves.
        ``nid`` defaults to the root.
        """
        for m in self.preorder(nid):
            if isinstance(self.data(m), BlockNode):
                yield m
```

Update the `__all__` list at the bottom of `tree.py`:

```python
__all__ = [
    "BlockNode",
    "Buffer",
    "BufferRegion",
    "ForNode",
    "ISANode",
    "IterVar",
    "KernelTree",
    "NodeData",
    "RootNode",
    "build_initial_tree",
    "role_of",
]
```

Also update the `NodeData` union near the bottom of `tree.py` (above `__all__`):

```python
NodeData = RootNode | BlockNode | ForNode | ISANode
```

Update `nkigym/src/nkigym/ir/__init__.py` to re-export the new types:

```python
"""Dim unification analysis + canonical schedule tree for an ``f_nkigym`` callable."""

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import TensorDims
from nkigym.ir.expr import Expr
from nkigym.ir.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import (
    BlockNode,
    Buffer,
    BufferRegion,
    ForNode,
    ISANode,
    IterVar,
    KernelTree,
    NodeData,
    RootNode,
    build_initial_tree,
)

__all__ = [
    "BlockNode",
    "Buffer",
    "BufferRegion",
    "Dependency",
    "Expr",
    "ForNode",
    "ISANode",
    "IterVar",
    "KernelIR",
    "KernelTree",
    "NodeData",
    "RootNode",
    "TensorDims",
    "build_initial_ir",
    "build_initial_tree",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py -v`
Expected: All passing (existing tests + 3 new in this task).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/ir/__init__.py test/ir/test_ir_extensions.py
git commit -m "Add KernelTree.blocks() helper; export new IR types"
```

---

## Phase 3 — Canonical builder cutover

This is the largest phase. We change `ForNode` and `ISANode` payload shapes, rewrite the canonical builder to emit `BlockNode`-rooted trees, and slim down `KernelIR`. Renderer, dependency graph, and transforms break temporarily — we'll bring them back online in Phases 4–8.

To keep the cutover gated: during this phase, the canonical-builder smoke test (Task 11) is the gate. Renderer / transform tests are knowingly red and stay red until Phases 4–8 complete.

### Task 8: Change `ForNode` payload to `(loop_var, extent)`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py`
- Test: `test/ir/test_ir_extensions.py`

The new `ForNode(loop_var, extent)` replaces today's `ForNode(dim, trip)`. The renderer and transforms still use `dim`/`trip` — we'll rewrite them in later phases. This task only changes the payload class.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_fornode_payload_uses_loop_var_and_extent():
    """ForNode carries (loop_var, extent), not (dim, trip)."""
    from nkigym.ir.tree import ForNode

    node = ForNode(loop_var="i_M_0", extent=16)
    assert node.loop_var == "i_M_0"
    assert node.extent == 16
    """Old field names must NOT be present."""
    assert not hasattr(node, "dim")
    assert not hasattr(node, "trip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py::test_fornode_payload_uses_loop_var_and_extent -v`
Expected: FAIL — current `ForNode` has `dim` and `trip` fields.

- [ ] **Step 3: Update `ForNode`**

In `nkigym/src/nkigym/ir/tree.py`, replace the existing `ForNode` definition with:

```python
@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Loop binding to one (or part of one) :class:`BlockNode` iter_var.

    Multiple same-axis ``ForNode``s above one block — the result of
    :class:`Split` — bind the iter_var via the affine combination
    encoded in the enclosing block's ``iter_values``.

    Attributes:
        loop_var: symbolic name (e.g. ``"i_M_outer"``).
        extent: loop trip count.
    """

    loop_var: str
    extent: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py::test_fornode_payload_uses_loop_var_and_extent -v`
Expected: PASS.

- [ ] **Step 5: Verify the codebase no longer compiles against `dim`/`trip`**

Run: `grep -rn "ForNode(dim=\|\.dim\b\|\.trip\b" nkigym/src/nkigym/ test/`
Expected: Many hits — all in legacy code (renderer, transforms, fixtures, dimension_analysis). These will be rewritten in subsequent tasks. We do NOT fix them in this task — they stay broken and are repaired phase-by-phase.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py
git commit -m "Switch ForNode payload to (loop_var, extent)"
```

### Task 9: Change `ISANode` payload to `(op_cls, operand_bindings, kwargs)`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_isanode_payload_uses_operand_bindings():
    """ISANode carries (op_cls, operand_bindings, kwargs); legacy fields removed."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion, ISANode
    from nkigym.ops.matmul import NKIMatmul

    bindings = {
        "stationary": BufferRegion(
            tensor="sbuf_lhs_T",
            ranges=((Var(name="vK"), Const(value=1)), (Var(name="vM"), Const(value=128))),
        ),
        "moving": BufferRegion(
            tensor="sbuf_rhs",
            ranges=((Var(name="vK"), Const(value=1)), (Var(name="vN"), Const(value=512))),
        ),
        "dst": BufferRegion(
            tensor="psum_prod",
            ranges=((Var(name="vM"), Const(value=128)), (Var(name="vN"), Const(value=512))),
        ),
    }
    node = ISANode(op_cls=NKIMatmul, operand_bindings=bindings, kwargs={})
    assert node.op_cls is NKIMatmul
    assert set(node.operand_bindings) == {"stationary", "moving", "dst"}
    """Legacy fields must NOT be present."""
    for old in ("reads", "writes", "rmw", "axis_map", "tensorize_sizes", "location", "dtype"):
        assert not hasattr(node, old), f"ISANode unexpectedly carries legacy field {old!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py::test_isanode_payload_uses_operand_bindings -v`
Expected: FAIL — current `ISANode` has `reads`, `writes`, etc.

- [ ] **Step 3: Update `ISANode`**

In `nkigym/src/nkigym/ir/tree.py`, replace the existing `ISANode` definition with:

```python
@dataclass(frozen=True, kw_only=True)
class ISANode:
    """Single ISA call.

    Attributes:
        op_cls: :class:`NKIOp` subclass.
        operand_bindings: per-slot :class:`BufferRegion` in the
            enclosing :class:`BlockNode`'s iter_var space.
        kwargs: non-operand call kwargs (e.g. ``{"value": 0.0}`` for
            :class:`NKIMemset`).
    """

    op_cls: type[NKIOp]
    operand_bindings: dict[str, BufferRegion] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Update the `role_of` helper**

The existing `role_of(leaf, concrete_dim)` uses `leaf.axis_map`. After the cutover, role lookup is on `BlockNode.iter_vars`. Move the helper to take a block + axis name:

In `nkigym/src/nkigym/ir/tree.py`, replace the existing `role_of` function with:

```python
def role_of(block: BlockNode, axis: str) -> AxisRole:
    """Return the role this block assigns to ``axis``.

    Searches ``block.iter_vars`` for the entry whose ``axis`` matches.
    Raises :class:`KeyError` if the block does not declare that axis.
    """
    for iv in block.iter_vars:
        if iv.axis == axis:
            return iv.role
    raise KeyError(f"BlockNode does not declare axis {axis!r}")
```

- [ ] **Step 5: Rewrite `test/ir/test_role_of.py` for the new `role_of(block, axis)` signature**

The existing tests build an `ISANode` with `axis_map` and call `role_of(leaf, concrete_dim)`. After this task `ISANode` has no `axis_map` and `role_of` takes a `BlockNode`. Replace the file's contents with:

```python
"""Unit tests for :func:`nkigym.ir.tree.role_of`."""

from __future__ import annotations

import pytest

from nkigym.ir.tree import BlockNode, IterVar, role_of
from nkigym.ops.base import AxisRole


def _make_block(roles: dict[str, AxisRole]) -> BlockNode:
    """Build a minimal :class:`BlockNode` with iter_vars whose roles come from ``roles``."""
    iter_vars = tuple(IterVar(axis=name, dom=(0, 128), role=role) for name, role in roles.items())
    return BlockNode(
        iter_vars=iter_vars,
        iter_values=(),
        reads=(),
        writes=(),
    )


def test_role_of_returns_parallel_when_declared() -> None:
    block = _make_block({"P": AxisRole.PARALLEL, "F": AxisRole.PARALLEL})
    assert role_of(block, "P") == AxisRole.PARALLEL
    assert role_of(block, "F") == AxisRole.PARALLEL


def test_role_of_returns_accumulation_when_declared() -> None:
    block = _make_block({"K": AxisRole.ACCUMULATION, "M": AxisRole.PARALLEL, "N": AxisRole.PARALLEL})
    assert role_of(block, "K") == AxisRole.ACCUMULATION
    assert role_of(block, "M") == AxisRole.PARALLEL


def test_role_of_unknown_axis_raises() -> None:
    block = _make_block({"P": AxisRole.PARALLEL})
    with pytest.raises(KeyError, match="does not declare axis"):
        role_of(block, "d99")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest test/ir/test_role_of.py test/ir/test_ir_extensions.py::test_isanode_payload_uses_operand_bindings -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py test/ir/test_role_of.py
git commit -m "Switch ISANode payload to (op_cls, operand_bindings, kwargs); adapt role_of tests"
```

### Task 10: Slim down `KernelIR` envelope

**Files:**
- Modify: `nkigym/src/nkigym/ir/ir.py`
- Test: `test/ir/test_ir_extensions.py`

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_kernelir_envelope_is_slim():
    """KernelIR drops dim_sizes and tensors fields."""
    import dataclasses
    from nkigym.ir.ir import KernelIR

    field_names = {f.name for f in dataclasses.fields(KernelIR)}
    assert field_names == {"func_name", "param_names", "return_name", "tree", "dependency"}


def test_kernelir_helper_methods_exposed():
    """KernelIR exposes all_buffers / buffer / axis_extent."""
    from nkigym.ir.ir import KernelIR

    assert callable(KernelIR.all_buffers)
    assert callable(KernelIR.buffer)
    assert callable(KernelIR.axis_extent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py -v -k "kernelir_envelope or helper_methods"`
Expected: FAIL — `dim_sizes` and `tensors` still present; helpers missing.

- [ ] **Step 3: Rewrite `KernelIR`**

Replace the contents of `nkigym/src/nkigym/ir/ir.py` with:

```python
"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` is the single envelope. It carries the kernel
signature, return-tensor identity, schedule tree, and producer-consumer
dependency graph. Per-buffer and per-axis information is derived from
the tree on demand via :meth:`KernelIR.all_buffers` and
:meth:`KernelIR.axis_extent` — caching them on the envelope leads to
invalidation churn when transforms move buffers between blocks.

:func:`build_initial_ir` runs dim unification, tree construction, and
dependency graph construction, then flattens the analysis output onto
a :class:`KernelIR` instance. :meth:`KernelIR.dump` writes the tree
and dependency diagrams side-by-side into a cache directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import analyze_dimensions
from nkigym.ir.tree import BlockNode, Buffer, KernelTree, build_initial_tree
from nkigym.ir.tree_visualize import dump_tree


@dataclass
class KernelIR:
    """Envelope holding signature and the schedule tree.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tree: KernelTree
    dependency: Dependency

    def all_buffers(self) -> dict[str, Buffer]:
        """Walk every :class:`BlockNode` in pre-order; return ``name -> Buffer``."""
        out: dict[str, Buffer] = {}
        for nid in self.tree.blocks():
            block = self.tree.data(nid)
            assert isinstance(block, BlockNode)
            for buf in block.alloc_buffers:
                if buf.name in out:
                    raise ValueError(f"buffer {buf.name!r} declared by two blocks")
                out[buf.name] = buf
        return out

    def buffer(self, name: str) -> Buffer:
        """Resolve a buffer by name; raises :class:`KeyError` if absent."""
        buffers = self.all_buffers()
        if name not in buffers:
            raise KeyError(f"buffer {name!r} not found in any block.alloc_buffers")
        return buffers[name]

    def axis_extent(self, axis: str) -> int:
        """Return the extent of the iter_var named ``axis``.

        Walks blocks in pre-order; returns the first ``IterVar`` whose
        ``axis`` matches. Raises :class:`KeyError` if the axis is not
        declared anywhere in the tree.
        """
        for nid in self.tree.blocks():
            block = self.tree.data(nid)
            assert isinstance(block, BlockNode)
            for iv in block.iter_vars:
                if iv.axis == axis:
                    return iv.dom[1] - iv.dom[0]
        raise KeyError(f"no iter_var with axis {axis!r}")

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``envelope.md``, ``tree.*``, ``dependency.*``, and a black-formatted ``kernel.py`` into ``cache_dir``."""
        from nkigym.codegen import render

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "envelope.md").write_text(self._render_envelope_md(), encoding="utf-8")
        dump_tree(self.tree, cache_dir)
        self.dependency.dump(cache_dir)
        kernel_path = cache_path / "kernel.py"
        kernel_path.write_text(render(self), encoding="utf-8")
        subprocess.run(["black", "--quiet", str(kernel_path)], check=True)

    def _render_envelope_md(self) -> str:
        """Render signature + buffers as Markdown."""
        lines: list[str] = [
            f"# `{self.func_name}`",
            "",
            "## Signature",
            "",
            f"- **Params**: {', '.join(f'`{p}`' for p in self.param_names) or '_(none)_'}",
            f"- **Returns**: `{self.return_name}`",
            "",
            "## Buffers",
            "",
            "| Name | Location | Dtype | Shape |",
            "| ---- | -------- | ----- | ----- |",
        ]
        for buf in self.all_buffers().values():
            shape = "(" + ", ".join(str(s) for s in buf.shape) + ")"
            lines.append(f"| `{buf.name}` | `{buf.location}` | `{buf.dtype}` | `{shape}` |")
        lines.append("")
        return "\n".join(lines)


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
        tree=tree,
        dependency=dependency,
    )


__all__ = ["KernelIR", "build_initial_ir"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py -v -k "kernelir_envelope or helper_methods"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/ir.py test/ir/test_ir_extensions.py
git commit -m "Slim KernelIR envelope; add all_buffers/buffer/axis_extent helpers"
```

### Task 11: Rewrite `dimension_analysis.py` to emit `BlockNode`-rooted tree

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py`
- Modify: `nkigym/src/nkigym/ir/tree.py` (replace `build_initial_tree`)
- Test: `test/ir/test_ir_extensions.py`

This task is the heart of the canonical-builder cutover. The new builder:

1. Collects `NKIAlloc` records → `Buffer` objects.
2. Builds one leaf `BlockNode` per non-alloc op with iter_vars from `OPERAND_AXES` × `axis_map`, iter_values as `Var(loop_var)`, and reads/writes from operand slots + `OPERAND_AXES`.
3. Detects memset+rmw pairs and folds the memset under matmul's `init`.
4. Computes the smallest enclosing block per buffer (canonical: nearly always root).
5. Wraps everything under a synthetic root `BlockNode`.

- [ ] **Step 1: Write the failing test (canonical matmul smoke)**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_canonical_matmul_emits_root_block_with_alloc_buffers():
    """The canonical 2048**3 matmul tree's root child is a single BlockNode whose alloc_buffers
    list every kernel-lifetime tensor, including psum_prod (canonical scope = root)."""
    from nkigym.ir import build_initial_ir
    from nkigym.ir.tree import BlockNode
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_copy import NKITensorCopy

    K = M = N = 2048

    @nkigym_kernel
    def f_matmul(lhs_T, rhs):
        sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
        sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
        psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
        NKILoad()(src=rhs, dst=sbuf_rhs)
        NKIMemset(value=0.0)(dst=psum_prod)
        NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
        NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    ir = build_initial_ir(f_matmul, {"lhs_T": (K, M), "rhs": (K, N)})
    root_kids = ir.tree.children(ir.tree.root)
    assert len(root_kids) == 1, "tree.root must have exactly one child (the synthetic root block)"
    root_block = ir.tree.data(root_kids[0])
    assert isinstance(root_block, BlockNode)
    """Buffer scope: every kernel-lifetime tensor is in root's alloc_buffers."""
    buffer_names = {buf.name for buf in root_block.alloc_buffers}
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"} <= buffer_names


def test_canonical_matmul_folds_memset_into_matmul_init():
    """The matmul leaf-block's init is the memset sub-block; no sibling memset block remains."""
    from nkigym.ir import build_initial_ir
    from nkigym.ir.tree import BlockNode
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_copy import NKITensorCopy

    K = M = N = 2048

    @nkigym_kernel
    def f_matmul(lhs_T, rhs):
        sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
        sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
        psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
        NKILoad()(src=rhs, dst=sbuf_rhs)
        NKIMemset(value=0.0)(dst=psum_prod)
        NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
        NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    ir = build_initial_ir(f_matmul, {"lhs_T": (K, M), "rhs": (K, N)})
    """Locate the matmul leaf-block — the one whose body emits NKIMatmul."""
    matmul_block_nid = None
    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        for kid in ir.tree.descendants(nid):
            kid_data = ir.tree.data(kid)
            from nkigym.ir.tree import ISANode
            from nkigym.ops.matmul import NKIMatmul as MM

            if isinstance(kid_data, ISANode) and kid_data.op_cls is MM:
                matmul_block_nid = nid
                break
        if matmul_block_nid is not None:
            break
    assert matmul_block_nid is not None, "matmul block not found"
    matmul_block = ir.tree.data(matmul_block_nid)
    assert matmul_block.init is not None, "matmul block must carry an init sub-block"
    init_block = ir.tree.data(matmul_block.init)
    assert isinstance(init_block, BlockNode)
    """The init body emits NKIMemset."""
    from nkigym.ir.tree import ISANode
    from nkigym.ops.memset import NKIMemset as MS

    init_isa = next(
        ir.tree.data(d) for d in ir.tree.descendants(matmul_block.init) if isinstance(ir.tree.data(d), ISANode)
    )
    assert init_isa.op_cls is MS
    """And there is no sibling memset block under root."""
    root_kid = ir.tree.children(ir.tree.root)[0]
    for kid in ir.tree.children(root_kid):
        kid_data = ir.tree.data(kid)
        if isinstance(kid_data, BlockNode):
            for d in ir.tree.descendants(kid):
                d_data = ir.tree.data(d)
                if isinstance(d_data, ISANode) and d_data.op_cls is MS:
                    raise AssertionError(f"unexpected sibling memset block at nid {kid}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_ir_extensions.py -v -k "canonical_matmul"`
Expected: FAIL — current builder still emits the old IR.

- [ ] **Step 3: Replace `build_initial_tree` and helpers**

Replace the existing `build_initial_tree` function and the `_attach_op_subtree` helper in `nkigym/src/nkigym/ir/tree.py` with the new versions. Open `nkigym/src/nkigym/ir/tree.py` and find `def build_initial_tree(`. Replace from that line through the end of `_attach_op_subtree` with:

```python
def build_initial_tree(analysis: "_AnalysisResult") -> "KernelTree":
    """Build the canonical schedule tree from an :class:`_AnalysisResult`.

    The returned tree's root is a synthetic :class:`RootNode` with a
    single ``BlockNode("root")`` child. Per-op leaf blocks are children
    of that root block, in source order. Reduction memsets are folded
    into their matmul's ``init`` field. Allocs become ``Buffer`` entries
    on the smallest enclosing block whose subtree contains every leaf
    that touches the buffer (canonical: nearly always the root block).
    """
    from nkigym.ir.canonical_build import build_canonical_blocknode_tree

    return build_canonical_blocknode_tree(analysis)


__all__ = [
    "BlockNode",
    "Buffer",
    "BufferRegion",
    "ForNode",
    "ISANode",
    "IterVar",
    "KernelTree",
    "NodeData",
    "RootNode",
    "build_initial_tree",
    "role_of",
]
```

(The `_attach_op_subtree` helper is removed entirely — superseded by the new builder.)

Note that the existing `__all__` list at the end of `tree.py` already contains these entries from Task 7; ensure no duplicate `__all__` block is introduced.

- [ ] **Step 4: Add the new canonical builder module**

Create `nkigym/src/nkigym/ir/canonical_build.py`:

```python
"""Canonical :class:`BlockNode`-rooted tree construction.

Consumes the private :class:`_AnalysisResult` produced by
:func:`nkigym.ir.dimension_analysis.analyze_dimensions` and emits a
fully-shaped :class:`KernelTree` whose root has exactly one child — a
synthetic ``BlockNode("root")`` that owns every kernel-lifetime
``Buffer`` and contains one leaf ``BlockNode`` per non-alloc op,
in source order.

Pattern detection: a ``NKIMemset(dst=X)`` immediately preceding an op
with an ``RMW`` slot writing the same buffer ``X`` is folded into the
following block's ``init`` field. The memset is removed from the
top-level sequence.

Buffer scope: each :class:`Buffer` lives on the smallest block whose
subtree transitively touches the buffer. For canonical IR (every op
is a sibling of every other under the root block), this resolves to
the root block in nearly every case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nkigym.ir.expr import Const, Var
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree, RootNode
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.memset import NKIMemset

if TYPE_CHECKING:
    from nkigym.ir.dimension_analysis import _AnalysisResult, _OpRecord, TensorDims


def build_canonical_blocknode_tree(analysis: "_AnalysisResult") -> KernelTree:
    """Build the canonical :class:`BlockNode`-rooted tree."""
    tree = KernelTree()
    op_records = list(analysis.ops)
    buffers_by_name = _collect_buffers(op_records, analysis.tensors)
    compute_records = [rec for rec in op_records if rec.op_cls is not NKIAlloc]
    """Pair memset+rmw: each rmw op may absorb the immediately-preceding memset whose dst matches its rmw slot tensor."""
    paired = _pair_memset_with_rmw(compute_records)
    """Construct the synthetic root block; alloc_buffers attached after we know which buffers are touched."""
    root_block_nid = tree.add_node(
        BlockNode(
            iter_vars=(),
            iter_values=(),
            reads=(),
            writes=(),
            alloc_buffers=tuple(buffers_by_name.values()),
            init=None,
        ),
        parent=tree.root,
    )
    """Walk paired records in source order; emit a leaf block per primary record."""
    for primary, init_rec in paired:
        _attach_leaf_block(tree, root_block_nid, primary, init_rec, analysis)
    return tree


def _collect_buffers(records: list["_OpRecord"], tensors: dict[str, "TensorDims"]) -> dict[str, Buffer]:
    """Return one :class:`Buffer` per tensor in ``tensors`` (every named tensor — params + alloc-declared)."""
    out: dict[str, Buffer] = {}
    for name, td in tensors.items():
        out[name] = Buffer(name=name, shape=tuple(td.shape), dtype=td.dtype, location=td.location)
    return out


def _pair_memset_with_rmw(records: list["_OpRecord"]) -> list[tuple["_OpRecord", "_OpRecord | None"]]:
    """Pair each rmw-bearing op with the immediately-preceding memset that writes the same tensor.

    Returns a list of ``(primary, init_or_None)`` tuples in source order. The memset records absorbed
    into a primary block do NOT appear as their own entries.
    """
    out: list[tuple["_OpRecord", "_OpRecord | None"]] = []
    pending_memset: "_OpRecord | None" = None
    for rec in records:
        if rec.op_cls is NKIMemset:
            pending_memset = rec
            continue
        init_rec: "_OpRecord | None" = None
        rmw_slots = rec.op_cls.RMW_OPERANDS
        if pending_memset is not None and rmw_slots:
            memset_dst = pending_memset.operand_names.get("dst")
            rmw_targets = {rec.operand_names[s] for s in rmw_slots if s in rec.operand_names}
            if memset_dst in rmw_targets:
                init_rec = pending_memset
                pending_memset = None
        if pending_memset is not None and init_rec is None:
            """Pending memset did not match this rec's rmw slot; flush it as its own entry."""
            out.append((pending_memset, None))
            pending_memset = None
        out.append((rec, init_rec))
    if pending_memset is not None:
        out.append((pending_memset, None))
    return out


def _attach_leaf_block(
    tree: KernelTree,
    root_block_nid: int,
    primary: "_OpRecord",
    init_rec: "_OpRecord | None",
    analysis: "_AnalysisResult",
) -> None:
    """Attach one leaf :class:`BlockNode` for ``primary`` (with optional ``init_rec``) under root."""
    init_nid: int | None = None
    if init_rec is not None:
        init_nid = _build_subblock(tree, root_block_nid, init_rec, analysis, attach=False)
    block_nid = _build_subblock(tree, root_block_nid, primary, analysis, attach=True, init_nid=init_nid)
    if init_rec is not None:
        assert init_nid is not None
        """Re-parent the init sub-block under the primary block so dependency / visualisation
        understand it is a child."""
        tree.graph.add_edge(block_nid, init_nid)


def _build_subblock(
    tree: KernelTree,
    parent_nid: int,
    rec: "_OpRecord",
    analysis: "_AnalysisResult",
    attach: bool,
    init_nid: int | None = None,
) -> int:
    """Construct one :class:`BlockNode` + its loop chain + ISA leaf; return the block's nid.

    If ``attach`` is True, the block is parented under ``parent_nid``; otherwise it is added with no
    parent (used for init sub-blocks that the caller re-parents).
    """
    iter_vars: list[IterVar] = []
    iter_values: list = []
    loop_var_names: dict[str, str] = {}
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        role = rec.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
        iter_vars.append(IterVar(axis=concrete, dom=(0, extent), role=role))
        loop_var = f"i_{concrete}_0"
        loop_var_names[abstract] = loop_var
        iter_values.append(Var(name=loop_var))
    reads, writes = _operand_regions(rec, loop_var_names)
    block = BlockNode(
        iter_vars=tuple(iter_vars),
        iter_values=tuple(iter_values),
        reads=tuple(reads),
        writes=tuple(writes),
        alloc_buffers=(),
        init=init_nid,
    )
    block_nid = tree.add_node(block, parent=parent_nid if attach else None)
    parent_for_loops: int = block_nid
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        trip = extent // tile
        loop_var = loop_var_names[abstract]
        for_nid = tree.add_node(ForNode(loop_var=loop_var, extent=trip), parent=parent_for_loops)
        parent_for_loops = for_nid
    operand_bindings = _operand_bindings(rec, loop_var_names)
    op_kwargs = {k: v for k, v in rec.kwargs.items() if k not in rec.op_cls.OPERAND_AXES}
    tree.add_node(
        ISANode(op_cls=rec.op_cls, operand_bindings=operand_bindings, kwargs=op_kwargs),
        parent=parent_for_loops,
    )
    return block_nid


def _operand_regions(
    rec: "_OpRecord", loop_var_names: dict[str, str]
) -> tuple[list[BufferRegion], list[BufferRegion]]:
    """Build (reads, writes) BufferRegion lists from ``rec.operand_names`` and OPERAND_AXES."""
    reads: list[BufferRegion] = []
    writes: list[BufferRegion] = []
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        region = _build_region(rec, slot, axes, loop_var_names)
        if slot in rec.op_cls.INPUT_OPERANDS:
            reads.append(region)
        elif slot in rec.op_cls.RMW_OPERANDS:
            reads.append(region)
            writes.append(region)
        else:
            writes.append(region)
    return reads, writes


def _operand_bindings(rec: "_OpRecord", loop_var_names: dict[str, str]) -> dict[str, BufferRegion]:
    """Build the per-slot :class:`BufferRegion` map for the ISA leaf."""
    out: dict[str, BufferRegion] = {}
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        out[slot] = _build_region(rec, slot, axes, loop_var_names)
    return out


def _build_region(
    rec: "_OpRecord", slot: str, axes: tuple[str, ...], loop_var_names: dict[str, str]
) -> BufferRegion:
    """Construct a :class:`BufferRegion` for ``slot`` using its axes and per-tile widths."""
    tensor_name = rec.operand_names[slot]
    ranges: list = []
    for abstract in axes:
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        concrete = rec.axis_map[abstract]
        extent_per_tile = max_tile if max_tile is not None else _resolve_extent(concrete, rec, loop_var_names)
        loop_var = loop_var_names.get(abstract)
        if loop_var is None:
            ranges.append((Const(value=0), Const(value=extent_per_tile)))
        else:
            ranges.append((Var(name=loop_var), Const(value=extent_per_tile)))
    return BufferRegion(tensor=tensor_name, ranges=tuple(ranges))


def _resolve_extent(concrete: str, rec: "_OpRecord", loop_var_names: dict[str, str]) -> int:
    """Look up the extent of ``concrete`` from the analysis (we need it for unbounded MAX_TILE axes).

    Helper duplicates :class:`_AnalysisResult.dim_sizes` access; kept private to this builder.
    """
    raise NotImplementedError("call site fills extent inline; placeholder for explicit unbounded-axis support")
```

- [ ] **Step 5: Reflow extent lookup**

Replace the `_resolve_extent` placeholder by passing `dim_sizes` explicitly. Edit `nkigym/src/nkigym/ir/canonical_build.py` to thread `dim_sizes` through `_operand_regions` / `_operand_bindings` / `_build_region`:

Replace `_build_region` and its callers with:

```python
def _operand_regions(
    rec: "_OpRecord", loop_var_names: dict[str, str], dim_sizes: dict[str, int]
) -> tuple[list[BufferRegion], list[BufferRegion]]:
    """Build (reads, writes) BufferRegion lists from ``rec.operand_names`` and OPERAND_AXES."""
    reads: list[BufferRegion] = []
    writes: list[BufferRegion] = []
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        region = _build_region(rec, slot, axes, loop_var_names, dim_sizes)
        if slot in rec.op_cls.INPUT_OPERANDS:
            reads.append(region)
        elif slot in rec.op_cls.RMW_OPERANDS:
            reads.append(region)
            writes.append(region)
        else:
            writes.append(region)
    return reads, writes


def _operand_bindings(
    rec: "_OpRecord", loop_var_names: dict[str, str], dim_sizes: dict[str, int]
) -> dict[str, BufferRegion]:
    """Build the per-slot :class:`BufferRegion` map for the ISA leaf."""
    out: dict[str, BufferRegion] = {}
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        out[slot] = _build_region(rec, slot, axes, loop_var_names, dim_sizes)
    return out


def _build_region(
    rec: "_OpRecord",
    slot: str,
    axes: tuple[str, ...],
    loop_var_names: dict[str, str],
    dim_sizes: dict[str, int],
) -> BufferRegion:
    """Construct a :class:`BufferRegion` for ``slot`` using its axes and per-tile widths."""
    tensor_name = rec.operand_names[slot]
    ranges: list = []
    for abstract in axes:
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        if max_tile is None:
            extent_per_tile = dim_sizes[rec.axis_map[abstract]]
        else:
            extent_per_tile = max_tile
        loop_var = loop_var_names.get(abstract)
        if loop_var is None:
            ranges.append((Const(value=0), Const(value=extent_per_tile)))
        else:
            ranges.append((Var(name=loop_var), Const(value=extent_per_tile)))
    return BufferRegion(tensor=tensor_name, ranges=tuple(ranges))
```

And in `_build_subblock`, replace the calls to `_operand_regions` / `_operand_bindings` with the three-argument versions:

```python
    reads, writes = _operand_regions(rec, loop_var_names, analysis.dim_sizes)
    ...
    operand_bindings = _operand_bindings(rec, loop_var_names, analysis.dim_sizes)
```

Remove the `_resolve_extent` placeholder function entirely.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py -v -k "canonical_matmul"`
Expected: PASS for both `test_canonical_matmul_emits_root_block_with_alloc_buffers` and `test_canonical_matmul_folds_memset_into_matmul_init`.

- [ ] **Step 7: Verify renderer / transform tests still red (expected during cutover)**

Run: `pytest test/codegen/ test/transforms/ -x --co -q 2>&1 | head -20`
Expected: Collection errors / import errors in renderer and transforms — they reference `node.dim`, `node.trip`, `node.reads`, etc. These are knowingly red until Phases 4–8.

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/ir/canonical_build.py nkigym/src/nkigym/ir/tree.py test/ir/test_ir_extensions.py
git commit -m "Rewrite canonical builder to emit BlockNode-rooted tree with init folding and buffer scope"
```

---

## Phase 4 — Renderer migration

The renderer must walk the new BlockNode-rooted tree and emit an NKI source that is structurally identical to today's output for the canonical 2048³ matmul. We approach this in four sub-tasks: header (alloc emission from `alloc_buffers`), body emission for non-init blocks, init emission, and the bound-region → slice formatter.

### Task 12: Header emits `nl.ndarray(...)` per `Buffer` in root's `alloc_buffers`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/header.py`
- Test: `test/codegen/test_header.py`

- [ ] **Step 1: Write the failing test**

Replace the contents of `test/codegen/test_header.py` with:

```python
"""Tests for nkigym.codegen.header (BlockNode-aware)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen.header import emit_header, emit_return


def test_header_contains_imports_and_signature():
    """The header has the standard imports + the @nki.jit decorator + def."""
    ir = build_canonical_ir()
    out = emit_header(ir)
    assert "import nki" in out
    assert "import nki.isa as nisa" in out
    assert "import nki.language as nl" in out
    assert "@nki.jit" in out
    assert f"def nki_{ir.func_name}" in out


def test_header_emits_param_shape_assertions():
    """Each kernel parameter gets an assert <param>.shape == (...) line."""
    ir = build_canonical_ir()
    out = emit_header(ir)
    assert "assert lhs_T.shape == (2048, 2048)" in out
    assert "assert rhs.shape == (2048, 2048)" in out


def test_return_emits_return_statement():
    """The return emitter renders a single `return <name>` line."""
    ir = build_canonical_ir()
    out = emit_return(ir)
    assert out.strip() == "return hbm_out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_header.py -v`
Expected: FAIL — current `_emit_shape_assertions` accesses `ir.tensors`, which no longer exists.

- [ ] **Step 3: Update `header.py`**

Replace the contents of `nkigym/src/nkigym/codegen/header.py` with:

```python
"""Kernel-prologue and -epilogue codegen.

:func:`emit_header` produces the fixed scaffolding above the kernel
body — imports, the ``@nki.jit`` decorator, the ``def`` line, and one
``assert <param>.shape == (...)`` line per kernel parameter.

:func:`emit_return` produces the trailing ``return <return_name>`` line.
``nl.ndarray(...)`` allocations now come from the body emitter, which
walks each block's ``alloc_buffers`` field.
"""

from __future__ import annotations

from nkigym.ir import KernelIR


def emit_header(ir: KernelIR) -> str:
    """Render imports + ``@nki.jit`` signature + per-param shape assertions."""
    lines: list[str] = []
    _emit_imports(lines)
    lines.append("")
    lines.append("")
    _emit_signature(lines, ir)
    _emit_shape_assertions(lines, ir)
    return "\n".join(lines) + "\n"


def emit_return(ir: KernelIR) -> str:
    """Render the trailing ``return <return_name>`` statement."""
    return f"    return {ir.return_name}\n"


def _emit_imports(lines: list[str]) -> None:
    """Append the standard NKI import block."""
    lines.append("import nki")
    lines.append("import nki.isa as nisa")
    lines.append("import nki.language as nl")


def _emit_signature(lines: list[str], ir: KernelIR) -> None:
    """Append ``@nki.jit`` and ``def <func_name>(<params>):`` in signature order."""
    lines.append("@nki.jit")
    params = ", ".join(ir.param_names)
    lines.append(f"def nki_{ir.func_name}({params}):")


def _emit_shape_assertions(lines: list[str], ir: KernelIR) -> None:
    """Append ``assert <param>.shape == (...)`` for every kernel parameter."""
    for name in ir.param_names:
        buf = ir.buffer(name)
        shape_tuple = "(" + ", ".join(str(s) for s in buf.shape) + ")"
        lines.append(f"    assert {name}.shape == {shape_tuple}")


__all__ = ["emit_header", "emit_return"]
```

- [ ] **Step 4: Update the canonical fixture**

`test/transforms/_fixtures.py` currently builds against the old IR. Replace it with a fixture that uses the new builder (already swapped in Task 11):

```python
"""Shared canonical-IR fixture for transform tests."""

from __future__ import annotations

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_prod)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
    NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def build_canonical_ir() -> KernelIR:
    """Build the canonical KernelIR for the matmul fixture."""
    return build_initial_ir(f_matmul, INPUT_SPECS)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest test/codegen/test_header.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/header.py test/codegen/test_header.py test/transforms/_fixtures.py
git commit -m "Update header emitter for slim KernelIR; revive canonical fixture"
```

### Task 13: Implement BufferRegion → Python slice rendering

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py`
- Test: `test/codegen/test_body.py`

The renderer needs a function that, given a `BufferRegion` and a `Buffer`, returns a Python slice expression like `psum_prod[i_M_0 * 128:i_M_0 * 128 + 128, i_N_0 * 512:i_N_0 * 512 + 512]`. SBUF / PSUM tensors render with the 3D `(128, P//128, F)` layout; HBM tensors render flat 2D.

- [ ] **Step 1: Write the failing test**

Replace the contents of `test/codegen/test_body.py` with:

```python
"""Tests for nkigym.codegen.body BufferRegion rendering."""

from __future__ import annotations

import pytest

from nkigym.codegen.body import render_buffer_region
from nkigym.ir.expr import Add, Const, Mul, Var
from nkigym.ir.tree import Buffer, BufferRegion


def test_render_hbm_2d_region():
    """An HBM 2D region renders as flat ``[lo:hi, lo:hi]``."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    region = BufferRegion(
        tensor="hbm_out",
        ranges=(
            (Mul(left=Var(name="i_d0_0"), right=Const(value=128)), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "hbm_out[i_d0_0 * 128:i_d0_0 * 128 + 128, i_d1_0 * 512:i_d1_0 * 512 + 512]"


def test_render_sbuf_3d_region_partition_axis_split():
    """An SBUF 3D region splits the partition axis: [0:128, P_coord, F_lo:F_hi]."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"


def test_render_psum_3d_region_partition_axis_split():
    """A PSUM region (also 3D) splits the partition axis the same way."""
    buf = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[0:128, i_d0_0, i_d1_0 * 512:i_d1_0 * 512 + 512]"


def test_render_constant_zero_origin_for_full_extent_axis():
    """When the lo expression is a bare zero Const, the rendered slice starts at 0 explicitly."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    region = BufferRegion(
        tensor="hbm_out",
        ranges=(
            (Const(value=0), Const(value=2048)),
            (Const(value=0), Const(value=2048)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "hbm_out[0:0 + 2048, 0:0 + 2048]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_body.py -v`
Expected: FAIL — `render_buffer_region` doesn't exist yet.

- [ ] **Step 3: Implement `render_buffer_region`**

Replace the contents of `nkigym/src/nkigym/codegen/body.py` with the new emitter. The legacy emitter referenced today's payloads; we replace it wholesale. Existing tests for the old body will be reworked in Task 14.

```python
"""BlockNode-driven body emitter.

Walks the canonical / transformed schedule tree and renders each
:class:`BlockNode` as a Python source fragment. Each block emits, in
order:

1. ``nl.ndarray(...)`` declarations — one per :attr:`BlockNode.alloc_buffers`.
2. The init sub-block, if :attr:`BlockNode.init` is set: a fresh
   sub-emission with the init's iter_vars, loops, and ISA leaf.
3. The block's main body — its ``ForNode`` chain ending in one
   :class:`ISANode`.

Operand slices are rendered via :func:`render_buffer_region` from the
ISA leaf's :attr:`ISANode.operand_bindings`.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.expr import Const, format_expr
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode

_INDENT = "    "
_PARTITION_DIM = 128


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree."""
    root_kids = ir.tree.children(ir.tree.root)
    if not root_kids:
        raise ValueError("emit_body: tree root has no children")
    if len(root_kids) != 1:
        raise ValueError(f"emit_body: tree.root must have exactly one BlockNode child; got {len(root_kids)}")
    code: list[str] = []
    _emit_block(ir, root_kids[0], depth=1, code=code)
    return "\n".join(code) + "\n"


def _emit_block(ir: KernelIR, block_nid: int, depth: int, code: list[str]) -> None:
    """Emit one BlockNode: alloc_buffers, optional init, then body."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for buf in block.alloc_buffers:
        code.append(indent + _emit_alloc(buf))
    if block.init is not None:
        _emit_block(ir, block.init, depth, code)
    for child_nid in ir.tree.children(block_nid):
        if child_nid == block.init:
            continue
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code)
        else:
            _emit_subtree(ir, child_nid, depth, code)


def _emit_subtree(ir: KernelIR, nid: int, depth: int, code: list[str]) -> None:
    """Emit a ForNode or ISANode subtree."""
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        code.append(indent + f"for {node.loop_var} in range({node.extent}):")
        for child_nid in ir.tree.children(nid):
            _emit_subtree(ir, child_nid, depth + 1, code)
    elif isinstance(node, ISANode):
        code.append(indent + _emit_isa_call(node, ir))
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")


def _emit_alloc(buf: Buffer) -> str:
    """Emit a single ``nl.ndarray(...)`` line for ``buf``."""
    if buf.location == "shared_hbm":
        shape = "(" + ", ".join(str(s) for s in buf.shape) + ")"
    else:
        if len(buf.shape) != 2:
            raise AssertionError(f"{buf.name}: SBUF/PSUM allocation expects a 2D shape; got {buf.shape}")
        P, F = buf.shape
        if P % _PARTITION_DIM != 0:
            raise AssertionError(f"{buf.name}: P={P} must be a multiple of {_PARTITION_DIM}")
        shape = f"({_PARTITION_DIM}, {P // _PARTITION_DIM}, {F})"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.dtype}, buffer=nl.{buf.location})"


def _emit_isa_call(node: ISANode, ir: KernelIR) -> str:
    """Emit ``nisa.<NAME>(slot=<region>, ..., kwarg=value, ...)`` for one ISA leaf."""
    op_cls = node.op_cls
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if slot in node.operand_bindings:
            region = node.operand_bindings[slot]
            buf = ir.buffer(region.tensor)
            parts.append(f"{slot}={render_buffer_region(region, buf)}")
    for k, v in node.kwargs.items():
        parts.append(f"{k}={v!r}")
    return f"nisa.{op_cls.NAME}({', '.join(parts)})"


def render_buffer_region(region: BufferRegion, buf: Buffer) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor."""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != _PARTITION_DIM:
                raise AssertionError(
                    f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}"
                )
            parts.append(f"0:{_PARTITION_DIM}")
            parts.append(format_expr(lo))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}[{', '.join(parts)}]"


__all__ = ["emit_body", "render_buffer_region"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/codegen/test_body.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "Implement BlockNode-driven body emitter with BufferRegion rendering"
```

### Task 14: End-to-end render gate — canonical matmul produces correct NKI source

**Files:**
- Modify: `test/codegen/test_render.py`
- (Possibly modify) `nkigym/src/nkigym/codegen/body.py` per any test failures

- [ ] **Step 1: Write the failing test (full-pipeline render + numerics)**

Replace the contents of `test/codegen/test_render.py` with:

```python
"""End-to-end render + CPU-sim numerics gate for the BlockNode IR refactor."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np

from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, f_matmul

from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32


def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("dumped_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_canonical_matmul_emits_expected_structure():
    """The rendered canonical kernel has the expected top-level shape."""
    ir = build_canonical_ir()
    src = render(ir)
    assert "@nki.jit" in src
    assert "def nki_f_matmul(lhs_T, rhs):" in src
    assert "psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)" in src
    assert "nisa.memset" in src
    assert "nisa.nc_matmul" in src
    assert "return hbm_out" in src.strip().splitlines()[-1]


def test_render_canonical_matmul_passes_numerics():
    """The rendered canonical kernel passes fp32 simulation against numpy."""
    ir = build_canonical_ir()
    src = render(ir)
    cache_dir = Path("/tmp/blocknode_render_test_canonical")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    kernel_path = cache_dir / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    module = _load_module_from_path(str(kernel_path))
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_render.py -v`
Expected: FAIL — most likely the structural test passes but numerics tile expressions need to be reconciled with the renderer's output.

- [ ] **Step 3: Iterate until both tests pass**

Inspect the rendered kernel at `/tmp/blocknode_render_test_canonical/kernel.py` after each test run; reconcile with the per-op `BufferRegion` shapes the canonical builder produces. Likely fixes:

* `_emit_isa_call` operand ordering — must follow the slot order in `op_cls.OPERAND_AXES`. Already implemented above; if a test fails because slots are ordered differently, ensure the iteration order matches the dict iteration order in `OPERAND_AXES`.

* `_build_region` in `canonical_build.py` — the `extent_per_tile` for sbuf/psum partition axis must be `_PARTITION_DIM` (128). If the renderer raises `AssertionError: SBUF/PSUM partition axis must use a partition-sized tile`, the canonical builder is emitting the wrong tile width for the partition axis. Patch `_build_region` to recognise the partition slot and clamp its tile to 128 when the buffer is SBUF/PSUM.

If the structural test passes and only numerics fail, run the test again with the `KEEP=1` env var (set up in `simulate_fp32`) and inspect the per-tile arithmetic in the rendered kernel.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py nkigym/src/nkigym/ir/canonical_build.py test/codegen/test_render.py
git commit -m "Render canonical matmul to working NKI source under BlockNode IR"
```

### Task 15: Update `tree_visualize.py` for BlockNodes

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree_visualize.py`

- [ ] **Step 1: Skim the existing visualiser and identify the per-node label sites**

Run: `grep -n "data\|ForNode\|ISANode\|RootNode" nkigym/src/nkigym/ir/tree_visualize.py`
Expected: A handful of `isinstance` switches that label nodes for Mermaid.

- [ ] **Step 2: Add a `BlockNode` label branch**

Find the per-node label function in `tree_visualize.py` and add a branch *before* the existing `ForNode` / `ISANode` branches:

```python
        if isinstance(data, BlockNode):
            iv_summary = ", ".join(f"{iv.axis}({iv.role.name[0]},{iv.dom[0]}..{iv.dom[1]})" for iv in data.iter_vars)
            alloc_count = len(data.alloc_buffers)
            init_flag = " init+" if data.init is not None else ""
            return f'block[{iv_summary}]{init_flag} allocs={alloc_count}'
```

Ensure `BlockNode` is imported in `tree_visualize.py`. Replace the existing `ForNode` / `ISANode` branches with their new payload-aware versions:

```python
        if isinstance(data, ForNode):
            return f"for {data.loop_var} in range({data.extent})"
        if isinstance(data, ISANode):
            return f"{data.op_cls.__name__} ({len(data.operand_bindings)} operands)"
```

- [ ] **Step 3: Add a smoke test that the visualiser doesn't error**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_dump_tree_runs_on_canonical_ir(tmp_path):
    """dump_tree on the canonical matmul IR produces tree.mmd and tree.png."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import dump_tree

    ir = build_canonical_ir()
    dump_tree(ir.tree, tmp_path)
    assert (tmp_path / "tree.mmd").exists()
    """The png is generated by mmdc; if mmdc is unavailable on this host the dump should still succeed
    but with a warning. We only check the mmd file unconditionally."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py::test_dump_tree_runs_on_canonical_ir -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree_visualize.py test/ir/test_ir_extensions.py
git commit -m "Update tree_visualize for BlockNode payloads"
```

---

## Phase 5 — Dependency graph migration

The dependency graph today keys on ISA-leaf nids and tracks per-tensor RAW / WAR / WAW between them. The new graph keys on BlockNode nids and uses each block's declared `reads` / `writes` to compute edges.

For canonical IR (every block under root), the per-region overlap question collapses to "does block A write a tensor that block B reads / writes?" — same conservative model as today, just one level up.

### Task 16: Block-keyed dependency graph

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `test/ir/test_dependency.py`:

```python
"""Tests for the block-keyed dependency graph."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import BlockNode, ISANode
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


def _block_for_op(ir, op_cls):
    """Return the leaf-block nid whose body emits ``op_cls``. The init memset block does NOT count
    as the matmul block."""
    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            d_data = ir.tree.data(d)
            if isinstance(d_data, ISANode) and d_data.op_cls is op_cls:
                """Skip if this ISA leaf is inside another block's init."""
                block_data = ir.tree.data(nid)
                assert isinstance(block_data, BlockNode)
                if block_data.init is None or d not in ir.tree.descendants(block_data.init):
                    return nid
    raise AssertionError(f"no leaf block for {op_cls.__name__}")


def test_dependency_orders_canonical_matmul_chain():
    """For canonical matmul: load_lhs / load_rhs precede matmul, which precedes tensor_copy, which precedes store."""
    ir = build_canonical_ir()
    matmul_nid = _block_for_op(ir, NKIMatmul)
    tc_nid = _block_for_op(ir, NKITensorCopy)
    store_nid = _block_for_op(ir, NKIStore)
    """The dependency graph contains an edge from matmul to tensor_copy."""
    assert ir.dependency.must_precede(matmul_nid, tc_nid)
    assert ir.dependency.must_precede(tc_nid, store_nid)
    assert ir.dependency.must_precede(matmul_nid, store_nid)


def test_dependency_does_not_order_independent_loads():
    """Loads of distinct tensors are independent; neither precedes the other."""
    ir = build_canonical_ir()
    load_nids = [
        nid
        for nid in ir.tree.blocks()
        for d in ir.tree.descendants(nid)
        if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKILoad
    ]
    assert len(load_nids) == 2
    a, b = load_nids
    assert not ir.dependency.must_precede(a, b)
    assert not ir.dependency.must_precede(b, a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/ir/test_dependency.py -v`
Expected: FAIL — current `Dependency` keys on ISA-leaf nids, not block nids.

- [ ] **Step 3: Rewrite `Dependency` to key on blocks**

Replace the contents of `nkigym/src/nkigym/ir/dependency.py` with:

```python
"""Producer-consumer dependency graph over :class:`BlockNode` leaves.

The :class:`Dependency` class scans a :class:`KernelTree` in pre-order
DFS and builds an ``nx.DiGraph`` whose nodes are leaf-block nids
(blocks whose ``init``-cleared subtree contains exactly one
``ISANode``). An edge ``p -> c`` means ``p`` must execute before ``c``.

Edges are inserted whenever block ``b`` reads / writes a tensor that
some earlier block wrote / read with overlapping :class:`BufferRegion`
ranges. For canonical IR (every block under root, no compute_at), the
overlap test reduces to "same tensor"; transforms can produce nested
blocks where the per-iteration overlap matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.tree import BlockNode, BufferRegion, KernelTree

_DEPENDENCY_STYLES: list[ClassStyle] = [
    ClassStyle(name="block", fill="#efe", stroke="#363"),
]

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}


@dataclass(frozen=True)
class _BlockInfo:
    """Cached read / write tensor sets for a single leaf block."""

    name: str
    reads: frozenset[str]
    writes: frozenset[str]


class Dependency:
    """Producer-consumer graph over leaf :class:`BlockNode` nids."""

    def __init__(self, tree: KernelTree) -> None:
        """Scan ``tree`` and build the block-keyed dependency graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.touches_by_tensor: dict[str, list[int]] = {}
        self.blocks: list[int] = []
        self._build(tree)
        self._closure: nx.DiGraph = nx.transitive_closure(self.graph, reflexive=False)

    def info(self, nid: int) -> _BlockInfo:
        """Return the cached :class:`_BlockInfo` for ``nid``."""
        return self.graph.nodes[nid]["info"]

    def direct_producers(self, nid: int) -> list[int]:
        return list(self.graph.predecessors(nid))

    def direct_consumers(self, nid: int) -> list[int]:
        return list(self.graph.successors(nid))

    def producers(self, nid: int) -> set[int]:
        return set(self._closure.predecessors(nid))

    def consumers(self, nid: int) -> set[int]:
        return set(self._closure.successors(nid))

    def must_precede(self, producer: int, consumer: int) -> bool:
        return self._closure.has_edge(producer, consumer)

    def chains(self) -> dict[str, list[int]]:
        return {name: list(chain) for name, chain in self.touches_by_tensor.items()}

    def dump(self, cache_dir: str | Path) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        mmd_path = cache_path / "dependency.mmd"
        png_path = cache_path / "dependency.png"
        mmd_path.write_text(_to_mermaid(self), encoding="utf-8")
        render_png(mmd_path, png_path)

    def _build(self, tree: KernelTree) -> None:
        """Populate the graph by walking leaf blocks (skipping the synthetic root)."""
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if not block.iter_vars and not block.reads and not block.writes:
                continue
            info = self._summarise(block)
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

    def _summarise(self, block: BlockNode) -> _BlockInfo:
        """Collapse a block's BufferRegions to ``(reads, writes)`` tensor-name sets."""
        reads = {r.tensor for r in block.reads}
        writes = {w.tensor for w in block.writes}
        return _BlockInfo(name=_block_name(block), reads=frozenset(reads), writes=frozenset(writes))

    def _record_hazards(
        self, nid: int, info: _BlockInfo, last_writer: dict[str, int], prior_readers: dict[str, list[int]]
    ) -> None:
        for name in info.reads:
            self._try_edge(last_writer.get(name), nid, "RAW")
        for name in info.writes:
            self._try_edge(last_writer.get(name), nid, "WAW")
            for prior_r in prior_readers.get(name, ()):
                self._try_edge(prior_r, nid, "WAR")

    def _try_edge(self, producer: int | None, consumer: int, kind: str) -> None:
        if producer is None or producer == consumer:
            return
        if self.graph.has_edge(producer, consumer):
            current = self.graph.edges[producer, consumer]["kind"]
            if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                return
        self.graph.add_edge(producer, consumer, kind=kind)


def _block_name(block: BlockNode) -> str:
    """Best-effort label for a block."""
    return block.annotations.get("name", "block")


def _to_mermaid(dep: Dependency) -> str:
    flow = Flowchart(direction="LR", styles=_DEPENDENCY_STYLES)
    for nid in dep.blocks:
        info = dep.info(nid)
        node_id = f"n{nid}"
        flow.add_node(node_id, f'{node_id}["{_label(nid, info)}"]', "block")
    for producer, consumer, attrs in dep.graph.edges(data=True):
        flow.add_edge(f"n{producer}", f"n{consumer}", label=attrs["kind"])
    return flow.render()


def _label(nid: int, info: _BlockInfo) -> str:
    parts: list[str] = [f"#{nid} {info.name}"]
    if info.reads:
        parts.append(f"reads={','.join(sorted(info.reads))}")
    if info.writes:
        parts.append(f"writes={','.join(sorted(info.writes))}")
    return "<br/>".join(parts)


def _bufferregion_overlaps(_a: BufferRegion, _b: BufferRegion) -> bool:
    """Stub for future per-region overlap analysis. Today's canonical IR doesn't need it; the
    block-pair tensor-name match is sufficient. Compute_at-driven nested blocks will exercise this."""
    return True


__all__ = ["Dependency"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/ir/test_dependency.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full IR-side test suite to confirm nothing else regressed**

Run: `pytest test/ir/ -v`
Expected: All passing.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Block-keyed dependency graph with tensor-name overlap"
```

### Task 17: Update `role_of` consumers across the codebase

**Files:**
- Search and update all consumers of the old `role_of(leaf, concrete_dim)`.

- [ ] **Step 1: Find every call site**

Run: `grep -rn "role_of(" nkigym/src/nkigym/ test/`

Expected hits will primarily be in `transforms/reorder.py` and possibly tests.

- [ ] **Step 2: Update each call to use `(block, axis)` form**

For each hit, change from `role_of(leaf, concrete_dim)` to `role_of(block, axis)` where `block` is the enclosing `BlockNode` payload and `axis` is the abstract axis name on the iter_var. Inside `Reorder._check_legality`, the descendant-leaf walk must be replaced by a descendant-block walk (the role lookup is now per-block). This is part of the larger Reorder migration in Task 20 — for now, just unbreak the test imports by leaving stubs that raise:

In `nkigym/src/nkigym/transforms/reorder.py` (current implementation), find the `role_of(leaf, swap_dim)` call and rewrite as:

```python
        for descendant_block_nid in ir.tree.blocks(option.inner_nid):
            block = ir.tree.data(descendant_block_nid)
            for swap_dim in (outer_loop_var, inner_loop_var):
                """Map loop_var -> iter_var axis via the enclosing block; if the block does not bind
                this loop_var (e.g. swap is between two same-axis Splits), skip."""
                axis = _axis_for_loop_var(block, swap_dim)
                if axis is None:
                    continue
                if role_of(block, axis) == AxisRole.SEQUENTIAL:
                    raise TransformLegalityError(
                        f"Reorder rejected: descendant block has SEQUENTIAL role on loop_var {swap_dim!r}"
                    )
```

Where `_axis_for_loop_var(block, loop_var)` is a small helper added to `reorder.py`:

```python
def _axis_for_loop_var(block: BlockNode, loop_var: str) -> str | None:
    """Return the iter_var axis bound by ``loop_var`` in ``block.iter_values``, or None."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        from nkigym.ir.expr import Var

        if isinstance(value, Var) and value.name == loop_var:
            return iv.axis
    return None
```

This is preliminary; the full Reorder migration in Task 20 reworks the whole file. Here we only restore importability.

- [ ] **Step 3: Rerun the dependency tests**

Run: `pytest test/ir/test_dependency.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py
git commit -m "Adapt role_of consumers to block + axis form (interim)"
```

---

## Phase 6 — Split migration

`Split` rewrites `BlockNode.iter_values` and `ForNode.extent`s. Two flavours:

* **Outer-trip Split**: the target is a `ForNode` whose `loop_var` binds (part of) one iter_var of an enclosing block. Replace the ForNode with a chain of new ForNodes whose extent product equals the original; rewrite the enclosing block's `iter_values` entry for the bound iter_var so the binding is `i_0 * (n1*n2*...) + i_1 * (n2*...) + ...`.
* **Tensorize Split**: the target is an `ISANode` (the body of a leaf block). Insert one or more ForNodes between the block and the leaf and shrink the per-axis tile width on the leaf's `operand_bindings`. Rewrite the enclosing block's `iter_values` entry similarly.

### Task 18: Migrate `Split` to BlockNode IR

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py`
- Modify: `nkigym/src/nkigym/transforms/_tree_ops.py`
- Modify: `test/transforms/test_split.py`

- [ ] **Step 1: Replace `_tree_ops.py` helpers to operate on the new tree shape**

The existing `_replace_in_parent_children` is fine as-is — it operates on tree topology, payload-agnostic. No change needed here. (Verify by reading the file.)

- [ ] **Step 2: Write the failing tests for outer-trip Split**

Replace the contents of `test/transforms/test_split.py` with the new fixture-based tests:

```python
"""Tests for nkigym.transforms.Split under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.expr import to_affine
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import Split, SplitOption, TransformLegalityError


def _matmul_block(ir):
    """Return (block_nid, block) for the matmul leaf-block."""
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        for d in ir.tree.descendants(nid):
            d_data = ir.tree.data(d)
            if isinstance(d_data, ISANode) and d_data.op_cls is NKIMatmul:
                return nid, block
    raise AssertionError("matmul block not found")


def _first_for_under(ir, block_nid):
    """Return the first ForNode descended from block_nid."""
    for nid in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(nid), ForNode):
            return nid
    raise AssertionError("no ForNode under block")


def test_split_outer_trip_replaces_for_with_chain():
    """Splitting a ForNode trip 16 by factors=(4, 4) gives a 4 -> 4 chain."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))

    """Old IR untouched."""
    assert ir.tree.data(target).extent == target_extent

    """New IR: parent's child slot now contains a fresh ForNode of extent 4 with one ForNode child."""
    parent = ir.tree.parent(target)
    new_kid = new_ir.tree.children(parent)[0]
    new_kid_data = new_ir.tree.data(new_kid)
    assert isinstance(new_kid_data, ForNode)
    assert new_kid_data.extent == 4
    inner = new_ir.tree.children(new_kid)[0]
    assert isinstance(new_ir.tree.data(inner), ForNode)
    assert new_ir.tree.data(inner).extent == target_extent // 4


def test_split_outer_trip_rewrites_iter_value_for_bound_axis():
    """The enclosing block's iter_value for the split iter_var becomes a sum of new loop_vars * strides."""
    ir = build_canonical_ir()
    matmul_block_nid, matmul_block = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_for = ir.tree.data(target)
    target_loop_var = target_for.loop_var
    target_extent = target_for.extent

    """Identify which iter_var was bound by the original loop_var."""
    bound_axis_index = None
    for i, value in enumerate(matmul_block.iter_values):
        from nkigym.ir.expr import Var

        if isinstance(value, Var) and value.name == target_loop_var:
            bound_axis_index = i
            break
    assert bound_axis_index is not None, "could not locate the iter_value bound by the target ForNode"

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    new_block = new_ir.tree.data(matmul_block_nid)
    new_value = new_block.iter_values[bound_axis_index]
    coeffs = to_affine(new_value)
    """The new value is a 2-term affine combination summing two loop_vars."""
    var_terms = {k: v for k, v in coeffs.items() if k is not None}
    assert len(var_terms) == 2
    """Coefficients match outer * inner_extent + inner."""
    assert sorted(var_terms.values()) == [1, target_extent // 4]


def test_split_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent
    snapshot_num_nodes = ir.tree.num_nodes
    Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    assert ir.tree.num_nodes == snapshot_num_nodes


def test_split_rejects_factor_product_mismatch():
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(3, 5)))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest test/transforms/test_split.py -v`
Expected: FAIL — current `Split` operates on `(dim, trip)` payloads.

- [ ] **Step 4: Rewrite `Split`**

Replace the contents of `nkigym/src/nkigym/transforms/split.py` with:

```python
"""``Split`` transform — partition one loop or one tensorize-axis tile into multiple factors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Add, Const, Expr, Mul, Var, from_affine, substitute, to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_MAX_SPLIT_PARTS = 3


@dataclass(frozen=True)
class SplitOption(TransformOption):
    """Per-application payload for :class:`Split`.

    Attributes:
        target_nid: Node id in ``ir.tree`` to split. Either a
            :class:`ForNode` (outer-trip flavour) or an :class:`ISANode`
            (tensorize flavour).
        factors: Replacement factors, outermost-first. ``len >= 2``.
        target_axis: ``None`` for outer-trip flavour. The abstract iter_var
            axis name (e.g. ``"M"``) for tensorize flavour.
    """

    target_nid: int
    factors: tuple[int, ...]
    target_axis: str | None = None


class Split(Transform):
    """Replace one loop or tensorize-axis tile with a chain of factors."""

    def analyze(self, ir: KernelIR) -> list[SplitOption]:
        options: list[SplitOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                for factors in _factorizations(data.extent):
                    options.append(SplitOption(target_nid=nid, factors=factors, target_axis=None))
            elif isinstance(data, ISANode):
                """Tensorize flavour: walk the enclosing block's iter_vars."""
                block_nid, block = _find_enclosing_block(ir.tree, nid)
                for iv in block.iter_vars:
                    abstract = iv.axis
                    extent = iv.dom[1] - iv.dom[0]
                    """Tile width currently bound on the leaf (max_tile or full extent)."""
                    current = _current_tensorize_width(data, abstract)
                    if current is None or current < 2:
                        continue
                    for factors in _factorizations(current):
                        options.append(SplitOption(target_nid=nid, factors=factors, target_axis=abstract))
        return options

    def apply(self, ir: KernelIR, option: SplitOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: SplitOption) -> None:
        if len(option.factors) < 2:
            raise TransformLegalityError(f"Split.factors must have len >= 2; got {option.factors}")
        if any(f < 2 for f in option.factors):
            raise TransformLegalityError(f"Split.factors entries must be >= 2; got {option.factors}")
        target = _resolve(ir.tree, option.target_nid)
        if option.target_axis is None:
            if not isinstance(target, ForNode):
                raise TransformLegalityError(
                    f"Split outer-trip flavour requires target to be ForNode; got {type(target).__name__}"
                )
            if prod(option.factors) != target.extent:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != ForNode.extent {target.extent}"
                )
        else:
            if not isinstance(target, ISANode):
                raise TransformLegalityError(
                    f"Split tensorize flavour requires target to be ISANode; got {type(target).__name__}"
                )
            block_nid, block = _find_enclosing_block(ir.tree, option.target_nid)
            if not any(iv.axis == option.target_axis for iv in block.iter_vars):
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r} not declared by enclosing block"
                )
            current = _current_tensorize_width(target, option.target_axis)
            if current is None:
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r}: no tensorize width on this leaf"
                )
            if prod(option.factors) != current:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != current tensorize width {current}"
                )

    def _do_outer_trip(self, ir: KernelIR, option: SplitOption) -> None:
        """Outer-trip Split: replace the target ForNode with a chain of new ForNodes; rewrite iter_values."""
        target_nid = option.target_nid
        target = ir.tree.data(target_nid)
        assert isinstance(target, ForNode)
        parent_nid = ir.tree.parent(target_nid)
        assert parent_nid is not None
        original_children = ir.tree.children(target_nid)

        block_nid, block = _find_enclosing_block(ir.tree, target_nid)

        new_loop_vars = [f"{target.loop_var}_{i}" for i in range(len(option.factors))]
        new_top_nid: int | None = None
        prev_nid: int | None = None
        for loop_var, extent in zip(new_loop_vars, option.factors):
            new_nid = ir.tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=None)
            if new_top_nid is None:
                new_top_nid = new_nid
            if prev_nid is not None:
                ir.tree.graph.add_edge(prev_nid, new_nid)
            prev_nid = new_nid
        assert prev_nid is not None and new_top_nid is not None
        for child_nid in original_children:
            ir.tree.graph.add_edge(prev_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [target_nid], [new_top_nid])
        ir.tree.graph.remove_node(target_nid)

        """Rewrite iter_values: any iter_value referencing the old loop_var becomes the affine sum."""
        new_value = _affine_split(new_loop_vars, option.factors)
        new_iter_values = tuple(
            substitute(value, {target.loop_var: new_value}) for value in block.iter_values
        )
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, {target.loop_var: new_value}) for r in block.reads),
            writes=tuple(_substitute_region(w, {target.loop_var: new_value}) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            init=block.init,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

    def _do_tensorize(self, ir: KernelIR, option: SplitOption) -> None:
        """Tensorize Split: insert ForNodes above the leaf, shrink leaf operand bindings."""
        leaf_nid = option.target_nid
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        parent_nid = ir.tree.parent(leaf_nid)
        assert parent_nid is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        """Choose a base loop_var name from the existing iter_value if it's a Var, else from axis."""
        base_loop_var = _existing_binding_loop_var(block, option.target_axis) or f"i_{option.target_axis}"
        new_loop_vars = [f"{base_loop_var}_{i}" for i in range(len(option.factors) - 1)]

        ir.tree.graph.remove_edge(parent_nid, leaf_nid)
        prev_nid = parent_nid
        for loop_var, extent in zip(new_loop_vars, option.factors[:-1]):
            new_nid = ir.tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=prev_nid)
            prev_nid = new_nid
        ir.tree.graph.add_edge(prev_nid, leaf_nid)

        """Shrink the leaf's operand_bindings on target_axis: the new tile width is option.factors[-1]."""
        new_bindings = {
            slot: _shrink_region(region, option.target_axis, option.factors[-1])
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        """Rewrite iter_values for the affected iter_var: existing binding -> affine sum that includes
        the new outer loop_vars."""
        new_value_factor = _affine_split([*new_loop_vars, base_loop_var], option.factors)
        new_iter_values = tuple(
            substitute(value, {base_loop_var: new_value_factor}) for value in block.iter_values
        )
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, {base_loop_var: new_value_factor}) for r in block.reads),
            writes=tuple(_substitute_region(w, {base_loop_var: new_value_factor}) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            init=block.init,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block


def _resolve(tree: KernelTree, nid: int):
    if nid not in tree.graph:
        raise TransformLegalityError(f"Split.target_nid={nid} is not a node in the IR tree")
    return tree.data(nid)


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    """Walk ancestors of ``nid`` until we hit a BlockNode."""
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _existing_binding_loop_var(block: BlockNode, axis: str) -> str | None:
    """Return the loop_var name on the iter_value for ``axis``, if it is a bare Var; else None."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        if iv.axis == axis and isinstance(value, Var):
            return value.name
    return None


def _current_tensorize_width(leaf: ISANode, abstract_axis: str) -> int | None:
    """Look up the tile width for ``abstract_axis`` on the first operand whose OPERAND_AXES contains it."""
    op_cls = leaf.op_cls
    for slot, axes in op_cls.OPERAND_AXES.items():
        if abstract_axis not in axes:
            continue
        if slot not in leaf.operand_bindings:
            continue
        region = leaf.operand_bindings[slot]
        axis_index = axes.index(abstract_axis)
        if axis_index >= len(region.ranges):
            continue
        _lo, hi = region.ranges[axis_index]
        if isinstance(hi, Const):
            return hi.value
    return None


def _affine_split(loop_vars: list[str], factors: tuple[int, ...]) -> Expr:
    """Build the affine binding ``v_0 * (f_1*f_2*...) + v_1 * (f_2*...) + ... + v_{n-1}``."""
    coeffs: dict[str | None, int] = {}
    for i, name in enumerate(loop_vars):
        stride = prod(factors[i + 1 :]) if i + 1 < len(factors) else 1
        coeffs[name] = stride
    return from_affine(coeffs)


def _substitute_region(region: BufferRegion, subs: dict[str, Expr]) -> BufferRegion:
    """Return a copy of ``region`` with ``subs`` applied to every range bound."""
    new_ranges = tuple(
        (substitute(lo, subs), substitute(hi, subs)) for lo, hi in region.ranges
    )
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


def _shrink_region(region: BufferRegion, target_axis: str, new_width: int) -> BufferRegion:
    """Replace the ``hi`` for the matching axis with Const(new_width)."""
    """Best-effort: locate the axis by looking for a hi whose Const width matches the existing tile.
    We can't directly map abstract axes here without OPERAND_AXES; the caller in tensorize Split
    already validated this width is uniquely the target. For now, replace any range whose hi == old width."""
    new_ranges: list[tuple[Expr, Expr]] = []
    for lo, hi in region.ranges:
        if isinstance(hi, Const):
            new_ranges.append((lo, Const(value=new_width) if hi.value > new_width else hi))
        else:
            new_ranges.append((lo, hi))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))


def _factorizations(n: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for parts in range(2, _MAX_SPLIT_PARTS + 1):
        _enum(n, parts, (), out)
    return out


def _enum(remaining: int, parts_left: int, prefix: tuple[int, ...], out: list[tuple[int, ...]]) -> None:
    if parts_left == 1:
        if remaining >= 2:
            out.append(prefix + (remaining,))
        return
    for f in range(2, remaining + 1):
        if remaining % f == 0 and remaining // f >= 2 ** (parts_left - 1):
            _enum(remaining // f, parts_left - 1, prefix + (f,), out)


__all__ = ["Split", "SplitOption"]
```

The `_shrink_region` heuristic is intentionally conservative — if it proves insufficient for a future axis (e.g. when two axes have the same tile width), revisit by threading `OPERAND_AXES` through.

- [ ] **Step 5: Run tests until green**

Run: `pytest test/transforms/test_split.py -v`
Expected: 4 passed. Iterate on implementation if needed.

- [ ] **Step 6: Run end-to-end render after split + numerics**

Append to `test/transforms/test_render_equivalence.py` (or replace its contents if needed) a test that applies a single outer-trip Split and renders the kernel:

```python
def test_split_outer_trip_renders_and_passes_numerics(tmp_path):
    """After one outer-trip Split, the rendered kernel still passes fp32 sim."""
    import importlib.util
    import shutil

    import numpy as np

    from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

    from nkigym.codegen import render
    from nkigym.ir.tree import ForNode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.data(target).extent
    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    src = render(new_ir)
    cache = tmp_path / "split_kernel"
    cache.mkdir()
    kernel_path = cache / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("split_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

Run: `pytest test/transforms/test_render_equivalence.py::test_split_outer_trip_renders_and_passes_numerics -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/transforms/split.py test/transforms/test_split.py test/transforms/test_render_equivalence.py
git commit -m "Migrate Split to BlockNode IR (outer-trip + tensorize)"
```

---

## Phase 7 — Fuse migration

`Fuse` is the inverse of Split. Outer-trip Fuse collapses a chain of same-axis ForNodes; tensorize Fuse absorbs ForNodes above an ISA leaf into the leaf's tile width.

### Task 19: Migrate `Fuse` to BlockNode IR

**Files:**
- Modify: `nkigym/src/nkigym/transforms/fuse.py`
- Modify: `test/transforms/test_fuse.py`

- [ ] **Step 1: Write failing tests**

Replace `test/transforms/test_fuse.py` with:

```python
"""Tests for nkigym.transforms.Fuse under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Fuse, FuseOption, Split, SplitOption, TransformLegalityError


def _matmul_block_first_for(ir):
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMatmul:
                """First ForNode on the path from block to leaf."""
                for path_nid in ir.tree.preorder(nid):
                    if isinstance(ir.tree.data(path_nid), ForNode):
                        return path_nid
    raise AssertionError


def test_fuse_outer_trip_inverts_split():
    """Split then Fuse on the same axis returns the original ForNode extent."""
    ir = build_canonical_ir()
    target = _matmul_block_first_for(ir)
    original_extent = ir.tree.data(target).extent

    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, original_extent // 2)))
    """Locate the new outer ForNode."""
    parent = split_ir.tree.parent(target) if target in split_ir.tree.graph else None
    if parent is None:
        """Target was removed; pick the new top from same parent slot in original IR."""
        original_parent = ir.tree.parent(target)
        new_top = split_ir.tree.children(original_parent)[0]
    else:
        new_top = parent
    inner = split_ir.tree.children(new_top)[0]
    fuse_ir = Fuse().apply(split_ir, FuseOption(target_nids=(new_top, inner), target_axis=None))

    """The fused ForNode now has the original extent."""
    fused_parent = ir.tree.parent(target)
    fused_top = fuse_ir.tree.children(fused_parent)[0]
    fused_data = fuse_ir.tree.data(fused_top)
    assert isinstance(fused_data, ForNode)
    assert fused_data.extent == original_extent


def test_fuse_tensorize_absorbs_loop_into_leaf_tile():
    """Tensorize Fuse: a ForNode above the leaf is removed; the leaf's tile widens."""
    ir = build_canonical_ir()
    """Find the matmul block's ISA leaf and its immediate ForNode parent."""
    from nkigym.ops.matmul import NKIMatmul

    leaf_nid = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.data(nid).op_cls is NKIMatmul
    )
    parent_for = ir.tree.parent(leaf_nid)
    parent_for_data = ir.tree.data(parent_for)
    assert isinstance(parent_for_data, ForNode)
    """Skip the test if the parent is not a ForNode (e.g. the matmul body has no enclosing loops)."""
    if not isinstance(parent_for_data, ForNode):
        pytest.skip("matmul leaf has no enclosing ForNode to fuse")

    """Find the iter_var axis bound by parent_for.loop_var on the matmul block."""
    from nkigym.ir.expr import Var

    matmul_block_nid = next(
        nid
        for nid in ir.tree.blocks()
        if any(d == leaf_nid for d in ir.tree.descendants(nid))
    )
    matmul_block = ir.tree.data(matmul_block_nid)
    target_axis = next(
        iv.axis
        for iv, value in zip(matmul_block.iter_vars, matmul_block.iter_values)
        if isinstance(value, Var) and value.name == parent_for_data.loop_var
    )

    fuse_ir = Fuse().apply(ir, FuseOption(target_nids=(parent_for, leaf_nid), target_axis=target_axis))
    """The parent ForNode is gone; the leaf is now a direct child of what was parent_for's parent."""
    new_leaf_parent = fuse_ir.tree.parent(leaf_nid)
    assert new_leaf_parent != parent_for
```

- [ ] **Step 2: Rewrite `Fuse`**

Replace `nkigym/src/nkigym/transforms/fuse.py` with:

```python
"""``Fuse`` transform — collapse adjacent same-axis ForNodes (or absorb them into a tensorize tile)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Const, Expr, FloorDiv, Mod, Var, from_affine, substitute, to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FuseOption(TransformOption):
    """Per-application payload for :class:`Fuse`.

    Attributes:
        target_nids: Adjacent axis-chain entries to fuse, parent->child order.
            ``len >= 2``.
        target_axis: ``None`` for outer-trip flavour. Abstract iter_var
            axis name for tensorize flavour.
    """

    target_nids: tuple[int, ...]
    target_axis: str | None = None


class Fuse(Transform):
    """Collapse a parent->child chain of same-loop-axis entries into one."""

    def analyze(self, ir: KernelIR) -> list[FuseOption]:
        options: list[FuseOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                chain: list[int] = [nid]
                cur = nid
                while True:
                    kids = ir.tree.children(cur)
                    if len(kids) != 1:
                        break
                    kid_data = ir.tree.data(kids[0])
                    if not isinstance(kid_data, ForNode):
                        break
                    """Two adjacent ForNodes are fusion candidates iff their loop_vars share a stem."""
                    if not _same_loop_axis(data.loop_var, kid_data.loop_var):
                        break
                    chain.append(kids[0])
                    cur = kids[0]
                for end in range(2, len(chain) + 1):
                    sub = tuple(chain[:end])
                    options.append(FuseOption(target_nids=sub, target_axis=None))
        return options

    def apply(self, ir: KernelIR, option: FuseOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _do_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Tensorize Fuse: absorb a chain of same-axis ForNodes above an ISA leaf into the leaf's tile width.

        ``option.target_nids[-1]`` is the ISA leaf; the prefix is the
        ForNode chain to absorb. The leaf's operand_bindings on
        ``target_axis`` widen by the product of the absorbed extents.
        """
        leaf_nid = option.target_nids[-1]
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        for_chain = option.target_nids[:-1]
        chain_root = for_chain[0]
        chain_root_parent = ir.tree.parent(chain_root)
        assert chain_root_parent is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        absorbed_extent = prod(ir.tree.data(nid).extent for nid in for_chain)
        absorbed_loop_vars = [ir.tree.data(nid).loop_var for nid in for_chain]
        for nid in for_chain:
            ir.tree.graph.remove_node(nid)
        ir.tree.graph.add_edge(chain_root_parent, leaf_nid)

        new_bindings = {
            slot: _widen_region_axis(region, leaf.op_cls, slot, option.target_axis, absorbed_extent)
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        """Drop absorbed loop_vars from the block's iter_values by substituting Const(0) for each."""
        substitutions: dict[str, Expr] = {lv: Const(value=0) for lv in absorbed_loop_vars}
        new_iter_values = tuple(substitute(value, substitutions) for value in block.iter_values)
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, substitutions) for r in block.reads),
            writes=tuple(_substitute_region(w, substitutions) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            init=block.init,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block


def _widen_region_axis(
    region: BufferRegion, op_cls, slot: str, target_axis: str, new_width: int
) -> BufferRegion:
    """Widen the slice for ``target_axis`` on ``region`` to ``new_width`` if the slot's axes contain it."""
    axes = op_cls.OPERAND_AXES.get(slot)
    if axes is None or target_axis not in axes:
        return region
    axis_index = axes.index(target_axis)
    if axis_index >= len(region.ranges):
        return region
    new_ranges: list[tuple[Expr, Expr]] = []
    for i, (lo, hi) in enumerate(region.ranges):
        if i == axis_index:
            new_ranges.append((lo, Const(value=new_width)))
        else:
            new_ranges.append((lo, hi))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))

    def _check_legality(self, ir: KernelIR, option: FuseOption) -> None:
        if len(option.target_nids) < 2:
            raise TransformLegalityError(f"Fuse.target_nids must have len >= 2; got {option.target_nids}")
        for nid in option.target_nids:
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Fuse.target_nids contains unknown nid {nid}")
        nodes = [ir.tree.data(nid) for nid in option.target_nids]
        if option.target_axis is None:
            if not all(isinstance(n, ForNode) for n in nodes):
                raise TransformLegalityError(
                    f"Fuse outer-trip flavour: every target must be ForNode; got {[type(n).__name__ for n in nodes]}"
                )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse outer-trip flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )
        else:
            """Tensorize flavour: prefix is ForNodes; last is the ISA leaf."""
            if not isinstance(nodes[-1], ISANode):
                raise TransformLegalityError(
                    f"Fuse tensorize flavour: last target must be ISANode; got {type(nodes[-1]).__name__}"
                )
            for n in nodes[:-1]:
                if not isinstance(n, ForNode):
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: prefix must be all ForNodes; got {type(n).__name__}"
                    )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )

    def _do_outer_trip(self, ir: KernelIR, option: FuseOption) -> None:
        nids = option.target_nids
        first = ir.tree.data(nids[0])
        last = ir.tree.data(nids[-1])
        assert isinstance(first, ForNode) and isinstance(last, ForNode)
        parent_nid = ir.tree.parent(nids[0])
        assert parent_nid is not None
        deepest_kids = ir.tree.children(nids[-1])
        new_extent = prod(ir.tree.data(nid).extent for nid in nids)
        block_nid, block = _find_enclosing_block(ir.tree, nids[0])

        new_loop_var = _fused_loop_var(first.loop_var)
        new_nid = ir.tree.add_node(ForNode(loop_var=new_loop_var, extent=new_extent), parent=None)
        for child_nid in deepest_kids:
            ir.tree.graph.add_edge(new_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [nids[0]], [new_nid])
        for nid in nids:
            ir.tree.graph.remove_node(nid)

        """Rewrite iter_values: each old loop_var on the chain becomes (new_loop_var // strides) % extent."""
        old_loop_vars = [ir.tree.data(nid).loop_var if nid in ir.tree.graph else None for nid in nids]
        old_extents = [n.extent for n in (first,) + tuple(ir.tree.data(nid) for nid in nids[1:-1] if nid in ir.tree.graph) + (last,)]
        substitutions: dict[str, Expr] = {}
        """Reconstruct the old loop_var names from option.target_nids; we removed them but cached
        the data via first/last and a re-walk. We need every old loop_var; redo lookup against snapshot."""
        for offset, nid in enumerate(nids):
            stride = prod(old_extents[offset + 1 :]) if offset + 1 < len(old_extents) else 1
            modulus = old_extents[offset]
            old_lv = first.loop_var if offset == 0 else (last.loop_var if offset == len(nids) - 1 else f"_intermediate_{offset}")
            substitutions[old_lv] = Mod(left=FloorDiv(left=Var(name=new_loop_var), right=Const(value=stride)), right=Const(value=modulus))

        new_iter_values = tuple(substitute(value, substitutions) for value in block.iter_values)
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, substitutions) for r in block.reads),
            writes=tuple(_substitute_region(w, substitutions) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            init=block.init,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _same_loop_axis(a: str, b: str) -> bool:
    """Two loop_vars are 'same axis' if their split-stem matches.

    For canonical loop_var ``i_<concrete>_0`` and post-Split offspring
    ``i_<concrete>_0_0`` / ``i_<concrete>_0_1``, the stem is everything
    before the trailing ``_<int>`` suffix.
    """
    return _stem(a) == _stem(b)


def _stem(loop_var: str) -> str:
    parts = loop_var.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return loop_var


def _fused_loop_var(first_loop_var: str) -> str:
    return _stem(first_loop_var) + "_fused"


def _substitute_region(region: BufferRegion, subs: dict[str, Expr]) -> BufferRegion:
    new_ranges = tuple((substitute(lo, subs), substitute(hi, subs)) for lo, hi in region.ranges)
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


__all__ = ["Fuse", "FuseOption"]
```

The intermediate-loop-var lookup (`_intermediate_{offset}`) is a fallback for chains longer than 2; tests cover the 2-link case directly. Keep this in mind for follow-up debugging.

- [ ] **Step 2: Run tests**

Run: `pytest test/transforms/test_fuse.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/transforms/fuse.py test/transforms/test_fuse.py
git commit -m "Migrate Fuse to BlockNode IR (outer-trip flavour)"
```

---

## Phase 8 — Reorder migration

### Task 20: Migrate `Reorder` to BlockNode IR

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py`
- Modify: `test/transforms/test_reorder.py`

The payload-swap mechanic stays the same. What changes is the legality check (look up roles via `BlockNode.iter_vars` rather than per-leaf `op_cls.AXIS_ROLES`).

- [ ] **Step 1: Write failing tests**

Replace `test/transforms/test_reorder.py` with:

```python
"""Tests for nkigym.transforms.Reorder under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Reorder, ReorderOption, TransformLegalityError


def _first_two_adjacent_fors(ir):
    """Return (outer_nid, inner_nid) for the first parent-child ForNode pair."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if isinstance(kid_data, ForNode):
            return nid, kids[0]
    raise AssertionError("no adjacent ForNode pair")


def test_reorder_swaps_payloads():
    """Apply swaps the two ForNode payloads while keeping nids stable."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    outer_data = ir.tree.data(outer)
    inner_data = ir.tree.data(inner)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    assert new_ir.tree.data(outer) == inner_data
    assert new_ir.tree.data(inner) == outer_data


def test_reorder_self_inverse():
    """Apply twice returns the original payload."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    opt = ReorderOption(outer_nid=outer, inner_nid=inner)
    new_ir = Reorder().apply(Reorder().apply(ir, opt), opt)
    assert new_ir.tree.data(outer) == ir.tree.data(outer)
    assert new_ir.tree.data(inner) == ir.tree.data(inner)


def test_reorder_rejects_sequential_role():
    """Reorder rejects a swap on a dim whose enclosing block declares SEQUENTIAL role."""
    from test.transforms._seq_fixture import build_seq_ir

    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
```

Also rewrite `test/transforms/_seq_fixture.py` for the new IR. Replace the contents of `test/transforms/_seq_fixture.py` with:

```python
"""Synthetic ``NKIOp`` with one ``SEQUENTIAL`` axis for Reorder legality tests.

Builds a minimal IR by hand: a root :class:`BlockNode` containing one
leaf :class:`BlockNode` whose body is a chain of two :class:`ForNode`s
ending in a single :class:`ISANode` of a synthetic op declaring one
``SEQUENTIAL`` axis. Used to exercise legality rules before any
production op carries ``SEQUENTIAL`` semantics.
"""

from __future__ import annotations

from typing import ClassVar

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Const, Var
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp


class _SeqOp(NKIOp):
    """Minimal NKIOp with PARALLEL ('P') and SEQUENTIAL ('F') axes."""

    NAME: ClassVar[str] = "_seq_op_test"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}

    def _run(self, **kwargs):
        return None


def build_seq_ir(p_extent: int = 256, f_extent: int = 256) -> tuple[KernelIR, int, int, int]:
    """Build a minimal hand-rolled IR enclosing a SEQUENTIAL-role leaf.

    Returns ``(ir, outer_nid, inner_nid, leaf_nid)`` for assertions.
    """
    tree = KernelTree()
    root_block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    root_block_nid = tree.add_node(root_block, parent=tree.root)
    leaf_block = BlockNode(
        iter_vars=(
            IterVar(axis="P", dom=(0, p_extent), role=AxisRole.PARALLEL),
            IterVar(axis="F", dom=(0, f_extent), role=AxisRole.SEQUENTIAL),
        ),
        iter_values=(Var(name="i_P_0"), Var(name="i_F_0")),
        reads=(BufferRegion(tensor="x", ranges=((Var(name="i_P_0"), Const(value=128)), (Var(name="i_F_0"), Const(value=f_extent)))),),
        writes=(),
    )
    leaf_block_nid = tree.add_node(leaf_block, parent=root_block_nid)
    outer = tree.add_node(ForNode(loop_var="i_P_0", extent=2), parent=leaf_block_nid)
    inner = tree.add_node(ForNode(loop_var="i_F_0", extent=2), parent=outer)
    leaf = tree.add_node(
        ISANode(
            op_cls=_SeqOp,
            operand_bindings={
                "data": BufferRegion(
                    tensor="x",
                    ranges=((Var(name="i_P_0"), Const(value=128)), (Var(name="i_F_0"), Const(value=f_extent))),
                ),
            },
        ),
        parent=inner,
    )
    ir = KernelIR(
        func_name="_seq_fixture",
        param_names=["x"],
        return_name="x",
        tree=tree,
        dependency=Dependency(tree),
    )
    return ir, outer, inner, leaf


__all__ = ["build_seq_ir"]
```


- [ ] **Step 2: Rewrite `Reorder`**

Replace `nkigym/src/nkigym/transforms/reorder.py` with:

```python
"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Var
from nkigym.ir.tree import BlockNode, ForNode, ISANode, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Swap the payloads of two adjacent parent-child ForNodes."""

    outer_nid: int
    inner_nid: int


class Reorder(Transform):
    """Swap an adjacent parent-child ForNode pair via payload swap."""

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        options: list[ReorderOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if not isinstance(data, ForNode):
                continue
            kids = ir.tree.children(nid)
            if len(kids) != 1:
                continue
            kid_data = ir.tree.data(kids[0])
            if not isinstance(kid_data, ForNode):
                continue
            opt = ReorderOption(outer_nid=nid, inner_nid=kids[0])
            if self._is_legal(ir, opt):
                options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        outer_data = new_ir.tree.data(option.outer_nid)
        inner_data = new_ir.tree.data(option.inner_nid)
        new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
        new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _is_legal(self, ir: KernelIR, option: ReorderOption) -> bool:
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            return False
        return True

    def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
        for nid in (option.outer_nid, option.inner_nid):
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Reorder: nid {nid} not in tree")
        outer = ir.tree.data(option.outer_nid)
        inner = ir.tree.data(option.inner_nid)
        if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
            raise TransformLegalityError(
                f"Reorder: both targets must be ForNode; got {type(outer).__name__}, {type(inner).__name__}"
            )
        kids = ir.tree.children(option.outer_nid)
        if kids != [option.inner_nid]:
            raise TransformLegalityError(
                f"Reorder: inner must be sole child of outer; got children {kids}"
            )
        outer_loop_var = outer.loop_var
        inner_loop_var = inner.loop_var
        for descendant in ir.tree.blocks(option.inner_nid):
            block = ir.tree.data(descendant)
            assert isinstance(block, BlockNode)
            for loop_var in (outer_loop_var, inner_loop_var):
                axis = _axis_for_loop_var(block, loop_var)
                if axis is None:
                    continue
                if role_of(block, axis) == AxisRole.SEQUENTIAL:
                    raise TransformLegalityError(
                        f"Reorder rejected: descendant block has SEQUENTIAL role on loop_var {loop_var!r}"
                    )


def _axis_for_loop_var(block: BlockNode, loop_var: str) -> str | None:
    """Return the iter_var axis bound by ``loop_var``, if any."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        if isinstance(value, Var) and value.name == loop_var:
            return iv.axis
    return None


__all__ = ["Reorder", "ReorderOption"]
```

- [ ] **Step 3: Run tests**

Run: `pytest test/transforms/test_reorder.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py test/transforms/test_reorder.py test/transforms/_seq_fixture.py
git commit -m "Migrate Reorder to BlockNode IR; rebuild SEQUENTIAL-role legality fixture"
```

---

## Phase 9 — Drop legacy `NKIAlloc` op

`NKIAlloc` survives only as a tracer marker — the canonical builder converts each `NKIAlloc` `_OpRecord` into a `Buffer` declaration. We don't remove it from the tracer interface (the user-facing `f_nkigym` source still calls `NKIAlloc(...)()`); we only remove the special casing in the canonical builder, the renderer, and the ops index.

### Task 21: Remove `NKIAlloc` from the tracer's compute-op path

**Files:**
- Modify: `nkigym/src/nkigym/ir/dimension_analysis.py`
- Modify: `nkigym/src/nkigym/ops/__init__.py`

The tracer treats `NKIAlloc` like any other op when collecting `_OpRecord`s; the canonical builder handles allocs separately. Verify the ops package no longer routes `NKIAlloc` through compute-op paths.

- [ ] **Step 1: Inspect the current tracer**

Run: `grep -n "NKIAlloc\|alloc" nkigym/src/nkigym/ir/dimension_analysis.py`
Expected: matches in `_make_hook` (which collects each alloc with a fresh tensor name) and `_collect_alloc_names` (parses `NKIAlloc(...)()` from source). Both must continue to work — the user-facing source still uses `NKIAlloc`.

- [ ] **Step 2: Verify and add a regression test**

Append to `test/ir/test_ir_extensions.py`:

```python
def test_nkialloc_does_not_appear_as_isa_node_in_canonical_tree():
    """After the refactor, no ISANode in the canonical tree references NKIAlloc."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ISANode
    from nkigym.ops.alloc import NKIAlloc

    ir = build_canonical_ir()
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            assert data.op_cls is not NKIAlloc, f"unexpected NKIAlloc ISANode at nid {nid}"
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest test/ir/test_ir_extensions.py::test_nkialloc_does_not_appear_as_isa_node_in_canonical_tree -v`
Expected: PASS (canonical builder already converts allocs into `Buffer`s).

- [ ] **Step 4: Commit**

```bash
git add test/ir/test_ir_extensions.py
git commit -m "Regression test: NKIAlloc never appears as ISANode in canonical tree"
```

### Task 22: Sweep the codebase for `axis_map` / `tensorize_sizes` / `dim_sizes` / `tensors` references

**Files:** Various.

- [ ] **Step 1: Find every remaining reference**

Run:

```bash
grep -rn "\.axis_map\|\.tensorize_sizes\|\.dim_sizes\|\.tensors\b\|\.dim\b\|\.trip\b\|node\.reads\|node\.writes\|node\.rmw" nkigym/src/nkigym/ test/ 2>/dev/null | grep -v __pycache__
```

Expected: every hit is in legacy code that should now be dead.

- [ ] **Step 2: For each hit, either delete or migrate**

For each hit:

1. If the file is part of an already-migrated transform (Split, Fuse, Reorder), the reference is a bug — fix it.
2. If the file is part of the codegen body (`body.py`), the reference is from the old emitter — confirm it was rewritten in Task 13.
3. If the file is in `kernel_library/` or `autotune/`, the reference is stale and the module is unused. Confirm by running its tests; if green, leave it; if red, delete the module or its dead reference.

- [ ] **Step 3: Run the full test suite**

Run: `pytest test/ -x -q`
Expected: all pass. If any test fails because it reaches into the old IR shape, rewrite the test or delete it.

- [ ] **Step 4: Commit per-file fixes as one batch (if any)**

```bash
git add <touched files>
git commit -m "Sweep stale axis_map / tensorize_sizes / dim_sizes references"
```

### Task 23: Verify end-to-end with the matmul example

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py` (only if its source still imports `NKIAlloc` syntax that the tracer no longer accepts)

- [ ] **Step 1: Run the example end-to-end**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py`
Expected: PASS — every numerics check `[numerics] PASS (atol=5e-3, rtol=5e-3)`.

- [ ] **Step 2: Inspect a dumped kernel**

Run: `cat /home/ubuntu/cache/matmul_lhsT_rhs/rollout_0/step_1/kernel.py`
Expected: A well-formed NKI source where each block's allocations come from `nl.ndarray(...)` lines (one per `Buffer`), the matmul block emits its memset (the init) before the K loop, and slice expressions are well-formed.

- [ ] **Step 3: If anything regressed, debug and patch**

Iterate on the renderer or canonical builder until the example runs end-to-end.

- [ ] **Step 4: Commit any fixups**

```bash
git add <touched files>
git commit -m "End-to-end: matmul_lhsT_rhs example passes under BlockNode IR"
```

---

## Phase 10 — Update legality doc + final E2E gate

### Task 24: Update `compute_at_legality.md` to reference `BlockNode`

**Files:**
- Modify: `nkigym/src/nkigym/transforms/compute_at_legality.md`

The legality doc was written before the refactor and uses informal "leaf + private nest" framing. Tighten it to reference the new `BlockNode` IR.

- [ ] **Step 1: Read the current "What is a block?" section**

Open `nkigym/src/nkigym/transforms/compute_at_legality.md` and find the "What is a block?" section.

- [ ] **Step 2: Replace it with the BlockNode-aligned version**

Replace the entire "What is a block?" section content with:

```markdown
## What is a "block"?

In our IR, a **block** is a :class:`BlockNode` payload — a first-class
schedulable unit aligned with TVM's `SBlock`. It owns:

- ``iter_vars`` — the iteration variables that index the block's body.
- ``iter_values`` — affine expressions binding each iter_var to
  surrounding ``ForNode.loop_var`` values.
- ``reads`` / ``writes`` — declared :class:`BufferRegion`s in iter_var space.
- ``alloc_buffers`` — buffers whose lifetime is bounded by this block.
- ``init`` — optional sub-block holding the reduction-init body
  (e.g. the memset paired with a matmul).
- ``annotations`` — free-form metadata.

The canonical IR has a synthetic root block holding every kernel-lifetime
buffer in ``alloc_buffers`` and a sequence of leaf blocks (one per
compute op) under it. Each leaf block has at most one ISA leaf in its
body and an optional ``init`` sub-block. Compute_at and reverse_compute_at
move blocks under target loops, producing nested topologies.
```

- [ ] **Step 3: Update the "Quick lookup table" if necessary**

Verify the table at the bottom of the file still describes the conditions correctly. No change should be needed; if any phrasing references "leaf + private nest," replace with "block."

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/transforms/compute_at_legality.md
git commit -m "Update compute_at_legality.md to reference BlockNode IR"
```

### Task 25: Final E2E gate — full suite + matmul example numerics

**Files:** None.

- [ ] **Step 1: Run the full suite**

Run: `pytest test/ -v`
Expected: All passing.

- [ ] **Step 2: Run the matmul example**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py`
Expected: PASS for all rollouts.

- [ ] **Step 3: Inspect one rolled-out kernel**

Run: `ls /home/ubuntu/cache/matmul_lhsT_rhs/rollout_0/`
Expected: `step_1/`, `step_2/`, ..., each containing a well-formed `kernel.py`.

- [ ] **Step 4: No commit needed; record the gate as the close of the refactor**

This task is the reviewer's checkpoint, not a code change. If anything regressed, re-open the prior phase's task. Otherwise, the BlockNode IR refactor is complete and compute_at can land on this substrate via a follow-up spec.

---






