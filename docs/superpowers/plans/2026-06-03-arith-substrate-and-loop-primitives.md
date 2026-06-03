# Arith Substrate + Loop Primitives (Split / Fuse / Reorder) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port TVM TensorIR's `arith` subsystem (Analyzer / RewriteSimplifier / IterMap / IntSet) into the custom IR as a shared, separately-tested substrate, then rebuild `split` / `fuse` / `reorder` as thin clients that call it instead of inlining ad-hoc arithmetic.

**Architecture:** A new `nkigym/ir/arith/` package mirrors TVM `src/arith/` file-for-file, depending on nothing above it. Verification uses TVM run as a live oracle in two layers: Layer A asserts each ported arith primitive matches `tvm.arith` on a generated corpus (in-process, clean); Layer B asserts each transform's loop structure matches TVM's `sch.split/fuse/reorder`. The existing hand-ladder byte-exact gate (`kernel_transforms.py`) stays primary.

**Tech Stack:** Python 3.12, the `kernel-env` venv, networkx-backed `KernelTree`, the built TVM fork at `/home/ubuntu/tvm` (used as test-only oracle, never shipped).

**Reference — TVM source to mirror (read these before each phase):**
- `src/arith/analyzer.cc` — `Analyzer` facade (`Simplify`, `CanProve`, `CanProveEqual`, `Bind`).
- `src/arith/rewrite_simplify.cc` — `RewriteSimplifier::Impl::VisitExpr_` per-node rules + `TryCompare`.
- `src/arith/canonical_simplify.cc` — sum-of-products canonical form.
- `src/arith/iter_affine_map.cc` — `DetectIterMap`, `IterMapSimplify`, `NormalizeIterMapToExpr`.
- `src/arith/int_set.cc` — `IntSet` (Interval / Union / Intersect / EvalSet).
- `src/s_tir/schedule/primitive/loop_transformation.cc` — `Split` (line 396), `Fuse` (line 876), `SubstituteVarAndCollectOpaqueBlock` (line 50), `IterMapSimplifyBlockBinding` (line 82).
- `src/s_tir/schedule/primitive/reorder_block_iter_var.cc` — `Reorder`.

**Environment for every command below:**
```bash
source ~/venvs/kernel-env/bin/activate
# nkigym tests:
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src
# TVM oracle (test-only):
TVM_LIBRARY_PATH=/home/ubuntu/tvm/build/lib PYTHONPATH=$PYTHONPATH:/home/ubuntu/tvm/python
```

**Verified TVM oracle entry points** (confirmed working in the built fork):
```python
import tvm.tirx as T               # T.Var, T.Add, T.Mul, T.FloorDiv, T.FloorMod, T.IntImm
from tvm import arith, ir
a = arith.Analyzer(); a.bind(x, ir.Range(0, 128))
a.simplify(expr)                   # (x*512+3)%512 -> 3
a.can_prove(pred)                  # x < 128 -> True
arith.detect_iter_map(indices, input_iters, predicate, check_level)
arith.IntSet                       # interval / union / intersect
```

---

## File Structure

**Created:**
- `nkigym/src/nkigym/ir/arith/__init__.py` — package exports.
- `nkigym/src/nkigym/ir/arith/expr.py` — Expr AST (moved from `ir/expr.py`, extended).
- `nkigym/src/nkigym/ir/arith/analyzer.py` — `Analyzer` facade.
- `nkigym/src/nkigym/ir/arith/rewrite_simplify.py` — `RewriteSimplifier`.
- `nkigym/src/nkigym/ir/arith/canonical_simplify.py` — `CanonicalSimplifier`.
- `nkigym/src/nkigym/ir/arith/iter_map.py` — `detect_iter_map` / `iter_map_simplify`.
- `nkigym/src/nkigym/ir/arith/int_set.py` — `IntSet` (subsumes `interval.py` logic later; not this spec).
- `test/ir/arith/_tvm_bridge.py` — test-only `Expr ↔ tvm PrimExpr` bridge.
- `test/ir/arith/test_expr.py`, `test_analyzer.py`, `test_rewrite_simplify.py`, `test_iter_map.py`, `test_int_set.py` — Layer-A oracle tests.
- `test/transforms/_tvm_struct_oracle.py` — test-only Layer-B structural comparator.

**Modified:**
- `nkigym/src/nkigym/ir/__init__.py` — re-export `arith.expr` names (kill call-site churn).
- `nkigym/src/nkigym/transforms/split.py` — rewrite as arith client.
- `nkigym/src/nkigym/transforms/reorder.py` — rewrite as arith client.
- `nkigym/src/nkigym/transforms/fuse.py` — rewrite as arith client.
- Every current importer of `nkigym.ir.expr` — repoint to the `ir/__init__` re-export.

**Deleted:**
- `nkigym/src/nkigym/ir/expr.py` — moved into `ir/arith/expr.py`.
- Inline affine/division logic inside `split.py` / `fuse.py` / `reorder.py`.

**NOT touched (verified by import audit — still load-bearing elsewhere):**
- `transforms/_domain_solve.py` (only `_code_motion.py`/compute_at imports it → Spec 2).
- `ir/interval.py` (only `ir/dependency.py` imports it → folded into `int_set.py` in Spec 2).
- `transforms/compute_at.py`, `reverse_compute_at.py`, `_code_motion.py`, `_normalize.py`, `_tile_region.py`.

---

## Phase 0 — Expr extension + TVM bridge

### Task 1: Move `ir/expr.py` → `ir/arith/expr.py` with a re-export shim

**Files:**
- Create: `nkigym/src/nkigym/ir/arith/__init__.py`
- Create: `nkigym/src/nkigym/ir/arith/expr.py` (content moved from `ir/expr.py`)
- Modify: `nkigym/src/nkigym/ir/__init__.py`
- Delete: `nkigym/src/nkigym/ir/expr.py`

- [ ] **Step 1: Find every importer of `nkigym.ir.expr`**

Run:
```bash
grep -rln "nkigym.ir.expr\|from nkigym.ir import.*\(Const\|Var\|Add\|Mul\|FloorDiv\|Mod\|Expr\|to_affine\|from_affine\|substitute\|format_expr\|NonAffineError\)" nkigym/src test | grep -v ".pyc"
```
Expected: a list including `tree.py`, `interval.py`, `dependency.py`, `_normalize.py`, `split.py`, `fuse.py`, `reorder.py`, `_tile_region.py`, and several tests. Record it.

- [ ] **Step 2: Create the arith package and move expr.py verbatim**

```bash
mkdir -p nkigym/src/nkigym/ir/arith
git mv nkigym/src/nkigym/ir/expr.py nkigym/src/nkigym/ir/arith/expr.py
```

Create `nkigym/src/nkigym/ir/arith/__init__.py`:
```python
"""Ported TVM ``arith`` substrate: Expr AST, Analyzer, simplifiers, IterMap, IntSet.

Mirrors TVM ``src/arith/`` file-for-file. Depends on nothing above it in the IR
stack; ``transforms/`` and the rest of ``ir/`` call into here, never the reverse.
"""

from nkigym.ir.arith.expr import (
    Add,
    Const,
    Expr,
    FloorDiv,
    Mod,
    Mul,
    NonAffineError,
    Var,
    format_expr,
    from_affine,
    substitute,
    to_affine,
)

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

- [ ] **Step 3: Add a re-export shim in `ir/__init__.py`**

So existing `from nkigym.ir.expr import X` becomes `from nkigym.ir import X`. Inspect the current `ir/__init__.py` first (`Read nkigym/src/nkigym/ir/__init__.py`), then add, near the other re-exports:
```python
from nkigym.ir.arith.expr import (
    Add,
    Const,
    Expr,
    FloorDiv,
    Mod,
    Mul,
    NonAffineError,
    Var,
    format_expr,
    from_affine,
    substitute,
    to_affine,
)
```
and add those names to `__all__`.

- [ ] **Step 4: Repoint every importer found in Step 1**

Change each `from nkigym.ir.expr import ...` to `from nkigym.ir.arith.expr import ...` (the canonical new path). Leave `from nkigym.ir import ...` forms alone (the shim covers them). Use the Step-1 list; edit each file.

- [ ] **Step 5: Run the full suite to verify the move is transparent**

Run:
```bash
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest -q
```
Expected: same pass count as baseline (226 passed), 0 import errors.

- [ ] **Step 6: Verify old module is gone**

Run: `test ! -f nkigym/src/nkigym/ir/expr.py && echo "DELETED"`
Expected: `DELETED`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Move ir/expr.py -> ir/arith/expr.py with re-export shim (arith package scaffold)"
```

### Task 2: Extend Expr with Sub / Min / Max / predicate nodes + opaque non-affine carry

**Files:**
- Modify: `nkigym/src/nkigym/ir/arith/expr.py`
- Modify: `nkigym/src/nkigym/ir/arith/__init__.py` (export new nodes)
- Test: `test/ir/arith/test_expr.py`

- [ ] **Step 1: Write the failing test**

Create `test/ir/arith/__init__.py` (empty) and `test/ir/arith/test_expr.py`:
```python
from nkigym.ir.arith.expr import Add, Const, EQ, LE, LT, Max, Min, Mul, Sub, Var, substitute


def test_new_nodes_construct_and_substitute():
    e = Sub(left=Var(name="x"), right=Const(value=1))
    assert isinstance(e, Sub)
    out = substitute(e, {"x": Const(value=5)})
    assert out == Sub(left=Const(value=5), right=Const(value=1))


def test_min_max_predicate_nodes():
    assert isinstance(Min(left=Var(name="a"), right=Var(name="b")), Min)
    assert isinstance(Max(left=Var(name="a"), right=Var(name="b")), Max)
    assert isinstance(LT(left=Var(name="i"), right=Const(value=128)), LT)
    assert isinstance(LE(left=Var(name="i"), right=Const(value=128)), LE)
    assert isinstance(EQ(left=Var(name="i"), right=Const(value=0)), EQ)


def test_predicate_substitute_recurses():
    p = LT(left=Add(left=Mul(left=Var(name="i"), right=Const(value=512)), right=Var(name="j")), right=Const(value=2048))
    out = substitute(p, {"i": Const(value=3)})
    assert out == LT(
        left=Add(left=Mul(left=Const(value=3), right=Const(value=512)), right=Var(name="j")),
        right=Const(value=2048),
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/arith/test_expr.py -q`
Expected: FAIL — `ImportError: cannot import name 'Sub'`.

- [ ] **Step 3: Add the new dataclasses to `expr.py`**

After the existing `Mod` dataclass, add:
```python
@dataclass(frozen=True, kw_only=True)
class Sub:
    """Binary subtraction."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Min:
    """Binary minimum."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Max:
    """Binary maximum."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class LT:
    """Predicate ``left < right``."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class LE:
    """Predicate ``left <= right``."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class EQ:
    """Predicate ``left == right``."""

    left: "Expr"
    right: "Expr"
```

Update the `Expr` union:
```python
Expr = Const | Var | Add | Sub | Mul | FloorDiv | Mod | Min | Max | LT | LE | EQ
```

- [ ] **Step 4: Extend `substitute` to recurse into the new nodes**

In `substitute`, before the final `raise TypeError`, add handling that recurses left/right for `Sub, Min, Max, LT, LE, EQ` (mirror the existing `Add` branch). Build a result var; single return at the bottom (codegen style: one return per function). Concretely, replace the body with a dispatch that covers every node type and returns once:
```python
def substitute(expr: Expr, subs: dict[str, Expr]) -> Expr:
    """Replace each ``Var(name)`` in ``expr`` by ``subs[name]`` recursively."""
    result: Expr
    if isinstance(expr, Const):
        result = expr
    elif isinstance(expr, Var):
        result = subs.get(expr.name, expr)
    elif isinstance(expr, (Add, Sub, Mul, FloorDiv, Mod, Min, Max, LT, LE, EQ)):
        cls = type(expr)
        result = cls(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    else:
        raise TypeError(f"Unknown Expr node {type(expr).__name__}")
    return result
```

- [ ] **Step 5: Make `to_affine` carry non-affine subterms opaquely (do NOT raise)**

The current `_accumulate` raises `NonAffineError` on `Var*Var`, non-const div/mod. For the simplifier we need non-affine subterms to survive as opaque atoms. Add an opaque-carry path: introduce a module-level helper `affine_or_opaque(expr)` that returns the affine dict when affine, else `{(_opaque_key(expr)): 1}` where the key is the expr itself (frozen dataclasses are hashable). KEEP `to_affine` raising for callers that require pure affinity (it is used by `_normalize.py` and `format_expr`); add a SEPARATE non-raising entry point so existing behavior is unchanged:
```python
def affine_terms(expr: Expr) -> dict[object, int]:
    """Affine coefficients, carrying non-affine subterms as opaque atoms.

    Like :func:`to_affine` but never raises: a subterm that is not affine in
    Vars (``Var*Var``, non-const divisor) becomes a single opaque key (the
    subterm itself) with coefficient 1, so predicates and partially-affine
    expressions remain representable for the simplifier.
    """
    ...
```
Add a focused unit test in `test_expr.py` asserting `affine_terms(Mul(Var x, Var y))` yields `{Mul(...): 1}` not a raise, and `affine_terms(Add(Mul(Var i, Const 512), Var j))` yields `{Var(i): 512, Var(j): 1}`.

- [ ] **Step 6: Export the new names**

Add `EQ, LE, LT, Max, Min, Sub, affine_terms` to `expr.py`'s `__all__`, to `ir/arith/__init__.py`, and to the `ir/__init__.py` shim.

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=... python -m pytest test/ir/arith/test_expr.py -q`
Expected: PASS (all new tests).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Extend Expr: Sub/Min/Max/LT/LE/EQ + affine_terms opaque non-affine carry"
```

### Task 3: Test-only Expr ↔ TVM PrimExpr bridge

**Files:**
- Create: `test/ir/arith/_tvm_bridge.py`
- Test: `test/ir/arith/test_tvm_bridge.py`

- [ ] **Step 1: Write the failing test**

Create `test/ir/arith/test_tvm_bridge.py` (the bridge maps our `Mod`→`tvm.tirx.FloorMod`, `FloorDiv`→`tvm.tirx.FloorDiv`):
```python
import pytest

tvm = pytest.importorskip("tvm")
from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul, Var
from test.ir.arith._tvm_bridge import to_tvm, from_tvm


def test_roundtrip_affine():
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    assert from_tvm(to_tvm(e)) == e


def test_roundtrip_div_mod():
    e = Mod(left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512))
    assert from_tvm(to_tvm(e)) == e
```

- [ ] **Step 2: Run to verify it fails**

Run: `TVM_LIBRARY_PATH=/home/ubuntu/tvm/build/lib PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/tvm/python python -m pytest test/ir/arith/test_tvm_bridge.py -q`
Expected: FAIL — `ModuleNotFoundError: test.ir.arith._tvm_bridge`.

- [ ] **Step 3: Implement the bridge**

Create `test/ir/arith/_tvm_bridge.py`:
```python
"""Test-only bridge between our Expr AST and TVM PrimExpr (for the arith oracle).

Never imported by shipped ``nkigym/`` code — only by Layer-A oracle tests.
``int32`` throughout (our extents/indices are all int32-range).
"""
from __future__ import annotations

import tvm.tirx as T

from nkigym.ir.arith.expr import Add, Const, EQ, Expr, FloorDiv, LE, LT, Max, Min, Mod, Mul, Sub, Var

_I32 = "int32"


def to_tvm(expr: Expr, env: dict[str, "T.Var"] | None = None) -> "T.PrimExpr":
    """Lower our Expr to a tvm.tirx PrimExpr. ``env`` interns Var names -> T.Var."""
    env = {} if env is None else env

    def rec(e: Expr):
        if isinstance(e, Const):
            return T.IntImm(_I32, e.value)
        if isinstance(e, Var):
            if e.name not in env:
                env[e.name] = T.Var(e.name, _I32)
            return env[e.name]
        if isinstance(e, Add):
            return T.Add(rec(e.left), rec(e.right))
        if isinstance(e, Sub):
            return T.Sub(rec(e.left), rec(e.right))
        if isinstance(e, Mul):
            return T.Mul(rec(e.left), rec(e.right))
        if isinstance(e, FloorDiv):
            return T.FloorDiv(rec(e.left), rec(e.right))
        if isinstance(e, Mod):
            return T.FloorMod(rec(e.left), rec(e.right))
        if isinstance(e, Min):
            return T.Min(rec(e.left), rec(e.right))
        if isinstance(e, Max):
            return T.Max(rec(e.left), rec(e.right))
        if isinstance(e, LT):
            return rec(e.left) < rec(e.right)
        if isinstance(e, LE):
            return rec(e.left) <= rec(e.right)
        if isinstance(e, EQ):
            return rec(e.left) == rec(e.right)
        raise TypeError(f"to_tvm: unknown node {type(e).__name__}")

    return rec(expr)


def from_tvm(pe: "T.PrimExpr") -> Expr:
    """Lift a tvm.tirx PrimExpr back to our Expr (inverse of to_tvm for the supported subset)."""
    import tvm.tirx as T

    if isinstance(pe, T.IntImm):
        return Const(value=int(pe.value))
    if isinstance(pe, T.Var):
        return Var(name=pe.name)
    binops = {T.Add: Add, T.Sub: Sub, T.Mul: Mul, T.FloorDiv: FloorDiv, T.FloorMod: Mod, T.Min: Min, T.Max: Max}
    for tvm_cls, our_cls in binops.items():
        if isinstance(pe, tvm_cls):
            return our_cls(left=from_tvm(pe.a), right=from_tvm(pe.b))
    raise TypeError(f"from_tvm: unsupported PrimExpr {type(pe).__name__}")
```
(Verify the exact attribute names `pe.value` / `pe.name` / `pe.a` / `pe.b` against the built TVM first by inspecting one node in a REPL; fix the accessors if they differ.)

- [ ] **Step 4: Run tests**

Run: `TVM_LIBRARY_PATH=... PYTHONPATH=...:/home/ubuntu/tvm/python python -m pytest test/ir/arith/test_tvm_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Test-only Expr<->TVM PrimExpr bridge for the arith oracle"
```

---

## Phase 1 — Arith core

### Task 4: `RewriteSimplifier` — per-node rewrite rules (Add/Sub/Mul/FloorDiv/Mod)

**Files:**
- Create: `nkigym/src/nkigym/ir/arith/rewrite_simplify.py`
- Test: `test/ir/arith/test_rewrite_simplify.py`

Mirror `src/arith/rewrite_simplify.cc` `VisitExpr_` for each node. Port ONLY the rule families the corpus exercises; the oracle (Task 6) tells you which are missing.

- [ ] **Step 1: Write the failing oracle test**

Create `test/ir/arith/test_rewrite_simplify.py`:
```python
import pytest

tvm = pytest.importorskip("tvm")
from tvm import arith as tarith, ir as tir_ir

from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.rewrite_simplify import RewriteSimplifier
from test.ir.arith._tvm_bridge import from_tvm, to_tvm

CASES = [
    Add(left=Const(value=2), right=Const(value=3)),
    Add(left=Var(name="x"), right=Const(value=0)),
    Mul(left=Var(name="x"), right=Const(value=1)),
    Mod(left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)),
    FloorDiv(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=512)),
]


@pytest.mark.parametrize("expr", CASES)
def test_simplify_matches_tvm(expr):
    ours = RewriteSimplifier().simplify(expr)
    a = tarith.Analyzer()
    tvm_simplified = from_tvm(a.simplify(to_tvm(expr)))
    assert ours == tvm_simplified, f"{ours} != {tvm_simplified}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `TVM_LIBRARY_PATH=... PYTHONPATH=...:/home/ubuntu/tvm/python python -m pytest test/ir/arith/test_rewrite_simplify.py -q`
Expected: FAIL — `ModuleNotFoundError: ...rewrite_simplify`.

- [ ] **Step 3: Implement `RewriteSimplifier`**

Create `nkigym/src/nkigym/ir/arith/rewrite_simplify.py`. Structure mirrors TVM: a recursive `simplify(expr)` that post-order rewrites each node, with a `_visit_<Node>` per type applying TVM's rules. Start with the rule families the CASES need (const-fold, identity `x+0`/`x*1`, the `(x*c + r) % c -> r` and `(x*c)//c -> x` rules from `rewrite_simplify.cc` Mod/Div visitors). Use `affine_terms` for the affine canonicalization fast path. Single return per function (build a `result` var). Reference: `rewrite_simplify.cc:415` (Add), `:566` (Sub), `:755` (Mul), `:794` (Div), `:947` (Mod). Transcribe the rules faithfully — do not invent.

- [ ] **Step 4: Run tests**

Run: `TVM_LIBRARY_PATH=... python -m pytest test/ir/arith/test_rewrite_simplify.py -q`
Expected: PASS for all CASES. If a case fails, the printed `ours != tvm` names the missing rule — read that VisitExpr_ in TVM, port it, rerun.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "arith: RewriteSimplifier (Add/Sub/Mul/FloorDiv/Mod rules) matching TVM oracle"
```

### Task 5: `RewriteSimplifier.TryCompare` — inequality proving over bound vars

**Files:**
- Modify: `nkigym/src/nkigym/ir/arith/rewrite_simplify.py`
- Test: `test/ir/arith/test_rewrite_simplify.py` (add cases)

- [ ] **Step 1: Write the failing test**

Append to `test/ir/arith/test_rewrite_simplify.py`:
```python
def test_can_prove_lt_with_bounds():
    """(i0*2 + i1) < 4  with i0 in [0,2), i1 in [0,2)  -> provable True (mirrors Split predicate elision)."""
    from nkigym.ir.arith.expr import LT
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 2)
    rs.bind("i1", 0, 2)
    pred = LT(left=Add(left=Mul(left=Var(name="i0"), right=Const(value=2)), right=Var(name="i1")), right=Const(value=4))
    assert rs.can_prove(pred) is True


def test_cannot_prove_false_bound():
    from nkigym.ir.arith.expr import LT
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 4)
    pred = LT(left=Var(name="i0"), right=Const(value=2))
    assert rs.can_prove(pred) is False
```
Cross-check each against `tvm.arith.Analyzer` with the same `bind` ranges (add an oracle assertion mirroring Task 4's pattern).

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `RewriteSimplifier` has no `bind`/`can_prove`.

- [ ] **Step 3: Implement `bind` + `TryCompare` + `can_prove`**

Mirror `rewrite_simplify.cc:179 TryCompare` and `analyzer.cc:192 CanProve`: `bind(name, lo, hi)` stores a `ConstIntBound`; `can_prove(LT(a,b))` simplifies `a - b`, computes its const-int upper bound from the bound vars, returns `upper < 0`. Port the const-int-bound propagation (`const_int_bound.cc` subset: bounds of `Add/Mul/Var/Const`). Single return.

- [ ] **Step 4: Run tests**

Expected: PASS, matching the TVM oracle assertions.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "arith: RewriteSimplifier.TryCompare + can_prove (const-int-bound) matching TVM"
```

### Task 6: `Analyzer` facade + `CanProveEqual`

**Files:**
- Create: `nkigym/src/nkigym/ir/arith/analyzer.py`
- Test: `test/ir/arith/test_analyzer.py`

- [ ] **Step 1: Write the failing oracle test**

Create `test/ir/arith/test_analyzer.py`:
```python
import pytest

tvm = pytest.importorskip("tvm")
from tvm import arith as tarith

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var
from test.ir.arith._tvm_bridge import from_tvm, to_tvm


def test_simplify_matches_tvm():
    a = Analyzer()
    a.bind("x", 0, 128)
    e = Mod(left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512))
    ours = a.simplify(e)
    ta = tarith.Analyzer()
    expected = from_tvm(ta.simplify(to_tvm(e)))
    assert ours == expected == Const(value=3)


def test_can_prove_equal():
    a = Analyzer()
    lhs = Add(left=Var(name="x"), right=Var(name="y"))
    rhs = Add(left=Var(name="y"), right=Var(name="x"))
    assert a.can_prove_equal(lhs, rhs) is True
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError: ...analyzer`.

- [ ] **Step 3: Implement `Analyzer`**

Create `nkigym/src/nkigym/ir/arith/analyzer.py`. Mirror `analyzer.cc`: holds a `RewriteSimplifier` (and later a `CanonicalSimplifier`); `simplify(expr, steps=2)` runs the simplifier(s) to fixpoint/`steps`; `bind(name, lo, hi)` forwards to the simplifier; `can_prove(pred)` forwards; `can_prove_equal(a, b)` = `can_prove(EQ(a,b))` via `simplify(Sub(a,b)) == Const(0)`. Single return per method.

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "arith: Analyzer facade (simplify/bind/can_prove/can_prove_equal) matching TVM"
```

### Task 7: `iter_map` — DetectIterMap / IterMapSimplify

**Files:**
- Create: `nkigym/src/nkigym/ir/arith/iter_map.py`
- Test: `test/ir/arith/test_iter_map.py`

This is the binding-simplifier Split/Fuse rely on (TVM `IterMapSimplifyBlockBinding`).

- [ ] **Step 1: Write the failing oracle test**

Create `test/ir/arith/test_iter_map.py`:
```python
import pytest

tvm = pytest.importorskip("tvm")

from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.iter_map import iter_map_simplify


def test_split_recombine_collapses():
    """Split: i in [0,16) -> (i0 in [0,4), i1 in [0,4)); binding i0*4+i1 simplifies back to one iter."""
    binding = Add(left=Mul(left=Var(name="i0"), right=Const(value=4)), right=Var(name="i1"))
    ranges = {"i0": (0, 4), "i1": (0, 4)}
    out = iter_map_simplify([binding], ranges)
    """Result is detected as a single affine iter of extent 16 (i0*4+i1)."""
    assert out is not None and len(out) == 1


def test_fuse_split_inverse():
    """Fuse then split-recover: fused in [0,16); (fused//4, fused%4) detected as the two original iters."""
    ranges = {"fused": (0, 16)}
    hi = FloorDiv(left=Var(name="fused"), right=Const(value=4))
    lo = Mod(left=Var(name="fused"), right=Const(value=4))
    out = iter_map_simplify([hi, lo], ranges)
    assert out is not None and len(out) == 2
```
Cross-check shape against `tvm.arith.detect_iter_map` on the same inputs (add an oracle assertion comparing the simplified bindings' structure).

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError: ...iter_map`.

- [ ] **Step 3: Implement `detect_iter_map` + `iter_map_simplify`**

Create `nkigym/src/nkigym/ir/arith/iter_map.py`. Mirror `iter_affine_map.cc`: represent each index as an `IterSumExpr` (sum of `IterSplitExpr` = `mark * scale` with `lower_factor`/`extent`), detect whether the bindings form a valid affine iter map over the input iters, and `iter_map_simplify` returns the normalized bindings (or `None` if not a valid iter map). Port the subset needed for split (recombine `i0*f + i1`) and fuse (`fused//f`, `fused%f`). Use the `Analyzer` for the const arithmetic. This is the largest single port — reference `iter_affine_map.cc:1431 DetectIterMap` and `:2153 IterMapSimplify`.

- [ ] **Step 4: Run tests**

Expected: PASS, matching the TVM oracle structure.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "arith: iter_map detect/simplify (split-recombine + fuse-inverse) matching TVM"
```

### Task 8: `IntSet` skeleton (Interval / Union / Intersect / EvalSet)

**Files:**
- Create: `nkigym/src/nkigym/ir/arith/int_set.py`
- Test: `test/ir/arith/test_int_set.py`

Minimal here (heavy use is Spec 2); enough that split/reorder/fuse don't need ad-hoc interval logic.

- [ ] **Step 1: Write the failing oracle test**

Create `test/ir/arith/test_int_set.py`:
```python
import pytest

tvm = pytest.importorskip("tvm")

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mul, Var
from nkigym.ir.arith.int_set import IntSet


def test_eval_affine_over_bound_var():
    """EvalSet(x*512 + j) with x in [0,4), j in [0,512) -> interval [0, 2047]."""
    a = Analyzer()
    a.bind("x", 0, 4)
    a.bind("j", 0, 512)
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    s = IntSet.eval(e, a)
    assert s.min_value == Const(value=0)
    assert s.max_value == Const(value=2047)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError: ...int_set`.

- [ ] **Step 3: Implement `IntSet`**

Create `nkigym/src/nkigym/ir/arith/int_set.py`. Mirror `int_set.cc`: `IntSet` = interval `[min, max]`; `eval(expr, analyzer)` walks the expr propagating intervals (`Add` adds, `Mul` by const scales, `Var` -> its bound); `union`/`intersect` over two sets. Single return per function.

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "arith: IntSet skeleton (Interval/Union/Intersect/EvalSet) matching TVM"
```

---

## Phase 2 — Split + Reorder as arith clients

### Task 9: Layer-B structural oracle harness

**Files:**
- Create: `test/transforms/_tvm_struct_oracle.py`
- Test: `test/transforms/test_tvm_struct_oracle.py`

- [ ] **Step 1: Write the failing test**

Create `test/transforms/test_tvm_struct_oracle.py`:
```python
import pytest

tvm = pytest.importorskip("tvm")
from test.transforms._tvm_struct_oracle import tvm_split_loopnest


def test_tvm_split_loopnest_shape():
    """A 16-extent loop split (4,4) -> nested (4,4) loops with binding i0*4+i1."""
    nest = tvm_split_loopnest(extent=16, factors=[4, 4])
    assert nest.extents == [4, 4]
    assert nest.binding == "i0*4 + i1"  # normalized string form
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the harness**

Create `test/transforms/_tvm_struct_oracle.py`. Build a minimal TVM `prim_func` with one loop, run `tvm.tir.Schedule(...).split(loop, factors)`, and extract a normalized `LoopNest` summary (list of extents outer→inner + the simplified binding expr as a canonical string). This gives the structural target our transforms must match. Reference: `tvm.tir.Schedule.split` (Python binding at `loop_transformation.cc:1206`).

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Layer-B TVM structural oracle harness (split loopnest extraction)"
```

### Task 10: Rewrite `Split` as an arith client (outer-trip flavour)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py`
- Test: `test/transforms/test_split.py` (existing — must stay green) + new oracle test

- [ ] **Step 1: Read the current split.py and the TVM Split**

`Read nkigym/src/nkigym/transforms/split.py`; re-read `loop_transformation.cc:396 Split`. The faithful structure: build `substitute_value = Σ var_i·Π(factor_j, j>i)` over new loop vars; `analyzer.bind` each var to `[0, factor_i)`; predicate `substitute_value < extent`, elided iff `analyzer.can_prove`; regen nested loops; `iter_map_simplify` the bindings. In our IR, the binding lives in `BlockNode.iter_values` and region `lo`s — so the affine `substitute_value` is what `normalize_block` recomputes. The arith client REPLACES the hand-rolled affine in `_do_outer_trip`/`normalize` with `analyzer`-driven construction.

- [ ] **Step 2: Add a failing oracle test**

In `test/transforms/test_split.py`, add:
```python
def test_split_matches_tvm_structure():
    import pytest
    pytest.importorskip("tvm")
    from test.transforms._tvm_struct_oracle import tvm_split_loopnest
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.transforms import Split, SplitOption
    from nkigym.ir.tree import ForNode, ISANode
    ir = build_canonical_ir()
    mm = next(n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul")
    mloop = next(a for a in ir.tree.ancestors(mm) if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d1_0")
    out = Split().apply(ir, SplitOption(target_nid=mloop, factors=(4, 4), target_axis=None))
    """The split M loop (extent 16) becomes nested (4,4) with the TVM-matching binding."""
    nest = tvm_split_loopnest(extent=16, factors=[4, 4])
    # extract our loop nest for d1 and compare extents + binding to nest
    ...
```
Fill in the extraction (walk the d1 ForNodes in `out`, collect extents, format the `iter_value`); assert equality to `nest`.

- [ ] **Step 3: Run to verify it fails or passes**

Run: `PYTHONPATH=... python -m pytest test/transforms/test_split.py -q`
Expected: the new test FAILs if our binding diverges from TVM; existing tests still PASS.

- [ ] **Step 4: Rewrite `_do_outer_trip` to use the Analyzer**

Replace the hand-rolled affine construction with `Analyzer`-driven `substitute_value` + predicate check. Keep `normalize_block` for the tree-representation reconciliation (it is our equivalent of `IterMapSimplifyBlockBinding`), but feed it the arith-constructed binding rather than re-deriving. Delete the now-dead inline affine helpers in `split.py` that the Analyzer subsumes.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=... python -m pytest test/transforms/test_split.py -q`
Expected: ALL pass (existing byte-exact + new oracle).

- [ ] **Step 6: Run the hand ladder**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python kernel_transforms.py 2>&1 | grep -c "pass=True"`
Expected: `16`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Split (outer-trip): arith-client rewrite, matches TVM structure + hand ladder"
```

### Task 11: Split tensorize flavour as arith client + predicate generality

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py`
- Test: `test/transforms/test_split.py`

- [ ] **Step 1: Add a failing test for non-divisible split predicate**

Add a test that splits a non-divisible extent (e.g. an extent-6 loop by factors (4, 2) is divisible; instead use (4,...) where product != extent is illegal — TVM emits a PREDICATE for the ragged case). Mirror TVM: assert that when factors' product exceeds extent, the generated nest carries a guard predicate (representable now via the LT node), rather than being rejected. Cross-check against `tvm_split_loopnest` with ragged factors.

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — current code rejects non-divisible via `TransformLegalityError`.

- [ ] **Step 3: Implement the predicate path in `_do_tensorize`**

Mirror TVM `Split`'s `BlockPredicateAppender`: when `not analyzer.can_prove(substitute_value < extent)`, attach the predicate to the block. Use the new `LT` node + `Analyzer.can_prove`.

- [ ] **Step 4: Run tests + hand ladder**

Run both as in Task 10 Steps 5-6. Expected: all PASS, ladder 16.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Split: predicate path for ragged factors (arith can_prove), matching TVM"
```

### Task 12: Rewrite `Reorder` as an arith client

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py`
- Test: `test/transforms/test_reorder.py`

- [ ] **Step 1: Read current reorder.py + TVM Reorder**

`Read nkigym/src/nkigym/transforms/reorder.py`; read `reorder_block_iter_var.cc`. Reorder swaps loop order and re-simplifies bindings — lightest arith (just binding re-simplification via `iter_map_simplify`).

- [ ] **Step 2: Add failing oracle test**

In `test/transforms/test_reorder.py`, add a test comparing a reorder's resulting loop order + bindings to TVM `sch.reorder` on the equivalent nest (extend `_tvm_struct_oracle.py` with `tvm_reorder_loopnest`).

- [ ] **Step 3: Run to verify it fails**

Expected: FAIL (the oracle test).

- [ ] **Step 4: Rewrite reorder to call `iter_map_simplify`**

Replace any inline binding logic with the arith call; keep the payload-swap topology change.

- [ ] **Step 5: Run tests + hand ladder**

Expected: all PASS, ladder 16.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Reorder: arith-client rewrite (iter_map_simplify), matches TVM + hand ladder"
```

---

## Phase 3 — Fuse as arith client

### Task 13: Rewrite `Fuse` as an arith client

**Files:**
- Modify: `nkigym/src/nkigym/transforms/fuse.py`
- Test: `test/transforms/test_fuse.py`

- [ ] **Step 1: Read current fuse.py + TVM Fuse**

`Read nkigym/src/nkigym/transforms/fuse.py`; re-read `loop_transformation.cc:876 Fuse`. Faithful structure: `fused_extent = Π extents`; `substitute_value[i] = floordiv(floormod(fused, lower_{i+1}), lower_i)` (and `floordiv(fused, lower)` for the outermost); substitute; regen single loop; `iter_map_simplify` proves the `//`,`%` recover the originals.

- [ ] **Step 2: Add failing oracle test**

In `test/transforms/test_fuse.py`, add a test comparing a fuse's result to TVM `sch.fuse` (extend `_tvm_struct_oracle.py` with `tvm_fuse_loopnest`): assert the single fused loop's extent and the recovered `//`/`%` bindings match.

- [ ] **Step 3: Run to verify it fails**

Expected: FAIL (the oracle test).

- [ ] **Step 4: Rewrite fuse to build TVM's substitute_value via the Analyzer**

Replace the inline fuse arithmetic with the `floordiv/floormod` construction + `iter_map_simplify`. Delete the dead inline helpers.

- [ ] **Step 5: Run tests + hand ladder**

Expected: all PASS, ladder 16.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Fuse: arith-client rewrite (floordiv/floormod + iter_map_simplify), matches TVM"
```

### Task 14: Remove dead arithmetic + full regression

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py`, `fuse.py`, `reorder.py`
- Verify: whole suite

- [ ] **Step 1: Grep for now-dead inline arithmetic in the three transforms**

Run:
```bash
grep -nE "to_affine|from_affine|_factorizations|prod\(|// |% " nkigym/src/nkigym/transforms/split.py nkigym/src/nkigym/transforms/fuse.py nkigym/src/nkigym/transforms/reorder.py
```
Identify any inline affine/division logic the Analyzer now subsumes; remove it (keep `_factorizations` — it is option enumeration, not arithmetic). Build a result var, single return.

- [ ] **Step 2: Run full pytest**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest -q`
Expected: 0 failed (>= baseline 226 + new arith/oracle tests).

- [ ] **Step 3: Run the hand ladder**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python kernel_transforms.py 2>&1 | grep -c "pass=True"`
Expected: `16`.

- [ ] **Step 4: Run the MDP example over split/fuse/reorder only**

Temporarily set `examples/matmul_lhsT_rhs.py` `transforms=[Split(), Fuse(), Reorder()]` (drop ComputeAt/ReverseComputeAt — Spec 2), run it, confirm all `[numerics] PASS`. Revert the edit after (do not commit the transform-list change).
Run: `source ~/venvs/kernel-env/bin/activate && PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py 2>&1 | tail -5`
Expected: clean exit, all PASS.

- [ ] **Step 5: Run the Layer-A oracle suite once more (regression)**

Run: `TVM_LIBRARY_PATH=/home/ubuntu/tvm/build/lib PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src:/home/ubuntu/tvm/python python -m pytest test/ir/arith -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Remove dead inline arithmetic from split/fuse/reorder; full regression green"
```

---

## Self-Review notes

- **Spec coverage:** arith package (Tasks 1-8) ✓; split/reorder/fuse as clients (Tasks 10-13) ✓; TVM oracle Layer A (Tasks 4-8 tests) + Layer B (Task 9) ✓; removals (Tasks 1, 14) ✓; expr extension (Task 2) ✓.
- **Honest scope:** compute_at/reverse untouched; `_domain_solve.py`/`interval.py` NOT deleted (Spec 2). The MDP smoke test (Task 14 Step 4) drops the two code-motion transforms because they are out of scope.
- **Type consistency:** `Analyzer.simplify/bind/can_prove/can_prove_equal`, `RewriteSimplifier.simplify/bind/can_prove`, `iter_map_simplify`, `IntSet.eval/union/intersect` used consistently across tasks.
- **Risk:** Task 7 (`iter_map`) is the largest port and the one split/fuse correctness hinges on; if it stalls, fall back to keeping `normalize_block` as the binding simplifier for the divisible cases (already proven) and scope `iter_map` to only what the oracle demands.
