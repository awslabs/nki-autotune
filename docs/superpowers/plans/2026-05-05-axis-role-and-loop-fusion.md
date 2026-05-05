# AxisRole + LoopForest IR + FuseOuterLoop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `NKIOp.BLOCKING_AXES` with a three-valued `AxisRole` classification, introduce a `LoopForest` tree IR as the analysis surface for structural kernel rewrites, re-base the renderer as a tree walker, and deliver `FuseOuterLoop` as the first `KernelRewrite` wired into a new `"tune"` stage of `nkigym_compile`.

**Architecture:** `OpGraph` → `LoopForest` → rendered NKI source. Axis roles live on each `NKIOp` class and get resolved to a per-dim `dim_role` on each `ParsedOp`. `LoopForest` is a list of trees (one per op) consumed by a generic walker that emits `for` headers and dispatches body emission to phase-keyed functions. `FuseOuterLoop` is a tree rewrite at the outermost boundary between adjacent op trees, legal only when both roots are PARALLEL on the same dim with the same trip count.

**Tech Stack:** Python 3.12, `dataclasses`, `enum.Enum`, `typing.Protocol`, `numpy`, `nki` (kernel venv at `~/venvs/kernel-env`).

**Source of truth:** `docs/superpowers/specs/2026-05-05-axis-role-and-loop-fusion-design.md`.

**Conventions followed throughout this plan:**
- Activate the venv at the start of every run: `source ~/venvs/kernel-env/bin/activate`.
- All tests live under `test/codegen/` and are executed via `python -m pytest test/codegen/<file>::<case> -v` from the repo root.
- Code style per `/home/ubuntu/.claude/rules/code_style.md`: triple-quoted docstrings only (no `#` comments except tooling directives), type hints with modern syntax (`X | None`, `list[...]`), no inline `#` comments.

---

## File Structure

**New files (created by this plan):**

| Path | Responsibility |
|---|---|
| `nkigym/src/nkigym/codegen/loop_forest.py` | `AxisRole`-aware `LoopNode` / `BodyLeaf` / `LoopForest` dataclasses; `build_canonical_forest`. |
| `nkigym/src/nkigym/rewrites/__init__.py` | `KernelRewrite` protocol, package marker. |
| `nkigym/src/nkigym/rewrites/fuse_outer_loop.py` | `FuseOuterLoop` dataclass (is_legal / apply) + `enumerate_fusion_atoms`. |
| `examples/rmsnorm_matmul_tuned.py` | End-to-end `tune`-stage demo with an explicit rewrite list. |
| `test/codegen/test_axis_role.py` | Layer-1 tests: `AxisRole` + `ParsedOp.dim_role`. |
| `test/codegen/test_loop_forest.py` | Layer-2 tests: canonical forest, invariant, fusion atoms. |
| `test/codegen/test_rewrites.py` | Layer-2 tests: `KernelRewrite` protocol + `FuseOuterLoop`. |

**Modified files:**

| Path | What changes |
|---|---|
| `nkigym/src/nkigym/ops/base.py` | Add `AxisRole` enum; replace `BLOCKING_AXES` with `AXIS_ROLES`. |
| `nkigym/src/nkigym/ops/matmul.py` | `AXIS_ROLES = {"K": AxisRole.ACCUMULATION}`; drop `BLOCKING_AXES`. |
| `nkigym/src/nkigym/ops/activation_reduce.py` | `AXIS_ROLES = {"F": AxisRole.ACCUMULATION}`; drop `BLOCKING_AXES`. |
| `nkigym/src/nkigym/ops/{activation,dma_transpose,load,store,tensor_scalar,transpose}.py` | Drop `BLOCKING_AXES`. |
| `nkigym/src/nkigym/codegen/graph.py` | Add `ParsedOp.dim_role`; populate in `_build_parsed_ops`; switch `_touched_dims` from `BLOCKING_AXES` to `AXIS_ROLES`. |
| `nkigym/src/nkigym/codegen/render.py` | Walker-based renderer; generalised slice helpers; `(op_kind, phase)`-keyed body emitters. |
| `nkigym/src/nkigym/compile.py` | Add `"tune"` stage + `rewrites` / `seed` kwargs. |
| `test/codegen/test_render.py` | Rename loop-variable assertions from `i_block_<d>` / `i_tile_<d>` to `i_<d>_0` / `i_<d>_1`. |
| `test/codegen/test_compile.py` | Add `tune`-stage integration tests. |
| `test/codegen/test_graph.py` | Add `dim_role`-field coverage. |

Each phase below ships its own commits. Git is used on the existing `dev_1` branch.

---

## Phase A — `AxisRole` + `dim_role`

Add the new enum, replace `BLOCKING_AXES` with `AXIS_ROLES`, add `dim_role` to `ParsedOp`, keep the renderer unchanged. End-state: Layer-1 tests pass; every existing test still passes.

### Task A1: Add `AxisRole` enum to `nkigym/ops/base.py`

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Test: `test/codegen/test_axis_role.py`

- [ ] **Step 1: Create test file with failing tests.**

Create `test/codegen/test_axis_role.py`:

```python
"""Layer-1 tests: AxisRole enum, NKIOp.AXIS_ROLES, ParsedOp.dim_role."""

from nkigym.ops.base import AxisRole, NKIOp


def test_axis_role_has_three_values() -> None:
    """AxisRole enumerates exactly PARALLEL, SEQUENTIAL, ACCUMULATION."""
    assert {r.name for r in AxisRole} == {"PARALLEL", "SEQUENTIAL", "ACCUMULATION"}


def test_axis_role_values_are_stable_strings() -> None:
    """AxisRole values are stable lowercase strings for readable reprs."""
    assert AxisRole.PARALLEL.value == "parallel"
    assert AxisRole.SEQUENTIAL.value == "sequential"
    assert AxisRole.ACCUMULATION.value == "accumulation"


def test_nkiop_axis_roles_defaults_to_empty() -> None:
    """NKIOp's default AXIS_ROLES is an empty dict (every axis PARALLEL)."""
    assert NKIOp.AXIS_ROLES == {}
```

- [ ] **Step 2: Run the test — expect failure.**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: ImportError on `AxisRole` (not yet defined).

- [ ] **Step 3: Add the `AxisRole` enum and `NKIOp.AXIS_ROLES` attribute.**

Edit `nkigym/src/nkigym/ops/base.py` — add the enum near the top of the file (after the existing `_DEFAULT_OUTPUT_ROLE = "sbuf"` line), and add `AXIS_ROLES` as a new ClassVar on `NKIOp`.

In the imports near the top of `base.py` (currently starts with `import functools`), add `from enum import Enum` after the existing `import functools` line.

Add after the `_DEFAULT_OUTPUT_ROLE = "sbuf"` line:

```python


class AxisRole(str, Enum):
    """Per-op classification of how a loop axis carries state across iterations.

    PARALLEL iterations are independent and safe to fuse with another
    op's PARALLEL loop on the same dim. SEQUENTIAL iterations carry
    non-associative state (prefix scan, running state) and must not
    fuse with a PARALLEL loop on the same dim. ACCUMULATION iterations
    contribute to an associative reducer (sum, max); the accumulator
    is live across iterations, so fusion with another nest's PARALLEL
    loop on the same dim is illegal.
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ACCUMULATION = "accumulation"
```

On the `NKIOp` class, add `AXIS_ROLES` as a new ClassVar alongside the existing `BLOCKING_AXES`. Do **not** delete `BLOCKING_AXES` yet — subsequent tasks remove it per-subclass after their own tests pass.

In the `NKIOp` class body (around line 104 where `BLOCKING_AXES` is declared) add:

```python
    AXIS_ROLES: ClassVar[dict[str, "AxisRole"]] = {}
```

And update the class docstring `Attributes:` block to describe `AXIS_ROLES` alongside `BLOCKING_AXES`. Add this line in the Attributes block:

```
    AXIS_ROLES: Per-op axis → role classification. Omitted axes default
        to ``AxisRole.PARALLEL``. Replaces ``BLOCKING_AXES`` — a
        ``BLOCKING_AXES`` entry corresponds to an
        ``AxisRole.ACCUMULATION`` entry.
```

- [ ] **Step 4: Run the test — expect pass.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run the full existing test suite — expect all still pass.**

```bash
python -m pytest test/codegen -v
```

Expected: all tests pass (no regressions; `BLOCKING_AXES` still works because we haven't removed it).

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/ops/base.py test/codegen/test_axis_role.py
git commit -m "Add AxisRole enum + NKIOp.AXIS_ROLES class attribute"
```

---

### Task A2: Set `AXIS_ROLES` on `NKIMatmul` and `NKIActivationReduce`

**Files:**
- Modify: `nkigym/src/nkigym/ops/matmul.py`
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`
- Modify: `test/codegen/test_axis_role.py`

- [ ] **Step 1: Extend the test file with AXIS_ROLES assertions for these two ops.**

Append to `test/codegen/test_axis_role.py`:

```python
def test_matmul_axis_roles_marks_k_as_accumulation() -> None:
    """NKIMatmul's K axis is the accumulation axis; M and N default to PARALLEL."""
    from nkigym.ops.matmul import NKIMatmul

    assert NKIMatmul.AXIS_ROLES == {"K": AxisRole.ACCUMULATION}


def test_activation_reduce_axis_roles_marks_f_as_accumulation() -> None:
    """NKIActivationReduce's F axis is the accumulation axis; P defaults to PARALLEL."""
    from nkigym.ops.activation_reduce import NKIActivationReduce

    assert NKIActivationReduce.AXIS_ROLES == {"F": AxisRole.ACCUMULATION}
```

- [ ] **Step 2: Run — expect failure.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: two new tests fail because `AXIS_ROLES` is the base-class `{}` for both ops.

- [ ] **Step 3: Set `AXIS_ROLES` on `NKIMatmul`.**

Edit `nkigym/src/nkigym/ops/matmul.py` — add an import and a new ClassVar on `NKIMatmul`.

Add `AxisRole` to the import on line 13:

```python
from nkigym.ops.base import AxisRole, NKIOp
```

Add `AXIS_ROLES` just below the existing `BLOCKING_AXES` line (~line 24):

```python
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
```

Do **not** delete `BLOCKING_AXES` yet.

- [ ] **Step 4: Set `AXIS_ROLES` on `NKIActivationReduce`.**

Edit `nkigym/src/nkigym/ops/activation_reduce.py` — add an import and a new ClassVar.

Add `AxisRole` to the import on line 19:

```python
from nkigym.ops.base import AxisRole, NKIOp
```

Add `AXIS_ROLES` just below the existing `BLOCKING_AXES` line (~line 86):

```python
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
```

- [ ] **Step 5: Run the axis-role tests — expect pass.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Run the full existing test suite — expect all still pass.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions.

- [ ] **Step 7: Commit.**

```bash
git add nkigym/src/nkigym/ops/matmul.py nkigym/src/nkigym/ops/activation_reduce.py test/codegen/test_axis_role.py
git commit -m "Set AXIS_ROLES on NKIMatmul + NKIActivationReduce"
```

---

### Task A3: Add `ParsedOp.dim_role` field

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Modify: `test/codegen/test_axis_role.py`

- [ ] **Step 1: Add a test for `dim_role` population.**

Append to `test/codegen/test_axis_role.py`:

```python
def test_parsed_op_has_dim_role_for_every_touched_dim() -> None:
    """ParsedOp.dim_role has an entry per touched_dim with the op's role."""
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
    k_dim = matmul_op.axis_map["K"]
    m_dim = matmul_op.axis_map["M"]
    n_dim = matmul_op.axis_map["N"]
    assert matmul_op.dim_role[k_dim] == AxisRole.ACCUMULATION
    assert matmul_op.dim_role[m_dim] == AxisRole.PARALLEL
    assert matmul_op.dim_role[n_dim] == AxisRole.PARALLEL
    assert set(matmul_op.dim_role.keys()) == set(matmul_op.touched_dims)


def test_same_concrete_dim_can_carry_different_roles_across_ops() -> None:
    """In rmsnorm+matmul, d1 is ACCUMULATION in activation_reduce and PARALLEL in tensor_scalar."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar

    @nkigym_kernel
    def kernel(lhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rms_inv = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        out = NKIStore()(data=lhs_rms)
        return out

    specs = {"lhs": ((128, 256), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    ar = next(op for op in g.ops if op.op_cls.__name__ == "NKIActivationReduce")
    ts = next(op for op in g.ops if op.op_cls.__name__ == "NKITensorScalar")
    f_dim = ar.axis_map["F"]
    assert ar.dim_role[f_dim] == AxisRole.ACCUMULATION
    assert ts.dim_role[f_dim] == AxisRole.PARALLEL
```

- [ ] **Step 2: Run — expect failure (`dim_role` field doesn't exist yet).**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: the two new tests fail with `AttributeError: ... has no attribute 'dim_role'`.

- [ ] **Step 3: Add `dim_role` field to `ParsedOp` and populate it.**

Edit `nkigym/src/nkigym/codegen/graph.py`.

At the top of the file, update the import line `from nkigym.ops.base import NKIOp` to also pull in `AxisRole`:

```python
from nkigym.ops.base import AxisRole, NKIOp
```

Extend the `ParsedOp` dataclass (around lines 62-86) — add `dim_role` as the final field with a docstring entry. The docstring already has an Attributes block; append:

```
        dim_role: Concrete ``dim_id`` → :class:`AxisRole` for every dim
            in ``touched_dims``. Resolved from ``op_cls.AXIS_ROLES`` via
            ``axis_map``; PARALLEL is the default for any dim not named
            in ``AXIS_ROLES``.
```

After the existing `touched_dims: tuple[str, ...]` line, add:

```python
    dim_role: dict[str, AxisRole]
```

In `_build_parsed_ops` (around line 466), add logic to compute `dim_role` and pass it to `ParsedOp`:

Replace the body of the function with:

```python
    """Assemble per-op records with canonicalised ``touched_dims``."""
    ops: list[ParsedOp] = []
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        dim_role = _resolve_dim_role(raw.op_cls, axis_map, touched)
        ops.append(
            ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
                dim_role=dim_role,
            )
        )
    return ops


def _resolve_dim_role(
    op_cls: type[NKIOp], axis_map: dict[str, str], touched: tuple[str, ...]
) -> dict[str, AxisRole]:
    """Map every ``dim_id`` in ``touched`` to the op's role for that dim."""
    abstract_role = getattr(op_cls, "AXIS_ROLES", {})
    concrete: dict[str, AxisRole] = {}
    for abstract, dim_id in axis_map.items():
        if dim_id in touched and abstract in abstract_role:
            concrete[dim_id] = abstract_role[abstract]
    for dim_id in touched:
        concrete.setdefault(dim_id, AxisRole.PARALLEL)
    return concrete
```

- [ ] **Step 4: Run the axis-role tests — expect pass.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Run the full existing test suite — expect all still pass.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions. `test_graph.py` may have one or two tests that construct `ParsedOp` or compare by dataclass equality; those need a `dim_role` entry added too.

If `test_graph.py` fails, fix any `ParsedOp(...)` constructor calls in the tests by adding a `dim_role={}` kwarg (or the expected mapping if the test is about a real op). Inspect the failure output — typically `TypeError: ParsedOp.__init__() missing 1 required positional argument: 'dim_role'`.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_axis_role.py test/codegen/test_graph.py
git commit -m "Add ParsedOp.dim_role populated from AXIS_ROLES"
```

---

### Task A4: Switch `_touched_dims` from `BLOCKING_AXES` to `AXIS_ROLES`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`

- [ ] **Step 1: Verify existing behaviour of `_touched_dims` through a checkpoint test.**

Before changing the implementation, confirm today's `touched_dims` output is correct on the canonical matmul and activation_reduce cases. The existing `test_graph.py::test_parse_and_resolve_touched_dims_ordering` test does exactly this. Re-run it to establish the baseline:

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_graph.py::test_parse_and_resolve_touched_dims_ordering -v
```

Expected: pass.

- [ ] **Step 2: Change the implementation.**

Edit `nkigym/src/nkigym/codegen/graph.py`, `_touched_dims` (around lines 487-507).

Replace:

```python
    blocking = [axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map]
    for d in blocking:
        if d not in ordered:
            ordered.append(d)
```

with:

```python
    non_parallel_axes = getattr(op_cls, "AXIS_ROLES", {}).keys()
    for abstract in non_parallel_axes:
        if abstract in axis_map and axis_map[abstract] not in ordered:
            ordered.append(axis_map[abstract])
```

Do **not** remove `BLOCKING_AXES` from the op classes yet — it stays as an unused leftover until Task A5.

- [ ] **Step 3: Run the existing suite — expect all pass.**

```bash
python -m pytest test/codegen -v
```

Expected: `test_parse_and_resolve_touched_dims_ordering` still passes, `test_render.py`'s rmsnorm+matmul end-to-end still passes, everything else still passes.

- [ ] **Step 4: Commit.**

```bash
git add nkigym/src/nkigym/codegen/graph.py
git commit -m "Derive touched_dims' non-parallel axes from AXIS_ROLES"
```

---

### Task A5: Delete `BLOCKING_AXES` from every op class

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py`
- Modify: `nkigym/src/nkigym/ops/matmul.py`
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`
- Modify: `nkigym/src/nkigym/ops/{activation,dma_transpose,load,store,tensor_scalar,transpose}.py`
- Modify: `test/codegen/test_axis_role.py`

- [ ] **Step 1: Add migration-completeness tests.**

Append to `test/codegen/test_axis_role.py`:

```python
def test_blocking_axes_removed_from_base_nkiop() -> None:
    """Migration complete: BLOCKING_AXES must not exist on NKIOp."""
    assert not hasattr(NKIOp, "BLOCKING_AXES"), (
        "NKIOp.BLOCKING_AXES should be deleted — use AXIS_ROLES."
    )


def test_blocking_axes_removed_from_every_op_subclass() -> None:
    """No NKIOp subclass may carry a BLOCKING_AXES attribute."""
    from nkigym.ops.activation import NKIActivation
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.dma_transpose import NKIDMATranspose
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose

    for cls in (
        NKIActivation,
        NKIActivationReduce,
        NKIDMATranspose,
        NKILoad,
        NKIMatmul,
        NKIStore,
        NKITensorScalar,
        NKITranspose,
    ):
        assert "BLOCKING_AXES" not in cls.__dict__, (
            f"{cls.__name__} still declares BLOCKING_AXES; delete it."
        )
```

- [ ] **Step 2: Run — expect failures on every op that still has `BLOCKING_AXES`.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: both new tests fail.

- [ ] **Step 3: Remove `BLOCKING_AXES` from each file.**

In `nkigym/src/nkigym/ops/base.py`:
- Delete the line `BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()` on the `NKIOp` class.
- Remove the `BLOCKING_AXES` entry from the class docstring's `Attributes` block.

In `nkigym/src/nkigym/ops/matmul.py`:
- Delete the line `BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})`.

In `nkigym/src/nkigym/ops/activation_reduce.py`:
- Delete the line `BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"F"})` and its triple-quoted docstring comment below it (the one explaining the F reduction).

For the triple-quoted docstring, replace it — instead of living as a standalone string after `BLOCKING_AXES`, attach the same explanation to `AXIS_ROLES` so the reasoning about F being a reduction axis stays in the source. Keep the existing `AXIS_ROLES = {"F": AxisRole.ACCUMULATION}` line, and add a docstring immediately above it on a separate line:

```python
    """The F axis is a reduction axis — the op iterates over all F tiles
    before its output is complete. Render emits an F-loop memset prologue
    (on the reduce accumulator) and places downstream consumers outside
    this F-loop, symmetric to how matmul's K dim is handled."""
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
```

Actually class-level docstrings don't attach cleanly to individual ClassVars — instead, use a regular triple-quoted "module note" string as a sibling line. Python treats standalone triple-quoted strings in class bodies as valid but no-op. Leave the existing explanatory triple-quoted string in place (around line 89 in `activation_reduce.py`), just remove the `BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"F"})` line on line 86.

In `nkigym/src/nkigym/ops/activation.py`, `nkigym/src/nkigym/ops/dma_transpose.py`, `nkigym/src/nkigym/ops/load.py`, `nkigym/src/nkigym/ops/store.py`, `nkigym/src/nkigym/ops/tensor_scalar.py`, `nkigym/src/nkigym/ops/transpose.py`:
- Delete the line `BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()` from each. Also delete any adjacent triple-quoted docstring that only explained `BLOCKING_AXES`.

- [ ] **Step 4: Run the axis-role tests — expect pass.**

```bash
python -m pytest test/codegen/test_axis_role.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Run the full existing test suite — expect all still pass.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions. The renderer still works because `_touched_dims` was switched to `AXIS_ROLES` in Task A4 before this deletion.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/ops/*.py test/codegen/test_axis_role.py
git commit -m "Delete BLOCKING_AXES from NKIOp and every subclass"
```

---

## Phase B — `LoopForest` IR + `build_canonical_forest`

Introduce the tree IR. Renderer stays on today's per-op emitters; canonical forest is built but not yet consumed.

### Task B1: Create `loop_forest.py` with dataclasses and invariant helper

**Files:**
- Create: `nkigym/src/nkigym/codegen/loop_forest.py`
- Create: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write tests for the dataclasses and a helper to check the per-dim product invariant.**

Create `test/codegen/test_loop_forest.py`:

```python
"""Layer-2 tests: LoopForest IR, canonical forest, invariant."""

from nkigym.codegen.loop_forest import (
    BodyLeaf,
    LoopForest,
    LoopNode,
    check_invariant,
)
from nkigym.ops.base import AxisRole


def test_body_leaf_defaults_phase_to_main() -> None:
    """BodyLeaf defaults to phase='main' for single-phase ops."""
    leaf = BodyLeaf(op_idx=0)
    assert leaf.phase == "main"


def test_body_leaf_accepts_explicit_phase() -> None:
    """Multi-phase ops name their phase explicitly."""
    leaf = BodyLeaf(op_idx=3, phase="psum_init")
    assert leaf.phase == "psum_init"


def test_loop_node_stores_dim_trip_role_and_children() -> None:
    """LoopNode exposes dim_id, trip_count, role, and children."""
    node = LoopNode(
        dim_id="d0",
        trip_count=16,
        role=AxisRole.PARALLEL,
        children=[BodyLeaf(op_idx=0)],
    )
    assert node.dim_id == "d0"
    assert node.trip_count == 16
    assert node.role is AxisRole.PARALLEL
    assert len(node.children) == 1


def test_check_invariant_passes_on_simple_2n_shape() -> None:
    """A 2-level block+tile chain (16 × 1 = 16) satisfies the invariant."""
    tree: LoopNode = LoopNode(
        "d0", 16, AxisRole.PARALLEL,
        [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])],
    )
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16}
    op_touched = {0: ("d0",)}
    check_invariant(forest, num_tiles, op_touched)


def test_check_invariant_raises_on_product_mismatch() -> None:
    """A chain where product of same-dim trips != num_tiles(d) fails the invariant."""
    tree: LoopNode = LoopNode(
        "d0", 8, AxisRole.PARALLEL,
        [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])],
    )
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16}
    op_touched = {0: ("d0",)}
    try:
        check_invariant(forest, num_tiles, op_touched)
    except ValueError as exc:
        assert "d0" in str(exc)
    else:
        raise AssertionError("check_invariant did not raise on product mismatch")


def test_check_invariant_raises_when_dim_missing_from_ancestors() -> None:
    """A BodyLeaf whose op references a dim with no ancestor LoopNodes fails."""
    tree: LoopNode = LoopNode(
        "d0", 16, AxisRole.PARALLEL,
        [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])],
    )
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16, "d1": 4}
    op_touched = {0: ("d0", "d1")}
    try:
        check_invariant(forest, num_tiles, op_touched)
    except ValueError as exc:
        assert "d1" in str(exc)
    else:
        raise AssertionError("check_invariant did not raise on missing dim")
```

- [ ] **Step 2: Run — expect ImportError.**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: ImportError, module not yet created.

- [ ] **Step 3: Create `nkigym/src/nkigym/codegen/loop_forest.py`.**

Create the file:

```python
"""``LoopForest`` IR — tree-level analysis surface for structural kernel rewrites.

Each `OpGraph` op lowers to one `LoopForest` tree. A tree's interior
`LoopNode`s describe the loop structure emitted at render time; its
leaves are `BodyLeaf` markers that name which op (and which phase of
that op, for multi-phase ops like matmul) runs at that tree position.
Transforms such as :class:`FuseOuterLoop` operate on the forest as data
rather than on source text.
"""

from dataclasses import dataclass, field
from typing import Union

from nkigym.ops.base import AxisRole


@dataclass
class BodyLeaf:
    """Marks where an op (or an op phase) runs inside a loop nest.

    Attributes:
        op_idx: Index into ``OpGraph.ops`` of the op this leaf represents.
        phase: Phase name for multi-phase ops. Single-phase ops use the
            default ``"main"``. Matmul phases: ``"psum_init"``,
            ``"compute"``, ``"drain"``. ActivationReduce phases:
            ``"reducer_init"``, ``"reduce_step"``, ``"post_op"``.
    """

    op_idx: int
    phase: str = "main"


@dataclass
class LoopNode:
    """A single loop at one tree depth.

    Attributes:
        dim_id: Concrete ``OpGraph.dims`` key this loop iterates.
        trip_count: Iteration count (``num_tiles(d)`` for a "block" tier,
            ``1`` for a "tile" tier in the 2N-per-dim canonical form;
            any divisor of ``num_tiles(d)`` under structural transforms).
        role: ``AxisRole`` for this op's use of ``dim_id``. After
            fusion the merged ``LoopNode``'s role is ``PARALLEL`` by
            construction (fusion requires both sides PARALLEL).
        children: Nested ``LoopNode``s and/or terminal ``BodyLeaf``s, in
            emission order.
    """

    dim_id: str
    trip_count: int
    role: AxisRole
    children: list[Union["LoopNode", BodyLeaf]] = field(default_factory=list)


LoopForest = list[Union[LoopNode, BodyLeaf]]
"""A list of root-level entries, one per op, in program order."""


def check_invariant(
    forest: LoopForest, num_tiles: dict[str, int], op_touched: dict[int, tuple[str, ...]]
) -> None:
    """Validate the per-dim product invariant on every ``BodyLeaf``.

    Walks the forest from each root; at every ``BodyLeaf`` verifies that
    for each ``dim_id d`` in the leaf's op's ``touched_dims``, the
    product of ancestor ``LoopNode.trip_count`` where ``dim_id == d``
    equals ``num_tiles[d]``.

    Raises:
        ValueError: A body leaf violates the invariant; the message
            names the offending dim and op index.
    """
    for entry in forest:
        _check_node(entry, ancestor_trips={}, num_tiles=num_tiles, op_touched=op_touched)


def _check_node(
    node: Union[LoopNode, BodyLeaf],
    ancestor_trips: dict[str, list[int]],
    num_tiles: dict[str, int],
    op_touched: dict[int, tuple[str, ...]],
) -> None:
    """Recursive helper for :func:`check_invariant`."""
    if isinstance(node, BodyLeaf):
        touched = op_touched.get(node.op_idx, ())
        for d in touched:
            trips = ancestor_trips.get(d, [])
            if not trips:
                raise ValueError(
                    f"BodyLeaf(op_idx={node.op_idx}, phase={node.phase!r}) references dim "
                    f"{d!r} but no ancestor LoopNode iterates it"
                )
            product = 1
            for t in trips:
                product *= t
            expected = num_tiles[d]
            if product != expected:
                raise ValueError(
                    f"BodyLeaf(op_idx={node.op_idx}, phase={node.phase!r}) dim {d!r}: "
                    f"ancestor trip product {product} != num_tiles {expected}"
                )
        return
    ancestor_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _check_node(child, ancestor_trips, num_tiles, op_touched)
    ancestor_trips[node.dim_id].pop()
```

- [ ] **Step 4: Run the loop-forest tests — expect pass.**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Run the full suite — confirm nothing broke.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Add LoopForest IR dataclasses with invariant checker"
```

---

### Task B2: Implement `build_canonical_forest` for single-phase ops

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Modify: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Add tests.**

Append to `test/codegen/test_loop_forest.py`:

```python
def _parse(kernel, specs):
    from nkigym.codegen.graph import parse_and_resolve

    return parse_and_resolve(kernel, specs)


def test_canonical_forest_load_kernel_shape() -> None:
    """A 2D NKILoad op produces a 4-deep chain: d0 block / d0 tile / d1 block / d1 tile / BodyLeaf."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(x):
        y = NKILoad()(data=x)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    """Two roots — one per op."""
    assert len(forest) == 2
    load_tree = forest[0]
    assert isinstance(load_tree, LoopNode)
    assert load_tree.dim_id == "d0"
    assert load_tree.trip_count == g.dims["d0"].num_tiles
    assert load_tree.role is AxisRole.PARALLEL
    inner_d0 = load_tree.children[0]
    assert isinstance(inner_d0, LoopNode)
    assert inner_d0.dim_id == "d0"
    assert inner_d0.trip_count == 1
    outer_d1 = inner_d0.children[0]
    assert isinstance(outer_d1, LoopNode)
    assert outer_d1.dim_id == "d1"
    assert outer_d1.trip_count == g.dims["d1"].num_tiles
    inner_d1 = outer_d1.children[0]
    assert isinstance(inner_d1, LoopNode)
    assert inner_d1.dim_id == "d1"
    assert inner_d1.trip_count == 1
    leaf = inner_d1.children[0]
    assert isinstance(leaf, BodyLeaf)
    assert leaf.op_idx == 0
    assert leaf.phase == "main"


def test_canonical_forest_invariant_holds_on_load_store() -> None:
    """check_invariant passes on the canonical forest of a load+store kernel."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(x):
        y = NKILoad()(data=x)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    check_invariant(forest, num_tiles, op_touched)
```

- [ ] **Step 2: Run — expect failure (`build_canonical_forest` not defined).**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `build_canonical_forest` for single-phase ops.**

Edit `nkigym/src/nkigym/codegen/loop_forest.py`. Add at the top:

```python
from nkigym.codegen.graph import OpGraph, ParsedOp
```

(If there's a circular import risk — the `graph.py` → `loop_forest.py` direction doesn't yet exist, so this import is safe.)

Append to the module:

```python
def build_canonical_forest(op_graph: "OpGraph") -> LoopForest:
    """Produce the canonical forest — one tree per op, in program order.

    Each op's tree is a 2N-deep chain over its ``touched_dims``: for dim
    ``d_k`` we emit a ``LoopNode(d_k, num_tiles(d_k))`` followed by a
    nested ``LoopNode(d_k, 1)`` ("block" tier then "tile" tier). At the
    deepest point, multi-phase ops place phase leaves per op-class
    rules; single-phase ops place one ``BodyLeaf(op_idx, "main")``.
    """
    return [_build_tree(op, op_graph) for op in op_graph.ops]


def _build_tree(op: "ParsedOp", op_graph: "OpGraph") -> LoopNode:
    """Build the 2N-per-dim chain for ``op`` with phase leaves at the tip."""
    deepest_children = _build_leaves(op, op_graph)
    return _wrap_dims(op.touched_dims, op, op_graph, deepest_children)


def _wrap_dims(
    dims: tuple[str, ...],
    op: "ParsedOp",
    op_graph: "OpGraph",
    inner_children: list[Union[LoopNode, BodyLeaf]],
) -> LoopNode:
    """Wrap ``inner_children`` in a 2N-per-dim chain over ``dims``."""
    if not dims:
        raise ValueError(f"Op {op.idx}: cannot build tree — no touched_dims")
    node_children: list[Union[LoopNode, BodyLeaf]] = inner_children
    for d in reversed(dims):
        role = op.dim_role[d]
        num_t = op_graph.dims[d].num_tiles
        tile_node = LoopNode(dim_id=d, trip_count=1, role=role, children=node_children)
        block_node = LoopNode(dim_id=d, trip_count=num_t, role=role, children=[tile_node])
        node_children = [block_node]
    head = node_children[0]
    assert isinstance(head, LoopNode)
    return head


def _build_leaves(op: "ParsedOp", op_graph: "OpGraph") -> list[Union[LoopNode, BodyLeaf]]:
    """Return the deepest-point children list for ``op``'s tree.

    Dispatch on op-class name — single-phase ops return a single
    ``[BodyLeaf(op.idx, "main")]``; multi-phase ops (matmul,
    activation_reduce) receive custom builders added in later tasks.
    """
    builder = _LEAF_BUILDERS.get(op.op_cls.__name__, _build_leaves_default)
    return builder(op, op_graph)


def _build_leaves_default(op: "ParsedOp", op_graph: "OpGraph") -> list[Union[LoopNode, BodyLeaf]]:
    """Single-phase default: one BodyLeaf(op_idx, 'main')."""
    _ = op_graph
    return [BodyLeaf(op_idx=op.idx, phase="main")]


_LEAF_BUILDERS: dict = {}
"""Populated with multi-phase builders by later tasks (matmul, activation_reduce)."""
```

- [ ] **Step 4: Run the loop-forest tests — expect pass.**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Build canonical forest for single-phase ops"
```

---

### Task B3: Add multi-phase leaf builder for `NKIMatmul`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Modify: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Add a test for matmul's tree shape.**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_canonical_forest_matmul_has_three_phase_leaves() -> None:
    """Matmul's innermost N-tile node contains psum_init, K chain ending in compute, drain."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    """Drill in: M-block, M-tile, N-block, N-tile."""
    n_tile = tree.children[0].children[0].children[0]
    children = n_tile.children
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "psum_init"
    """Children[1] is the K chain (block -> tile -> compute leaf)."""
    k_block = children[1]
    assert isinstance(k_block, LoopNode) and k_block.trip_count > 1
    k_tile = k_block.children[0]
    compute_leaf = k_tile.children[0]
    assert isinstance(compute_leaf, BodyLeaf) and compute_leaf.phase == "compute"
    """Children[2] is the drain leaf at the same depth as psum_init."""
    assert isinstance(children[2], BodyLeaf) and children[2].phase == "drain"


def test_canonical_forest_matmul_invariant_holds() -> None:
    """Matmul's canonical forest satisfies the per-dim product invariant."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    check_invariant(forest, num_tiles, op_touched)
```

- [ ] **Step 2: Run — expect the matmul shape test to fail (wrong children layout).**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: `test_canonical_forest_matmul_has_three_phase_leaves` fails.

- [ ] **Step 3: Implement the matmul leaf builder.**

Edit `nkigym/src/nkigym/codegen/loop_forest.py`. Add a function and register it on `_LEAF_BUILDERS`:

```python
def _build_leaves_matmul(op: "ParsedOp", op_graph: "OpGraph") -> list[Union[LoopNode, BodyLeaf]]:
    """Matmul: outer M/N chain surrounds a K chain plus psum_init + drain leaves.

    The outer dims ``M`` and ``N`` are consumed by the surrounding
    ``_wrap_dims`` call. Here we build the innermost children list for
    the N-tile node: ``[psum_init, <K chain ending in compute>, drain]``.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = op_graph.dims[k_dim].num_tiles
    compute_leaf = BodyLeaf(op_idx=op.idx, phase="compute")
    k_tile = LoopNode(dim_id=k_dim, trip_count=1, role=k_role, children=[compute_leaf])
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[k_tile])
    return [
        BodyLeaf(op_idx=op.idx, phase="psum_init"),
        k_block,
        BodyLeaf(op_idx=op.idx, phase="drain"),
    ]


_LEAF_BUILDERS["NKIMatmul"] = _build_leaves_matmul
```

Matmul's `touched_dims` places K last; `_wrap_dims` currently wraps every dim. Because this builder handles K internally, we must skip K when wrapping. Update `_build_tree` to skip dims handled by a custom builder:

Replace `_build_tree` with:

```python
def _build_tree(op: "ParsedOp", op_graph: "OpGraph") -> LoopNode:
    """Build the 2N-per-dim chain for ``op`` with phase leaves at the tip."""
    deepest_children = _build_leaves(op, op_graph)
    wrap_dims = _dims_to_wrap(op)
    return _wrap_dims(wrap_dims, op, op_graph, deepest_children)


def _dims_to_wrap(op: "ParsedOp") -> tuple[str, ...]:
    """Return the dims the outer wrapper should build around the leaves.

    Multi-phase builders may handle some interior dims themselves (e.g.
    matmul builds K internally). For those builders, the dims they
    consume are dropped from the outer wrap.
    """
    skip = _BUILDER_INTERIOR_DIMS.get(op.op_cls.__name__, lambda _op: ())(op)
    return tuple(d for d in op.touched_dims if d not in skip)


_BUILDER_INTERIOR_DIMS: dict = {"NKIMatmul": lambda op: {op.axis_map["K"]}}
"""Builders that handle some dims internally must register them here."""
```

- [ ] **Step 4: Run the loop-forest tests — expect pass.**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: all matmul tests pass.

- [ ] **Step 5: Run the full suite — expect no regressions.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Build canonical forest for NKIMatmul with phase leaves"
```

---

### Task B4: Add multi-phase leaf builder for `NKIActivationReduce`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Modify: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Add tests for activation_reduce's tree shape (with and without post_op).**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_canonical_forest_activation_reduce_with_post_op_has_three_leaves() -> None:
    """ActivationReduce with post_op emits reducer_init + F chain ending in reduce_step + post_op."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=xs)
        out = NKIStore()(data=m)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Three children: reducer_init, F chain, post_op."""
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "reducer_init"
    f_block = children[1]
    assert isinstance(f_block, LoopNode)
    f_tile = f_block.children[0]
    reduce_leaf = f_tile.children[0]
    assert isinstance(reduce_leaf, BodyLeaf) and reduce_leaf.phase == "reduce_step"
    assert isinstance(children[2], BodyLeaf) and children[2].phase == "post_op"


def test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf() -> None:
    """When the op has no post_op, the post_op leaf is omitted."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
        out = NKIStore()(data=m)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Two children: reducer_init, F chain. No post_op."""
    assert len(children) == 2
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "reducer_init"
    assert isinstance(children[1], LoopNode)
    for c in children:
        if isinstance(c, BodyLeaf):
            assert c.phase != "post_op"


def test_canonical_forest_rmsnorm_matmul_invariant_holds() -> None:
    """The full rmsnorm+matmul canonical forest satisfies the invariant."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose

    @nkigym_kernel
    def k(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=lhs_sbuf)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = _parse(k, specs)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    check_invariant(forest, num_tiles, op_touched)
```

- [ ] **Step 2: Run — expect the three new tests to fail.**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: three new tests fail.

- [ ] **Step 3: Implement the activation_reduce leaf builder.**

Edit `nkigym/src/nkigym/codegen/loop_forest.py`. Add:

```python
def _build_leaves_activation_reduce(
    op: "ParsedOp", op_graph: "OpGraph"
) -> list[Union[LoopNode, BodyLeaf]]:
    """ActivationReduce: reducer_init + F chain ending in reduce_step + optional post_op.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: a ``F block / F tile / BodyLeaf(reduce_step)`` chain sits
    between ``reducer_init`` and the optional ``post_op`` leaf.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf])
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile])
    leaves: list[Union[LoopNode, BodyLeaf]] = [
        BodyLeaf(op_idx=op.idx, phase="reducer_init"),
        f_block,
    ]
    if op.op_kwargs.get("post_op") is not None:
        leaves.append(BodyLeaf(op_idx=op.idx, phase="post_op"))
    return leaves


_LEAF_BUILDERS["NKIActivationReduce"] = _build_leaves_activation_reduce
_BUILDER_INTERIOR_DIMS["NKIActivationReduce"] = lambda op: {op.axis_map["F"]}
```

- [ ] **Step 4: Run the loop-forest tests — expect pass.**

```bash
python -m pytest test/codegen/test_loop_forest.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full suite — expect no regressions.**

```bash
python -m pytest test/codegen -v
```

Expected: no regressions.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Build canonical forest for NKIActivationReduce with optional post_op"
```

---

## Phase C — Walker-based renderer

Replace per-op emitters with a generic `LoopForest` walker + phase-keyed body emitters. The emitted source becomes byte-identical to today modulo loop-variable renaming (`i_block_<d>` → `i_<d>_0`, `i_tile_<d>` → `i_<d>_1`).

### Task C1: Generalise slice helpers to accept `path_ordinals` and `path_trips`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Update existing slice-helper tests to use the new signature.**

The existing tests `test_sbuf_tile_slice_2d`, `test_sbuf_tile_slice_1d`, and `test_hbm_tile_slice` in `test/codegen/test_render.py` today pass `p_tile` and `f_tile` args and expect the string forms `"i_block_<d> + i_tile_<d>"`. Replace them with versions that pass the generalised `path_ordinals` + `path_trips` dicts and assert the new loop-variable naming.

Edit `test/codegen/test_render.py`, replace the three existing tests:

```python
def test_sbuf_tile_slice_2d() -> None:
    """Per-tile SBUF slice collapses (i_d0_0 + i_d0_1) / (i_d1_0 + i_d1_1) offsets."""
    from nkigym.codegen.render import _sbuf_tile_slice

    ordinals = {"d0": 2, "d1": 2}
    trips = {"d0": [16, 1], "d1": [8, 1]}
    slice_expr = _sbuf_tile_slice("sbuf_lhs", ("d0", "d1"), p_tile=128, f_tile=128,
                                  path_ordinals=ordinals, path_trips=trips)
    assert slice_expr == (
        "sbuf_lhs[0:128, i_d0_0 + i_d0_1, "
        "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )


def test_sbuf_tile_slice_1d() -> None:
    """1D slice uses the partition-axis path ordinals only."""
    from nkigym.codegen.render import _sbuf_tile_slice

    ordinals = {"d0": 2}
    trips = {"d0": [4, 1]}
    slice_expr = _sbuf_tile_slice("sbuf_rms", ("d0",), p_tile=128, f_tile=1,
                                  path_ordinals=ordinals, path_trips=trips)
    assert slice_expr == "sbuf_rms[0:128, i_d0_0 + i_d0_1, 0:1]"


def test_hbm_tile_slice() -> None:
    """HBM slice parenthesises compound + expressions before multiplying."""
    from nkigym.codegen.render import _hbm_tile_slice

    ordinals = {"d0": 2, "d1": 2}
    trips = {"d0": [16, 1], "d1": [16, 1]}
    slice_expr = _hbm_tile_slice("lhs", ("d0", "d1"), p_tile=128, f_tile=128,
                                 path_ordinals=ordinals, path_trips=trips)
    assert slice_expr == (
        "lhs[(i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128, "
        "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )
```

- [ ] **Step 2: Run — expect failure on these three updated tests (signature mismatch).**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_render.py -v -k "tile_slice or hbm_tile_slice"
```

Expected: three tests fail.

- [ ] **Step 3: Update `_sbuf_tile_slice`, `_hbm_tile_slice`, and `_swapped_dst_tile_slice` in `render.py`.**

Edit `nkigym/src/nkigym/codegen/render.py`. Replace the three existing helper functions (lines ~131-188) with generic versions:

```python
def _slot_expr(path_ordinals: dict[str, int], path_trips: dict[str, list[int]], dim_id: str) -> str:
    """Return the sum-of-ordinals expression for ``dim_id``.

    For dim ``d`` with ``k = path_ordinals[d]`` same-dim ancestors on
    the current path, the slot is the sum over ``i_<d>_{idx} * prod_of_tail_trips``
    for ``idx = 0..k-1``. For the canonical 2N form (trips ``[t_0, 1]``)
    the tail product is 1 for both terms so the slot collapses to
    ``i_<d>_0 + i_<d>_1``.

    Raises:
        ValueError: ``dim_id`` has no open ancestor loops on the path.
    """
    k = path_ordinals.get(dim_id, 0)
    if k == 0:
        raise ValueError(f"No open LoopNode on path for dim {dim_id!r}")
    trips = path_trips[dim_id]
    terms: list[str] = []
    for idx in range(k):
        tail_prod = 1
        for t in trips[idx + 1 :]:
            tail_prod *= t
        if tail_prod == 1:
            terms.append(f"i_{dim_id}_{idx}")
        else:
            terms.append(f"i_{dim_id}_{idx} * {tail_prod}")
    return " + ".join(terms)


def _sbuf_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_ordinals: dict[str, int],
    path_trips: dict[str, list[int]],
) -> str:
    """Return the SBUF ``[p_tile, p_slot, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot = _slot_expr(path_ordinals, path_trips, p_axis)
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_ordinals, path_trips, f_axis)
    f_slot = f"({f_slot_inner})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"


def _hbm_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_ordinals: dict[str, int],
    path_trips: dict[str, list[int]],
) -> str:
    """Return the HBM ``[p_range, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot_inner = _slot_expr(path_ordinals, path_trips, p_axis)
    p_slot = f"({p_slot_inner})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_ordinals, path_trips, f_axis)
    f_slot = f"({f_slot_inner})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )


def _swapped_dst_tile_slice(
    dst_name: str,
    src_p_axis: str,
    src_f_axis: str,
    tile: int,
    path_ordinals: dict[str, int],
    path_trips: dict[str, list[int]],
) -> str:
    """SBUF slice for a transpose's dst tensor (swapped axes)."""
    p_slot = _slot_expr(path_ordinals, path_trips, src_f_axis)
    f_slot_inner = _slot_expr(path_ordinals, path_trips, src_p_axis)
    f_slot = f"({f_slot_inner})"
    return (
        f"{_sbuf_name(dst_name)}[0:{tile}, {p_slot}, "
        f"{f_slot} * {tile} : {f_slot} * {tile} + {tile}]"
    )
```

The existing per-op emitters (`_emit_load`, `_emit_matmul`, etc.) still call these helpers with the old signature, so they'll break. That's intentional — Phase C replaces every caller. For Task C1 we're only updating the helpers; compile errors from unupdated callers surface immediately in Step 4. Temporarily update the per-op emitter call sites to pass empty dicts so the suite keeps running while we roll the walker forward:

In each of `_emit_load`, `_emit_store`, `_emit_matmul`, `_emit_transpose`, `_emit_dma_transpose`, `_emit_activation`, `_emit_tensor_scalar`, `_emit_activation_reduce`, wherever they call `_sbuf_tile_slice`, `_hbm_tile_slice`, or `_swapped_dst_tile_slice`, update the call to pass `path_ordinals` and `path_trips` reconstructed from the still-hand-written loops. For this task, that's a mechanical change: each emitter that opens a block+tile pair for dim `d` with trip count `num_tiles(d)` must build `ordinals = {d: 2, ...}` and `trips = {d: [num_tiles(d), 1], ...}`.

Rather than threading new arguments through every emitter, keep the emitters producing source exactly as today but pre-compute the ordinals/trips for the open nest and pass them in. Concretely, at the top of each emitter, after the loop headers have been written, compute:

```python
path_ordinals = {d: 2 for d in touched_dims_already_opened}
path_trips = {d: [op_graph.dims[d].num_tiles, 1] for d in touched_dims_already_opened}
```

For `_emit_matmul` and `_emit_activation_reduce` the K or F loop is opened later, so maintain the dict incrementally.

This is a lot of mechanical work — it will **all** be thrown away in Task C5 when the walker takes over. The simpler path: skip updating the per-op emitters in Task C1 entirely by keeping the old helper *names* as thin wrappers that reconstruct the trivial case, and renaming the new helpers. Specifically, **rename** the new helpers to `_sbuf_tile_slice_generic`, `_hbm_tile_slice_generic`, `_swapped_dst_tile_slice_generic`, and keep the original `_sbuf_tile_slice` / `_hbm_tile_slice` / `_swapped_dst_tile_slice` unchanged for now. The tests in Step 1 reference the generic names instead:

Adjust the test imports in Step 1 to use `_sbuf_tile_slice_generic` / `_hbm_tile_slice_generic`. Delete the old helpers in Task C5 once the walker replaces all callers.

Actually — for clarity, do this: **introduce the generic helpers under new names; keep the existing helpers untouched**. The old helpers stay until Task C5 deletes them with the last per-op emitter. This avoids any interim hack.

Revised Step 3:

Keep `_sbuf_tile_slice`, `_hbm_tile_slice`, `_swapped_dst_tile_slice` as-is. Add three new functions `_sbuf_tile_slice_generic`, `_hbm_tile_slice_generic`, `_swapped_dst_tile_slice_generic` with the code shown at the top of Step 3 (just under new names). Also add `_slot_expr` as a new helper.

In Step 1, update the test imports to use the `_generic` suffixed names.

- [ ] **Step 4: Run — expect the three updated tests pass and no regressions elsewhere.**

```bash
python -m pytest test/codegen -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add path-ordinal slice helpers for the forest walker"
```

---

### Task C2: Add the `render_forest` walker + body-emitter registry

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Add a walker test that renders a minimal forest for a load+store kernel.**

Append to `test/codegen/test_render.py`:

```python
def test_walker_emits_for_headers_with_path_ordinal_names() -> None:
    """The walker emits 'for i_<d>_<ordinal>' headers with the correct trip count."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import _Writer, render_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.base import AxisRole
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def k(x):
        y = NKILoad()(data=x)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = parse_and_resolve(k, specs)

    """Trivial minimal forest — a single one-loop tree with a marker body."""
    marker_tree = LoopNode(
        "d0", 4, AxisRole.PARALLEL,
        [LoopNode("d0", 1, AxisRole.PARALLEL,
                  [BodyLeaf(op_idx=0, phase="_marker_")])],
    )

    """Register a marker emitter used only by this test."""
    from nkigym.codegen.render import _BODY_EMITTERS

    def _marker_emitter(w, op_graph, op, path_ordinals, path_trips):
        w.line(f"MARK(i_{'d0'}_0={path_ordinals['d0']-2}, trips={path_trips['d0']!r})")

    _BODY_EMITTERS[("NKILoad", "_marker_")] = _marker_emitter

    w = _Writer()
    render_forest(w, g, [marker_tree])
    src = w.getvalue()
    assert "for i_d0_0 in range(4):" in src
    assert "for i_d0_1 in range(1):" in src
    assert "MARK(" in src


def test_body_emitter_registry_has_register_helper() -> None:
    """_register_body decorator stores a function keyed on (op_kind, phase)."""
    from nkigym.codegen.render import _BODY_EMITTERS, _register_body

    @_register_body("TestOp", "test_phase")
    def _emit_test(w, op_graph, op, path_ordinals, path_trips):
        w.line("TEST")

    assert _BODY_EMITTERS[("TestOp", "test_phase")] is _emit_test
    """Clean up the test registration so it doesn't pollute other tests."""
    del _BODY_EMITTERS[("TestOp", "test_phase")]
```

- [ ] **Step 2: Run — expect import / attribute failures.**

```bash
python -m pytest test/codegen/test_render.py::test_walker_emits_for_headers_with_path_ordinal_names -v
```

Expected: failure on import of `render_forest` or `_register_body`.

- [ ] **Step 3: Implement the walker and registry.**

Edit `nkigym/src/nkigym/codegen/render.py`. Append at the end:

```python
from collections.abc import Callable

from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode


_BODY_EMITTERS: dict[tuple[str, str], Callable] = {}
"""Per ``(op_kind, phase)`` body emitter.

A body emitter receives ``(writer, op_graph, parsed_op, path_ordinals,
path_trips)`` and emits that phase's source lines without any loop
headers — the walker is responsible for opening and closing the loops
that frame the body.
"""


def _register_body(op_kind: str, phase: str = "main"):
    """Decorator: register a body emitter for ``(op_kind, phase)``."""

    def wrap(fn: Callable) -> Callable:
        _BODY_EMITTERS[(op_kind, phase)] = fn
        return fn

    return wrap


def render_forest(w: _Writer, op_graph: "OpGraph", forest: LoopForest) -> None:
    """Walk ``forest`` and emit NKI source for every node.

    ``path_ordinals`` tracks how many same-dim ``LoopNode``s are open
    above the current position; ``path_trips`` carries their trip
    counts so body emitters can build correct slot expressions.
    """
    path_ordinals: dict[str, int] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in forest:
        _emit_node(w, op_graph, entry, path_ordinals, path_trips)


def _emit_node(
    w: _Writer,
    op_graph: "OpGraph",
    node: LoopNode | BodyLeaf,
    path_ordinals: dict[str, int],
    path_trips: dict[str, list[int]],
) -> None:
    """Emit one forest node (recursive for ``LoopNode``, delegating for ``BodyLeaf``)."""
    if isinstance(node, BodyLeaf):
        op = op_graph.ops[node.op_idx]
        emitter = _BODY_EMITTERS.get((op.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(
                f"No body emitter registered for ({op.op_cls.__name__!r}, {node.phase!r})"
            )
        emitter(w, op_graph, op, path_ordinals, path_trips)
        return
    k = path_ordinals.get(node.dim_id, 0)
    w.line(f"for i_{node.dim_id}_{k} in range({node.trip_count}):")
    w.indent()
    path_ordinals[node.dim_id] = k + 1
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, op_graph, child, path_ordinals, path_trips)
    path_trips[node.dim_id].pop()
    path_ordinals[node.dim_id] = k
    w.dedent()
```

- [ ] **Step 4: Run — expect pass.**

```bash
python -m pytest test/codegen/test_render.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add render_forest walker and body-emitter registry"
```

---

### Task C3: Port single-phase ops (load/store/activation/tensor_scalar/transpose/dma_transpose) to body emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Add a walker-level end-to-end test for a load+store kernel.**

Append to `test/codegen/test_render.py`:

```python
def test_render_forest_load_store_cpu_sim_matches() -> None:
    """Rendering a passthrough kernel via the walker and simulating it matches numpy."""
    import nki
    import numpy as np

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _Writer,
        _hbm_name,
        render_forest,
    )
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def passthrough(x):
        xs = NKILoad()(data=x)
        out = NKIStore()(data=xs)
        return out

    specs = {"x": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(passthrough, specs)
    forest = build_canonical_forest(g)

    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, g)
    w.indent()
    _emit_param_asserts(w, g)
    _emit_hbm_output(w, g)
    _emit_sbuf_allocations(w, g)
    render_forest(w, g, forest)
    w.line(f"return {_hbm_name(g.return_name)}")
    w.dedent()
    src = w.getvalue()

    assert "for i_d0_0 in range" in src
    assert "for i_d0_1 in range(1):" in src
    assert "nisa.dma_copy(" in src

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

- [ ] **Step 2: Run — expect failure (no body emitter for `NKILoad`/`NKIStore` `"main"`).**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_render.py::test_render_forest_load_store_cpu_sim_matches -v
```

Expected: `ValueError: No body emitter registered for ('NKILoad', 'main')`.

- [ ] **Step 3: Register body emitters for single-phase ops.**

Edit `nkigym/src/nkigym/codegen/render.py`. Append at the end (after `render_forest` and the registry are defined):

```python
@_register_body("NKILoad", "main")
def _body_load(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.dma_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_expr = _sbuf_tile_slice_generic(
        _sbuf_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    src_expr = _hbm_tile_slice_generic(
        src_name, src_tensor.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIStore", "main")
def _body_store(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.dma_copy`` SBUF→HBM at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_expr = _hbm_tile_slice_generic(
        _hbm_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    src_expr = _sbuf_tile_slice_generic(
        _sbuf_name(src_name), src_tensor.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIActivation", "main")
def _body_activation(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.activation`` at the innermost open-loop point."""
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
    dst_expr = _sbuf_tile_slice_generic(
        _sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    src_expr = _sbuf_tile_slice_generic(
        _sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")


@_register_body("NKITensorScalar", "main")
def _body_tensor_scalar(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.tensor_scalar`` at the innermost open-loop point."""
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
    dst_expr = _sbuf_tile_slice_generic(
        _sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    data_expr = _sbuf_tile_slice_generic(
        _sbuf_name(data_name), data.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    op0_expr = _sbuf_tile_slice_generic(
        _sbuf_name(op0_name), op0.dim_ids, p_tile, 1, path_ordinals, path_trips
    )
    _ = f_axis
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")


@_register_body("NKITranspose", "main")
def _body_transpose(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit PSUM alloc + ``nc_transpose`` + ``tensor_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_expr = _sbuf_tile_slice_generic(
        _sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    dst_expr = _swapped_dst_tile_slice_generic(
        dst_name, src_p_axis, src_f_axis, p_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")


@_register_body("NKIDMATranspose", "main")
def _body_dma_transpose(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.dma_transpose`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    src_expr = _sbuf_tile_slice_generic(
        _sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    dst_expr = _swapped_dst_tile_slice_generic(
        dst_name, src_p_axis, src_f_axis, p_tile, path_ordinals, path_trips
    )
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")
```

- [ ] **Step 4: Run — expect the new test to pass, all existing tests still pass.**

```bash
python -m pytest test/codegen -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add body emitters for single-phase ops"
```

---

### Task C4: Port multi-phase ops (matmul, activation_reduce) to phase emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Add walker-level end-to-end tests for matmul and activation_reduce via the forest.**

Append to `test/codegen/test_render.py`:

```python
def _render_via_walker(op_graph) -> str:
    """Helper: render a full kernel through the forest walker."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _Writer,
        _hbm_name,
        render_forest,
    )

    forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph)
    render_forest(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()


def test_render_forest_matmul_cpu_sim_matches() -> None:
    """Full lhs_T @ rhs kernel rendered through the walker simulates correctly."""
    import nki
    import numpy as np
    from nkigym.codegen.graph import parse_and_resolve

    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    src = _render_via_walker(g)
    assert "nisa.nc_matmul(" in src
    assert "nisa.memset(psum_tile" in src
    assert "nisa.tensor_copy(" in src
    assert "for i_d0_0 in range" in src
    assert "for i_d0_1 in range(1):" in src

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


def test_render_forest_activation_reduce_cpu_sim_matches() -> None:
    """RMS kernel rendered through the walker simulates correctly."""
    import nki
    import numpy as np
    from nkigym.codegen.graph import parse_and_resolve

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = _render_via_walker(g)
    assert "nisa.memset(" in src
    assert "nisa.activation_reduce(" in src
    assert "nisa.activation(" in src
    assert "op=nl.rsqrt" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_rms_kernel"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = 1.0 / np.sqrt(np.mean(x * x, axis=1) + 1e-6)
    assert np.allclose(actual.reshape(-1), expected, atol=5e-4, rtol=5e-4)
```

- [ ] **Step 2: Run — expect failure (matmul/activation_reduce phases have no body emitters).**

```bash
python -m pytest test/codegen/test_render.py::test_render_forest_matmul_cpu_sim_matches -v
python -m pytest test/codegen/test_render.py::test_render_forest_activation_reduce_cpu_sim_matches -v
```

Expected: both fail with "No body emitter registered for ('NKIMatmul', 'psum_init')" or similar.

- [ ] **Step 3: Register matmul phase emitters.**

Append to `nkigym/src/nkigym/codegen/render.py`:

```python
@_register_body("NKIMatmul", "psum_init")
def _body_matmul_psum_init(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Allocate + memset the PSUM accumulator once per (M, N) tile."""
    _ = path_ordinals
    _ = path_trips
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")


@_register_body("NKIMatmul", "compute")
def _body_matmul_compute(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit one ``nisa.nc_matmul`` per K tile inside the K loop."""
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size
    stat_expr = _sbuf_tile_slice_generic(
        _sbuf_name(stat_name), stat.dim_ids, p_tile_K, p_tile_M, path_ordinals, path_trips
    )
    mov_expr = _sbuf_tile_slice_generic(
        _sbuf_name(mov_name), mov.dim_ids, p_tile_K, f_tile_N, path_ordinals, path_trips
    )
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKIMatmul", "drain")
def _body_matmul_drain(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Drain the PSUM accumulator into the output SBUF once the K loop closes."""
    out_name = op.output_names[0]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    out_expr = _sbuf_tile_slice_generic(
        _sbuf_name(out_name), out.dim_ids, p_tile_M, f_tile_N, path_ordinals, path_trips
    )
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")
```

- [ ] **Step 4: Register activation_reduce phase emitters.**

Append:

```python
@_register_body("NKIActivationReduce", "reducer_init")
def _body_ar_reducer_init(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Memset the output reducer slot to the reduction identity."""
    dst_name = op.output_names[0]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    identity = _REDUCE_IDENTITY[reduce_op]
    p_slot = _slot_expr(path_ordinals, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(f"nisa.memset({dst_slot}, value={identity})")


@_register_body("NKIActivationReduce", "reduce_step")
def _body_ar_reduce_step(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Per-F-tile activation_reduce + merge into the running accumulator."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    p_axis = op.axis_map["P"]
    f_axis = op.axis_map["F"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    p_slot = _slot_expr(path_ordinals, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(f"tmp_red = nl.ndarray(({p_tile}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    w.line(f"scratch = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    src_expr = _sbuf_tile_slice_generic(
        _sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_ordinals, path_trips
    )
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst=scratch[0:{p_tile}, 0:{f_tile}],")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res=tmp_red[0:{p_tile}, 0:1],")
    w.dedent()
    w.line(")")
    w.line(f"nisa.tensor_tensor({dst_slot}, {dst_slot}, tmp_red[0:{p_tile}, 0:1], op={merge})")


@_register_body("NKIActivationReduce", "post_op")
def _body_ar_post_op(w, op_graph, op, path_ordinals, path_trips) -> None:
    """Emit the closing post-reduction activation (e.g. rsqrt)."""
    dst_name = op.output_names[0]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    post_op = op.op_kwargs["post_op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    p_slot = _slot_expr(path_ordinals, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(
        f"nisa.activation(dst={dst_slot}, op=nl.{post_op}, data={dst_slot}, scale={scale}, bias={bias})"
    )
```

- [ ] **Step 5: Run — expect pass.**

```bash
python -m pytest test/codegen -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add phase body emitters for NKIMatmul and NKIActivationReduce"
```

---

### Task C5: Switch `render` to use the walker and delete the old per-op emitters

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render.py`

- [ ] **Step 1: Update every existing test-render assertion from old to new naming.**

In `test/codegen/test_render.py`, find every assertion that matches either `"for i_block_"` or `"for i_tile_"` in a kernel-source string and update to the new naming scheme. Grep first:

```bash
grep -n "i_block_\|i_tile_" test/codegen/test_render.py
```

Current tests touched by this rename: `test_render_emits_header_and_allocations` (no loop assertion — untouched), `test_render_load_store_kernel` (checks `"for i_block_"` via a substring test — update to `"for i_d0_0"` or `"for i_"` generically), and any test that calls `render(g)` on an end-to-end kernel.

Replace every substring assertion of the form `assert "for i_block_..." in src` (with the exact existing strings) with the new form. Current assertions using the old names are:
- `test_render_load_store_kernel` line containing `"for i_block_"` — replace with `"for i_d0_0"` and `"for i_d0_1"`.

Do this mechanically for every test file that checks loop-variable names.

- [ ] **Step 2: Run — expect the updated tests to fail (still on the old renderer).**

```bash
python -m pytest test/codegen/test_render.py -v
```

Expected: any test that checks new names fails because `render` still uses the old per-op emitters producing old names.

- [ ] **Step 3: Replace `render` with a walker-based implementation.**

Edit `nkigym/src/nkigym/codegen/render.py`.

At the top of the file, update the module docstring's mention of "per-op emitter" to "forest walker" (cosmetic).

Add an import for `build_canonical_forest`:

```python
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, build_canonical_forest
```

Replace the existing `render` function (lines ~54-67) with:

```python
def render(op_graph: OpGraph, forest: LoopForest | None = None) -> str:
    """Render ``op_graph`` to NKI kernel source via the forest walker.

    When ``forest`` is ``None``, a canonical forest is built from
    ``op_graph`` — matches today's default behaviour. Callers with a
    transformed forest (e.g. after fusion rewrites) pass it explicitly.
    """
    if forest is None:
        forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph)
    render_forest(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()
```

Delete the following functions/decorators (the legacy per-op emitters and the registry they used):
- `_emit_op`
- `_register_emitter` + `_EMITTERS`
- `_open_block_tile_loops`
- `_close_loops` (only used by the legacy emitters)
- `_op_header_comment` (still fine to keep if nobody references it; delete if unused)
- `_emit_load`, `_emit_store`, `_emit_matmul`, `_emit_transpose`, `_emit_dma_transpose`, `_emit_activation`, `_emit_tensor_scalar`, `_emit_activation_reduce`
- The old `_sbuf_tile_slice`, `_hbm_tile_slice`, `_swapped_dst_tile_slice` wrappers (keep only the `_generic` versions).

Rename the three helpers by dropping the `_generic` suffix (so the names become `_sbuf_tile_slice`, `_hbm_tile_slice`, `_swapped_dst_tile_slice` again with the new signature). Update every body-emitter call site to use the unsuffixed names.

- [ ] **Step 4: Run — expect all tests pass.**

```bash
python -m pytest test/codegen -v
```

Expected: every test passes, including the end-to-end CPU-sim tests for matmul, rmsnorm, rmsnorm+matmul.

- [ ] **Step 5: Regenerate the rmsnorm_matmul example kernel cache (spot-check).**

```bash
python examples/matmul_lhsT_rhs.py
```

(If the example doesn't exist or isn't importable, run `python -m pytest test/codegen/test_render.py::test_render_matmul_lhsT_rhs -v` as a proxy.) Check that the printed / cached kernel uses `i_d0_0` / `i_d0_1` style names and no `i_block_` / `i_tile_`.

```bash
grep -E "i_block_|i_tile_" /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py 2>/dev/null
```

Expected: empty output (no matches).

- [ ] **Step 6: Commit.**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Switch render() to the LoopForest walker; delete legacy per-op emitters"
```

---

## Phase D — `FuseOuterLoop` rewrite + `tune` stage

### Task D1: Define the `KernelRewrite` protocol

**Files:**
- Create: `nkigym/src/nkigym/rewrites/__init__.py`
- Create: `test/codegen/test_rewrites.py`

- [ ] **Step 1: Test that `KernelRewrite` is importable and has the expected interface.**

Create `test/codegen/test_rewrites.py`:

```python
"""Layer-2 tests: KernelRewrite protocol and FuseOuterLoop."""

from typing import get_type_hints

from nkigym.rewrites import KernelRewrite


def test_kernel_rewrite_is_runtime_checkable_protocol() -> None:
    """KernelRewrite is a typing.Protocol exposing is_legal + apply."""
    assert hasattr(KernelRewrite, "is_legal")
    assert hasattr(KernelRewrite, "apply")


def test_kernel_rewrite_protocol_accepts_minimal_impl() -> None:
    """A class with is_legal and apply satisfies KernelRewrite structurally."""

    class NoopRewrite:
        def is_legal(self, op_graph, forest):
            _ = op_graph, forest
            return True

        def apply(self, op_graph, forest):
            return op_graph, forest

    r: KernelRewrite = NoopRewrite()
    assert r.is_legal(None, []) is True
    new_g, new_f = r.apply(None, [])
    assert new_g is None
    assert new_f == []


def test_type_hints_are_resolvable() -> None:
    """Protocol methods have resolvable type hints (smoke test for future tooling)."""
    hints = get_type_hints(KernelRewrite.is_legal)
    assert "forest" in hints
```

- [ ] **Step 2: Run — expect ImportError.**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: module not found.

- [ ] **Step 3: Create `nkigym/src/nkigym/rewrites/__init__.py`.**

```python
"""Kernel rewrites — performance-related transforms applied by the ``tune`` stage.

Every rewrite is a :class:`KernelRewrite`: it answers whether the
current ``(OpGraph, LoopForest)`` pair admits the transform and, if
so, returns the post-transform pair. Structural rewrites (the common
case today, e.g. :class:`FuseOuterLoop`) leave the ``OpGraph``
untouched. Graph rewrites mutate the ``OpGraph``; the ``tune`` stage
handles the forest contract per the rewrite's semantics.
"""

from typing import Protocol, runtime_checkable

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest


@runtime_checkable
class KernelRewrite(Protocol):
    """A performance-related kernel transform."""

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return ``True`` when the rewrite is applicable to the current state."""
        ...

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Return the post-transform ``(op_graph, forest)`` pair.

        Callers must check :meth:`is_legal` first; ``apply`` on an
        illegal input is not guaranteed to raise.
        """
        ...


__all__ = ["KernelRewrite"]
```

- [ ] **Step 4: Run — expect pass.**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/rewrites/__init__.py test/codegen/test_rewrites.py
git commit -m "Add KernelRewrite protocol"
```

---

### Task D2: Implement `FuseOuterLoop.is_legal`

**Files:**
- Create: `nkigym/src/nkigym/rewrites/fuse_outer_loop.py`
- Modify: `test/codegen/test_rewrites.py`

- [ ] **Step 1: Add is_legal tests.**

Append to `test/codegen/test_rewrites.py`:

```python
import pytest

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.rewrites.fuse_outer_loop import FuseOuterLoop


EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=EPS)(
        data=lhs_sbuf
    )
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def _canonical():
    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    return g, forest


def test_is_legal_accepts_parallel_pair_with_matching_dim_and_trip_count() -> None:
    """ActivationReduce and TensorScalar share d0 PARALLEL; fusing them is legal."""
    g, forest = _canonical()
    atom = FuseOuterLoop(boundary=(2, 3), dim_id="d0")
    assert atom.is_legal(g, forest) is True


def test_is_legal_rejects_non_adjacent_boundary() -> None:
    """Non-adjacent (i, i+2) boundary is illegal."""
    g, forest = _canonical()
    atom = FuseOuterLoop(boundary=(2, 4), dim_id="d0")
    assert atom.is_legal(g, forest) is False


def test_is_legal_rejects_boundary_out_of_range() -> None:
    """Boundary past end of forest is illegal."""
    g, forest = _canonical()
    atom = FuseOuterLoop(boundary=(len(forest) - 1, len(forest)), dim_id="d0")
    assert atom.is_legal(g, forest) is False


def test_is_legal_rejects_dim_mismatch() -> None:
    """Boundary where both sides are LoopNode but on different dims is illegal."""
    g, forest = _canonical()
    atom = FuseOuterLoop(boundary=(2, 3), dim_id="dZZZ")
    assert atom.is_legal(g, forest) is False


def test_is_legal_rejects_trip_count_mismatch() -> None:
    """Synthetic forest: mismatched trips reject fusion."""
    forest = [
        LoopNode("d0", 8, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseOuterLoop(boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_role_mismatch() -> None:
    """Synthetic forest: PARALLEL vs ACCUMULATION rejects fusion."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseOuterLoop(boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_body_leaf_on_either_side() -> None:
    """A BodyLeaf at a boundary position is not a LoopNode — refuse fusion."""
    forest = [
        BodyLeaf(op_idx=0),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseOuterLoop(boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False
```

- [ ] **Step 2: Run — expect ImportError.**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: `FuseOuterLoop` not found.

- [ ] **Step 3: Implement the class.**

Create `nkigym/src/nkigym/rewrites/fuse_outer_loop.py`:

```python
"""``FuseOuterLoop`` rewrite — fuse the outermost loops of adjacent op trees.

See ``docs/superpowers/specs/2026-05-05-axis-role-and-loop-fusion-design.md``
§5 for the design.
"""

from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode
from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class FuseOuterLoop:
    """Fuse the outer-root loops of the adjacent trees at ``boundary``.

    Attributes:
        boundary: ``(i, i+1)`` position pair in the forest.
        dim_id: Concrete dim the two roots must share (guards against
            stale bindings when the caller supplies a list).
    """

    boundary: tuple[int, int]
    dim_id: str

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return ``True`` when the atom can be applied at ``boundary``."""
        _ = op_graph
        i, j = self.boundary
        if j != i + 1 or j >= len(forest) or i < 0:
            return False
        a = forest[i]
        b = forest[j]
        if not isinstance(a, LoopNode) or not isinstance(b, LoopNode):
            return False
        return (
            a.dim_id == self.dim_id
            and b.dim_id == self.dim_id
            and a.role == AxisRole.PARALLEL
            and b.role == AxisRole.PARALLEL
            and a.trip_count == b.trip_count
        )

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Merge the two adjacent roots into one ``LoopNode`` with concatenated children."""
        i, j = self.boundary
        a = forest[i]
        b = forest[j]
        assert isinstance(a, LoopNode) and isinstance(b, LoopNode)
        merged = LoopNode(
            dim_id=self.dim_id,
            trip_count=a.trip_count,
            role=AxisRole.PARALLEL,
            children=[*a.children, *b.children],
        )
        new_forest: LoopForest = [*forest[:i], merged, *forest[j + 1 :]]
        return op_graph, new_forest
```

- [ ] **Step 4: Run — expect all is_legal tests pass.**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: all new tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/rewrites/fuse_outer_loop.py test/codegen/test_rewrites.py
git commit -m "Implement FuseOuterLoop.is_legal and apply"
```

---

### Task D3: Implement `enumerate_fusion_atoms`

**Files:**
- Modify: `nkigym/src/nkigym/rewrites/fuse_outer_loop.py`
- Modify: `test/codegen/test_rewrites.py`

- [ ] **Step 1: Add enumeration tests.**

Append to `test/codegen/test_rewrites.py`:

```python
def test_enumerate_fusion_atoms_rmsnorm_matmul_canonical() -> None:
    """Canonical rmsnorm+matmul yields the expected adjacent-PARALLEL atoms."""
    from nkigym.rewrites.fuse_outer_loop import enumerate_fusion_atoms

    g, forest = _canonical()
    atoms = enumerate_fusion_atoms(forest)

    """Op indices: 0=Load(lhs), 1=Load(rhs), 2=ActivationReduce,
    3=TensorScalar, 4=Transpose, 5=Matmul, 6=Store. All roots iterate d0
    as their outermost loop with PARALLEL role — except ActivationReduce
    and TensorScalar/Transpose/Matmul/Store whose roots are PARALLEL on
    d0. Expect atoms at every adjacent pair where both roots are PARALLEL
    on the same outermost dim with matching trip counts."""
    atom_pairs = {(a.boundary, a.dim_id) for a in atoms}

    """Load(rhs) and ActivationReduce(lhs_sbuf) touch different dims at
    root (Load(rhs) starts with rhs's partition dim, ActivationReduce
    starts with lhs's partition dim d0). So (1,2) is rejected by dim
    mismatch. Sanity: at least (2,3) and (3,4) and (5,6) must be
    present."""
    assert ((2, 3), "d0") in atom_pairs
    assert ((3, 4), "d0") in atom_pairs
    assert ((5, 6), "d0") in atom_pairs


def test_enumerate_returns_empty_on_empty_forest() -> None:
    """Enumeration on an empty forest returns an empty list."""
    from nkigym.rewrites.fuse_outer_loop import enumerate_fusion_atoms

    assert enumerate_fusion_atoms([]) == []


def test_enumerate_returns_empty_when_no_adjacent_pair_matches() -> None:
    """Forest where adjacent trees differ on dim_id yields no atoms."""
    from nkigym.rewrites.fuse_outer_loop import enumerate_fusion_atoms

    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    assert enumerate_fusion_atoms(forest) == []
```

- [ ] **Step 2: Run — expect ImportError.**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: `enumerate_fusion_atoms` not defined.

- [ ] **Step 3: Implement the enumerator.**

Append to `nkigym/src/nkigym/rewrites/fuse_outer_loop.py`:

```python
def enumerate_fusion_atoms(forest: LoopForest) -> list[FuseOuterLoop]:
    """Return every legal :class:`FuseOuterLoop` at an adjacent root boundary.

    Enumeration inspects root-level pairs only — per §5.4 of the design,
    outermost-only by construction. Callers typically re-enumerate after
    every :meth:`FuseOuterLoop.apply` because apply-time index shifts
    invalidate indices produced earlier.
    """
    atoms: list[FuseOuterLoop] = []
    for i in range(len(forest) - 1):
        a = forest[i]
        b = forest[i + 1]
        if not isinstance(a, LoopNode) or not isinstance(b, LoopNode):
            continue
        if (
            a.dim_id == b.dim_id
            and a.role == AxisRole.PARALLEL
            and b.role == AxisRole.PARALLEL
            and a.trip_count == b.trip_count
        ):
            atoms.append(FuseOuterLoop(boundary=(i, i + 1), dim_id=a.dim_id))
    return atoms
```

- [ ] **Step 4: Run — expect pass.**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/rewrites/fuse_outer_loop.py test/codegen/test_rewrites.py
git commit -m "Add enumerate_fusion_atoms"
```

---

### Task D4: Test `FuseOuterLoop.apply` preserves the invariant and composes

**Files:**
- Modify: `test/codegen/test_rewrites.py`

- [ ] **Step 1: Add composition and invariant tests.**

Append to `test/codegen/test_rewrites.py`:

```python
def test_apply_shrinks_forest_length_by_one() -> None:
    """Fusing at (i, i+1) removes one entry from the forest."""
    g, forest = _canonical()
    original_len = len(forest)
    atom = FuseOuterLoop(boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    assert len(new_forest) == original_len - 1


def test_apply_preserves_invariant_on_rmsnorm_matmul() -> None:
    """After fusing activation_reduce ↔ tensor_scalar, check_invariant still holds."""
    from nkigym.codegen.loop_forest import check_invariant

    g, forest = _canonical()
    atom = FuseOuterLoop(boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    check_invariant(new_forest, num_tiles, op_touched)


def test_apply_merged_root_holds_both_subtrees() -> None:
    """The merged LoopNode's children = old-A.children ++ old-B.children."""
    g, forest = _canonical()
    a_children_count = len(forest[2].children)
    b_children_count = len(forest[3].children)
    atom = FuseOuterLoop(boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    merged = new_forest[2]
    assert isinstance(merged, LoopNode)
    assert merged.dim_id == "d0"
    assert merged.role is AxisRole.PARALLEL
    assert len(merged.children) == a_children_count + b_children_count


def test_re_enumeration_after_apply_finds_next_atom_at_shifted_boundary() -> None:
    """After fusing (2,3), the former (3,4) atom appears at (2,3) in the new forest."""
    from nkigym.rewrites.fuse_outer_loop import enumerate_fusion_atoms

    g, forest = _canonical()
    atoms0 = enumerate_fusion_atoms(forest)
    first_legal = next(a for a in atoms0 if a.boundary == (2, 3) and a.dim_id == "d0")
    _, forest1 = first_legal.apply(g, forest)
    atoms1 = enumerate_fusion_atoms(forest1)
    """(3,4) in the original is now (2,3) in forest1 — that tensor_scalar↔transpose
    edge should still be enumerated."""
    assert any(a.boundary == (2, 3) and a.dim_id == "d0" for a in atoms1)
```

- [ ] **Step 2: Run — expect pass (no implementation change needed).**

```bash
python -m pytest test/codegen/test_rewrites.py -v
```

Expected: all tests pass. If any fail, fix the offending logic in `FuseOuterLoop.apply` or `enumerate_fusion_atoms`.

- [ ] **Step 3: Commit.**

```bash
git add test/codegen/test_rewrites.py
git commit -m "Test FuseOuterLoop.apply preserves invariant + composes with re-enumeration"
```

---

### Task D5: Add the `"tune"` stage to `nkigym_compile`

**Files:**
- Modify: `nkigym/src/nkigym/compile.py`
- Modify: `test/codegen/test_compile.py`

- [ ] **Step 1: Add integration tests for the tune stage.**

Append to `test/codegen/test_compile.py`:

```python
from nkigym.rewrites.fuse_outer_loop import FuseOuterLoop


def test_tune_stage_with_empty_rewrites_produces_kernel_tuned_file(tmp_path):
    """`tune` with `rewrites=[]` writes kernel_tuned.py and CPU-sim succeeds."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(
        _rmsnorm_matmul_numpy, specs, tmp_path,
        stages=["tune"], rewrites=[],
    )
    assert (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_applies_fuse_outer_loop_d0(tmp_path):
    """Applying FuseOuterLoop on activation_reduce↔tensor_scalar d0 still matches numpy."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    rewrites = [FuseOuterLoop(boundary=(2, 3), dim_id="d0")]
    nkigym_compile(
        _rmsnorm_matmul_numpy, specs, tmp_path,
        stages=["tune"], rewrites=rewrites,
    )
    kernel_source = (tmp_path / "kernel_tuned.py").read_text()
    """Smoke check: the merged d0 loop shows the activation_reduce body
    immediately followed by the tensor_scalar body inside the same block."""
    assert "activation_reduce" in kernel_source
    assert "tensor_scalar" in kernel_source


def test_tune_stage_composes_multiple_atoms(tmp_path):
    """Apply (2,3) then — on the shifted forest — (2,3) again to cover two fusions."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    rewrites = [
        FuseOuterLoop(boundary=(2, 3), dim_id="d0"),
        FuseOuterLoop(boundary=(2, 3), dim_id="d0"),
    ]
    nkigym_compile(
        _rmsnorm_matmul_numpy, specs, tmp_path,
        stages=["tune"], rewrites=rewrites,
    )
    assert (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_rejects_illegal_rewrite(tmp_path):
    """A rewrite that fails is_legal raises ValueError before writing any artifact."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    bogus = FuseOuterLoop(boundary=(99, 100), dim_id="d0")
    with pytest.raises(ValueError, match="illegal"):
        nkigym_compile(
            _rmsnorm_matmul_numpy, specs, tmp_path,
            stages=["tune"], rewrites=[bogus],
        )
    assert not (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_random_draw_is_reproducible(tmp_path):
    """Running tune twice with rewrites=None + same seed produces identical output."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], seed=42)
    first = (tmp_path / "kernel_tuned.py").read_text()
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], seed=42)
    second = (tmp_path / "kernel_tuned.py").read_text()
    assert first == second
```

- [ ] **Step 2: Run — expect failure (tune stage not implemented).**

```bash
source ~/venvs/kernel-env/bin/activate
python -m pytest test/codegen/test_compile.py -v
```

Expected: `ValueError: Unknown stage 'tune'`.

- [ ] **Step 3: Add the `"tune"` stage implementation.**

Edit `nkigym/src/nkigym/compile.py`.

Extend imports near the top:

```python
import random
```

Add:

```python
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.render import render
from nkigym.rewrites import KernelRewrite
from nkigym.rewrites.fuse_outer_loop import enumerate_fusion_atoms
```

Change `_KNOWN_STAGES`:

```python
_KNOWN_STAGES = ("synthesis", "initial_codegen", "tune")
```

Add `rewrites` and `seed` kwargs on `nkigym_compile`:

```python
def nkigym_compile(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    stages: list[str],
    rewrites: list[KernelRewrite] | None = None,
    seed: int = 0,
) -> None:
    """Run the nkigym compilation pipeline through the given ``stages``.

    Args:
        f_numpy: Plain-numpy reference — drives both synthesis and the
            CPU-sim accuracy check.
        input_specs: ``{param_name: (shape, dtype)}`` for every parameter.
        cache_dir: Directory to write stage artifacts.
        stages: Ordered list of stage names to run. Supported:
            ``"synthesis"``, ``"initial_codegen"``, ``"tune"``.
        rewrites: Consumed only by the ``"tune"`` stage. When ``None``,
            the tune stage randomises from :func:`enumerate_fusion_atoms`
            using ``seed``. When a list is given, rewrites are applied in
            order; each rewrite's legality is checked against the current
            ``(op_graph, forest)`` state.
        seed: Seeds the random rewrite draw when ``rewrites`` is ``None``.

    Raises:
        ValueError: An unknown stage name is in ``stages``, a stage's
            required input artifact is missing, or an explicit rewrite
            is illegal on the current state.
        AssertionError: The CPU-sim result of the rendered kernel
            diverges from ``f_numpy``.
    """
    for stage in stages:
        if stage not in _KNOWN_STAGES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {_KNOWN_STAGES}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        if stage == "synthesis":
            _run_synthesis(f_numpy, input_specs, cache_path)
        elif stage == "initial_codegen":
            _run_initial_codegen(f_numpy, input_specs, cache_path)
        elif stage == "tune":
            _run_tune(f_numpy, input_specs, cache_path, rewrites=rewrites, seed=seed)
```

Add the `_run_tune` helper:

```python
def _run_tune(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    rewrites: list[KernelRewrite] | None,
    seed: int,
) -> None:
    """Apply structural rewrites, render, write kernel_tuned.py, CPU-sim check."""
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"tune requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = _load_f_nkigym(f_nkigym_path)
    op_graph = parse_and_resolve(f_nkigym, input_specs)
    forest = build_canonical_forest(op_graph)

    if rewrites is None:
        rng = random.Random(seed)
        while True:
            atoms = enumerate_fusion_atoms(forest)
            candidates = [a for a in atoms if rng.random() < 0.5]
            if not candidates:
                break
            chosen = candidates[0]
            op_graph, forest = chosen.apply(op_graph, forest)
    else:
        for r in rewrites:
            if not r.is_legal(op_graph, forest):
                raise ValueError(f"{r!r} illegal on current state")
            op_graph, forest = r.apply(op_graph, forest)

    kernel_source = render(op_graph, forest=forest)
    (cache_path / "kernel_tuned.py").write_text(kernel_source)
    _cpu_sim_check(kernel_source, f_nkigym.__name__, f_numpy, input_specs)
```

- [ ] **Step 4: Run — expect pass.**

```bash
python -m pytest test/codegen/test_compile.py -v
```

Expected: all tune-stage tests pass, plus existing `initial_codegen` tests still pass.

- [ ] **Step 5: Commit.**

```bash
git add nkigym/src/nkigym/compile.py test/codegen/test_compile.py
git commit -m "Add 'tune' stage to nkigym_compile with explicit + random paths"
```

---

### Task D6: Add `examples/rmsnorm_matmul_tuned.py`

**Files:**
- Create: `examples/rmsnorm_matmul_tuned.py`

- [ ] **Step 1: Write the example file.**

Create `examples/rmsnorm_matmul_tuned.py`:

```python
"""End-to-end demo of the `tune` stage on the rmsnorm+matmul kernel.

Runs:

    1. ``"synthesis"`` — synthesise ``f_nkigym`` from numpy (skipped if
       ``f_nkigym.py`` already exists in the cache).
    2. ``"initial_codegen"`` — render the canonical kernel.
    3. ``"tune"`` — apply an explicit list of structural rewrites
       (currently: two ``FuseOuterLoop`` atoms) and CPU-sim the result.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul_tuned.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym import nkigym_compile
from nkigym.rewrites.fuse_outer_loop import FuseOuterLoop


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` golden."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = lhs * rms_inv
    return normed @ rhs


if __name__ == "__main__":
    cache_dir = Path("/home/ubuntu/cache/rmsnorm_matmul_tuned")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    """Rewrite list: fuse activation_reduce↔tensor_scalar on d0, then fuse
    the now-adjacent tensor_scalar↔transpose pair (shifted to (2,3) in the
    forest after the first apply)."""
    rewrites = [
        FuseOuterLoop(boundary=(2, 3), dim_id="d0"),
        FuseOuterLoop(boundary=(2, 3), dim_id="d0"),
    ]

    nkigym_compile(
        rmsnorm_matmul_numpy,
        INPUT_SPECS,
        cache_dir,
        stages=["synthesis", "initial_codegen", "tune"],
        rewrites=rewrites,
    )
    print(f"[rmsnorm_matmul_tuned] canonical kernel: {cache_dir / 'kernel.py'}")
    print(f"[rmsnorm_matmul_tuned] tuned kernel:     {cache_dir / 'kernel_tuned.py'}")
```

- [ ] **Step 2: Smoke-run the example.**

```bash
source ~/venvs/kernel-env/bin/activate
python examples/rmsnorm_matmul_tuned.py
```

Expected: the script runs to completion, prints both kernel paths, and the CPU-sim check inside `tune` passes. If the synthesis stage hangs (agent SDK call), manually write `f_nkigym.py` matching the source in `test/codegen/test_compile.py::_F_NKIGYM_SOURCE` to the cache dir and run with `stages=["initial_codegen", "tune"]` instead.

- [ ] **Step 3: Run the full test suite one more time — expect all pass.**

```bash
python -m pytest test/codegen -v
```

Expected: every test passes.

- [ ] **Step 4: Commit.**

```bash
git add examples/rmsnorm_matmul_tuned.py
git commit -m "Add rmsnorm_matmul_tuned.py example demonstrating the tune stage"
```

---

## Phase E — Cache invalidation sweep

### Task E1: Regenerate any cached kernel library entries

**Files:**
- Possibly modify: `kernel_library/` entries (optional).

- [ ] **Step 1: Inventory cached kernels still using old loop-variable names.**

```bash
grep -rn "i_block_\|i_tile_" /home/ubuntu/nki-autotune/kernel_library 2>/dev/null | head -20
```

Expected: either no matches (nothing cached with old names), or hand-written reference kernels that are intentionally kept.

- [ ] **Step 2: Document in a final commit.**

If no matches, there's nothing to regenerate — skip to Step 3.

If matches exist, note whether each file is:
- An auto-generated kernel (then regenerate by running the corresponding example / test);
- A hand-written reference kernel (then leave untouched — `kernel_hand_*.py` files per the repo convention are not rendered output).

- [ ] **Step 3: Final commit — no code changes, just a note.**

If nothing needed updating:

```bash
git commit --allow-empty -m "Cache sweep complete — no stale i_block_/i_tile_ references in kernel_library/"
```

If files were regenerated, include them:

```bash
git add kernel_library/
git commit -m "Regenerate cached kernels with path-ordinal loop variable names"
```

---

## Self-Review

**Spec coverage walk-through** (every numbered section of the spec maps to at least one task):

- §2.1 AxisRole enum → Task A1.
- §2.2 AXIS_ROLES replaces BLOCKING_AXES → Tasks A1, A2, A5.
- §2.3 ParsedOp.dim_role → Task A3.
- §2.4 LoopForest IR → Task B1.
- §2.5 Multi-phase ops → Tasks B3, B4.
- §3 Canonical forest → Tasks B2, B3, B4.
- §4.1 Walker → Task C2.
- §4.2 Path-ordinal naming → Task C1 (generic helpers); Task C5 (rename back).
- §4.3 Slot expressions → Task C1.
- §4.4 Body emitter registry → Task C2.
- §4.5 What stays put in render.py → Task C5 (cleanup).
- §5.1–5.4 FuseOuterLoop → Tasks D2, D3.
- §5.5 Composition example (rmsnorm+matmul) → Task D4.
- §5.6 Out of scope → documented in spec; plan does not introduce tile-fusion atoms.
- §6 KernelRewrite protocol → Task D1.
- §7 tune stage → Task D5.
- §8.1–8.2 Layer 1 + Layer 2 tests → Tasks A1, A2, A3, A5 (axis-role); B1, B2, B3, B4 (forest); D2, D3, D4 (rewrites).
- §8.3 Layer 3 (rendered source) → Tasks C3, C4, C5.
- §8.4 Layer 3 (tune integration) → Task D5.
- §8.5 rmsnorm_matmul_tuned.py → Task D6.
- §9 Phase A / B / C / D / E → mapped 1:1 to plan phases.
- §11 File changes → every row mapped to a task.

**Placeholder scan:** every test step has concrete code; every command is exact; every file path is absolute-relative-to-repo-root; no `TODO` / `TBD` / "implement later" anywhere. ✓

**Type consistency:**
- `FuseOuterLoop` — same field names used in every task (`boundary: tuple[int, int]`, `dim_id: str`).
- `LoopNode` — same fields (`dim_id`, `trip_count`, `role`, `children`) referenced in every task.
- `BodyLeaf` — always `(op_idx, phase)` with `phase` defaulting to `"main"`.
- `path_ordinals: dict[str, int]` and `path_trips: dict[str, list[int]]` — consistent signatures on every helper.
- Phase names match: matmul's `psum_init` / `compute` / `drain`; activation_reduce's `reducer_init` / `reduce_step` / `post_op`.

**Scope check:** single milestone, five linear phases, one design doc. ✓
