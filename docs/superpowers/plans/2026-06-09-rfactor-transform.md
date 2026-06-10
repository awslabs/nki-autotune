# RFactor Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RFactor` rewrite transform to `nkigym` that faithfully ports `tir.Schedule.rfactor` — splitting a reduction loop into an rf-buffer-filling block plus a separate write-back block.

**Architecture:** RFactor consumes a `ForNode` bound to an `ACCUMULATION` axis (produced by a prior `Split`) and emits TVM's terminal form: a `[factor, *tile(out)]` rf-buffer, an **rf-block** that fills it (factored loop role-flipped reduction→parallel), and a separate **wb-block** that reduces the slots into the original output. Loops are NOT reordered. The transform follows the shipped `Transform` idiom (`analyze`/`apply`, deep-copy + mutate + re-check + loud `TransformLegalityError`), mirroring `SoftwarePipeline`. Two new pieces of op infrastructure are required: a `REDUCE_COMBINATOR` declaration on `NKIOp` (mirrors TVM's `CommReducer`) and a `NKITensorTensor` op (the `"rmw"` wb-block combine `out += B_rf[ko]`).

**Tech Stack:** Python 3.12, `networkx` (IR tree), `numpy` (CPU sim). Tests via `pytest`. Renderer is generic (emits `nisa.<NAME>(...)` from `OPERAND_AXES`) so **no codegen changes are needed**. Unit tests run on the Kaizen desktop via `transport/remote_pytest.sh` (no local Python env).

**Spec:** `docs/superpowers/specs/2026-06-07-rfactor-transform-design.md`
**TVM reference:** `.claude/rules/tvm_knowledge.md` (source-verified rfactor semantics)

---

## Scope

This plan implements **only the RFactor atom (`"rmw"` recipe) + its two infra dependencies + tests.** Explicitly OUT of scope (per spec §8): the `"slot"` recipe, the rf-buffer→fused fold (blocked on the parked `_check_no_reduction_axis_covered` narrowing), the hoist of `ko` outside M, `SoftwarePipeline` composition, and byte-exact reproduction of `kernel_hand_90.92mfu.py`.

## File Structure

**New files:**
- `nkigym/src/nkigym/ops/tensor_tensor.py` — `NKITensorTensor` op (elementwise tensor⊕tensor; the wb-block combine). One responsibility: the ISA op + its CPU sim.
- `nkigym/src/nkigym/transforms/rfactor.py` — `RFactor` transform + `RFactorOption`. One responsibility: the rf-buffer + wb-block emission and its legality.
- `test/ops/test_tensor_tensor.py` — unit tests for `NKITensorTensor` (render + sim).
- `test/transforms/test_rfactor.py` — unit tests for `RFactor` (analyze, apply byte-exact, legality, dependency-order).
- `test/transforms/_rfactor_fixtures.py` — shared fixture: canonical matmul IR + `Split(K)` → the pre-RFactor state, plus node-discovery helpers. Mirrors `_pipeline_fixtures.py`.

**Modified files:**
- `nkigym/src/nkigym/ops/base.py` — add `REDUCE_COMBINATOR` class attribute + `ReduceCombinator` dataclass on `NKIOp`.
- `nkigym/src/nkigym/ops/matmul.py` — set `NKIMatmul.REDUCE_COMBINATOR = ReduceCombinator(combiner="add", identity=0.0)`.
- `nkigym/src/nkigym/transforms/__init__.py` — export `RFactor`, `RFactorOption`.

## Conventions (read before starting)

- **Code style** (`.claude/rules/code_style.md`): triple-quoted block comments only (no `#` comments except tooling directives); Google/NumPy docstrings on every function/class; modern type hints (`X | None`, `list`, `dict`); single return per function where practical; loud failures (no silent `except`). A pre-commit hook reformats with `black`/`isort` — re-stage and retry if it aborts.
- **Running tests:** there is NO local Python env. Run every `pytest` via
  `AWS_PROFILE=kaizen-access transport/remote_pytest.sh <args>`. Refresh creds first if expired (see `.claude/rules/learnings.md` / kaizen-desktop skill: `ada credentials update ... --profile=kaizen-access --once` and `... --profile=cluster-role --once`).
- **2 PRE-EXISTING desktop test failures** (NOT regressions): `test_dump_tree_runs_on_canonical_ir`, `test_fuse_tensorize_matmul_n_renders_and_sims`. Verify any "new" failure against the parent commit before treating it as yours.
- **Commit cadence:** commit after each task's tests pass. Branch is `dev_1` (not `main`) — commit there. End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

---

### Task 1: `ReduceCombinator` + `REDUCE_COMBINATOR` on `NKIOp`

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py` (add dataclass + ClassVar near `RFACTOR_RECIPE`, ~line 143)
- Modify: `nkigym/src/nkigym/ops/matmul.py:18-34` (declare it on `NKIMatmul`)
- Test: `test/ops/test_reduce_combinator.py` (create)

- [ ] **Step 1: Write the failing test**

Create `test/ops/test_reduce_combinator.py`:

```python
"""Tests for the REDUCE_COMBINATOR reducer declaration on NKIOp."""

from __future__ import annotations

from nkigym.ops.base import NKIOp, ReduceCombinator
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy


def test_matmul_declares_sum_reducer() -> None:
    """NKIMatmul exposes the sum reducer (combiner='add', identity=0.0)."""
    rc = NKIMatmul.REDUCE_COMBINATOR
    assert isinstance(rc, ReduceCombinator)
    assert rc.combiner == "add"
    assert rc.identity == 0.0


def test_non_reduction_op_has_no_reducer() -> None:
    """An op with no reduction (tensor_copy) declares REDUCE_COMBINATOR = None."""
    assert NKITensorCopy.REDUCE_COMBINATOR is None


def test_base_default_is_none() -> None:
    """The base NKIOp default is None."""
    assert NKIOp.REDUCE_COMBINATOR is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/ops/test_reduce_combinator.py -q`
Expected: FAIL — `ImportError: cannot import name 'ReduceCombinator'` (or `AttributeError` on `REDUCE_COMBINATOR`).

- [ ] **Step 3: Add `ReduceCombinator` + the ClassVar in `base.py`**

In `nkigym/src/nkigym/ops/base.py`, add the dataclass near the top (after the imports / before `class NKIOp`). `dataclass` is already imported if not, add `from dataclasses import dataclass`:

```python
@dataclass(frozen=True)
class ReduceCombinator:
    """An op's commutative-associative reducer, mirroring TVM's CommReducer.

    Attributes:
        combiner: the ``nl.*`` op name applied in the RFactor wb-block combine
            (e.g. ``"add"`` for a sum reduction).
        identity: the value RFactor memsets both the rf-block slot and the
            wb-block accumulator to before reducing (e.g. ``0.0`` for sum).
    """

    combiner: str
    identity: float
```

Then inside `class NKIOp`, REPLACE the existing `RFACTOR_RECIPE` docstring block with a corrected one and add `REDUCE_COMBINATOR` immediately after it:

```python
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = None
    """Which RFactor recipe this op supports, or ``None`` if not rfactorable.

    Both recipes emit the same rf-buffer + write-back-block shape; they differ
    only in how the rf-block's per-slot accumulate lowers:

    - ``"rmw"``: ops with a HW accumulator (matmul). Per-slot accumulate is
      HW ``+=`` into a PSUM slot, drained to the SBUF rf-buffer; the wb-block
      combine is a ``tensor_tensor``.
    - ``"slot"``: ops with no HW accumulator (activation_reduce). Each slot is
      written directly in SBUF; the wb-block closes with a ``tensor_reduce``.
    - ``None``: RFactor legality rejects any atom targeting this op.
    """

    REDUCE_COMBINATOR: ClassVar["ReduceCombinator | None"] = None
    """The op's commutative-associative reducer, or ``None`` if not a reduction.

    RFactor reads this to synthesize the rf-block per-slot init, the wb-block
    init (both ``memset`` to ``identity``), and the wb-block combine (an ISA op
    applying ``combiner``). Must be set on every op whose ``RFACTOR_RECIPE`` is
    not ``None``.
    """
```

- [ ] **Step 4: Declare it on `NKIMatmul`**

In `nkigym/src/nkigym/ops/matmul.py`, add the import and the class attribute. Edit the import line at the top:

```python
from nkigym.ops.base import AxisRole, NKIOp, ReduceCombinator, _operand_role
```

Add this line in the `NKIMatmul` ClassVar block (after `RFACTOR_RECIPE`, ~line 29):

```python
    REDUCE_COMBINATOR: ClassVar[ReduceCombinator | None] = ReduceCombinator(combiner="add", identity=0.0)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/ops/test_reduce_combinator.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ops/base.py nkigym/src/nkigym/ops/matmul.py test/ops/test_reduce_combinator.py
git commit -m "feat(ops): add ReduceCombinator + REDUCE_COMBINATOR (matmul=sum)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `NKITensorTensor` op (elementwise tensor⊕tensor)

**Files:**
- Create: `nkigym/src/nkigym/ops/tensor_tensor.py`
- Test: `test/ops/test_tensor_tensor.py`

The renderer is generic (`codegen/body.py:166-182` emits `nisa.<NAME>(slot=region, …, kwarg=value)` from `OPERAND_AXES` + `operand_bindings` + `kwargs`), so adding the op class is sufficient — no codegen change.

- [ ] **Step 1: Write the failing test**

Create `test/ops/test_tensor_tensor.py`:

```python
"""Tests for the NKITensorTensor elementwise op (CPU sim + operand classification)."""

from __future__ import annotations

import numpy as np
import pytest

from nkigym.ops.tensor_tensor import NKITensorTensor


@pytest.mark.parametrize("op,expected", [("add", 5.0), ("subtract", 1.0), ("multiply", 6.0)])
def test_run_applies_op_elementwise(op: str, expected: float) -> None:
    """CPU sim applies the named op elementwise over data1, data2."""
    data1 = np.full((3, 4), 3.0, dtype=np.float32)
    data2 = np.full((3, 4), 2.0, dtype=np.float32)
    out = NKITensorTensor()._run(data1=data1, data2=data2, op=op)
    np.testing.assert_allclose(out, np.full((3, 4), expected, dtype=np.float32), atol=1e-6)


def test_operand_axes_and_rmw() -> None:
    """data1 is the RMW accumulator; data2 is the read-only input; slots mirror the ISA."""
    assert NKITensorTensor.OPERAND_AXES == {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    assert NKITensorTensor.RMW_OPERANDS == frozenset({"data1"})
    assert NKITensorTensor.INPUT_OPERANDS == frozenset({"data2"})


def test_name_is_isa_call() -> None:
    """NAME matches the nisa call the generic renderer emits."""
    assert NKITensorTensor.NAME == "tensor_tensor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/ops/test_tensor_tensor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.ops.tensor_tensor'`.

- [ ] **Step 3: Write the op**

Create `nkigym/src/nkigym/ops/tensor_tensor.py`:

```python
"""Elementwise tensor-tensor op: maps to ``nisa.tensor_tensor``.

Applies ``dst = data1 <op> data2`` over two same-shape ``(P, F)`` SBUF tensors.
RFactor's ``"rmw"`` write-back block uses this as the running combine
``out_sbuf = out_sbuf + B_rf[ko]`` — ``data1`` is the RMW accumulator, ``data2``
the per-slot partial.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role

_OPS: dict[str, Any] = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply}


class NKITensorTensor(NKIOp):
    """Elementwise ``dst = data1 <op> data2`` over two ``(P, F)`` tensors."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data2"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """``data1`` and ``data2`` must both be SBUF-resident."""
        for slot in ("data1", "data2"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKITensorTensor({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``data1 <op> data2`` elementwise."""
        return _OPS[kwargs["op"]](kwargs["data1"], kwargs["data2"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/ops/test_tensor_tensor.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/tensor_tensor.py test/ops/test_tensor_tensor.py
git commit -m "feat(ops): add NKITensorTensor (elementwise tensor-tensor, RMW data1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: RFactor pre-state fixture (canonical matmul + Split(K))

This fixture builds the *input* every RFactor test/apply needs: the canonical
matmul IR with K split into `(ko, ki)`, one PSUM accumulator, `ko`/`ki` both
`ACCUMULATION`. It mirrors `test/transforms/_pipeline_fixtures.py`.

**Files:**
- Create: `test/transforms/_rfactor_fixtures.py`

First discover the canonical nids (deterministic — `build_initial_ir` is stable):

- [ ] **Step 1: Print the canonical matmul tree to read its nids**

Run:
```bash
AWS_PROFILE=kaizen-access transport/remote_pytest.sh -q -s -c /dev/null \
  --no-header -p no:cacheprovider \
  --pyargs /dev/stdin <<'PY'
PY
```
That form is awkward; instead add a throwaway print via a one-off test file
`test/transforms/_rfactor_scout.py`:

```python
"""Throwaway: print canonical matmul tree nids to author the fixture. Delete after."""

from test.transforms._fixtures import build_canonical_ir


def test_print_tree() -> None:
    ir = build_canonical_ir()
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        kind = type(data).__name__
        label = getattr(data, "loop_var", getattr(data, "op_cls", ""))
        print(nid, kind, label)
    assert True
```

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/_rfactor_scout.py -q -s`
Expected: a printed node list. **Record the `ForNode` nid whose `loop_var` is the K loop** (`i_d0_0`) and the matmul `ISANode` nid. Note the K extent (2048/128 = 16 trips). These nids feed the fixture's `Split` call and `RFactorOption`.

- [ ] **Step 2: Delete the scout file**

```bash
rm test/transforms/_rfactor_scout.py
```

- [ ] **Step 3: Write the fixture**

Create `test/transforms/_rfactor_fixtures.py`. Replace `K_LOOP_NID` / `factors` /
`MATMUL_LEAF` placeholders with the values read in Step 1 (e.g. K loop `i_d0_0` is
the canonical `nid` you recorded; factors `(2, 8)` split 16 trips into `ko=2, ki=8`):

```python
"""Shared fixture: canonical matmul IR with K split into (ko, ki) — the RFactor input."""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption
from test.transforms._fixtures import build_canonical_ir


def split_k_ir() -> KernelIR:
    """Canonical matmul IR after Split(K -> ko=2, ki=8). One PSUM accumulator.

    The K loop (``i_d0_0``, 16 trips) is split outer-trip into ko=2 over ki=8.
    Both resulting loops bind the matmul's K axis (ACCUMULATION).
    """
    ir = build_canonical_ir()
    k_loop = _k_loop_nid(ir)
    return Split().apply(ir, SplitOption(target_nid=k_loop, factors=(2, 8), target_axis=None))


def _k_loop_nid(ir: KernelIR) -> int:
    """Return the ForNode nid binding the matmul K loop (loop_var 'i_d0_0')."""
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var == "i_d0_0"
    )


def ko_loop_nid(ir: KernelIR) -> int:
    """Return the OUTER K loop (ko) ForNode nid in a post-Split IR.

    After Split, two K loops exist; ko is the outer (first in preorder among the
    matmul's K-axis ForNodes). Used as RFactorOption.target_loop_nid.
    """
    matmul_leaf = matmul_leaf_nid(ir)
    k_loops = [
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith("i_d0_")
    ]
    return k_loops[0]


def matmul_leaf_nid(ir: KernelIR) -> int:
    """Return the nc_matmul ISANode nid."""
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
```

- [ ] **Step 4: Smoke-test the fixture**

Add to `test/transforms/test_rfactor.py` (created fully in Task 5; for now just this
one test to validate the fixture):

```python
from test.transforms._rfactor_fixtures import ko_loop_nid, split_k_ir
from nkigym.ir.tree import ForNode


def test_fixture_splits_k() -> None:
    """split_k_ir yields a tree with an outer K loop (ko) of extent 2."""
    ir = split_k_ir()
    ko = ko_loop_nid(ir)
    node = ir.tree.data(ko)
    assert isinstance(node, ForNode)
    assert node.extent == 2
```

- [ ] **Step 5: Run the smoke test**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/test_rfactor.py::test_fixture_splits_k -q`
Expected: PASS. If the K extent or `ko` ordering differs from the printed tree, fix the fixture's `factors` / `_k_loop_nid` to match Step 1's reading (do NOT loosen the assertion).

- [ ] **Step 6: Commit**

```bash
git add test/transforms/_rfactor_fixtures.py test/transforms/test_rfactor.py
git commit -m "test(rfactor): add canonical-matmul + Split(K) fixture

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `RFactor` transform — skeleton, analyze, legality

Implement the transform class with `analyze` + legality first (no emission yet);
`apply` raises `NotImplementedError` until Task 5. This lets the analyze/legality
tests pass independently and keeps the emission (the hard part) isolated.

**Files:**
- Create: `nkigym/src/nkigym/transforms/rfactor.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`

- [ ] **Step 1: Write failing analyze/legality tests**

Append to `test/transforms/test_rfactor.py`:

```python
import pytest

from nkigym.transforms import RFactor, RFactorOption, TransformLegalityError
from test.transforms._rfactor_fixtures import matmul_leaf_nid, split_k_ir


def test_analyze_finds_the_ko_loop() -> None:
    """analyze offers exactly the rfactorable reduction loops (ko and ki)."""
    ir = split_k_ir()
    opts = RFactor().analyze(ir)
    target_nids = {o.target_loop_nid for o in opts}
    """Both K loops bind an ACCUMULATION axis of an op with RFACTOR_RECIPE='rmw'."""
    assert len(target_nids) >= 1
    for o in opts:
        node = ir.tree.data(o.target_loop_nid)
        assert node.loop_var.startswith("i_d0_")


def test_apply_rejects_non_forNode() -> None:
    """A target that is not a ForNode is rejected loudly."""
    ir = split_k_ir()
    mm = matmul_leaf_nid(ir)
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, RFactorOption(target_loop_nid=mm, factor_axis=0))


def test_apply_rejects_parallel_loop() -> None:
    """A PARALLEL loop (e.g. the M loop) is not a reduction loop → rejected."""
    ir = split_k_ir()
    from nkigym.ir.tree import ForNode
    m_loop = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var.startswith("i_d1_")
    )
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, RFactorOption(target_loop_nid=m_loop, factor_axis=0))
```

- [ ] **Step 2: Run to verify failure**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/test_rfactor.py -q`
Expected: FAIL — `ImportError: cannot import name 'RFactor'`.

- [ ] **Step 3: Write the transform skeleton + analyze + legality**

Create `nkigym/src/nkigym/transforms/rfactor.py`:

```python
"""``RFactor`` transform — faithful port of TVM ``tir.Schedule.rfactor``.

Splits a reduction loop into an rf-block (fills a ``[factor, *tile(out)]``
rf-buffer; factored loop flipped reduction→parallel) and a separate wb-block
(reduces the slots into the original output). Loops are NOT reordered. See
``docs/superpowers/specs/2026-06-07-rfactor-transform-design.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Factor the reduction loop ``target_loop_nid``.

    Attributes:
        target_loop_nid: the ForNode (a reduction/ACCUMULATION loop) to factor.
        factor_axis: position of the prepended ``[factor]`` dim in the rf-buffer
            (TVM ``rfactor(loop, factor_axis)``; default 0).
    """

    target_loop_nid: int
    factor_axis: int = 0


class RFactor(Transform):
    """Faithful ``tir.Schedule.rfactor``: rf-buffer + write-back block."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Enumerate every ForNode that binds an ACCUMULATION axis of an
        rfactorable op (RFACTOR_RECIPE not None)."""
        options: list[RFactorOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            if self._rfactorable(ir, nid):
                options.append(RFactorOption(target_loop_nid=nid, factor_axis=0))
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Re-check legality, deep-copy, emit rf-block + wb-block, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._emit_rmw(new_ir, option)
        return new_ir

    def _rfactorable(self, ir: KernelIR, loop_nid: int) -> bool:
        """True iff ``loop_nid`` binds an ACCUMULATION axis whose owning op
        declares RFACTOR_RECIPE='rmw' and a REDUCE_COMBINATOR."""
        leaf = self._owning_matmul_leaf(ir, loop_nid)
        result = False
        if leaf is not None:
            op_cls = ir.tree.data(leaf).op_cls
            block = self._enclosing_block(ir, leaf)
            axis = self._loop_axis(ir, loop_nid, block)
            if (
                op_cls.RFACTOR_RECIPE == "rmw"
                and op_cls.REDUCE_COMBINATOR is not None
                and axis is not None
                and self._role_of(block, axis) == AxisRole.ACCUMULATION
            ):
                result = True
        return result

    def _check_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise TransformLegalityError if the option is not a valid rmw rfactor."""
        nid = option.target_loop_nid
        if nid not in ir.tree.graph or not isinstance(ir.tree.data(nid), ForNode):
            raise TransformLegalityError(f"RFactor target {nid} is not a ForNode in the tree")
        if not self._rfactorable(ir, nid):
            raise TransformLegalityError(
                f"RFactor target loop {nid} does not bind an ACCUMULATION axis of an "
                f"op with RFACTOR_RECIPE='rmw' + a REDUCE_COMBINATOR"
            )

    def _emit_rmw(self, ir: KernelIR, option: RFactorOption) -> None:
        """Emit the rf-buffer + rf-block + wb-block (Task 5)."""
        raise NotImplementedError("RFactor._emit_rmw lands in Task 5")

    """--- helpers (small, pure) ---"""

    def _owning_matmul_leaf(self, ir: KernelIR, loop_nid: int) -> int | None:
        """The single ISA leaf under ``loop_nid`` whose op is rfactorable, or None."""
        leaves = [
            d
            for d in ir.tree.descendants(loop_nid)
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls.RFACTOR_RECIPE is not None
        ]
        return leaves[0] if len(leaves) == 1 else None

    def _enclosing_block(self, ir: KernelIR, nid: int) -> BlockNode:
        """Nearest enclosing BlockNode payload of ``nid``."""
        for anc in reversed(ir.tree.ancestors(nid)):
            if isinstance(ir.tree.data(anc), BlockNode):
                return ir.tree.data(anc)
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _loop_axis(self, ir: KernelIR, loop_nid: int, block: BlockNode) -> str | None:
        """The concrete axis the loop's loop_var binds, via the block's iter_values."""
        from nkigym.ir.arith.expr import to_affine

        loop_var = ir.tree.data(loop_nid).loop_var
        for iv, value in zip(block.iter_vars, block.iter_values):
            if loop_var in to_affine(value):
                return iv.axis
        return None

    def _role_of(self, block: BlockNode, axis: str) -> AxisRole:
        """Role the block assigns to ``axis`` (default PARALLEL if absent)."""
        for iv in block.iter_vars:
            if iv.axis == axis:
                return iv.role
        return AxisRole.PARALLEL


__all__ = ["RFactor", "RFactorOption"]
```

- [ ] **Step 4: Export from the transforms package**

In `nkigym/src/nkigym/transforms/__init__.py`, add the import (alphabetical, after `Reorder`) and the two `__all__` entries:

```python
from nkigym.transforms.rfactor import RFactor, RFactorOption
```
Add `"RFactor",` and `"RFactorOption",` to `__all__`.

- [ ] **Step 5: Run analyze/legality tests**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/test_rfactor.py -q -k "analyze or rejects or fixture"`
Expected: PASS for `test_fixture_splits_k`, `test_analyze_finds_the_ko_loop`, `test_apply_rejects_non_forNode`, `test_apply_rejects_parallel_loop`.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/rfactor.py nkigym/src/nkigym/transforms/__init__.py test/transforms/test_rfactor.py
git commit -m "feat(transforms): RFactor skeleton + analyze + legality

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `RFactor._emit_rmw` — rf-buffer + rf-block + wb-block, byte-exact

The emission. Co-developed against a hand fixture per the byte-exact workflow
(`.claude/rules/learnings.md`: "gate = `render(apply(before_IR)) == <hand kernel>`").

**Files:**
- Modify: `nkigym/src/nkigym/transforms/rfactor.py` (implement `_emit_rmw`)
- Create: `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` (the expected post-RFactor hand kernel)
- Modify: `test/transforms/test_rfactor.py` (byte-exact + sim + dep-order tests)

**Emission algorithm** (mirrors TVM `RFactorBlockCreator` / `WriteBackBlockCreator`,
spec §3.4). All steps mutate the deep-copied `ir` in place:

1. Resolve from `option.target_loop_nid`: the `ko` ForNode, the matmul leaf
   (`_owning_matmul_leaf`), its enclosing block, the matmul `dst` buffer name
   (the PSUM accumulator) and the op's `REDUCE_COMBINATOR` (`identity`, `combiner`).
2. **rf-buffer**: add a new `Buffer` `B_rf` via the block's `alloc_buffers`, shape =
   the accumulator's logical shape with `factor` prepended at `option.factor_axis`
   (`factor` = `ko` ForNode extent), location = op `OUTPUT_LOCATION` (`"sbuf"`).
3. **rf-block (mutate the existing matmul block)**: set the `ko` `IterVar.role` to
   `PARALLEL`; redirect the matmul `dst` region and the existing drain `tensor_copy`
   to index `B_rf` by `ko` at `factor_axis` (slot-indexed); keep the per-slot
   `memset` (to `identity`) sibling inside `ko`.
4. **wb-block (new sibling block after the rf-block)**: a `memset(out_sbuf, identity)`
   init + a combine leaf `NKITensorTensor(data1=out_sbuf, data2=B_rf[ko], op=combiner)`,
   with the `ko` `IterVar.role` = `ACCUMULATION`. Build via `tree.add_node` of a
   `BlockNode` + the `ko` `ForNode` + the `ISANode`s, spliced after the rf-block
   (`_replace_in_parent_children`). Mirror `canonical_build._build_memset_subblock`
   for the memset sibling.
5. **No reorder** — leave the loop nest order untouched.
6. Tail (exactly as `SoftwarePipeline.apply` / `ComputeAt.apply`): `place_buffers(ir.tree)`,
   then `ir.dependency = Dependency(ir.tree)`.

- [ ] **Step 1: Author the expected hand kernel by rendering once**

Temporarily make `_emit_rmw` a `pass` (no-op) is NOT enough — instead, implement the
algorithm above, then render and inspect:

```python
"""Throwaway scout: render the post-RFactor IR to author the hand fixture. Delete after."""

from nkigym.codegen import render
from nkigym.transforms import RFactor, RFactorOption
from test.transforms._rfactor_fixtures import ko_loop_nid, split_k_ir


def test_dump_rfactor_render() -> None:
    ir = split_k_ir()
    out = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    print(render(out))
    assert True
```

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/_rfactor_render_scout.py -q -s`
Iterate on `_emit_rmw` until the printed kernel matches the spec §3.1 "after" shape
(rf-block fills `B_rf[ko]`, separate wb-block reduces it into the output). Then copy
the printed source verbatim into `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py`
as the function `matmul_lhsT_rhs_nkigym`. Delete the scout file.

- [ ] **Step 2: Write the byte-exact + sim + dep-order tests**

Append to `test/transforms/test_rfactor.py`:

```python
import numpy as np

from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.ir.tree import ForNode
from nkigym.ops.base import AxisRole
from test.transforms._ladder_compare import assert_matches_hand
from test.transforms._rfactor_fixtures import ko_loop_nid, split_k_ir

_HAND = "kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py"


def test_apply_byte_exact() -> None:
    """render(apply(Split→RFactor)) is AST-identical to the hand kernel."""
    ir = split_k_ir()
    out = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    assert_matches_hand(render(out), _HAND, func_name="matmul_lhsT_rhs_nkigym")


def test_apply_sim_matches_matmul() -> None:
    """The rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    ir = split_k_ir()
    out = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    src = render(out)
    rng = np.random.default_rng(0)
    inputs = {"lhs_T": rng.standard_normal((2048, 2048)).astype(np.float32),
              "rhs": rng.standard_normal((2048, 2048)).astype(np.float32)}
    import importlib.util, tempfile, os
    path = os.path.join(tempfile.gettempdir(), "rfactor_sim.py")
    with open(path, "w") as h:
        h.write(src)
    spec = importlib.util.spec_from_file_location("rf", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.matmul_lhsT_rhs_nkigym)(**inputs))
    np.testing.assert_allclose(actual, inputs["lhs_T"].T @ inputs["rhs"], atol=5e-3, rtol=5e-3)


def test_ko_roles_split_across_blocks() -> None:
    """ko is PARALLEL in the rf-block (matmul) and ACCUMULATION in the wb-block."""
    ir = split_k_ir()
    out = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    roles = set()
    for nid in out.tree.blocks():
        block = out.tree.data(nid)
        for iv in block.iter_vars:
            if iv.axis == "d0":
                roles.add(iv.role)
    assert AxisRole.PARALLEL in roles
    assert AxisRole.ACCUMULATION in roles
```

- [ ] **Step 3: Run to verify failure (emission not yet matching)**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/test_rfactor.py -q`
Expected: the byte-exact/sim/roles tests FAIL until `_emit_rmw` is correct (or the
hand fixture is mismatched). Iterate `_emit_rmw` ↔ fixture until green.

- [ ] **Step 4: Implement `_emit_rmw` per the algorithm above**

Replace the `raise NotImplementedError` body in `rfactor.py` with the emission
(steps 1-6 of the algorithm). Use: `tree.add_node(BlockNode(...))`,
`tree.add_node(ForNode(...))`, `tree.add_node(ISANode(...))`,
`_replace_in_parent_children` (from `nkigym.transforms._tree_ops`),
`dataclasses.replace` for buffer/iter_var edits, `place_buffers` + `Dependency` tail.
The hand fixture from Step 1 is the exact target.

- [ ] **Step 5: Run all RFactor tests**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/test_rfactor.py -q`
Expected: PASS (all). If sim passes but byte-exact fails, the IR is behaviorally
right but a region/name differs — fix the emission, not the oracle (per the
byte-exact-means-same learning).

- [ ] **Step 6: Full regression (no new failures)**

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh test/transforms/ test/ops/ test/ir/ -q`
Expected: 0 failed except the 2 pre-existing (`test_dump_tree_runs_on_canonical_ir`,
`test_fuse_tensorize_matmul_n_renders_and_sims`). Any other failure is a regression — fix it.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/transforms/rfactor.py kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py test/transforms/test_rfactor.py
git commit -m "feat(transforms): RFactor emission (rf-buffer + wb-block), byte-exact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Hand-written best-MFU transform sequence (RFactor in the TRACE)

Build on `examples/tune_matmul_lhsT_rhs.py` to hand-author a transform sequence from
canonical that drives toward the best MFU using RFactor. This is a profiling/example
driver, not a unit test — it runs on Trn2 via the example harness and prints MFU.

**Honest scope note:** the *full* 90.92% kernel needs the rf-buffer→fused **fold**
(ComputeAt + compact_shapes) and the `ko`-outside-M hoist — the fold is **parked**
(blocked on narrowing `_check_no_reduction_axis_covered`; spec §7, tvm_knowledge.md).
So this task produces **the best kernel reachable with RFactor + the shipped
transforms today**, and documents the remaining gap. It does NOT claim 90.92% until
the fold lands. If RFactor's multi-slot rf-buffer PSUM/SBUF-OOMs at full size, that
is expected (no capacity gate) — the example profiles only what fits, exactly as the
existing script's docstring explains.

**Files:**
- Create: `examples/tune_matmul_lhsT_rhs_rfactor.py` (a sibling driver; do not break the
  existing `tune_matmul_lhsT_rhs.py`, which encodes the validated 83% TRACE)

- [ ] **Step 1: Copy the existing driver as the starting point**

```bash
cp examples/tune_matmul_lhsT_rhs.py examples/tune_matmul_lhsT_rhs_rfactor.py
```

- [ ] **Step 2: Discover the RFactor target nid in this example's canonical tree**

The example's `f_nkigym` + `INPUT_SPECS` differ from the test fixture, so its nids
differ. Add a temporary print to the new file's `main()` (or a scratch test) that
builds the canonical IR and prints the tree (as in Task 3 Step 1), replay the
example's existing Split atoms first, then read the `ko` ForNode nid. Record it.

Run: `AWS_PROFILE=kaizen-access transport/remote_pytest.sh -q -s` against a scratch
print test, OR run the driver with a debug print on the desktop via
`transport/kaizen.sh --name default --cmd "python examples/tune_matmul_lhsT_rhs_rfactor.py --cache /home/weittang/cache" --cache /home/weittang/cache` and read the dumped tree under the cache dir.

- [ ] **Step 3: Author the RFactor-bearing TRACE**

In `examples/tune_matmul_lhsT_rhs_rfactor.py`, edit the imports to add RFactor:

```python
from nkigym.transforms import (
    ComputeAt, ComputeAtOption, Reorder, ReorderOption, RFactor, RFactorOption,
    SoftwarePipeline, SoftwarePipelineOption, Split, SplitOption, Fuse, ReverseComputeAt,
)
```

Replace the `TRACE` list with a sequence that: (a) `Split`s K into `(ko, ki)`,
(b) applies `RFactor(ko)` (literal nid from Step 2), (c) applies whatever shipped
transforms (Reorder / ComputeAt / SoftwarePipeline) still legally compose on the
post-RFactor tree to maximize overlap. Each entry is `(Transform(), Option(literal_nids))`.
Add `RFactor()` to the `transforms=[...]` list passed to `KernelMDP` in `_validate_trace`
(near line 202) so the replay engine recognizes it:

```python
transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt(), SoftwarePipeline(), RFactor()],
```

Build the TRACE incrementally: after each atom, the harness sim-checks (it raises on
mismatch). Add one atom at a time, re-run, keep only atoms that stay sim-clean.

- [ ] **Step 4: Update the module docstring**

Rewrite the top docstring to describe the RFactor-based sequence honestly: what it
does, the MFU it reaches, and that the rf-buffer→fused fold (for the full SOTA) is
parked. Reference `docs/superpowers/specs/2026-06-07-rfactor-transform-design.md` §7.

- [ ] **Step 5: Run on Trn2 and read MFU**

Run: `AWS_PROFILE=kaizen-access transport/kaizen.sh --name default --cmd "python examples/tune_matmul_lhsT_rhs_rfactor.py --cache /home/weittang/cache" --cache /home/weittang/cache`
Expected: prints the tuned kernel's MFU + the neuronx-cc baseline. Record the number.
If the kernel PSUM/SBUF-OOMs (compile failure), trim the TRACE to the largest tiling
that fits and note the OOM'd variant in the docstring (do not gate it in the transform).

- [ ] **Step 6: Sim-validate locally (the harness already does this) and commit**

The driver sim-checks every rung in `_validate_trace`. Once it runs clean on Trn2:

```bash
git add examples/tune_matmul_lhsT_rhs_rfactor.py
git commit -m "feat(examples): hand-tuned RFactor matmul TRACE + Trn2 MFU profile

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Update learnings with the measured RFactor MFU**

Append a one-line measured result to `.claude/rules/learnings.md` (the Matmul MFU
ladder bullet): the MFU RFactor reaches, and whether it closed or only narrowed the
83.4%→90.92% gap. State plainly if the fold is still the remaining lever.

```bash
git add .claude/rules/learnings.md
git commit -m "docs(learnings): record measured RFactor matmul MFU

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final self-review checklist (run before handoff)

- [ ] Every spec §3–§6 requirement maps to a task (REDUCE_COMBINATOR→T1, NKITensorTensor→T2, fixture→T3, analyze/legality→T4, emission/byte-exact/sim/dep-order→T5).
- [ ] No placeholder nids remain: T3 Step 1 and T6 Step 2 explicitly read real nids before use.
- [ ] Type/name consistency: `RFactorOption(target_loop_nid, factor_axis)`, `ReduceCombinator(combiner, identity)`, `NKITensorTensor(data1, data2, dst, op)`, `B_rf`/`out_sbuf` used consistently across T4/T5.
- [ ] Out-of-scope items (slot recipe, fold, hoist) are referenced but not implemented; T6 documents the parked fold honestly rather than overclaiming 90.92%.
