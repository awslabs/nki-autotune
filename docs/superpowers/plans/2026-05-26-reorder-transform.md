# Reorder Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `Reorder` as the third `nkigym/transforms/` rewrite — adjacent ForNode payload swap with TVM-aligned legality (PARALLEL/ACCUMULATION descendant leaves only) — so the MDP env can express the matmul nest's loop-order rewrites (e.g. `kernel_transforms.py` k5→k6, k8→k9).

**Architecture:** New `Reorder(Transform)` class in `nkigym/src/nkigym/transforms/reorder.py`. `ReorderOption(TransformOption)` carries `(outer_nid, inner_nid)` — both ForNodes, with inner being outer's sole child. `apply` deep-copies the IR then mutates two `graph.nodes[nid]["data"]` payloads (no topology splice, no nid churn — trivially self-inverse). `_check_legality` raises `TransformLegalityError` on any rule violation; `analyze` enumerates every legal adjacent-pair candidate. The transform is role-aware: descendant ISA leaves are walked and `role_of(leaf, swap_dim) == AxisRole.SEQUENTIAL` rejects (matches TVM `tir.schedule.Reorder`'s `kDataPar | kCommReduce` filter).

**Tech Stack:** Python 3.12, networkx 3.x, pytest (`pythonpath=["nkigym/src"]` configured in `pyproject.toml`), kernel-env venv at `~/venvs/kernel-env/bin/activate`.

**Spec:** `docs/superpowers/specs/2026-05-26-reorder-transform-design.md`

---

## File Plan

| Path | Responsibility |
|---|---|
| `nkigym/src/nkigym/transforms/reorder.py` | NEW — `Reorder`, `ReorderOption`, plus a private `_is_legal` mirroring `Fuse._is_legal` |
| `nkigym/src/nkigym/transforms/__init__.py` | EDIT — re-export `Reorder`, `ReorderOption` |
| `test/transforms/test_reorder.py` | NEW — 11 tests covering analyze / apply / legality / render / role rejection / preservation |
| `test/transforms/_seq_fixture.py` | NEW — synthetic `NKIOp` subclass with one `SEQUENTIAL` axis + builder for an IR enclosing it under two ForNodes (used by the SEQ-rejection test) |
| `test/environment/test_mdp.py` | EDIT — add `test_mdp_with_reorder_random_rollout` |
| `examples/matmul_lhsT_rhs.py` | EDIT — include `Reorder()` in the env's transform list; bump rollout knobs for the smoke run, then revert |

No existing source modules outside `transforms/__init__.py` are modified. The transform is purely additive.

---

## Sequencing

Tasks land in this order so every commit passes the full test suite:

1. **Task 1** — Stub `Reorder` + `ReorderOption` so the import surface compiles (analyze returns `[]`, apply raises `NotImplementedError`).
2. **Task 2** — Implement structural legality (rules 1–3: nids exist, both ForNode, sole-child).
3. **Task 3** — Implement payload-swap apply + `Dependency` rebuild.
4. **Task 4** — Implement role-aware legality (rule 4: descendant-leaf SEQUENTIAL rejection).
5. **Task 5** — Implement `analyze` (enumerate adjacent pairs, filter via `_is_legal`).
6. **Task 6** — Render-equivalence test (lhs_T load K-axis loop appears in the swapped position) + end-to-end fp32 sim.
7. **Task 7** — MDP env integration test.
8. **Task 8** — Example smoke run (`examples/matmul_lhsT_rhs.py`); revert rollout knobs.

Tasks 1–5 each end with a green `pytest` run. Tasks 6–8 are wider regressions.

---

## Common Setup

Every test/run command assumes the kernel venv is active:

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
```

The repo has a pre-commit hook (`check-python-style.py`) that runs on `.py` edits. Use the existing tooling — autoflake, isort, black are wired in.

---

## Task 1: Stub `Reorder` + `ReorderOption` package skeleton

**Files:**
- Create: `nkigym/src/nkigym/transforms/reorder.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`
- Create: `test/transforms/test_reorder.py`

- [ ] **Step 1: Write the failing import test**

Create `test/transforms/test_reorder.py`:

```python
"""Tests for :class:`nkigym.transforms.Reorder`."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Reorder, ReorderOption, TransformLegalityError


def test_reorder_imports_and_instantiates():
    """``Reorder`` and ``ReorderOption`` are public and constructable."""
    t = Reorder()
    opt = ReorderOption(outer_nid=0, inner_nid=1)
    assert opt.outer_nid == 0
    assert opt.inner_nid == 1
    assert t.analyze(build_canonical_ir()) == []
```

- [ ] **Step 2: Run test, expect import failure**

```bash
pytest test/transforms/test_reorder.py::test_reorder_imports_and_instantiates -v
```

Expected: FAIL with `ImportError: cannot import name 'Reorder' from 'nkigym.transforms'`.

- [ ] **Step 3: Write the stub module**

Create `nkigym/src/nkigym/transforms/reorder.py`:

```python
"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode, role_of
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Per-application payload for :class:`Reorder`.

    Attributes:
        outer_nid: nid of the parent :class:`ForNode` to swap.
        inner_nid: nid of its sole :class:`ForNode` child.
    """

    outer_nid: int
    inner_nid: int


class Reorder(Transform):
    """Swap an adjacent parent-child ForNode pair via payload swap.

    See ``docs/superpowers/specs/2026-05-26-reorder-transform-design.md``.
    """

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        """Enumerate every legal adjacent-pair Reorder option."""
        return []

    def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, swap payloads, return new IR."""
        raise NotImplementedError("Reorder.apply lands in Task 3")


__all__ = ["Reorder", "ReorderOption"]
```

- [ ] **Step 4: Update `__init__.py` to re-export**

Modify `nkigym/src/nkigym/transforms/__init__.py`. Replace:

```python
"""Rewrite transforms over :class:`nkigym.ir.KernelIR`."""

from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.fuse import Fuse, FuseOption
from nkigym.transforms.split import Split, SplitOption

__all__ = ["Fuse", "FuseOption", "Split", "SplitOption", "Transform", "TransformLegalityError", "TransformOption"]
```

with:

```python
"""Rewrite transforms over :class:`nkigym.ir.KernelIR`."""

from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.fuse import Fuse, FuseOption
from nkigym.transforms.reorder import Reorder, ReorderOption
from nkigym.transforms.split import Split, SplitOption

__all__ = [
    "Fuse",
    "FuseOption",
    "Reorder",
    "ReorderOption",
    "Split",
    "SplitOption",
    "Transform",
    "TransformLegalityError",
    "TransformOption",
]
```

- [ ] **Step 5: Run test, expect pass**

```bash
pytest test/transforms/test_reorder.py::test_reorder_imports_and_instantiates -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py nkigym/src/nkigym/transforms/__init__.py test/transforms/test_reorder.py
git commit -m "feat: stub Reorder transform + ReorderOption skeleton"
```

---

## Task 2: Structural legality (rules 1–3)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py` (add `_check_legality`)
- Modify: `test/transforms/test_reorder.py`

- [ ] **Step 1: Add the structural-legality tests**

Append to `test/transforms/test_reorder.py`:

```python
def _find_first_for_with_trip(ir, trip: int) -> int:
    """Return the nid of the first :class:`ForNode` with the given trip count."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == trip:
            return nid
    raise AssertionError(f"no ForNode with trip={trip}")


def _find_matmul_outer_pair(ir) -> tuple[int, int]:
    """Return the (K-outer, M-middle) ForNode pair of the matmul nest.

    Canonical IR places the matmul at axis order K, M, N (outermost-first).
    The K-outer ForNode has dim 'd0', trip 16, and exactly one ForNode child
    (the M-middle, dim 'd1', trip 16). Walks pre-order looking for that pair.
    """
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not (isinstance(data, ForNode) and data.dim == "d0" and data.trip == 16):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if not (isinstance(kid_data, ForNode) and kid_data.dim == "d1" and kid_data.trip == 16):
            continue
        return nid, kids[0]
    raise AssertionError("matmul K/M outer ForNode pair not found")


def test_reorder_rejects_unknown_nid():
    """Both nids must exist in the tree."""
    ir = build_canonical_ir()
    outer, _ = _find_matmul_outer_pair(ir)
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=999_999))


def test_reorder_rejects_non_for_target():
    """``inner_nid`` must be a ``ForNode`` (passing an ``ISANode`` raises)."""
    ir = build_canonical_ir()
    outer, _ = _find_matmul_outer_pair(ir)
    isa_nid = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode))
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=isa_nid))


def test_reorder_rejects_non_adjacent():
    """``inner_nid`` must be the sole child of ``outer_nid``."""
    ir = build_canonical_ir()
    outer, middle = _find_matmul_outer_pair(ir)
    """The N-innermost ForNode lives under ``middle``, not directly under ``outer``."""
    n_inner = ir.tree.children(middle)[0]
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=n_inner))


def test_reorder_rejects_outer_with_siblings():
    """``outer_nid`` must have exactly one child (single-element child list)."""
    ir = build_canonical_ir()
    """The matmul middle (M) ForNode has the matmul-N inner as its sole child,
    but the matmul-N inner has the ISA leaf as its sole child. We need a ForNode
    with multiple children to test rejection. Apply Split to introduce such a pair?
    Easier: synthesize the failure by constructing an option pointing at a ForNode
    that does have multiple children in the canonical IR.

    The canonical IR places several blocks (load, load, memset, matmul, copy, store)
    as children of the root, not of any ForNode. So no ForNode has siblings in
    canonical. We construct the test by first applying a Split to create the
    multi-child structure, then attempting Reorder on it."""
    from nkigym.transforms import Split, SplitOption

    target = _find_first_for_with_trip(ir, 16)
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    """``split_ir``'s split chain still has single children. Manufacture
    a multi-child ForNode by direct graph mutation in a copy."""
    import copy

    bad_ir = copy.deepcopy(split_ir)
    """Find any ForNode in bad_ir; add a synthetic ForNode child alongside its real child."""
    for nid in bad_ir.tree.preorder():
        data = bad_ir.tree.data(nid)
        if isinstance(data, ForNode):
            outer_n = nid
            existing_kids = bad_ir.tree.children(outer_n)
            if not existing_kids:
                continue
            inner_n = existing_kids[0]
            """Add a synthetic sibling under outer_n."""
            sibling_nid = bad_ir.tree.add_node(ForNode(dim=data.dim, trip=2), parent=outer_n)
            with pytest.raises(TransformLegalityError):
                Reorder()._check_legality(bad_ir, ReorderOption(outer_nid=outer_n, inner_nid=inner_n))
            return
    raise AssertionError("no ForNode found for sibling-injection test")
```

- [ ] **Step 2: Run tests, expect failures**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 4 new tests fail (no `_check_legality` method yet).

- [ ] **Step 3: Add `_check_legality` (rules 1–3 only) to `reorder.py`**

Insert into `class Reorder` after `apply`:

```python
def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
    """Raise :class:`TransformLegalityError` on any rule violation."""
    """1. Both nids exist."""
    if option.outer_nid not in ir.tree.graph:
        raise TransformLegalityError(
            f"Reorder.outer_nid={option.outer_nid} is not a node in the IR tree"
        )
    if option.inner_nid not in ir.tree.graph:
        raise TransformLegalityError(
            f"Reorder.inner_nid={option.inner_nid} is not a node in the IR tree"
        )
    """2. Both are ForNodes."""
    outer = ir.tree.data(option.outer_nid)
    inner = ir.tree.data(option.inner_nid)
    if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
        raise TransformLegalityError(
            f"Reorder requires both targets to be ForNode; got "
            f"outer={type(outer).__name__}, inner={type(inner).__name__}"
        )
    """3. Inner is the sole child of outer (perfect-nest of two)."""
    kids = ir.tree.children(option.outer_nid)
    if kids != [option.inner_nid]:
        raise TransformLegalityError(
            f"Reorder requires inner_nid={option.inner_nid} to be the sole child of "
            f"outer_nid={option.outer_nid}; got children {kids}"
        )
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 4 structural-legality tests pass; the import test still passes.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py test/transforms/test_reorder.py
git commit -m "feat: Reorder structural legality (nid-exists, ForNode-only, sole-child)"
```

---

## Task 3: Payload-swap apply + self-inverse

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py` (replace `apply` body)
- Modify: `test/transforms/test_reorder.py`

- [ ] **Step 1: Add the apply tests**

Append to `test/transforms/test_reorder.py`:

```python
def test_reorder_apply_swaps_payloads():
    """Applying ``Reorder`` swaps the two ForNode payloads; topology unchanged."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    parent = ir.tree.parent(outer)
    inner_kids_before = ir.tree.children(inner)
    parent_kids_before = ir.tree.children(parent)

    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))

    """outer_nid now carries inner's payload, inner_nid now carries outer's payload."""
    assert new_ir.tree.data(outer).dim == "d1"
    assert new_ir.tree.data(outer).trip == 16
    assert new_ir.tree.data(inner).dim == "d0"
    assert new_ir.tree.data(inner).trip == 16

    """Topology is bitwise unchanged."""
    assert new_ir.tree.children(parent) == parent_kids_before
    assert new_ir.tree.children(outer) == [inner]
    assert new_ir.tree.children(inner) == inner_kids_before


def test_reorder_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    snapshot_payloads = {nid: ir.tree.data(nid) for nid in ir.tree.preorder()}

    Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))

    for nid, data in snapshot_payloads.items():
        assert ir.tree.data(nid) == data, f"payload at nid={nid} mutated"


def test_reorder_self_inverse():
    """Applying the same ``Reorder`` twice restores the original payloads."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    option = ReorderOption(outer_nid=outer, inner_nid=inner)

    once = Reorder().apply(ir, option)
    twice = Reorder().apply(once, option)

    """Twice-applied IR has the same payload at every nid as the original."""
    original_nids = list(ir.tree.preorder())
    twice_nids = list(twice.tree.preorder())
    assert original_nids == twice_nids
    for nid in original_nids:
        assert ir.tree.data(nid) == twice.tree.data(nid)
```

- [ ] **Step 2: Run tests, expect failures**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 3 new tests fail (`Reorder.apply` raises `NotImplementedError`).

- [ ] **Step 3: Replace `apply` body in `reorder.py`**

Replace the `apply` method body in `class Reorder`:

```python
def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
    """Re-check legality, deep-copy ``ir``, swap the two payloads, return."""
    self._check_legality(ir, option)
    new_ir = copy.deepcopy(ir)
    outer_data = new_ir.tree.data(option.outer_nid)
    inner_data = new_ir.tree.data(option.inner_nid)
    new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
    new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
    new_ir.dependency = Dependency(new_ir.tree)
    return new_ir
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 7 tests pass (1 import + 4 structural + 3 apply).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py test/transforms/test_reorder.py
git commit -m "feat: Reorder.apply via payload swap; self-inverse + deep-copy"
```

---

## Task 4: Role-aware legality (SEQUENTIAL rejection)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py` (extend `_check_legality`)
- Create: `test/transforms/_seq_fixture.py`
- Modify: `test/transforms/test_reorder.py`

- [ ] **Step 1: Build the synthetic SEQUENTIAL fixture**

Create `test/transforms/_seq_fixture.py`:

```python
"""Synthetic ``NKIOp`` with one ``SEQUENTIAL`` axis for Reorder legality tests.

No production op carries SEQUENTIAL today (matmul has ACCUMULATION on K;
RMSNorm/softmax prefix-scan ops will, when they ship). To verify the
legality rule before those ops land, we declare a minimal subclass and
build a hand-rolled IR that encloses one such leaf under two ForNodes
on its mapped dims.
"""

from __future__ import annotations

from typing import ClassVar

import networkx as nx

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode, KernelTree
from nkigym.ops.base import AxisRole, NKIOp


class _SeqOp(NKIOp):
    """Minimal NKIOp with one PARALLEL ('P') and one SEQUENTIAL ('F') axis."""

    NAME: ClassVar[str] = "_seq_op_test"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}

    def _run(self, **kwargs):
        """Concrete stub so the class is instantiable; never called by legality checks."""
        return None


def build_seq_ir(p_extent: int = 256, f_extent: int = 256) -> tuple[KernelIR, int, int, int]:
    """Build a minimal IR: ``RootNode → ForNode(d0, trip=2) → ForNode(d1, trip=2) → _SeqOp leaf``.

    The leaf reads tensor 'x' with axis_map {'P': 'd0', 'F': 'd1'} so its
    SEQUENTIAL role on 'F' resolves to dim 'd1'. Returns ``(ir, outer_nid,
    inner_nid, leaf_nid)``. ``KernelIR``'s required dataclass fields
    (`func_name`, `param_names`, `return_name`) are populated with stubs;
    the legality check reads only ``ir.tree``.
    """
    tree = KernelTree()
    outer = tree.add_node(ForNode(dim="d0", trip=2), parent=tree.root)
    inner = tree.add_node(ForNode(dim="d1", trip=2), parent=outer)
    """Build the leaf payload. tensorize_sizes covers each axis (trip * tensorize == extent)."""
    leaf_data = ISANode(
        op_cls=_SeqOp,
        reads=("x",),
        writes=(),
        rmw=(),
        tensorize_sizes={"P": p_extent // 2, "F": f_extent // 2},
        axis_map={"P": "d0", "F": "d1"},
        kwargs={},
    )
    leaf_nid = tree.add_node(leaf_data, parent=inner)

    ir = KernelIR(
        func_name="_seq_fixture",
        param_names=[],
        return_name="",
        dim_sizes={"d0": p_extent, "d1": f_extent},
        tensors={},
        tree=tree,
        dependency=Dependency(tree),
    )
    return ir, outer, inner, leaf_nid
```

> **Note:** `KernelIR` is a `@dataclass` with required positional fields
> (`func_name`, `param_names`, `return_name`, `dim_sizes`, `tensors`,
> `tree`, `dependency`) — see `nkigym/src/nkigym/ir/ir.py:28-48`. Populate
> them with stubs; the Reorder legality check only reads `ir.tree`.
> Do NOT modify `KernelIR` itself.

- [ ] **Step 2: Add the role-rejection test**

Append to `test/transforms/test_reorder.py`:

```python
def test_reorder_rejects_sequential_role():
    """A leaf with SEQUENTIAL role on either swap dim must reject the reorder."""
    from test.transforms._seq_fixture import build_seq_ir

    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=inner))


def test_reorder_allows_accumulation_parallel():
    """K/M (ACCUM/PARALLEL) and M/N (PARALLEL/PARALLEL) must pass legality."""
    ir = build_canonical_ir()
    """K-outer / M-middle is ACCUM (K) and PARALLEL (M)."""
    outer_km, inner_km = _find_matmul_outer_pair(ir)
    Reorder()._check_legality(ir, ReorderOption(outer_nid=outer_km, inner_nid=inner_km))

    """M-middle / N-inner is PARALLEL (M) and PARALLEL (N)."""
    n_inner = ir.tree.children(inner_km)[0]
    Reorder()._check_legality(ir, ReorderOption(outer_nid=inner_km, inner_nid=n_inner))
```

- [ ] **Step 3: Run tests, expect SEQUENTIAL test to fail**

```bash
pytest test/transforms/test_reorder.py::test_reorder_rejects_sequential_role -v
pytest test/transforms/test_reorder.py::test_reorder_allows_accumulation_parallel -v
```

Expected: SEQUENTIAL rejection fails (no role check yet); ACCUM/PARALLEL allow passes.

- [ ] **Step 4: Extend `_check_legality` with rule 4**

Append to `_check_legality` in `reorder.py`:

```python
"""4. No descendant ISA leaf has SEQUENTIAL role on either swapped dim."""
for leaf_nid in ir.tree.leaves(option.inner_nid):
    leaf = ir.tree.data(leaf_nid)
    if not isinstance(leaf, ISANode) or leaf.op_cls is NKIAlloc:
        continue
    leaf_dims = set(leaf.axis_map.values())
    for swap_dim in (outer.dim, inner.dim):
        if swap_dim in leaf_dims and role_of(leaf, swap_dim) == AxisRole.SEQUENTIAL:
            raise TransformLegalityError(
                f"Reorder rejected: leaf {leaf.op_cls.__name__} has SEQUENTIAL role "
                f"on dim {swap_dim!r}"
            )
```

- [ ] **Step 5: Run all reorder tests, expect pass**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 9 tests pass (1 import + 4 structural + 3 apply + 2 role).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py test/transforms/test_reorder.py test/transforms/_seq_fixture.py
git commit -m "feat: Reorder rejects descendant SEQUENTIAL role (matches TVM kDataPar|kCommReduce)"
```

---

## Task 5: `analyze` enumerates legal adjacent pairs

**Files:**
- Modify: `nkigym/src/nkigym/transforms/reorder.py` (replace `analyze`)
- Modify: `test/transforms/test_reorder.py`

- [ ] **Step 1: Add the analyze tests**

Append to `test/transforms/test_reorder.py`:

```python
def test_reorder_analyze_canonical_matmul():
    """Canonical matmul nest yields at least the K/M and M/N adjacent ForNode pairs."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    pairs = {(opt.outer_nid, opt.inner_nid) for opt in options}

    """K/M outer pair must be present."""
    km_outer, km_inner = _find_matmul_outer_pair(ir)
    assert (km_outer, km_inner) in pairs

    """M/N pair must also be present (M-middle has the matmul N-inner as sole child)."""
    n_inner = ir.tree.children(km_inner)[0]
    assert (km_inner, n_inner) in pairs


def test_reorder_analyze_skips_single_for_subtrees():
    """Subtrees whose only ForNode has no ForNode child (loads, stores, memset)
    must not surface options."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    pairs = {(opt.outer_nid, opt.inner_nid) for opt in options}

    """The lhs_T load nest is a single ForNode → ISANode chain (after the
    canonical IR builds a per-axis chain, but with trip-1 ForNodes on
    axes whose MAX_TILE_SIZE is None, see `tree.py:_attach_op_subtree`).
    Walk every ForNode; ensure no option pairs a ForNode with an ISA child."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) == 1 and isinstance(ir.tree.data(kids[0]), ISANode):
            assert (nid, kids[0]) not in pairs


def test_reorder_analyze_returns_only_legal_options():
    """Every option ``analyze`` returns must apply without raising."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    assert options, "expected at least one Reorder option on the canonical IR"
    for opt in options:
        Reorder().apply(ir, opt)
```

- [ ] **Step 2: Run tests, expect failures**

```bash
pytest test/transforms/test_reorder.py -v -k analyze
```

Expected: 3 analyze tests fail (analyze still returns `[]`).

- [ ] **Step 3: Replace `analyze` body**

Replace `analyze` in `reorder.py`:

```python
def analyze(self, ir: KernelIR) -> list[ReorderOption]:
    """Enumerate every legal adjacent-pair ForNode swap."""
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

def _is_legal(self, ir: KernelIR, option: ReorderOption) -> bool:
    """Wrapper around :meth:`_check_legality` that returns a bool.

    Used by :meth:`analyze` to filter candidate options without raising.
    Production-path callers must use :meth:`_check_legality` directly so
    illegal options raise loudly. Mirrors :meth:`Fuse._is_legal`.
    """
    legal = True
    try:
        self._check_legality(ir, option)
    except TransformLegalityError:
        legal = False
    return legal
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest test/transforms/test_reorder.py -v
```

Expected: 12 tests pass (1 import + 4 structural + 3 apply + 2 role + 3 analyze).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/reorder.py test/transforms/test_reorder.py
git commit -m "feat: Reorder.analyze enumerates legal adjacent ForNode pairs"
```

---

## Task 6: Render-equivalence + end-to-end fp32 sim

**Files:**
- Modify: `test/transforms/test_reorder.py`

- [ ] **Step 1: Add the render + sim tests**

Append to `test/transforms/test_reorder.py`:

```python
def test_reorder_render_swap_visible():
    """Rendering the Reorder-applied IR shows ``i_d1_0`` outside ``i_d0_0`` in the matmul nest."""
    from nkigym.codegen import render

    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    src = render(new_ir)

    """The matmul nest in the rendered source must put i_d1_0 (M, was inner)
    above i_d0_0 (K, was outer)."""
    matmul_pos = src.find("nisa.nc_matmul")
    assert matmul_pos != -1
    """Walk backward from the matmul call; the immediate enclosing loop on the
    matmul nest is now i_d1_0 (M-axis), and the next enclosing one is i_d0_0 (K-axis).
    Find the K loop offset and the M loop offset; M must come BEFORE K (smaller offset)."""
    k_pos = src.rfind("for i_d0_0 in range(16):", 0, matmul_pos)
    m_pos = src.rfind("for i_d1_0 in range(16):", 0, matmul_pos)
    assert k_pos != -1 and m_pos != -1, f"K loop at {k_pos}, M loop at {m_pos}"
    assert m_pos < k_pos, f"after Reorder, M loop ({m_pos}) must precede K loop ({k_pos}) above matmul"


def test_reorder_round_trip_render_sim():
    """End-to-end: Reorder → render → fp32 sim → matches numpy golden."""
    import importlib.util
    import os
    import shutil
    import tempfile

    import numpy as np

    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    src = render(new_ir)

    """Write the rendered kernel to a temp path and import it as a module."""
    tmpdir = tempfile.mkdtemp()
    try:
        kernel_path = os.path.join(tmpdir, "kernel.py")
        with open(kernel_path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("dumped_reorder_kernel", kernel_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        K, M, N = 2048, 2048, 2048
        rng = np.random.default_rng(0)
        lhs_T = rng.standard_normal((K, M)).astype(np.float32)
        rhs = rng.standard_normal((K, N)).astype(np.float32)
        expected = lhs_T.T @ rhs
        actual = np.asarray(simulate_fp32(module.nki_f_matmul)(lhs_T, rhs))
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

> **Note on the kernel function name:** the rendered kernel's top-level
> function is `nki_<f_name>` where `<f_name>` is the original `@nkigym_kernel`
> callable's `__name__`. The fixture's callable is `f_matmul`, so the rendered
> function is `nki_f_matmul`. Verify by inspecting `emit_header.py:_emit_signature`
> if the assertion fails — the prefix is consistent across the codebase per
> `examples/matmul_lhsT_rhs.py`'s use of `module.nki_f_nkigym`.

- [ ] **Step 2: Run tests**

```bash
pytest test/transforms/test_reorder.py::test_reorder_render_swap_visible test/transforms/test_reorder.py::test_reorder_round_trip_render_sim -v
```

Expected: PASS. The fp32 sim test takes ~3-5 seconds (2048³ matmul under CPU sim).

- [ ] **Step 3: Commit**

```bash
git add test/transforms/test_reorder.py
git commit -m "test: Reorder render-equivalence + end-to-end fp32 sim against numpy golden"
```

---

## Task 7: MDP env integration test

**Files:**
- Modify: `test/environment/test_mdp.py`

- [ ] **Step 1: Add the rollout test**

Append to `test/environment/test_mdp.py`:

```python
def test_mdp_with_reorder_random_rollout():
    """A random rollout with [Split, Fuse, Reorder] runs without raising."""
    import random

    from nkigym.transforms import Reorder

    rng = random.Random(42)
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder()])
    state = env.reset()

    """5 random steps; legality is checked inside step (loud failure on illegal)."""
    for _ in range(5):
        actions = env.legal_actions(state)
        if not actions:
            break
        action = rng.choice(actions)
        state = env.step(state, action)
        """Confirm the resulting state is itself legal: legal_actions runs analyze
        across all transforms, which exercises tree-walk over the new IR."""
        env.legal_actions(state)


def test_mdp_legal_actions_includes_reorder_options_on_canonical():
    """``legal_actions`` on the canonical IR must include at least one Reorder action."""
    from nkigym.transforms import Reorder

    reorder = Reorder()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), reorder])
    state = env.reset()
    actions = env.legal_actions(state)
    reorder_actions = [(tr, opt) for tr, opt in actions if tr is reorder]
    assert reorder_actions, "expected at least one Reorder action on canonical IR"
```

- [ ] **Step 2: Run tests, expect pass**

```bash
pytest test/environment/test_mdp.py -v
```

Expected: all existing MDP tests pass + 2 new Reorder tests.

- [ ] **Step 3: Commit**

```bash
git add test/environment/test_mdp.py
git commit -m "test: MDP env integration with Reorder transform"
```

---

## Task 8: Example smoke run

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py`

- [ ] **Step 1: Update the example to include Reorder + bump rollout knobs**

Modify `examples/matmul_lhsT_rhs.py`:

Replace:

```python
from nkigym.transforms import Fuse, Split

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}
NUM_ROLLOUTS = 4
MAX_STEPS = 5
SEED = 42
```

with:

```python
from nkigym.transforms import Fuse, Reorder, Split

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}
NUM_ROLLOUTS = 8
MAX_STEPS = 6
SEED = 42
```

And replace:

```python
env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse()])
```

with:

```python
env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder()])
```

- [ ] **Step 2: Run the example**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
```

Expected: every rollout step prints `[numerics] PASS (atol=0.005, rtol=0.005)`. Rendered kernels live under `/home/ubuntu/cache/matmul_lhsT_rhs/rollout_<k>/step_<t>/`.

- [ ] **Step 3: Revert the rollout knobs**

After the smoke run is clean, revert `NUM_ROLLOUTS` and `MAX_STEPS` to their pre-smoke values:

```python
NUM_ROLLOUTS = 4
MAX_STEPS = 5
```

(Leave `transforms=[Split(), Fuse(), Reorder()]` and the `Reorder` import.)

- [ ] **Step 4: Re-run the example to confirm reverted knobs still work**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
```

Expected: every step still passes; total runtime drops back to ~30-40 seconds.

- [ ] **Step 5: Commit**

```bash
git add examples/matmul_lhsT_rhs.py
git commit -m "feat: include Reorder in matmul_lhsT_rhs example MDP transform list"
```

---

## Verification

After Task 8, run the full test suite to confirm nothing regressed:

```bash
pytest -v
```

Expected: every test passes. Spot-check the new tests:

```bash
pytest test/transforms/test_reorder.py -v --tb=short
pytest test/environment/test_mdp.py -v --tb=short
```

Inspect a dumped kernel to confirm Reorder shows up in MDP rollouts:

```bash
ls /home/ubuntu/cache/matmul_lhsT_rhs/rollout_0/
```

Expect `step_1/`, `step_2/`, etc., each with `kernel.py`, `ir.md`, `tree.png`.

Compare a Reorder-applied step's `kernel.py` against the canonical `step_0` to confirm the loop nesting differs in the matmul region — exact form depends on the random rollout, but at least one of the 8 rollouts should hit a Reorder action given the action space's size.

---

## Out of Scope

Per the spec:

- TVM-style permutation-list reorder (any permutation of N≥2 ForNodes on a chain). Composition of adjacent swaps reaches every permutation.
- Reorder across SEQUENTIAL boundaries with a more elaborate dependency model.
- Reorder over `compute_at`-sunk producer-consumer subtrees (`compute_at` lands first).
- Cross-block reorder (siblings under the tree root).
