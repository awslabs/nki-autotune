# Loop-Role Placement Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop `ForNode.loop_type` from the IR and move per-axis role lookup to a per-leaf helper, so future `Reorder` and `compute_at` transforms can answer role questions correctly when one ForNode encloses multiple ISA leaves.

**Architecture:** `ForNode` becomes role-neutral (matches TVM TIR's `For`). Role information is derived per-leaf via a new `role_of(leaf, concrete_dim)` helper that consults `op_cls.AXIS_ROLES + ISANode.axis_map`. `Split` and `Fuse` lose all role checks; their correctness depends only on dim and tree structure (matches TVM Split/Fuse). Visualizer drops the role suffix on loop labels.

**Tech Stack:** Python 3.12, networkx, dataclasses, pytest. Kernel virtualenv at `~/venvs/kernel-env`.

**Spec:** `docs/superpowers/specs/2026-05-18-loop-role-placement-refactor-design.md`

---

## File Plan

| Path | Responsibility |
|---|---|
| `nkigym/src/nkigym/ir/tree.py` | Drop `loop_type` field from `ForNode`; remove role lookup from `_attach_op_subtree`; add `role_of(leaf, dim)` helper |
| `nkigym/src/nkigym/ir/tree_visualize.py` | Drop `data.loop_type.name` from loop-node label |
| `nkigym/src/nkigym/transforms/split.py` | Drop the two `loop_type=` kwargs and the `op_cls.AXIS_ROLES.get(...)` lookup; drop unused `AxisRole` import |
| `nkigym/src/nkigym/transforms/fuse.py` | Drop role-equality + SEQ-rejection from `analyze`, `_check_outer_trip`, `_check_tensorize`, `_do_apply_outer_trip`; drop unused `AxisRole` import |
| `test/codegen/test_body.py` | Drop `loop_type=` kwargs at lines 193, 194, 303; drop `AxisRole` import if unused |
| `test/transforms/test_split.py` | Drop the `loop_type` assertion at line 192 |
| `test/transforms/test_fuse.py` | Update stale docstring on `test_fuse_analyze_no_tensorize_options_on_canonical` |
| `test/ir/test_role_of.py` | NEW — unit tests for `role_of(leaf, dim)` |
| `examples/matmul_lhsT_rhs.py` | Temporarily bump `NUM_ROLLOUTS=32, MAX_STEPS=8` for end-to-end smoke; revert after smoke passes |

---

## Sequencing

Tasks land in this order so each commit leaves the test suite green:

1. **Task 1** — Add `role_of` helper alongside the existing `loop_type` field (tests pass on top of unchanged IR).
2. **Task 2** — Strip role logic from `Split` (no longer reads `loop_type`).
3. **Task 3** — Strip role logic from `Fuse` (no longer reads `loop_type`).
4. **Task 4** — Update `test_body`, `test_split`, `test_fuse` to stop constructing/asserting `loop_type`.
5. **Task 5** — Drop `loop_type` from `ForNode` data class and from `_attach_op_subtree` and visualizer (the field is now dead).
6. **Task 6** — End-to-end smoke via `examples/matmul_lhsT_rhs.py`; revert rollout knobs.

The ordering keeps the IR data-model change (Task 5) at the very end so every intermediate commit compiles.

---

## Common Setup

Every test/run command in this plan assumes the kernel venv is active:

```bash
source ~/venvs/kernel-env/bin/activate
```

`pyright` checks should pass after every task. Run from repo root:

```bash
cd /home/ubuntu/nki-autotune
```

---

### Task 1: Add `role_of` helper

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (append helper near bottom)
- Test: `test/ir/test_role_of.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `test/ir/test_role_of.py`:

```python
"""Unit tests for :func:`nkigym.ir.tree.role_of`."""

from __future__ import annotations

import pytest

from nkigym.ir.tree import ISANode, role_of
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul


def _make_leaf(op_cls: type, axis_map: dict[str, str]) -> ISANode:
    """Build a minimal :class:`ISANode` for role lookup tests.

    Only ``op_cls`` and ``axis_map`` are consulted by :func:`role_of`,
    so the other payload fields are left at their defaults.
    """
    return ISANode(op_cls=op_cls, axis_map=axis_map)


def test_role_of_load_returns_parallel_for_both_axes() -> None:
    """``NKILoad`` does not declare any non-PARALLEL axes."""
    leaf = _make_leaf(NKILoad, {"P": "d0", "F": "d1"})
    assert role_of(leaf, "d0") == AxisRole.PARALLEL
    assert role_of(leaf, "d1") == AxisRole.PARALLEL


def test_role_of_matmul_k_is_accumulation() -> None:
    """``NKIMatmul`` declares ``K`` as ``ACCUMULATION``; ``M`` and ``N`` are ``PARALLEL``."""
    leaf = _make_leaf(NKIMatmul, {"K": "d0", "M": "d1", "N": "d2"})
    assert role_of(leaf, "d0") == AxisRole.ACCUMULATION
    assert role_of(leaf, "d1") == AxisRole.PARALLEL
    assert role_of(leaf, "d2") == AxisRole.PARALLEL


def test_role_of_activation_reduce_f_is_accumulation() -> None:
    """``NKIActivationReduce`` declares ``F`` as ``ACCUMULATION``; ``P`` is ``PARALLEL``."""
    leaf = _make_leaf(NKIActivationReduce, {"P": "d0", "F": "d1"})
    assert role_of(leaf, "d0") == AxisRole.PARALLEL
    assert role_of(leaf, "d1") == AxisRole.ACCUMULATION


def test_role_of_unmapped_dim_raises() -> None:
    """Asking about a dim the leaf does not touch is a loud failure."""
    leaf = _make_leaf(NKILoad, {"P": "d0", "F": "d1"})
    with pytest.raises(KeyError, match="no axis mapping"):
        role_of(leaf, "d99")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/ir/test_role_of.py -v
```

Expected: FAIL with `ImportError: cannot import name 'role_of'`.

- [ ] **Step 3: Add the helper**

Append at the bottom of `nkigym/src/nkigym/ir/tree.py`, immediately before the `__all__` line:

```python
def role_of(leaf: ISANode, concrete_dim: str) -> AxisRole:
    """Return the role this leaf assigns to ``concrete_dim``.

    Walks ``leaf.axis_map`` to find the abstract axis that maps to
    ``concrete_dim``, then consults ``leaf.op_cls.AXIS_ROLES``
    (defaulting to :attr:`AxisRole.PARALLEL` for axes not listed).
    Raises ``KeyError`` if the leaf does not touch ``concrete_dim``.
    """
    matched_role: AxisRole | None = None
    for abstract, dim_id in leaf.axis_map.items():
        if dim_id == concrete_dim:
            matched_role = leaf.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
            break
    if matched_role is None:
        raise KeyError(f"{leaf.op_cls.__name__} has no axis mapping {concrete_dim}")
    return matched_role
```

Update the `__all__` list at the bottom of the file to include `"role_of"`:

```python
__all__ = ["ForNode", "ISANode", "KernelTree", "NodeData", "RootNode", "build_initial_tree", "role_of"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/ir/test_role_of.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the existing IR + transform suites to confirm no regression**

```bash
pytest test/ir test/transforms test/codegen -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_role_of.py
git commit -m "Add role_of(leaf, dim) helper deriving axis role per-leaf"
```

---

### Task 2: Strip role logic from `Split`

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py` (lines 13, 149, 177, 183)

- [ ] **Step 1: Drop the `AxisRole` reads in `_do_apply_outer_trip`**

Open `nkigym/src/nkigym/transforms/split.py`. Locate `_do_apply_outer_trip` (line ~132). Replace the `add_node` call on line 149:

```python
"""Before"""
new_nid = ir.tree.add_node(ForNode(dim=target.dim, trip=trip, loop_type=target.loop_type), parent=None)

"""After"""
new_nid = ir.tree.add_node(ForNode(dim=target.dim, trip=trip), parent=None)
```

- [ ] **Step 2: Drop the `AxisRole` reads in `_do_apply_tensorize`**

In `_do_apply_tensorize` (line ~168). Replace the role lookup and the `add_node` call:

```python
"""Before"""
loop_type = leaf.op_cls.AXIS_ROLES.get(option.target_axis, AxisRole.PARALLEL)
...
new_nid = ir.tree.add_node(ForNode(dim=concrete_dim, trip=trip, loop_type=loop_type), parent=prev)

"""After"""
new_nid = ir.tree.add_node(ForNode(dim=concrete_dim, trip=trip), parent=prev)
```

The `loop_type = ...` line is deleted entirely.

- [ ] **Step 3: Drop the now-unused `AxisRole` import**

At the top of `nkigym/src/nkigym/transforms/split.py`, line ~13:

```python
"""Before"""
from nkigym.ops.base import AxisRole

"""After (delete the line)"""
```

- [ ] **Step 4: Run the Split tests**

```bash
pytest test/transforms/test_split.py -v
```

Expected: every test passes EXCEPT `test_split_tensorize_apply_inserts_outer_for` — that one still asserts on `loop_type`. Don't fix that yet (Task 4 owns the test edit). For now confirm the failure is *only* on the `loop_type` assertion at line ~192.

- [ ] **Step 5: Run pyright on `split.py`**

```bash
pyright nkigym/src/nkigym/transforms/split.py
```

Expected: 0 errors. (The unused-import warning is gone since the import was deleted.)

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/split.py
git commit -m "Strip AxisRole reads from Split; new ForNodes carry only dim+trip"
```

---

### Task 3: Strip role logic from `Fuse`

**Files:**
- Modify: `nkigym/src/nkigym/transforms/fuse.py` (lines 13, 58, 62, 73-74, 82-84, 142-143, 149-153, 179-181, 195-199, 245)

- [ ] **Step 1: Drop role checks in `analyze`'s outer-trip walk**

Open `nkigym/src/nkigym/transforms/fuse.py`. Locate `analyze` (line ~43). The outer-trip chain walk (line ~48-66) has two role-related lines — the chain-break role check (~line 58) and the SEQ early stop (~line 62-63). Edit the loop body:

```python
"""Before (lines ~50-63)"""
if isinstance(data, ForNode):
    chain = [nid]
    cur = nid
    while True:
        kids = ir.tree.children(cur)
        if len(kids) != 1:
            break
        kid_data = ir.tree.data(kids[0])
        if not isinstance(kid_data, ForNode):
            break
        if kid_data.dim != data.dim or kid_data.loop_type != data.loop_type:
            break
        chain.append(kids[0])
        cur = kids[0]
        if data.loop_type == AxisRole.SEQUENTIAL:
            break

"""After"""
if isinstance(data, ForNode):
    chain = [nid]
    cur = nid
    while True:
        kids = ir.tree.children(cur)
        if len(kids) != 1:
            break
        kid_data = ir.tree.data(kids[0])
        if not isinstance(kid_data, ForNode):
            break
        if kid_data.dim != data.dim:
            break
        chain.append(kids[0])
        cur = kids[0]
```

- [ ] **Step 2: Drop role checks in `analyze`'s tensorize walk**

Same `analyze` method, the `elif isinstance(data, ISANode)` branch (line ~69-97). Drop the `expected_role` lookup, the SEQ skip, and the role check on the upward walk:

```python
"""Before (lines ~69-90)"""
elif isinstance(data, ISANode):
    if data.op_cls is NKIAlloc:
        continue
    for axis, dim in data.axis_map.items():
        expected_role = data.op_cls.AXIS_ROLES.get(axis, AxisRole.PARALLEL)
        if expected_role == AxisRole.SEQUENTIAL:
            continue
        chain_above: list[int] = []
        walker = ir.tree.parent(nid)
        prev = nid
        while walker is not None and walker != ir.tree.root:
            wdata = ir.tree.data(walker)
            if not isinstance(wdata, ForNode):
                break
            if wdata.dim != dim or wdata.loop_type != expected_role:
                break
            kids = ir.tree.children(walker)
            if kids != [prev]:
                break
            chain_above.insert(0, walker)
            prev = walker
            walker = ir.tree.parent(walker)

"""After"""
elif isinstance(data, ISANode):
    if data.op_cls is NKIAlloc:
        continue
    for axis, dim in data.axis_map.items():
        chain_above: list[int] = []
        walker = ir.tree.parent(nid)
        prev = nid
        while walker is not None and walker != ir.tree.root:
            wdata = ir.tree.data(walker)
            if not isinstance(wdata, ForNode):
                break
            if wdata.dim != dim:
                break
            kids = ir.tree.children(walker)
            if kids != [prev]:
                break
            chain_above.insert(0, walker)
            prev = walker
            walker = ir.tree.parent(walker)
```

- [ ] **Step 3: Drop role checks in `_check_outer_trip`**

Locate `_check_outer_trip` (line ~134). Drop the SEQ rejection and the chain `loop_type`-equality check:

```python
"""Before (lines ~141-153)"""
first = nodes[0]
if first.loop_type == AxisRole.SEQUENTIAL:
    raise TransformLegalityError("Fuse outer-trip flavor: SEQUENTIAL loops cannot be fused")
for n in nodes[1:]:
    if n.dim != first.dim:
        raise TransformLegalityError(
            f"Fuse outer-trip flavor: all entries must share dim; got {first.dim!r} vs {n.dim!r}"
        )
    if n.loop_type != first.loop_type:
        raise TransformLegalityError(
            f"Fuse outer-trip flavor: all entries must share loop_type; got "
            f"{first.loop_type} vs {n.loop_type}"
        )

"""After"""
first = nodes[0]
for n in nodes[1:]:
    if n.dim != first.dim:
        raise TransformLegalityError(
            f"Fuse outer-trip flavor: all entries must share dim; got {first.dim!r} vs {n.dim!r}"
        )
```

- [ ] **Step 4: Drop role checks in `_check_tensorize`**

Locate `_check_tensorize` (line ~162). Drop the `expected_role` computation, the SEQ rejection, and the prefix-loop-type check:

```python
"""Before (lines ~177-200)"""
concrete_dim = leaf.axis_map[option.target_axis]
expected_role = leaf.op_cls.AXIS_ROLES.get(option.target_axis, AxisRole.PARALLEL)
if expected_role == AxisRole.SEQUENTIAL:
    raise TransformLegalityError("Fuse tensorize flavor: SEQUENTIAL axes cannot be fused")

for_chain_nids = option.target_nids[:-1]
for nid in for_chain_nids:
    data = ir.tree.data(nid)
    if not isinstance(data, ForNode):
        raise TransformLegalityError(
            f"Fuse tensorize flavor: prefix entries must be ForNode; got {type(data).__name__}"
        )
    if data.dim != concrete_dim:
        raise TransformLegalityError(
            f"Fuse tensorize flavor: prefix dim must match leaf axis concrete dim "
            f"({concrete_dim!r}); got {data.dim!r}"
        )
    if data.loop_type != expected_role:
        raise TransformLegalityError(
            f"Fuse tensorize flavor: prefix loop_type must match leaf axis role "
            f"({expected_role}); got {data.loop_type}"
        )

"""After"""
concrete_dim = leaf.axis_map[option.target_axis]

for_chain_nids = option.target_nids[:-1]
for nid in for_chain_nids:
    data = ir.tree.data(nid)
    if not isinstance(data, ForNode):
        raise TransformLegalityError(
            f"Fuse tensorize flavor: prefix entries must be ForNode; got {type(data).__name__}"
        )
    if data.dim != concrete_dim:
        raise TransformLegalityError(
            f"Fuse tensorize flavor: prefix dim must match leaf axis concrete dim "
            f"({concrete_dim!r}); got {data.dim!r}"
        )
```

- [ ] **Step 5: Drop the `loop_type=` kwarg in `_do_apply_outer_trip`**

Locate `_do_apply_outer_trip` (line ~230). Replace the `add_node` call:

```python
"""Before (line ~245)"""
new_nid = ir.tree.add_node(ForNode(dim=first.dim, trip=new_trip, loop_type=first.loop_type), parent=None)

"""After"""
new_nid = ir.tree.add_node(ForNode(dim=first.dim, trip=new_trip), parent=None)
```

- [ ] **Step 6: Drop the now-unused `AxisRole` import**

At the top of `nkigym/src/nkigym/transforms/fuse.py`, line ~13:

```python
"""Before"""
from nkigym.ops.base import AxisRole

"""After (delete the line)"""
```

- [ ] **Step 7: Run the Fuse tests**

```bash
pytest test/transforms/test_fuse.py -v
```

Expected: all green.

- [ ] **Step 8: Run pyright on `fuse.py`**

```bash
pyright nkigym/src/nkigym/transforms/fuse.py
```

Expected: 0 errors.

- [ ] **Step 9: Run the full transform + IR + codegen suites**

```bash
pytest test/transforms test/ir test/codegen -q
```

Expected: every test green except the existing `test_split_tensorize_apply_inserts_outer_for` `loop_type` assertion (Task 4 owns that fix).

- [ ] **Step 10: Commit**

```bash
git add nkigym/src/nkigym/transforms/fuse.py
git commit -m "Strip AxisRole reads from Fuse; legality is structural-only"
```

---

### Task 4: Update tests that construct or assert `loop_type`

**Files:**
- Modify: `test/codegen/test_body.py` (lines 18, 193, 194, 303)
- Modify: `test/transforms/test_split.py` (line 192)
- Modify: `test/transforms/test_fuse.py` (docstring on `test_fuse_analyze_no_tensorize_options_on_canonical`)

- [ ] **Step 1: Drop `loop_type=` kwargs from `test/codegen/test_body.py`**

Line 193:
```python
"""Before"""
outer_nid = tree.add_node(ForNode(dim="d0", trip=8, loop_type=AxisRole.PARALLEL), parent=parent_nid)

"""After"""
outer_nid = tree.add_node(ForNode(dim="d0", trip=8), parent=parent_nid)
```

Line 194:
```python
"""Before"""
inner_nid = tree.add_node(ForNode(dim="d0", trip=2, loop_type=AxisRole.PARALLEL), parent=outer_nid)

"""After"""
inner_nid = tree.add_node(ForNode(dim="d0", trip=2), parent=outer_nid)
```

Line 303:
```python
"""Before"""
tree.graph.nodes[d0_loop_nid]["data"] = ForNode(dim="d0", trip=8, loop_type=AxisRole.PARALLEL)

"""After"""
tree.graph.nodes[d0_loop_nid]["data"] = ForNode(dim="d0", trip=8)
```

- [ ] **Step 2: Drop the now-unused `AxisRole` import from `test/codegen/test_body.py`**

Line 18:
```python
"""Before"""
from nkigym.ops.base import AxisRole

"""After (delete the line)"""
```

- [ ] **Step 3: Drop the `loop_type` assertion from `test/transforms/test_split.py`**

Locate `test_split_tensorize_apply_inserts_outer_for` (line ~169). Drop the assertion at line 192:

```python
"""Before (lines 188-192)"""
new_parent_data = new_ir.tree.data(new_parent)
assert isinstance(new_parent_data, ForNode)
assert new_parent_data.trip == 16
assert new_parent_data.dim == leaf_data.axis_map["F"]
assert new_parent_data.loop_type == AxisRole.PARALLEL

"""After"""
new_parent_data = new_ir.tree.data(new_parent)
assert isinstance(new_parent_data, ForNode)
assert new_parent_data.trip == 16
assert new_parent_data.dim == leaf_data.axis_map["F"]
```

If `AxisRole` is no longer referenced in `test_split.py` after this edit, drop the `from nkigym.ops.base import AxisRole` import on line 10 too:

```bash
grep -n "AxisRole" test/transforms/test_split.py
```

If grep returns no hits beyond the import line, delete the import.

- [ ] **Step 4: Update the stale docstring on `test_fuse_analyze_no_tensorize_options_on_canonical`**

Locate the test in `test/transforms/test_fuse.py` (line ~226). The current docstring cites "matches its op axis role" as the filter reason, which is doubly stale (the role filter is gone after this refactor; even before, the actual filter was `chain_trip_product < 2`). Rewrite:

```python
"""Before (lines 226-231)"""
def test_fuse_analyze_no_tensorize_options_on_canonical():
    """On the canonical IR, no leaf has an enclosing same-dim ForNode that matches its op axis role,
    because canonical trip-1 outer loops are dropped — so tensorize Fuse should yield no options."""
    ir = build_canonical_ir()
    options = [opt for opt in Fuse().analyze(ir) if opt.target_axis is not None]
    assert options == []

"""After"""
def test_fuse_analyze_no_tensorize_options_on_canonical():
    """On the canonical IR, no enclosing same-dim ForNode chain has trip-product >= 2, so
    tensorize Fuse should yield no options."""
    ir = build_canonical_ir()
    options = [opt for opt in Fuse().analyze(ir) if opt.target_axis is not None]
    assert options == []
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest test/transforms test/ir test/codegen -q
```

Expected: every test green.

- [ ] **Step 6: Run pyright on the edited test files**

```bash
pyright test/codegen/test_body.py test/transforms/test_split.py test/transforms/test_fuse.py
```

Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add test/codegen/test_body.py test/transforms/test_split.py test/transforms/test_fuse.py
git commit -m "Drop loop_type kwargs and assertions from tests"
```

---

### Task 5: Drop `loop_type` from `ForNode` and the canonical builder

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (`ForNode` dataclass, `_attach_op_subtree`)
- Modify: `nkigym/src/nkigym/ir/tree_visualize.py` (line 52)

- [ ] **Step 1: Drop the `loop_type` field from `ForNode`**

Open `nkigym/src/nkigym/ir/tree.py`. Replace the `ForNode` dataclass (lines ~47-60):

```python
"""Before"""
@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Trip-loop payload.

    Attributes:
        dim: Concrete dim id (e.g. ``"d0"``).
        trip: Loop trip count (``extent // tile_size``).
        loop_type: Per-op axis classification
            (``PARALLEL`` / ``SEQUENTIAL`` / ``ACCUMULATION``).
    """

    dim: str
    trip: int
    loop_type: AxisRole

"""After"""
@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Trip-loop payload.

    Attributes:
        dim: Concrete dim id (e.g. ``"d0"``).
        trip: Loop trip count (``extent // tile_size``).
    """

    dim: str
    trip: int
```

- [ ] **Step 2: Drop the role lookup in `_attach_op_subtree`**

Locate `_attach_op_subtree` (line ~197). Replace the loop body that creates ForNodes (lines ~229-231):

```python
"""Before"""
if emit_loops:
    role = op.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
    parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile, loop_type=role), parent=parent)

"""After"""
if emit_loops:
    parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile), parent=parent)
```

The `AxisRole` import in `tree.py` stays — `role_of` still uses it.

- [ ] **Step 3: Drop the `loop_type` reference in the visualizer**

Open `nkigym/src/nkigym/ir/tree_visualize.py`. Locate line 52:

```python
"""Before"""
return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}<br/>{data.loop_type.name}"]', "loop")

"""After"""
return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}"]', "loop")
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest test/transforms test/ir test/codegen -q
```

Expected: every test green.

- [ ] **Step 5: Run pyright on the edited files**

```bash
pyright nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/ir/tree_visualize.py
```

Expected: 0 errors.

- [ ] **Step 6: Confirm no `loop_type` references remain in source**

```bash
grep -rn "loop_type" nkigym/ test/
```

Expected: no output (every reference removed).

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/ir/tree_visualize.py
git commit -m "Drop loop_type field from ForNode; ForNode is role-neutral"
```

---

### Task 6: End-to-end smoke via `examples/matmul_lhsT_rhs.py`

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py` (lines 39-40, then revert)

- [ ] **Step 1: Bump rollout knobs**

Open `examples/matmul_lhsT_rhs.py`. Change lines 39-40:

```python
"""Before"""
NUM_ROLLOUTS = 4
MAX_STEPS = 5

"""After"""
NUM_ROLLOUTS = 32
MAX_STEPS = 8
```

- [ ] **Step 2: Run the example**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
```

Expected: every step's `[numerics] PASS` line prints. The script exercises Split + Fuse over the canonical matmul IR for 32 independent rollouts of up to 8 steps each. If any rollout fails numerics or raises `TransformLegalityError`, the refactor changed semantics — stop and investigate before committing.

If the run takes too long to land, lower `NUM_ROLLOUTS` to 16; the goal is widened coverage, not exhaustive search.

- [ ] **Step 3: Revert the rollout knobs**

After the smoke run is clean, revert lines 39-40 back to:

```python
NUM_ROLLOUTS = 4
MAX_STEPS = 5
```

- [ ] **Step 4: Confirm the revert**

```bash
git diff examples/matmul_lhsT_rhs.py
```

Expected: no diff (file matches HEAD).

- [ ] **Step 5: Final full-suite check**

```bash
pytest test -q
```

Expected: every test green.

- [ ] **Step 6: Final grep for stragglers**

```bash
grep -rn "loop_type" nkigym/ test/ examples/
```

Expected: no output.

```bash
grep -rn "AxisRole" nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/transforms/
```

Expected: only references in `tree.py` (used by `role_of` and `ISANode` field defaults — still legitimate).

- [ ] **Step 7: No commit**

This task is verification-only. The earlier 5 commits already capture the refactor.

---

## Self-Review

**Spec coverage:**
- §"Data Model Change" — Task 5, Step 1.
- §"Role Lookup Helper" — Task 1, Step 3.
- §"Canonical Builder Change" — Task 5, Step 2.
- §"Split Change" — Task 2.
- §"Fuse Change" — Task 3.
- §"Visualization Change" — Task 5, Step 3.
- §"Tests" → `test_body.py` — Task 4, Steps 1-2.
- §"Tests" → `test_split.py` — Task 4, Step 3.
- §"Tests" → `test_fuse.py` — Task 4, Step 4.
- §"Tests" → `test_role_of.py` (new) — Task 1, Step 1.
- §"End-to-end smoke" — Task 6.

Every spec section has a task. No gaps.

**Placeholder scan:** No "TBD" / "TODO" / "implement later" / unspecified handling. Every code change shows full before/after. Every command shows expected output.

**Type consistency:** `role_of(leaf, concrete_dim)` signature is identical across spec, helper definition (Task 1 Step 3), and tests (Task 1 Step 1). `ForNode(dim, trip)` (no `loop_type`) signature is identical across all construction sites in Tasks 2, 3, 4, 5.
