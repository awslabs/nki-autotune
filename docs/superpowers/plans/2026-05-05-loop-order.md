# Loop Reorder Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ReorderLoops` — a local-perfect-nest loop interchange rewrite — plus `LoopNode.reduce_op` and forest-state hashing, and wire them into the `tune` stage.

**Architecture:** Four-phase build on top of the existing `LoopForest` IR: (A) add `reduce_op` to `LoopNode` and populate it at canonical-forest build time; (B) add forest-state hashing utilities; (C) add the `ReorderLoops` atom + enumerator; (D) extend the `tune` stage's random-draw loop to include reorder atoms and use the hash to break self-inverse cycles.

**Tech Stack:** Python 3.12, dataclasses, `nki` CPU simulator, pytest.

**Spec:** `docs/superpowers/specs/2026-05-05-loop-order-design.md`.

---

## File Structure

### Files to modify

- `nkigym/src/nkigym/codegen/loop_forest.py` — extend `LoopNode` with `reduce_op`; populate in matmul/activation_reduce leaf builders; add `_resolve_node`, `_canonical_key`, `hash_forest` utilities.
- `nkigym/src/nkigym/tune/stage.py` — include reorder atoms in the random draw; add hash-based cycle break.
- `test/codegen/test_loop_forest.py` — add reduce_op population + `hash_forest` tests.
- `test/codegen/test_compile.py` — add tune-stage integration tests (reorder-only, reorder+fuse, cycle termination).

### Files to create

- `nkigym/src/nkigym/tune/reorder_loops.py` — `ReorderLoops` atom, `_roles_commute`, `enumerate_reorder_atoms`.
- `test/codegen/test_reorder_loops.py` — Layer-2 tree-level tests mirroring `test_tune.py`'s `FuseLoops` section.

### One-file-per-responsibility

The new atom module mirrors `fuse_loops.py`'s shape (atom + enumerator + module-local `_rewrite_forest_at`-style helpers). Do not unify the two rewrites into a single module — the spec's `KernelRewrite` protocol lets them live independently, and future atoms (hoist, distribute) will follow the same per-rewrite layout.

---

## Phase A — `LoopNode.reduce_op`

### Task A1: Extend `LoopNode` with `reduce_op`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py:33-52`
- Test: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_loop_node_reduce_op_defaults_to_none() -> None:
    """LoopNode without an explicit reduce_op defaults to None."""
    from nkigym.codegen.loop_forest import LoopNode
    from nkigym.ops.base import AxisRole

    node = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.PARALLEL)
    assert node.reduce_op is None


def test_loop_node_reduce_op_accepts_explicit_value() -> None:
    """LoopNode can be constructed with reduce_op='add' for ACC loops."""
    from nkigym.codegen.loop_forest import LoopNode
    from nkigym.ops.base import AxisRole

    node = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.ACCUMULATION, reduce_op="add")
    assert node.reduce_op == "add"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_loop_forest.py::test_loop_node_reduce_op_defaults_to_none test/codegen/test_loop_forest.py::test_loop_node_reduce_op_accepts_explicit_value -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'reduce_op'` or attribute-missing error.

- [ ] **Step 3: Add the field after `children` (preserves positional callers)**

The existing codebase has many 4-positional call sites like
`LoopNode("d0", 4, AxisRole.PARALLEL, [child])` — the 4th positional
is `children`. Insert `reduce_op` **after** `children`, not between
`role` and `children`, so those call sites stay valid.

Edit `nkigym/src/nkigym/codegen/loop_forest.py`. Replace the current `LoopNode` definition (around lines 33–52) with:

```python
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
        reduce_op: Reducer name for ACCUMULATION loops (``"add"``,
            ``"max"``, ...). ``None`` for PARALLEL / SEQUENTIAL loops.
            Used by :class:`ReorderLoops` to detect associative-
            compatible ACC×ACC swaps.
    """

    dim_id: str
    trip_count: int
    role: AxisRole
    children: "list[LoopNode | BodyLeaf]" = field(default_factory=list)
    reduce_op: str | None = None
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_loop_forest.py -v
```

Expected: new tests PASS, all existing tests PASS — existing positional calls `LoopNode("d0", 4, AxisRole.PARALLEL, [...])` bind to the first four fields unchanged; `reduce_op` defaults to `None`.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Add LoopNode.reduce_op field for ACC associativity checks"
```

### Task A2: Populate `reduce_op` in matmul leaf builder

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py:210-224` (`_build_leaves_matmul`)
- Test: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_canonical_forest_matmul_k_loops_carry_reduce_op_add() -> None:
    """Matmul's K-chain LoopNodes (block + tile) carry reduce_op='add'."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    n_tile = tree.children[0].children[0].children[0]
    k_block = n_tile.children[1]
    assert isinstance(k_block, LoopNode)
    assert k_block.reduce_op == "add"
    k_tile = k_block.children[0]
    assert isinstance(k_tile, LoopNode)
    assert k_tile.reduce_op == "add"


def test_canonical_forest_matmul_m_n_loops_have_no_reduce_op() -> None:
    """Matmul's M and N loops are PARALLEL and carry reduce_op=None."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    m_block = tree
    m_tile = m_block.children[0]
    n_block = m_tile.children[0]
    n_tile = n_block.children[0]
    for node in (m_block, m_tile, n_block, n_tile):
        assert isinstance(node, LoopNode)
        assert node.reduce_op is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_loop_forest.py::test_canonical_forest_matmul_k_loops_carry_reduce_op_add -v
```

Expected: FAIL — `k_block.reduce_op` is `None` (default).

- [ ] **Step 3: Populate `reduce_op="add"` in matmul's K-chain**

Edit `nkigym/src/nkigym/codegen/loop_forest.py` — replace `_build_leaves_matmul` (around lines 210–224) with:

```python
def _build_leaves_matmul(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """Matmul: ``[psum_init leaf, <K chain ending in compute leaf>, drain leaf]``.

    The outer M and N dims are consumed by ``_wrap_dims``. The K dim is
    handled here so the body placement mirrors the physical kernel:
    PSUM init lives outside K, ``nc_matmul`` fires inside K, drain
    runs after K closes. The K-chain LoopNodes carry ``reduce_op="add"``
    because nc_matmul's PSUM accumulator is summation.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = op_graph.dims[k_dim].num_tiles
    compute_leaf = BodyLeaf(op_idx=op.idx, phase="compute")
    k_tile = LoopNode(dim_id=k_dim, trip_count=1, role=k_role, children=[compute_leaf], reduce_op="add")
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[k_tile], reduce_op="add")
    return [BodyLeaf(op_idx=op.idx, phase="psum_init"), k_block, BodyLeaf(op_idx=op.idx, phase="drain")]
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_loop_forest.py -v
```

Expected: PASS (both new tests + all existing tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Populate reduce_op='add' on matmul K-chain in canonical forest"
```

### Task A3: Populate `reduce_op` in activation_reduce leaf builder

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py:247-266` (`_build_leaves_activation_reduce`)
- Test: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_canonical_forest_activation_reduce_f_loops_carry_kwargs_reduce_op() -> None:
    """ActivationReduce F-chain LoopNodes carry reduce_op from op_kwargs."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    f_block = p_tile.children[1]
    assert isinstance(f_block, LoopNode)
    assert f_block.reduce_op == "add"
    f_tile = f_block.children[0]
    assert isinstance(f_tile, LoopNode)
    assert f_tile.reduce_op == "add"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_loop_forest.py::test_canonical_forest_activation_reduce_f_loops_carry_kwargs_reduce_op -v
```

Expected: FAIL — `f_block.reduce_op` is `None`.

- [ ] **Step 3: Populate `reduce_op` from `op.op_kwargs["reduce_op"]`**

Edit `nkigym/src/nkigym/codegen/loop_forest.py` — replace `_build_leaves_activation_reduce` (around lines 247–266) with:

```python
def _build_leaves_activation_reduce(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce: ``[reducer_init leaf, <F chain ending in reduce_step leaf>, post_op leaf?]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: an ``F-block / F-tile / BodyLeaf(reduce_step)`` chain sits
    between ``reducer_init`` and the optional ``post_op`` leaf. The
    F-chain LoopNodes carry the reducer name from
    ``op.op_kwargs["reduce_op"]`` so :class:`ReorderLoops` can detect
    associative-compatible ACC×ACC pairs. The ``post_op`` leaf is
    included only when ``op.op_kwargs["post_op"]`` is not ``None``.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile], reduce_op=reduce_op)
    leaves: list[LoopNode | BodyLeaf] = [BodyLeaf(op_idx=op.idx, phase="reducer_init"), f_block]
    if op.op_kwargs.get("post_op") is not None:
        leaves.append(BodyLeaf(op_idx=op.idx, phase="post_op"))
    return leaves
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_loop_forest.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Populate reduce_op on activation_reduce F-chain from op_kwargs"
```

---

## Phase B — Forest hashing utilities

### Task B1: Add `_canonical_key` + `hash_forest`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py` (append at end of file)
- Test: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_hash_forest_is_deterministic_across_invocations() -> None:
    """Two calls on the same forest produce the same hash."""
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest

    g = _parse(_rmsnorm_matmul, _RMSNORM_MATMUL_SPECS)
    forest = build_canonical_forest(g)
    assert hash_forest(forest) == hash_forest(forest)


def test_hash_forest_equals_for_independently_built_canonical_forests() -> None:
    """Two independent canonical forests of the same op_graph hash equal."""
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest

    g = _parse(_rmsnorm_matmul, _RMSNORM_MATMUL_SPECS)
    f1 = build_canonical_forest(g)
    f2 = build_canonical_forest(g)
    assert hash_forest(f1) == hash_forest(f2)


def test_hash_forest_distinguishes_trip_count_change() -> None:
    """Forests differing only in a trip count hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    f2 = [LoopNode("d0", 8, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_dim_id_change() -> None:
    """Forests differing only in dim_id at a node hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    f2 = [LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_reduce_op_change() -> None:
    """Forests differing only in reduce_op hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")]
    f2 = [LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="max")]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_leaf_phase_change() -> None:
    """Leaves differing only in phase hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="a")])]
    f2 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="b")])]
    assert hash_forest(f1) != hash_forest(f2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_loop_forest.py -k hash_forest -v
```

Expected: FAIL with `ImportError: cannot import name 'hash_forest'`.

- [ ] **Step 3: Implement `_canonical_key` and `hash_forest`**

Append to `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
def _canonical_key(node: "LoopNode | BodyLeaf") -> tuple:
    """Recursive structural key for a node.

    Two nodes (and their subtrees) produce equal keys iff they have the
    same tree shape, dim_ids, trip counts, roles, reduce_ops, and leaf
    op_idx / phase tags.
    """
    if isinstance(node, BodyLeaf):
        return ("leaf", node.op_idx, node.phase)
    return (
        "node",
        node.dim_id,
        node.trip_count,
        node.role.value,
        node.reduce_op,
        tuple(_canonical_key(c) for c in node.children),
    )


def hash_forest(forest: LoopForest) -> int:
    """Return a deterministic structural hash of ``forest``.

    Used by the ``tune`` stage's random-draw loop to break cycles
    caused by self-inverse rewrites (e.g. ``ReorderLoops`` applied
    twice restores the prior state).

    Covers only the forest — current structural rewrites leave
    ``op_graph`` untouched. Once graph rewrites land, extend the hash
    to include ``op_graph``.
    """
    return hash(tuple(_canonical_key(e) for e in forest))
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_loop_forest.py -k hash_forest -v
```

Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Add hash_forest for structural forest-state equality"
```

### Task B2: Add `_resolve_node` helper

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py` (append)
- Test: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_loop_forest.py`:

```python
def test_resolve_node_returns_forest_root_entry_for_single_index_path() -> None:
    """path=(0,) returns forest[0]."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    node_a = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    node_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    forest = [node_a, node_b]
    assert _resolve_node(forest, (0,)) is node_a
    assert _resolve_node(forest, (1,)) is node_b


def test_resolve_node_walks_nested_children() -> None:
    """path=(0, 0) returns forest[0].children[0]."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    leaf = BodyLeaf(op_idx=0)
    inner = LoopNode("d1", 1, AxisRole.PARALLEL, [leaf])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    assert _resolve_node(forest, (0, 0)) is inner
    assert _resolve_node(forest, (0, 0, 0)) is leaf


def test_resolve_node_returns_none_for_empty_path() -> None:
    """Empty path is invalid (no target node)."""
    from nkigym.codegen.loop_forest import _resolve_node

    assert _resolve_node([], ()) is None


def test_resolve_node_returns_none_for_out_of_range_index() -> None:
    """Path index beyond children length returns None."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert _resolve_node(forest, (99,)) is None
    assert _resolve_node(forest, (0, 99)) is None


def test_resolve_node_returns_none_when_traversing_through_body_leaf() -> None:
    """Can't descend into a BodyLeaf's 'children'; returns None."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    leaf = BodyLeaf(op_idx=0)
    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [leaf])]
    """path=(0, 0, 0) would try to descend into leaf.children."""
    assert _resolve_node(forest, (0, 0, 0)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_loop_forest.py -k resolve_node -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `_resolve_node`**

Append to `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
def _resolve_node(forest: LoopForest, path: tuple[int, ...]) -> "LoopNode | BodyLeaf | None":
    """Walk ``path`` from the forest root; return the node at that position.

    Returns ``None`` when the path is invalid — empty, out of range, or
    traversing through a ``BodyLeaf``.
    """
    if not path:
        return None
    siblings: list[LoopNode | BodyLeaf] = list(forest)
    node: LoopNode | BodyLeaf | None = None
    for idx in path:
        if idx < 0 or idx >= len(siblings):
            return None
        node = siblings[idx]
        if isinstance(node, BodyLeaf):
            """Consumed the terminal leaf; further path indices are
            invalid."""
            siblings = []
        else:
            siblings = node.children
    return node if siblings is not None else None
```

**Bug check for Step 3 code.** The last-loop-iteration behaviour needs attention: after consuming `path[-1]`, `node` points at the target; if that target is a BodyLeaf, the loop sets `siblings=[]` but we never use it. So `return node` is correct when `path` is fully consumed. The mid-walk guard is the `idx >= len(siblings)` check at loop top of the next iteration — that fires when the current `node` was a BodyLeaf and a further index is requested (because `siblings=[]`). Good.

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_loop_forest.py -k resolve_node -v
```

Expected: PASS (all five tests).

- [ ] **Step 5: Run the full loop_forest test module**

```bash
pytest test/codegen/test_loop_forest.py -v
```

Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "Add _resolve_node helper for path-indexed forest lookup"
```

---

## Phase C — `ReorderLoops` atom

### Task C1: Scaffold `reorder_loops.py` with `ReorderLoops` dataclass + `_roles_commute`

**Files:**
- Create: `nkigym/src/nkigym/tune/reorder_loops.py`
- Create: `test/codegen/test_reorder_loops.py`

- [ ] **Step 1: Write the failing tests**

Create `test/codegen/test_reorder_loops.py`:

```python
"""Layer-2 tests: ReorderLoops atom and enumerator."""

from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
from nkigym.ops.base import AxisRole


def test_reorder_loops_is_frozen_dataclass_with_path_outer_inner_dims() -> None:
    """ReorderLoops exposes path, outer_dim, inner_dim as frozen fields."""
    from nkigym.tune.reorder_loops import ReorderLoops

    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.path == (0,)
    assert atom.outer_dim == "d0"
    assert atom.inner_dim == "d1"


def test_roles_commute_par_par_true() -> None:
    """Two PARALLEL loops always commute."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.PARALLEL)
    b = LoopNode("d1", 4, AxisRole.PARALLEL)
    assert _roles_commute(a, b) is True


def test_roles_commute_par_acc_true_both_orderings() -> None:
    """PAR×ACC and ACC×PAR both commute (one is outer or the other)."""
    from nkigym.tune.reorder_loops import _roles_commute

    par = LoopNode("d0", 4, AxisRole.PARALLEL)
    acc = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(par, acc) is True
    assert _roles_commute(acc, par) is True


def test_roles_commute_acc_acc_same_reduce_op_true() -> None:
    """Two ACCs with the same reduce_op commute (associative+commutative)."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op="add")
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(a, b) is True


def test_roles_commute_acc_acc_different_reduce_op_false() -> None:
    """ACCs with different reduce_ops do not commute."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op="add")
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="max")
    assert _roles_commute(a, b) is False


def test_roles_commute_acc_acc_missing_reduce_op_false() -> None:
    """An ACC node with reduce_op=None cannot commute even with another ACC."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op=None)
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(a, b) is False


def test_roles_commute_any_sequential_false() -> None:
    """A SEQUENTIAL loop never commutes."""
    from nkigym.tune.reorder_loops import _roles_commute

    seq = LoopNode("d0", 4, AxisRole.SEQUENTIAL)
    par = LoopNode("d1", 4, AxisRole.PARALLEL)
    acc = LoopNode("d2", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(seq, par) is False
    assert _roles_commute(par, seq) is False
    assert _roles_commute(seq, acc) is False
    assert _roles_commute(acc, seq) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_reorder_loops.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.tune.reorder_loops'`.

- [ ] **Step 3: Create `reorder_loops.py` with `ReorderLoops` and `_roles_commute`**

Create `nkigym/src/nkigym/tune/reorder_loops.py`:

```python
"""``ReorderLoops`` rewrite — local perfect-nest loop interchange.

Swaps a ``LoopNode`` with its unique ``LoopNode`` child. Classical
rectangular polyhedral interchange: legality depends only on the
swap pair (not the surrounding forest), so the atom composes cleanly
with ``FuseLoops`` and the future hoist primitive.

The atom's identity is ``(path, outer_dim, inner_dim)``:

* ``path`` — tuple of child indices from the forest root down to the
  outer ``LoopNode`` of the swap pair.
* ``outer_dim`` / ``inner_dim`` — dim ids the two loops iterate;
  guard the atom against stale bindings when the caller stores a
  rewrite list across intervening rewrites.

Self-inverse: applying the same atom twice restores the original
forest (structurally). The ``tune`` stage uses forest-state hashing
(:func:`nkigym.codegen.loop_forest.hash_forest`) to break cycles in
the random-draw loop.
"""

from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, _resolve_node
from nkigym.ops.base import AxisRole


def _roles_commute(a: LoopNode, b: LoopNode) -> bool:
    """Return True iff swapping loops ``a`` and ``b`` preserves semantics.

    * Any SEQUENTIAL involvement → False (non-associative state).
    * PAR×PAR → True.
    * Mixed PAR / ACC → True (ordering changes accumulator footprint,
      not correctness).
    * ACC×ACC → True iff both have the same non-None reduce_op.
    """
    result: bool
    if a.role == AxisRole.SEQUENTIAL or b.role == AxisRole.SEQUENTIAL:
        result = False
    elif a.role == AxisRole.ACCUMULATION and b.role == AxisRole.ACCUMULATION:
        result = a.reduce_op is not None and a.reduce_op == b.reduce_op
    else:
        result = True
    return result


@dataclass(frozen=True)
class ReorderLoops:
    """Swap a LoopNode with its unique LoopNode child.

    Attributes:
        path: Child indices from the forest root down to (and including)
            the outer LoopNode of the swap pair. A length-1 path
            ``(idx,)`` targets ``forest[idx]``; a length-2 path
            ``(idx, j)`` targets ``forest[idx].children[j]``. Empty
            path is invalid (:meth:`is_legal` returns False).
        outer_dim: Dim id the outer loop iterates; guards against stale
            atoms after unrelated rewrites.
        inner_dim: Dim id the inner loop iterates; guards against stale
            atoms after unrelated rewrites.
    """

    path: tuple[int, ...]
    outer_dim: str
    inner_dim: str

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return True when the swap pair exists, is locally perfect, and role-commutes."""
        _ = op_graph
        result: bool
        outer = _resolve_node(forest, self.path)
        if not isinstance(outer, LoopNode):
            result = False
        elif outer.dim_id != self.outer_dim:
            result = False
        elif len(outer.children) != 1:
            result = False
        else:
            inner = outer.children[0]
            if not isinstance(inner, LoopNode):
                result = False
            elif inner.dim_id != self.inner_dim:
                result = False
            elif not _roles_commute(outer, inner):
                result = False
            else:
                result = True
        return result

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Swap outer and inner; grandchildren subtree is passed by reference."""
        raise NotImplementedError
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_reorder_loops.py -v
```

Expected: PASS (all seven tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/reorder_loops.py test/codegen/test_reorder_loops.py
git commit -m "Scaffold ReorderLoops atom with _roles_commute helper"
```

### Task C2: Implement `is_legal` with full coverage

The `is_legal` code was included in Task C1 as part of the scaffold; Task C2 adds the `is_legal` tests.

**Files:**
- Test: `test/codegen/test_reorder_loops.py`

- [ ] **Step 1: Add `is_legal` test cases**

Append to `test/codegen/test_reorder_loops.py`:

```python
def test_is_legal_accepts_par_par_cross_dim_perfect_pair() -> None:
    """Classic positive case: outer PAR loop with one PAR LoopNode child on a different dim."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_par_par_same_dim_perfect_pair() -> None:
    """Same-dim swap is not excluded — future tiles_per_block uses it."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 16, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d0")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_par_acc_perfect_pair() -> None:
    """Mixed PAR × ACC is legal (ordering affects footprint, not correctness)."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_acc_acc_same_reduce_op() -> None:
    """ACC × ACC with matching reduce_op is legal."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_rejects_acc_acc_different_reduce_op() -> None:
    """Different reduce_ops do not commute."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="max")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_any_sequential() -> None:
    """SEQUENTIAL never commutes."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.SEQUENTIAL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_stale_outer_dim() -> None:
    """outer_dim not matching the resolved node → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="dZZZ", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_stale_inner_dim() -> None:
    """inner_dim not matching the child → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="dZZZ")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_out_of_range_path() -> None:
    """Path indexing past the forest root → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    atom = ReorderLoops(path=(99,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_path_terminating_at_body_leaf() -> None:
    """Path whose last step lands on a BodyLeaf → False (not a LoopNode)."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    """path=(0, 0) lands on the BodyLeaf."""
    atom = ReorderLoops(path=(0, 0), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_outer_with_multiple_children() -> None:
    """Local imperfect-nest (>1 child) → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    child_a = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    child_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [child_a, child_b])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_outer_whose_child_is_body_leaf() -> None:
    """Locally-perfect shape must have child = LoopNode, not BodyLeaf."""
    from nkigym.tune.reorder_loops import ReorderLoops

    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_empty_path() -> None:
    """Empty path has no target node → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    atom = ReorderLoops(path=(), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False
```

- [ ] **Step 2: Run tests**

```bash
pytest test/codegen/test_reorder_loops.py -v
```

Expected: PASS (all 13 new tests plus the 7 from Task C1).

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_reorder_loops.py
git commit -m "Add ReorderLoops.is_legal tests for role/path/shape coverage"
```

### Task C3: Implement `apply`

**Files:**
- Modify: `nkigym/src/nkigym/tune/reorder_loops.py`
- Test: `test/codegen/test_reorder_loops.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_reorder_loops.py`:

```python
def test_apply_swaps_outer_and_inner_at_forest_root() -> None:
    """Apply swaps outer ↔ inner; grandchildren pass through by reference."""
    from nkigym.tune.reorder_loops import ReorderLoops

    leaf = BodyLeaf(op_idx=0)
    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [leaf])
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, new_forest = atom.apply(None, forest)
    assert len(new_forest) == 1
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert new_outer.dim_id == "d1"
    assert new_outer.trip_count == 4
    assert new_outer.role is AxisRole.PARALLEL
    assert len(new_outer.children) == 1
    new_inner = new_outer.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.dim_id == "d0"
    assert new_inner.trip_count == 8
    assert new_inner.role is AxisRole.PARALLEL
    assert new_inner.children[0] is leaf, "grandchildren subtree must be reference-equal"


def test_apply_preserves_reduce_op_across_swap() -> None:
    """reduce_op field travels with each node through the swap."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, new_forest = atom.apply(None, forest)
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert new_outer.reduce_op == "add"
    new_inner = new_outer.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.reduce_op == "add"


def test_apply_at_nested_path() -> None:
    """path=(0, 0) swaps the inner two of three nested loops, preserving the outermost."""
    from nkigym.tune.reorder_loops import ReorderLoops

    innermost = LoopNode("d2", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    middle = LoopNode("d1", 4, AxisRole.PARALLEL, [innermost])
    outermost = LoopNode("d0", 4, AxisRole.PARALLEL, [middle])
    forest = [outermost]
    atom = ReorderLoops(path=(0, 0), outer_dim="d1", inner_dim="d2")
    _, new_forest = atom.apply(None, forest)
    top = new_forest[0]
    assert isinstance(top, LoopNode)
    assert top.dim_id == "d0"
    new_middle = top.children[0]
    assert isinstance(new_middle, LoopNode)
    assert new_middle.dim_id == "d2"
    new_inner = new_middle.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.dim_id == "d1"
    assert new_inner.children[0] is innermost.children[0]


def test_apply_is_self_inverse_by_structural_hash() -> None:
    """Applying the same atom twice produces a forest with the starting hash."""
    from nkigym.codegen.loop_forest import hash_forest
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, [inner])
    forest = [outer]
    first = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, f1 = first.apply(None, forest)
    """After the first swap the outer dim is d1, inner dim is d0.
    To swap back we need a fresh atom that matches the new state."""
    reverse = ReorderLoops(path=(0,), outer_dim="d1", inner_dim="d0")
    _, f2 = reverse.apply(None, f1)
    assert hash_forest(forest) == hash_forest(f2)


def test_apply_rmsnorm_matmul_preserves_check_invariant() -> None:
    """Reordering d0(T=16) ↔ d1(T=16) inside tensor_scalar's chain keeps invariant."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest, check_invariant, compute_phase_touched
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose
    from nkigym.tune.reorder_loops import ReorderLoops

    @nkigym_kernel
    def rmm(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=1e-6)(
            data=lhs_sbuf
        )
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(rmm, specs)
    forest = build_canonical_forest(g)
    """tensor_scalar is op index 3. Its tree shape is d0-block/d0-tile/d1-block/d1-tile/leaf.
    Swap d0-tile (at path (3, 0)) with d1-block (its child)."""
    atom = ReorderLoops(path=(3, 0), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(g, forest) is True
    _, new_forest = atom.apply(g, forest)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(new_forest, num_tiles, op_touched, phase_touched)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_reorder_loops.py -k apply -v
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `apply` + supporting helpers**

Edit `nkigym/src/nkigym/tune/reorder_loops.py`. Replace the `apply` body with:

```python
    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Swap outer and inner; grandchildren subtree is passed by reference."""
        new_forest = _rewrite_at_path(forest, self.path, _swap_pair)
        return op_graph, new_forest
```

Append module-level helpers after `ReorderLoops`:

```python
def _swap_pair(outer: LoopNode) -> LoopNode:
    """Swap ``outer`` with its unique LoopNode child.

    outer → inner → grandchildren  becomes
    inner → outer → grandchildren.
    """
    assert len(outer.children) == 1
    inner = outer.children[0]
    assert isinstance(inner, LoopNode)
    new_outer = LoopNode(
        dim_id=outer.dim_id,
        trip_count=outer.trip_count,
        role=outer.role,
        children=list(inner.children),
        reduce_op=outer.reduce_op,
    )
    return LoopNode(
        dim_id=inner.dim_id,
        trip_count=inner.trip_count,
        role=inner.role,
        children=[new_outer],
        reduce_op=inner.reduce_op,
    )


def _rewrite_at_path(
    forest: LoopForest,
    path: tuple[int, ...],
    transform,
) -> LoopForest:
    """Return a new forest with ``transform`` applied to the node at ``path``.

    Ancestors along ``path`` are reconstructed; everything outside the
    edit site is passed through by reference.
    """
    if len(path) == 1:
        idx = path[0]
        target = forest[idx]
        assert isinstance(target, LoopNode)
        replacement = transform(target)
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_at_path(parent.children, rest, transform)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_reorder_loops.py -v
```

Expected: PASS (all prior tests + new `apply` tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/reorder_loops.py test/codegen/test_reorder_loops.py
git commit -m "Implement ReorderLoops.apply with structural sharing"
```

### Task C4: Implement `enumerate_reorder_atoms`

**Files:**
- Modify: `nkigym/src/nkigym/tune/reorder_loops.py`
- Test: `test/codegen/test_reorder_loops.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_reorder_loops.py`:

```python
def test_enumerate_empty_forest_returns_empty_list() -> None:
    """Empty forest has no atoms."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    assert enumerate_reorder_atoms([]) == []


def test_enumerate_single_loop_node_over_leaf_returns_empty() -> None:
    """No parent→child LoopNode pair present → no atoms."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert enumerate_reorder_atoms(forest) == []


def test_enumerate_finds_par_par_pair_at_forest_root() -> None:
    """A single perfect parent→child LoopNode pair yields one atom."""
    from nkigym.tune.reorder_loops import ReorderLoops, enumerate_reorder_atoms

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atoms = enumerate_reorder_atoms(forest)
    assert atoms == [ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")]


def test_enumerate_finds_nested_chain_yields_multiple_atoms() -> None:
    """3-deep chain yields two atoms (parent-child pair at each non-leaf level)."""
    from nkigym.tune.reorder_loops import ReorderLoops, enumerate_reorder_atoms

    innermost = LoopNode("d2", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    middle = LoopNode("d1", 4, AxisRole.PARALLEL, [innermost])
    outermost = LoopNode("d0", 4, AxisRole.PARALLEL, [middle])
    forest = [outermost]
    atoms = enumerate_reorder_atoms(forest)
    assert ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1") in atoms
    assert ReorderLoops(path=(0, 0), outer_dim="d1", inner_dim="d2") in atoms
    assert len(atoms) == 2


def test_enumerate_skips_imperfect_pairs() -> None:
    """Outer with >1 child does not contribute a reorder atom at that level."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    child_a = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    child_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [child_a, child_b])
    forest = [outer]
    """No swap at path=(0,) is legal. child_a/child_b are each locally
    perfect over a BodyLeaf, so no further atoms below either."""
    atoms = enumerate_reorder_atoms(forest)
    assert atoms == []


def test_enumerate_skips_role_incompatible_pairs() -> None:
    """PAR×SEQ is skipped even when locally perfect."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    inner = LoopNode("d1", 4, AxisRole.SEQUENTIAL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    assert enumerate_reorder_atoms(forest) == []


def test_enumerate_rmsnorm_matmul_canonical_finds_expected_atoms() -> None:
    """Canonical rmsnorm+matmul yields the expected reorder set.

    Every op-tree is a 2N-deep chain of same-dim block+tile pairs, so
    the enumerator finds ``(path(k), outer_dim=d, inner_dim=d)`` atoms
    at every non-leaf level where roles commute. Inside matmul the K
    sub-chain is ACC×ACC-same-reduce-op ('add') — those same-dim pairs
    are also legal.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    @nkigym_kernel
    def rmm(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=1e-6)(
            data=lhs_sbuf
        )
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(rmm, specs)
    forest = build_canonical_forest(g)
    atoms = enumerate_reorder_atoms(forest)
    """Sanity: at least one atom exists in every 2D-or-deeper op's chain
    (load, rhs_load, tensor_scalar, transpose, store are 2D ops; each
    has block→tile same-dim pair legal as PAR×PAR)."""
    paths = {a.path for a in atoms}
    assert (0,) in paths
    assert (1,) in paths
    assert (3,) in paths
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_reorder_loops.py -k enumerate -v
```

Expected: FAIL — `ImportError: cannot import name 'enumerate_reorder_atoms'`.

- [ ] **Step 3: Implement the enumerator**

Append to `nkigym/src/nkigym/tune/reorder_loops.py`:

```python
def enumerate_reorder_atoms(forest: LoopForest) -> list[ReorderLoops]:
    """Return every legal :class:`ReorderLoops` atom in ``forest``.

    Walks the forest recursively: at every LoopNode whose single child
    is another LoopNode and whose role commutes with that child,
    emits one atom.
    """
    atoms: list[ReorderLoops] = []
    _collect_reorder(forest, path=(), atoms=atoms)
    return atoms


def _collect_reorder(
    siblings: list[LoopNode | BodyLeaf],
    path: tuple[int, ...],
    atoms: list[ReorderLoops],
) -> None:
    """Recursive helper for :func:`enumerate_reorder_atoms`."""
    for idx, node in enumerate(siblings):
        if isinstance(node, LoopNode):
            if len(node.children) == 1 and isinstance(node.children[0], LoopNode):
                inner = node.children[0]
                if _roles_commute(node, inner):
                    atoms.append(
                        ReorderLoops(
                            path=path + (idx,),
                            outer_dim=node.dim_id,
                            inner_dim=inner.dim_id,
                        )
                    )
            _collect_reorder(node.children, path=path + (idx,), atoms=atoms)
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_reorder_loops.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/reorder_loops.py test/codegen/test_reorder_loops.py
git commit -m "Add enumerate_reorder_atoms for all legal local-perfect pairs"
```

---

## Phase D — `tune` stage integration

### Task D1: Include reorder atoms in random draw; add cycle break

**Files:**
- Modify: `nkigym/src/nkigym/tune/stage.py:18-20,66-74`
- Test: `test/codegen/test_compile.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_compile.py`:

```python
def test_tune_stage_applies_reorder_loops_inside_tensor_scalar(tmp_path: Path) -> None:
    """An explicit ReorderLoops on tensor_scalar's inner chain still matches numpy."""
    from nkigym.tune.reorder_loops import ReorderLoops

    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    """tensor_scalar is op index 3; its tree is a 2N chain over (d0, d1).
    Swap the d0-tile (depth 1, path (3, 0)) with its d1-block child."""
    rewrites = [ReorderLoops(path=(3, 0), outer_dim="d0", inner_dim="d1")]
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_compose_reorder_then_fuse(tmp_path: Path) -> None:
    """Reorder exposes a new fusion boundary; composed pipeline still CPU-sim-passes."""
    from nkigym.tune.fuse_loops import FuseLoops
    from nkigym.tune.reorder_loops import ReorderLoops

    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    """Apply an unrelated reorder first, then the canonical forest-root
    FuseLoops on (2, 3) — both should still be legal and the CPU-sim
    gate passes on the combined result."""
    rewrites = [
        ReorderLoops(path=(3, 0), outer_dim="d0", inner_dim="d1"),
        FuseLoops(path=(), boundary=(2, 3), dim_id="d0"),
    ]
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_random_draw_terminates_when_only_self_inverse_atom_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With enumerators stubbed to emit a single self-inverse atom, tune terminates on hash cycle."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.ops.base import AxisRole
    from nkigym.tune import stage as stage_mod
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])

    def _only_root_atom(forest):
        """Return the unique reorder at the outermost two-level pair (if present)."""
        if not forest:
            return []
        top = forest[0]
        if not isinstance(top, LoopNode) or len(top.children) != 1:
            return []
        child = top.children[0]
        if not isinstance(child, LoopNode):
            return []
        return [ReorderLoops(path=(0,), outer_dim=top.dim_id, inner_dim=child.dim_id)]

    monkeypatch.setattr(stage_mod, "enumerate_reorder_atoms", _only_root_atom)
    monkeypatch.setattr(stage_mod, "enumerate_fusion_atoms", lambda forest: [])

    class _AlwaysHeads:
        def random(self):
            return 0.0

    monkeypatch.setattr(stage_mod.random, "Random", lambda _seed: _AlwaysHeads())

    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], seed=0)
    """If the cycle break didn't fire, this would loop forever (or until
    Python recursion/OOM). Reaching this line means the hash_forest
    cycle detector stopped the loop after the first revisit."""
    assert (tmp_path / "kernel_tuned.py").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_compile.py -k tune_stage_applies_reorder_loops -v
```

Expected: FAIL — `ReorderLoops` isn't yet included in the tune stage's random enumerator, but the explicit-list path works. Re-reading: the explicit-list test **should already pass** because `ReorderLoops` implements the `KernelRewrite` protocol and `run_tune`'s explicit path dispatches through it. Check and confirm by running:

```bash
pytest test/codegen/test_compile.py::test_tune_stage_applies_reorder_loops_inside_tensor_scalar -v
```

If it PASSES, skip Step 3's first half (the explicit-list path is already covered). If it FAILS, something else is wrong — read the traceback and fix before proceeding.

The cycle-termination test (`test_tune_stage_random_draw_terminates_when_only_self_inverse_atom_exists`) WILL fail because `enumerate_reorder_atoms` is not imported into the stage yet.

- [ ] **Step 3: Extend `run_tune` — import reorder enumerator and add hash cycle break**

Edit `nkigym/src/nkigym/tune/stage.py`. Replace lines 18–20 (import block) with:

```python
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest
from nkigym.codegen.render import render
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms
```

Replace the random-draw block (currently lines 66–74, the `if rewrites is None:` branch) with:

```python
    if rewrites is None:
        rng = random.Random(seed)
        seen: set[int] = {hash_forest(forest)}
        while True:
            atoms = enumerate_fusion_atoms(forest) + enumerate_reorder_atoms(forest)
            candidates = [a for a in atoms if rng.random() < 0.5]
            if not candidates:
                break
            chosen = candidates[0]
            op_graph, forest = chosen.apply(op_graph, forest)
            h = hash_forest(forest)
            if h in seen:
                break
            seen.add(h)
```

Update the `run_tune` docstring to reflect the richer atom pool and the cycle-break behaviour. Replace the docstring's second paragraph (starting `"Two paths:"`) with:

```
    Two paths:

    * **Explicit** (``rewrites`` is a list): apply each rewrite in order,
      checking legality before each ``apply`` because the caller's
      boundary / path indices may become stale after prior applies.
    * **Random** (``rewrites`` is ``None``): draw atoms from
      ``enumerate_fusion_atoms`` + ``enumerate_reorder_atoms`` between
      applies, pick the first one whose coin-flip survives, and apply
      it. Terminates when no atom survives the draw or when the
      resulting forest state repeats (hash-detected cycle — reorder
      atoms are self-inverse, so a self-canceling pair must stop the
      loop).
```

- [ ] **Step 4: Run tests**

```bash
pytest test/codegen/test_compile.py -v
```

Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/stage.py test/codegen/test_compile.py
git commit -m "Wire ReorderLoops into tune stage random draw with cycle break"
```

### Task D2: End-to-end validation — run the example

**Files:**
- Execute: `examples/rmsnorm_matmul.py`

- [ ] **Step 1: Confirm the example runs end-to-end**

```bash
source ~/venvs/kernel-env/bin/activate
python examples/rmsnorm_matmul.py
```

Expected: both `kernel.py` and `kernel_tuned.py` land in the cache dir with no CPU-sim mismatches.

Inspect the tuned kernel to confirm reorder atoms participated:

```bash
diff /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py /home/ubuntu/cache/rmsnorm_matmul_compile/kernel_tuned.py
```

Expected: a non-empty diff demonstrating structural change.

- [ ] **Step 2: Run the full test suite**

```bash
pytest test -v
```

Expected: all tests pass.

- [ ] **Step 3: No commit**

This task is validation only.

---

## Self-Review Notes

- **Spec coverage.**
  - §2 (`LoopNode.reduce_op` + population) — Tasks A1, A2, A3.
  - §3 (atom signature) — Task C1.
  - §4 (legality + `_resolve_node` + `_roles_commute`) — Tasks B2, C1, C2.
  - §5 (apply + `_rewrite_at_path`) — Task C3.
  - §6 (enumeration) — Task C4.
  - §7 (tune stage integration + hash cycle break) — Tasks B1, D1.
  - §8 (testing layers 1-3) — Tasks A1-A3 (layer 1), C1-C4 (layer 2), D1 (layer 3).
  - §9 (migration phases A-E) — Tasks split along the same phase boundaries. Phase E (cache reseed) is documentation-only; noted in the final commit message.
- **Placeholder scan.** Every `Step N` either names an exact test, shows exact code, or runs an exact command. No TBDs, no "similar to Task N" shortcuts.
- **Type consistency.** `ReorderLoops(path, outer_dim, inner_dim)` identity is used identically across C1, C2, C3, C4, D1 tests and source. `hash_forest` signature `LoopForest -> int` is consistent between B1 definition and D1 consumer. `_resolve_node(forest, path) -> LoopNode | BodyLeaf | None` is consistent between B2 definition and C1 consumer.
- **Positional-call safety.** Task A1's `LoopNode` edit places `reduce_op` after `children` deliberately — every existing call site uses 4 positional args (the 4th being `children`), so extending the dataclass with a new keyword field at the tail preserves them.
