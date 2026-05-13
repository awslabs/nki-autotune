# Canonical 1N + ComputeAt Partial-Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the canonical IR from 2N to 1N form (one `LoopNode` per `(dim, nest-position)`, no trip-1 partners), ban tail-siblings across atoms, and fix `ComputeAt`/`ReverseComputeAt` regeneration so partial-coverage ancestors produce a residual inner loop instead of silently dropping it.

**Architecture:** Three source changes: (1) `canonical.py` drops the tile-partner wrap in `_wrap_dims` + matmul K + activation-reduce F builders; (2) `split.py` rejects non-divisor factors via `AtomLegalityError`; (3) `compute_at.py` swaps `_ancestor_dims` for `_ancestor_trip_products`, uses per-dim trip products to compute a residual `L(d, num_tiles/covered)` loop when an ancestor only partially covers the dim. `reverse_compute_at.py` picks up the new helpers via its existing shared-import pattern. `inject_software_pipeline.py` loses an unreachable trip-1-descent branch.

**Tech Stack:** Python 3.12, pytest, `nki.simulate` for CPU sim; existing `nkigym` codegen + tune infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-08-canonical-1N-and-computeat-partial-coverage-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `nkigym/src/nkigym/codegen/canonical.py` | Modify | Emit 1N trees: `_wrap_dims`, `_build_leaves_matmul`, `_build_leaves_activation_reduce` each drop their trip-1 tile partner. |
| `nkigym/src/nkigym/tune/split.py` | Modify | `is_legal` + `apply` reject non-divisor factors via `AtomLegalityError`. Delete tail-sibling emission. |
| `nkigym/src/nkigym/tune/compute_at.py` | Modify | New `_ancestor_trip_products`; `_wrap_leaf_with_dims` takes `(dim, trip)` pairs; `ComputeAt.apply` uses trip products + raises `AtomLegalityError` on indivisible residual. Delete `_ancestor_dims`. |
| `nkigym/src/nkigym/tune/reverse_compute_at.py` | Modify | Mirror `ComputeAt.apply`'s new math; update shared-helper import. |
| `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py` | Modify | Delete the `trip_count == 1` descent branch in `_emit_pipelined_leaf` (unreachable on 1N). |
| `test/tune/test_split.py` | Modify | Delete tail-sibling tests; add divisor-guard tests. |
| `test/tune/test_compute_at.py` | Modify | Add partial-coverage trio. |
| `test/tune/test_reverse_compute_at.py` | Modify | Add partial-coverage trio. |
| `test/codegen/test_canonical.py`, `test/codegen/test_place_buffers.py`, `test/codegen/test_dep_cache.py`, `test/codegen/test_ir.py`, `test/codegen/test_software_pipeline_unit.py`, `test/codegen/test_multi_buffer_unit.py`, `test/codegen/test_batch.py`, `test/tune/test_reorder.py`, `test/tune/test_decompose_reduction.py`, `test/tune/test_end_to_end_template_kernel.py` | Modify as needed | Update any structural assertions that assume 2N. |
| `debug_kernel_0000_chain.py` | Verify | End-to-end Bug A fix check: step 2 must report `ok`. |
| `.claude/rules/learnings.md` | Modify | Record 1N decision + tail-sibling ban. |
| `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md` | Modify | Add Bug A landed row. |

---

## Task 1: Red test — ComputeAt regenerates residual trip on partial-coverage ancestor

**Files:**
- Modify: `test/tune/test_compute_at.py`

- [ ] **Step 1: Append a local `_mod_hand` helper + imports for hand-built modules**

At the top of `test/tune/test_compute_at.py` (after the existing imports), add:

```python
from nkigym.ir.dep_cache import DepCache
from nkigym.ir.ir import DimInfo, KernelIR, Tensor
from nkigym.ops.base import AxisRole


def _mod_hand(body: list, dims: dict[str, DimInfo], tensors: dict[str, Tensor]) -> KernelIR:
    """Build a minimal KernelIR from hand-rolled body / dims / tensors."""
    return KernelIR(
        func_name="f",
        param_names=[],
        return_name=next(iter(tensors)) if tensors else "x",
        tensors=tensors,
        dims=dims,
        body=body,
        dep=DepCache(scopes={}),
    )
```

- [ ] **Step 2: Append the residual-regen test**

Append to `test/tune/test_compute_at.py`:

```python
def test_apply_regenerates_residual_trip_on_partial_ancestor() -> None:
    """Partial-coverage ancestor → residual inner loop regenerated.

    Setup: forest body has two root subtrees.
    * Subtree 0: producer ``write={"p"}`` with no ancestor — sibling at the root.
    * Subtree 1: ``L(d0, trip=2)`` wrapping consumer ``reads={"p"}``.
    ``dims["d0"].num_tiles = 16``. Move producer under subtree 1's
    root loop (target_path=(1,)); producer's dim_role has ``d0``.
    Expected: producer appended under ``L(d0, 2)`` with a regenerated
    ``L(d0, 8)`` between target and producer leaf, covering the residual
    ``16 / 2 = 8`` trips.
    """
    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={},
        writes=("p",),
        kwargs={},
        axis_map={},
        dim_role={"d0": AxisRole.PARALLEL},
        op_local_buffers={},
    )
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        writes=(),
        kwargs={},
        axis_map={},
        dim_role={"d0": AxisRole.PARALLEL},
        op_local_buffers={},
    )
    consumer_outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[consumer])
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(2048,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer, consumer_outer], dims=dims, tensors=tensors)
    atom = ComputeAt(leaf_path=(0,), target_loop_path=(1,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    """After apply: body collapses from length 2 to 1 — producer removed
    from body[0], consumer_outer shifts to body[0]. Under the new
    consumer_outer, expect the original consumer + a regenerated
    L(d0, 8) wrapping a clone of producer."""
    assert len(new_module.body) == 1
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 2
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert len(regen_wrappers) == 1, f"expected one regenerated d0 loop, got {len(regen_wrappers)}"
    residual_loop = regen_wrappers[0]
    assert residual_loop.trip_count == 8
    residual_child = residual_loop.children[0]
    assert isinstance(residual_child, BodyLeaf)
    assert residual_child.writes == ("p",)
```

- [ ] **Step 3: Run the test, verify RED**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/tune/test_compute_at.py::test_apply_regenerates_residual_trip_on_partial_ancestor -v`
Expected: FAIL at `assert len(regen_wrappers) == 1, ...` with actual `0` — current `_ancestor_dims` (set-membership) returns `{"d0"}` when target is `L(d0, 2)`, so `needed=[]` and no inner wrapper is generated at all. After Task 4's fix, one wrapper with `trip_count=8` will be produced.

- [ ] **Step 4: Commit the failing test**

```bash
git add test/tune/test_compute_at.py
git commit -m "test: red for computeat partial-coverage residual regen"
```

---

## Task 2: Red test — indivisible residual raises AtomLegalityError

**Files:**
- Modify: `test/tune/test_compute_at.py`

- [ ] **Step 1: Append the second red test**

Append to `test/tune/test_compute_at.py`:

```python
def test_apply_raises_on_indivisible_residual() -> None:
    """Ancestor trip does not divide num_tiles → AtomLegalityError."""
    from nkigym.tune import AtomLegalityError

    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        writes=("p",),
        dim_role={"d0": AxisRole.PARALLEL},
    )
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        dim_role={"d0": AxisRole.PARALLEL},
    )
    """num_tiles=17 with ancestor trip=3 → residual 17/3 does not divide."""
    consumer_outer = LoopNode("d0", 3, AxisRole.PARALLEL, children=[consumer])
    dims = {"d0": DimInfo(dim_id="d0", total_size=17 * 128, tile_size=128, num_tiles=17)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(17 * 128,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer, consumer_outer], dims=dims, tensors=tensors)
    atom = ComputeAt(leaf_path=(0,), target_loop_path=(1,))
    assert atom.is_legal(module)
    try:
        atom.apply(module)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for indivisible residual")
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest test/tune/test_compute_at.py::test_apply_raises_on_indivisible_residual -v`
Expected: FAIL — current code silently wraps with `L(d0, num_tiles=17) → L(d0, 1)`, no exception raised.

- [ ] **Step 3: Commit**

```bash
git add test/tune/test_compute_at.py
git commit -m "test: red for computeat indivisible-residual rejection"
```

---

## Task 3: Implement `_ancestor_trip_products` helper

**Files:**
- Modify: `nkigym/src/nkigym/tune/compute_at.py` (add helper, retain `_ancestor_dims` for now — deletion comes after call sites migrate)

- [ ] **Step 1: Add the helper**

In `nkigym/src/nkigym/tune/compute_at.py`, after `_ancestor_dims` (line 153):

```python
def _ancestor_trip_products(body: TreeIR, path: tuple[int, ...]) -> dict[str, int]:
    """Product of trip_counts per dim_id along ``path`` from body root.

    Walks the same path as :func:`_ancestor_dims` but multiplies each
    ancestor LoopNode's trip_count into a per-dim accumulator. Dims
    not on the path are absent from the result (callers treat absence
    as coverage of 1).
    """
    products: dict[str, int] = {}
    siblings: list[LoopNode | BodyLeaf] = list(body)
    for idx in path:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, LoopNode):
            products[node.dim_id] = products.get(node.dim_id, 1) * node.trip_count
            siblings = node.children
        else:
            break
    return products
```

- [ ] **Step 2: Add a direct unit test for the helper**

Append to `test/tune/test_compute_at.py`:

```python
def test_ancestor_trip_products_accumulates_same_dim() -> None:
    """Same-dim ancestors contribute their trips multiplicatively."""
    from nkigym.tune.compute_at import _ancestor_trip_products

    leaf = BodyLeaf(op_cls=object, phase="main")
    l3 = LoopNode("d0", 4, AxisRole.PARALLEL, children=[leaf])
    l2 = LoopNode("d0", 2, AxisRole.PARALLEL, children=[l3])
    l1 = LoopNode("d1", 3, AxisRole.PARALLEL, children=[l2])
    body = [l1]
    assert _ancestor_trip_products(body, (0, 0, 0)) == {"d0": 8, "d1": 3}
    assert _ancestor_trip_products(body, (0, 0)) == {"d0": 2, "d1": 3}
    assert _ancestor_trip_products(body, (0,)) == {"d1": 3}
    assert _ancestor_trip_products(body, ()) == {}
```

- [ ] **Step 3: Run the helper test, verify GREEN**

Run: `pytest test/tune/test_compute_at.py::test_ancestor_trip_products_accumulates_same_dim -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/compute_at.py test/tune/test_compute_at.py
git commit -m "feat: _ancestor_trip_products for per-dim trip accumulation"
```

---

## Task 4: Rewire `_wrap_leaf_with_dims` to accept `(dim, trip)` pairs

**Files:**
- Modify: `nkigym/src/nkigym/tune/compute_at.py:169-183`

- [ ] **Step 1: Replace the body**

Replace `_wrap_leaf_with_dims` (current lines 169-183) with:

```python
def _wrap_leaf_with_dims(
    leaf: BodyLeaf, dim_trips: list[tuple[str, int]], module: KernelIR
) -> LoopNode | BodyLeaf:
    """Wrap ``leaf`` in one LoopNode per ``(dim, trip)`` entry (1N form).

    Outermost-first. ``module`` is unused by the helper itself (trip
    comes from the caller); retained in the signature for call-site
    symmetry — callers passing the enclosing module don't need to be
    revised on every helper tweak. Returns ``leaf`` directly when
    ``dim_trips`` is empty.
    """
    _ = module
    if not dim_trips:
        return leaf
    node: LoopNode | BodyLeaf = leaf
    for d, trip in reversed(dim_trips):
        role = leaf.dim_role[d]
        node = LoopNode(dim_id=d, trip_count=trip, role=role, children=[node])
    return node
```

- [ ] **Step 2: Adapt `ComputeAt.apply` call site (lines 52-79)**

Replace lines ~73-77 (the `ancestor_dims = _ancestor_dims(...)` block through `regenerated = _wrap_leaf_with_dims(...)`) with:

```python
        ancestor_products = _ancestor_trip_products(body_without, new_target_path)
        leaf_dims = list(leaf.dim_role.keys())
        needed: list[tuple[str, int]] = []
        for d in leaf_dims:
            covered = ancestor_products.get(d, 1)
            num_t = module.dims[d].num_tiles
            if num_t == covered:
                continue
            if num_t % covered != 0:
                from nkigym.tune import AtomLegalityError

                raise AtomLegalityError(
                    f"ComputeAt.apply: ancestor coverage {covered} does not divide "
                    f"num_tiles[{d!r}]={num_t}"
                )
            residual = num_t // covered
            if residual > 1:
                needed.append((d, residual))
        regenerated = _wrap_leaf_with_dims(leaf, needed, module)
```

- [ ] **Step 3: Mirror in `ReverseComputeAt.apply`** (`nkigym/src/nkigym/tune/reverse_compute_at.py:52-84`)

Replace the analogous block (lines ~78-81) with the same logic. Update the shared-import list at line 54 to replace `_ancestor_dims` with `_ancestor_trip_products`:

```python
from nkigym.tune.compute_at import (
    _ancestor_trip_products,
    _append_under,
    _find_node_path,
    _remove_at_path,
    _rename_canonical,
    _wrap_leaf_with_dims,
)
```

And replace the legality block with the same logic as `ComputeAt.apply`, except the error message reads `"ReverseComputeAt.apply: ..."`.

- [ ] **Step 4: Delete `_ancestor_dims`**

Remove `_ancestor_dims` from `compute_at.py` (current lines 153-166). Confirm no other file imports it:

Run: `grep -rn "_ancestor_dims" /home/ubuntu/nki-autotune/nkigym/ /home/ubuntu/nki-autotune/test/`
Expected: no matches.

- [ ] **Step 5: Run Task 1 + 2 tests, verify GREEN**

Run: `pytest test/tune/test_compute_at.py::test_apply_regenerates_residual_trip_on_partial_ancestor test/tune/test_compute_at.py::test_apply_raises_on_indivisible_residual -v`
Expected: PASS.

- [ ] **Step 6: Run the full ComputeAt + ReverseComputeAt test files**

Run: `pytest test/tune/test_compute_at.py test/tune/test_reverse_compute_at.py -v`
Expected: all green. If any existing test asserts `L(d, num_tiles) → L(d, 1)` shape for regenerated leaves, update those assertions to the new 1N shape `L(d, residual)` (no trip-1 partner). Reason: tests were written against 2N; 1N is now the contract.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/tune/compute_at.py nkigym/src/nkigym/tune/reverse_compute_at.py test/tune/test_compute_at.py test/tune/test_reverse_compute_at.py
git commit -m "fix: computeat/reverse_computeat regenerate residual trip on partial coverage"
```

---

## Task 5: Mirror the partial-coverage trio onto ReverseComputeAt

**Files:**
- Modify: `test/tune/test_reverse_compute_at.py`

- [ ] **Step 1: Add imports + local `_mod_hand` helper to the reverse test file**

If not already present, add the same imports and `_mod_hand` helper used in Task 1 Step 1 to `test/tune/test_reverse_compute_at.py`:

```python
from nkigym.ir.dep_cache import DepCache
from nkigym.ir.ir import BodyLeaf, DimInfo, KernelIR, LoopNode, Tensor
from nkigym.ops.base import AxisRole
from nkigym.tune.reverse_compute_at import ReverseComputeAt


def _mod_hand(body: list, dims: dict[str, DimInfo], tensors: dict[str, Tensor]) -> KernelIR:
    return KernelIR(
        func_name="f",
        param_names=[],
        return_name=next(iter(tensors)) if tensors else "x",
        tensors=tensors,
        dims=dims,
        body=body,
        dep=DepCache(scopes={}),
    )
```

- [ ] **Step 2: Append the mirror trio**

Append three tests that exercise `ReverseComputeAt.apply`'s regeneration. `ReverseComputeAt` moves a consumer leaf under a target loop whose subtree contains a producer. Semantics: target must contain a write of a tensor the moved leaf reads.

```python
def test_reverse_apply_regenerates_residual_trip_on_partial_ancestor() -> None:
    """Partial-coverage ancestor → residual inner loop regenerated.

    Forest body:
    * Subtree 0: ``L(d0, trip=2)`` wrapping producer ``writes={"p"}``.
    * Subtree 1: consumer ``reads={"p"}`` at the root.

    Move the consumer under subtree 0's root loop. ``num_tiles[d0]=16``,
    ancestor ``covered=2`` → regenerated ``L(d0, 8)``.
    """
    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        writes=("p",),
        dim_role={"d0": AxisRole.PARALLEL},
    )
    producer_outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        dim_role={"d0": AxisRole.PARALLEL},
    )
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(2048,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 2
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert len(regen_wrappers) == 1
    residual_loop = regen_wrappers[0]
    assert residual_loop.trip_count == 8
    residual_child = residual_loop.children[0]
    assert isinstance(residual_child, BodyLeaf)
    assert residual_child.reads == {"data": "p"}


def test_reverse_apply_skips_fully_covered_dim() -> None:
    """Ancestor covers full num_tiles → no regeneration.

    Body:
    * Subtree 0: ``L(d0, trip=16)`` wrapping producer.
    * Subtree 1: consumer at the root.
    ``num_tiles[d0]=16`` → ``covered==num_tiles`` → no inner d0 loop.
    """
    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        writes=("p",),
        dim_role={"d0": AxisRole.PARALLEL},
    )
    producer_outer = LoopNode("d0", 16, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        dim_role={"d0": AxisRole.PARALLEL},
    )
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(2048,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 16
    """Target children: the original producer + the moved consumer
    (directly, no regen wrapper)."""
    moved = [c for c in new_target.children if isinstance(c, BodyLeaf) and c.reads == {"data": "p"}]
    assert len(moved) == 1
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert not regen_wrappers


def test_reverse_apply_raises_on_indivisible_residual() -> None:
    """Ancestor trip does not divide num_tiles → AtomLegalityError."""
    from nkigym.tune import AtomLegalityError

    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        writes=("p",),
        dim_role={"d0": AxisRole.PARALLEL},
    )
    producer_outer = LoopNode("d0", 3, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        dim_role={"d0": AxisRole.PARALLEL},
    )
    dims = {"d0": DimInfo(dim_id="d0", total_size=17 * 128, tile_size=128, num_tiles=17)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(17 * 128,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    try:
        atom.apply(module)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for indivisible residual")
```

- [ ] **Step 2: Run and verify GREEN**

Run: `pytest test/tune/test_reverse_compute_at.py -v`
Expected: all green (Task 4 already implemented the mirror).

- [ ] **Step 3: Commit**

```bash
git add test/tune/test_reverse_compute_at.py
git commit -m "test: reverse_computeat partial-coverage regen trio"
```

---

## Task 6: Red test — Split rejects non-divisor factor

**Files:**
- Modify: `test/tune/test_split.py`

- [ ] **Step 1: Inspect existing tail-sibling tests**

Run: `sed -n '30,65p' /home/ubuntu/nki-autotune/test/tune/test_split.py`
Expected: see `test_split_non_divisible_yields_two_siblings` and `test_split_factor_greater_than_trip` — both will be deleted in Task 7, not now.

- [ ] **Step 2: Add divisor-guard tests**

Append to `test/tune/test_split.py`:

```python
def test_split_is_legal_rejects_non_divisor() -> None:
    """is_legal returns False when factor does not divide trip_count."""
    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 17, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    assert not atom.is_legal(mod)


def test_split_apply_raises_on_non_divisor() -> None:
    """apply raises AtomLegalityError when factor does not divide trip_count."""
    from nkigym.tune import AtomLegalityError

    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 17, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    try:
        atom.apply(mod)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for non-divisor factor")
```

- [ ] **Step 3: Run, verify RED**

Run: `pytest test/tune/test_split.py::test_split_is_legal_rejects_non_divisor test/tune/test_split.py::test_split_apply_raises_on_non_divisor -v`
Expected: FAIL on both — current `is_legal` only checks `factor >= 1`; current `apply` handles non-divisor via tail-siblings.

- [ ] **Step 4: Commit**

```bash
git add test/tune/test_split.py
git commit -m "test: red for split non-divisor rejection"
```

---

## Task 7: Implement Split divisor guard + delete tail-sibling path

**Files:**
- Modify: `nkigym/src/nkigym/tune/split.py:28-66`
- Modify: `test/tune/test_split.py` (delete obsolete tail-sibling tests)

- [ ] **Step 1: Update `Split.is_legal`**

Replace lines 28-36 of `nkigym/src/nkigym/tune/split.py`:

```python
    def is_legal(self, module: KernelIR) -> bool:
        """Target must be a LoopNode; ``factor`` must be a positive divisor of ``trip_count``."""
        result: bool
        if self.factor < 1:
            result = False
        else:
            target = resolve_node(module.body, self.loop_path)
            if not isinstance(target, LoopNode):
                result = False
            else:
                result = target.trip_count % self.factor == 0
        return result
```

- [ ] **Step 2: Update `Split.apply`**

Replace lines 38-66 with:

```python
    def apply(self, module: KernelIR) -> KernelIR:
        """Replace target with a single outer × inner pair.

        Rejects non-divisor factors via :class:`AtomLegalityError` — the
        old tail-sibling emission path is gone. Tail-siblings violated
        the 1N invariant (sibling subtrees with the same dim and
        mismatched trips make downstream atoms brittle; see spec
        `docs/superpowers/specs/2026-05-08-canonical-1N-and-computeat-partial-coverage-design.md`).
        Canonical-rename runs across the whole body after replacement,
        matching the post-apply contract of ``ComputeAt`` et al.
        """
        from nkigym.tune import AtomLegalityError
        from nkigym.tune.compute_at import _rename_canonical

        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, LoopNode)
        n = target.trip_count
        f = self.factor
        if n % f != 0:
            raise AtomLegalityError(
                f"Split.apply: factor {f} does not divide trip_count {n} at {self.loop_path}"
            )
        outer_trip = n // f
        replacement = [_make_split_pair(target, outer_trip=outer_trip, inner_trip=f)]
        new_body = _replace_with_siblings(module.body, self.loop_path, replacement)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)
```

- [ ] **Step 3: Delete obsolete tail-sibling tests**

From `test/tune/test_split.py`, delete:
- `test_split_non_divisible_yields_two_siblings`
- `test_split_factor_greater_than_trip`

These assertions (two siblings with different trips) are no longer reachable semantics.

- [ ] **Step 4: Run test_split, verify GREEN**

Run: `pytest test/tune/test_split.py -v`
Expected: all green. Task 6's red tests now pass; obsolete tests deleted.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/split.py test/tune/test_split.py
git commit -m "fix: split rejects non-divisor factor via AtomLegalityError"
```

---

## Task 8: Migrate canonical builder to 1N

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py:626-696`

- [ ] **Step 1: Rewrite `_wrap_dims`**

Replace lines 626-641:

```python
def _wrap_dims(
    wrap: tuple[str, ...], op: _ParsedOp, dims: dict[str, DimInfo], inner_children: list[LoopNode | BodyLeaf]
) -> LoopNode:
    """Wrap ``inner_children`` in a 1N-per-dim chain over ``wrap``.

    Each dim contributes one ``LoopNode`` with ``trip_count = num_tiles``.
    The outer wrapper matches the dim ordering in ``touched_dims``
    (outermost first). Arithmetic-intensity dials (tiles-per-block)
    live in ``Split``, not here.
    """
    if not wrap:
        raise ValueError(f"Op {op.idx}: cannot build tree — no touched_dims to wrap")
    node_children: list[LoopNode | BodyLeaf] = inner_children
    for d in reversed(wrap):
        role = op.dim_role[d]
        num_t = dims[d].num_tiles
        block_node = LoopNode(dim_id=d, trip_count=num_t, role=role, children=node_children)
        node_children = [block_node]
    head = node_children[0]
    assert isinstance(head, LoopNode)
    return head
```

- [ ] **Step 2: Rewrite `_build_leaves_matmul`**

Replace lines 662-677:

```python
def _build_leaves_matmul(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """Matmul: ``[psum_init leaf, <K chain ending in compute leaf>, drain leaf]``.

    The outer M and N dims are consumed by ``_wrap_dims``. The K dim is
    handled here so the body placement mirrors the physical kernel:
    PSUM init lives outside K, ``nc_matmul`` fires inside K, drain runs
    after K closes. The K loop carries ``reduce_op="add"`` because
    nc_matmul's PSUM accumulator is summation.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = dims[k_dim].num_tiles
    compute_leaf = _make_leaf(op, "compute")
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[compute_leaf], reduce_op="add")
    return [_make_leaf(op, "psum_init"), k_block, _make_leaf(op, "drain")]
```

- [ ] **Step 3: Rewrite `_build_leaves_activation_reduce`**

Replace lines 680-696:

```python
def _build_leaves_activation_reduce(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce Pattern 2: ``[<F loop with reduce_step>, reduce_close]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: one F loop holds the per-tile ``reduce_step`` BodyLeaf that
    writes each tile's partial sum into a distinct slot of the op-local
    ``slot_vec``. After the F loop exits, ``reduce_close`` folds the
    slot vector via one ``nisa.tensor_reduce`` into the op's ``(P, 1)``
    output.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = _make_leaf(op, "reduce_step")
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    return [f_block, _make_leaf(op, "reduce_close")]
```

- [ ] **Step 4: Run canonical + downstream tests, verify GREEN**

Run: `pytest test/codegen/test_canonical.py -v`
Expected: most tests green. If any assertion hardcodes `len(km.body[i].children) == 1` expecting a trip-1 child, update the assertion — the new canonical has no trip-1 partner so the body leaf is the direct child of the block loop. Fix inline by updating the assertion to match the new structure.

Run: `pytest test/codegen/ -v`
Expected: may surface broken assertions in `test_place_buffers.py`, `test_software_pipeline_unit.py`, etc. For each failure, read the assertion, compare against the new 1N structure, and update the assertion to match. Commit updates in Task 9 below.

- [ ] **Step 5: Commit the canonical source change**

```bash
git add nkigym/src/nkigym/codegen/canonical.py
git commit -m "refactor: canonical builder emits 1N trees"
```

(Test fixture updates happen in Task 9 as a focused commit.)

---

## Task 9: Update test fixtures for 1N canonical

**Files:**
- Modify any `test/codegen/*.py` or `test/tune/*.py` whose structural assertions break after Task 8.

- [ ] **Step 1: Run the full suite to collect breakages**

Run: `pytest test/ -v 2>&1 | tail -80`
Expected: list of failing tests. Go through each one:

- If the test constructs a `KernelIR` directly via `_mod([...])` or similar, it doesn't go through `build_initial_ir` — unaffected.
- If the test calls `build_initial_ir` and asserts on tree shape, update the assertion.

Likely affected:
- `test/codegen/test_canonical.py` — any `km.body[i]` walk expecting two-tier wrap.
- `test/codegen/test_place_buffers.py` — if it uses `build_initial_ir`.
- `test/codegen/test_dep_cache.py` — same.
- `test/tune/test_end_to_end_template_kernel.py` — end-to-end pipeline may tolerate but depth changes.

- [ ] **Step 2: Update each broken assertion inline**

For each failure, load the test, find the assertion, update it to match the new 1N shape. Examples:
- `assert outer.children[0].trip_count == 1` → remove assertion or change to `assert outer.children[0] is leaf`.
- `assert len(wrap_chain) == 2 * num_dims` → `assert len(wrap_chain) == num_dims`.

Do NOT change any test that was asserting *behaviour* (e.g. "ComputeAt successfully moves leaf"). Only update tests asserting 2N *structure*.

- [ ] **Step 3: Run the full suite**

Run: `pytest test/ -v`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add test/
git commit -m "test: update fixture structural assertions for 1N canonical"
```

---

## Task 10: Drop unreachable trip-1 descent branch in pipeline emitter

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py:247-274`

- [ ] **Step 1: Simplify `_emit_pipelined_leaf`**

In `inject_software_pipeline.py`, delete the `if child.trip_count == 1:` branch and its whole trivial-descent block (~lines 251-274 in current file). The remaining `else` branch (non-trivial descent, emits the `for` header) becomes the unconditional path for LoopNode children.

The cleaned function should look like (schematically, after the LoopNode recursion entry):

```python
    if isinstance(child, LoopNode):
        if not _subtree_has_firing_leaf(child, stages, fire_if_stage_le, fire_if_stage_gt):
            return
        existing = path_names.setdefault(child.dim_id, [])
        loop_var = child.name if child.name is not None else f"i_{child.dim_id}_{len(existing)}"
        w.line(f"for {loop_var} in range({child.trip_count}):")
        w.indent()
        existing.append(loop_var)
        path_trips.setdefault(child.dim_id, []).append(child.trip_count)
        try:
            for grandchild in child.children:
                _emit_pipelined_leaf(
                    w,
                    module,
                    grandchild,
                    path_names,
                    path_trips,
                    stages,
                    dim_id,
                    constant_loop_var,
                    fire_if_stage_le,
                    fire_if_stage_gt,
                )
        finally:
            path_trips[child.dim_id].pop()
            existing.pop()
            w.dedent()
        return
```

Update the function's docstring to remove the paragraph about "Trivial ``trip_count == 1`` LoopNodes skip the ``for`` header" — this case is no longer reachable.

- [ ] **Step 2: Run software-pipeline tests**

Run: `pytest test/codegen/test_software_pipeline_unit.py -v`
Expected: green. If any test constructed a hand-built trip-1 LoopNode inside a pipelined subtree to exercise the deleted branch, delete that test — the branch is unreachable and the test no longer serves a purpose.

- [ ] **Step 3: Run the full suite**

Run: `pytest test/ -v`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/codegen/lowering/inject_software_pipeline.py test/codegen/test_software_pipeline_unit.py
git commit -m "refactor: drop unreachable trip-1 descent in pipelined emitter"
```

---

## Task 11: End-to-end verification with `debug_kernel_0000_chain.py`

**Files:**
- Verify: `debug_kernel_0000_chain.py` (no code change; just run)

- [ ] **Step 1: Clean the dump directory**

Run: `rm -f /home/ubuntu/cache/matmul_lhsT_rhs_debug/step_*.py /home/ubuntu/cache/matmul_lhsT_rhs_debug/chain_summary.md`

- [ ] **Step 2: Rerun the chain replay**

Run: `source ~/venvs/kernel-env/bin/activate && python /home/ubuntu/nki-autotune/debug_kernel_0000_chain.py 2>&1 | tee /tmp/kernel_0000_chain_after_fix.log | tail -40`
Expected: chain may still match kernel_0000 byte-for-byte OR may produce a different byte sequence (the 1N migration changes `hash_state` so pool exploration can diverge — that's allowed per spec). What matters:
- **Step 2 (`ReverseComputeAt`) is now `ok`** (was `DIVERGED` with max_abs=nan). This is the Bug A fix evidence.
- Step 15 (`SoftwarePipeline`) may still be `DIVERGED` — Bug B is out of scope.

- [ ] **Step 3: If step 2 is still DIVERGED, debug**

If step 2 reports `DIVERGED`, inspect `/home/ubuntu/cache/matmul_lhsT_rhs_debug/step_02_ReverseComputeAt_DIVERGED.py` and compare against step 1's output. The store leaf should have a `for i_d1_1 in range(8):` (or similar residual) wrapping its inner dma_copy. If the residual loop is missing, return to Task 4 and verify the `_wrap_leaf_with_dims` + `_ancestor_trip_products` wiring.

- [ ] **Step 4: Full test suite sanity check**

Run: `pytest test/ -v 2>&1 | tail -10`
Expected: all green.

- [ ] **Step 5: No commit required**

This task is verification only.

---

## Task 12: Update followups doc + learnings

**Files:**
- Modify: `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md`
- Modify: `.claude/rules/learnings.md`

- [ ] **Step 1: Update followups progress table**

In `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md`, after the progress table (line 12), add a new landed row:

```markdown
| Bug A — ComputeAt/ReverseComputeAt partial-coverage regen | Landed | <paste final SHA> |
| Canonical 1N migration + tail-sibling ban | Landed | <paste final SHA> |
```

Also add the corresponding entries to the "New followups surfaced" section:

```markdown
- **Bug A fixed, 1N canonical landed.** `_ancestor_dims` (presence-only) replaced with
  `_ancestor_trip_products` (per-dim multiplicative accumulation). Canonical builder
  emits 1N trees; `Split` rejects non-divisor factors. Tail-siblings no longer appear
  in any atom's output.
```

- [ ] **Step 2: Append learning entries**

Append to the "Architecture" or "Workflow" section of `.claude/rules/learnings.md`:

```markdown
- **Canonical 1N + tail-sibling ban**: canonical tree emits one `LoopNode` per (dim, nest-position) — no trip-1 partner. AI-dial (tiles-per-block) lives exclusively in `Split`. `Split.apply` raises `AtomLegalityError` on non-divisor factor. `ComputeAt`/`ReverseComputeAt` regeneration uses `_ancestor_trip_products` (multiplicative, not set-union), producing `L(d, num_tiles/covered)` residual when partial coverage arises and raising on indivisible. Bug A fix — step 2 of kernel_tuned_0000 chain no longer NaN. *(2026-05-08)*
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-08-ir-refactor-followups.md .claude/rules/learnings.md
git commit -m "docs: followups + learnings for bug A fix and 1N canonical"
```

- [ ] **Step 4: Backfill commit SHAs into followups doc**

Run: `git log --oneline -20`
Copy the SHAs for the Task 4 and Task 8 commits into the progress table rows added above. Amend:

```bash
git add docs/superpowers/plans/2026-05-08-ir-refactor-followups.md
git commit --amend --no-edit
```

---

## Success Criteria Checklist

- [ ] `pytest test/` passes (all tests green, structural assertions updated for 1N).
- [ ] `test/tune/test_split.py` contains new divisor-guard tests; tail-sibling tests deleted.
- [ ] `test/tune/test_compute_at.py` contains partial-coverage trio.
- [ ] `test/tune/test_reverse_compute_at.py` contains partial-coverage trio.
- [ ] `debug_kernel_0000_chain.py` reports step 2 as `ok` (max_abs ≈ 1.3e-4).
- [ ] Canonical 2048³ matmul CPU-sim unchanged (max_abs ≈ 1.3e-4).
- [ ] `_ancestor_dims` deleted; `_ancestor_trip_products` lives in `compute_at.py`.
- [ ] Followups doc + learnings updated.
