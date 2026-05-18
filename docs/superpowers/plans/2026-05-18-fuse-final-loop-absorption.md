# Fuse Final-Loop Absorption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the renderer to lower an ISA leaf with zero enclosing loops on a dim when its `tensorize_size` equals the full axis extent, making Fuse and Split symmetric and unblocking the random-policy MDP smoke test.

**Architecture:** Drop the "ISA leaves require ≥1 enclosing loop per touched dim" branch from `_check_axis_coverage` so the universal coverage rule (`product(loop trips) * tensorize_size == axis_extent`) is the sole legality constraint. Adjust `_render_tensor_slice` to default the slice coord to `"0"` when no enclosing loops on a dim contribute terms. Two-line production change plus a regression test.

**Tech Stack:** Python 3.12, pytest, networkx-backed `KernelTree`. Kernel virtualenv at `~/venvs/kernel-env`.

**Spec:** `docs/superpowers/specs/2026-05-18-fuse-final-loop-absorption-design.md`

---

## File Plan

| Path | Responsibility |
|---|---|
| `nkigym/src/nkigym/codegen/body.py` | Drop the "ISA leaves require enclosing loop" branch in `_check_axis_coverage`; default `coord` to `"0"` for empty terms in `_render_tensor_slice`; switch to `enclosing_loops.get(..., [])` pattern; rewrite the affected `_check_axis_coverage` docstring lines |
| `test/codegen/test_body.py` | Add `test_zero_enclosing_loops_with_full_extent_tensorize` regression test; verify the existing `test_axis_extent_mismatch_raises` still passes |
| `examples/matmul_lhsT_rhs.py` | Re-run the smoke at `NUM_ROLLOUTS=32, MAX_STEPS=8`; revert knobs to `4, 5` once green |

---

## Sequencing

1. **Task 1** — Add the regression test. The test fails on current `body.py` (raises the to-be-removed AssertionError).
2. **Task 2** — Apply the two-line fix in `body.py` (drop the ISA-leaf-needs-loop branch, default `coord` to `"0"`, swap to `.get(...)`). Test passes.
3. **Task 3** — Smoke run on `examples/matmul_lhsT_rhs.py` with bumped knobs; revert; final-suite check.

---

## Common Setup

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
```

---

### Task 1: Add the regression test for zero-loop full-extent tensorize

**Files:**
- Modify: `test/codegen/test_body.py` (append a new test before `test_axis_extent_mismatch_raises`, around line 285)

The matmul fixture (`_matmul` at line 42-56 of `test_body.py`) already builds a 2048×2048×2048 matmul IR exposing every compute op. We mutate the `NKIStore` leaf so its F-axis tensorize is the full extent (2048) and its enclosing d2-dim ForNode is removed — the post-Fuse final-loop-absorption shape.

- [ ] **Step 1: Write the failing test**

Add this test to `test/codegen/test_body.py` immediately before `def test_axis_extent_mismatch_raises()` (around line 285). The fixture scaffolding (`_matmul`, `_MATMUL_INPUT_SPECS`, the `NKIStore` import, `build_initial_ir`, `emit_body`, `tree`, `ForNode`, `ISANode`) is already in scope earlier in the file.

```python
def test_zero_enclosing_loops_with_full_extent_tensorize() -> None:
    """An ISA leaf with ``tensorize_size == axis_extent`` and zero enclosing
    loops on that dim renders correctly.

    Fuse can absorb every enclosing ForNode on a dim when the resulting
    tensorize_size reaches the axis extent. Codegen must accept that
    shape (universal coverage rule: ``Π trips × tensorize == extent``).
    Mutate the ``NKIStore`` leaf to ``tensorize_sizes['F'] == 2048`` and
    detach its d2-dim ForNode chain — the same shape Fuse produces in
    the random-policy MDP rollouts. ``emit_body`` must not raise, and
    the rendered F slice must spell as the full-axis range starting at 0.
    """
    ir = build_initial_ir(_matmul, _MATMUL_INPUT_SPECS)
    tree = ir.tree
    store_isa_nid = next(
        nid for nid in tree.preorder() if isinstance(tree.data(nid), ISANode) and tree.data(nid).op_cls is NKIStore
    )
    store = tree.data(store_isa_nid)
    f_dim = store.axis_map["F"]

    """Walk the store's enclosing chain and detach every f_dim ForNode,
    reconnecting the chain so the store sits directly under the next
    non-f_dim ancestor."""
    chain = list(tree.ancestors(store_isa_nid)) + [store_isa_nid]
    for ancestor_nid, child_nid in zip(chain, chain[1:]):
        ancestor = tree.data(ancestor_nid)
        if isinstance(ancestor, ForNode) and ancestor.dim == f_dim:
            grandparent = tree.parent(ancestor_nid)
            assert grandparent is not None
            tree.graph.remove_edge(grandparent, ancestor_nid)
            tree.graph.remove_edge(ancestor_nid, child_nid)
            tree.graph.add_edge(grandparent, child_nid)
            tree.graph.remove_node(ancestor_nid)

    """Bump the store's F tensorize to the full axis extent."""
    new_tensorize = dict(store.tensorize_sizes)
    new_tensorize["F"] = ir.dim_sizes[f_dim]
    tree.graph.nodes[store_isa_nid]["data"] = ISANode(
        op_cls=store.op_cls,
        reads=store.reads,
        writes=store.writes,
        rmw=store.rmw,
        tensorize_sizes=new_tensorize,
        axis_map=store.axis_map,
        kwargs=store.kwargs,
        location=store.location,
        dtype=store.dtype,
    )

    body = emit_body(ir)
    store_lines = [line for line in body.splitlines() if "nisa.store(" in line]
    assert store_lines, f"NKIStore call missing from body:\n{body}"
    """The store's F-axis slice must cover the full extent. With zero F-dim loops,
    the rendered slice expression for F resolves to 0:N (after black) or
    the literal ``(0)*N:(0+1)*N``; either way ``0`` and ``2048`` appear together."""
    for line in store_lines:
        assert "2048" in line, f"NKIStore slice should reference the full F extent (2048): {line!r}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest test/codegen/test_body.py::test_zero_enclosing_loops_with_full_extent_tensorize -v
```

Expected: FAIL with `AssertionError: NKIStore on axis d2: ISA leaf has no enclosing loop covering this dim`.

- [ ] **Step 3: Commit the failing test**

```bash
git add test/codegen/test_body.py
git commit -m "Add regression test for zero-loop full-extent tensorize"
```

(The commit lands a deliberately-failing test. The next task makes it pass.)

---

### Task 2: Drop the ≥1-loop rule from `_check_axis_coverage` and default empty `coord` to `"0"`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (lines 177-202 and 205-226)

- [ ] **Step 1: Drop the ISA-leaf-needs-loop branch in `_check_axis_coverage`**

Open `nkigym/src/nkigym/codegen/body.py`. Locate `_check_axis_coverage` (line 177). Replace lines 177-202 with:

```python
def _check_axis_coverage(node: ISANode, ir: KernelIR, enclosing_loops: dict[str, list[int]]) -> None:
    """Assert ``Π enclosing trips × tensorize_size == dim_extent`` for every axis in ``axis_map``.

    The coverage rule is universal: any combination of enclosing-loop
    trip product and ``tensorize_sizes[axis]`` whose product equals the
    axis extent is legal. Zero enclosing loops are allowed when the
    leaf's ``tensorize_size`` already covers the full extent (empty
    product = 1). Names the leaf's op class and the failing concrete
    dim so error messages point at the offending node.
    """
    for abstract_axis, concrete_axis in node.axis_map.items():
        enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
        trip_product = math.prod(enclosing_axis_loops)
        tensorize_size = node.tensorize_sizes[abstract_axis]
        axis_extent = ir.dim_sizes[concrete_axis]
        assert trip_product * tensorize_size == axis_extent, (
            f"{node.op_cls.__name__} on axis {concrete_axis}: trip product {trip_product} "
            f"* tensorize size {tensorize_size} != axis extent {axis_extent} "
            f"(enclosing trips {enclosing_axis_loops})"
        )
```

The diff:
- The `if node.op_cls is not NKIAlloc and concrete_axis not in enclosing_loops: raise AssertionError(...)` branch is gone.
- The docstring is rewritten — the second paragraph (about ISA leaves needing a loop variable to spell, even at trip 1) no longer applies.
- The line that reads `enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])` is unchanged.

The `NKIAlloc` import at the top of `body.py` stays — `emit_alloc` still uses it.

- [ ] **Step 2: Default `coord` to `"0"` and switch to `.get(...)` in `_render_tensor_slice`**

Locate `_render_tensor_slice` (line 205). Replace lines 211-220 (the per-axis loop body inside the function) with the version that handles empty `terms`:

```python
def _render_tensor_slice(
    node: ISANode, ir: KernelIR, kwarg: str, tensor_name: str, enclosing_loops: dict[str, list[int]]
) -> str:
    """Render a single operand as ``<name>[<slice>]`` per its location and axes."""
    axes = node.op_cls.OPERAND_AXES[kwarg]
    location = ir.tensors[tensor_name].location
    slice_strs: list[str] = []
    for counter, abstract_axis in enumerate(axes):
        concrete_axis = node.axis_map[abstract_axis]
        enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
        tensorize_size = node.tensorize_sizes[abstract_axis]
        terms = [
            f"i_{concrete_axis}_{cardinal}*{math.prod(enclosing_axis_loops[cardinal + 1 :])}"
            for cardinal in range(len(enclosing_axis_loops))
        ]
        coord = " + ".join(terms) if terms else "0"
        if counter == 0 and location != "shared_hbm":
            slice_strs.append(f"0:{tensorize_size}")
            slice_strs.append(f"{coord}")
        else:
            slice_strs.append(f"({coord})*{tensorize_size}:({coord}+1)*{tensorize_size}")
    return f"{tensor_name}[{', '.join(slice_strs)}]"
```

The diff against the previous version:
- Line 214 changed: `enclosing_axis_loops = enclosing_loops[concrete_axis]` → `enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])`. Same call surface as `_check_axis_coverage`.
- Line 220 changed: `coord = " + ".join(terms)` → `coord = " + ".join(terms) if terms else "0"`. When there are no enclosing loops on this dim, the slice uses literal coord `0`.

The downstream branches at lines 221-225 are unchanged.

- [ ] **Step 3: Run the regression test added in Task 1**

```bash
pytest test/codegen/test_body.py::test_zero_enclosing_loops_with_full_extent_tensorize -v
```

Expected: PASS.

- [ ] **Step 4: Confirm `test_axis_extent_mismatch_raises` still passes**

```bash
pytest test/codegen/test_body.py::test_axis_extent_mismatch_raises -v
```

Expected: PASS. The test mutates the matmul K-loop trip from 16 → 8 (1024 ≠ 2048); the line-198 assertion still fires.

- [ ] **Step 5: Run the full codegen + transforms + IR suites**

```bash
pytest test/codegen test/transforms test/ir -q
```

Expected: every test green.

- [ ] **Step 6: Confirm the renderer's existing tests still match the new slice spelling**

Several existing tests exercise the slice rendering on the canonical IR (`test_isa_calls_reference_i_dim_cardinal_loop_vars` at line 161, `test_split_loop_increments_cardinal_for_same_dim_ancestor` at line 202, `test_same_dim_ancestor_emits_linear_combination_in_slice` at line 230). Those leaves all have non-empty enclosing-loop chains, so `terms` is non-empty and `coord` retains the joined-loop-vars form. The default `"0"` only kicks in for genuinely empty chains. Confirm by running:

```bash
pytest test/codegen/test_body.py -v
```

Expected: every test green, including the regression test from Task 1.

- [ ] **Step 7: Commit the production fix**

```bash
git add nkigym/src/nkigym/codegen/body.py
git commit -m "Allow zero-loop full-extent tensorize coverage in renderer"
```

---

### Task 3: End-to-end smoke via `examples/matmul_lhsT_rhs.py`

**Files:**
- Modify: `examples/matmul_lhsT_rhs.py` (lines 39-40, then revert)

- [ ] **Step 1: Bump the rollout knobs**

In `examples/matmul_lhsT_rhs.py`:

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

Expected: every step's `[numerics] PASS` line prints. The earlier blocker (rollout 3 step 4 — Fuse absorbs all d2 loops on `NKIStore`) is no longer a codegen failure; the rendered kernel runs and matches numpy golden under fp32 sim.

If the run takes more than ~10 minutes, lower `NUM_ROLLOUTS` to 16 and rerun. Do not lower `MAX_STEPS` below 8 — the failing scenario surfaces around step 3-4.

- [ ] **Step 3: Revert the rollout knobs**

After the smoke is clean, revert `examples/matmul_lhsT_rhs.py` lines 39-40:

```python
NUM_ROLLOUTS = 4
MAX_STEPS = 5
```

- [ ] **Step 4: Confirm the revert is clean**

```bash
git diff examples/matmul_lhsT_rhs.py
```

Expected: no diff.

- [ ] **Step 5: Final full-suite check**

```bash
pytest test -q
```

Expected: every test green.

- [ ] **Step 6: No commit**

This task is verification-only. Tasks 1-2 already captured the fix.

---

## Self-Review

**Spec coverage:**
- §"Design" → `_check_axis_coverage` change — Task 2, Step 1.
- §"Design" → `_render_tensor_slice` change — Task 2, Step 2.
- §"Tests" → new regression test — Task 1.
- §"Tests" → `test_axis_extent_mismatch_raises` still passes — Task 2, Step 4.
- §"Tests" → smoke rerun — Task 3.

Every spec item has a task. No gaps.

**Placeholder scan:** No "TBD"/"TODO"/"implement later". Every code change shows full before/after. Every command shows expected output.

**Type consistency:** Function signatures (`_check_axis_coverage`, `_render_tensor_slice`) unchanged. Test fixture imports (`build_initial_ir`, `_matmul`, `_MATMUL_INPUT_SPECS`, `NKIStore`, `ISANode`, `ForNode`, `tree.ancestors`, `tree.parent`, `tree.children`, `tree.graph.remove_edge`/`add_edge`/`remove_node`/`nodes[...]["data"]`) all match the existing surface used by `test_split_d0_into_outer_8_inner_2` (line 172) and `test_axis_extent_mismatch_raises` (line 285).
