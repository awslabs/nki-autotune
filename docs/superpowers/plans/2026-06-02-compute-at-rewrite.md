# ComputeAt / ReverseComputeAt Greenfield Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ComputeAt` (sink a producer) and `ReverseComputeAt` (lift a consumer) as greenfield transforms with TVM-style region-regen move mechanics, reproducing all 7 hand-ladder code-motion rungs byte-exact plus one new partial-coverage rung.

**Architecture:** Delete the pre-`normalize_block` move engine. Build a shared `_move(ir, block_nid, target_loop_nid, index, is_reverse)` that derives each moved-block iter-var's domain from the required region (`SolveBlockVarDomain`-analogue over the affine `Expr` AST), regenerates residual loops, splices under the target, then reconciles via `normalize_block` → `place_buffers` → `compact_shapes` → `Dependency`. `ComputeAt`/`ReverseComputeAt` are thin `Transform` faces differing only in dependency-check direction (consumers vs producers) and the forward-only output-block guard. Full coverage is the degenerate solver case (every covered axis solves to the target var, no residual).

**Tech Stack:** Python 3.12, networkx, pytest, numpy, NKI CPU simulator. Hand-rolled affine `Expr` AST (`nkigym/ir/expr.py`, shipped). No new external deps.

**Spec:** `docs/superpowers/specs/2026-06-02-compute-at-rewrite-design.md`.

**Legality reference:** `nkigym/src/nkigym/transforms/compute_at_legality.md` (six conditions; refreshed in Task 1).

**Environment:** `source ~/venvs/kernel-env/bin/activate`; run from repo root `/home/ubuntu/nki-autotune`; tests run with `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src`. A `check-python-style.py` hook runs on `.py` edits; the commit hook runs `autoflake`/`isort`/`black`.

---

## Key API facts (verified against current code — read before starting)

- **`KernelTree`** (`nkigym/ir/tree.py`): `add_node(data, parent=None) -> int`,
  `data(nid)`, `children(nid) -> list[int]`, `parent(nid) -> int | None`,
  `ancestors(nid) -> list[int]` (root-first), `descendants(nid) -> set[int]`,
  `preorder(nid=None)`, `blocks(nid=None)`, `.root`, `.graph` (nx.DiGraph;
  payload at `graph.nodes[nid]["data"]`). Edges parent→child; child order =
  successor insertion order.
- **Payloads** (all `@dataclass(frozen=True, kw_only=True)`):
  - `ForNode(loop_var: str, extent: int)`.
  - `ISANode(op_cls, operand_bindings: dict[str, BufferRegion], kwargs: dict)`.
  - `IterVar(axis: str, dom: tuple[int,int], role: AxisRole)`.
  - `BufferRegion(tensor: str, ranges: tuple[tuple[Expr, Expr], ...])` —
    each range is `(lo, width)`; **width is a `Const`** (not `hi`).
  - `BlockNode(iter_vars, iter_values, reads, writes, alloc_buffers=(),
    annotations={}, axis_map={})`. `iter_values[i]` is the affine `Expr`
    binding `iter_vars[i]` to surrounding loop_var symbols.
  - `Buffer(name, shape, dtype, location)`; `physical_shape()`,
    `physical_dtype()`.
- **expr** (`nkigym/ir/expr.py`): `to_affine(expr) -> dict[str|None,int]`
  (constant under key `None`; raises `NonAffineError`), `from_affine(coeffs)`,
  `substitute(expr, {name: Expr})`, `Const(value=)`, `Var(name=)`,
  `Add/Mul(left=,right=)`, `format_expr`.
- **`normalize_block(tree, block_nid)`** (`transforms/_normalize.py`): drops
  trip-1 ForNodes, dense-renames loop vars to `i_{dim}_{N}`, recomputes
  `iter_values` + every region `lo` from surviving loops. Call it on the
  **fork block** after a move.
- **`place_buffers(tree)`** (`ir/buffer_placement.py`): LCA-of-touchers
  placement, in place, idempotent.
- **`compact_shapes(tree)`** + **`rebased_region(region, buf, tree)`**
  (`codegen/compact.py`): bbox shape materialization + read-time anchor
  subtraction. Renderer already calls `rebased_region`.
- **`Dependency(tree)`** (`ir/dependency.py`): `producers(nid) -> set[int]`,
  `consumers(nid) -> set[int]`, `must_precede(p, c)`. `KernelIR.dependency`
  holds the instance; rebuild with `Dependency(new_ir.tree)`.
- **`_tree_ops`**: `_replace_in_parent_children(tree, parent, old, new)`
  (order-preserving), `_block_local_descendants(tree, block_nid)` (does NOT
  enter sub-blocks).
- **`Transform`** base (`transforms/base.py`): override `analyze(ir) ->
  list`, `apply(ir, option) -> KernelIR`. `TransformLegalityError`,
  `TransformOption` (frozen marker).
- **`KernelIR`**: `.tree`, `.dependency`, `.return_name`. `apply` pattern:
  `self._check_legality(ir, option); new = copy.deepcopy(ir); <mutate
  new.tree>; new.dependency = Dependency(new.tree); return new`.
- **Fixtures** (`test/transforms/_fixtures.py`): `build_canonical_ir() ->
  KernelIR`, `INPUT_SPECS`, `K=M=N=2048`. Canonical buffer names:
  `sbuf_lhs_T`, `sbuf_rhs`, `psum_prod`, `sbuf_prod`, `hbm_out`; kernel fn
  `nki_f_matmul`. The hand ladder (`kernel_transforms.py`) uses `psum_acc`
  and `kernel_N` — the byte oracle normalizes these two name skews.

---

## File Structure

New:
- `nkigym/src/nkigym/transforms/_domain_solve.py` — the region-regen core:
  `required_region`, `solve_iter_domains`, `regen_and_rebind`. Pure
  functions over tree + Expr; the one genuinely new piece.
- `nkigym/src/nkigym/transforms/compute_at.py` — `ComputeAt`,
  `ComputeAtOption`, forward legality (`_check_consumers_visited` + output
  guard).
- `test/transforms/_ladder_compare.py` — `assert_matches_hand(rendered_src,
  hand_fn)`: normalize-then-equate byte oracle.
- `test/transforms/test_compute_at.py` — forward byte-exact rungs + legality
  + partial-coverage.

Rewritten:
- `nkigym/src/nkigym/transforms/_code_motion.py` — `_move(...)` over
  `_domain_solve` (replaces the bare-Var collapse engine).
- `nkigym/src/nkigym/transforms/reverse_compute_at.py` — thin face over
  `_move`; keep the existing (correct) `_check_producers_visited` /
  `_root_sibling_of`.
- `test/transforms/test_reverse_compute_at.py` — byte-exact reverse rungs +
  PSUM-hoist + legality.

Modified:
- `nkigym/src/nkigym/transforms/__init__.py` — export `ComputeAt`,
  `ComputeAtOption`.
- `nkigym/src/nkigym/transforms/compute_at_legality.md` — drop
  `init`/`NKIAlloc`/`dst=`; refresh "What is a block?".
- `kernel_transforms.py` — add one partial-coverage hand kernel
  (`kernel_partial`).

---

## Phase outline

1. Legality-doc refresh + delete the stale engine (Task 1).
2. Region-regen core `_domain_solve.py` (Tasks 2–4) — unit-tested in isolation.
3. Shared `_move` (Task 5).
4. `ReverseComputeAt` face + byte-oracle helper (Task 6).
5. `ComputeAt` face (Task 7); fill `build_ladder_state` + byte-exact all 7 rungs (Task 8).
6. Partial-coverage hand kernel + region-regen test (Task 9).
7. PSUM-hoist E2E + full regression (Task 10).

---

## Phase 1 — Refresh doc, delete stale engine

### Task 1: Refresh legality doc; delete the pre-normalize move engine

**Files:**
- Modify: `nkigym/src/nkigym/transforms/compute_at_legality.md`
- Delete: `nkigym/src/nkigym/transforms/_code_motion.py`,
  `nkigym/src/nkigym/transforms/reverse_compute_at.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py` (drop the
  ReverseComputeAt exports temporarily — re-added in Task 6)

- [ ] **Step 1: Refresh `compute_at_legality.md`**

In the "What is a block?" section, delete the `init` bullet (lines ~28–29:
"`init` — optional sub-block...") and the two code blocks showing
`init=memset_block` / `└── memset_block(... init ...)`. Replace the
canonical-shape code block with the decomposed-canonical form (memset is a
*sibling* block, no `init`):

```text
root_block(alloc_buffers=[sbuf_lhs_T, sbuf_rhs, psum_prod, sbuf_prod, hbm_out])
├── memset_block  → ForNode(i_d1_0) → ForNode(i_d2_0) → ISANode(NKIMemset)
└── matmul_block  → ForNode(i_d0_0) → ForNode(i_d1_0) → ForNode(i_d2_0) → ISANode(NKIMatmul)
```

In the condition-2 examples and condition-5 examples, replace every
`NKIMatmul()(... dst=psum_prod)` and `NKIMemset(value=0.0)(dst=psum_prod)`
with the SSA form `psum_prod = NKIMatmul()(stationary=..., moving=...)` /
`psum_prod = NKIMemset(value=0.0)()`. Delete the "Note on allocs" block
(lines ~337–343) referencing `NKIAlloc` — buffers are no longer ISA leaves.
The six conditions and the quick-lookup table stay unchanged.

- [ ] **Step 2: Delete the stale engine and its exports**

```bash
git rm nkigym/src/nkigym/transforms/_code_motion.py nkigym/src/nkigym/transforms/reverse_compute_at.py
```

In `nkigym/src/nkigym/transforms/__init__.py`, remove the
`from nkigym.transforms.reverse_compute_at import ...` line and the
`"ReverseComputeAt"`, `"ReverseComputeAtOption"` entries from `__all__`.
Also `git rm test/transforms/test_code_motion.py
test/transforms/test_reverse_compute_at.py` (both rewritten later; the old
ones test deleted internals).

- [ ] **Step 3: Verify the suite is green without the deleted modules**

Run: `source ~/venvs/kernel-env/bin/activate && PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest -q`
Expected: PASS (the only removals are the two transforms + their tests; nothing else imports them — verify with `grep -rn "reverse_compute_at\|_code_motion" nkigym/ test/` returning nothing).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Delete pre-normalize compute_at engine; refresh legality doc for SSA IR"
```

---

## Phase 2 — Region-regen core (`_domain_solve.py`)

The core models one moved block's relocation as: (a) which axes the target's
enclosing loops cover, (b) for each moved-block iter-var, the domain it must
sweep, (c) regenerate residual ForNodes + rebind. All three are pure
functions over the tree and the affine `Expr` AST, unit-testable without a
full transform.

### Task 2: `axis_loops` helpers — map dims to loops on both sides

**Files:**
- Create: `nkigym/src/nkigym/transforms/_domain_solve.py`
- Test: `test/transforms/test_domain_solve.py`

The move needs, for the moved block and for the target's enclosing nest, a
map *concrete dim → ordered list of (loop_var, extent)*. Unlike the deleted
`_loop_var_to_axis` (bare-Var only), this reads `iter_values` via `to_affine`
so it works on tiled (affine) bindings.

- [ ] **Step 1: Write the failing test**

Create `test/transforms/test_domain_solve.py`:

```python
"""Tests for nkigym.transforms._domain_solve."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._domain_solve import dim_loops_of_block, enclosing_dim_loops


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _leaf_in(ir, block_nid: int) -> int:
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ISANode):
            return d
    raise AssertionError("no ISA leaf")


def test_dim_loops_of_block_canonical_matmul():
    """The canonical matmul block owns d0,d1,d2 loops, each trip 16/16/4."""
    ir = build_canonical_ir()
    mm = _block_for_op(ir, "NKIMatmul")
    loops = dim_loops_of_block(ir.tree, mm)
    assert set(loops) == {"d0", "d1", "d2"}
    assert [e for _v, e in loops["d0"]] == [16]
    assert [e for _v, e in loops["d2"]] == [4]


def test_enclosing_dim_loops_of_matmul_inner_loop():
    """The dims covered at/above the matmul's innermost loop are d0,d1,d2."""
    ir = build_canonical_ir()
    mm = _block_for_op(ir, "NKIMatmul")
    leaf = _leaf_in(ir, mm)
    innermost = ir.tree.ancestors(leaf)[-1]
    assert isinstance(ir.tree.data(innermost), ForNode)
    enclosing = enclosing_dim_loops(ir.tree, innermost)
    assert set(enclosing) == {"d0", "d1", "d2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q`
Expected: FAIL with `ModuleNotFoundError: nkigym.transforms._domain_solve`.

- [ ] **Step 3: Implement the two helpers**

Create `nkigym/src/nkigym/transforms/_domain_solve.py`:

```python
"""Region-regen core for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop. These pure functions
derive, from the affine ``iter_values`` of both the moved block and the
target's enclosing nest, which dims the target covers and what residual
domain each moved iter-var must sweep, then regenerate residual ForNodes
and rebind the moved block's regions. Works on tiled (affine) bindings,
not only bare-Var ones.
"""

from __future__ import annotations

from math import prod

from nkigym.ir.expr import Expr, from_affine, to_affine
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants


def _loopvar_to_dim(tree: KernelTree, block_nid: int) -> dict[str, str]:
    """Map each loop_var the block binds to its concrete dim, via iter_values.

    A loop_var binds the iter_var whose iter_value affine mentions it
    (iter_values are affine over a single dim's loops). Works for tiled
    bindings (``i_d1_0*512 + i_d1_1*128``), not just bare Vars.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                out[name] = iv.axis
    return out


def dim_loops_of_block(tree: KernelTree, block_nid: int) -> dict[str, list[tuple[str, int]]]:
    """Map each concrete dim to its block-owned ForNodes as ``(loop_var, extent)`` outer→inner."""
    lv_to_dim = _loopvar_to_dim(tree, block_nid)
    out: dict[str, list[tuple[str, int]]] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in lv_to_dim:
            out.setdefault(lv_to_dim[data.loop_var], []).append((data.loop_var, data.extent))
    return out


def enclosing_dim_loops(tree: KernelTree, target_loop_nid: int) -> dict[str, list[tuple[str, int]]]:
    """Map each concrete dim to the ForNodes at/above ``target_loop_nid`` within its block.

    Walks ``[target_loop_nid, *ancestors]`` up to (not into) the enclosing
    block, reading that block's loopvar→dim map. Outer→inner order.
    """
    block_nid = _enclosing_block(tree, target_loop_nid)
    lv_to_dim = _loopvar_to_dim(tree, block_nid)
    chain = [target_loop_nid, *reversed(tree.ancestors(target_loop_nid))]
    out: dict[str, list[tuple[str, int]]] = {}
    for nid in chain:
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in lv_to_dim:
            out.setdefault(lv_to_dim[data.loop_var], []).insert(0, (data.loop_var, data.extent))
    return out


def _enclosing_block(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor of ``nid``."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


__all__ = ["dim_loops_of_block", "enclosing_dim_loops"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/_domain_solve.py test/transforms/test_domain_solve.py
git commit -m "Add _domain_solve dim-loop helpers (affine-aware, both sides)"
```

### Task 3: `solve_iter_domains` — covered vs residual per dim

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_domain_solve.py`
- Test: `test/transforms/test_domain_solve.py`

For each dim the moved block iterates, compare its total trip-product against
the target's enclosing trip-product on that dim:
- target covers `≥` the dim → the dim is fully covered: it binds to the
  target's loops, no residual.
- target covers a proper divisor → residual = `moved_product //
  target_product`, regenerated as one residual ForNode.
- target doesn't iterate the dim → the whole dim stays residual.

This returns, per dim, `(covered_loops_from_target, residual_extent)`. Full
coverage ⇒ `residual_extent == 1` (no residual loop). The "not a clean
divisor" case raises loudly.

- [ ] **Step 1: Write the failing test**

Append to `test/transforms/test_domain_solve.py`:

```python
def test_solve_iter_domains_full_cover():
    """Moved d1 trip 16, target enclosing d1 trip 16 -> covered, residual 1."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 16)], "d0": [("i_d0_0", 16)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d1"].residual_extent == 1
    assert solved["d1"].target_loops == [("i_d1_0", 16)]


def test_solve_iter_domains_partial_cover_residual():
    """Moved d1 trip 16, target covers trip 4 -> residual 4."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 4)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d1"].residual_extent == 4
    assert solved["d1"].target_loops == [("i_d1_0", 4)]


def test_solve_iter_domains_uncovered_dim_all_residual():
    """A moved dim the target does not iterate stays fully residual."""
    from nkigym.transforms._domain_solve import solve_iter_domains

    moved = {"d2": [("i_d2_0", 4)]}
    target = {"d1": [("i_d1_0", 16)]}
    solved = solve_iter_domains(moved, target)
    assert solved["d2"].residual_extent == 4
    assert solved["d2"].target_loops == []


def test_solve_iter_domains_indivisible_raises():
    """A target coverage that does not divide the moved extent is illegal."""
    import pytest

    from nkigym.transforms._domain_solve import DomainSolveError, solve_iter_domains

    moved = {"d1": [("i_d1_0", 16)]}
    target = {"d1": [("i_d1_0", 5)]}
    with pytest.raises(DomainSolveError, match="divide"):
        solve_iter_domains(moved, target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q -k solve_iter`
Expected: FAIL with `ImportError: cannot import name 'solve_iter_domains'`.

- [ ] **Step 3: Implement `solve_iter_domains`**

Append to `_domain_solve.py` (add `from dataclasses import dataclass` at top):

```python
class DomainSolveError(ValueError):
    """Raised when a target's coverage does not cleanly divide a moved dim."""


@dataclass(frozen=True)
class DimDomain:
    """How one moved-block dim is re-domained under the target.

    Attributes:
        target_loops: the target's enclosing ``(loop_var, extent)`` on this
            dim that the moved dim binds to (empty if the target doesn't
            iterate the dim).
        residual_extent: trip count of the residual loop regenerated below
            the insertion point (1 = fully covered, no residual loop).
    """

    target_loops: list[tuple[str, int]]
    residual_extent: int


def solve_iter_domains(
    moved: dict[str, list[tuple[str, int]]],
    target: dict[str, list[tuple[str, int]]],
) -> dict[str, DimDomain]:
    """Per moved dim, split its iteration into target-covered + residual.

    ``moved`` / ``target`` are ``dim_loops_of_block`` / ``enclosing_dim_loops``
    outputs. For each moved dim, ``moved_product`` is the product of its
    trips; ``target_product`` the product of the target's trips on that dim
    (1 if absent). Requires ``target_product`` to divide ``moved_product``;
    residual = ``moved_product // target_product``.
    """
    out: dict[str, DimDomain] = {}
    for dim, loops in moved.items():
        moved_product = prod(e for _v, e in loops)
        target_loops = target.get(dim, [])
        target_product = prod(e for _v, e in target_loops)
        if moved_product % target_product != 0:
            raise DomainSolveError(
                f"dim {dim}: target coverage {target_product} does not divide moved extent {moved_product}"
            )
        out[dim] = DimDomain(target_loops=target_loops, residual_extent=moved_product // target_product)
    return out
```

Update `__all__` to add `DomainSolveError`, `DimDomain`, `solve_iter_domains`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/_domain_solve.py test/transforms/test_domain_solve.py
git commit -m "Add solve_iter_domains: covered/residual split per moved dim"
```

### Task 4: `regen_and_rebind` — regenerate residual loops + rebind regions

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_domain_solve.py`
- Test: `test/transforms/test_domain_solve.py`

Given the solved domains, mutate the moved block in place: drop its existing
ForNodes (all of them — covered and residual alike), regenerate ONE residual
ForNode per dim with `residual_extent > 1` (named with a temporary stem;
`normalize_block` dense-renames after splice), and rebind the block's
`iter_values` + reads + writes + leaf operand_bindings so each covered dim
binds to its target loop var and each residual dim binds to its new residual
loop var. The actual binding math is deferred to `normalize_block` (which
recomputes iter_values + region lo's from the surviving loops); this function
only sets up the loop topology + the iter_values skeleton `normalize_block`
expects (one Var per surviving loop).

The key simplification: after this function leaves the moved block with
exactly its residual ForNodes (chained) and iter_values referencing the
target loop vars (covered) and residual loop vars (residual),
`normalize_block` on the fork block recomputes everything correctly — the
same contract Split/Fuse rely on.

- [ ] **Step 1: Write the failing test**

Append to `test/transforms/test_domain_solve.py`:

```python
def test_regen_and_rebind_full_cover_drops_all_loops():
    """Full-coverage move: the moved block keeps no ForNodes (all covered)."""
    from nkigym.ir.tree import ForNode
    from nkigym.transforms._domain_solve import (
        DimDomain,
        dim_loops_of_block,
        regen_and_rebind,
    )

    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")  # first load (lhs_T): dims d0, d1
    solved = {
        dim: DimDomain(target_loops=[(f"i_{dim}_0", e[0][1])], residual_extent=1)
        for dim, e in dim_loops_of_block(ir.tree, load).items()
    }
    regen_and_rebind(ir.tree, load, solved)
    remaining = [d for d in ir.tree.descendants(load) if isinstance(ir.tree.data(d), ForNode)]
    assert remaining == []


def test_regen_and_rebind_residual_keeps_one_loop():
    """Partial cover (residual 4 on d1) leaves exactly one residual ForNode on d1."""
    from nkigym.ir.tree import ForNode
    from nkigym.transforms._domain_solve import (
        DimDomain,
        dim_loops_of_block,
        regen_and_rebind,
    )

    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    loops = dim_loops_of_block(ir.tree, load)
    solved = {
        "d0": DimDomain(target_loops=[("i_d0_0", 16)], residual_extent=1),
        "d1": DimDomain(target_loops=[("i_d1_0", 4)], residual_extent=4),
    }
    regen_and_rebind(ir.tree, load, solved)
    remaining = [ir.tree.data(d) for d in ir.tree.descendants(load) if isinstance(ir.tree.data(d), ForNode)]
    assert len(remaining) == 1
    assert remaining[0].extent == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q -k regen`
Expected: FAIL with `ImportError: cannot import name 'regen_and_rebind'`.

- [ ] **Step 3: Implement `regen_and_rebind`**

Append to `_domain_solve.py`:

```python
def regen_and_rebind(tree: KernelTree, block_nid: int, solved: dict[str, DimDomain]) -> None:
    """Drop the moved block's ForNodes; regenerate residual loops; rebind iter_values.

    After this, the block's body is reached through one residual ForNode per
    dim with ``residual_extent > 1`` (chained outer→inner in iter_vars order),
    and ``iter_values`` bind each dim's iter_var to the affine over its target
    loops (covered) plus its residual loop var. The block's reads/writes and
    leaf operand_bindings keep their tensor structure; ``normalize_block``
    (called by the caller after splice) recomputes the region ``lo`` offsets
    from the surviving loops. This function only fixes loop topology + the
    iter_values skeleton.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)

    """Detach the block's body leaf (single ISA leaf) from its current loop chain."""
    body_leaf = _single_body_leaf(tree, block_nid)
    _strip_block_loops(tree, block_nid)

    """Regenerate residual ForNodes (one per dim with residual_extent > 1), chained."""
    residual_vars: dict[str, str] = {}
    parent = block_nid
    for iv in block.iter_vars:
        dom = solved.get(iv.axis)
        if dom is None or dom.residual_extent <= 1:
            continue
        loop_var = f"i_{iv.axis}__resid"
        new_for = tree.add_node(ForNode(loop_var=loop_var, extent=dom.residual_extent), parent=parent)
        residual_vars[iv.axis] = loop_var
        parent = new_for
    tree.graph.add_edge(parent, body_leaf)

    """Rebuild iter_values: covered dims -> affine over target loops; residual -> its loop var;
    both -> sum. normalize_block recomputes region lo's from these."""
    new_values: list[Expr] = []
    for iv in block.iter_vars:
        dom = solved.get(iv.axis)
        new_values.append(_dim_binding(dom, residual_vars.get(iv.axis)))
    tree.graph.nodes[block_nid]["data"] = replace(block, iter_values=tuple(new_values))


def _dim_binding(dom: "DimDomain | None", residual_var: str | None) -> Expr:
    """Affine binding for one dim: target-loop affine + residual-loop term.

    target loops ``l_0(t_0)..l_{k-1}(t_{k-1})`` contribute ``Σ l_j * (Π
    inner-target-trips * residual_extent)``; the residual loop contributes
    ``+ residual_var``. With no target loops and no residual the dim is
    loopless (``Const(0)``). The exact element scaling is re-derived by
    ``normalize_block``; here we only need every surviving loop var to appear
    so the loopvar→dim map is recoverable.
    """
    coeffs: dict[str | None, int] = {None: 0}
    if dom is not None:
        inner = dom.residual_extent
        for loop_var, extent in reversed(dom.target_loops):
            coeffs[loop_var] = inner
            inner *= extent
    if residual_var is not None:
        coeffs[residual_var] = 1
    return from_affine(coeffs)


def _single_body_leaf(tree: KernelTree, block_nid: int) -> int:
    """Return the one ISA leaf in the block's local scope."""
    leaves = [n for n in _block_local_descendants(tree, block_nid) if isinstance(tree.data(n), ISANode)]
    if len(leaves) != 1:
        raise DomainSolveError(f"block {block_nid} must have exactly one ISA leaf; got {len(leaves)}")
    return leaves[0]


def _strip_block_loops(tree: KernelTree, block_nid: int) -> None:
    """Remove every ForNode in the block's local scope, leaving the leaf detached."""
    for nid in _block_local_descendants(tree, block_nid):
        if isinstance(tree.data(nid), ForNode):
            tree.graph.remove_node(nid)
```

Add `from dataclasses import replace` at the top. Update `__all__` to add
`regen_and_rebind`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_domain_solve.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/_domain_solve.py test/transforms/test_domain_solve.py
git commit -m "Add regen_and_rebind: strip loops, regenerate residuals, skeleton iter_values"
```

---

## Phase 3 — Shared `_move`

### Task 5: `_move` — splice + reconcile

**Files:**
- Create: `nkigym/src/nkigym/transforms/_code_motion.py` (re-created)
- Test: `test/transforms/test_code_motion.py` (re-created)

Compose the Task 2–4 pieces into the structural move: solve domains,
regen+rebind, splice the moved block under the target at `index`, then
`normalize_block` on the **fork block** (the block whose subtree now holds
both the moved block and the target).

- [ ] **Step 1: Write the failing test**

Create `test/transforms/test_code_motion.py`:

```python
"""Tests for nkigym.transforms._code_motion._move (structural move)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _move


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _innermost_for(ir, block_nid: int) -> int:
    leaf = next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))
    return ir.tree.ancestors(leaf)[-1]


def test_move_lifts_tensor_copy_under_matmul_inner_loop():
    """Lifting tensor_copy under the matmul's innermost loop nests it there."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _innermost_for(ir, mm)
    _move(ir, block_nid=tc, target_loop_nid=target, index=-1, is_reverse=True)
    assert tc in ir.tree.descendants(target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_code_motion.py -q`
Expected: FAIL with `ModuleNotFoundError: nkigym.transforms._code_motion`.

- [ ] **Step 3: Implement `_move`**

Create `nkigym/src/nkigym/transforms/_code_motion.py`:

```python
"""Shared structural move for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop via region-regen: solve each
moved-block dim into target-covered + residual (``_domain_solve``),
regenerate residual loops + rebind, splice under the target at ``index``,
then ``normalize_block`` reconciles names / trip-1 / region offsets on the
fork block. Direction (``is_reverse``) does not change the structural steps;
the caller's legality check differs.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, KernelTree
from nkigym.transforms._domain_solve import (
    dim_loops_of_block,
    enclosing_dim_loops,
    regen_and_rebind,
    solve_iter_domains,
)
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _replace_in_parent_children


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` in place (region-regen).

    Caller has checked legality and deep-copied. ``index`` follows TVM
    convention: ``-1`` append, ``-2`` prepend, ``>=0`` explicit slot among
    the target loop's children. ``is_reverse`` is structurally inert.
    """
    tree = ir.tree
    moved = dim_loops_of_block(tree, block_nid)
    target = enclosing_dim_loops(tree, target_loop_nid)
    solved = solve_iter_domains(moved, target)
    regen_and_rebind(tree, block_nid, solved)
    _splice_under_target(tree, block_nid, target_loop_nid, index)
    fork = _enclosing_block_of(tree, target_loop_nid)
    normalize_block(tree, fork)


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach ``block_nid`` from its parent and insert under the target loop at ``index``."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    children = tree.children(target_loop_nid)
    if index == -1:
        pos = len(children)
    elif index == -2:
        pos = 0
    else:
        pos = index
    new_order = children[:pos] + [block_nid] + children[pos:]
    for child in children:
        tree.graph.remove_edge(target_loop_nid, child)
    for child in new_order:
        tree.graph.add_edge(target_loop_nid, child)


def _enclosing_block_of(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor of ``nid`` (the fork block)."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


__all__ = ["_move"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_code_motion.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py test/transforms/test_code_motion.py
git commit -m "Add _move: region-regen splice + normalize_block reconcile"
```

---

## Phase 4 — ReverseComputeAt face + byte oracle helper

### Task 6: `ReverseComputeAt` face + byte oracle helper

**Files:**
- Create: `nkigym/src/nkigym/transforms/reverse_compute_at.py` (re-created)
- Create: `test/transforms/_ladder_compare.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`
- Test: `test/transforms/test_reverse_compute_at.py` (re-created)

Re-create the thin face: port the *correct* legality logic from the deleted
file (`_check_producers_visited`, `_root_sibling_of`, the ForNode/descendant
guards), call `_move(..., is_reverse=True)`, then `place_buffers` →
`compact_shapes` → `Dependency`. Also build the byte oracle helper.

- [ ] **Step 1: Write the byte-oracle helper**

Create `test/transforms/_ladder_compare.py`:

```python
"""Byte-exact comparison of rendered transform output vs a hand kernel.

Normalize-then-equate: strip comments/blanks, black-format both sides, and
rename-normalize the two known skews (kernel function name and the
accumulator buffer psum_prod/psum_acc) to fixed placeholders. Everything
else must match character-for-character.
"""

from __future__ import annotations

import inspect
import re

import black


def _normalize(src: str) -> str:
    src = re.sub(r"#.*", "", src)
    src = re.sub(r"\bnki_f_\w+\b", "KFN", src)
    src = re.sub(r"\bkernel_\w+\b", "KFN", src)
    src = re.sub(r"\bpsum_acc\b", "PACC", src)
    src = re.sub(r"\bpsum_prod\b", "PACC", src)
    formatted = black.format_str(src, mode=black.Mode())
    return "\n".join(line for line in formatted.splitlines() if line.strip())


def assert_matches_hand(rendered_src: str, hand_fn) -> None:
    """Assert ``rendered_src`` equals ``hand_fn``'s source after normalization."""
    hand_src = inspect.getsource(hand_fn)
    got = _normalize(rendered_src)
    want = _normalize(hand_src)
    assert got == want, f"rendered != hand kernel\n--- got ---\n{got}\n--- want ---\n{want}"
```

- [ ] **Step 2: Write the failing test (legality + canonical lift render+sim)**

Create `test/transforms/test_reverse_compute_at.py`. The byte-exact reverse
rungs (k11→k12, k13→k14) need before-states reached via forward ComputeAt
moves, which don't exist until Task 7 — so they land in Task 8 (the ladder
task) once `build_ladder_state` is wired. **This task** ships the face plus
what is provable now: legality rejections + a canonical full-extent lift that
renders and fp32-sims. Write:

```python
"""Tests for nkigym.transforms.ReverseComputeAt."""

from __future__ import annotations

from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import ReverseComputeAt, ReverseComputeAtOption, TransformLegalityError


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _first_for_in(ir, block_nid: int) -> int:
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ForNode):
            return d
    raise AssertionError("no ForNode")


def test_reverse_rejects_non_fornode_target():
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=mm, index=-1))


def test_reverse_rejects_target_inside_moved_block():
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    own = _first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="descendant|ancestor|own"):
        ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=own, index=-1))


def test_reverse_lift_tensor_copy_under_matmul_renders_and_sims():
    """Full-extent lift of tensor_copy under the matmul M-loop renders + sims."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    m_loop = _first_for_in(ir, mm)
    new_ir = ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=m_loop, index=-1))
    assert tc in new_ir.tree.descendants(m_loop)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    import importlib.util
    import tempfile, pathlib
    src = render(new_ir)
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_reverse_compute_at.py -q`
Expected: FAIL — `ImportError: cannot import name 'ReverseComputeAt'`.

- [ ] **Step 4: Implement the face**

Create `nkigym/src/nkigym/transforms/reverse_compute_at.py`:

```python
"""``ReverseComputeAt`` — lift a consumer block under a producer's loop.

See ``compute_at_legality.md`` (conditions 1-3, 5b, 6). Structural move is
the shared ``_move``; this face owns the producer-direction legality.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _move
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReverseComputeAtOption(TransformOption):
    """Lift consumer ``block_nid`` under ``target_loop_nid`` at ``index``."""

    block_nid: int
    target_loop_nid: int
    index: int


class ReverseComputeAt(Transform):
    """Lift a consumer block under a producer's loop."""

    def apply(self, ir: KernelIR, option: ReverseComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, lift, re-derive geometry, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid,
              index=option.index, is_reverse=True)
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[ReverseComputeAtOption]:
        """Enumerate (consumer, target loop, index) triples passing legality."""
        options: list[ReverseComputeAtOption] = []
        leaf_blocks = [
            nid for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                for index in self._legal_indices(ir, block_nid, target_nid):
                    opt = ReverseComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    try:
                        self._check_legality(ir, opt)
                    except TransformLegalityError:
                        continue
                    options.append(opt)
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children."""
        children = ir.tree.children(target_nid)
        producers = ir.dependency.producers(block_nid)
        consumers = ir.dependency.consumers(block_nid)
        lp = -1
        fc = len(children)
        for i, child in enumerate(children):
            sub = ir.tree.descendants(child) | {child}
            if sub & producers:
                lp = i
            if sub & consumers and i < fc:
                fc = i
        return list(range(lp + 1, fc + 1))

    def _check_legality(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Conditions 1-3, 5b. (6 enumerated by _legal_indices.)"""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"ReverseComputeAt requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot lift under its own loop)"
            )
        self._check_producers_visited(ir, option)

    def _check_producers_visited(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Condition 5b: every producer is under the target OR an earlier root-sibling."""
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for producer in ir.dependency.producers(option.block_nid):
            if producer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(producer):
                continue
            producer_root = self._root_sibling_of(ir, producer)
            if producer_root not in root_order:
                raise TransformLegalityError(f"producer block {producer} not under a root-sibling")
            if root_order.index(producer_root) < target_index:
                continue
            raise TransformLegalityError(
                f"producer block {producer} runs after the target loop "
                f"(root index {root_order.index(producer_root)} >= target {target_index})"
            )

    @staticmethod
    def _root_sibling_of(ir: KernelIR, nid: int) -> int:
        """Return the direct child of tree.root that is nid or an ancestor of it."""
        if nid in ir.tree.children(ir.tree.root):
            return nid
        for anc in ir.tree.ancestors(nid):
            if anc in ir.tree.children(ir.tree.root):
                return anc
        raise TransformLegalityError(f"node {nid} has no root-sibling ancestor")


__all__ = ["ReverseComputeAt", "ReverseComputeAtOption"]
```

In `nkigym/src/nkigym/transforms/__init__.py` re-add the import and the two
`__all__` entries (`ReverseComputeAt`, `ReverseComputeAtOption`).

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_reverse_compute_at.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Add ReverseComputeAt face (region-regen _move) + byte-oracle helper"
```

## Phase 5 — ComputeAt + byte-exact rungs

### Task 7: `ComputeAt` face

**Files:**
- Create: `nkigym/src/nkigym/transforms/compute_at.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`
- Test: `test/transforms/test_compute_at.py`

Forward face: mirror `ReverseComputeAt` but check **consumers** (5a) and add
the **output-block guard** (condition 4). Structural move is the same `_move`
with `is_reverse=False`.

- [ ] **Step 1: Write the failing test (legality + canonical sink render+sim)**

Create `test/transforms/test_compute_at.py`:

```python
"""Tests for nkigym.transforms.ComputeAt."""

from __future__ import annotations

from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import ComputeAt, ComputeAtOption, TransformLegalityError


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _first_for_in(ir, block_nid: int) -> int:
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ForNode):
            return d
    raise AssertionError("no ForNode")


def test_compute_at_rejects_sinking_output_store():
    """Condition 4: sinking the store (writes hbm_out = return) is illegal."""
    ir = build_canonical_ir()
    store = _block_for_op(ir, "NKIStore")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _first_for_in(ir, mm)
    with pytest.raises(TransformLegalityError, match="output|return"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=store, target_loop_nid=target, index=-1))


def test_compute_at_rejects_non_fornode_target():
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=mm, index=-1))


def test_compute_at_sink_load_under_matmul_renders_and_sims():
    """Sink lhs_T load under the matmul's inner loop; full coverage; render + sim."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    leaf = next(d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ISANode))
    inner = ir.tree.ancestors(leaf)[-1]
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=inner, index=-2))
    assert load in new_ir.tree.descendants(inner)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    import importlib.util, tempfile, pathlib
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(new_ir))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_compute_at.py -q`
Expected: FAIL — `ImportError: cannot import name 'ComputeAt'`.

- [ ] **Step 3: Implement the forward face**

Create `nkigym/src/nkigym/transforms/compute_at.py` mirroring
`reverse_compute_at.py`, with these differences:
- `ComputeAtOption(block_nid, target_loop_nid, index)`.
- `apply` calls `_move(..., is_reverse=False)`.
- `_check_legality` adds condition 4 BEFORE the producer/consumer check:

```python
        moved = ir.tree.data(option.block_nid)
        leaf = next(
            (ir.tree.data(d) for d in ir.tree.descendants(option.block_nid)
             if isinstance(ir.tree.data(d), ISANode)), None,
        )
        if leaf is not None:
            for region in leaf.operand_bindings.values():
                if region.tensor == ir.return_name:
                    raise TransformLegalityError(
                        f"ComputeAt cannot sink the output block (writes return {ir.return_name})"
                    )
```

- and replaces `_check_producers_visited` with `_check_consumers_visited`
  (condition 5a — mirror: every CONSUMER is under target OR a LATER
  root-sibling, `>` instead of `<`):

```python
    def _check_consumers_visited(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Condition 5a: every consumer is under target OR a later root-sibling."""
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for consumer in ir.dependency.consumers(option.block_nid):
            if consumer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(consumer):
                continue
            consumer_root = self._root_sibling_of(ir, consumer)
            if consumer_root not in root_order:
                raise TransformLegalityError(f"consumer block {consumer} not under a root-sibling")
            if root_order.index(consumer_root) > target_index:
                continue
            raise TransformLegalityError(
                f"consumer block {consumer} runs before the target loop "
                f"(root index {root_order.index(consumer_root)} <= target {target_index})"
            )
```

`analyze`, `_legal_indices`, `_root_sibling_of` are identical to
`ReverseComputeAt` (copy them; DRY is acceptable to break here — the two
faces are intentionally parallel and small). Re-export `ComputeAt`,
`ComputeAtOption` in `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_compute_at.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Add ComputeAt face: consumer-direction legality + output guard"
```

### Task 8: Fill `build_ladder_state`; byte-exact all 7 rungs

**Files:**
- Modify: `test/transforms/_fixtures.py`
- Test: `test/transforms/test_compute_at.py`, `test/transforms/test_reverse_compute_at.py`

Now both transforms exist. Build `build_ladder_state(n)` ONE rung at a time,
sim-gating each, because two parameters of each move CANNOT be known
statically and MUST be derived empirically: (a) the `index` insertion slot,
and (b) the exact loop-var names after the previous rung's `normalize_block`
dense-rename. Do NOT guess these — derive them by the procedure below.

**The verified rung table** (transform + locator per the `kernel_transforms.py`
sources; loop vars are the canonical/dense names at that rung):

| n→n+1 | transform | locator (op, axis/loop) | params |
|---|---|---|---|
| 0→1 | Split | load_lhsT leaf, `target_axis="d1"` | `factors=(16,128)` |
| 1→2 | ComputeAt | load_lhsT block → matmul innermost loop | full cover d0,d1 |
| 2→3 | Split | load_rhs leaf, `target_axis="d2"` | `factors=(4,512)` |
| 3→4 | ComputeAt | load_rhs block → matmul `d2` loop | full cover d0,d2 |
| 4→5 | Split | memset leaf, `target_axis="d2"` | `factors=(4,512)` |
| 5→6 | Reorder | matmul outer pair `d0`↔`d1` | adjacent swap |
| 6→7 | ComputeAt | memset block → matmul `d1` loop | cover d1; d2 residual |
| 7→8 | ComputeAt ×2 | load_lhsT then load_rhs block → matmul inner `d2` loop | two applications |
| 8→9 | Reorder | matmul `d0`↔`d2` | adjacent swap |
| 9→10 | ComputeAt | memset block → matmul `d2` loop | cover d2 |
| 10→11 | Split | tensor_copy leaf, `target_axis="d2"` | `factors=(4,512)` |
| 11→12 | ReverseComputeAt | tensor_copy block → matmul `d2` loop | the PSUM hoist |
| 12→13 | Split | store leaf, `target_axis="d2"` | `factors=(4,512)` |
| 13→14 | ReverseComputeAt | store block → tensor_copy `d2` loop | — |

Note 7→8 is the one **multi-move** rung (two `ComputeAt` applications in one
ladder step). All Split/Reorder rungs are already-shipped transforms.

- [ ] **Step 1: Add the locator helpers + an empty rung list**

Append to `test/transforms/_fixtures.py`:

```python
def _ladder_helpers():
    """Return (blk, leaf, loop, inner) target-locators bound to a fresh closure."""
    from nkigym.ir.tree import ForNode, ISANode

    def blk(ir, op_name, which=0):
        found = [
            nid for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
            and ir.tree.data(next(d for d in ir.tree.descendants(nid)
                                  if isinstance(ir.tree.data(d), ISANode))).op_cls.__name__ == op_name
        ]
        return found[which]

    def leaf(ir, block_nid):
        return next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))

    def loop(ir, block_nid, loop_var):
        return next(d for d in ir.tree.preorder(block_nid)
                    if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == loop_var)

    def inner(ir, block_nid):
        return ir.tree.ancestors(leaf(ir, block_nid))[-1]

    return blk, leaf, loop, inner


def build_ladder_state(n: int) -> KernelIR:
    """Replay the kernel_transforms.py transform sequence from canonical to kernel_n.

    Rungs are appended one at a time (see plan Task 8). Each lambda takes ir,
    applies exactly one transform, returns the new ir. Raises NotImplementedError
    for rungs not yet wired so a partial build fails loud.
    """
    from nkigym.transforms import (  # noqa: F401
        ComputeAt, ComputeAtOption, Reorder, ReorderOption,
        ReverseComputeAt, ReverseComputeAtOption, Split, SplitOption,
    )

    blk, leaf, loop, inner = _ladder_helpers()
    rungs: list = []
    """--- rungs appended below, one per Step-2 iteration ---"""
    if n > len(rungs):
        raise NotImplementedError(f"build_ladder_state({n}): only {len(rungs)} rungs wired")
    ir = build_canonical_ir()
    for rung in rungs[:n]:
        ir = rung(ir)
    return ir
```

- [ ] **Step 2: Derive and append each rung 0→1 … 13→14, sim-gating every one**

For each row of the rung table, in order, do this loop (do NOT batch):

  1. Append the rung lambda to `rungs` using the locators. Split/Reorder
     params come straight from the table. For each `ComputeAt`/
     `ReverseComputeAt`, leave `index=-2` (earliest legal) for a producer
     sink and `index=-1` (latest legal) for a consumer lift as the first
     guess.
  2. Render `build_ladder_state(n+1)` and CPU-sim it against the numpy
     golden (reuse the render+sim block from Task 6's test). If sim fails,
     the move landed at the wrong slot or the wrong loop — inspect the
     rendered source vs `kernel_{n+1}`, adjust the `index` (it must be a slot
     in `analyze`'s legal gap) or the located loop var, and retry.
  3. Once it sims, assert byte-exact against the hand kernel (Step 3/4
     parametrize covers this) before appending the next rung.

The loop vars in the table are the dense names AFTER the prior rung's
`normalize_block`; if a `loop(...)` lookup raises StopIteration, print
`render(build_ladder_state(n))` to read the actual current loop var and use
that. This is expected — names are normalized, not guessed.

- [ ] **Step 3: Add byte-exact forward-rung tests**

Append to `test/transforms/test_compute_at.py`:

```python
import kernel_transforms as KT
from test.transforms._fixtures import build_ladder_state
from test.transforms._ladder_compare import assert_matches_hand


@pytest.mark.parametrize("before_n, hand", [(1, KT.kernel_2), (3, KT.kernel_4),
                                            (6, KT.kernel_7), (7, KT.kernel_8), (9, KT.kernel_10)])
def test_compute_at_rung_byte_exact(before_n, hand):
    """Each forward ComputeAt rung reproduces its hand kernel byte-exact."""
    from nkigym.codegen import render
    # build_ladder_state(before_n) is the kernel_{before_n} state; the next
    # rung is the ComputeAt that yields kernel_{before_n+1} == hand.
    ir = build_ladder_state(before_n + 1)
    assert_matches_hand(render(ir), hand)
```

- [ ] **Step 3: Run forward byte-exact tests**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_compute_at.py -q`
Expected: all pass. If a rung mismatches, the diff in `assert_matches_hand`
shows exactly which line — fix in `_move`/`_domain_solve`, not the renderer.

- [ ] **Step 4: Add byte-exact reverse-rung tests**

Append to `test/transforms/test_reverse_compute_at.py`:

```python
import kernel_transforms as KT
from test.transforms._fixtures import build_ladder_state
from test.transforms._ladder_compare import assert_matches_hand


@pytest.mark.parametrize("before_n, hand", [(11, KT.kernel_12), (13, KT.kernel_14)])
def test_reverse_rung_byte_exact(before_n, hand):
    """Each ReverseComputeAt rung reproduces its hand kernel byte-exact."""
    from nkigym.codegen import render
    ir = build_ladder_state(before_n + 1)
    assert_matches_hand(render(ir), hand)
```

- [ ] **Step 5: Run reverse byte-exact tests**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_reverse_compute_at.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Byte-exact reproduction of all 7 ladder code-motion rungs"
```

---

## Phase 6 — Partial coverage + PSUM hoist + regression

### Task 9: Partial-coverage hand kernel + region-regen byte oracle

**Files:**
- Modify: `kernel_transforms.py`
- Test: `test/transforms/test_compute_at.py`

The 7 ladder rungs are all full coverage. Add ONE partial-coverage hand
kernel as the byte oracle for residual regeneration: a load whose own loop is
`range(16)` sunk under a target covering `range(4)`, leaving a `range(4)`
residual.

- [ ] **Step 1: Add the hand kernel**

Append to `kernel_transforms.py` a `kernel_partial(lhs_T, rhs)` that is
`kernel_1` (load_lhsT split to d1 `range(16)`) with the load sunk under a
matmul M-loop that has been Split to `(4,4)` — so the load's d1 `range(16)`
sinks under the outer `range(4)`, regenerating a `range(4)` residual inside.
Write the explicit nki source (modeled on `kernel_2` but with the residual
inner `for i_d1_resid in range(4)` over the load), and add it to `_main`'s
sim loop so it is CPU-sim-verified.

- [ ] **Step 2: Write the failing byte-exact test**

Append to `test/transforms/test_compute_at.py`:

```python
def test_compute_at_partial_coverage_byte_exact():
    """A range(16) load sunk under a range(4) target regenerates a range(4) residual."""
    import kernel_transforms as KT
    from nkigym.codegen import render
    from nkigym.transforms import Split, SplitOption

    # canonical -> split load d1 (16,128) -> split matmul M-loop (4,4) -> ComputeAt load under matmul M_outer(range 4)
    ir = build_canonical_ir()
    load0 = _block_for_op(ir, "NKILoad")
    leaf0 = next(d for d in ir.tree.preorder(load0) if isinstance(ir.tree.data(d), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=leaf0, factors=(16, 128), target_axis="d1"))
    mm = _block_for_op(ir, "NKIMatmul")
    m_loop = next(d for d in ir.tree.preorder(mm)
                  if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0")
    ir = Split().apply(ir, SplitOption(target_nid=m_loop, factors=(4, 4)))
    mm = _block_for_op(ir, "NKIMatmul")
    m_outer = next(d for d in ir.tree.preorder(mm)
                   if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0")
    load0 = _block_for_op(ir, "NKILoad")
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load0, target_loop_nid=m_outer, index=-2))
    assert_matches_hand(render(new_ir), KT.kernel_partial)
```

- [ ] **Step 3: Run; iterate on `_move` until byte-exact + the hand kernel sims**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_compute_at.py::test_compute_at_partial_coverage_byte_exact -q`
and `python kernel_transforms.py` (the partial kernel must print `pass=True`).
Expected: both green. The diff pinpoints any residual-regen mismatch.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Add partial-coverage hand kernel + region-regen byte-exact test"
```

### Task 10: PSUM-hoist assertion + full regression

**Files:**
- Test: `test/transforms/test_reverse_compute_at.py`

- [ ] **Step 1: Write the PSUM-hoist assertion test**

Append to `test/transforms/test_reverse_compute_at.py`:

```python
def test_psum_hoist_descends_and_compacts():
    """After k11->k12, psum_prod is declared inside the matmul block and compacted to one tile."""
    from test.transforms._fixtures import build_ladder_state
    from nkigym.ir.tree import BlockNode

    ir = build_ladder_state(12)
    decls = {buf.name: (nid, buf) for nid in ir.tree.blocks()
             for buf in ir.tree.data(nid).alloc_buffers}
    nid, buf = decls["psum_prod"]
    assert nid != ir.tree.root, "psum_prod did not descend from root"
    assert buf.shape == (128, 512), f"psum_prod not compacted to one tile: {buf.shape}"
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_reverse_compute_at.py::test_psum_hoist_descends_and_compacts -q`
Expected: PASS. (If `buf.shape` differs, the spec's PSUM-hoist claim or
`compact_shapes` interaction needs a fix — debug there, not by loosening the
assert.)

- [ ] **Step 3: Full regression + ladder sim**

Run:
```
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest -q
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python kernel_transforms.py
```
Expected: full suite green; every `kernel_N` (including `kernel_partial`)
prints `pass=True`.

- [ ] **Step 4: Wire both transforms into the example (optional smoke)**

In `examples/matmul_lhsT_rhs.py`, add `ComputeAt(), ReverseComputeAt()` to the
`KernelMDP(..., transforms=[...])` list. Run the example; confirm rollouts
stay numerically correct (every `[numerics] PASS`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "PSUM-hoist assertion; wire ComputeAt/ReverseComputeAt into MDP"
```

---

## Out of scope

- multi_buffer, software_pipeline, decompose/compose_reduction.
- Multi-leaf compound blocks (each block keeps one ISA leaf).
- General >2-factor residual regeneration beyond the one partial-coverage
  oracle (the solver handles it; only one hand kernel pins it).
