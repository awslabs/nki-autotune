# Fold-inlining Coverage Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine two over-conservative legality guards so the existing transform set can inline the RFactor fold into the matmul's innermost loop, driving canonical → a kernel_target-equivalent (closing the 46% → ~90% MFU gap).

**Architecture:** Both guards reject moves that the move's own region-regen makes legal. Barrier 2 (`ir/dependency.py`): the backward-edge check treats a frozen full-N COVER edge as a hazard even when the move rebinds the consumer to the covering loop — fix by skipping a COVER edge whose producer-loop the move re-covers. Barrier 1 (`transforms/_code_motion.py`): the ACCUMULATION-coverage guard rejects covering ANY reduction axis — fix by allowing coverage when the covering loop is the moved block's OWN carry loop (init dominates). Neither touches transform mechanics, only the two pure legality predicates.

**Tech Stack:** Python 3.12, networkx (sidecar dependency DiGraph), nkigym IR (`KernelTree`/`BlockNode`/`ForNode`/`ISANode`), CPU-sim + Trn2 HW via `transport/ssh_host.sh` to gym-1.

## Global Constraints

- **Transform legality = behavior/dep-order + ISA well-formedness ONLY; NEVER resource capacity.** A dep-legal kernel that over-subscribes PSUM/SBUF is a VALID transform output.
- **Loud failures only:** no silent raises, no `try/except` to adapt around malformed IR. Single return per function (no branching returns).
- **No new transform types.** Only the two legality predicates change; `_move`/`regen_and_rebind`/`solve_iter_domains` are untouched.
- **Code style:** triple-quoted block comments only (no `#`), modern type hints (`X | None`, `list`/`dict`), all functions typed + docstringed, `black` line-length 120, `isort`.
- **Compare loop NIDs, never loop_vars, for the B1 discriminator:** after RFactor a foreign K-loop and the fold's own ko-loop are both named `i_d0_0` (dim `d0`); a var comparison silently admits the foreign case → NaN.
- **Run tests on remote only:** dev box has no Python env. Unit tests via `transport/remote_pytest.sh` (`PYTHONPATH=.:nkigym/src:autotune/src`); sim/HW via `transport/ssh_host.sh --host gym-1 --cmd "python examples/kernel_transforms.py" --cache /home/weittang/workplace/cache/transforms`.
- **Controller owns all remote runs.** Subagents edit + commit only; the controller runs gym-1 sims/HW and reports results back.

## File Structure

- `nkigym/src/nkigym/ir/dependency.py` — MODIFY: `_first_backward` (COVER-skip), `first_backward_edge_for_insertion` + `first_backward_edge` (thread `skip_cover_loops`). One responsibility: the leaf-keyed dependency graph + ordering queries.
- `nkigym/src/nkigym/transforms/_code_motion.py` — MODIFY: `_check_move_realizable` (return `solved`), `_check_move_preserves_dependencies` (compute + pass covered-loop nids), `_check_no_reduction_axis_covered` (own-carry-loop discriminator). One responsibility: shared structural move + its pure legality predicates.
- `test/ir/test_dependency.py` — MODIFY: B2 unit test (COVER edge skipped iff loop is re-covered).
- `test/transforms/test_code_motion.py` — MODIFY: B1 unit tests (fold-move allowed; foreign-move + replication stay rejected — the golden-verdict gate).
- `examples/kernel_transforms.py` — MODIFY: rewrite `_build_ladder` to the N-outermost fold-inlining path (the matmul workload's single example file).

---

## Task 1: Barrier 2 — COVER-aware backward-edge check

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py` (`_first_backward`, `first_backward_edge`, `first_backward_edge_for_insertion`)
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` (`_check_move_realizable`, `_check_move_preserves_dependencies`)
- Test: `test/ir/test_dependency.py`

**Interfaces:**
- Consumes: `solve_iter_domains(moved, target) -> dict[str, DimDomain]` where `DimDomain.target_loops: list[tuple[str, int]]` (loop_var, extent) per covered dim; `enclosing_dim_loops(tree, target_loop_nid)`; `dim_loops_of_block(tree, block_nid)`.
- Produces: `_check_move_realizable(ir, block_nid, target_loop_nid) -> dict[str, DimDomain]` (now RETURNS solved). `Dependency.first_backward_edge_for_insertion(moved_leaf_nid, target_loop_nid, index, skip_cover_loops: frozenset[int] = frozenset())`. `skip_cover_loops` = set of producer-loop NIDs whose COVER edges into the moved leaf are dissolved by the move.

- [ ] **Step 1: Write the failing unit test for the COVER-skip**

Add to `test/ir/test_dependency.py`:

```python
def test_first_backward_skips_cover_edge_when_loop_recovered():
    """A COVER edge L->consumer is NOT backward when the move re-binds the
    consumer's covered dim to L (skip_cover_loops contains L). A RAW edge is
    never skipped."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    dep = Dependency(ir.tree)
    cover_edges = [(a, b) for a, b, d in dep.graph.edges(data=True) if d.get("kind") == "COVER"]
    assert cover_edges, "fixture must have at least one COVER edge for this test"
    loop_nid, consumer = cover_edges[0]
    assert isinstance(ir.tree.data(loop_nid), ForNode)
    leaf = consumer if isinstance(ir.tree.data(consumer), ISANode) else dep._resolve(consumer)

    """span = loop AFTER leaf so the COVER edge loop->leaf reads backward
    (span(loop).end < span(leaf).start is FALSE). Without the skip it is
    flagged; with loop in skip_cover_loops the COVER edge is dissolved -> None."""
    backward_span = lambda n: (1.0, 1.0) if n == loop_nid else (0.0, 0.0)  # noqa: E731
    without_skip = dep._first_backward(leaf, backward_span)
    with_skip = dep._first_backward(leaf, backward_span, skip_cover_loops=frozenset({loop_nid}))
    assert without_skip == (loop_nid, leaf)
    assert with_skip is None
```

- [ ] **Step 2: Run it to verify it fails**

Run (controller): `transport/remote_pytest.sh test/ir/test_dependency.py::test_first_backward_skips_cover_edge_when_loop_recovered`
Expected: FAIL — `_first_backward()` got an unexpected keyword argument `skip_cover_loops`.

- [ ] **Step 3: Add `skip_cover_loops` to `_first_backward`**

In `nkigym/src/nkigym/ir/dependency.py`, replace `_first_backward`:

```python
    def _first_backward(
        self,
        moved_leaf_nid: int,
        span: Callable[[int], tuple[float, float]],
        skip_cover_loops: frozenset[int] = frozenset(),
    ) -> tuple[int, int] | None:
        """Return the first edge incident to ``moved_leaf_nid`` that ``span`` ranks
        backward (``span(a).end < span(b).start`` violated), else ``None``.

        A ``COVER`` edge ``L -> moved_leaf_nid`` is SKIPPED when ``L`` is in
        ``skip_cover_loops`` — the move re-binds the moved block's covered dim to
        ``L``, dissolving the full-extent coverage that froze the edge. Only COVER
        edges are skippable; RAW/WAW/WAR/CARRY are real hazards and always checked.
        """
        result: tuple[int, int] | None = None
        for a, b, attrs in self.graph.edges(data=True):
            if a != moved_leaf_nid and b != moved_leaf_nid:
                continue
            if attrs.get("kind") == "COVER" and b == moved_leaf_nid and a in skip_cover_loops:
                continue
            if not (span(a)[1] < span(b)[0]):
                result = (a, b)
                break
        return result
```

- [ ] **Step 4: Thread `skip_cover_loops` through the two public queries**

In `first_backward_edge_for_insertion`, change the signature and final call:

```python
    def first_backward_edge_for_insertion(
        self, moved_leaf_nid: int, target_loop_nid: int, index: int, skip_cover_loops: frozenset[int] = frozenset()
    ) -> tuple[int, int] | None:
```

and its last line:

```python
        return self._first_backward(moved_leaf_nid, span, skip_cover_loops=skip_cover_loops)
```

In `first_backward_edge`, add the same parameter and pass it through:

```python
    def first_backward_edge(
        self, moved_leaf_nid: int, tree: KernelTree | None = None, skip_cover_loops: frozenset[int] = frozenset()
    ) -> tuple[int, int] | None:
```

and its last line:

```python
        return self._first_backward(moved_leaf_nid, span, skip_cover_loops=skip_cover_loops)
```

- [ ] **Step 5: Run the unit test to verify it passes**

Run (controller): `transport/remote_pytest.sh test/ir/test_dependency.py::test_first_backward_skips_cover_edge_when_loop_recovered`
Expected: PASS.

- [ ] **Step 6: Compute and pass `skip_cover_loops` from the move's legality check**

In `nkigym/src/nkigym/transforms/_code_motion.py`, make `_check_move_realizable` RETURN solved (last line `return solved`; change annotation to `-> dict`). Then in `_check_move_preserves_dependencies`, replace the realizability call + edge query:

```python
    solved = _check_move_realizable(ir, block_nid, target_loop_nid)
    target_nid_by_var = {ir.tree.data(nid).loop_var: nid
                         for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
                         if isinstance(ir.tree.data(nid), ForNode)}
    skip_cover_loops = frozenset(
        target_nid_by_var[lv]
        for dom in solved.values()
        for lv, _ext in dom.target_loops
        if lv in target_nid_by_var
    )
    moved_leaf = ir.dependency._resolve(block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(
        moved_leaf, target_loop_nid, index, skip_cover_loops=skip_cover_loops
    )
```

Confirm `ForNode` is imported in `_code_motion.py` (the existing import is `from nkigym.ir.tree import BlockNode, KernelTree, role_of`; add `ForNode` to it). Remove the now-duplicated `_check_move_realizable(...)` line that previously preceded the query (it is now the first line of the replacement block, capturing `solved`).

- [ ] **Step 7: Verify no regression in the transform suite**

Run (controller): `transport/remote_pytest.sh test/transforms/ test/ir/test_dependency.py`
Expected: PASS (all existing tests green — the default `skip_cover_loops=frozenset()` preserves behavior everywhere except a deliberately re-covered loop).

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py nkigym/src/nkigym/transforms/_code_motion.py test/ir/test_dependency.py
git commit -m "fix(deps): skip a COVER edge the move re-covers (Barrier 2)"
```

## Task 2: Barrier 1 — init-domination discriminator (NaN-capable; golden gate)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` (`_check_no_reduction_axis_covered`)
- Test: `test/transforms/test_code_motion.py`

**Interfaces:**
- Consumes: `ir.dependency.graph` with `CARRY` edges `writer_leaf -> loop_nid` (added by `_add_carry_edges` when a buffer's init dominates a non-PARALLEL loop); `BlockNode.iter_vars` (axis+role); `ISANode.op_cls.RMW_OPERANDS`; `_loopvar_to_dim`-equivalent already inside the guard via `solved` keys (dims). `target_loop_nid` + `ir.tree.ancestors` to resolve covering-loop NIDs.
- Produces: refined `_check_no_reduction_axis_covered(ir, block_nid, target_loop_nid, solved)` — allows covering an ACCUMULATION dim ONLY when every covering loop NID is one the moved block's own accumulator-init dominates (a `CARRY` edge from the block's init-writer into that loop NID exists). Rejects otherwise, unchanged message.

- [ ] **Step 1: Write the failing test — the fold move is ALLOWED**

Add to `test/transforms/test_code_motion.py`. This drives the probed-clean N-outermost prefix (canonical → Split(K) → Split(M) → Reorder×6 → RFactor(ko)), splits the fold to the matmul tile prefix, then asserts ReverseComputeAt of the fold under the matmul `i_d1_1` no longer raises:

```python
def test_reverse_compute_at_allows_fold_covering_its_own_ko():
    """The two-stage fold accumulates across its ENCLOSING ko (its sbuf_prod
    memset dominates ko via a CARRY edge), so covering ko by that loop is SAFE
    and must be allowed — the kernel_target fold-inlining precondition."""
    import pytest

    from test.transforms._fixtures import INPUT_SPECS, f_matmul

    from nkigym.environment import KernelMDP
    from nkigym.transforms import (
        Reorder, ReorderOption, ReverseComputeAt, ReverseComputeAtOption,
        RFactor, RFactorOption, Split, SplitOption, TransformLegalityError,
    )

    def mm_loop(state, loop_var):
        from nkigym.ir.tree import ForNode, ISANode
        leaf = next(n for n in state.tree.preorder()
                    if isinstance(state.tree.data(n), ISANode)
                    and state.tree.data(n).op_cls.__name__ == "NKIMatmul")
        return next(a for a in state.tree.ancestors(leaf)
                    if isinstance(state.tree.data(a), ForNode) and state.tree.data(a).loop_var == loop_var)

    def fold_blk(state):
        from nkigym.ir.tree import ISANode
        for nid in state.tree.blocks():
            leaves = [d for d in state.tree.descendants(nid) if isinstance(state.tree.data(d), ISANode)]
            if len(leaves) == 1 and state.tree.data(leaves[0]).op_cls.__name__ == "NKITensorTensor":
                return nid
        raise AssertionError("no fold block")

    def fold_leaf(state):
        from nkigym.ir.tree import ISANode
        return next(d for d in state.tree.descendants(fold_blk(state)) if isinstance(state.tree.data(d), ISANode))

    def fold_loop(state, loop_var):
        from nkigym.ir.tree import ForNode
        return next(d for d in state.tree.descendants(fold_blk(state))
                    if isinstance(state.tree.data(d), ForNode) and state.tree.data(d).loop_var == loop_var)

    env = KernelMDP(f_matmul, INPUT_SPECS,
                    transforms=[Split(), Reorder(), ReverseComputeAt(), RFactor()])
    s = env.reset()
    s = Split().apply(s, SplitOption(target_nid=mm_loop(s, "i_d0_0"), factors=(2, 8), target_axis=None))
    s = Split().apply(s, SplitOption(target_nid=mm_loop(s, "i_d1_0"), factors=(4, 4), target_axis=None))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d1_1"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d1_0"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_0"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d1_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d1_1")))
    s = RFactor().apply(s, RFactorOption(target_loop_nid=mm_loop(s, "i_d0_0"), factor_axis=0))
    s = Split().apply(s, SplitOption(target_nid=fold_leaf(s), factors=(4, 512), target_axis="d2"))
    s = Split().apply(s, SplitOption(target_nid=fold_loop(s, "i_d1_0"), factors=(4, 4), target_axis=None))
    moved = ReverseComputeAt().apply(
        s, ReverseComputeAtOption(block_nid=fold_blk(s), target_loop_nid=mm_loop(s, "i_d1_1"), index=-1)
    )
    assert fold_blk(moved) in moved.tree.descendants(mm_loop(moved, "i_d1_1"))
```

- [ ] **Step 2: Run it to verify it fails (the guard currently rejects)**

Run (controller): `transport/remote_pytest.sh test/transforms/test_code_motion.py::test_reverse_compute_at_allows_fold_covering_its_own_ko`
Expected: FAIL — `TransformLegalityError: ... would cover reduction axis 'd0' (ACCUMULATION) ...`.

- [ ] **Step 3: Add a helper that finds the block's accumulator-init carry loops**

In `nkigym/src/nkigym/transforms/_code_motion.py`, add above `_check_no_reduction_axis_covered`:

```python
def _own_carry_loop_nids(ir: KernelIR, block_nid: int) -> set[int]:
    """Loop NIDs that the moved block's accumulator-init DOMINATES.

    The block's RMW operand (its carried accumulator) is initialised by a
    sibling memset whose write CARRYs into the reduction loop — recorded as a
    ``CARRY`` edge ``init_writer -> loop_nid`` in ``ir.dependency``. Covering a
    reduction axis by one of these loops is the SAFE enclosing-reduction case
    (init dominates); covering by any other loop is foreign (init does not
    dominate -> NaN). Returns the loop NIDs into which the block's own
    accumulator's writers carry.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    leaf = next(d for d in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(d), ISANode))
    isa = ir.tree.data(leaf)
    acc_tensors = {
        isa.operand_bindings[slot].tensor for slot in isa.op_cls.RMW_OPERANDS if slot in isa.operand_bindings
    }
    out: set[int] = set()
    for writer in ir.dependency.touches_by_tensor.get(next(iter(acc_tensors)), ()) if acc_tensors else ():
        for _w, loop_nid, attrs in ir.dependency.graph.out_edges(writer, data=True):
            if attrs.get("kind") == "CARRY":
                out.add(loop_nid)
    return out
```

Add `from nkigym.ir.tree import ISANode` to the tree import (extend the existing line to `BlockNode, ForNode, ISANode, KernelTree, role_of`).

- [ ] **Step 4: Refine the guard to consult the own-carry loops by NID**

Replace the body loop of `_check_no_reduction_axis_covered` (keep the docstring's intent; update it to describe the discriminator). The covering loop NIDs for a dim are the target's enclosing ForNodes on that dim:

```python
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    own_carry = _own_carry_loop_nids(ir, block_nid)
    target_nid_by_var = {ir.tree.data(nid).loop_var: nid
                         for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
                         if isinstance(ir.tree.data(nid), ForNode)}
    result: None = None
    for dim, domain in solved.items():
        if not domain.target_loops:
            continue
        try:
            role = role_of(block, dim)
        except KeyError:
            continue
        if role != AxisRole.ACCUMULATION:
            continue
        covering_nids = {target_nid_by_var[lv] for lv, _e in domain.target_loops if lv in target_nid_by_var}
        if covering_nids and covering_nids <= own_carry:
            continue
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) would cover reduction axis "
            f"{dim!r} (ACCUMULATION) with enclosing loops {domain.target_loops} the block's own "
            f"init does not dominate; a foreign covering loop breaks init-domination"
        )
    return result
```

- [ ] **Step 5: Run the allow-test to verify it now passes**

Run (controller): `transport/remote_pytest.sh test/transforms/test_code_motion.py::test_reverse_compute_at_allows_fold_covering_its_own_ko`
Expected: PASS.

- [ ] **Step 6: Run the GOLDEN-VERDICT gate — the foreign-loop + replication rejects MUST still fire**

Run (controller): `transport/remote_pytest.sh "test/transforms/test_code_motion.py::test_compute_at_rejects_covering_matmul_reduction_axis" "test/transforms/test_code_motion.py::test_compute_at_rejects_replicating_reduction_over_untiled_output_dim"`
Expected: PASS (both still raise `TransformLegalityError`). If EITHER now passes-through (move admitted), STOP — the discriminator is too loose (a foreign loop entered `own_carry`); do not proceed.

- [ ] **Step 7: Full suite regression**

Run (controller): `transport/remote_pytest.sh test/transforms/ test/ir/`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py test/transforms/test_code_motion.py
git commit -m "fix(legality): allow covering a reduction axis the block's own init dominates (Barrier 1)"
```

## Task 3: Rewrite `_build_ladder` to the N-outermost fold-inlining path

**Files:**
- Modify: `examples/kernel_transforms.py` (`_build_ladder`, module docstring; reuse existing semantic locators `_loop`, `_op_blk`, `_op_leaf`, `_psum_memset_blk`, `_psum_memset_leaf`, `_blk_loop`)

**Interfaces:**
- Consumes: the Task 1 + Task 2 guard refinements (both committed). Existing locators in the file are semantic (track nids across structural change), so they keep working.
- Produces: an updated ladder whose final rung renders a kernel_target-equivalent (N-outermost, fold inlined under the matmul `i_d1_1`, store sunk under `i_d2_0`). `kernel_target` itself is unchanged (the reference).

> NOTE: rung locators/exact order are pinned empirically (the gym-1 sim gate below is the authority). The sequence below is the probed-clean spine (k0–k9) plus the now-unblocked co-location rungs; if a co-location rung's index/target needs adjusting, fix it against the sim gate, do not invent a new transform.

- [ ] **Step 1: Replace the `steps` list in `_build_ladder`**

The probed-clean spine is k1–k9 (Split K, Split M, Reorder×6, RFactor). Append the co-location rungs. Use the existing `_loop`/`_op_blk`/`_op_leaf`/`_blk_loop` locators:

```python
    steps = [
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_1"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_1"))),
        lambda ir: RFactor().apply(ir, RFactorOption(target_loop_nid=_loop(ir, "i_d0_0"), factor_axis=0)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")),
        lambda ir: ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=_psum_memset_blk(ir), target_loop_nid=_loop(ir, "i_d2_0"), index=0)),
        lambda ir: ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=_op_blk(ir, "NKITensorCopy"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorTensor"), factors=(4, 512), target_axis="d2")),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_blk_loop(ir, _op_blk(ir, "NKITensorTensor"), "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=_op_blk(ir, "NKITensorTensor"), target_loop_nid=_loop(ir, "i_d1_1"), index=-1)),
        lambda ir: ComputeAt().apply(ir, ComputeAtOption(block_nid=_op_blk(ir, "NKIStore"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)),
    ]
```

- [ ] **Step 2: Update the module docstring**

Rewrite the docstring's ladder description to the N-outermost path (nest `N > ko > Mo > Mi > ki`, fold inlined under `i_d1_1`, store sunk under `i_d2_0`). Remove the "~45pp gap / fold-inlining blocked" paragraph (now resolved); state the new expected HW result is filled in after the gym-1 run (Step 4). Keep it factual — do not assert an MFU number before measuring.

- [ ] **Step 3: Per-rung CPU-sim on gym-1 (the correctness gate)**

Run (controller): `transport/ssh_host.sh --host gym-1 --cmd "python examples/kernel_transforms.py" --cache /home/weittang/workplace/cache/transforms`
Expected: every rendered rung prints `[sim] ... pass=True` (~1.4e-4), including the new co-location rungs and `kernel_target`. If a co-location rung sims FALSE or its `apply` raises, STOP: re-pin that rung's locator/target against the dumped tree (the rung order is empirical) — do NOT loosen a guard further to force it.

- [ ] **Step 4: Read the HW MFU from the same run + record it**

The same `profile(...)` call compiles + profiles every rung on Trn2. From the printed output / `<cache>/results.json`, record the final fold-inlined rung's MFU and `kernel_target`'s. Update the module docstring with the measured number (e.g. "final rung XX.X% MFU vs kernel_target 90.8%").
Expected: the fold-inlined rung runs on HW (not BIR exit 70) and exceeds k15's 46.08%. If it does not exceed 46%, that is a real finding — report it; do not silently keep the old ladder.

- [ ] **Step 5: Commit**

```bash
git add examples/kernel_transforms.py
git commit -m "feat(example): N-outermost fold-inlining ladder reaching kernel_target"
```

## Self-Review (completed by plan author)

- **Spec coverage:** B2 → Task 1; B1 → Task 2; fold-inlining ladder + validation (per-rung sim, golden-verdict gate, HW MFU) → Tasks 2.6 + 3.3 + 3.4. Out-of-scope items (buffer-list, codegen placement, lhs_T tuning) correctly omitted.
- **Type consistency:** `_check_move_realizable` returns `dict` (solved) — consumed in Task 1.6 and unchanged for Task 2 (which reads `solved` as before). `skip_cover_loops: frozenset[int]` consistent across `_first_backward`/`first_backward_edge`/`first_backward_edge_for_insertion`. `_own_carry_loop_nids -> set[int]` compared against `covering_nids: set[int]` (NID-to-NID, per the global constraint).
- **Placeholder scan:** none — every code step shows the code; the only deferred item (exact co-location rung order) is explicitly gated on the gym-1 sim, which is the correct authority for an empirical sequence, not a placeholder.
