# Unified CodeMotion + span-promotion legality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CARRY/COVER/barrier code-motion legality machinery with a single span-promotion rule, and merge `ComputeAt` + `ReverseComputeAt` into one `CodeMotion` transform.

**Architecture:** Code-motion legality already tests one law — a producer's tree-span must end before its consumer's begins — via a preorder span. It is fooled only by a loop-carried region (an accumulator RMW'd loop-invariantly), whose live range is the whole loop, not a point. Fix: promote such an access's span to the carrying loop's full span, on demand, inside `span()`. This makes CARRY/COVER edges and `skip_cover_loops` unnecessary. Separately, the two code-motion faces are already one operation (`is_reverse` is inert, `analyze()` bodies are identical), so they collapse into one `CodeMotion`; the only behavioral change is dropping the `ComputeAt`-only output-block guard, which is required so the k11→k12 store-sink stays legal.

**Tech Stack:** Python 3.12, `networkx` (dependency DiGraph), `pytest`. No new dependencies. The `arith` substrate (`to_affine`) and `interval.regions_disjoint` already ship.

## Global Constraints

- **Design doc (source of truth):** `docs/superpowers/specs/2026-07-03-code-motion-span-promotion-design.md`. Every task serves it.
- **No unrolling / no oracle module.** The check is purely symbolic.
- **Transform legality = behavior/dep-order + ISA well-formedness ONLY; never resource capacity.** Do not add PSUM/SBUF fit gates.
- **Directions frozen from the pre-move program.** The legality check reads edge directions from `ir.dependency` (built on the current, pre-move tree) and evaluates spans at the proposed position. NEVER rebuild `Dependency` on a moved tree to get directions.
- **Loud failures only.** No silent raises, no `try/except` to adapt around malformed IR. Reject bad input with a `TransformLegalityError` (legality) or `ValueError`/`AssertionError` (malformed IR).
- **Single return per function.** User rejects branching returns. Use a `result` variable set in branches, returned once.
- **Comments are triple-quoted block comments (`"""..."""`), never `#`.** Tooling directives (`# type: ignore`) exempt. No inline comments.
- **Modern type hints** (`list`/`dict`/`X | None`), Google/NumPy docstrings on every function.
- **Tests run on gym-1, not locally.** The dev box has no Neuron env. Run via:
  `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest <files>" --cache /home/weittang/workplace/cache/<name>`
  `ssh_host.sh` sets `PYTHONPATH=.:nkigym/src:autotune/src`, requires `--cache` even for pytest, and `--cmd` needs a `.py` token (enumerate test files; a bare `test/` dir is rejected). **To select a single test, use `-k <substring>` (the `path::node-id` form has no `.py` token and is rejected); e.g. `python -m pytest test/ir/test_dependency.py -k hazard_edges_record -v`.**
- **Pre-existing failing set is 7** (transform-legality/code-motion, e.g. `target_loop_nid not in tree`), identical at the parent commit. Verify any "new" failure against the parent before treating it as a regression.
- **Byte-exact ladder gate:** `k0…k27` in `examples/manual_transforms.py` must CPU-sim clean, and `examples/kernel_transforms.py` must rebuild the transform-driven ladder. A wrong intermediate rung means the check mis-verdicted — fix the check, not the ladder.
- **Commit frequently**, ending messages with the Co-Authored-By trailer for Claude Opus 4.8 (1M context).

## File Structure

Files created or modified across the plan, and each one's responsibility:

- `nkigym/src/nkigym/ir/dependency.py` **(modify)** — add `tensor` to each hazard edge; add the on-demand span-promotion helpers (`_promoted_span`, `_access_invariant_across`, `_tensor_carried_across`) and wire them into the two `span()` closures; later, delete the CARRY/COVER builders + `skip_cover_loops`.
- `nkigym/src/nkigym/transforms/_code_motion.py` → **rename to** `code_motion.py` — hosts the shared move/legality helpers AND the new public `CodeMotion` transform + `CodeMotionOption`. Delete `_own_carry_loop_nids`, `_check_no_reduction_axis_covered`, `is_reverse`, `skip_cover_loops`.
- `nkigym/src/nkigym/transforms/compute_at.py` **(delete)** — folded into `CodeMotion`.
- `nkigym/src/nkigym/transforms/reverse_compute_at.py` **(delete)** — folded into `CodeMotion`.
- `nkigym/src/nkigym/transforms/__init__.py` **(modify)** — swap the two face exports for `CodeMotion`/`CodeMotionOption`.
- `nkigym/src/nkigym/transforms/compute_at_legality.md` → **rename to** `code_motion_legality.md` — de-stale and reframe for the merged transform.
- `test/ir/test_dependency.py` **(modify)** — rewrite the internal-reaching tests (`_carry_loops_of_leaf` imports) to assert the new check's verdicts; add span-promotion unit tests.
- `test/transforms/test_code_motion.py` **(modify)** — absorb the face-specific tests; retarget option construction to `CodeMotion`; add the store-sink-allowed verdict.
- `test/transforms/test_compute_at.py` **(delete)** — merged into `test_code_motion.py`.
- `test/transforms/test_reverse_compute_at.py` **(delete)** — merged into `test_code_motion.py`.
- `test/transforms/_fixtures.py`, `test/transforms/_pipeline_fixtures.py`, `test/transforms/test_reorder.py`, `test/transforms/test_split.py` **(modify)** — retarget `ComputeAt`/`ReverseComputeAt` construction to `CodeMotion`.
- `examples/kernel_transforms.py`, `examples/matmul_lhsT_rhs.py` **(modify)** — retarget the option construction to `CodeMotion`.

---

## Task 1: Record the conflicting tensor on every hazard edge

Span-promotion is **per-tensor**: a leaf can have one carried edge (`memset→fold` on `sbuf_prod`, carried across `ko`) and one non-carried edge (`copy→fold` on `sbuf_rfactor`) at once. Promoting at leaf granularity would wrongly promote the fold across `ko` for the `sbuf_rfactor` edge too, rejecting the legal per-`ko` copy-before-fold. So each edge must carry the tensor it is about. `_try_edge` already receives `tensor` but discards it; store it. (Probed on gym-1: no leaf pair in the canonical IR shares more than one conflicting tensor, so a single `tensor` attribute per edge is sufficient — no need for a set.)

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py:398-408` (`_try_edge`)
- Test: `test/ir/test_dependency.py`

**Interfaces:**
- Produces: hazard edges now have attrs `{"kind": str, "tensor": str}`. Consumers reading `attrs["kind"]` are unaffected; `_first_backward` (Task 3) will read `attrs["tensor"]`.

- [ ] **Step 1: Write the failing test**

Add to `test/ir/test_dependency.py`:

```python
def test_hazard_edges_record_conflicting_tensor():
    """Each RAW/WAW/WAR edge carries the tensor it is about, so span-promotion
    can key on the shared buffer per edge (a leaf may have one carried and one
    non-carried edge at once)."""
    from test.transforms._fixtures import build_canonical_ir

    ir = build_canonical_ir()
    dep = ir.dependency
    edges_without_tensor = [
        (a, b) for a, b, attrs in dep.graph.edges(data=True) if "tensor" not in attrs
    ]
    assert not edges_without_tensor, f"edges missing tensor attr: {edges_without_tensor}"
    for _a, _b, attrs in dep.graph.edges(data=True):
        assert isinstance(attrs["tensor"], str) and attrs["tensor"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_hazard_edges_record_conflicting_tensor -v" --cache /home/weittang/workplace/cache/span1`
Expected: FAIL — edges have only `kind`, not `tensor`.

- [ ] **Step 3: Store the tensor on the edge**

In `_try_edge`, change the two `add_edge` sites so the tensor is recorded. Replace the body's tail:

```python
    def _try_edge(self, producer: int | None, consumer: int, kind: str, tensor: str) -> None:
        """Insert a hazard edge, skipping self-loops and missing producers.

        The edge records both the hazard ``kind`` and the ``tensor`` the two
        leaves conflict on. Span-promotion reads ``tensor`` to decide, per edge,
        whether the shared buffer is carried across an enclosing loop.
        """
        result: None = None
        if producer is None or producer == consumer:
            result = None
        elif self._provably_disjoint(producer, consumer, tensor, kind):
            result = None
        else:
            keep = True
            if self.graph.has_edge(producer, consumer):
                current = self.graph.edges[producer, consumer]["kind"]
                if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                    keep = False
            if keep:
                self.graph.add_edge(producer, consumer, kind=kind, tensor=tensor)
        return result
```

(The single-return refactor keeps the user's one-return rule; behavior is identical to the current early-returns plus the new `tensor=tensor`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_hazard_edges_record_conflicting_tensor -v" --cache /home/weittang/workplace/cache/span1`
Expected: PASS.

- [ ] **Step 5: Guard the mermaid dump still renders**

The `_to_mermaid` helper reads `attrs["kind"]` (line 581) — unaffected by the added `tensor`. Confirm the existing dependency-dump test (if any) still passes:

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py -v" --cache /home/weittang/workplace/cache/span1`
Expected: PASS (no new failures vs the pre-existing set).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Dependency: record conflicting tensor on each hazard edge

Span-promotion is per-tensor; a leaf can have a carried and a non-carried
edge simultaneously (fold: memset->fold on sbuf_prod carried, copy->fold on
sbuf_rfactor not). Store the tensor _try_edge already receives.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 2: Span-promotion predicates (pure module functions)

Two on-demand predicates, both read from the evaluated tree — nothing stored. `_access_invariant_across(tree, access_nid, loop_nid, tensor)` — is the leaf's access to `tensor` invariant across the loop (loop-var absent from every region `lo` of that tensor's operands)? `_tensor_carried_across(tree, loop_nid, tensor)` — does some ISA leaf enclosed by the loop RMW `tensor` invariantly across it? Both are ROLE-BLIND: they consult only regions + `RMW_OPERANDS`, never `AxisRole`. This is deliberate and verified (post-RFactor the matmul's K axis is PARALLEL for both `ko` and `ki`, yet `psum` genuinely carries across them; invariance already excludes the indexed loops `i_d1_1`/`i_d2_0` because those vars DO appear in the psum offset).

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py` (add module-level helpers near `_carry_loops_of_leaf`)
- Test: `test/ir/test_dependency.py`

**Interfaces:**
- Consumes: `KernelTree` (`.data`, `.ancestors`), `ISANode.operand_bindings`, `op_cls.RMW_OPERANDS`, `arith.to_affine`.
- Produces:
  - `_leaf_operand_regions(tree, leaf_nid, tensor, rmw_only=False) -> list[BufferRegion]` — the leaf's operand regions naming `tensor` (all slots, or only `RMW_OPERANDS` slots).
  - `_access_invariant_across(tree, leaf_nid, loop_var, tensor) -> bool` — True iff `loop_var` appears in NO `lo` of any of `leaf_nid`'s regions on `tensor`. False if the leaf touches no such region (cannot claim invariance for an absent access).
  - `_tensor_carried_across(tree, loop_nid, tensor) -> bool` — True iff some ISA leaf in `tree.descendants(loop_nid)` has an `rmw` region on `tensor` invariant across `loop_nid`'s var.

- [ ] **Step 1: Write the failing tests**

Add to `test/ir/test_dependency.py`:

```python
def test_tensor_carried_across_psum_over_kloop():
    """psum (matmul rmw, offset invariant across K) is carried across the K loop;
    a pure-read operand (sbuf_lhs_T) is not."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "psum_prod") is True
    assert _tensor_carried_across(ir.tree, kloop, "sbuf_lhs_T") is False


def test_access_invariant_across_matches_offset_var():
    """The matmul's psum access is invariant across K (K-var absent from the psum
    offset) but NOT across a loop whose var indexes the psum offset."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.dependency import _access_invariant_across
    from nkigym.ir.tree import ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    assert _access_invariant_across(ir.tree, matmul_leaf, "i_d0_0", "psum_prod") is True
    assert _access_invariant_across(ir.tree, matmul_leaf, "i_d1_0", "psum_prod") is False
```

(The canonical matmul writes `psum_prod[.., i_d1_0, i_d2_0*512:+512]` — K-var `i_d0_0` absent from the offset → invariant; `i_d1_0` present → not invariant. Verify the exact loop_vars against `build_canonical_ir` when implementing; adjust the literals if the canonical names differ.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_tensor_carried_across_psum_over_kloop test/ir/test_dependency.py::test_access_invariant_across_matches_offset_var -v" --cache /home/weittang/workplace/cache/span2`
Expected: FAIL — `_tensor_carried_across` / `_access_invariant_across` not defined.

- [ ] **Step 3: Implement the predicates**

Add near `_carry_loops_of_leaf` in `dependency.py`:

```python
def _leaf_operand_regions(
    tree: KernelTree, leaf_nid: int, tensor: str, rmw_only: bool
) -> list[BufferRegion]:
    """Regions of ``tensor`` bound by ``leaf_nid``'s operands.

    With ``rmw_only`` True, only ``RMW_OPERANDS`` slots are considered — the
    read-modify-written accumulators (matmul ``dst``, tensor_tensor ``data1``)
    that can carry a value across a loop.
    """
    data = tree.data(leaf_nid)
    regions: list[BufferRegion] = []
    if isinstance(data, ISANode):
        slots = data.op_cls.RMW_OPERANDS if rmw_only else data.operand_bindings.keys()
        for slot in slots:
            region = data.operand_bindings.get(slot)
            if region is not None and region.tensor == tensor:
                regions.append(region)
    return regions


def _access_invariant_across(tree: KernelTree, leaf_nid: int, loop_var: str, tensor: str) -> bool:
    """True iff ``leaf_nid``'s access to ``tensor`` does NOT depend on ``loop_var``.

    Invariant means ``loop_var`` appears in no axis offset (``lo``) of any operand
    region naming ``tensor`` — so every iteration of the loop touches the same
    slice (a live-across / replicated access). A leaf that touches no such region
    is NOT invariant (there is no access to be invariant about).
    """
    regions = _leaf_operand_regions(tree, leaf_nid, tensor, rmw_only=False)
    invariant = bool(regions)
    for region in regions:
        if any(loop_var in to_affine(lo) for lo, _w in region.ranges):
            invariant = False
            break
    return invariant


def _tensor_carried_across(tree: KernelTree, loop_nid: int, tensor: str) -> bool:
    """True iff some ISA leaf inside ``loop_nid`` RMWs ``tensor`` invariantly across it.

    A carried tensor is live across the whole loop: an rmw access whose offset
    omits the loop var accumulates into the same slice every iteration. This is
    role-blind — it reads only regions + ``RMW_OPERANDS``, never the axis role
    (post-RFactor the matmul's K axis is PARALLEL yet psum still carries).
    """
    loop = tree.data(loop_nid)
    assert isinstance(loop, ForNode), f"_tensor_carried_across: {loop_nid} is not a ForNode"
    loop_var = loop.loop_var
    carried = False
    for nid in tree.descendants(loop_nid):
        if not isinstance(tree.data(nid), ISANode):
            continue
        rmw_regions = _leaf_operand_regions(tree, nid, tensor, rmw_only=True)
        if rmw_regions and not any(
            loop_var in to_affine(lo) for region in rmw_regions for lo, _w in region.ranges
        ):
            carried = True
            break
    return carried
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_tensor_carried_across_psum_over_kloop test/ir/test_dependency.py::test_access_invariant_across_matches_offset_var -v" --cache /home/weittang/workplace/cache/span2`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Dependency: add role-blind carried/invariance predicates for span-promotion

_tensor_carried_across + _access_invariant_across read regions + RMW_OPERANDS
only (no axis role) — post-RFactor matmul K is PARALLEL yet psum carries.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 3: Promote spans in `_first_backward`, per edge tensor

`_first_backward` walks each edge incident to the moved leaf. For an edge `(a, b, tensor)`, before the `span(a).end < span(b).start` test, widen each endpoint's span to any enclosing loop `L` across which that endpoint's access to `tensor` is invariant AND `tensor` is carried across `L`. The promotion is **per edge tensor** (Task 1's attr) and evaluated on the tree the `span` closure sees. Crucially, the moved leaf's enclosing loops are the TARGET's (from Task 4's `for_insertion` closure the moved leaf is scored at the proposed position) — so a memset sunk under the matmul's K loop promotes to K-span there.

Promotion needs, per endpoint, the enclosing loops on the evaluated tree. `first_backward_edge` reads them from `eval_tree.ancestors(nid)`. `first_backward_edge_for_insertion` (Task 4) passes the target's ancestor loops for the moved leaf. So `_first_backward` takes an `enclosing_loops: Callable[[int], list[int]]` that returns the ForNode ancestor nids for an endpoint at its evaluated position, plus the eval tree for region lookups.

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py` (`_first_backward` 206-229, and the `first_backward_edge` caller 91-124)
- Test: `test/ir/test_dependency.py`

**Interfaces:**
- Consumes: Task 1 edge `tensor` attr; Task 2 predicates.
- Produces: `_first_backward(self, moved_leaf_nid, span, eval_tree, enclosing_loops)` — the `skip_cover_loops` parameter is REMOVED. Promotion helper `_promoted_span(base_span, endpoint_nid, tensor, eval_tree, enclosing_loops, order) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing test — memset sunk into K rejects, outside K allows**

Add to `test/ir/test_dependency.py` (this pins the historical NaN bug at the new mechanism):

```python
def test_span_promotion_rejects_memset_sunk_into_kloop():
    """Sinking the psum memset INTO the K loop must read as a backward edge:
    psum is carried across K, so both memset and matmul promote to K-span and
    memset.end < matmul.start is false."""
    import copy
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    memset_blk = next(
        nid for nid in ir.tree.blocks()
        if any(isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMemset
               for d in ir.tree.descendants(nid))
    )
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=memset_blk, target_loop_nid=kloop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    memset_leaf = next(
        n for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKIMemset
    )
    assert dep.first_backward_edge(memset_leaf) is not None
```

(This mirrors the existing `test_first_backward_edge_flags_memset_sunk_under_kloop`; it must still pass with CARRY edges GONE and promotion doing the work. Keep both until Task 6 deletes CARRY, then the old one is retargeted.)

- [ ] **Step 2: Run test to verify it fails**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_span_promotion_rejects_memset_sunk_into_kloop -v" --cache /home/weittang/workplace/cache/span3`
Expected: at this point it may PASS via the still-present CARRY edge. That is fine — the test pins the VERDICT; Step 3 makes promotion the mechanism, and Task 6 removes CARRY so this test proves promotion alone suffices. If it fails now, the promotion wiring in Step 3 is what fixes it.

- [ ] **Step 3: Add `_promoted_span` and rewire `_first_backward`**

Add the helper and rewrite `_first_backward` + the `first_backward_edge` closure. New `_promoted_span`:

```python
def _promoted_span(
    base: tuple[float, float],
    endpoint_nid: int,
    tensor: str,
    eval_tree: KernelTree,
    enclosing_loops: list[int],
    span_of_loop: Callable[[int], tuple[float, float]],
) -> tuple[float, float]:
    """Widen ``base`` to any enclosing loop across which ``endpoint_nid``'s access
    to ``tensor`` is invariant and ``tensor`` is carried.

    A carried, loop-invariant access is live across the whole loop, so its
    effective span is the loop's span, not the leaf point. Promoting BOTH
    endpoints of a carried edge is what turns "memset lexically first inside K"
    into a backward edge (memset.end == K.end is not < matmul.start inside K).
    """
    lo, hi = base
    for loop_nid in enclosing_loops:
        loop = eval_tree.data(loop_nid)
        if not isinstance(loop, ForNode):
            continue
        if not _access_invariant_across(eval_tree, endpoint_nid, loop.loop_var, tensor):
            continue
        if not _tensor_carried_across(eval_tree, loop_nid, tensor):
            continue
        l_lo, l_hi = span_of_loop(loop_nid)
        lo = min(lo, l_lo)
        hi = max(hi, l_hi)
    return (lo, hi)
```

Rewrite `_first_backward` to read the edge tensor and promote each endpoint:

```python
    def _first_backward(
        self,
        moved_leaf_nid: int,
        span: Callable[[int], tuple[float, float]],
        eval_tree: KernelTree,
        enclosing_loops: Callable[[int], list[int]],
    ) -> tuple[int, int] | None:
        """Return the first edge incident to ``moved_leaf_nid`` that ``span`` ranks
        backward after per-tensor span-promotion, else ``None``.

        Each edge carries the ``tensor`` its two leaves conflict on. Before the
        ``span(a).end < span(b).start`` test, each endpoint's span is promoted to
        any enclosing loop across which its access to that tensor is invariant and
        the tensor is carried (a live-across accumulator). ``enclosing_loops(nid)``
        returns the ForNode ancestor nids of an endpoint at its evaluated position
        — for the moved leaf under an insertion query, the TARGET's ancestors.
        """
        result: tuple[int, int] | None = None
        for a, b, attrs in self.graph.edges(data=True):
            if a != moved_leaf_nid and b != moved_leaf_nid:
                continue
            tensor = attrs.get("tensor")
            if tensor is None:
                continue
            span_a = _promoted_span(span(a), a, tensor, eval_tree, enclosing_loops(a), span)
            span_b = _promoted_span(span(b), b, tensor, eval_tree, enclosing_loops(b), span)
            if not (span_a[1] < span_b[0]):
                result = (a, b)
                break
        return result
```

**Why `attrs.get("tensor")` + skip-if-None:** CARRY/COVER edges (still built in `_build` until Task 6) have no `tensor` attr and route through loop-nid endpoints, not real accesses — promotion does not apply to them, so skip them. RAW/WAW/WAR edges all carry `tensor` (Task 1). After Task 6 there are no tensor-less edges, but the guard stays harmless. This is why the memset-into-K reject in this task's test comes from PROMOTION on the RAW `memset→matmul` edge, not from the soon-to-be-deleted CARRY edge.

Update the `first_backward_edge` caller (91-124) to pass `eval_tree` and an `enclosing_loops` reading `eval_tree.ancestors`:

```python
        eval_tree = tree if tree is not None else self._tree
        order = {n: i for i, n in enumerate(eval_tree.preorder())}

        def span(nid: int) -> tuple[float, float]:
            idxs = [order[d] for d in (eval_tree.descendants(nid) | {nid}) if d in order]
            if not idxs:
                raise KeyError(f"dependency endpoint {nid} absent from the evaluated tree")
            return (min(idxs), max(idxs))

        def enclosing_loops(nid: int) -> list[int]:
            return [a for a in eval_tree.ancestors(nid) if isinstance(eval_tree.data(a), ForNode)]

        return self._first_backward(moved_leaf_nid, span, eval_tree, enclosing_loops)
```

Remove the `skip_cover_loops` parameter from `first_backward_edge`'s signature and docstring. (CARRY/COVER edge-builders still run in `_build` for now — Task 6 removes them; the added `tensor` attr and promotion coexist with them harmlessly because promotion only widens spans, and a still-present CARRY edge is a loop-nid endpoint the promotion loop skips via the `isinstance(..., ForNode)` guards on real accesses. Verify no CARRY loop-nid endpoint breaks the edge walk — see Step 4.)

- [ ] **Step 4: Run the memset test + full dependency suite**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py -v" --cache /home/weittang/workplace/cache/span3`
Expected: PASS for the new test and no NEW failures vs the pre-existing set. (The `attrs.get("tensor")` guard in Step 3 already handles the tensor-less CARRY/COVER edges that still exist at this point, so the edge walk does not KeyError.)

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Dependency: per-tensor span-promotion in _first_backward

Promote a carried, loop-invariant endpoint's span to the carrying loop, per
edge tensor. Removes skip_cover_loops from first_backward_edge. CARRY/COVER
still built in _build (removed in a later task); tensor-less edges are skipped.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 4: Wire promotion into `first_backward_edge_for_insertion` (the moved-leaf case)

`for_insertion` scores the moved leaf at the analytic `moved_pos` on `self._tree` (pre-move tree, moved leaf relocated). For promotion, the moved leaf's enclosing loops are the TARGET's — `target_loop_nid` plus the target's ForNode ancestors — because after the splice it becomes their descendant. Every other endpoint keeps its own `self._tree` ForNode ancestors (minus the moved subtree). This is what makes the memset-into-K case promote at the proposed position: the memset's post-splice enclosers include K, and psum is carried across K.

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py:126-171` (`first_backward_edge_for_insertion`)
- Test: `test/ir/test_dependency.py`

**Interfaces:**
- Consumes: Task 3's `_first_backward(self, moved_leaf_nid, span, eval_tree, enclosing_loops)`.
- Produces: `first_backward_edge_for_insertion(self, moved_leaf_nid, target_loop_nid, index)` — `skip_cover_loops` REMOVED.

- [ ] **Step 1: Write the failing test — pure insertion query agrees with an actual move**

Add to `test/ir/test_dependency.py`:

```python
def test_for_insertion_rejects_memset_into_kloop_without_moving():
    """The pure insertion query flags memset-sunk-into-K as backward, matching
    the actual-move result, via span-promotion (no CARRY edge needed)."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    dep = ir.dependency
    memset_blk = next(
        nid for nid in ir.tree.blocks()
        if any(isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMemset
               for d in ir.tree.descendants(nid))
    )
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved_leaf = dep._resolve(memset_blk)
    assert dep.first_backward_edge_for_insertion(moved_leaf, kloop, 0) is not None
```

- [ ] **Step 2: Run test to verify it fails or passes-via-CARRY**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py::test_for_insertion_rejects_memset_into_kloop_without_moving -v" --cache /home/weittang/workplace/cache/span4`
Expected: PASS if CARRY still present; the point is it must STILL pass after Step 3 rewires it to promotion and after Task 6 deletes CARRY.

- [ ] **Step 3: Rewrite the closure to supply target-based enclosers for the moved leaf**

Replace the tail of `first_backward_edge_for_insertion` (from the `def span` through the return). Drop `skip_cover_loops` from the signature:

```python
    def first_backward_edge_for_insertion(
        self, moved_leaf_nid: int, target_loop_nid: int, index: int
    ) -> tuple[int, int] | None:
        """[keep the existing docstring's first paragraphs; drop the skip_cover_loops
        mention. Add:] Span-promotion evaluates the moved leaf's enclosing loops as
        the TARGET's nest (``target_loop_nid`` + its ForNode ancestors), since the
        splice makes it their descendant; every other endpoint keeps its own
        ``self._tree`` ForNode ancestors minus the moved subtree.
        """
        order: dict[int, float] = {n: i for i, n in enumerate(self._tree.preorder())}
        owner_block = self._owner_block.get(moved_leaf_nid, moved_leaf_nid)
        moved_subtree = self._tree.descendants(owner_block) | {owner_block}
        moved_pos = self._effective_insertion_position(order, target_loop_nid, index, moved_subtree)
        enclosers = set(self._tree.ancestors(target_loop_nid)) | {target_loop_nid}
        target_loops = [
            n for n in (target_loop_nid, *self._tree.ancestors(target_loop_nid))
            if isinstance(self._tree.data(n), ForNode)
        ]

        def span(nid: int) -> tuple[float, float]:
            if nid == moved_leaf_nid:
                return (moved_pos, moved_pos)
            positions = [order[d] for d in (self._tree.descendants(nid) | {nid}) - moved_subtree if d in order]
            if nid in enclosers:
                positions.append(moved_pos)
            if not positions:
                raise KeyError(f"dependency endpoint {nid} absent from the tree")
            return (min(positions), max(positions))

        def enclosing_loops(nid: int) -> list[int]:
            if nid == moved_leaf_nid:
                return target_loops
            return [
                a for a in self._tree.ancestors(nid)
                if a not in moved_subtree and isinstance(self._tree.data(a), ForNode)
            ]

        return self._first_backward(moved_leaf_nid, span, self._tree, enclosing_loops)
```

**Note on the promoted-loop span:** `_promoted_span` calls `span(loop_nid)` for a promoted loop. For the moved leaf's target loops, `span` uses the `for_insertion` closure — a target loop is in `enclosers` so its span already includes `moved_pos` and its own subtree, giving the correct "moved leaf spans all of K" width. Confirm the target loop nid is reachable by `span` (it is a real node in `self._tree`).

- [ ] **Step 4: Run the insertion test + the code-motion suite**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py test/transforms/test_code_motion.py -v" --cache /home/weittang/workplace/cache/span4`
Expected: PASS for the new test; no NEW failures. (`_check_move_preserves_dependencies` still passes `skip_cover_loops` — Task 5 fixes that caller. If a signature mismatch errors here, it is expected until Task 5; run only the dependency file this step: `python -m pytest test/ir/test_dependency.py -v`.)

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Dependency: moved-leaf enclosers are the target nest in for_insertion

Span-promotion of the moved leaf uses target_loop_nid + its ForNode ancestors,
so memset-into-K promotes to K-span at the proposed position. Drops
skip_cover_loops from first_backward_edge_for_insertion.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 5: Switch the code-motion caller to span-promotion only

`_check_move_preserves_dependencies` (in `_code_motion.py`) builds `skip_cover_loops` and calls `first_backward_edge_for_insertion` with it, and `_check_same_loop_prefix` calls `_check_no_reduction_axis_covered`. Both go away: the caller calls `first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)` with no skip set, and the reduction-axis-covered call is dropped. The reduction-covering verdict is now delivered by span-promotion. The structural `_check_no_reduction_replicated` stays.

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` — `_check_move_preserves_dependencies` (266-315), `_check_same_loop_prefix` (159 removal)
- Test: `test/transforms/test_code_motion.py` (the four pinned verdicts)

**Interfaces:**
- Consumes: Task 4's `first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)`.
- Produces: `_check_move_preserves_dependencies(ir, block_nid, target_loop_nid, index)` — the unused `is_reverse` parameter is DROPPED; internally no `skip_cover_loops` construction.

- [ ] **Step 1: Verify the four pinned verdicts BEFORE editing (baseline)**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/test_code_motion.py -v" --cache /home/weittang/workplace/cache/span5`
Expected: the four pinned tests PASS (this is the baseline; they must still pass after the switch). Record which currently pass.

- [ ] **Step 2: Drop `skip_cover_loops` construction from the caller**

In `_check_move_preserves_dependencies`, remove the `covered_vars` / `target_nid_by_var` / `skip_cover_loops` block and simplify the call:

```python
    def _check_move_preserves_dependencies(
        ir: KernelIR, block_nid: int, target_loop_nid: int, index: int
    ) -> None:
        """[keep docstring; drop the skip_cover_loops paragraph, note span-promotion
        now delivers reduction-init domination + coverage.]"""
        _check_same_loop_prefix(ir, block_nid, target_loop_nid)
        moved_leaf = ir.dependency._resolve(block_nid)
        offending = ir.dependency.first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)
        result: None = None
        if offending is not None:
            a, b = offending
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
                f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
                f"reduction loop, nor a consumer precede its producer)"
            )
        return result
```

Drop the `is_reverse` parameter here (it was unused). Update both call sites in `compute_at.py:116` and `reverse_compute_at.py:105` to drop `is_reverse=...` — they will be replaced entirely in Task 7, but keep them compiling now.

- [ ] **Step 3: Drop the reduction-axis-covered call from `_check_same_loop_prefix`**

In `_check_same_loop_prefix` remove line 159 (`_check_no_reduction_axis_covered(...)`). Keep `_check_no_reduction_replicated` (line 160). The function still returns `target_seq`.

- [ ] **Step 4: Delete the stale-nid verdict tests; write fresh op-based ones**

CONTROLLER FINDING (verified on gym-1, worktree at base `e7ff491`): 4 of 6 tests in
`test/transforms/test_code_motion.py` ALREADY FAIL at base — `test_sunk_block_residual_loop_does_not_shadow_enclosing_name`, `test_compute_at_rejects_covering_matmul_reduction_axis`, `test_compute_at_rejects_replicating_reduction_over_untiled_output_dim`, `test_compute_at_memset_sink_across_block_wall_sims_correct` — because they hardcode literal nids in `KernelMDP` traces that have DRIFTED (`target_loop_nid=23 not in tree`). These are pre-existing failures, unrelated to span-promotion. Per the user's direction: **DELETE these 4 stale-nid tests and write FRESH tests for the new span-promotion path** using robust op-based lookups (no hardcoded nids), mirroring the passing tests in `test_ir/test_dependency.py`.

Delete the 4 stale tests from `test_code_motion.py`. KEEP the 2 passing tests (`test_move_lifts_tensor_copy_under_matmul_inner_loop`, `test_reverse_compute_at_allows_fold_covering_its_own_ko`). Add fresh verdict tests using `_block_for_op` (already in the file) + `ir.dependency.first_backward_edge_for_insertion` on the PRE-move `ir` (production path, no `_move`, no hardcoded nid). Controller-verified pattern (memset-into-K reject returns `(9, 14)` non-None):

```python
def test_span_promotion_rejects_memset_sunk_into_matmul_kloop():
    """Init-domination: sinking the psum memset INTO the matmul's K loop is
    rejected — psum_prod is carried across K, span-promotion widens both to
    K-span so the init can no longer precede the accumulation. Verdict via the
    production insertion query on the pre-move tree (no hardcoded nids)."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, "NKIMemset")
    matmul_blk = _block_for_op(ir, "NKIMatmul")
    matmul_leaf = next(d for d in ir.tree.preorder(matmul_blk) if isinstance(ir.tree.data(d), ISANode))
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(memset_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, 0) is not None


def test_span_promotion_allows_memset_before_matmul_kloop():
    """The legal placement (memset OUTSIDE K, at the top of the matmul nest) is
    allowed — memset is not enclosed by K, so no promotion, and it precedes the
    accumulation. index=-2 (prepend, before child 0) puts it before the K loop."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, "NKIMemset")
    matmul_blk = _block_for_op(ir, "NKIMatmul")
    matmul_leaf = next(d for d in ir.tree.preorder(matmul_blk) if isinstance(ir.tree.data(d), ISANode))
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(memset_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, -2) is None
```

The **replication reject** verdict (`_check_no_reduction_replicated`, a structural guard KEPT by this plan) and the **memset-sink-sims-correct** and **loop-var-shadow** verdicts are best re-established as part of the full-ladder validation (Step 5) rather than brittle standalone traces — the ladder exercises the real co-location moves and CPU-sims them. If a compact standalone reject test for `_check_no_reduction_replicated` is cheap to author with op-based lookups, add one; otherwise rely on the ladder. The controller will confirm the ladder covers the replication path.

**Do NOT** re-add hardcoded-nid `KernelMDP` traces. Every new test uses `_block_for_op` / op-name / loop_var lookups so it survives nid drift.

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/test_code_motion.py -v" --cache /home/weittang/workplace/cache/span5`
Expected: the 2 kept tests + the fresh verdict tests PASS; the 4 stale tests are GONE (not failing).

- [ ] **Step 5: Run the full ladder CPU-sim on gym-1 (the real gate)**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python examples/kernel_transforms.py" --cache /home/weittang/workplace/cache/ladder5`
Expected: every rung CPU-sim PASS (max_abs ~1e-4). This proves span-promotion admits every legal ladder move and the byte-exact structure is unchanged.

Also: `transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" --cache /home/weittang/workplace/cache/manual5`
Expected: k0…k27 all CPU-sim PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py nkigym/src/nkigym/transforms/compute_at.py nkigym/src/nkigym/transforms/reverse_compute_at.py test/transforms/test_code_motion.py
git commit -m "CodeMotion: legality via span-promotion only (drop skip_cover_loops + axis-covered call)

_check_move_preserves_dependencies no longer builds a cover-skip set;
_check_same_loop_prefix no longer calls _check_no_reduction_axis_covered.
Reduction-init domination + coverage now come from span-promotion. Deleted 4
stale-hardcoded-nid verdict tests (failing at base); wrote fresh op-based
span-promotion verdict tests. Full ladder CPU-sim green on gym-1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 6: Delete the CARRY/COVER machinery (gym-1-gated on SoftwarePipeline)

Now that span-promotion carries the legality, remove the dead edge-builders and helpers. This includes the SoftwarePipeline migration hazard: `must_precede` reads `_closure`, which today folds in CARRY/COVER edges. Diagnose on gym-1 which safeguard applies (clean-delete vs relocate) BEFORE deleting.

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py` — delete `_add_carry_edges`, `_add_coverage_edges`, `_carry_loops_of_leaf`, `_tiled_write_loops_of_leaf`, `_reads_independently_of_loop`; the `_build` tail calls only the base hazard walk.
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` — delete `_own_carry_loop_nids`, `_check_no_reduction_axis_covered`.
- Test: `test/ir/test_dependency.py` (retarget the `_carry_loops_of_leaf` imports at 185/205), `test/transforms/test_software_pipeline.py`.

**Interfaces:**
- Produces: `Dependency.graph` with only RAW/WAW/WAR edges (each with `tensor`). `_closure` no longer includes CARRY/COVER.

- [ ] **Step 1: Diagnose the SoftwarePipeline hazard on gym-1**

Temporarily make `_build` skip `_add_carry_edges` (comment out the call via a local edit — do NOT commit), then run:

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/test_software_pipeline.py -v" --cache /home/weittang/workplace/cache/span6`
Expected: record PASS or FAIL.
- **All PASS → safeguard (a) clean-delete.** Proceed to Step 2.
- **Any FAIL → safeguard (b) relocate.** A `must_precede` answer changed. Do NOT delete the carried relation from the model; instead keep the base RAW edges (which are what SoftwarePipeline actually needs) and confirm the failing `must_precede(a,b)` is recoverable from a base RAW/transitive path. If a genuinely CARRY-only precedence is required by SoftwarePipeline, STOP and report — that is a design escalation (the spec's (b) branch), not a mechanical delete.

Revert the temporary edit before Step 2.

- [ ] **Step 2: Write/retarget the tests that reach into deleted internals**

`test/ir/test_dependency.py` imports `_carry_loops_of_leaf` at lines ~185 and ~205. Rewrite those two tests to assert the SAME verdict via the new mechanism. The 185 test ("matmul carries psum over K") becomes an assertion on `_tensor_carried_across` (Task 2):

```python
def test_matmul_carries_psum_over_kloop():
    """The matmul RMWs psum invariantly across K, so psum is carried across the
    K loop (replaces the old _carry_loops_of_leaf assertion)."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "psum_prod") is True
```

The 205 test ("a pure load does not carry") becomes:

```python
def test_load_does_not_carry_over_kloop():
    """A pure-read load (never rmw) is not carried across any loop (replaces the
    old _carry_loops_of_leaf == {} assertion)."""
    from test.transforms._fixtures import build_canonical_ir
    from nkigym.ir.dependency import _tensor_carried_across
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.matmul import NKIMatmul

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    assert _tensor_carried_across(ir.tree, kloop, "sbuf_lhs_T") is False
```

- [ ] **Step 3: Delete the builders and helpers**

- In `dependency.py`: delete `_add_carry_edges`, `_add_coverage_edges`, `_carry_loops_of_leaf`, `_tiled_write_loops_of_leaf`, `_reads_independently_of_loop`, and `_enclosing_block_nid`/`_loopvar_to_axis` IF now unused (grep first). Change the `_build` tail: replace `self._add_carry_edges(tree)` (line 276) with nothing (the base walk is complete).
- In `_code_motion.py`: delete `_own_carry_loop_nids` and `_check_no_reduction_axis_covered`.
- Grep to confirm no remaining callers:

```bash
grep -rn "_add_carry_edges\|_add_coverage_edges\|_carry_loops_of_leaf\|_tiled_write_loops_of_leaf\|_reads_independently_of_loop\|_own_carry_loop_nids\|_check_no_reduction_axis_covered\|skip_cover_loops\|CARRY\|COVER" nkigym/ test/ examples/
```

Expected: no hits in `nkigym/` source (comments/docstrings mentioning the removed concepts should also be cleaned). Test/example hits only where a retargeted test references the new mechanism.

- [ ] **Step 4: Run dependency + software-pipeline + code-motion suites**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/ir/test_dependency.py test/transforms/test_software_pipeline.py test/transforms/test_code_motion.py -v" --cache /home/weittang/workplace/cache/span6`
Expected: PASS, no NEW failures vs the pre-existing 7.

- [ ] **Step 5: Full ladder re-run**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" --cache /home/weittang/workplace/cache/manual6`
Expected: k0…k27 CPU-sim PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Dependency: delete CARRY/COVER edge-builders and code-motion helpers

Span-promotion replaces them. SoftwarePipeline must_precede confirmed green on
gym-1 after removal (base RAW closure sufficed). Retargeted the two
_carry_loops_of_leaf tests to _tensor_carried_across.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 7: Create the unified `CodeMotion` transform

Add the public `CodeMotion` + `CodeMotionOption` to `_code_motion.py` (renamed to `code_motion.py` in Task 8). The `apply`/`analyze`/`_legal_indices`/`_check_legality` bodies are the union of the two faces — identical modulo the option class — with `is_reverse` gone and NO output-block guard.

**Files:**
- Modify: `nkigym/src/nkigym/transforms/_code_motion.py` (add the class; still named with underscore until Task 8)
- Test: `test/transforms/test_code_motion.py`

**Interfaces:**
- Consumes: `_move`, `_check_move_preserves_dependencies` (Task 5 shape), `place_buffers`, `compact_shapes`, `Dependency`.
- Produces:
  - `CodeMotionOption(block_nid: int, target_loop_nid: int, index: int)` — frozen dataclass, no `is_reverse`.
  - `CodeMotion` with `apply(ir, CodeMotionOption) -> KernelIR`, `analyze(ir) -> list[CodeMotionOption]`.

- [ ] **Step 1: Write the failing tests — store-sink allowed, and a producer-sink still works**

Add to `test/transforms/test_code_motion.py`. First the NEW store-sink verdict (the case the dropped output guard would have rejected). Build the k11-shape state via the existing `_fixtures` ladder helper up to the store-sink point, then assert `CodeMotion` ALLOWS sinking the store:

```python
def test_code_motion_allows_output_store_sink():
    """The output store (writes the return tensor) may sink under the drain's N
    loop — the dropped output-block guard would have rejected it; span-promotion
    permits it (drain writes the sbuf_prod slice the store reads, same N-iter).
    This is the _fixtures rung_13_14 move, done via CodeMotion."""
    from test.transforms._fixtures import build_ladder_state, _ladder_helpers
    from nkigym.transforms.code_motion import CodeMotion, CodeMotionOption

    state = build_ladder_state(13)
    blk, _leaf, _loop, _inner, _mm_loop, tc_loop = _ladder_helpers()
    store_blk = blk(state, "NKIStore")
    d2 = tc_loop(state, "i_d2_0")
    opt = CodeMotionOption(block_nid=store_blk, target_loop_nid=d2, index=-1)
    new_ir = CodeMotion().apply(state, opt)
    assert new_ir is not None
    assert any(o.block_nid == store_blk and o.target_loop_nid == d2 for o in CodeMotion().analyze(state))
```

This reuses the shipped `build_ladder_state(n)` (replays the `_fixtures` rung sequence to state `n`) and `_ladder_helpers()` (returns `blk`, `leaf`, `loop`, `inner`, `mm_loop`, `tc_loop` lookups). `build_ladder_state(13)` is the state just before `rung_13_14` sinks the store block under the tensor_copy's `i_d2_0` — exactly the store-sink the old `ComputeAt` output guard would have blocked but `ReverseComputeAt` (and now `CodeMotion`) allows. No new fixture helper is needed. NOTE: `build_ladder_state` itself imports `ComputeAt`/`ReverseComputeAt` today; Task 9 retargets it to `CodeMotion`, so this test's `build_ladder_state(13)` call keeps working across Task 9's edit.

- [ ] **Step 2: Run test to verify it fails**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/test_code_motion.py::test_code_motion_allows_output_store_sink -v" --cache /home/weittang/workplace/cache/span7`
Expected: FAIL — `nkigym.transforms.code_motion` / `CodeMotion` not defined (module still `_code_motion`, class absent).

- [ ] **Step 3: Add `CodeMotion` + `CodeMotionOption`**

Append to `_code_motion.py` (imports at top: `copy`, `dataclass`, `compact_shapes`, `place_buffers`, `Dependency`, `Transform`, `TransformOption`, `ForNode`, `ISANode`):

```python
@dataclass(frozen=True)
class CodeMotionOption(TransformOption):
    """Relocate ``block_nid`` under ``target_loop_nid`` at child slot ``index``.

    One option type for both directions of motion: sinking a producer under a
    consumer's loop and lifting a consumer under a producer's loop are the same
    structural splice, distinguished only by the dependency graph — not a flag.
    """

    block_nid: int
    target_loop_nid: int
    index: int


class CodeMotion(Transform):
    """Relocate one block under a target loop (the merged ComputeAt/ReverseComputeAt).

    Legality is dependency-ordering (span-promotion) + the structural same-prefix
    merge + the reduction-replication guard. There is NO output-block guard: the
    block writing the return tensor is relocatable when ordering permits (e.g. the
    k11->k12 store-sink under the matmul's N loop).
    """

    def apply(self, ir: KernelIR, option: CodeMotionOption) -> KernelIR:
        """Re-check legality, deep-copy, move, re-derive geometry, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[CodeMotionOption]:
        """Enumerate (block, target loop, index) triples passing legality."""
        options: list[CodeMotionOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                for index in self._legal_indices(ir, block_nid, target_nid):
                    opt = CodeMotionOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    try:
                        self._check_legality(ir, opt)
                    except TransformLegalityError:
                        continue
                    options.append(opt)
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children.

        Bounded below by the last child holding a producer of the moved block and
        above by the first child holding a consumer — symmetric in both, which is
        why one enumeration serves producer-sink and consumer-lift alike.
        """
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

    def _check_legality(self, ir: KernelIR, option: CodeMotionOption) -> None:
        """Structural checks (target/block in graph, target a ForNode, target not a
        descendant of the block) then span-promotion ordering. No output guard."""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"CodeMotion requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot move under its own loop)"
            )
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index)
```

Also update `_move`'s signature to drop `is_reverse` (delete the parameter and the "structurally inert" docstring line); update `__all__` to add `CodeMotion`, `CodeMotionOption`. **Ripple:** dropping `is_reverse` from `_move` breaks every existing caller passing it. Grep and fix them now:

```bash
grep -rn "_move(.*is_reverse" nkigym/ test/
```

Update each `_move(..., is_reverse=False)` / `is_reverse=True)` call (in `test/ir/test_dependency.py` and the Task 3 test added there) to drop the `is_reverse=...` argument. Run `python -m pytest test/ir/test_dependency.py -v` on gym-1 after, to confirm no call site was missed.

- [ ] **Step 4: Run the store-sink test + a producer-sink regression**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/test_code_motion.py -v" --cache /home/weittang/workplace/cache/span7`
Expected: `test_code_motion_allows_output_store_sink` PASS; the four pinned verdicts still PASS (they construct via `ComputeAt`/`ReverseComputeAt` still — those faces delegate to the shared helpers, unaffected until Task 8).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/transforms/_code_motion.py test/transforms/test_code_motion.py test/transforms/_fixtures.py
git commit -m "CodeMotion: add unified transform (no is_reverse, no output guard)

One CodeMotion/CodeMotionOption replacing the two faces' identical analyze/apply
bodies. Store-sink (k11->k12 shape) now allowed — span-promotion governs the
output block like any other.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 8: Rename the module, delete the two faces, update `__init__`

**Files:**
- Rename: `nkigym/src/nkigym/transforms/_code_motion.py` → `code_motion.py` (`git mv`)
- Delete: `nkigym/src/nkigym/transforms/compute_at.py`, `reverse_compute_at.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py`
- Rename: `nkigym/src/nkigym/transforms/compute_at_legality.md` → `code_motion_legality.md`

**Interfaces:**
- Produces: `from nkigym.transforms import CodeMotion, CodeMotionOption`. `ComputeAt`/`ReverseComputeAt` no longer importable.

- [ ] **Step 1: git mv the module and legality doc**

```bash
cd /workplace/weittang/nki-autotune
git mv nkigym/src/nkigym/transforms/_code_motion.py nkigym/src/nkigym/transforms/code_motion.py
git mv nkigym/src/nkigym/transforms/compute_at_legality.md nkigym/src/nkigym/transforms/code_motion_legality.md
```

- [ ] **Step 2: Delete the two faces**

```bash
git rm nkigym/src/nkigym/transforms/compute_at.py nkigym/src/nkigym/transforms/reverse_compute_at.py
```

- [ ] **Step 3: Update `__init__.py`**

Replace lines 4 and 7 (the two face imports) with one, and swap the `__all__` entries:

```python
from nkigym.transforms.code_motion import CodeMotion, CodeMotionOption
```

`__all__`: remove `"ComputeAt"`, `"ComputeAtOption"`, `"ReverseComputeAt"`, `"ReverseComputeAtOption"`; add `"CodeMotion"`, `"CodeMotionOption"` (keep alphabetical-ish order consistent with the file).

- [ ] **Step 4: Fix internal imports of the old module path**

Grep for `_code_motion` importers and retarget to `code_motion`:

```bash
grep -rn "transforms._code_motion\|transforms\.compute_at\|transforms\.reverse_compute_at" nkigym/ test/ examples/
```

Update each (`test_code_motion.py:8` imports `_move` from `transforms._code_motion` → `transforms.code_motion`; any other hits).

- [ ] **Step 5: De-stale `code_motion_legality.md`**

Edit the renamed doc: retitle to "CodeMotion legality"; replace the `ComputeAt`/`ReverseComputeAt` two-face framing with the single `CodeMotion`; fix the stale `_check_move_realizable` reference (the shipped code uses `_check_same_loop_prefix` + span-promotion); state the output-block guard is dropped and why (k11→k12). Keep it concise — integrate, do not append.

- [ ] **Step 6: Import smoke test on gym-1**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -c 'from nkigym.transforms import CodeMotion, CodeMotionOption; print(CodeMotion, CodeMotionOption)'" --cache /home/weittang/workplace/cache/span8`
Expected: prints the class + option; no ImportError.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "transforms: rename _code_motion->code_motion, delete ComputeAt/ReverseComputeAt

One public CodeMotion replaces the two faces. __init__ exports swapped;
legality doc renamed + de-staled.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 9: Retarget all callers and fold the face-specific tests

**Files:**
- Modify: `examples/kernel_transforms.py`, `examples/matmul_lhsT_rhs.py`
- Modify: `test/transforms/_fixtures.py`, `_pipeline_fixtures.py`, `test_reorder.py`, `test_split.py`
- Merge + delete: `test/transforms/test_compute_at.py`, `test_reverse_compute_at.py` → `test_code_motion.py`

**Interfaces:**
- Consumes: `CodeMotion`, `CodeMotionOption`.

- [ ] **Step 1: Retarget the examples**

In `examples/kernel_transforms.py`: change the imports (`ComputeAt`, `ComputeAtOption`, `ReverseComputeAt`, `ReverseComputeAtOption` → `CodeMotion`, `CodeMotionOption`), the `KernelMDP(..., transforms=[...])` list (replace both faces with a single `CodeMotion()`), and the 7 option call sites (`ComputeAtOption(...)`/`ReverseComputeAtOption(...)` → `CodeMotionOption(...)`, same args; `ComputeAt().apply`/`ReverseComputeAt().apply` → `CodeMotion().apply`). Same in `examples/matmul_lhsT_rhs.py` (import + `KernelMDP` transforms list).

- [ ] **Step 2: Retarget the fixtures and other transform tests**

In `_fixtures.py`, `_pipeline_fixtures.py`, `test_reorder.py`, `test_split.py`: replace `ComputeAt`/`ReverseComputeAt`/`*Option` imports and call sites with `CodeMotion`/`CodeMotionOption`. The moves are unchanged — only the class/option names. (Where a comment says "ComputeAt"/"ReverseComputeAt" describing the move, update to "CodeMotion".)

- [ ] **Step 3: Fold the face-specific test files**

Move the tests from `test_compute_at.py` and `test_reverse_compute_at.py` into `test_code_motion.py`, renaming any `ComputeAt`/`ReverseComputeAt` construction to `CodeMotion`. Where two tests are the same verdict via different faces (e.g. a producer-sink in `test_compute_at.py` and its mirror in `test_reverse_compute_at.py`), keep both as distinct scenarios (they exercise different tree shapes) — preserve every assertion. Then:

```bash
git rm test/transforms/test_compute_at.py test/transforms/test_reverse_compute_at.py
```

- [ ] **Step 4: Grep for any surviving reference**

```bash
grep -rn "ComputeAt\|ReverseComputeAt\|compute_at\|reverse_compute_at" nkigym/ test/ examples/
```

Expected: only `code_motion` / `CodeMotion` (and the renamed legality doc). No `ComputeAt`/`ReverseComputeAt` symbol survives. Historical mentions in `docs/` design specs are fine (they are dated records).

- [ ] **Step 5: Full suite + both ladders on gym-1**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest test/transforms/ test/ir/test_dependency.py -v" --cache /home/weittang/workplace/cache/span9`
Expected: PASS, no NEW failures vs the pre-existing 7.

Run: `transport/ssh_host.sh --host gym-1 --cmd "python examples/kernel_transforms.py" --cache /home/weittang/workplace/cache/ladder9`
Expected: every rung CPU-sim PASS.

Run: `transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" --cache /home/weittang/workplace/cache/manual9`
Expected: k0…k27 CPU-sim PASS (crucially k11→k12 store-sink under `CodeMotion`).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "transforms: retarget all callers to CodeMotion; fold face tests

examples + fixtures + reorder/split tests use CodeMotion/CodeMotionOption;
test_compute_at.py + test_reverse_compute_at.py folded into test_code_motion.py.
Full suite + both ladders green on gym-1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Task 10: Update the shipped design records

**Files:**
- Modify: `.claude/rules/learnings.md`, `.claude/rules/tvm_knowledge.md` (the parked block-granular question)

**Interfaces:** none (documentation).

- [ ] **Step 1: Update learnings**

Add a Verification/Architecture bullet recording the shipped mechanism: "Code-motion legality = per-tensor SPAN-PROMOTION (a carried, loop-invariant access's span widens to the carrying loop); CARRY/COVER/skip_cover_loops/reduction-axis-covered DELETED. `ComputeAt`+`ReverseComputeAt` merged into `CodeMotion` (no output guard — store-sink k11→k12 legal)." Convert the date. Keep it one line.

- [ ] **Step 2: Update the parked TVM question**

In `tvm_knowledge.md`, the "⏳ TO REVISIT: block-granular dependency model" section: note it is resolved for the code-motion path by span-promotion (the loop-span cases are handled by widening a carried access's span, no region-cover port), the SoftwarePipeline static layer is untouched. Keep the historical record; append a dated resolution line, do not delete the section.

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/learnings.md .claude/rules/tvm_knowledge.md
git commit -m "docs: record span-promotion + CodeMotion merge in learnings/tvm_knowledge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
