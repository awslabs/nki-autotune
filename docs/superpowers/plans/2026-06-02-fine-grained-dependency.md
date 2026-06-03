# Fine-Grained Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-key the `Dependency` graph from block-granular to ISANode/ForNode nids and add carried-state domination edges, so `ComputeAt`/`ReverseComputeAt` legality becomes a single move-simulating, span-based check that correctly rejects sinking an accumulator's init/drain into its reduction loop (and a consumer before its producer).

**Architecture:** `Dependency` builds a graph whose nodes are ISA-leaf nids plus the ForNode nids that are carry-loop endpoints. Flow edges (RAW/WAW/WAR, region-overlap gated) connect leaf→leaf exactly as today (canonical graph stays isomorphic). New carry edges connect producer-leaf→carry-loop and carry-loop→consumer-leaf whenever a non-PARALLEL loop carries a buffer (the operand whose region omits the loop's axis). The `kind` label is provenance only — legality reads it never. `Dependency` exposes a pure `first_backward_edge(leaf)` span-check over its own tree; the transform faces (in `transforms/`, where `_move` lives — `ir/` must not import `transforms/`) call a shared `_check_move_preserves_dependencies` that deep-copies, `_move`s, rebuilds `Dependency`, and asks `first_backward_edge`. One span rule (`span(a).end < span(b).start`) covers reduction-init domination AND consumer-before-producer ordering, replacing both faces' bespoke condition-5 logic. Public query methods accept a block nid or a leaf nid (block→owned-leaf bijection), so existing call sites are unaffected.

**Tech Stack:** Python 3.12, networkx, pytest, numpy, NKI CPU simulator. No new deps.

**Spec:** `docs/superpowers/specs/2026-06-02-fine-grained-dependency-design.md`.

**Environment:** `source ~/venvs/kernel-env/bin/activate`; run from repo root `/home/ubuntu/nki-autotune`; tests run with `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src`. HEAD `0f5f815`, suite **215 passed** (includes the ad-hoc guard `51c2361` + its test, which Task 5 removes). A `check-python-style.py` hook runs on `.py` edits; commit hook runs autoflake/isort/black.

---

## Key API facts (verified against current code — read before starting)

- **`Dependency`** (`nkigym/ir/dependency.py`): builds `self.graph` (nx.DiGraph) over **block nids** today, `self._closure = nx.transitive_closure(graph)`. Public: `producers(nid)`, `consumers(nid)`, `must_precede(a,b)`, `direct_producers/consumers`, `info(nid)`, `chains()`, `dump(dir)`. Build walks `_nodes_in_execution_order(tree)` (single-leaf-owning blocks ordered by their owned leaf's pre-order position), `_record_hazards` over `last_writer`/`prior_readers`, `_try_edge` gated by `_provably_disjoint` (which calls `regions_disjoint` from `ir/interval.py`).
- **block↔leaf is a bijection** at dependency-node level: every dependency block owns exactly one *direct* ISA leaf (nearest-enclosing-block). Canonical: block 1→leaf 3 (load), 4→6 (load), 7→9 (memset), 10→14 (matmul), 15→17 (tensor_copy), 18→20 (store).
- **Role access**: `role_of(block, axis)` (`ir/tree.py`) returns the `AxisRole` for an axis; raises `KeyError` if absent. `AxisRole` (`ops/base.py`) ∈ {`PARALLEL`, `SEQUENTIAL`, `ACCUMULATION`}. A block's iter_var carries `role`; the matmul block's `d0` (K) is `ACCUMULATION`.
- **loopvar→axis**: a ForNode's `loop_var` maps to a block iter-var axis via the block's `iter_values` affine (`to_affine(value)` keys). Helpers exist: `_domain_solve._loopvar_to_dim(tree, block_nid)` and `_enclosing_block(tree, nid)`.
- **carried buffer**: for an ISA leaf `M` whose op has a non-PARALLEL axis `a` (from the enclosing block's iter_var role on the loop's bound axis), the carried buffer is the operand whose `OPERAND_AXES` tuple does NOT contain `a`. Verified: matmul `dst(M,N)` omits `K`→`psum_prod`; inputs `stationary(K,M)`/`moving(K,N)` contain `K`. The memset block `.writes` includes `psum_prod` (producer detection works on block writes).
- **`BlockNode`** payload: `iter_vars` (each `IterVar(axis,dom,role)`), `iter_values`, `reads`/`writes` (`BufferRegion`s), `alloc_buffers`, `axis_map`. `ISANode`: `op_cls`, `operand_bindings: dict[slot, BufferRegion]`. `ForNode`: `loop_var`, `extent`.
- **`KernelTree`**: `data(nid)`, `children/parent/ancestors/descendants/preorder/blocks`, `.root`, `.graph`.
- **compute_at faces**: `compute_at.py` / `reverse_compute_at.py` each have `apply` (check legality → `_move` → place/compact/Dependency), `analyze`, `_legal_indices`, `_check_legality` (conditions: target-in-graph, is-ForNode, block-in-graph, target-not-descendant-of-block, [ComputeAt: output guard], then `_check_consumers_visited`/`_check_producers_visited` + `_root_sibling_of`). `compute_at.py` ALSO has the ad-hoc `_check_no_writer_under_accumulation` from `51c2361` (Task 5 deletes it).

---

## File Structure

- `nkigym/src/nkigym/ir/dependency.py` — EDIT. Re-key graph to leaf/loop nids; add carry edges; add pure `first_backward_edge()`; public methods resolve block→leaf.
- `nkigym/src/nkigym/transforms/_code_motion.py` — EDIT. Add `_check_move_preserves_dependencies` (deep-copy → `_move` → rebuild Dependency → `first_backward_edge`).
- `nkigym/src/nkigym/transforms/compute_at.py` — EDIT. Delete `_check_consumers_visited` + `_check_no_writer_under_accumulation` + `_root_sibling_of`; legality calls `_check_move_preserves_dependencies`.
- `nkigym/src/nkigym/transforms/reverse_compute_at.py` — EDIT. Delete `_check_producers_visited` + `_root_sibling_of`; legality calls `_check_move_preserves_dependencies`.
- `test/ir/test_dependency.py` — EDIT. Existing `must_precede` assertions work unchanged (block→leaf resolve); add carry-edge unit tests.
- `test/transforms/test_compute_at.py` — EDIT. The `51c2361` rejection test re-expressed against the model.

---

## Phase outline

1. Carry-loop + carried-buffer detection helpers (Task 1) — pure, unit-tested.
2. Re-key the graph to leaf nids + block→leaf resolve on public API (Task 2) — canonical graph isomorphic; existing tests pass.
3. Add carry-domination edges to the graph (Task 3) — the new edges + units.
4. Pure `first_backward_edge()` span-check (Task 4).
5. Move-sim wrapper + rewrite both faces' legality onto it; delete bespoke checks + the ad-hoc guard (Task 5).
6. Full regression: ladder byte-exact, MDP reduction-class clean (Task 6).

---

## Phase 1 — Detection helpers

### Task 1: carry-loop + carried-buffer detection

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py`

Pure helpers (module-level functions, no graph state): given the tree and an ISA-leaf nid, find its enclosing non-PARALLEL loops and the buffer each carries.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_dependency.py`:

```python
def test_carry_loops_of_matmul_leaf():
    """The matmul leaf's K loop (d0, ACCUMULATION) carries psum_prod; M/N (PARALLEL) carry nothing."""
    from nkigym.ir.dependency import _carry_loops_of_leaf
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    carries = _carry_loops_of_leaf(ir.tree, matmul_leaf)
    """carries maps carry-loop nid -> carried buffer name."""
    carried_buffers = set(carries.values())
    assert carried_buffers == {"psum_prod"}, carried_buffers
    """Exactly one carry loop (the single canonical K loop, ACCUMULATION)."""
    assert len(carries) == 1
    (kloop_nid,) = carries
    from nkigym.ir.tree import ForNode
    assert isinstance(ir.tree.data(kloop_nid), ForNode)
    assert ir.tree.data(kloop_nid).loop_var == "i_d0_0"


def test_carry_loops_empty_for_pure_parallel_leaf():
    """A load leaf (all-PARALLEL axes) has no carry loops."""
    from nkigym.ir.dependency import _carry_loops_of_leaf
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    load_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKILoad
    )
    assert _carry_loops_of_leaf(ir.tree, load_leaf) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k carry`
Expected: FAIL `ImportError: cannot import name '_carry_loops_of_leaf'`.

- [ ] **Step 3: Implement the helpers**

Add to `nkigym/src/nkigym/ir/dependency.py` (add imports `from nkigym.ir.tree import role_of` and `from nkigym.ops.base import AxisRole` at top; `_loopvar_to_dim` and `_enclosing_block` already importable from `nkigym.transforms._domain_solve` — but to avoid a transforms→ir dependency inversion, inline the two tiny helpers here instead):

```python
def _enclosing_block_nid(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor of ``nid``."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


def _loopvar_to_axis(block: BlockNode) -> dict[str, str]:
    """Map each loop_var bound by the block to its concrete iter-var axis (via iter_values affine)."""
    from nkigym.ir.expr import to_affine

    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                out[name] = iv.axis
    return out


def _carry_loops_of_leaf(tree: KernelTree, leaf_nid: int) -> dict[int, str]:
    """Map each enclosing non-PARALLEL loop of ``leaf_nid`` to the buffer it carries.

    A loop carries state when its bound axis has SEQUENTIAL or ACCUMULATION
    role for the leaf's enclosing block. The carried buffer is the leaf
    operand whose ``OPERAND_AXES`` tuple omits that loop's axis (the value
    live across the loop). Loops whose axis is PARALLEL, or which carry no
    such operand, are skipped.
    """
    data = tree.data(leaf_nid)
    assert isinstance(data, ISANode)
    block_nid = _enclosing_block_nid(tree, leaf_nid)
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    lv_to_axis = _loopvar_to_axis(block)
    inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    op_axes = data.op_cls.OPERAND_AXES
    out: dict[int, str] = {}
    for anc in tree.ancestors(leaf_nid):
        anc_data = tree.data(anc)
        if not isinstance(anc_data, ForNode):
            continue
        concrete = lv_to_axis.get(anc_data.loop_var)
        if concrete is None:
            continue
        if role_of(block, concrete) == AxisRole.PARALLEL:
            continue
        abstract = inverse_axis_map.get(concrete)
        for slot, axes in op_axes.items():
            if abstract is not None and abstract not in axes and slot in data.operand_bindings:
                out[anc] = data.operand_bindings[slot].tensor
    return out
```

Note: `concrete` axis (e.g. `d0`) maps to the op's abstract axis (`K`) via `block.axis_map`'s inverse; `OPERAND_AXES` is keyed by abstract names, so the omission test uses `abstract`. A loop with no non-PARALLEL role, or whose carried-axis no operand omits, contributes nothing.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k carry`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Add carry-loop / carried-buffer detection to dependency"
```

---

## Phase 2 — Re-key to leaf nids

### Task 2: graph nodes = leaf nids; public API resolves block→leaf

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py`

The graph currently keys hazards on the owning-block nid. Re-key to the **owned ISA-leaf nid**. Because block↔leaf is a bijection, add a `_resolve(nid)` that maps a block nid → its owned leaf nid (and a leaf/loop nid → itself), and call it at the top of every public query so existing call sites passing block nids keep working.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_dependency.py`:

```python
def test_dependency_graph_keyed_on_leaf_nids():
    """The dependency graph nodes are ISA-leaf nids, not block nids."""
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    for node in ir.dependency.graph.nodes:
        assert isinstance(ir.tree.data(node), ISANode), f"node {node} is not an ISA leaf"


def test_must_precede_accepts_block_or_leaf_nids():
    """must_precede works whether given block nids (legacy) or leaf nids (resolved either way)."""
    ir = build_canonical_ir()
    matmul_blk = _block_for_op(ir, NKIMatmul)
    store_blk = _block_for_op(ir, NKIStore)
    from nkigym.ir.tree import ISANode

    def leaf_of(blk):
        return next(
            d for d in ir.tree.preorder(blk)
            if isinstance(ir.tree.data(d), ISANode)
            and next(a for a in reversed(ir.tree.ancestors(d)) if isinstance(ir.tree.data(a), BlockNode)) == blk
        )

    """Block nids (legacy call style) still order correctly."""
    assert ir.dependency.must_precede(matmul_blk, store_blk)
    """Leaf nids resolve to the same answer."""
    assert ir.dependency.must_precede(leaf_of(matmul_blk), leaf_of(store_blk))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k "leaf_nids or accepts_block"`
Expected: FAIL — `test_dependency_graph_keyed_on_leaf_nids` fails (nodes are block nids today).

- [ ] **Step 3: Re-key the build + add `_resolve`**

In `dependency.py`:

1. Add a leaf-owner map built once in `__init__` (before `_build`):

```python
        self._leaf_of_block: dict[int, int] = {}
        self._owner_block: dict[int, int] = {}
```

2. Change `_build` to key on the leaf nid. Replace the `for nid in self._nodes_in_execution_order(tree):` loop body to use the owned leaf nid as the graph key, recording the block↔leaf maps. Concretely, rewrite `_nodes_in_execution_order` to return `(leaf_nid, block_nid)` pairs (it already finds the owner per leaf):

```python
    @staticmethod
    def _leaves_in_execution_order(tree: KernelTree) -> list[tuple[int, int]]:
        """Return (leaf_nid, owning_block_nid) pairs in ISA pre-order."""
        ordered: list[tuple[int, int]] = []
        seen: set[int] = set()
        for leaf in tree.preorder():
            if not isinstance(tree.data(leaf), ISANode):
                continue
            owner = next(a for a in reversed(tree.ancestors(leaf)) if isinstance(tree.data(a), BlockNode))
            if owner in seen:
                raise AssertionError(f"block {owner} owns more than one ISA leaf; dependency model requires one")
            seen.add(owner)
            ordered.append((leaf, owner))
        return ordered
```

And `_build` becomes:

```python
    def _build(self, tree: KernelTree) -> None:
        buffers = self._buffer_map(tree)
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for leaf_nid, block_nid in self._leaves_in_execution_order(tree):
            self._leaf_of_block[block_nid] = leaf_nid
            self._owner_block[leaf_nid] = block_nid
            block = tree.data(block_nid)
            assert isinstance(block, BlockNode)
            info = self._summarise(block_nid, block, tree, buffers)
            self.graph.add_node(leaf_nid, info=info)
            self.blocks.append(leaf_nid)
            for name in info.reads | info.writes:
                self.touches_by_tensor.setdefault(name, []).append(leaf_nid)
            self._record_hazards(leaf_nid, info, last_writer, prior_readers)
            for name in info.writes:
                last_writer[name] = leaf_nid
                prior_readers.pop(name, None)
            for name in info.reads - info.writes:
                prior_readers.setdefault(name, []).append(leaf_nid)
```

(`_summarise`, `_record_hazards`, `_try_edge`, `_provably_disjoint` are unchanged — they operate on whatever nid is the graph key. `info` extents are still gathered from the block's descendant ForNodes.)

3. Add `_resolve` and call it in the public queries:

```python
    def _resolve(self, nid: int) -> int:
        """Map a block nid to its owned ISA-leaf nid; a leaf/loop nid maps to itself."""
        return self._leaf_of_block.get(nid, nid)
```

In `producers`, `consumers`, `must_precede`, `direct_producers`, `direct_consumers`, `info`: wrap each `nid` argument with `self._resolve(nid)` before indexing the graph/closure. E.g.:

```python
    def producers(self, nid: int) -> set[int]:
        return set(self._closure.predecessors(self._resolve(nid)))

    def must_precede(self, producer: int, consumer: int) -> bool:
        return self._closure.has_edge(self._resolve(producer), self._resolve(consumer))
```

Keep return values as leaf nids — the existing transform callers compare set membership against block nids, so ALSO resolve on the comparison side (moot after Task 5, which replaces the faces' producer/consumer-set logic with the move-sim check). For now the dep tests pass block nids in and check booleans, which resolve symmetrically.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q`
Expected: all pass (the existing `must_precede` block-nid tests resolve through `_resolve`; the two new tests pass).

- [ ] **Step 5: Full transforms regression (the faces still call producers/consumers with block nids)**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/ -q`
Expected: all pass. The faces' `producers(block_nid)`/`consumers(block_nid)` now return LEAF nids, but their `_check_*_visited` compares against `ir.tree.descendants(...)` / root-sibling sets that contain block nids → could mismatch. IF any transform test fails here, that is EXPECTED churn that Task 5 fixes by replacing the faces' bespoke checks with the move-sim legality. Record which fail; do NOT patch the faces in this task. If NONE fail (because resolve happens to align), note that. Re-run the dependency + ir suite to confirm the graph re-key itself is clean: `python -m pytest test/ir/ -q`.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Re-key dependency graph to ISA-leaf nids; public API resolves block->leaf"
```

> **Note for executor:** if Step 5 shows transform-test failures, commit anyway (the graph re-key is correct and self-consistent); Task 5 restores green by replacing the faces' legality. If you prefer a continuously-green tree, do Task 5 immediately after this commit before running the full suite.

---

## Phase 3 — Carry edges

### Task 3: add carry-domination edges to the graph

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py`

After flow edges are built, add carry edges: for each leaf `M` with carry loops (Task 1), connect every producer/consumer of the carried buffer to the carry loop.

- [ ] **Step 1: Write the failing test**

Append to `test/ir/test_dependency.py`:

```python
def test_carry_edges_memset_dominates_kloop_and_kloop_dominates_drain():
    """Canonical: memset_leaf -> K_loop and K_loop -> tensor_copy_leaf carry edges exist."""
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset

    ir = build_canonical_ir()
    dep = ir.dependency

    def leaf(op_cls):
        return next(
            n for n in ir.tree.preorder()
            if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is op_cls
        )

    memset_leaf = leaf(NKIMemset)
    matmul_leaf = leaf(NKIMatmul)
    tc_leaf = leaf(NKITensorCopy)
    kloop = next(a for a in ir.tree.ancestors(matmul_leaf)
                 if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0")

    assert dep.graph.has_edge(memset_leaf, kloop), "memset must dominate the K loop"
    assert dep.graph.has_edge(kloop, tc_leaf), "K loop must dominate the drain (tensor_copy)"


def test_no_carry_edge_for_input_loads():
    """The lhs_T load (writes sbuf_lhs_T, indexed by K) gets NO edge to the K loop."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    dep = ir.dependency
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(a for a in ir.tree.ancestors(matmul_leaf)
                 if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0")
    load_leaves = [
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKILoad
    ]
    for ll in load_leaves:
        assert not dep.graph.has_edge(ll, kloop), f"load {ll} must NOT be forced to dominate K"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k "carry_edges or no_carry_edge"`
Expected: `test_carry_edges_*` FAILS (no such edges yet); `test_no_carry_edge_for_input_loads` passes vacuously.

- [ ] **Step 3: Add carry edges in `_build`**

At the END of `_build` (after the flow-edge loop), add a pass that adds carry edges, then recompute `_closure` (the `_closure` is built in `__init__` after `_build` returns, so just adding to `self.graph` here is enough — verify `__init__` order). Append to `_build`:

```python
        self._add_carry_edges(tree)

    def _add_carry_edges(self, tree: KernelTree) -> None:
        """For each leaf with carry loops, add producer->loop and loop->consumer edges.

        A buffer carried across a non-PARALLEL loop ``L`` must be fully
        produced before ``L`` (init dominates) and only consumed after ``L``
        (drain post-dominates). Producers/consumers of the carried buffer are
        the graph's existing writers/readers of that tensor (``touches_by_tensor``
        filtered by write/read side). The reducer leaf itself is exempt.
        """
        for leaf_nid in list(self.graph.nodes):
            carries = _carry_loops_of_leaf(tree, leaf_nid)
            for loop_nid, tensor in carries.items():
                self.graph.add_node(loop_nid)
                for other in self.touches_by_tensor.get(tensor, ()):
                    if other == leaf_nid:
                        continue
                    info = self.graph.nodes[other]["info"]
                    if tensor in info.writes:
                        self.graph.add_edge(other, loop_nid, kind="CARRY")
                    if tensor in info.reads and tensor not in info.writes:
                        self.graph.add_edge(loop_nid, other, kind="CARRY")
```

Note: a producer is a leaf that WRITES the carried tensor (memset writes psum_prod); a pure consumer READS but does not write it (tensor_copy reads psum_prod, writes sbuf_prod). The matmul itself both reads+writes psum_prod (RMW) AND is the reducer — it's excluded by `other == leaf_nid` for its own carry loop. The loop nid is added as a graph node so the closure includes it.

Confirmed: `__init__` builds `self._closure` on the line AFTER `self._build(tree)` returns (`dependency.py:52-53`), so carry edges added at the end of `_build` are in `self.graph` before the closure is computed — no reordering needed.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q`
Expected: all pass, including the two carry-edge tests.

- [ ] **Step 5: Confirm flow graph still isomorphic (no spurious flow changes)**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/ test/codegen/ -q`
Expected: all pass (carry edges are additive; flow edges + region tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Add carry-domination edges (producer->carry-loop->consumer) to dependency"
```

---

## Phase 4 — pure span-based ordering check

### Task 4: `first_backward_edge()` — the single span-based check (kind-agnostic)

**Files:**
- Modify: `nkigym/src/nkigym/ir/dependency.py`
- Test: `test/ir/test_dependency.py`

The legality question, answered with ONE rule reading no edge kind: after
the proposed move, does any dependency edge incident to the moved node
point **backward** in execution order? `Dependency.first_backward_edge`
answers it as a pure check on a given tree; the move-sim wrapper (Task 5)
applies it to the proposed move.

The model: every graph node has a **preorder span** `[start, end]` over
the tree — a leaf is a point (`start == end`); a loop spans its whole
subtree (the K-loop spans all positions inside it). An edge `a → b`
("a before b") is **satisfied** iff `span(a).end < span(b).start` and
**violated (backward)** otherwise. A carry edge `memset → K-loop` and a
flow edge `matmul → tensor_copy` are checked by the **same** comparison;
the loop endpoint's wider span is precisely what encodes "must be entirely
outside-and-before the loop." This subsumes the reduction-init hole AND
the consumer-before-producer hole in one rule.

`Dependency` exposes a **pure** span-check over its own tree
(`first_backward_edge`). The move-simulation that *uses* it (deep-copy →
`_move` → rebuild `Dependency` → check) lives in the transform faces
(Task 5), because `_move` is in `transforms/` and **`ir/` must not import
`transforms/`** (verified: `ir/` is transforms-free today; `_code_motion`
imports `from nkigym.ir import KernelIR`, so `dependency.py` importing
`_code_motion` would be a circular import AND violate the ir↛transforms
invariant). Keeping the pure check in `ir/` and the simulation in
`transforms/` respects the direction.

- [ ] **Step 1: Write the failing tests (both holes + the legal case)**

These build the moved tree by hand (`_move` on a deep copy), rebuild
`Dependency` on it, and assert `first_backward_edge(moved_leaf)`. Append
to `test/ir/test_dependency.py`:

```python
def test_first_backward_edge_flags_memset_sunk_under_kloop():
    """After sinking the memset (writer of psum_prod, carried over K) under the
    K loop, the memset->K-loop carry edge points backward."""
    import copy
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, NKIMemset)
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(a for a in ir.tree.ancestors(matmul_leaf)
                 if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0")
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=memset_blk, target_loop_nid=kloop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    memset_leaf = next(
        n for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKIMemset
    )
    assert dep.first_backward_edge(memset_leaf) is not None


def test_first_backward_edge_flags_consumer_before_producer():
    """Sinking the tensor_copy (consumer of psum_prod) under the MEMSET's loop puts it
    before the matmul that produces psum_prod -> backward flow edge matmul->tensor_copy."""
    import copy
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.ops.memset import NKIMemset
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    tc_blk = _block_for_op(ir, NKITensorCopy)
    memset_blk = _block_for_op(ir, NKIMemset)
    memset_loop = next(
        d for d in ir.tree.preorder(memset_blk) if isinstance(ir.tree.data(d), ForNode)
    )
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=tc_blk, target_loop_nid=memset_loop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    tc_leaf = next(
        n for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKITensorCopy
    )
    assert dep.first_backward_edge(tc_leaf) is not None


def test_first_backward_edge_allows_load_under_kloop():
    """Sinking the lhs_T load (writes sbuf_lhs_T, NOT carried over K) under K is legal -> None."""
    import copy
    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms._code_motion import _move

    ir = build_canonical_ir()
    load_blk = _block_for_op(ir, NKILoad)
    matmul_leaf = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    kloop = next(a for a in ir.tree.ancestors(matmul_leaf)
                 if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0")
    moved = copy.deepcopy(ir)
    _move(moved, block_nid=load_blk, target_loop_nid=kloop, index=0, is_reverse=False)
    dep = Dependency(moved.tree)
    load_leaf = next(
        n for n in moved.tree.preorder()
        if isinstance(moved.tree.data(n), ISANode) and moved.tree.data(n).op_cls is NKILoad
    )
    assert dep.first_backward_edge(load_leaf) is None
```

(`_block_for_op(ir, NKILoad)` returns the FIRST load block, lhs_T — fine.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k backward`
Expected: FAIL `AttributeError: 'Dependency' object has no attribute 'first_backward_edge'`.

- [ ] **Step 3: Implement `first_backward_edge` (pure span-check on Dependency's OWN tree)**

Store the tree handle in `__init__` (`self._tree = tree` before
`self._build(tree)`). Add to `dependency.py`:

```python
    def first_backward_edge(self, moved_leaf_nid: int) -> tuple[int, int] | None:
        """Return the first dependency edge incident to ``moved_leaf_nid`` that
        points backward in this tree's execution order, else ``None``.

        One rule, no edge-kind. Each node has a preorder span ``[start, end]``
        over the tree (a leaf is a point; a loop spans its whole subtree). An
        edge ``a -> b`` ("a before b") is satisfied iff ``span(a).end <
        span(b).start`` and backward otherwise. A carry edge to a loop and a
        flow edge to a leaf are checked identically; the loop's wider span
        encodes "outside-and-before the whole loop". Callers that want to test
        a *proposed* move build the moved tree, construct a fresh
        ``Dependency`` on it, and call this with the moved leaf nid.
        """
        order = {n: i for i, n in enumerate(self._tree.preorder())}

        def span(nid: int) -> tuple[int, int]:
            idxs = [order[d] for d in (self._tree.descendants(nid) | {nid}) if d in order]
            return (min(idxs), max(idxs))

        result: tuple[int, int] | None = None
        for a, b in self.graph.edges():
            if a != moved_leaf_nid and b != moved_leaf_nid:
                continue
            if not (span(a)[1] < span(b)[0]):
                result = (a, b)
                break
        return result
```

No new imports, no IR handle, no cycle — purely reads `self._tree` and
`self.graph`. `Dependency.__init__` signature is unchanged (`tree` only).
The Step-1 tests are the real gate for this pure check; the
move-simulating wrapper that calls it is added + tested in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/ir/test_dependency.py -q -k backward`
Expected: 3 passed. Also confirm no import cycle: `python -c "import nkigym.ir.dependency"`.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/dependency.py test/ir/test_dependency.py
git commit -m "Add Dependency.first_backward_edge(): pure span-based ordering check"
```

---

## Phase 5 — Rewrite the faces' legality

### Task 5: ComputeAt + ReverseComputeAt legality via move-sim dependency check; delete bespoke checks

**Files:**
- Modify: `nkigym/src/nkigym/transforms/compute_at.py`, `nkigym/src/nkigym/transforms/reverse_compute_at.py`
- Test: `test/transforms/test_compute_at.py`

Add a shared move-simulation legality helper to `_code_motion.py` (where
`_move` already lives — no new import direction), then both faces call it,
deleting their bespoke condition-5 checks (`_check_consumers_visited` /
`_check_producers_visited` / `_root_sibling_of`) and the ad-hoc
`_check_no_writer_under_accumulation` (`51c2361`).

- [ ] **Step 1: Add `_check_move_preserves_dependencies` to `_code_motion.py`**

This is the move-simulating wrapper that the pure `first_backward_edge`
(Task 4) supports. It lives in `transforms/` (legal to import both `_move`
and `Dependency`). Add to `nkigym/src/nkigym/transforms/_code_motion.py`:

```python
def _check_move_preserves_dependencies(
    ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool
) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Simulates the move on a deep copy, rebuilds the Dependency graph on the
    moved tree, and asks ``first_backward_edge`` for the moved leaf. One
    span-based, edge-kind-agnostic rule — covers reduction-init domination
    and consumer-before-producer ordering alike.
    """
    import copy

    from nkigym.ir.dependency import Dependency
    from nkigym.ir.tree import BlockNode, ISANode
    from nkigym.transforms.base import TransformLegalityError

    sim = copy.deepcopy(ir)
    _move(sim, block_nid=block_nid, target_loop_nid=target_loop_nid, index=index, is_reverse=is_reverse)
    dep = Dependency(sim.tree)
    """The moved block's leaf nid in the simulated tree: blocks keep their nids
    across deepcopy, so block_nid still identifies the moved block; resolve to
    its owned leaf via the rebuilt graph."""
    moved_leaf = dep._resolve(block_nid)
    offending = dep.first_backward_edge(moved_leaf)
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
```

Export it in `_code_motion.py`'s `__all__` (or leave module-private and
import directly in the faces). Note: `copy.deepcopy(ir)` preserves node
nids, so `block_nid`/`target_loop_nid` are valid in `sim`. The simulated
move is thrown away — only its dependency check matters.

- [ ] **Step 2: Edit both faces' `_check_legality` to call it**

In `compute_at.py` `_check_legality`: replace the tail (the
`_check_no_writer_under_accumulation` call from `51c2361` AND
`self._check_consumers_visited(ir, option)`) with:

```python
        _check_move_preserves_dependencies(
            ir, option.block_nid, option.target_loop_nid, option.index, is_reverse=False
        )
```

(add `from nkigym.transforms._code_motion import _move, _check_move_preserves_dependencies`).
Delete the methods `_check_consumers_visited`, `_root_sibling_of`, and
`_check_no_writer_under_accumulation`, plus now-unused imports (`AxisRole`,
`role_of`, `_enclosing_block`, `_loopvar_to_dim` — grep to confirm before
removing). Keep conditions 1–4 (target-in-graph, is-ForNode, block-in-graph,
target-not-descendant-of-block, output-block guard).

In `reverse_compute_at.py` `_check_legality`: replace
`self._check_producers_visited(ir, option)` with the same call,
`is_reverse=True`. Delete `_check_producers_visited` and `_root_sibling_of`.
Keep conditions 1–3.

CAVEAT for the executor: `_check_move_preserves_dependencies` runs `_move`
(which itself calls `place_buffers`/`compact_shapes`? — NO: `_move` is the
structural move only; the faces call place/compact AFTER `_move`. The
simulation only needs the structural tree, so `_move` alone is right). If
`_move` raises on a structurally-impossible move (e.g. partial coverage),
that surfaces as the legality error too — acceptable (an illegal move
shouldn't simulate). Confirm `analyze`'s `try/except TransformLegalityError`
still filters these.

- [ ] **Step 3: Re-express the reduction-init rejection test + add the consumer-before-producer rejection**

In `test/transforms/test_compute_at.py`, update the `51c2361` test
`test_compute_at_rejects_sinking_writer_under_accumulation_loop` — still
rejected, now via the move-sim legality. The error message changed to
"reorders dependency edge", so match `"reorder|dependency"`:

```python
def test_compute_at_rejects_sinking_writer_under_accumulation_loop():
    """Sinking the memset (accumulator init) under the matmul K loop is rejected
    by the dependency model (memset->K-loop carry edge would point backward),
    not an ad-hoc role guard."""
    ir = build_canonical_ir()
    memset = _block_for_op(ir, "NKIMemset")
    mm = _block_for_op(ir, "NKIMatmul")
    kloop = next(
        d for d in ir.tree.preorder(mm)
        if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d0_0"
    )
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=memset, target_loop_nid=kloop, index=0))
    assert not any(
        o.block_nid == memset and o.target_loop_nid == kloop for o in ComputeAt().analyze(ir)
    )


def test_compute_at_rejects_consumer_sunk_before_producer():
    """Hole #1: sinking the tensor_copy (consumer of psum_prod) under the memset's
    loop would place it before the matmul producer -> rejected by the same model."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    memset = _block_for_op(ir, "NKIMemset")
    memset_loop = next(
        d for d in ir.tree.preorder(memset) if isinstance(ir.tree.data(d), ForNode)
    )
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=tc, target_loop_nid=memset_loop, index=0))
```

(`_block_for_op` in this test file takes an op-name string — keep the existing signature.)

- [ ] **Step 4: Run the transform suite**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/ -q`
Expected: all pass. The 14 byte-exact ladder rungs MUST stay green (the legal moves must still be offered + apply). If `analyze` now offers FEWER options and a ladder rung's option vanished, that's an over-rejection bug — the legality is too strict; debug the carry-edge/span logic, do NOT loosen the ladder test.

> **Performance note:** `analyze` calls `_check_legality` per candidate, and
> `_check_move_preserves_dependencies` now deep-copies + `_move`s + rebuilds
> `Dependency` per candidate. `analyze` enumerates many candidates, so this is
> O(candidates × tree-size). For the matmul fixture this is fine (small tree,
> tens of candidates). If a later workload makes `analyze` slow, the fix is to
> short-circuit cheap structural rejections (conditions 1-4) BEFORE the
> simulation, and/or cache the per-candidate moved-graph — NOT to weaken the
> check. Note it; don't pre-optimize. (Learnings flag `analyze`-path
> regressions as a known sensitivity.)

- [ ] **Step 5: Full suite**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest -q`
Expected: 0 failed, 0 xfailed. (Count shifts: the `51c2361` test stays + the new consumer-before-producer test; net suite ≈216.)

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/compute_at.py nkigym/src/nkigym/transforms/reverse_compute_at.py nkigym/src/nkigym/transforms/_code_motion.py test/transforms/test_compute_at.py
git commit -m "ComputeAt/ReverseComputeAt legality via move-sim dependency check; delete bespoke 5a/5b + ad-hoc guard"
```

---

## Phase 6 — Regression + MDP

### Task 6: ladder byte-exact + MDP reduction-class clean

**Files:**
- Test: (no new files; verification + dump inspection)

- [ ] **Step 1: Byte-exact ladder + partial-coverage still green**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -m pytest test/transforms/test_compute_at.py test/transforms/test_reverse_compute_at.py -q`
Expected: all pass (7 code-motion rungs byte-exact, partial-coverage, all-14-states sim, PSUM-hoist assertion). This proves the move-sim legality did not over-reject any legal ladder move.

- [ ] **Step 2: kernel_transforms sim**

Run: `PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python kernel_transforms.py`
Expected: kernel_0..kernel_14 + kernel_partial all `pass=True`.

- [ ] **Step 3: MDP example — reduction class gone**

Run the example several times (unseeded random rollouts):
```bash
for i in 1 2 3 4 5; do PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -u examples/matmul_lhsT_rhs.py 2>&1 | grep -cE "memset.*psum|all.zero|nan|re-?init" || true; done
```
Expected: the memset-under-reduction-loop wrong-kernel class no longer appears. NOTE: the OTHER Task-13 holes (insertion-gap leaf-keying; the `_move` structural `multiple parents`) may still cause failures — that is expected and OUT OF SCOPE. Record which failure classes remain (capture the failing step's kernel + error) for the next task. Do NOT fix them here.

- [ ] **Step 4: Inspect the dependency dump shows the carry edge**

Run:
```bash
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src python -c "
from test.transforms._fixtures import build_canonical_ir
build_canonical_ir().dependency.dump('/home/ubuntu/cache/dep_check')
"
```
Confirm `/home/ubuntu/cache/dep_check/dependency.png` renders a `CARRY` edge from the memset leaf to the K-loop node (the edge the user asked to see). Read the PNG to verify.

- [ ] **Step 5: Commit any test/doc updates**

```bash
git add -A
git commit -m "Verify fine-grained dependency: ladder byte-exact, reduction-class MDP failures gone"
```

(If nothing changed in Step 4 beyond the cache dump, this commit may be empty — skip it.)

---

## Out of scope

- The other Task-13 legality holes (insertion-gap leaf-keying; `_move` structural double-parent) — separate fixes informed by this re-key.
- Per-`SBlockScope` graphs (deferred).
- `LoopPartition` + writers-as-list (its own spec; compatible with this re-key — the writers-list becomes per-leaf naturally).
