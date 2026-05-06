# Topological Loop Fusion

*Date: 2026-05-06*
*Status: Draft for review*

## 1. Context and Goal

`FuseLoops` today (`nkigym/src/nkigym/tune/fuse_loops.py`) merges two
**literally-adjacent** sibling `LoopNode`s that share `dim_id`, same
`trip_count`, and both carry `AxisRole.PARALLEL`. The literal-adjacency
restriction makes the atom miss fusions that are semantically valid but
blocked by an unrelated sibling sitting between the producer and
consumer.

Concrete example from the current `rmsnorm+matmul` pool
(`/home/ubuntu/cache/rmsnorm_matmul_compile/kernel_tuned_0000.py`): at
the forest root the sibling list is

```
[0] lhs_load    (outer d0)
[1] rhs_load    (outer d1)     ← unrelated; reads rhs, writes sbuf_rhs
[2] rmsnorm     (outer d0)     ← consumes sbuf_lhs produced by [0]
[3] transpose   (outer d1)
[4] matmul      (outer d0)
[5] store       (outer d3)
```

`lhs_load` and `rmsnorm` share `dim_id="d0"` and the three-field rule.
`rhs_load` at position 1 is independent of both. The fuse `(0, 2)` is
semantically legal — `rhs_load` can slide left of `lhs_load` without
breaking any RAW/WAR/WAW edge — but today's atom rejects it because
`boundary=(0, 2)` has `j > i+1`.

This design:

1. Adds a persistent op-level dependency graph (`DepGraph`) on
   `OpGraph`, built once at `parse_and_resolve`. Edges are inferred
   from tensor identities via `OPERAND_AXES` (reads) and the bound
   output names (writes).
2. Generalises `FuseLoops` to accept any pair `(i, j)` with
   `0 ≤ i < j < len(siblings)` whose intervening siblings commute
   with both endpoints, enabling **topological** sibling-adjacency
   fusion.
3. Leaves `ReorderLoops`, the renderer, the canonical-forest builder,
   and the `KernelRewrite` / sampler contracts untouched.

### 1.1 Non-goals

- No multi-way fusion in a single atom (three or more nests merged at
  once). Iterated pairwise fuse reaches the same state.
- No loop fission, no sibling-permute-only atom, no hoist atom.
- No changes to `ReorderLoops`, `build_canonical_forest`, `render`, or
  the `KernelRewrite` protocol.
- No changes to the frontier sampler (`enumerate_pool` / `sample_pool`)
  beyond the single call-site signature update on
  `enumerate_fusion_atoms`.
- No new CPU-sim infrastructure — the existing tune-stage gate already
  catches correctness regressions.

## 2. Data Model

### 2.1 `DepGraph`

New module `nkigym/src/nkigym/codegen/dep_graph.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DepGraph:
    """Op-level dependency DAG over an OpGraph.

    Nodes are ParsedOp.idx values. Edges are RAW/WAR/WAW relations
    inferred from tensor identities.
    """

    producer: dict[str, int | None]
    """tensor_name -> op.idx that writes it, or None for params."""

    consumers: dict[str, tuple[int, ...]]
    """tensor_name -> tuple of op.idx values that read it, in source order."""

    reads: dict[int, frozenset[str]]
    """op.idx -> tensor names it reads (OPERAND_AXES slots only)."""

    writes: dict[int, frozenset[str]]
    """op.idx -> tensor names it writes (output_names)."""
```

**Construction.** Pure function `build_dep_graph(ops) -> DepGraph` in
the same module. One walk over `ops`:

- `writes[op.idx] = frozenset(op.output_names)`.
- `reads[op.idx]` = frozenset of `op.operand_names[slot]` for every
  `slot in op.op_cls.OPERAND_AXES` where `slot` is present in
  `op.operand_names`. Restricting to `OPERAND_AXES` excludes output
  slots that happen to be named kwargs.
- `producer[t]` = the unique `op.idx` whose `writes` contains `t`. For
  tensors never written by any op (parameter tensors), `producer[t]`
  is `None`. Tensor names declared as function parameters are
  pre-populated with `None` so every tensor in `op_graph.tensors`
  appears as a key.
- `consumers[t]` = tuple of `op.idx` whose `reads` contains `t`, in
  source order.

Raises `ValueError` if two ops write the same tensor name. This is a
defensive check — `parse_and_resolve`'s SSA invariant already forbids
it, so the check should never fire in practice.

### 2.2 Integration with `OpGraph`

```python
@dataclass
class OpGraph:
    ...
    dep: DepGraph = field(default_factory=lambda: DepGraph({}, {}, {}, {}))
```

`parse_and_resolve` calls `build_dep_graph(ops)` after
`_build_parsed_ops` and attaches the result. No other call site
changes. The default is an empty `DepGraph`, kept only so that raw
`OpGraph` instances used by existing unit tests (constructed by hand
without going through `parse_and_resolve`) continue to work — those
tests do not exercise the dep surface.

### 2.3 Subtree helpers

Same module (`dep_graph.py`):

```python
def subtree_ops(node: LoopNode | BodyLeaf) -> frozenset[int]:
    """Collect BodyLeaf.op_idx values under node."""

def subtree_reads(node, dep: DepGraph) -> frozenset[str]:
    """Union of dep.reads over subtree_ops(node)."""

def subtree_writes(node, dep: DepGraph) -> frozenset[str]:
    """Union of dep.writes over subtree_ops(node)."""

def commutes(a, b, dep: DepGraph) -> bool:
    """Subtrees a and b commute iff no RAW/WAR/WAW edge between them."""
    wa, ra = subtree_writes(a, dep), subtree_reads(a, dep)
    wb, rb = subtree_writes(b, dep), subtree_reads(b, dep)
    return not (wa & rb) and not (ra & wb) and not (wa & wb)
```

No caching in this milestone — forests are small (tens of nodes) and
legality is called `O(sibling_pairs)` per enumeration.

## 3. `FuseLoops` Generalisation

### 3.1 Identity

Shape unchanged; semantics widened:

```python
@dataclass(frozen=True)
class FuseLoops:
    path: tuple[int, ...]
    boundary: tuple[int, int]    # (i, j) with j > i; old case was j = i + 1
    dim_id: str
```

`j > i+1` is the new case. For `j = i+1` the enumerator emits
byte-identical atoms and apply behaves byte-identically — strict
generalisation, zero regression surface.

### 3.2 Legality (`is_legal`)

```
1. Resolve path → siblings list; fail if stale.
2. Three-field match on siblings[i], siblings[j]: same dim_id,
   same trip_count, both PARALLEL.
3. Topological-adjacency: for every k with i < k < j:
    3a. commutes(siblings[k], siblings[i], dep) — survivor passes producer
    3b. commutes(siblings[k], siblings[j], dep) — survivor passes consumer
```

Note on 3b. Under the assumption the input forest is a valid
linearisation of its dep DAG, 3b is implied by the sibling ordering
already being topologically correct. Keeping both checks makes the
legality predicate self-contained and robust to future rewrites that
might produce non-sorted sibling lists.

### 3.3 Apply

Fuse `(i, j)` reads as "consumer at `j` absorbs producer at `i`".
Fused nest lands at `j`'s slot; intervening siblings slide left of
`i`'s original position, preserving their relative order:

```python
def apply(self, op_graph, forest):
    def edit(siblings):
        i, j = self.boundary
        producer, consumer = siblings[i], siblings[j]
        survivors = [siblings[k] for k in range(i + 1, j)]
        fused = _merge_pair(producer, consumer, self.dim_id)
        return [*siblings[:i], *survivors, fused, *siblings[j + 1:]]
    new_forest = _rewrite_at_path(forest, self.path, edit)
    return op_graph, new_forest
```

`_merge_pair` (existing helper, lightly renamed from `_merge_pair` in
the current file) produces:

```python
LoopNode(
    dim_id=dim_id,
    trip_count=consumer.trip_count,
    role=AxisRole.PARALLEL,
    children=[*producer.children, *consumer.children],
    name=consumer.name,
)
```

Producer-body runs first inside the merged header (preserves the RAW
edge). The fused loop inherits the **consumer**'s `name` — matches the
"consumer absorbs producer" framing and keeps the identifier stable
from the reader's perspective.

### 3.4 Enumerator

`enumerate_fusion_atoms(op_graph, forest)` — note the new
`op_graph` parameter — walks every children list (forest root and
every `LoopNode.children`), iterates pairs `(i, j)` with
`0 ≤ i < j < len(siblings)`, and emits one atom per pair that passes
both the three-field match (§3.2.2) and the topological-adjacency
check (§3.2.3). Legality is checked inline during enumeration.

Complexity per parent: `O(n² · d)` where `n` is sibling count and `d`
is the subtree size. For nkigym-scale forests (`n < 10`, `d < 30`)
this is trivial.

## 4. Interactions

### 4.1 Sampler (`batch.py`)

One call-site change:

```python
# was
frontier[h0] = enumerate_fusion_atoms(forest) + enumerate_reorder_atoms(forest)
# now
frontier[h0] = enumerate_fusion_atoms(op_graph, forest) + enumerate_reorder_atoms(forest)
```

Same on the re-enumeration line after an apply.

### 4.2 `hash_forest` dedup

`hash_forest` is used at `batch.py:72` (`if h_new in pool`) for
**state dedup across the rewrite DAG**, not for cycle-breaking. The
sampler's frontier is already graph-traversal: each pool node drops
out of the frontier when its atom list is exhausted, and confluent
paths (two different `(source, atom)` pairs reaching the same
destination) are filtered by the hash check. The topological
generalisation does not introduce new confluence patterns beyond what
`ReorderLoops` already produced.

### 4.3 Renderer and canonical forest

Unchanged. The renderer walks whatever forest it is given; loop names
propagate through `_merge_pair` as before. Canonical-forest build
continues to produce literal-adjacent linearizations; the generalised
fuse is what extends the reachable states from those starting points.

### 4.4 `ReorderLoops`

Unchanged. Interactions with `FuseLoops` remain the same: reorder can
expose a pair that the fuse subsequently merges, or vice versa. The
frontier exploration handles this naturally.

## 5. Testing

### 5.1 Unit tests for `dep_graph.py`

New file `nkigym/tests/codegen/test_dep_graph.py`:

- `build_dep_graph` on a two-op chain (`load → activation`):
  `producer["<load-out>"] == 0`, `consumers["<load-out>"] == (1,)`.
- `build_dep_graph` on the rmsnorm+matmul fixture: spot-check a handful
  of edges (`lhs_load` is producer of `sbuf_lhs`; both
  `activation_reduce` and `tensor_scalar` consume it).
- Raises `ValueError` on duplicate `output_names`.
- `subtree_reads` / `subtree_writes` / `commutes` on hand-built forest
  fragments: disjoint-write pair commutes; RAW, WAR, WAW each reject.

### 5.2 Unit tests for `FuseLoops` generalisation

Extend `nkigym/tests/tune/test_fuse_loops.py`:

- **Baseline preservation.** Every existing test (literal-adjacent
  fuse) still passes unchanged.
- **Topological fuse on synthetic three-sibling forest.** Three
  PARALLEL loops `[A, B, C]` where `A` writes `tA`, `B` writes `tB`
  (reads nothing from `A`), `C` reads `tA`. Enumerator emits
  `FuseLoops(path=(), boundary=(0,2), dim_id=...)`. Apply produces
  `[B, fused(A‖C)]` — `B` pushed left, fused at `C`'s slot.
- **Legality rejection: RAW blocker.** Same shape but `B` reads `tA`.
  Atom `(0, 2)` not emitted.
- **Legality rejection: WAR blocker.** `A` reads `tB`; `C` reads `tA`;
  `B` between them writes `tB`. Atom `(0, 2)` not emitted.
- **End-to-end on rmsnorm+matmul fixture.** After generalisation, the
  enumerator emits the `lhs_load ↔ rmsnorm` atom at root. Apply
  produces the expected forest; `hash_forest` of the result matches a
  hand-derived target.
- **Stale-atom guard.** Atom constructed against forest F0 is applied
  after an intervening `ReorderLoops` that mutates the subtree under
  `path`; `is_legal` returns False.

### 5.3 Integration test via the batch sampler

Extend the existing rmsnorm+matmul end-to-end smoke test: seed,
fixed pool size, confirm at least one sampled kernel shows the
`lhs_load ↔ rmsnorm` fuse (grep for the absence of a second top-level
`for i_d0_0 in range(16):` in the rendered source).

### 5.4 CPU-sim correctness gate

New file `nkigym/tests/tune/test_fuse_loops_cpu_sim.py`:

- **rmsnorm+matmul, topological fuse applied explicitly.** Build
  `(op_graph, forest)` via `parse_and_resolve` + `build_canonical_forest`.
  Apply `FuseLoops(path=(), boundary=(0, 2), dim_id="d0")`
  (lhs_load ↔ rmsnorm). Render, CPU-sim, elementwise fp32 match vs
  numpy reference at `atol=rtol=1e-5` (K=2048).
- **matmul_lhsT_rhs, topological fuse on lhs_load ↔ matmul.** Simpler
  workload; confirms the fuse applies cleanly across shapes.
- **Chained atoms.** Apply a topological fuse, then a second fuse
  (now literal-adjacent after the first apply), CPU-sim the final
  kernel. Confirms composition through the frontier is
  correctness-preserving.
- **Negative control.** Hand-build a case where fuse is illegal (RAW
  edge via intervening sibling). Enumerator must not emit the atom;
  belt-and-suspenders, force-apply via raw construction and confirm
  CPU-sim output diverges from the numpy reference — validates the
  legality rule is doing real work.

All CPU-sim tests use the shared fp32 validation harness (cast both
sides to fp32 regardless of `input_specs` dtypes).

### 5.5 End-to-end run on `examples/rmsnorm_matmul.py`

After implementation, run the example through `nkigym_compile`'s full
pipeline (synthesis + initial_codegen + batch tune, 100 kernels,
`seed=0`) and confirm:

1. All sampled kernels pass CPU-sim — no regression from the
   generalised enumerator producing illegal states.
2. At least one sampled kernel shows the `lhs_load ↔ rmsnorm`
   topological fuse in its rendered source.
3. `results.json` shows MFU numbers within the expected range (sanity
   floor — no catastrophic regression vs the current 79–80% SOTA).
   Full MFU sweep is a separate perf exercise, not part of this
   design's acceptance.

This is the acceptance gate for the milestone.

## 6. Files Touched

New:

- `nkigym/src/nkigym/codegen/dep_graph.py`
- `nkigym/tests/codegen/test_dep_graph.py`
- `nkigym/tests/tune/test_fuse_loops_cpu_sim.py`

Modified:

- `nkigym/src/nkigym/codegen/graph.py` — `OpGraph` gains `dep: DepGraph`
  field; `parse_and_resolve` builds it.
- `nkigym/src/nkigym/tune/fuse_loops.py` — widened `is_legal`,
  rewritten `apply`, `enumerate_fusion_atoms` signature change.
- `nkigym/src/nkigym/tune/batch.py` — update both
  `enumerate_fusion_atoms` call sites to pass `op_graph`.
- `nkigym/tests/tune/test_fuse_loops.py` — new cases per §5.2.

Unchanged:

- `nkigym/src/nkigym/codegen/loop_forest.py`
- `nkigym/src/nkigym/codegen/render.py`
- `nkigym/src/nkigym/tune/reorder_loops.py`
- `nkigym/src/nkigym/tune/stage.py`
- `nkigym/src/nkigym/tune/__init__.py` (`KernelRewrite` protocol)
