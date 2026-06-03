# Fine-Grained Dependency: ISANode↔ForNode edges + carried-state domination

## Status (2026-06-02)

NOT STARTED. Motivated by a real bug: `examples/matmul_lhsT_rhs.py`
produced a wrong kernel
(`/home/ubuntu/cache/matmul_lhsT_rhs/rollout_0/step_4/kernel.py`) where
`ComputeAt` sank the `psum_prod` memset **inside** the matmul's K
(accumulation) loop, re-zeroing the accumulator every K-step. A prior
ad-hoc `ComputeAt` guard (commit `51c2361`) papered over the symptom; this
spec replaces it with the correct dependency model. Supersedes the
reduction-domination locus question left open in the compute_at rewrite.

## The problem

The shipped `Dependency` (`nkigym/ir/dependency.py`) is a **global,
block-granular** DAG: nodes are leaf-block nids, edges are RAW/WAW/WAR
hazards between blocks (region-overlap gated). Two structural limits:

1. **It cannot express "init must dominate the whole accumulation loop."**
   The memset→matmul relationship is a block→block edge meaning "memset
   block before matmul block" — satisfied *per K-iteration* even when the
   memset sits **inside** K. The constraint that actually matters —
   "memset before the K loop *starts*" — has no representation, because
   the K loop is not a node in the graph.

2. **Blocks are no longer 1:1 with instructions.** After `ComputeAt`
   sinks a producer, the matmul block's subtree owns **two** ISA leaves
   (the matmul and the sunk memset, in a nested sub-block) — verified on
   `step_4`. The block-granular model survives only via a fragile
   nearest-enclosing-block leaf→owner mapping, and still cannot target a
   loop node.

The legality consequence: `ComputeAt._check_consumers_visited` accepts
sinking the memset under K because the matmul (consumer) *encloses* the
target loop — literally true at block granularity, semantically wrong for
the reduction.

## The model: one edge type, ISANode↔ForNode granularity

**Nodes** are tree nids of **ISANodes and ForNodes** (not BlockNodes —
blocks are organizational; the orderable atoms are instructions and
loops).

**One dependency-edge notion**, a producer→consumer hazard over an
overlapping `BufferRegion` (RAW/WAW/WAR), gated by the existing
`regions_disjoint`. What's new is that an **endpoint may be a loop**, not
only an instruction, in exactly one situation: **carried state across a
non-PARALLEL loop.**

### Carried-state domination (the new, fine-grained part)

A loop `L` carries state when its bound axis has a **non-PARALLEL** role
(`SEQUENTIAL` or `ACCUMULATION`) for an instruction `M` enclosed by `L`.
(`SEQUENTIAL` ⊇ `ACCUMULATION`: ACCUMULATION is the commutative-reduction
subcase of order-carried state. The enum has exactly three values, so the
predicate is `role != PARALLEL`. `Reorder` already uses the same
non-PARALLEL barrier — `reorder.py:83`.)

The **carried buffer** `B` is the operand of `M` whose region index
**omits** `L`'s axis — the value live across `L`'s iterations. (For
matmul, `dst=psum_prod (M,N)` omits `K`; for tensor_reduce, `dst (P)`
omits `F`. This is general — it does NOT require `RMW_OPERANDS`, which is
empty for the reduce ops.) Then:

- every **producer** of `B` (the init, e.g. memset): edge **`producer →
  L`** — must execute before `L` and stay **outside** it.
- every **consumer** of `B` (the drain, e.g. tensor_copy): edge **`L →
  consumer`** — must execute after `L` completes, **outside** it.
- `M` itself is exempt (it is the carry body; it lives inside `L`).

Inputs of `M` (e.g. `sbuf_lhs_T`, `sbuf_rhs`) are **not** `B`, so they get
no domination edge — they may sit inside `L` (prefetch) or be sunk under
PARALLEL loops freely. Verified: sinking `rhs_load` under the matmul's
PARALLEL M loop sims correct (`max_abs 1.3e-4`); sinking `memset` under
the ACCUMULATION K loop sims wrong (`max_abs 229`). The role of the loop,
not invariance of the index, is the discriminator — invariance over a
PARALLEL loop (rhs over M) is a redundant-recompute opportunity, not a
domination constraint.

### Worked example (canonical matmul, K split (2,2,4) as in step_4)

```
matmul nc_matmul enclosed by:  i_d0_0,i_d0_1,i_d0_2 (K, ACCUMULATION)
                               i_d1_0 (M, PARALLEL), i_d2_0 (N, PARALLEL)
  dst=psum_prod index = (M,N)  → omits the K axis → psum_prod is carried over K
  → memset (writes psum_prod)  gets edge  memset → i_d0_0  (outermost K loop)
  → tensor_copy (reads psum_prod) gets edge  i_d0_0 → tensor_copy
  → lhs_T/rhs loads (write sbuf_*, indexed by K) get NO K edge
```

When `L` is a contiguous outer run of carry loops (`i_d0_0/_1/_2` after a
Split), the edge endpoint is the **outermost** of that run (`i_d0_0`), so
a producer must dominate the entire K nest, not just the innermost K loop.

### Legality = one topology query (no edge-kind, no role logic)

There is **one** kind of dependency fact: an edge `a → b` meaning "a must
execute before b." Carry edges (`memset → K-loop`) and flow edges
(`matmul → tensor_copy`) are the **same** fact; the only difference is
whether an endpoint is a loop or a leaf. The `RAW`/`WAW`/`CARRY` label is
**provenance for the viz only — legality never reads it.**

Every node has a **preorder span** `[start, end]` over the tree: a leaf is
a point; a loop is its whole subtree (the K-loop spans all positions
inside it). An edge `a → b` is **satisfied** iff `a` finishes before `b`
begins — `span(a).end < span(b).start` — and **violated (backward)**
otherwise. An edge to a *loop* therefore means "must be entirely outside
and before that loop's span," which is exactly the init-dominates-the-
reduction constraint, falling out of the same comparison with no special
case.

`violates(moved_block, target_loop, index)` simulates the move on a copy,
recomputes preorder spans, and returns any incident edge of the moved
node that is now backward (or `None`). One span comparison covers
init-into-reduction-loop, consumer-before-producer, and any future
ordering hole. The transform asks this one query; it inspects no
`AxisRole`, `RMW_OPERANDS`, or edge kind.

## Architecture

### `Dependency` (`nkigym/ir/dependency.py`) — re-keyed + carry edges

- **Nodes**: ISANode nids and the ForNode nids that are carry-loop
  endpoints. Flow edges (RAW/WAW/WAR) connect ISANode nids; carry edges
  connect ISANode↔ForNode nids.
- `_build` walks ISA leaves in execution order (the existing
  `_nodes_in_execution_order`, but the unit is the **leaf**, not its
  owning block). `_BlockInfo` (regions/extents/buffers) is computed
  per-leaf, keyed by leaf nid. `touches_by_tensor` becomes leaf-keyed.
- Flow edges: the existing `_record_hazards` / `_try_edge` /
  `_provably_disjoint` logic, unchanged except keyed on leaf nids. Since
  blocks are 1:1 with leaves in canonical IR, the canonical flow graph is
  isomorphic to today's (regression-safe).
- **Carry edges (new)**: for each ISA leaf `M`, for each enclosing
  ForNode whose bound axis is non-PARALLEL for `M`'s block, identify the
  carried buffer `B` (operand whose region omits that axis); add
  `producer(B) → L_outer` and `L_outer → consumer(B)` edges, where
  `L_outer` is the outermost of the contiguous non-PARALLEL run. Producers
  / consumers of `B` are found from the same per-leaf region scan that
  feeds flow edges.

### Public API

Method names/signatures preserved; they now speak node nids:
- `producers(nid)`, `consumers(nid)`, `must_precede(a, b)` over the
  transitive closure of the unified (flow + carry) graph.
- **NEW** `violates(moved_leaf_nid, insertion_parent_nid) -> Edge | None`
  — the single legality query: returns the offending carry/flow edge if
  placing `moved_leaf_nid` under `insertion_parent_nid` would make it a
  descendant of a loop it must precede, or break post-domination of a loop
  it must follow; else `None`. The returned edge feeds a loud
  `TransformLegalityError` message.

### Transform consumers — net simplification

`compute_at.py` and `reverse_compute_at.py` currently hand-roll
`_check_consumers_visited` / `_check_producers_visited` / `_root_sibling_of`
over block producer/consumer sets, plus the ad-hoc ACCUMULATION guard
(`compute_at.py` from `51c2361`). All of that is **deleted** and replaced
by a single `ir.dependency.violates(moved_leaf, target_loop_or_index_parent)`
call. The faces translate their `block_nid` to its ISA-leaf nid for the
query. The directional condition-5 logic (5a consumers / 5b producers) is
subsumed: a flow edge `P → M` plus the descendant/post-dominance check
expresses "consumer stays after producer" in both directions uniformly.

The output-block guard (condition 4, ComputeAt-only) and the
target-not-ancestor structural checks (condition 3) stay in the faces —
they are topology rules, not dependency queries.

## Regression strategy

The risk is that re-keying + rewriting both faces' legality silently
changes the canonical graph or over-rejects a legal ladder move. Gated, in
order:

1. **Carry-edge units** (`test/ir/test_dependency.py`): on canonical
   matmul — assert `memset_leaf → K_loop` and `K_loop → drain_leaf`
   exist; `lhs_T_load` / `rhs_load` have NO edge to K; PARALLEL M/N loops
   create no carry edges. After a Split of K, the edge targets the
   outermost K sub-loop.
2. **Flow-graph isomorphism**: the canonical flow graph (re-keyed to leaf
   nids) is structurally identical to today's block graph
   (memset→matmul→copy→store). Update the existing `must_precede`
   assertions from block nids to leaf nids; semantics unchanged.
3. **Legality equivalence — the critical gate**: `violates()` REJECTS the
   `step_4` memset-under-K move and its split-K variants, AND still ALLOWS
   every legal ladder move. Re-run `build_ladder_state(1..14)`: all 7
   code-motion rungs still apply and stay **byte-exact** against the hand
   kernels; partial-coverage stays byte-exact; the strict byte oracle
   stays 240/240. Over-rejecting a legal rung is a HARD STOP (model too
   strict — fix the edge/query, do not loosen tests).
4. **Full regression**: `pytest -q` 0 failed/0 xfailed;
   `python kernel_transforms.py` all `pass=True`; then
   `examples/matmul_lhsT_rhs.py` runs clean across many unseeded rollouts
   **for the reduction class** (the memset-under-K family is gone). The
   other Task-13 holes (one-directional dep already subsumed by the
   bidirectional flow+carry edges; insertion-gap keyed on leaf; structural
   `multiple parents` in `_move`) are tracked separately — note which, if
   any, still surface.

## Files

```
nkigym/src/nkigym/ir/dependency.py   # EDIT — re-key to ISANode/ForNode nids;
                                     #   flow edges per-leaf; + carry-loop domination
                                     #   edges; + violates() query
nkigym/src/nkigym/transforms/compute_at.py          # EDIT — delete bespoke 5a/role guard;
                                                    #   legality = violates() query
nkigym/src/nkigym/transforms/reverse_compute_at.py  # EDIT — delete bespoke 5b; same query
test/ir/test_dependency.py           # EDIT — leaf-keyed must_precede; + carry-edge units
test/transforms/test_compute_at.py   # EDIT — the reduction-init rejection now via violates();
                                     #   the 51c2361 ad-hoc test folds into the model test
examples/matmul_lhsT_rhs.py          # (unchanged — already wires both transforms)
```

The reduction-domination viz falls out for free: `dependency.png` now
shows `memset → K-loop` (the edge the user drew), making the constraint
inspectable.

## Decisions captured

- **One edge notion, loop-or-instruction endpoint, kind-agnostic
  legality** — NOT carry-vs-flow edge types. An edge is just "a before b";
  legality is one span comparison (`span(a).end < span(b).start`) over the
  post-move tree, reading no edge kind. This subsumes reduction-init
  domination AND consumer-before-producer ordering in one rule. Rejected
  earlier framings that special-cased carry edges against enclosing loops
  (incomplete — missed the consumer-before-producer hole) or used two
  edge types (redundant).
- **Role-gated (non-PARALLEL), not reduction-gated** — `SEQUENTIAL` ⊇
  `ACCUMULATION`; gate on `role != PARALLEL` so general order-carried
  state (scans) is covered, consistent with `Reorder`'s existing barrier.
  Rejected keying on `RMW_OPERANDS` (empty for reduce ops; doesn't
  generalize).
- **Carried buffer = output operand whose index omits the carry axis** —
  general detection, no per-op flag.
- **Re-key the whole graph to ISANode/ForNode nids** (not "block graph +
  sidecar reduction facts") — because blocks already own multiple ISA
  leaves post-compute_at, so block granularity is structurally
  insufficient, and a loop must be a first-class node to be an endpoint.
- **One global graph, not per-scope** — re-keying to fine-grained nids +
  region overlap is sufficient under nesting; TVM's per-`SBlockScope`
  graph stays deferred (it buys scope-local reasoning we don't need here).
- **Replace, don't augment, the faces' legality** — `violates()` subsumes
  conditions 5a/5b and the ad-hoc ACCUMULATION guard; faces keep only the
  topology rules (3, 4).

## Out of scope

- Per-`SBlockScope` graphs (deferred, not required).
- The other Task-13 legality holes beyond what the unified flow+carry
  model subsumes (insertion-gap leaf-keying; the `_move` structural
  double-parent) — separate fixes, may be informed by this re-key.
- `LoopPartition` + writers-as-list (its own spec); this re-key is
  compatible with it (the writers-list becomes per-leaf naturally).
