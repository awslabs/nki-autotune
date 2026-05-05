# Loop Reorder Rewrite

*Date: 2026-05-05*
*Status: Draft for review*

## 1. Context and Goal

`FuseLoops` (the first structural rewrite shipped in the prior
milestone) can only merge adjacent sibling `LoopNode`s that already
share a `dim_id`. Many fusion opportunities are unreachable from the
canonical forest because the relevant same-dim loops aren't adjacent —
they sit at different depths inside separate op trees, or behind a
different-dim ancestor.

This design adds `ReorderLoops`, the second structural rewrite:
**swap a `LoopNode` with its unique `LoopNode` child** (polyhedral
rectangular interchange). Composed with `FuseLoops`, it unlocks the
full loop-nest-rearrangement search space reachable without loop
distribution.

### 1.1 Design principle

> Rewrites stay atomic. If the transform is mathematically valid, it
> is legal — even when its performance consequences depend on the
> current kernel parameters. Resource costs (SBUF / PSUM footprint,
> degenerate `for _ in range(1):` wrappers) are the profiler's and
> the CPU-sim / HW-OOM gate's concern, not the legality check's.

This matches `FuseLoops`'s existing discipline: the atom emits
whenever the three-field legality rule holds, regardless of whether
the merged loop improves performance for any specific kernel.

### 1.2 Non-goals

- **No loop distribution.** Reorder requires a locally-perfect pair.
  Splitting an imperfect nest into multiple perfect nests is a separate
  future atom — `Hoist` / `Distribute`.
- **No same-dim resplit.** Changing the split factor of a single dim
  is `tiles_per_block`'s job, not reorder's.
- **No whole-chain permutation primitive.** Any permutation of an
  N-deep chain decomposes into at most N−1 adjacent swaps. One
  primitive keeps the IR transform surface minimal.
- **No cross-op or non-adjacent reorder.** Only parent→child swaps.
  Arbitrary code motion is hoist, not reorder.
- **No `op_graph` mutation.** Reorder is a pure `LoopForest`
  transform.

### 1.3 Decomposition argument for local-only

Any semantically valid "reorder through a non-perfect nest" decomposes
into:
1. **Un-hoist / distribution** — split the imperfect nest into
   multiple perfect nests.
2. **Local interchange** — an adjacent parent→child swap on one of the
   resulting perfect subnests.
3. **Optional re-hoist / re-fusion** — recompose.

Any "reorder" across a non-perfect boundary that *cannot* be
decomposed this way is a different transformation (standalone
distribution, accumulator rematerialization, init-motion) that needs
its own atom. Forcing those into `ReorderLoops` would bundle semantics
that belong in separate rewrites.

Concrete example — canonical matmul's `for N_tile: { psum_init, for
K_block(...compute...), drain }`: swapping N and K requires moving
`psum_init`/`drain` to a different depth (= hoist). `ReorderLoops`
correctly refuses the imperfect pair; the tuner reaches the same
final state via `Hoist(psum_init)` + `Hoist(drain)` + local reorder of
`(N_tile, K_block)` once those are perfectly nested.

## 2. Data Model Change

### 2.1 `LoopNode.reduce_op`

Extend `LoopNode` in `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
@dataclass
class LoopNode:
    dim_id: str
    trip_count: int
    role: AxisRole
    reduce_op: str | None = None   # NEW — "add" | "max" | None
    children: list[LoopNode | BodyLeaf] = field(default_factory=list)
```

Semantics:

- `role == PARALLEL` or `role == SEQUENTIAL` → `reduce_op is None`.
- `role == ACCUMULATION` → `reduce_op in {"add", "max", ...}`
  (whatever string the enclosing op uses in `op_kwargs["reduce_op"]`,
  normalised to lowercase).

The field lets the reorder legality check distinguish
associative-compatible ACC pairs (same reducer) from incompatible
ones (different reducer mixed in the same loop body) as a pure
field-read — symmetric to `FuseLoops`'s three-field rule.

### 2.2 Population at forest build time

In `build_canonical_forest` (via the existing `_LEAF_BUILDERS`
dispatch):

- `NKIMatmul.compute` K-chain: `reduce_op="add"` (nc_matmul's PSUM is
  summation; the reducer is hard-wired).
- `NKIActivationReduce` F-chain: `reduce_op = op.op_kwargs["reduce_op"]`
  (currently `"add"` for rmsnorm's Σx²; extends to `"max"` when
  top-k ops are added).
- All other `LoopNode`s: `reduce_op=None` (PAR dims, SEQ dims if any).

### 2.3 Preservation under existing rewrites

`FuseLoops` only merges PAR×PAR pairs, so the merged node has
`reduce_op=None` by construction. No change needed in
`nkigym/src/nkigym/tune/fuse_loops.py`.

`ReorderLoops.apply` preserves each node's `reduce_op` field verbatim
(the swap is a position change, not a role change).

### 2.4 `compute_phase_touched` unchanged

Phase-touched dims are a property of op identity, not loop position —
they survive any interchange.

## 3. Atom Signature

New module `nkigym/src/nkigym/tune/reorder_loops.py`:

```python
@dataclass(frozen=True)
class ReorderLoops:
    """Swap a LoopNode with its unique LoopNode child — local perfect-nest interchange.

    Attributes:
        path: Child indices from the forest root down to (and including)
            the outer LoopNode of the swap pair. A length-1 path
            ``(idx,)`` targets ``forest[idx]``; a length-2 path
            ``(idx, j)`` targets ``forest[idx].children[j]``; and so on.
            Empty tuple is invalid (``is_legal`` returns false).
        outer_dim: Dim id the outer loop iterates; guards against stale
            atoms after unrelated rewrites.
        inner_dim: Dim id the inner loop iterates; guards against stale
            atoms after unrelated rewrites.
    """

    path: tuple[int, ...]
    outer_dim: str
    inner_dim: str
```

**Identity** = `(path, outer_dim, inner_dim)`. The three fields let
the tune stage's explicit-list path detect stale bindings when an
earlier rewrite shifts indices or changes dim placement — same
discipline as `FuseLoops`'s `(path, boundary, dim_id)`.

**No direction parameter.** The swap is its own inverse, so a
`ReorderLoops` applied twice is structurally a no-op. The enumerator
emits each (outer, inner) ordered pair exactly once.

## 4. Legality

```python
def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
    _ = op_graph
    outer = _resolve_node(forest, self.path)
    if not isinstance(outer, LoopNode):
        return False
    if outer.dim_id != self.outer_dim:
        return False
    if len(outer.children) != 1:
        return False
    inner = outer.children[0]
    if not isinstance(inner, LoopNode):
        return False
    if inner.dim_id != self.inner_dim:
        return False
    return _roles_commute(outer, inner)


def _roles_commute(a: LoopNode, b: LoopNode) -> bool:
    """True when the two loops can swap without changing semantics."""
    if a.role == AxisRole.SEQUENTIAL or b.role == AxisRole.SEQUENTIAL:
        return False
    if a.role == AxisRole.PARALLEL and b.role == AxisRole.PARALLEL:
        return True
    if a.role == AxisRole.ACCUMULATION and b.role == AxisRole.ACCUMULATION:
        return a.reduce_op is not None and a.reduce_op == b.reduce_op
    """Mixed PAR / ACC: both orderings valid. Accumulator-footprint
    consequences are handled by the CPU-sim / HW-OOM gates downstream."""
    return True
```

**Role table.**

| Outer × Inner | Legal | Note |
|---|---|---|
| PAR × PAR | ✓ | Iterations commute. |
| PAR × ACC | ✓ | Increases accumulator footprint (all ACC accumulators live simultaneously); footprint is a profiler concern. |
| ACC × PAR | ✓ | Same reasoning, symmetric. |
| ACC × ACC, same `reduce_op` | ✓ | Reducer is associative + commutative. |
| ACC × ACC, different `reduce_op` | ✗ | Mixed reducers do not commute. |
| anything × SEQ, SEQ × anything | ✗ | Non-associative state. |

**Same-dim pairs are not excluded.** Once `tiles_per_block` produces
3-level splits `[block(B), group(T), tile(1)]` with `B, T > 1`,
same-dim swap becomes the classic polyhedral permutation with real
cache / SBUF reuse consequences. Allowing it today (where one side is
always trip=1 in the 2N canonical form, emitting a `for _ in range(1):`
wrapper) costs nothing — legality stays purely semantic, and future
`tiles_per_block` states are handled without code changes.

**No tree-global analysis.** Five field reads plus one helper call.
The `outer.children == [inner]` check is purely local to the swap
pair; the rest of the forest (post-fusion, post-hoist, partially
distributed, etc.) can be arbitrarily non-perfect — `ReorderLoops`
does not care.

### 4.1 Shared helper — `_resolve_node`

New utility in `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
def _resolve_node(forest: LoopForest, path: tuple[int, ...]) -> LoopNode | BodyLeaf | None:
    """Walk ``path`` from the forest root; return the node at that
    position, or ``None`` if the path is invalid (out of range, or
    traverses a ``BodyLeaf``)."""
```

`fuse_loops._resolve_siblings` (which returns the children list at a
depth) stays as-is — the two helpers have different return contracts
and callers.

## 5. Apply

```python
def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
    new_forest = _rewrite_forest_at(forest, self.path, _swap_pair)
    return op_graph, new_forest


def _swap_pair(outer: LoopNode) -> LoopNode:
    """outer → inner → grandchildren  becomes  inner → outer → grandchildren."""
    assert len(outer.children) == 1
    inner = outer.children[0]
    assert isinstance(inner, LoopNode)
    new_outer = LoopNode(
        dim_id=outer.dim_id,
        trip_count=outer.trip_count,
        role=outer.role,
        reduce_op=outer.reduce_op,
        children=inner.children,
    )
    return LoopNode(
        dim_id=inner.dim_id,
        trip_count=inner.trip_count,
        role=inner.role,
        reduce_op=inner.reduce_op,
        children=[new_outer],
    )
```

**Structural sharing.** Grandchildren sub-forest is passed through by
reference; only the two swapped nodes are freshly constructed. Same
discipline as `FuseLoops._rewrite_forest`.

**Slot expressions adapt automatically.** The renderer's `_slot_expr`
reads `path_ordinals` + `path_trips` at walk time — it does not care
which dim sits at which tree depth. Slice indices rewrite themselves
after the swap with zero renderer changes.

**Path-ordinal naming stays stable.** For a cross-dim swap, per-dim
ancestor ordering at every enclosed body leaf is unchanged →
`path_ordinals[d]` at every descendant is unchanged → emitted loop
variable names (`i_<d>_<k>`) are unchanged. A same-dim swap likewise
preserves per-dim ordering: the ancestor list on dim `d` stays the
same length, and because the walker numbers ancestors root-outward,
`path_ordinals[d]` at every descendant is unchanged too — the two
same-dim ancestors have simply had their trip counts exchanged. No
renderer changes in any case.

### 5.1 Shared helper — `_rewrite_forest_at`

Extract a small shared helper from `FuseLoops._rewrite_forest` in
`nkigym/src/nkigym/codegen/loop_forest.py`:

```python
def _rewrite_forest_at(
    forest: LoopForest,
    path: tuple[int, ...],
    transform: Callable[[LoopNode], LoopNode],
) -> LoopForest:
    """Return a new forest with ``transform`` applied to the node at
    ``path``. Ancestors are reconstructed along the path; everything
    outside the edit site is passed through by reference."""
```

Both `FuseLoops` and `ReorderLoops` call it with module-local
`transform` functions (FuseLoops operates on a siblings list at
`path + (boundary[0],)`; ReorderLoops operates on the node at `path`
itself — minor signature difference worth preserving as **two**
helpers if forcing them together gets awkward; design keeps the door
open either way).

## 6. Enumeration

```python
def enumerate_reorder_atoms(forest: LoopForest) -> list[ReorderLoops]:
    atoms: list[ReorderLoops] = []
    _collect_reorder(forest, path=(), atoms=atoms)
    return atoms


def _collect_reorder(
    siblings: list[LoopNode | BodyLeaf],
    path: tuple[int, ...],
    atoms: list[ReorderLoops],
) -> None:
    for idx, node in enumerate(siblings):
        if isinstance(node, LoopNode):
            if len(node.children) == 1 and isinstance(node.children[0], LoopNode):
                inner = node.children[0]
                if _roles_commute(node, inner):
                    atoms.append(ReorderLoops(
                        path=path + (idx,),
                        outer_dim=node.dim_id,
                        inner_dim=inner.dim_id,
                    ))
            _collect_reorder(node.children, path=path + (idx,), atoms=atoms)
```

**One atom per semantically-valid local-perfect pair.** No trip-count
filter, no same-dim exclusion — `is_legal` accepts everything the
enumerator emits, and vice versa.

**Re-enumeration exposes follow-on atoms.** In a 3-deep chain
`A→B→C`, apply `swap(A, B)` → chain `B→A→C` → re-enumerate sees
`swap(A, C)` at depth 1 and `swap(B, A)` (reverse) at depth 0. The
hash-based cycle check in §7 catches the reverse.

**Enabling fusion.** After rmsnorm+matmul's canonical forest, no PAR
`d1` loop from one op is adjacent to another's PAR `d1` (activation_
reduce's `d1` is ACC). Rotating tensor_scalar's chain
`d0(T=16)→d0(T=1)→d1(T=16)→d1(T=1)` via a swap of the two inner
nodes → `d0(T=16)→d1(T=16)→d0(T=1)→d1(T=1)` exposes new sibling
alignments one level deeper, which a follow-on `FuseLoops` can
absorb. Reorder-then-fuse is the main composition target.

## 7. `tune` Stage Integration

### 7.1 Changes

`nkigym/src/nkigym/tune/stage.py` — three-line change:

```python
from nkigym.codegen.loop_forest import hash_forest
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms
```

In the random-draw loop:

```python
if rewrites is None:
    rng = random.Random(seed)
    seen = {hash_forest(forest)}
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
else:
    for r in rewrites:
        if not r.is_legal(op_graph, forest):
            raise ValueError(f"{r!r} illegal on current state")
        op_graph, forest = r.apply(op_graph, forest)
```

### 7.2 Forest state hashing

New utilities in `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
def _canonical_key(node: LoopNode | BodyLeaf) -> tuple:
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

    Two forests with the same tree shape, dim_ids, trip counts, roles,
    reduce_ops, and leaf op_idx / phase tags hash equal. Used by the
    ``tune`` stage's random-draw loop to break cycles caused by
    self-inverse rewrites (reorder applied twice restores the prior
    state).
    """
    return hash(tuple(_canonical_key(e) for e in forest))
```

### 7.3 Termination properties

- **Bounded.** Each iteration advances to a new hash or terminates →
  at most |reachable states| iterations from any seed.
- **Deterministic.** Same seed → same forest-hash sequence → same
  stop point.
- **Covers both atom families.** Cycle detection is forest-structural,
  so any pair of inverse rewrites (reorder, future commute-like
  atoms) is caught automatically.

### 7.4 Explicit-list path unchanged

`ReorderLoops` implements the `KernelRewrite` protocol; `run_tune`
iterates the caller's list and dispatches through the protocol. No
code change needed beyond the import.

### 7.5 `op_graph` hashing — future

Current structural rewrites (`FuseLoops`, `ReorderLoops`) leave
`op_graph` untouched, so `hash_forest` alone is sufficient. Once
graph rewrites land (algebraic simplifications, fusion-across-op
boundaries), the hash must cover `op_graph` too. Inline TODO comment
in `hash_forest` flags this.

## 8. Testing

Three layers.

### 8.1 `test/codegen/test_loop_forest.py` (extend — data-model layer)

- `build_canonical_forest` populates `reduce_op` correctly on matmul
  K-chain (`"add"`), activation_reduce F-chain (reads from
  `op_kwargs`), PAR/SEQ loops (`None`).
- `hash_forest` is deterministic: same forest → same hash across
  invocations.
- `hash_forest` distinguishes structurally-different forests (swap
  two dim_ids at a LoopNode → different hash).
- `hash_forest` treats semantically-equal forests as equal (two
  independent `build_canonical_forest` calls on the same op_graph
  hash the same).

### 8.2 `test/tune/test_reorder_loops.py` (new — tree layer)

Mirrors `test/tune/test_fuse_loops.py`'s structure.

*`is_legal` positives:*
- PAR × PAR cross-dim, locally-perfect → true.
- PAR × PAR same-dim (constructed synthetic 3-level chain) → true.
- PAR × ACC, locally-perfect → true.
- ACC × PAR, locally-perfect → true.
- ACC × ACC same `reduce_op` → true.

*`is_legal` negatives:*
- Stale path (out of range, path through BodyLeaf) → false.
- `outer_dim` or `inner_dim` mismatches actual node → false.
- Outer has >1 child (imperfect nest) → false.
- Outer's unique child is a BodyLeaf → false.
- Any SEQUENTIAL involvement → false.
- ACC × ACC with different `reduce_op` → false.

*`apply` correctness:*
- Post-swap, outer and inner dims exchange. Grandchildren subtree
  reference-equal to pre-apply.
- `check_invariant` passes on swapped forest.
- Double-apply of the same atom returns a forest `hash_forest`-equal
  to the starting state (self-inverse).
- `reduce_op` field preserved across swap.

*Enumeration:*
- On canonical rmsnorm+matmul, `enumerate_reorder_atoms` returns an
  expected set — spot-check count + a handful of entries.
- Re-enumeration after an apply yields a changed atom set.

### 8.3 `test/tune/test_stage.py` (extend — end-to-end layer)

- `test_tune_reorder_only_cpu_sim` — explicit `[ReorderLoops(...)]`
  list that swaps two PAR loops inside `NKITensorScalar`'s chain on
  rmsnorm+matmul; `nki.simulate` result matches numpy golden.
- `test_tune_reorder_and_fuse_compose` — explicit list
  `[ReorderLoops(...), FuseLoops(...)]` where the reorder exposes a
  new fusion boundary; CPU-sim matches.
- `test_tune_random_reproducible_with_reorder` — same seed produces
  the same `kernel_tuned.py` byte-for-byte.
- `test_tune_random_terminates_on_cycle` — a targeted unit test on
  the random loop with a seeded RNG and monkey-patched enumerators
  that can construct a cycle, proving the `hash_forest` break fires.

### 8.4 `examples/rmsnorm_matmul.py`

Unchanged. The example runs `stages=["synthesis", "initial_codegen",
"tune"]` with `seed=0`; rerunning after this milestone will produce
a (potentially different) `kernel_tuned.py` because the atom pool
now includes reorder. Flag in the commit message so cached
`kernel_tuned.py` expectations are re-seeded.

## 9. Migration Phases

Each phase independently mergeable; tests gate each phase.

**Phase A — `LoopNode.reduce_op` + population.**

- Add `reduce_op: str | None = None` to `LoopNode`.
- Populate in matmul and activation_reduce leaf builders inside
  `build_canonical_forest`.
- Renderer and `FuseLoops` unchanged.
- Gate: Layer 1 tests (population correctness) pass; every existing
  test still passes.

**Phase B — Forest hashing utilities.**

- `_canonical_key` + `hash_forest` in `loop_forest.py`.
- Layer 1 tests (determinism, distinguishing, equality) pass.

**Phase C — `ReorderLoops` atom.**

- `reorder_loops.py` with `ReorderLoops`, `_roles_commute`,
  `enumerate_reorder_atoms`.
- `_resolve_node` helper in `loop_forest.py`.
- Optional shared `_rewrite_forest_at` refactor — either commit
  alongside Phase C or defer as a cleanup after both rewrites land.
- Gate: Layer 2 tests pass.

**Phase D — `tune` stage integration.**

- Extend `run_tune`'s random path with the reorder enumerator and
  `hash_forest` cycle break.
- Gate: Layer 3 tests pass; `examples/rmsnorm_matmul.py` runs
  end-to-end with CPU-sim passing.

**Phase E — Cache / snapshot reseed.**

- Any cached `kernel_tuned.py` with `seed=0` may differ after this
  milestone. Non-blocking; documented in the commit message.

## 10. Risks and Mitigations

- **Hash collision on structurally-different forests.** `hash()` on
  a tuple of primitives uses Python's tuple hashing — standard
  collision risk is negligible in practice but theoretically
  non-zero. Mitigation: the cycle detection is a *termination*
  guard, not a *correctness* guard; a false positive just stops
  tuning one iteration early. Correctness is gated by CPU-sim at
  every stop.
- **Explicit-list staleness after reorder.** Reorder preserves
  path indices for everything outside the swap site but renames
  what's under the swapped pair. A caller-supplied list specified
  against the canonical forest has paths that become stale after
  the first apply — same situation as `FuseLoops`. Mitigated the
  same way: `is_legal` is checked immediately before each `apply`
  in the explicit path.
- **ACC × ACC swap reaches resource-bounded configurations.** Not a
  legality concern (by design — §1.1). Caught by CPU-sim's HW-OOM
  gate on the tuned kernel.
- **`_rewrite_forest_at` refactor scope creep.** If the shared helper
  turns out to be awkward (FuseLoops edits a children list, Reorder
  edits a single node), keep them as two module-local helpers. The
  design does not hinge on the refactor.

## 11. File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/codegen/loop_forest.py` | EDIT — add `LoopNode.reduce_op`; populate in matmul/activation_reduce leaf builders; add `_resolve_node`, `_canonical_key`, `hash_forest` utilities |
| `nkigym/src/nkigym/tune/reorder_loops.py` | NEW — `ReorderLoops` atom + `_roles_commute` + `enumerate_reorder_atoms` |
| `nkigym/src/nkigym/tune/stage.py` | EDIT — add reorder enumerator to random draw; add `hash_forest` cycle break |
| `test/codegen/test_loop_forest.py` | EDIT — add reduce_op population + `hash_forest` tests |
| `test/tune/test_reorder_loops.py` | NEW — Layer 2 tree tests for `ReorderLoops` |
| `test/tune/test_stage.py` | EDIT — add reorder-integration + compose-with-fuse + cycle-termination tests |
| `docs/superpowers/specs/2026-05-05-loop-order-design.md` | NEW — this spec |
