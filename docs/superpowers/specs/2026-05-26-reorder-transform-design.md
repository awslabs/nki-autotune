# Reorder Transform — Adjacent-Pair Payload Swap

## Problem

`Split` and `Fuse` redistribute axis extents along the
`[outer_trips, tensorize_size]` chain but do not change the *order* in
which loops nest. Many MFU-relevant rewrites — hoisting an
ACCUMULATION dim outside a PARALLEL loop, choosing row-major vs
column-major traversal of two PARALLEL dims — require swapping the
nesting order of two loops. `Reorder` is the third transform in the
queue (`Split`, `Fuse`, `Reorder`, `compute_at`, `multi_buffer`,
`software_pipeline`).

`kernel_transforms.py` k5→k6 and k8→k9 are both Reorder applications:
k5→k6 lifts d1 (M) outside d0 (K) on the matmul nest; k8→k9 lifts d2
(N) outside d0 (K). Both must be expressible as a single transform
step from the prior IR.

## Goals

1. Define `Reorder(Transform)` and `ReorderOption(TransformOption)`
   matching the `Split` / `Fuse` interface (`analyze` /
   `apply` / `_check_legality`, deep-copy semantics, loud failures).
2. Express the action as the smallest atomic move: swap one
   adjacent parent-child `ForNode` pair. Long-range reorders are
   reached by composing adjacent swaps.
3. Reject reorders that change a `SEQUENTIAL`-role iteration's
   ordering. PARALLEL / ACCUMULATION combinations are all legal
   (associative-reducer commutativity).
4. Self-inverse: `apply(apply(ir, opt), opt)` returns IR structurally
   identical to the input.

Non-goals:

- TVM-style permutation-list reorder (any permutation of N≥2
  ForNodes on a chain in one call). Composition of adjacent swaps
  reaches every permutation; the smaller atom is a better fit for the
  MDP action space.
- Reorder across `compute_at`-sunk producer-consumer subtrees.
  `compute_at` lands first; cross-leaf legality is a future question.
- Cross-block reorder (siblings under the tree root). The env
  composes block placement separately.

## Mental Model

In our IR, two adjacent ForNodes `for A: for B: <body>` correspond to
a perfect-nest-of-two segment: `A` has exactly one child, `B`. The
body under `B` can have any structure (multiple ISA leaves, nested
ForNodes, the lot). Swapping `A` and `B` produces `for B: for A:
<body>` — same iteration set, possibly different traversal order.

Two mechanisms can implement this swap:

1. **Topology splice** (what `Split` and `Fuse` do): remove edges
   `parent→A`, `A→B`, `B→body`; create new ForNode nids `A'`, `B'`
   with swapped payloads; re-attach. Side effects: nids change,
   sibling-order under `parent` must be carefully re-spliced (the
   trap that hit outer-trip Split per memory).
2. **Payload swap**: mutate `graph.nodes[A_nid]["data"]` and
   `graph.nodes[B_nid]["data"]` to each other. Tree topology, child
   order, all nids untouched. The body stays attached to `B_nid`
   which is still in the same spot — only `B_nid` now carries `A`'s
   payload, so the renderer emits `A`'s loop header at that depth.

Reorder uses payload-swap. It works because the iteration set on
every descendant leaf is preserved and `axis_map` / `tensorize_sizes`
on each leaf are untouched — only the nesting order changes. Trivially
self-inverse.

## TVM Reference

TVM's `tir.schedule.Reorder` (`src/tir/schedule/primitive/loop_transformation.cc`)
permits a permutation of N≥2 loops on a single chain. Its legality
is checked by `BlockPropertyError::CheckBlockIterTypeAndAffineBinding`
(line 157):

```cpp
if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
  throw BlockPropertyError(...);
}
```

For every block transitively under the reorder range, every
`iter_var` must be `kDataPar` (data-parallel) or `kCommReduce`
(commutative reduction). `kOrdered` (sequential) and `kOpaque`
fail. Plus a single-branch chain check (`GetLoopsInReorderRange`,
line 1046) and a dependent-loop check (the outer's bound must not
reference an inner var, line 1106) — the latter is moot for our
constant trip counts.

Mapping to our `AxisRole`:

| TVM `IterVarType` | NKI `AxisRole` | Reorder legal? |
|---|---|---|
| `kDataPar` | `PARALLEL` | yes |
| `kCommReduce` | `ACCUMULATION` | yes |
| `kOrdered` / `kOpaque` | `SEQUENTIAL` | no |

The single-branch check translates to "perfect-nest between the
outermost and innermost permuted loop". For the adjacent-pair
flavor (N=2), this collapses to "inner is the sole child of outer".

## Interface

### `ReorderOption`

```python
@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Per-application payload for :class:`Reorder`.

    Attributes:
        outer_nid: nid of the parent ``ForNode`` to swap.
        inner_nid: nid of its sole ``ForNode`` child.
    """

    outer_nid: int
    inner_nid: int
```

No `target_axis` field. Both targets are ForNodes whose dim is
recoverable from `tree.data(nid).dim`. Reorder has no tensorize
flavor — `tensorize_sizes` are scalar per-axis maps, not ordered, so
"reorder tensorize entries" has no meaning. Single flavor only.

### `Reorder.analyze`

Walks `tree.preorder()`. For every nid:

- If `tree.data(nid)` is a `ForNode` and `tree.children(nid)` has
  exactly one entry whose data is also a `ForNode`, construct
  `ReorderOption(outer_nid=nid, inner_nid=child)`.
- Pass through `_is_legal` (try/except wrapper around
  `_check_legality`, mirrors `Fuse._is_legal`) and emit on legal.

The role check (descendant-leaf walk) only fires inside `_is_legal`,
so analyze visits every adjacent ForNode pair once and filters by
the same predicate `apply` re-checks.

### `Reorder.apply`

```python
def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
    """Re-check legality, deep-copy ``ir``, swap the two payloads, return."""
    self._check_legality(ir, option)
    new_ir = copy.deepcopy(ir)
    outer_data = new_ir.tree.data(option.outer_nid)
    inner_data = new_ir.tree.data(option.inner_nid)
    new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
    new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
    new_ir.dependency = Dependency(new_ir.tree)
    return new_ir
```

`Dependency` rebuild is unconditional — the leaf set and pre-order
are unchanged after a payload swap, so the rebuild is a no-op, but
keeping the rebuild uniform across Split/Fuse/Reorder simplifies the
contract.

### `Reorder._check_legality`

```python
def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
    """Raise :class:`TransformLegalityError` on any rule violation."""
    """1. Both nids exist."""
    for nid in (option.outer_nid, option.inner_nid):
        if nid not in ir.tree.graph:
            raise TransformLegalityError(
                f"Reorder.{'outer_nid' if nid == option.outer_nid else 'inner_nid'}"
                f"={nid} is not a node in the IR tree"
            )
    """2. Both are ForNodes."""
    outer = ir.tree.data(option.outer_nid)
    inner = ir.tree.data(option.inner_nid)
    if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
        raise TransformLegalityError(
            f"Reorder requires both targets to be ForNode; got "
            f"outer={type(outer).__name__}, inner={type(inner).__name__}"
        )
    """3. Inner is the sole child of outer (perfect-nest of two)."""
    kids = ir.tree.children(option.outer_nid)
    if kids != [option.inner_nid]:
        raise TransformLegalityError(
            f"Reorder requires inner_nid={option.inner_nid} to be the sole child of "
            f"outer_nid={option.outer_nid}; got children {kids}"
        )
    """4. No descendant ISA leaf has SEQUENTIAL on either swapped dim."""
    for leaf_nid in ir.tree.leaves(option.inner_nid):
        leaf = ir.tree.data(leaf_nid)
        if not isinstance(leaf, ISANode) or leaf.op_cls is NKIAlloc:
            continue
        leaf_dims = set(leaf.axis_map.values())
        for swap_dim in (outer.dim, inner.dim):
            if swap_dim in leaf_dims and role_of(leaf, swap_dim) == AxisRole.SEQUENTIAL:
                raise TransformLegalityError(
                    f"Reorder rejected: leaf {leaf.op_cls.__name__} has SEQUENTIAL role "
                    f"on dim {swap_dim!r}"
                )
```

Notes:

- `NKIAlloc` leaves are top-level (no enclosing loops), so they will
  never appear as descendants of `inner_nid`. The early-continue is
  a defensive belt against future refactors.
- `swap_dim in leaf_dims` is required because `role_of` raises
  `KeyError` on unmapped dims. The leaf may not touch every dim in
  the swap pair (e.g. a memset leaf under a matmul nest with no K
  axis).
- Same-dim swaps (`outer.dim == inner.dim`, common after Split) pass
  the legality check naturally — the role check fires once per
  dim, and the renderer's flat-index combination
  `i_d_0 * inner_trip + i_d_1` produces the same iteration set in
  row-major order regardless of which loop is outer (associative
  for ACCUMULATION, independent for PARALLEL).

## Layout

```
nkigym/src/nkigym/transforms/
├── __init__.py        # ADD Reorder, ReorderOption to re-exports
├── base.py
├── split.py
├── fuse.py
└── reorder.py         # NEW — Reorder, ReorderOption

test/transforms/
└── test_reorder.py    # NEW
```

## Tests

### `test/transforms/test_reorder.py`

1. **`test_reorder_analyze_canonical_matmul`** — build canonical IR
   for the 2048³ `lhs_T @ rhs`. Assert `Reorder().analyze(ir)`
   returns options for the matmul nest's K/M and M/N adjacent pairs.
   Assert single-ForNode subtrees (loads, stores, memset) yield no
   options.

2. **`test_reorder_apply_swaps_payloads`** — pick the matmul's
   outer (K) / middle (M) ForNode pair. Apply. Assert
   `tree.data(outer_nid).dim == 'd1'` and `tree.data(inner_nid).dim
   == 'd0'`; assert the parent's child list and inner's children
   are bitwise unchanged (same nids, same order).

3. **`test_reorder_self_inverse`** — apply the same option twice.
   Assert the resulting IR's `tree.graph.nodes[nid]["data"]` matches
   the input for every nid.

4. **`test_reorder_round_trip_render`** — start canonical, apply
   Reorder on the matmul K/M pair, render via the codegen pipeline,
   fp32-sim against the numpy golden. Pass at `atol=rtol=5e-3`.

5. **`test_reorder_render_swap_visible`** — same as above but assert
   the rendered kernel emits `for i_d1_0 in range(16):\n    for
   i_d0_0 in range(16):` where the canonical had K before M.

6. **`test_reorder_rejects_non_adjacent`** — manufacture an option
   referencing a ForNode pair that is not parent-child (e.g.
   K_outer + N_innermost). `apply` raises.

7. **`test_reorder_rejects_outer_with_siblings`** — manufacture an
   option where `outer_nid` has multiple children. Raises.

8. **`test_reorder_rejects_non_for_target`** — pass an `ISANode`'s
   nid as `inner_nid`. Raises.

9. **`test_reorder_rejects_sequential_role`** — construct a fixture
   IR enclosing a leaf whose `op_cls.AXIS_ROLES[axis] == SEQUENTIAL`
   under two ForNodes on the swap dims. Apply Reorder. Raises.
   Uses a synthetic `NKIOp` subclass with one SEQUENTIAL axis (no
   shipped op needs this today, but the rule must be tested before
   prefix-scan ops land).

10. **`test_reorder_allows_accumulation_parallel`** — confirm K/M
    (ACCUM/PARALLEL) and M/N (PARALLEL/PARALLEL) reorders pass
    legality. Negative regression against accidentally rejecting
    non-SEQUENTIAL combinations.

11. **`test_reorder_preserves_input_ir`** — apply any legal option;
    assert original `ir.tree` is structurally unchanged. Mirrors
    `test_split_apply_preserves_input_ir`.

### `test/environment/test_mdp.py`

Add **`test_mdp_with_reorder_random_rollout`** — instantiate
`KernelMDP(transforms=[Split(), Fuse(), Reorder()])`, run a random
rollout under a fixed seed, assert no `TransformLegalityError` and
every dumped kernel passes fp32 numerics.

### `examples/matmul_lhsT_rhs.py`

Update `transforms=[Split(), Fuse()]` → `transforms=[Split(),
Fuse(), Reorder()]`. Bump `NUM_ROLLOUTS=8, MAX_STEPS=6` for the
smoke run, then revert to canonical knobs once the smoke run is
clean.

## File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/transforms/reorder.py` | NEW — `Reorder`, `ReorderOption` |
| `nkigym/src/nkigym/transforms/__init__.py` | EDIT — re-export `Reorder`, `ReorderOption` |
| `test/transforms/test_reorder.py` | NEW — 11 tests above |
| `test/environment/test_mdp.py` | EDIT — add Reorder rollout test |
| `examples/matmul_lhsT_rhs.py` | EDIT — include `Reorder()` in transforms list |
| `docs/superpowers/specs/2026-05-26-reorder-transform-design.md` | NEW — this spec |

## Out of Scope

- TVM-style permutation-list reorder (any permutation of N≥2
  ForNodes on a chain). Composition of adjacent swaps reaches every
  permutation.
- Reorder across `SEQUENTIAL` boundaries with a more elaborate
  dependency model (out-of-iter-set guarantees on associative
  reducers in nested SEQ contexts).
- Reorder over `compute_at`-sunk producer-consumer subtrees.
  `compute_at` lands first; whether the existing per-leaf role
  legality is sufficient or needs an extra producer/consumer
  adjacency check is a future question.
- Cross-block reorder (siblings under the tree root). The env
  composes block placement separately.
