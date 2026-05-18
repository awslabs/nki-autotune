# Loop-Role Placement Refactor

## Problem

`ForNode` in `nkigym/ir/tree.py` carries `loop_type: AxisRole`,
populated at canonical-build time from the per-op
`op_cls.AXIS_ROLES[abstract_axis]`. Today this is consumed by:

- `Fuse._check_outer_trip` and `_check_tensorize` — chain
  role-equality + `SEQUENTIAL` rejection.
- `Fuse.analyze` — same role checks during chain enumeration.
- `Split._do_apply_outer_trip` — copy parent's `loop_type` onto
  the new chain.
- `Split._do_apply_tensorize` — read `op_cls.AXIS_ROLES` to set the
  new ForNodes' role.
- `tree_visualize` — render the role in the diagram label.

The field is a snapshot of "what role does this dim play for the
unique leaf currently in this subtree?" That invariant only holds
because canonical IR has one ISA leaf per op subtree. Future
`compute_at` sinks producer leaves inside consumer loops, so one
ForNode can enclose multiple leaves where its dim plays different
roles per leaf — `loop_type` then becomes a single answer to a
question with many answers.

## Goals

1. Move loop role from `ForNode` to its authoritative source: each
   enclosed ISA leaf, via `op_cls.AXIS_ROLES + ISANode.axis_map`.
2. Make `ForNode` role-neutral, matching TVM TIR's `For`.
3. Provide a `role_of(leaf, concrete_dim)` helper as the single
   primitive future role-sensitive transforms call.
4. Strip role logic from `Split` and `Fuse` — neither transform's
   correctness depends on role information (see §"Role-blindness
   of Split and Fuse").

Non-goals:

- `Reorder` transform itself (separate PR — descendant-walk legality).
- `compute_at` (separate PR).
- Sidecar / cached role tracking on `ForNode` (rejected — cache
  invalidation under transforms is the failure mode this refactor
  removes).
- Cross-dim Fuse via divmod ("synthetic axis") — still out of scope,
  same as the Split/Fuse spec.

## Role-Blindness of Split and Fuse

Both transforms preserve the iteration set of every enclosed leaf
and the per-leaf `(abstract_axis → concrete_dim)` binding. Role is
a class attribute on `op_cls`, not a property of the loop:

- **Split outer-trip**: replaces `ForNode(d, T)` with a chain whose
  trips multiply to `T`, all on dim `d`. Iteration set unchanged.
- **Split tensorize**: inserts ForNodes above the leaf and decreases
  `tensorize_sizes[axis]`. The leaf's role for that axis is fixed
  by `op_cls.AXIS_ROLES`, independent of tile size.
- **Fuse outer-trip**: collapses a same-dim ForNode chain into one
  ForNode of the product trip. Iteration set unchanged.
- **Fuse tensorize**: bumps `tensorize_sizes[axis]` and removes
  loops. Same reasoning.

TVM's TIR `Split` and `Fuse` (`tir/schedule/primitive/loop_transformation.cc`)
have no role/iter_type checks for the same reason. The existing
`loop_type`-equality and `SEQUENTIAL` rejections in our `Fuse` are
defensive over-rejection that this refactor removes.

The only transforms that genuinely need role information are those
that change the relative iteration order across leaves — `Reorder`
(swaps two loops, may swap a SEQ leaf's iteration order) and
`compute_at` (sinks a producer past a non-commuting role boundary).

## Data Model Change

`ForNode` in `nkigym/ir/tree.py`:

```python
@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Trip-loop payload.

    Attributes:
        dim: Concrete dim id (e.g. ``"d0"``).
        trip: Loop trip count (``extent // tile_size``).
    """

    dim: str
    trip: int
```

`ISANode`, `RootNode`, `KernelTree` unchanged. `op_cls.AXIS_ROLES`
and `ISANode.axis_map` already encode every fact a future role
check needs.

## Role Lookup Helper

New helper in `nkigym/ir/tree.py`:

```python
def role_of(leaf: ISANode, concrete_dim: str) -> AxisRole:
    """Return the role this leaf assigns to ``concrete_dim``.

    Walks ``leaf.axis_map`` to find the abstract axis that maps to
    ``concrete_dim``, then consults
    ``leaf.op_cls.AXIS_ROLES`` (defaulting to
    ``AxisRole.PARALLEL`` for axes not listed). Raises ``KeyError``
    if the leaf does not touch ``concrete_dim``.
    """
    for abstract, dim_id in leaf.axis_map.items():
        if dim_id == concrete_dim:
            return leaf.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
    raise KeyError(f"{leaf.op_cls.__name__} has no axis mapping {concrete_dim}")
```

Pure function over a leaf payload. The descendant walk happens in
the *caller* (future `Reorder` / `compute_at` legality checks);
`role_of` answers the per-leaf role question. Loud failure on
unmapped dim — matches the project's "no silent raises" rule.

`NKIAlloc` leaves do not declare iteration roles, but they also
never enclose loops, so `role_of` is never called on them.

## Canonical Builder Change

`_attach_op_subtree` in `nkigym/ir/tree.py`:

```diff
-role = op.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
-parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile, loop_type=role), parent=parent)
+parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile), parent=parent)
```

`AxisRole` import in `tree.py` stays — still referenced by `role_of`'s
default and by the `ISANode` payload semantics.

## Split Change

`nkigym/src/nkigym/transforms/split.py`:

`_do_apply_outer_trip` — drop the `loop_type=target.loop_type` kwarg:

```diff
-new_nid = ir.tree.add_node(ForNode(dim=target.dim, trip=trip, loop_type=target.loop_type), parent=None)
+new_nid = ir.tree.add_node(ForNode(dim=target.dim, trip=trip), parent=None)
```

`_do_apply_tensorize` — drop the `op_cls.AXIS_ROLES` lookup and the
`loop_type=` kwarg:

```diff
-loop_type = leaf.op_cls.AXIS_ROLES.get(option.target_axis, AxisRole.PARALLEL)
 ...
-new_nid = ir.tree.add_node(ForNode(dim=concrete_dim, trip=trip, loop_type=loop_type), parent=prev)
+new_nid = ir.tree.add_node(ForNode(dim=concrete_dim, trip=trip), parent=prev)
```

`AxisRole` import removed if no other reference remains.
`_check_legality` and `analyze` are unchanged — they never read
`loop_type`.

## Fuse Change

`nkigym/src/nkigym/transforms/fuse.py` — all deletions.

`_check_outer_trip` — drop the `SEQUENTIAL` rejection and the chain
`loop_type`-equality check:

```diff
-first = nodes[0]
-if first.loop_type == AxisRole.SEQUENTIAL:
-    raise TransformLegalityError("Fuse outer-trip flavor: SEQUENTIAL loops cannot be fused")
-for n in nodes[1:]:
-    if n.dim != first.dim:
-        raise TransformLegalityError(...)
-    if n.loop_type != first.loop_type:
-        raise TransformLegalityError(...)
+first = nodes[0]
+for n in nodes[1:]:
+    if n.dim != first.dim:
+        raise TransformLegalityError(...)
```

`_check_tensorize` — drop the `expected_role` computation, the
`SEQUENTIAL` rejection, and the prefix `loop_type` check (keep the
`dim` equality check):

```diff
-expected_role = leaf.op_cls.AXIS_ROLES.get(axis, AxisRole.PARALLEL)
-if expected_role == AxisRole.SEQUENTIAL:
-    raise TransformLegalityError(...)
 ...
-if data.loop_type != expected_role:
-    raise TransformLegalityError(...)
```

`analyze` — drop role checks in both walks:

```diff
-if kid_data.dim != data.dim or kid_data.loop_type != data.loop_type:
+if kid_data.dim != data.dim:
     break
-if data.loop_type == AxisRole.SEQUENTIAL:
-    break
 ...
-expected_role = data.op_cls.AXIS_ROLES.get(axis, AxisRole.PARALLEL)
-if expected_role == AxisRole.SEQUENTIAL:
-    continue
 ...
-if wdata.dim != dim or wdata.loop_type != expected_role:
+if wdata.dim != dim:
     break
```

`_do_apply_outer_trip` — drop the `loop_type=` kwarg:

```diff
-new_nid = ir.tree.add_node(ForNode(dim=first.dim, trip=new_trip, loop_type=first.loop_type), parent=None)
+new_nid = ir.tree.add_node(ForNode(dim=first.dim, trip=new_trip), parent=None)
```

`AxisRole` import removed if no other reference remains.

After these deletions, Fuse's legality rule is:

1. `len(target_nids) >= 2`.
2. Every entry exists in the tree.
3. Same-dim across the chain (renderer constraint).
4. Single-child invariant — `children(nid_i) == [nid_{i+1}]` for
   each consecutive pair.
5. (Tensorize flavor) `new_tensorize <= MAX_TILE_SIZE[axis]`.

Same as TVM modulo our same-dim and `MAX_TILE_SIZE` constraints,
which are orthogonal additions justified by our renderer / HW
model.

## Visualization Change

`nkigym/src/nkigym/ir/tree_visualize.py` line 52 — drop the role
suffix from the loop-node label:

```diff
-return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}<br/>{data.loop_type.name}"]', "loop")
+return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}"]', "loop")
```

## Tests

### `test/codegen/test_body.py`

Lines 193–194 and 303 construct `ForNode(..., loop_type=AxisRole.PARALLEL)`.
Drop the kwarg. Drop the `AxisRole` import on line 18 if unused
afterwards.

### `test/transforms/test_split.py`

Line 192 asserts `new_parent_data.loop_type == AxisRole.PARALLEL`
post-tensorize-Split. Delete that assertion — the field no longer
exists. Surrounding test still verifies `dim` and `trip`, which is
what tensorize-Split actually decides.

### `test/transforms/test_fuse.py`

No direct `loop_type` references and no SEQ-rejection / role-mismatch
tests today (verified). Every existing test still passes after the
refactor — `test_fuse_rejects_dim_mismatch` keeps working (dim
check stays); `test_fuse_analyze_no_tensorize_options_on_canonical`
still passes (the actual filter on canonical is `chain_trip_product < 2`,
not role).

One docstring update — `test_fuse_analyze_no_tensorize_options_on_canonical`'s
docstring currently cites "matches its op axis role" as the filter
reason, which is doubly stale (the role filter is gone after the
refactor; even before, the actual filter was `chain_trip_product < 2`).
Rewrite the docstring to reflect the real reason.

### End-to-end smoke — `examples/matmul_lhsT_rhs.py`

After the refactor, run `examples/matmul_lhsT_rhs.py` with elevated
rollout count (`NUM_ROLLOUTS = 32`, `MAX_STEPS = 8`) and confirm
every dumped `kernel.py` passes the numerics check (`atol=rtol=5e-3`
fp32-sim against numpy golden). Reverts the rollout knobs back to
their committed values once the smoke run is clean.

The MDP environment exercises `Split` and `Fuse` over the canonical
matmul IR; if the role-field removal accidentally changes either
transform's semantics, a random rollout will surface it as either a
`TransformLegalityError` mismatch or a numerical divergence. Higher
rollout count + step depth widens coverage over the
loop-structure-rewrite space without code changes.

### `test/ir/test_role_of.py` (new)

Unit tests for `role_of(leaf, concrete_dim)`:

1. `NKILoad` leaf — `role_of(load, P_dim) == PARALLEL`,
   `role_of(load, F_dim) == PARALLEL`.
2. `NKIMatmul` leaf — `role_of(mm, K_dim) == ACCUMULATION`,
   `role_of(mm, M_dim) == PARALLEL`,
   `role_of(mm, N_dim) == PARALLEL`.
3. `NKIActivationReduce` leaf —
   `role_of(act_red, F_dim) == ACCUMULATION`,
   `role_of(act_red, P_dim) == PARALLEL`.
4. Unmapped dim — `role_of(leaf, "no_such_dim")` raises `KeyError`.

Build leaves via `build_canonical_ir(...)` from the existing
fixtures; pull the leaf payloads out by op-class lookup.

## File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/ir/tree.py` | EDIT — drop `loop_type` from `ForNode`; drop role lookup in `_attach_op_subtree`; add `role_of(leaf, dim)` helper |
| `nkigym/src/nkigym/ir/tree_visualize.py` | EDIT — drop `data.loop_type.name` from the loop-node label |
| `nkigym/src/nkigym/transforms/split.py` | EDIT — drop the two `loop_type=` kwargs and the `op_cls.AXIS_ROLES.get(...)` lookup; drop unused `AxisRole` import |
| `nkigym/src/nkigym/transforms/fuse.py` | EDIT — drop role-equality + SEQ-rejection from `analyze`, `_check_outer_trip`, `_check_tensorize`, `_do_apply_outer_trip`; drop unused `AxisRole` import |
| `test/codegen/test_body.py` | EDIT — drop `loop_type=` kwargs; drop `AxisRole` import if unused |
| `test/transforms/test_split.py` | EDIT — drop the `loop_type` assertion |
| `test/transforms/test_fuse.py` | EDIT — update the stale docstring on `test_fuse_analyze_no_tensorize_options_on_canonical` |
| `test/ir/test_role_of.py` | NEW — unit tests for `role_of(leaf, dim)` |
| `docs/superpowers/specs/2026-05-18-loop-role-placement-refactor-design.md` | NEW — this spec |
