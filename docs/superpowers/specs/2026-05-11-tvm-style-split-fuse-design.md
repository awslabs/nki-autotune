# TVM-Style Split + Fuse with Explicit Innermost Tile Loop

## Problem

Today's IR welds two orthogonal decisions into `NKIOp.TILE_LIMITS`:

1. **Loop structure** — how many `For` nodes exist and their extents (scheduling).
2. **Intrinsic tiling** — what slice width the ISA call consumes (engine/microkernel selection).

The renderer absorbs `TILE_LIMITS` axes into ISA slice widths implicitly. This:

- Diverges from TVM: `sch.split` always emits N `For`s for N factors; TVM never implicitly drops a loop. Collapse is a separate `Tensorize` primitive.
- Confuses `split` semantics: factors become both outer python `for` trips *and* implicit slice widths, depending on whether they match `TILE_LIMITS`.
- Makes the legality check per-atom ad hoc — each `split` asks "is inner ≤ TILE_LIMITS?" rather than checking a structural invariant.
- Blocks multi-intrinsic support (e.g. `nc_transpose` on TE vs VE engines) because the op carries one tile dict.

## Target Model

Adopt TVM's invariant: **loops are the only structural concept; the innermost loop on each op axis is the tile**. `product(all_loop_extents_on_axis) == axis_extent` always.

- Each axis of each op has an **innermost loop** whose extent is bounded by per-op `[MIN_TILE_SIZE, MAX_TILE_SIZE]`.
- Schedule atoms (`Split`, `Fuse`) preserve total extent. They cannot change the `axis_extent`; they redistribute it across the chain.
- Renderer emits every loop as a Python `for`, **except** the innermost tile loop per op axis, which becomes the slice width on the ISA call. This is the only "magic" step, and it's local to the body leaf.

Multi-intrinsic selection is deferred (low priority per user): a single fixed intrinsic per op in v1, `[MIN, MAX]` carried as two dicts. Multi-intrinsic adds a `BindIntrinsic` atom later without IR restructuring.

## IR Changes

**`LoopNode`:** unchanged. Still `(trip, name, body, tensor_name?)` etc.

**`NKIOp` class attrs:** replace `TILE_LIMITS: dict[str, int]` with:

```python
MIN_TILE_SIZE: ClassVar[dict[str, int]]
MAX_TILE_SIZE: ClassVar[dict[str, int | None]]  # None = unbounded
```

Every axis of every op has both entries. Axes not present in either dict are non-tile axes (do not get renderer absorption); for v1 all ops' axes are tile axes with MIN=128.

## Per-Op Table

| Op                   | P           | K           | M           | N           | F             |
|----------------------|-------------|-------------|-------------|-------------|---------------|
| NKIMatmul            | —           | (128, 128)  | (128, 128)  | (128, 512)  | —             |
| NKITranspose         | (128, 128)  | —           | —           | —           | (128, 128)    |
| NKIDMATranspose      | (128, 128)  | —           | —           | —           | (128, None)   |
| NKILoad              | (128, 128)  | —           | —           | —           | (128, None)   |
| NKIStore             | (128, 128)  | —           | —           | —           | (128, None)   |
| NKIMemset            | (128, 128)  | —           | —           | —           | (128, None)   |
| NKITensorCopy        | (128, 128)  | —           | —           | —           | (128, None)   |
| NKITensorReduce      | (128, 128)  | —           | —           | —           | (128, None)   |
| NKIActivation        | (128, 128)  | —           | —           | —           | (128, None)   |
| NKIActivationReduce  | (128, 128)  | —           | —           | —           | (128, None)   |
| NKITensorScalar      | (128, 128)  | —           | —           | —           | (128, None)   |
| NKIAlloc (SBUF)      | (128, 128)  | —           | —           | —           | —             |
| NKIAlloc (PSUM)      | (128, 128)  | —           | —           | —           | —             |
| NKIAlloc (HBM)       | —           | —           | —           | —           | —             |

`MAX=None` means unbounded; canonical build and `Split`/`Fuse` treat it as no upper check.

## Canonical Build

For each op axis with bounds `(MIN, MAX)`:

```python
if MAX is None:
    outer_trip, inner_trip = 1, axis_extent
else:
    assert axis_extent % MAX == 0, f"{op}.{axis}: extent {axis_extent} not divisible by MAX {MAX}"
    outer_trip, inner_trip = axis_extent // MAX, MAX
```

Emitted as nested `for outer in range(outer_trip): for inner in range(inner_trip): leaf(...)`. If `outer_trip == 1`, per canonical-1N convention the trip-1 outer is still emitted explicitly (legibility and loop-identity stability).

Examples with axis extent 2048:
- dma_copy F (MAX=None): `outer=1, inner=2048`.
- matmul N (MAX=512): `outer=4, inner=512`.
- matmul M (MAX=128): `outer=16, inner=128`.

## Atoms

1:1 with TVM MetaSchedule primitives.

### `Split(loop, factors)`

Replaces `loop` with `len(factors)` nested loops whose trips multiply to `loop.extent`.

**Legality:**
- `product(factors) == loop.extent`.
- For every op leaf whose tile-axis innermost loop is `loop` (before split): the new innermost extent `factors[-1]` must satisfy `MIN ≤ factors[-1] ≤ MAX` (None MAX = upper check skipped).
- For every op leaf whose tile-axis innermost is **not** `loop` (i.e. a descendant): no constraint — the innermost is unaffected.

TVM replay: `sch.split(loop, factors=list(factors))`.

### `Fuse(adjacent_loops)`

Fuses two or more adjacent loops on the **same axis** into one. Product preserved.

**Legality:**
- All input loops are adjacent ancestors in the tree (immediate parent/child chain) and on the same axis.
- For every op leaf affected by the fuse, compute the leaf's new innermost tile extent along that axis and check `MIN ≤ new_innermost ≤ MAX`. Two cases:
  - All fused loops sit strictly above the leaf's pre-fuse innermost: the leaf's innermost is unchanged; no check needed.
  - The fuse includes the leaf's pre-fuse innermost: the new innermost is the fused product; check bounds against it.

TVM replay: `sch.fuse(*loops)`.

### `Reorder(loops)`

Unchanged. Permutes the listed loops within their common scope.

**Legality:** unchanged from current. Matches TVM `sch.reorder` — same-chain + affine-binding constraints.

### `ComputeAt(block, loop)` / `ReverseComputeAt(block, loop)`

Unchanged semantics. When the chain hosting a moved block is later reshaped (via Split/Fuse on that axis), the block re-anchors at `min(old_depth, new_chain_length)` along the new chain. The block's own inner structure is untouched.

### No Reshape Atom

Earlier design iterations considered a `Reshape(axis, new_factors)` atom that rebuilt the chain. Dropped in favor of Split+Fuse because:

- Every Reshape is equivalently a Fuse (collapse chain) + Split (re-split with new factors) — two TVM primitives, exact replay.
- Split/Fuse are more legible to agents (smaller per-atom decision).
- Matches TVM's own primitive surface.

## Renderer

One rule change: for each op leaf, along each tile axis, the innermost ancestor loop on that axis becomes the slice width on the ISA call instead of a Python `for`.

**Algorithm:**
1. Walk the tree.
2. At each `LoopNode`: if no body leaf below has this loop as its innermost tile loop for *any* axis, emit `for i in range(trip):` and recurse.
3. If this loop is the innermost tile loop for at least one descendant leaf, emit no `for`; the leaf's ISA call uses `trip` as the slice width on that axis.

Rule 3 is the "tensorize" step. In TVM-boundary replay, this corresponds to a terminal `sch.tensorize(loop, intrin_name)` per leaf — not represented as a nkigym atom in v1, added by a boundary post-processor.

## Enumeration (for autotuner)

### `Split.analyze(module) -> list[Split]`

For each `LoopNode` in the tree:
- For each non-identity factorization of `loop.extent` (i.e. tuples of length ≥ 2 whose product equals extent):
  - Compute the new innermost extent (`factors[-1]`).
  - For each op leaf whose innermost tile loop is `loop`: check `MIN ≤ factors[-1] ≤ MAX`.
  - If all leaves pass, emit `Split(loop, factors)`.

Bounded by: factor count ≤ small constant (e.g. 3), each factor is a divisor of the current extent. When MIN=MAX (fixed tile), the innermost factor is forced to that value; the search only varies the outer factors' decomposition.

### `Fuse.analyze(module) -> list[Fuse]`

For each adjacent same-axis loop pair:
- Compute fused extent.
- Check `MIN ≤ fused_extent ≤ MAX` for affected innermost-tile leaves.
- If pass, emit `Fuse(outer, inner)`.

### `Reorder.analyze`, `ComputeAt.analyze`, `ReverseComputeAt.analyze`

Unchanged from today.

## TVM Replay

Every atom is 1:1 with a TVM MetaSchedule primitive:

| nkigym atom            | TVM call                           |
|------------------------|------------------------------------|
| `Split(loop, factors)` | `sch.split(loop, factors=factors)` |
| `Fuse(loops)`          | `sch.fuse(*loops)`                 |
| `Reorder(loops)`       | `sch.reorder(*loops)`              |
| `ComputeAt(blk, loop)` | `sch.compute_at(blk, loop)`        |
| `ReverseComputeAt(blk, loop)` | `sch.reverse_compute_at(blk, loop)` |

Boundary post-processor adds a terminal `sch.tensorize(innermost_loop, intrin_name_from_op_class)` per leaf when replaying INTO TVM, reflecting nkigym's implicit tile absorption.

Legality checks differ: nkigym validates MIN/MAX on Split/Fuse; TVM accepts the split regardless. When replaying FROM TVM, nkigym re-validates MIN/MAX and rejects illegal traces.

## Shared-Axis Semantics

Multiple ops can share an abstract axis (same `dim_id`). Legality check operates **per op leaf at the leaf's actual depth** on that axis, not globally:

- Ops at the same depth on the same axis-chain necessarily render with the same slice width.
- Ops at different depths (e.g. transpose inside a matmul's N-loop, where the matmul sits one level up) see different tiles — the deeper leaf has more ancestors contributing to the product, hence a smaller innermost.

`Split` and `Fuse` walk the subtree, find each affected leaf, and check that leaf's innermost-tile extent against its own op's `[MIN, MAX]`. No false conflicts.

## Migration Impact

1. Rename `TILE_LIMITS: dict` → two dicts `MIN_TILE_SIZE`, `MAX_TILE_SIZE` per op. Mechanical.
2. Canonical builder: for each axis, emit `for outer in range(extent/MAX): for inner in range(MAX):` (or `1, extent` if MAX=None).
3. Today's implicit "renderer absorbs TILE_LIMITS" becomes explicit "renderer absorbs innermost tile loop per leaf axis." Same emitted NKI source for today's kernels.
4. Replace today's `Split` legality ("inner factor divides op tile limit") with "innermost extent ∈ [MIN, MAX] per affected leaf." More precise.
5. Add `Fuse` atom. Today's code has no equivalent — this is genuinely new search capability.

## Out of Scope (v1)

- Multi-intrinsic per op (e.g. TE vs VE transpose). Deferred; `MIN/MAX` will become lists of dicts when added, plus a `BindIntrinsic` atom.
- `Tensorize` atom. Terminal tile absorption is implicit in the renderer; no atom representation.
- Non-tile axes (where `MIN/MAX` absent). All current ops have tile axes everywhere; this remains a future extension for ops with structural-only axes.

## Open Questions

None in v1. Multi-intrinsic design will close remaining gaps when prioritized.
