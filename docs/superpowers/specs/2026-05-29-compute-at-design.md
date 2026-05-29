# ComputeAt / ReverseComputeAt + Affine Region Overlap

## Problem

`ComputeAt` and `ReverseComputeAt` are the next transforms after Split,
Fuse, Reorder. They move a block under a target loop so producer and
consumer share an enclosing loop nest — the single-stage PSUM hoist is
the motivating case (sink the `psum_prod` memset and lift the
`tensor_copy` under the matmul's `(M, N)` loops so the per-tile PSUM
lifetime fits Trn2's 2 MiB).

Both transforms' legality is built on the producer-consumer dependency
graph (`Dependency.must_precede`, `producers`, `consumers`). That graph
currently decides "do two blocks conflict?" by **tensor-name match
only** — `Dependency._summarise` collapses each block's `BufferRegion`s
to a set of tensor names and `_bufferregion_overlaps` is a `return True`
stub. For canonical IR (every block touches the whole tensor) this is
exact. But compute_at produces blocks that touch *tiles*: two blocks can
write the same tensor on disjoint slices (`psum_prod[m=0]` vs
`psum_prod[m=1]`) with no real dependency. Tensor-name matching inserts a
spurious WAW edge and serializes them — which makes compute_at reject
legal moves or mis-place insertion points.

So region overlap is a **prerequisite** for trustworthy compute_at on
tiled IR, not a deferred nicety. This spec covers both, region overlap
first.

## Goals

1. Replace the `_bufferregion_overlaps` stub with **precise affine
   integer-set intersection** over the `Expr` AST. Two same-tensor
   regions conflict iff their per-axis index ranges actually intersect
   for some assignment of the loop variables they share.
2. Thread region overlap into `Dependency` so hazard edges
   (RAW/WAW/WAR) are emitted only on genuine region conflicts, not on
   tensor-name coincidence.
3. Ship `ComputeAt` (sink a producer block under a consumer's loop) and
   `ReverseComputeAt` (lift a consumer block under a producer's loop),
   enforcing the six legality conditions documented in
   `nkigym/src/nkigym/transforms/compute_at_legality.md`.
4. Full-coverage-only: a move is legal only when the target's enclosing
   same-axis loops cover the FULL extent of the moved block's same-axis
   loops; covered loops collapse into the target's shared scope,
   uncovered dims stay as the moved block's private inner loops.
   Partial-coverage cases must `Split` first.

Non-goals:
- Partial-coverage region regeneration (TVM's full
  `CalculateBlockVarDomain`). We require full coverage; Split composes.
- Multi-leaf compound blocks. compute_at may nest a block under another
  block's loops, but each block still has one ISA leaf.
- `decompose_reduction` / `compose_reduction`. The memset is already a
  sibling block (decomposed canonical); no reduction-init machinery.

## Decisions captured (from brainstorming)

- **Two transforms**, `ComputeAt` + `ReverseComputeAt`, mirroring TVM —
  not one flagged transform. Each has one direction.
- **Full-coverage-only** loop handling (no residual regeneration).
- **TVM's six legality conditions** (already written up in
  `compute_at_legality.md`), mapped onto our `BlockNode` IR + the
  block-keyed `Dependency` graph.
- **Region overlap = precise affine integer-set intersection** (not the
  conservative or same-loop-var-only subset). Needs a small integer-set
  layer over the `Expr` AST.
- **Single root-child granularity**: one application moves one block.
  Multi-block lifts (the PSUM-hoist trio) compose from single moves.

## Part A — Affine region overlap

### IntervalSet over Expr

Add `nkigym/src/nkigym/ir/interval.py`. An `IntervalSet` represents the
integer range a single `BufferRegion` axis covers, as an affine
expression in loop variables plus a constant width:

```python
@dataclass(frozen=True, kw_only=True)
class AffineInterval:
    """Half-open integer interval ``[base, base + width)`` where ``base``
    is an affine combination of loop-var symbols.

    Attributes:
        coeffs: affine coefficients of ``base`` (``to_affine`` form:
            ``{var_name: coeff, ..., None: const}``).
        width: constant tile width (``hi - lo`` of the BufferRegion range).
    """
    coeffs: dict[str | None, int]
    width: int
```

Built from a `BufferRegion` range `(lo, hi)` by:
`base_coeffs = to_affine(lo)`, `width = to_affine(hi)[None] - to_affine(lo).get(None, 0)` when `hi` is a bare width `Const`, OR `width = eval(hi - lo)` for the `(lo, lo+width)` form. (The canonical builder emits ranges as `(lo, width_const)` — see `_build_region`: the second tuple element is the width `Const`, not `hi`. The interval is `[lo, lo + width)`.)

> **Note on the range encoding.** `BufferRegion.ranges` entries are
> `(lo_expr, width_expr)` where `width_expr` is a `Const` tile width
> (confirmed in `canonical_build._build_region` and consumed by
> `body.render_buffer_region` as `lo : lo + width`). `AffineInterval`
> takes `coeffs = to_affine(lo)` and `width = width_const.value`.
>
> **Partition-axis hazard.** Axis 0 of SBUF/PSUM operands uses a DIFFERENT
> encoding: `lo` is the *bare* partition-tile index (`i_d0_0`, not
> `i_d0_0 * 128`) and `width` is `128`, because the 3D SBUF layout
> `(128, num_tiles, F)` indexes the tile dimension by raw index. So for
> axis 0 the interval in *tile-index space* is `[i_d0_0, i_d0_0 + 1)`
> (width 1), NOT `[i_d0_0, i_d0_0 + 128)`. The interval builder must
> detect the partition axis (axis 0 + SBUF/PSUM location + bare-Var lo)
> and use width 1 in tile space, or normalise all axes to a consistent
> space before comparing. Comparing a bare-index axis-0 against a
> scaled-offset axis with the same `intervals_disjoint` logic would be a
> units mismatch. Recommended: the interval builder consumes the
> `Buffer.location` + axis index and converts axis-0 of sbuf/psum to
> element space (`base *= 128`, `width = 128`) so every axis is in a
> uniform element coordinate before the disjointness test.

### Disjointness test

Two `AffineInterval`s `A = [a, a+wa)` and `B = [b, b+wb)` on the same
axis are **provably disjoint** iff the difference `d = a - b` (affine in
the shared loop vars) is bounded away from the overlap window
`(-wb, wa)` for every legal assignment of those vars. Concretely:

```python
def intervals_disjoint(a: AffineInterval, b: AffineInterval,
                       loop_extents: dict[str, int]) -> bool:
    """Return True iff [a.base, a.base+a.width) and [b.base, b.base+b.width)
    are disjoint for EVERY assignment of the loop vars within their extents.

    Overlap requires -b.width < (a.base - b.base) < a.width. We compute the
    integer range of (a.base - b.base) over the loop-var box (each var in
    [0, extent)) and check it cannot fall in the open overlap window.
    """
```

The difference `a.base - b.base` is affine: `sum(c_v * v) + c0`. Its
integer range over the box `v in [0, extent_v)` is
`[c0 + sum(min(0, c_v*(ext_v-1))), c0 + sum(max(0, c_v*(ext_v-1)))]`.
Disjoint iff that range does not intersect the open interval
`(-b.width, a.width)`.

Key cases this gets right:
- **Same loop var, different tiles**: `a.base = i_m*128`,
  `b.base = i_m*128` → difference is `0`, width window `(-128, 128)`
  contains 0 → **overlap** (same tile, same iteration — correct, it's a
  RAW within the shared loop).
- **Two DIFFERENT loop vars** `i_m` vs `i_m'` (e.g. one block's loop and
  another's): difference `i_m*128 - i_m'*128` ranges over
  `[-128*(ext-1), +128*(ext-1)]` which straddles 0 → **overlap assumed**
  (sound: when both are 0 they coincide). This is the conservative-correct
  fallback when vars are independent.
- **Same loop var, constant offset apart**: `a.base = i_m*128`,
  `b.base = i_m*128 + 128` → difference constant `128`, not in
  `(-128, 128)` → **disjoint** (adjacent tiles, correct).

Two regions on the same tensor overlap iff they overlap on **every**
axis (a box intersection is non-empty iff every dimension's projection
intersects). So: regions disjoint iff **any** axis is provably disjoint.

### Wiring into Dependency

`Dependency._summarise` currently keeps only tensor-name sets. Extend
`_BlockInfo` to keep the per-tensor `BufferRegion`s (read set and write
set), plus the enclosing-loop extents for each block (needed to bound
the affine difference). `_record_hazards` / `_try_edge` then gate each
candidate edge on `not regions_disjoint(producer_region, consumer_region,
extents)` for the conflicting tensor.

For canonical IR every region is the full tensor (`lo = i*tile`,
loops cover the whole extent) → never provably disjoint → all current
edges preserved. The matmul dependency chain is unchanged. The new code
only *removes* edges that were spurious, so existing dependency tests
stay green.

### Region-overlap tests (`test/ir/test_interval.py`, NEW)

1. Same-loop-var same-tile → overlap.
2. Same-loop-var adjacent tiles (offset = width) → disjoint.
3. Same-loop-var gap (offset > width) → disjoint.
4. Different loop vars → overlap (sound fallback).
5. Multi-axis: disjoint on one axis, overlapping on another → disjoint
   (box intersection empty).
6. Multi-axis: overlapping on all axes → overlap.
7. Negative-coefficient affine base (reversed iteration) → correct range
   bound.
8. `Dependency` regression: hand-build a 2-block IR writing disjoint
   tiles of one tensor; assert NO edge. And overlapping tiles → edge.

## Part B — ComputeAt / ReverseComputeAt

### Refresh the legality doc first

`compute_at_legality.md`'s "What is a block?" section still describes the
removed `init` field and bundled memset. Update it to the decomposed
canonical (memset is a sibling block; no `init`; `tree.root` is the root
BlockNode). The six conditions themselves are unchanged.

### Option payloads

```python
@dataclass(frozen=True)
class ComputeAtOption(TransformOption):
    """Sink producer ``block_nid`` under ``target_loop_nid``.

    Attributes:
        block_nid: root-child BlockNode to sink (the producer).
        target_loop_nid: a ForNode in a consumer block's nest; the
            producer is moved to execute inside this loop.
        index: insertion position among the target loop body's existing
            children, in TVM's convention (-1 = last legal slot,
            -2 = earliest legal slot, >=0 = explicit).
    """
    block_nid: int
    target_loop_nid: int
    index: int = -1


@dataclass(frozen=True)
class ReverseComputeAtOption(TransformOption):
    """Lift consumer ``block_nid`` under ``target_loop_nid`` (mirror)."""
    block_nid: int
    target_loop_nid: int
    index: int = -1
```

### The six legality conditions (from compute_at_legality.md)

Implemented in `_check_legality`, raising `TransformLegalityError`:

1. **Stage pipeline** — precondition: `Dependency` graph is acyclic
   (guaranteed by construction; assert).
2. **Block is complete/reduction (not opaque)** — the moved block's ISA
   leaf has no `SEQUENTIAL`-role iter_var (`role_of(block, axis)`).
3. **Target not ancestor of block** — `target_loop_nid` not in
   `tree.descendants(block_nid)` and vice versa; the two live on disjoint
   root-child subtrees.
4. **(ComputeAt only) block is not the output** — the moved producer's
   writes don't include `ir.return_name`.
5. **All required visited** —
   - ComputeAt: every consumer `C` of the moved producer is either a
     descendant of `target_loop_nid` OR in a root-sibling whose
     pre-order index is `> target_root_index`.
   - ReverseComputeAt: every producer `P` of the moved consumer is
     either a descendant of `target_loop_nid` OR in a root-sibling whose
     pre-order index is `< target_root_index`.
   (Uses the block-keyed `Dependency.consumers` / `producers`.)
6. **Insertion gap exists** — among the target loop body's children,
   the legal insert range is `(last_producer_position,
   first_consumer_position]`; reject if empty. `index` selects within it.

### Full-coverage move mechanics (`apply`)

`apply` deep-copies the IR, then:

1. Identify the moved block's same-axis loops and the target's enclosing
   same-axis loops (walk `tree.ancestors(target_loop_nid)` for ForNodes).
2. **Full-coverage check**: for each axis the moved block shares with the
   target's enclosing loops, the target's loop extent product on that
   axis must equal the moved block's loop extent product (full cover).
   Else raise (Split-first guidance in the message).
3. **Collapse covered loops**: drop the moved block's covered-axis
   ForNodes; rewrite the moved block's `iter_values` so the covered
   iter_vars bind to the target's loop vars instead (substitution, same
   machinery as Split/Fuse's binding rewrite, propagated into the leaf's
   `operand_bindings` via `_block_local_descendants`).
4. **Keep uncovered loops** as the moved block's private inner nest.
5. **Splice** the moved block (with its now-reduced loop nest) into the
   target loop body at the legal `index`, preserving sibling order
   (`_replace_in_parent_children` / explicit out-edge rewrite).
6. Rebuild `Dependency`.

ReverseComputeAt is the mirror (lift consumer, same collapse/splice).

### Buffer placement descends automatically (side effect)

**Decision: buffer descent happens automatically as a side effect of
every move.** After `apply` splices the moved block, it re-runs the
LCA-of-users placement over the affected buffers and relocates each
`Buffer` to the new LCA of its (possibly-now-co-located) touchers. No
separate `sink_buffer` primitive.

Mechanism: factor the LCA placement logic out of `canonical_build` into
a reusable `place_buffers(tree)` helper (in a shared module, e.g.
`nkigym/ir/buffer_placement.py`). `canonical_build` calls it once at
build time; `ComputeAt.apply` / `ReverseComputeAt.apply` call it again
after the block move. Because placement is a pure function of tree
topology + each block's reads/writes, re-running it is idempotent and
always yields the correct (lifetime-dominating) allocation point.

PSUM-hoist falls out for free: once the tensor_copy is lifted and the
memset is sunk under the matmul's shared `(M, N)` loops, all three
blocks touching `psum_prod` share that scope, their LCA descends from
root into the matmul block, and `place_buffers` moves `psum_prod`'s
`Buffer` into the matmul block's `alloc_buffers` — the renderer then
emits the `nl.ndarray(buffer=nl.psum)` inside the `(M, N)` loops, one
tile resident at a time.

This means `place_buffers` must be safe to call on a post-move tree
(buffers currently on a block whose LCA changed get moved; buffers
whose LCA is unchanged stay put). Implement it as: clear all
`alloc_buffers` on every block, recompute each buffer's LCA over its
touchers, re-attach. Pure recompute, no incremental diffing.

### Transform tests

`test/transforms/test_compute_at.py` (NEW) and
`test/transforms/test_reverse_compute_at.py` (NEW):

- analyze enumerates legal targets on a Split-tiled matmul.
- apply: PSUM-hoist trio — ReverseComputeAt the tensor_copy under
  matmul's M,N; ComputeAt/ReverseComputeAt the memset; assert resulting
  tree nests correctly and renders to working NKI (fp32 sim at 5e-3).
- each of the six legality conditions has a rejection test.
- self-inverse where applicable; input-IR-preservation.
- render+numerics gate after a full PSUM-hoist sequence.

## Layout

```
nkigym/src/nkigym/ir/
├── interval.py           # NEW — AffineInterval, intervals_disjoint, regions_disjoint
├── buffer_placement.py   # NEW — place_buffers(tree): LCA-of-users, extracted from canonical_build
├── canonical_build.py    # EDIT — call place_buffers() instead of inline LCA logic
└── dependency.py         # EDIT — region-gated hazard edges; _BlockInfo keeps regions+extents

nkigym/src/nkigym/transforms/
├── compute_at.py         # NEW — ComputeAt, ComputeAtOption
├── reverse_compute_at.py # NEW — ReverseComputeAt, ReverseComputeAtOption
├── _tree_ops.py          # reuse _block_local_descendants, _replace_in_parent_children
└── __init__.py           # EDIT — export the two transforms + options

nkigym/src/nkigym/transforms/compute_at_legality.md  # EDIT — refresh "What is a block?"

test/ir/test_interval.py              # NEW
test/ir/test_dependency.py            # EDIT — disjoint-tile no-edge test
test/transforms/test_compute_at.py    # NEW
test/transforms/test_reverse_compute_at.py  # NEW
examples/matmul_lhsT_rhs.py           # EDIT — add ComputeAt+ReverseComputeAt to transforms list
```

## Implementation order

1. **Extract `place_buffers`**: pull the LCA-of-users logic out of
   `canonical_build` into `buffer_placement.place_buffers(tree)`; have
   `canonical_build` call it. Pure refactor — canonical output
   byte-identical, all tests green.
2. **Part A**: `interval.py` + tests → wire into `Dependency` + regression
   test. Gate: all existing dependency/ir/codegen tests green; new
   disjoint-tile test passes.
3. **Legality doc refresh**.
4. **ComputeAt**: option, analyze, `_check_legality` (6 conditions),
   `apply` (full-coverage collapse + splice + `place_buffers` rerun).
   Tests including the buffer-descent side effect.
5. **ReverseComputeAt**: mirror. Tests.
6. **PSUM-hoist E2E**: drive the trio on the Split-tiled matmul, render,
   fp32-sim; assert `psum_prod` descended into the matmul block's
   `alloc_buffers`. Add the transforms to `examples/matmul_lhsT_rhs.py`'s
   MDP transform list; confirm random rollouts stay numerically correct.

## Out of scope

- Partial-coverage regeneration (Split composes to reach full coverage).
- multi_buffer, software_pipeline.
- decompose/compose_reduction.
