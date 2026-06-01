# ComputeAt / ReverseComputeAt + Affine Region Overlap

## Status (amended 2026-05-30)

- **Part A (affine region overlap)** — SHIPPED (`ea75f23`):
  `nkigym/ir/interval.py`, `nkigym/ir/buffer_placement.py`, region-gated
  `nkigym/ir/dependency.py`, plus `test/ir/test_interval.py`,
  `test/ir/test_dependency.py`, `test/ir/test_buffer_placement.py`.
- **Writers-as-list / `LoopPartition`** — MOVED OUT (2026-05-31) to
  `docs/superpowers/specs/2026-05-31-loop-partition-and-writers-list-design.md`,
  sequenced after this spec. `compute_at` does not need it (its only
  multi-writer pattern, memset+matmul, is non-disjoint and RMW-chains
  correctly under the current single-`last_writer`).
- **`ReverseComputeAt`** — legality (`_check_legality`) done; `apply`
  move mechanics are `NotImplementedError`.
- **`ComputeAt`** — not started.
- **Part C (buffer compaction)** — NEW in this amendment; not started.
  `compact_shapes(tree)` materializes the shrunk shape on the tree after
  every shape-affecting `apply` (symmetric with `place_buffers`);
  `rebased_region` projects the rebased index at read time.
- **Shipped groundwork (2026-05-30):** `Buffer.physical_shape()` (the
  2D-logical→3D-physical layout helper, now the single source of truth
  for allocs + PNG labels) and a single `PARTITION_DIM` constant in
  `ir/tree.py` (was four copies). These land ahead of Part C since the
  compaction wiring builds on them.

This amendment (2026-05-30) adds **Part C** and answers two design
questions raised against the hand-written `kernel_transforms.py` ladder:
whether TVM has these code-motion atoms, and how to carve them. The
answers reinforce the existing two-transform decision and surface the
one piece the original spec left implicit — buffer *shape* (as opposed
to buffer *placement*) after a move. Per the inspect-after-every-atom
workflow, shape compaction is materialized on the tree inside each
`apply`, NOT deferred to a single render-time pass.

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

### Does TVM have these atoms? (2026-05-30)

Yes, and the shape matches exactly. TVM TIR exposes **two** schedule
primitives, `compute_at(block, loop)` and `reverse_compute_at(block,
loop)`, both in `src/tir/schedule/primitive/compute_at.cc`. They share
**one** core implementation (`ComputeAtOrReverseComputeAtImpl(...,
is_compute_at)`); the only differences are direction (which neighbor set
bounds the legal insertion region) and the output-block guard
(compute_at only). TVM has **no** separate primitive for "full coverage"
vs "partial" vs "sink deeper" vs "remove same loops" — its region-cover
arithmetic computes, for whatever `loop` you pass, which of the moved
block's loops collapse and which regenerate as residual loops below the
insertion point. `preserve_unit_loops` decides whether trip-1 residuals
survive; our IR keeps trip-1 loops, so that flag is effectively `True`.

### How to carve the atoms — the four hand-written cases are ALL one `ComputeAt` (2026-05-30)

The `kernel_transforms.py` ladder names four producer-sink transitions
"move". All four are the **same** atom (`ComputeAt`: sink a producer)
applied with different `target_loop_nid`. The shape differences are
*outcomes* of the single region-cover step (collapse covered same-dim
loops, regenerate uncovered dims as residual inner loops), NOT separate
atoms:

| Transition | Block moved | Producer→consumer | Coverage outcome |
|---|---|---|---|
| k1→k2 | `load_lhsT` | producer → matmul | `(d0,d1)` exact-cover → block becomes loopless at insertion |
| k3→k4 | `load_rhs`  | producer → matmul | `(d0,d2)` covered; target's extra `d1` → redundant re-loads |
| k6→k7 | `memset`    | producer → matmul | `d1` covered; block's `d2` survives as residual inner loop |
| k9→k10 | `memset`   | producer → matmul | block's own `d2` matches matmul's sibling `d2` → merge into one |

(k7→k8 is a further `ComputeAt` sinking `load_lhsT` under the inner `d2`;
the reverse hoists k11→k14, labeled "reverse_move", are
`ReverseComputeAt` — lift a consumer.)

Carving into four atoms by coverage shape was **rejected**: the shapes
overlap (k3→k4 is exact-cover AND carries a redundant target loop — is it
"full" or "+loop"?) and share all mechanics. A single unified `Move` with
direction inferred was also rejected: legality must branch on direction
anyway (output-block guard, producer-set vs consumer-set), re-introducing
the two cases inside one atom and losing the clean TVM mirror. **Two
atoms by direction**, one shared move implementation — already the
existing spec's decision; the four cases confirm it.
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

### Multiple writers per tensor — out of scope here (moved 2026-05-31)

`Dependency` currently tracks a single `last_writer` per tensor, which is
correct for canonical IR and for `compute_at` (the only multi-writer
pattern compute_at touches — memset + matmul on `psum_prod` — is
*non-disjoint*: the matmul RMW-reads then overwrites, so the two RMW-chain
correctly as `memset→matmul→copy`). The latent forgotten-writer bug
arises only with *disjoint* same-tensor writers, which nothing in this
spec produces — only the future `LoopPartition` transform does. The
writers-as-list fix (and the per-scope-graph fidelity question) therefore
live in their own spec, sequenced after this one:
`docs/superpowers/specs/2026-05-31-loop-partition-and-writers-list-design.md`.
`compute_at` runs on the current single-`last_writer` `Dependency`
unchanged.

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

> **Placement, shape, and rebase are three passes (amended
> 2026-05-30).** The move proper is purely structural — it moves loops
> and rewrites covered-loop bindings (step 3); it never touches
> `Buffer.shape` or operand indices. Two derived passes run after it in
> every `apply`, both detailed in **Part C**:
> 1. `place_buffers` — descends *which block declares* each buffer to
>    its new LCA (e.g. `psum_prod` into the matmul block).
> 2. `compact_shapes` — shrinks `Buffer.shape` to the region bbox
>    (k1→k2: `sbuf_lhs_T (128,16,2048)→(128,1,128)`; PSUM hoist:
>    `psum_prod` to one `(M_tile, N_tile)`). Idempotent, materialized on
>    the tree like placement.
>
> The accompanying **index rebase** is NOT a pass — tree regions stay
> global-frame (for `Dependency`); rebase is applied at read-time by the
> inspection surfaces. So after each atom the tree carries descended
> placement + compacted shape, and rendering projects the rebased index.

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

## Part C — Codegen buffer compaction (amended 2026-05-30)

### Why this is the missing piece

The move atoms (Part B) are purely structural: they relocate loops and
re-bind covered iter_vars, then `place_buffers` descends each buffer's
*declaration* to its new LCA. But nothing shrinks `Buffer.shape`, and
the renderer (`codegen/body.py`) emits shape straight from `buf.shape`
and indices straight from the raw `BufferRegion`. So after k1→k2 the
renderer would emit:

```python
for i_d0_0 in range(16):
    for i_d1_0 in range(16):
        sbuf_lhs_T = nl.ndarray((128, 16, 2048), ...)   # full, re-declared each iter
        dma_copy(dst=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0)*128 : +128], ...)
```

— over-allocated and wrongly indexed versus the hand-written `kernel_2`
(`(128,1,128)`, indexed `[0:128, 0, 0:128]`). Compaction closes that gap.

### Shape follows region — a decoupled pass, materialized after each atom

TVM's `CompactBufferAllocation`
(`src/tir/transforms/compact_buffer_region.cc`) computes each buffer's
accessed region within its allocation scope and shrinks the `Allocate`
extents to that bounding box, decoupled from the schedule primitives
(`compute_at` &c. leave buffers full-sized). We take the **decoupling**
— shape is a separate region-derived computation, never folded into the
move — but place it like our own `place_buffers`: a pass that runs after
every `apply` and **materializes its result on the tree**, because every
atom is inspected.

So `Buffer.shape` is **authoritative tree state recomputed by
`compact_shapes`**, exactly as `alloc_buffers` is authoritative state
recomputed by `place_buffers` — not vestigial, not render-lazy. The move
atom itself never touches shape (decoupling preserved); the post-move
`compact_shapes` records it. Rejected alternatives: option 2 (the move
*itself* recomputes shape — couples a capacity side-effect into the
structural move, and would force every move to special-case which
buffers it touched); option 3 (a separate user-invoked `CompactBuffer`
*atom* — makes k1→k2 two atoms in the action space, and lets
intermediate IR over-allocate between them). Both are worse than a
non-atom pass that fires automatically post-move. The bbox analysis is
identical across all three; running it as an automatic post-pass (a)
keeps the move structural, (b) compacts after every atom for inspection,
and (c) fixes the latent Split/Fuse over-allocation for free.

### Run after every transform — symmetric with `place_buffers`

Every atom's result is rendered and verified before the next atom is
applied (the inspect-after-each-atom workflow). So intermediate
compaction is *consumed*, not discarded — `compact_shapes` runs **inside
every shape-affecting `apply`**, materialized on the tree just like
`place_buffers`. No work is wasted: the IR you inspect at step k already
shows step-k's compacted buffers.

Compaction has two halves that materialize differently:

- **Shape shrink — materialized on the tree.** `compact_shapes(tree)`
  recomputes each `Buffer.shape` as the region bounding box over its
  declaration scope and writes it back, just as `place_buffers`
  recomputes `alloc_buffers`. It is **idempotent**: the bbox is derived
  from regions + loop extents, ignoring the buffer's current shape, so
  `compact(compact(tree)) == compact(tree)`. Safe to store because
  **nothing reads `Buffer.shape` during a rollout** — `Dependency` keys
  overlap off regions + `location`, transform legality/LCA read
  topology + regions; the only `buf.shape` readers are the three
  render-time inspection surfaces. Materializing it is observably inert
  except to those surfaces.
- **Index rebase — a read-time projection, NOT materialized.** Tree
  `BufferRegion`s MUST stay in **global frame** (`lo = i_d0_0*128`,
  &c.), because the shipped Part A overlap math (`interval.py`) compares
  regions across blocks in that frame using loop extents. Rebasing on
  the tree would (a) break `Dependency` (mismatched frames) and (b) be
  non-idempotent (`i_d0_0 → 0` once, `→ -i_d0_0` twice — double
  subtract). So the anchor subtraction is applied when a region is
  *read* for display, never stored.

This is the same structural-vs-derived split as `place_buffers`, applied
within one pass: the **idempotent** half (shape) materializes on the
tree after every atom; the **non-idempotent** half (rebase) is a pure
projection computed by the shared read helper. Mirrors TVM's spirit
(`CompactBufferAllocation` shrinks the `Allocate`; index remapping is a
lowering rewrite) while honoring this repo's "inspect after every atom"
loop.

`compact.py` therefore exposes two entry points:
- `compact_shapes(tree) -> None` — in-place shape materialization,
  called by every shape-affecting `apply`. `ComputeAt`/`ReverseComputeAt`
  call it right after their `place_buffers` rerun (a move changes both
  LCA and shape); `Split`/`Fuse` call it alone (tensorize changes region
  widths but not LCA, so placement is untouched).
- `rebased_region(region, buf, tree) -> BufferRegion` — read-time anchor
  subtraction, called by all three inspection surfaces (`kernel.py`,
  `envelope.md`, `tree.png`) so each shows compacted-shape +
  rebased-index consistently against the global-frame tree.

Because all three dump artifacts read buffer geometry, all three call
the shared read helper — `kernel.py` (`body.py`), the `envelope.md`
buffers table (`ir._render_envelope_md`), and the `tree.png` alloc
labels (`Buffer.label`) — or they would disagree.

### The bounding-box analysis

Add `nkigym/src/nkigym/codegen/compact.py`. For each `Buffer`, over
every `BufferRegion` (across all ISA leaves) that touches it, classified
by the buffer's **declaration (LCA) block** — the block that holds it in
`alloc_buffers`:

- **Anchor loops** = loop vars bound *at or above* the declaration block.
  These index *which instance* of the buffer is live; they are
  **subtracted** from every region `lo` (the index rebases toward 0) and
  do **not** contribute to the shape.
- **Interior loops** = loop vars bound *below* the declaration block.
  These index *within* one instance; the shape extent on each axis is the
  bounding box of `lo + width` over the interior-loop box (each interior
  var ∈ `[0, extent)`), with anchors zeroed.

The per-axis extent is computed with the same affine machinery Part A
already ships (`to_affine`, `_affine_range` in `interval.py`) — the
bounding box of an affine `lo` plus its constant `width` over the
interior-loop box. The partition-axis normalization
(`_interval_for_axis`: axis-0 SBUF/PSUM bare-index → element space) is
reused so the 3D `(128, num_tiles, F)` layout's tile axis compacts in
tile space.

Worked, k1→k2 (`sbuf_lhs_T` declared inside the matmul's `(d0,d1)`):
- anchors = `{i_d0_0, i_d1_0}` (bound above the declaration).
- region `[0:128, i_d0_0, (i_d1_0)*128 : +128]`: subtract anchors →
  axis-1 `i_d0_0 → 0`, axis-2 `(i_d1_0)*128 → 0`. Rebased index
  `[0:128, 0, 0:128]`.
- no interior loops over those axes → extents are the bare widths:
  shape `(128, 1, 128)`. ✅ matches `kernel_2`.

Worked, `lhs_T` (a kernel **param**, declared at root, touched by the
sunk load): anchors = ∅ (declared at root, above nothing relevant is
subtracted because params keep their global frame), so its
HBM-side index `[(i_d0_0)*128 : +128, (i_d1_0)*128 : +128]` is
**unchanged** — only allocated (LCA) buffers rebase. Params are never
re-declared or resized.

### Wiring

**Shape (materialized in `apply`).** `compact_shapes(tree)` rewrites
each `Buffer.shape` to its compacted 2D *logical* extents. The 2D→3D
physical expansion stays in `Buffer.physical_shape()` (the single source
of truth shared with the tree visualization), so the stored shape is
logical and every reader that wants the allocation calls
`physical_shape()`. `_emit_alloc(buf)` and `Buffer.label()` are then
**unchanged** — they already read `physical_shape()`; the buffer they
read just already carries the compacted shape. This is the payoff of
materializing on the tree: no emitter signature changes for shape.

**Index rebase (read-time projection).** Tree regions stay global-frame,
so the emitters subtract anchors when they format a region:
- `render_buffer_region(region, buf)` → `render_buffer_region(region,
  buf, tree)`: rebase via `rebased_region` before formatting. Params
  (declared at root, no anchors) project to themselves — unchanged.
- `Buffer.label()` shows shape only (no region), so it needs no rebase;
  the alloc/region split keeps it simple.

Anchors for a buffer = the loop vars bound at-or-above its declaration
block, read from tree topology at projection time — no per-buffer anchor
set is stored. `rebased_region` lives in `compact.py` and is called by
`body.py` and any other surface that prints a region.

No transform writes a *rebased* region; the tree's region geometry stays
global-frame for `Dependency`. This honors "codegen direction: ir →
codegen, never reverse" — the rebase is a render-time read of ir state,
not a writeback.

### Bonus: fixes latent Split/Fuse over-allocation

`Split._do_tensorize` already shrinks regions and `Fuse._do_tensorize`
widens them, but neither updates `Buffer.shape` (today harmless —
CPU-sim has no capacity gate). Once compaction derives shape from
regions, those buffers compact automatically with no extra work.

### Compaction tests (`test/codegen/test_compact.py`, NEW)

1. Canonical IR: every buffer compacts to its full canonical shape
   (no anchors above root) → render byte-identical to current output.
2. k1→k2 hand-target: after `ComputeAt(load_lhsT, matmul d1-loop)`,
   `sbuf_lhs_T` compacts to `(128,1,128)`, index rebases to
   `[0:128, 0, 0:128]`. Assert rendered source matches `kernel_2`'s
   alloc + load lines.
3. k3→k4: `sbuf_rhs` compacts to `(128,1,512)`.
4. PSUM hoist: after the trio, `psum_prod` compacts to one
   `(M_tile, N_tile)` tile.
5. Param invariance: `lhs_T` / `rhs` HBM indices never rebase.

## Layout

```
nkigym/src/nkigym/ir/
├── interval.py           # NEW — AffineInterval, intervals_disjoint, regions_disjoint
├── buffer_placement.py   # NEW — place_buffers(tree): LCA-of-users, extracted from canonical_build
├── canonical_build.py    # EDIT — call place_buffers() instead of inline LCA logic
└── dependency.py         # region-gated hazard edges (shipped Part A); writers-as-list moved to the LoopPartition spec

nkigym/src/nkigym/codegen/
├── compact.py            # NEW — compact_shapes(tree) [materialize] + rebased_region(region,buf,tree) [read-time]
└── body.py               # EDIT — render_buffer_region rebases via compact.rebased_region (_emit_alloc unchanged: already reads physical_shape)

nkigym/src/nkigym/ir/ir.py # EDIT — _render_envelope_md rebases region display (shape via physical_shape, already compacted on tree)

nkigym/src/nkigym/transforms/
├── compute_at.py         # NEW — ComputeAt, ComputeAtOption
├── reverse_compute_at.py # EDIT — fill ReverseComputeAt.apply (legality already shipped)
├── _code_motion.py       # NEW — shared _compute_at_impl(ir, ..., is_reverse); runs place_buffers + compact_shapes
├── split.py, fuse.py     # EDIT — call compact_shapes(tree) in .apply (LCA unchanged by tensorize → no place_buffers needed)
├── _tree_ops.py          # reuse _block_local_descendants, _replace_in_parent_children
└── __init__.py           # EDIT — export ComputeAt + ComputeAtOption (reverse already exported)

nkigym/src/nkigym/transforms/compute_at_legality.md  # already refreshed (ea75f23)

test/codegen/test_compact.py          # NEW
test/transforms/test_compute_at.py    # NEW
test/transforms/test_reverse_compute_at.py  # EDIT — apply-mechanics tests (legality tests shipped)
test/transforms/test_render_equivalence.py  # EDIT — k1→k2, k3→k4 render-match + numerics
examples/matmul_lhsT_rhs.py           # EDIT — add ComputeAt+ReverseComputeAt to transforms list
```

Part A files (`ir/interval.py`, `ir/buffer_placement.py`,
`ir/dependency.py`, `test/ir/test_interval.py`,
`test/ir/test_dependency.py`, `test/ir/test_buffer_placement.py`) are
already shipped (`ea75f23`) — no longer in the work set.

## Implementation order

Steps 1–3 (extract `place_buffers`, Part A region overlap, legality-doc
refresh) are **DONE** (`ea75f23`). Remaining, resequenced so compaction
lands first — it stands alone (compacts existing canonical + Split/Fuse
output) and makes every subsequent move's render verifiable against
`kernel_transforms.py`:

4. **Part C — buffer compaction**: `codegen/compact.py` with two entry
   points — `compact_shapes(tree)` (in-place bbox shape materialization,
   idempotent, reusing `interval._affine_range` / partition
   normalization) and `rebased_region(region, buf, tree)` (read-time
   anchor subtraction). Call `compact_shapes` in the existing
   `Split`/`Fuse`.`apply` (their tensorize flavor changes region widths,
   so shape compacts; LCA is unchanged so no `place_buffers` needed);
   wire `rebased_region` into
   `body.render_buffer_region`. Gate: canonical tree shapes unchanged
   (anchors ∅ above root) and render byte-identical; Split/Fuse
   tensorize buffers now compact on the tree; `compact_shapes`
   idempotent (apply twice == once, like `place_buffers`);
   `test/codegen/test_compact.py` + existing codegen/render/transform
   tests green.
5. **Shared `_code_motion._compute_at_impl`**: full-coverage collapse +
   keep-uncovered-residual + splice at `index` + `place_buffers` rerun +
   `compact_shapes` rerun + `Dependency` rebuild. Direction-parameterized
   (`is_reverse`).
6. **`ReverseComputeAt.apply`**: call the shared impl (legality already
   shipped). Tests: fill `test_reverse_compute_at.py` apply mechanics;
   k11→k14 reverse hoists render-match + fp32-sim.
7. **`ComputeAt`**: option, analyze, `_check_legality` (6 conditions
   incl. output-block guard), `apply` via shared impl. Tests: k1→k2,
   k3→k4, k6→k7, k9→k10 each render-match `kernel_transforms.py` +
   fp32-sim at 5e-3.
8. **PSUM-hoist E2E**: drive the trio on the Split-tiled matmul, render,
   fp32-sim; assert `psum_prod` descended into the matmul block's
   `alloc_buffers` AND compacted to one tile. Add both transforms to
   `examples/matmul_lhsT_rhs.py`'s MDP transform list; confirm random
   rollouts stay numerically correct.

## Out of scope

- Partial-coverage regeneration (Split composes to reach full coverage).
- multi_buffer, software_pipeline.
- decompose/compose_reduction.
