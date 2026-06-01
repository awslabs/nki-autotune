# LoopPartition + Writers-as-List Dependency Model

## Status (2026-05-31)

NOT STARTED. Sequenced AFTER `compute_at` / `reverse_compute_at` (the
`2026-05-29-compute-at-design.md` spec). This spec owns two coupled,
not-yet-built pieces:

1. **`LoopPartition`** — a new transform (TVM's name) that cuts one
   loop's iteration space into consecutive sub-loops, each a sibling
   block writing a disjoint slice of the same buffer.
2. **Writers-as-list dependency model** — the `Dependency` change that
   makes the multiple-disjoint-writers pattern `LoopPartition` produces
   track correctly. Extracted here from the compute_at spec because it is
   motivated by — and only exercised by — `LoopPartition`.

These ship **together** so the transform that produces disjoint writers
co-validates the dependency fix end-to-end (build → partition → check
deps → render → sim), rather than landing the dependency change with only
synthetic tests.

## Part 1 — `LoopPartition` transform

### What it does (TVM `loop_partition`)

Mirrors `tvm.s_tir.Schedule.loop_partition`
(`python/tvm/s_tir/schedule/schedule.py`,
`src/s_tir/schedule/primitive/loop_transformation.cc`). Cuts a loop's
iteration space into consecutive ranges — NOT nested factors (that is the
already-shipped `Split`). Distinct primitive.

```
# Before — one loop, one block:
for i in range(16):
    <body writing out[i*128 : +128]>

# After LoopPartition(loop=i, factors=[8]):  (factor = length of first part)
for i in range(0, 8):     # partition 0 — sibling block, writes out[0:1024]
    <body>
for i in range(8, 16):    # partition 1 (derived tail) — sibling block, writes out[1024:2048]
    <body>
```

### Factor semantics (verified against TVM `concrete_schedule.cc`)

`factors` are **lengths**, not cut points. TVM's pipeline:
1. User passes lengths, e.g. `[2, 64]`.
2. Validation: `sum(factors)` must be **strictly less** than the loop
   extent (room for the derived tail). `[2,64]` sums to 66 < 128 ✓.
3. Cumulative-sum into boundaries: `[2, 66]`.
4. Append the extent: `[2, 66, 128]` → cuts at `0|2|66|128` → three
   consecutive loops `[0,2)`, `[2,66)`, `[66,128)`.

So `n` explicit length factors yield `n+1` partitions (the last is the
derived remainder). At most ONE factor may be `None`/`-1` — that slot
becomes the inferred remainder and you get exactly `n` partitions.

For our IR the factors are trip counts on the canonical loop (e.g.
`LoopPartition(loop_nid, factors=[8])` on a trip-16 loop → trips `[0,8)`
and `[8,16)`).

### Option payload

```python
@dataclass(frozen=True)
class LoopPartitionOption(TransformOption):
    """Cut ``target_loop_nid``'s iteration space into consecutive sub-loops.

    Attributes:
        target_loop_nid: the ForNode to partition.
        factors: per-partition lengths (trip counts), summing to strictly
            less than the loop extent; the derived tail is implicit. At
            most one entry may be None (the inferred remainder).
    """
    target_loop_nid: int
    factors: tuple[int | None, ...]
```

### Mechanics (`apply`)

Deep-copy, then for the target `ForNode` with parent block:
1. Resolve factors → consecutive `[lo, hi)` trip ranges (cumulative-sum +
   derived tail, mirroring `concrete_schedule`).
2. For each range, clone the target loop's subtree (its body block + ISA
   leaf) into a **new sibling BlockNode**, with the loop's `extent`
   replaced by the range length and the loop's start offset folded into
   the block's `iter_values` / `BufferRegion` lo (so partition `k`'s body
   writes `out[range_lo*tile : +range_len*tile]`).
3. Splice the new sibling blocks into the parent at the target's original
   position (preserving sibling order — `_replace_in_parent_children`),
   remove the original.
4. Re-run `place_buffers` + `compact_shapes` (buffer touched by N blocks
   now; LCA may change) + rebuild `Dependency`.

### Legality

- Target is a `ForNode`.
- `sum(explicit factors) < extent`; at most one `None`; each explicit
  factor `>= 1`.
- The loop's body block has no `SEQUENTIAL`-role axis on the partition
  dim (partitioning a sequential axis would reorder dependent state).
  PARALLEL / ACCUMULATION are fine (matches TVM's data-par/comm-reduce
  filter).

### Tests

- `analyze` enumerates length-factor options on the canonical matmul's
  outer loops.
- `apply`: partition a trip-16 loop by `[8]` → two sibling blocks with
  disjoint write regions; render + fp32-sim matches numpy golden.
- Round-trip: `LoopPartition` then `Fuse` of the two consecutive loops
  recovers the original (if Fuse supports consecutive-range merging — else
  note as non-invertible).

## Part 2 — Writers-as-list dependency model

> **Status:** Affine region overlap + `_BlockInfo` regions are SHIPPED
> (`ea75f23`, the compute_at spec's Part A). This part is the NOT-YET-BUILT
> extension that closes a latent multiple-writers bug, exercised by
> `LoopPartition` (Part 1).

**The bug.** `Dependency._build` threads `last_writer: dict[str, int]` —
exactly ONE writer nid per tensor (`last_writer[name] = nid` overwrites).
That is correct only while every tensor has a single writer (canonical
IR). After `LoopPartition` makes N sibling blocks each writing a disjoint
slice of buffer `B`:

- region overlap correctly suppresses WAW *between* the partitions
  (disjoint ranges → no partition→partition edge); but
- `last_writer["B"]` holds only the LAST partition, so a later
  whole-`B` reader gets a RAW edge from that one partition. The earlier
  partitions are orphaned — no path to their consumer — and could
  legally reorder after it. Latent today; a real correctness bug the
  moment `LoopPartition` (or any multi-writer transform) lands.

**TVM's model.** `SBlockScopeNode` keeps `buffer_writers:
map<Buffer, Array<StmtSRef>>` — a LIST of every writer in the scope
(`include/tvm/s_tir/sblock_scope.h`). Its constructor adds a hazard edge
from EACH prior writer. But TVM keys purely on `region->buffer` (the
buffer object), never the region's ranges (`src/s_tir/sblock_scope.cc`):
it has no region-overlap, so it WAW-chains disjoint `LoopPartition`
writers (conservative-serial, correct but coarse). Our region overlap is
strictly finer.

**The change** (`dependency.py` only): replace the single-writer slot
with a live-writers list, mirroring `buffer_writers`:

```python
live_writers: dict[str, list[int]]   # was: last_writer: dict[str, int]
```

`_record_hazards` iterates the list for RAW and WAW (WAR unchanged):

```python
for name in info.reads:
    for w in live_writers.get(name, ()):        # RAW from every live writer
        self._try_edge(w, nid, "RAW", name)
for name in info.writes:
    for w in live_writers.get(name, ()):        # WAW from every live writer
        self._try_edge(w, nid, "WAW", name)
    for prior_r in prior_readers.get(name, ()):  # WAR unchanged
        self._try_edge(prior_r, nid, "WAR", name)
```

`_try_edge` / `_provably_disjoint` are UNCHANGED — region overlap still
gates every candidate edge. So disjoint `LoopPartition` writers stay
parallel (no spurious WAW) while a whole-tensor reader edges from ALL
overlapping writers. This is the **finer-than-TVM** behavior (chosen
deliberately; TVM would serialize the partitions).

**Live-writer maintenance — append-and-prune by block-footprint
coverage.** After block `nid` writes tensor `T`, update
`live_writers[T]`: append `nid`, then remove any prior writer whose write
`nid` FULLY COVERS. Coverage is a NEW affine predicate (a small
`region_covers(outer, inner, extents)` helper alongside
`regions_disjoint` in `interval.py`) — and it must compare **block
footprints, not per-iteration tiles**. A block's footprint on an axis is
its region width × the trips of the enclosing loops that index that axis
(the tile swept over its own loop nest). `outer` covers `inner` iff, over
the loop box, `inner`'s footprint `[lo, lo + width·trips)` on every axis
lies within `outer`'s.

> **Why footprint, not tile.** The canonical matmul writes
> `psum_prod[..., i_d2_0*512 : +512]` inside a 4-trip `i_d2_0` loop —
> per-iteration tile 512, but block footprint 512·4 = 2048. The memset
> writes the full 2048 in one shot (trip 1). At *tile* granularity 512
> ⊉ 2048, so memset would wrongly stay live (→ a redundant `memset→copy`
> edge). At *footprint* granularity 2048 ⊇ 2048, so the matmul prunes the
> memset — leaving the chain `memset→matmul→copy`, identical to today.

Coverage is NOT the complement of disjoint (partial overlap is neither);
a writer that partially overlaps `nid` is neither pruned (not covered)
nor parallel (overlap → WAW edge formed) — it stays live AND ordered,
which is correct. Consequences:

- **Full-extent overwrite** (canonical: memset's whole-tensor write, or
  the matmul's footprint sweep) covers every prior writer of that span →
  prunes them → `live_writers[T]` collapses to a single live writer. This
  makes the canonical graph BYTE-IDENTICAL to today (`memset→matmul→copy`,
  no extra edges) — regression-safe.
- **Disjoint partition** (LoopPartition: `B[0:2]`, footprint 2) covers
  neither sibling → prunes none → all partitions stay live → a later
  whole-`T` reader RAW-edges from all. Within-block loop ordering (e.g.
  "tile 512:1024 not yet written after the first matmul iteration") is
  program order *inside* a block, below the block-to-block graph; the
  graph only needs the whole-region block→block edge, which it has.

`_BlockInfo` already carries the regions + extents needed for the
coverage test (shipped in Part A). Public API (`producers`,
`consumers`, `must_precede`, `direct_*`) is unchanged; consumers
(`reverse_compute_at._check_producers_visited`, the transforms'
`Dependency(tree)` rebuild) see a strictly-more-complete graph.

**Per-scope graphs deferred (still).** TVM keeps one `SBlockScope` PER
scope-root block (`block_info: map<scope_root → SBlockInfo>`,
`include/tvm/s_tir/schedule/state.h`). We keep ONE global graph — the
writers-list fix is topology-agnostic (works flat or nested), and region
overlap makes the global graph correct under nesting regardless (shared
enclosing-loop terms cancel in the affine difference). Per-scope is a
fidelity refactor that can land whenever scope-local legality reasoning
becomes worth it; it is not required by `LoopPartition` or `compute_at`.

**Tests** (`test/ir/test_dependency.py`, plus `region_covers` units in
`test/ir/test_interval.py`):
- Regression: full suite stays green — the canonical `memset→matmul→copy`
  graph is unchanged (the matmul's footprint sweep prunes the memset; no
  redundant `memset→copy` edge).
- **Footprint coverage** (`region_covers` unit): a 512-wide region over a
  4-trip loop COVERS a 2048-wide region over a 1-trip loop on the same
  axis (same 2048 span) → returns True. The naive per-tile comparison
  (512 ⊉ 2048) would return False — this test pins footprint granularity.
- Multi-writer bug: hand-build 3 sibling blocks writing disjoint slices
  of one tensor + 1 whole-tensor reader; assert the reader has RAW from
  ALL three writers, and the three writers have NO WAW among them.
- Prune: a full-tensor overwrite after a prior write → assert the prior
  writer is pruned (WAW chain, single live writer), not kept parallel.
- E2E: `LoopPartition` a matmul output loop, then a downstream
  whole-tensor consumer; assert deps + render + fp32-sim.

## Layout

```
nkigym/src/nkigym/ir/
├── interval.py           # EDIT — + region_covers(outer, inner, extents) (block-footprint coverage)
└── dependency.py         # EDIT — live_writers list + append-and-prune by coverage

nkigym/src/nkigym/transforms/
├── loop_partition.py     # NEW — LoopPartition, LoopPartitionOption
└── __init__.py           # EDIT — export

test/ir/test_interval.py          # EDIT — region_covers footprint units
test/ir/test_dependency.py        # EDIT — writers-list multi-writer + prune tests
test/transforms/test_loop_partition.py   # NEW
examples/matmul_lhsT_rhs.py       # EDIT — add LoopPartition to the MDP transforms list
```

## Implementation order

1. **Writers-list + `region_covers`** first (dependency.py + interval.py),
   with the regression + footprint + multi-writer + prune tests. Gate:
   full suite green (byte-identical canonical), new tests pass.
2. **`LoopPartition`** transform on top, with its own tests + the E2E test
   that drives a real disjoint-writer chain through the writers-list.
3. Add `LoopPartition` to `examples/matmul_lhsT_rhs.py`'s MDP transform
   list; confirm random rollouts stay numerically correct.

## Out of scope

- Per-scope dependency graphs (deferred indefinitely; not required).
- Predicated/partial-tile partitions (TVM's `preserve_unit_iters` and
  predicate insertion) — our factors divide evenly into trip counts.
