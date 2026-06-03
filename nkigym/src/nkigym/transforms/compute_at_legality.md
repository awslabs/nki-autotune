# `ComputeAt` / `ReverseComputeAt` legality

This file explains every legality condition the two compute_at-flavored
transforms enforce. Each condition is shown as a small "before / proposed move
/ verdict" code snippet in our IR's source-rendered form.

The transforms move one *block* under a target loop:

- `ComputeAt(block, target_loop)` — sink a producer block under a consumer's
  loop.
- `ReverseComputeAt(block, target_loop)` — lift a consumer block under a
  producer's loop.

Six conditions, mirroring TVM's `tir.schedule.{ComputeAt, ReverseComputeAt}`
(`src/tir/schedule/primitive/compute_at.cc`, lines 700–731).

## What is a "block"?

In our IR, a **block** is a `BlockNode` payload — a first-class schedulable
unit aligned with TVM's `SBlock`. It owns:

- `iter_vars` — the iteration variables that index the block's body.
- `iter_values` — affine expressions binding each iter_var to surrounding
  `ForNode.loop_var` values.
- `reads` / `writes` — declared `BufferRegion`s in iter_var space.
- `alloc_buffers` — buffers whose lifetime is bounded by this block.
- `annotations` — free-form metadata.

The canonical IR has a synthetic root block holding every kernel-lifetime
buffer in `alloc_buffers` and a sequence of leaf blocks (one per compute op)
under it. Each leaf block has at most one ISA leaf in its body. Compute_at and
reverse_compute_at move blocks under target loops, producing nested topologies.

In the **canonical** IR (before any compute_at), every leaf block is a direct
child of the synthetic root block. The matmul's reduction-init memset is a
*sibling* block emitted before the matmul (no `init` sub-block):

```
root_block(alloc_buffers=[sbuf_lhs_T, sbuf_rhs, psum_prod, sbuf_prod, hbm_out])
├── memset_block  → ForNode(i_d1_0) → ForNode(i_d2_0) → ISANode(NKIMemset)
└── matmul_block  → ForNode(i_d0_0) → ForNode(i_d1_0) → ForNode(i_d2_0) → ISANode(NKIMatmul)
```

After **compute_at / reverse_compute_at**, blocks no longer live flat at root —
moved blocks now share enclosing loops with the target's existing block(s).
Example, after lifting `tensor_copy` and sinking `memset` under the matmul's
`(m, n)` nest (the PSUM hoist):

```
root
└── ForNode(d_M, trip=16)              ← shared enclosing scope
    └── ForNode(d_N, trip=4)              (M and N are no longer "private"
        ├── ISANode(NKIMemset, ...)        to any single block)
        ├── ForNode(d_K, trip=16)
        │   └── ISANode(NKIMatmul, ...)
        └── ISANode(NKITensorCopy, ...)
```

Three blocks live under the shared `(M, N)` scope:

- **memset block** — body = the memset leaf; private nest = empty (no
  ForNodes above it within the shared scope).
- **matmul block** — body = the matmul leaf; private nest = the K-loop only
  (M and N are now shared, not private).
- **tensor_copy block** — body = the tensor_copy leaf; private nest = empty.

The fork point — the deepest ForNode whose subtree contains more than one ISA
leaf — is `ForNode(d_N, trip=4)`. Loops at or above the fork are shared; loops
strictly below the fork (along the path to a single leaf) are that leaf's
private nest.

The compute_at transforms operate at block granularity: one application moves
one block (identified by its ISA leaf nid) under a target loop. After the
move, the moved block's private nest may shrink (covered loops collapse into
the target's shared scope) and grow inner siblings (uncovered dims stay as
private loops below the target).

This matches TVM's "block" — the unit of schedulable computation — but
specialised to our IR: every block has exactly one ISA leaf (no multi-leaf
"compound blocks" like a fused matmul+activation; each op gets its own block).
What changes after compute_at is *where* the block lives, not its identity.

---

## 1. Stage-pipeline scope (no cycles)

The dependency graph over leaves must be acyclic — each tensor has one
linearly-ordered producer-consumer chain. This is a precondition on the IR
itself, not a per-call check; `Dependency` builds an `nx.DiGraph` from a single
pre-order pass that adds edges only "earlier leaf → later leaf", so cycles are
impossible by construction.

```python
# Always satisfied — no example needed. If a future op introduces a
# self-referential RMW pattern, Dependency would still order writes
# in emission order and never produce a back-edge.
```

---

## 2. Block is "complete" or "reduction" (not opaque)

TVM categorises every block by its iter_vars and read/write structure:

- **Complete block** — every iter_var is data-parallel (each iteration writes a
  distinct output cell), the block is the only writer of its outputs in the
  scope (*dominant*), and reads don't overlap with writes. Example: a
  pointwise add.
- **Reduction block** — like complete, but allows `kCommReduce` iter_vars
  (e.g. K in a matmul) provided there's an `init` statement zeroing the output
  and the reduction iter_vars don't appear in the output index. Example:
  matmul, where `(m, n)` are data-parallel and `k` is a commutative reduction.
- **Opaque block** — anything else. Order-dependent iter_vars (`kOrdered`,
  e.g. a prefix scan), data-dependent indexing (`kOpaque`), multiple writers,
  or read-write overlap that isn't a clean reduction pattern.

`compute_at` rejects opaque blocks because TVM's region-cover arithmetic can't
synthesize regenerated loops without knowing what slice of the output one
iteration touches.

In our IR this collapses to a single check: **the moved block has no
`SEQUENTIAL`-role axes on its ISA leaf**.

```python
# OK — NKILoad has all-PARALLEL axes; complete-block analogue.
sbuf_lhs_T = NKILoad()(src=lhs_T)

# OK — NKIMatmul has K as ACCUMULATION (TVM's kCommReduce) and M, N as
# PARALLEL; reduction-block analogue. RMW on `psum_prod` is the init+reducer
# pattern (memset zeroes psum_prod before the K loop, then nc_matmul
# accumulates into it).
psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)

# Hypothetical opaque block — a prefix-scan op declares its scan axis as
# SEQUENTIAL. Compute_at rejects: iterations aren't independent, sinking
# the block under a target loop could change traversal order.
class NKIPrefixScan(NKIOp):
    AXIS_ROLES = {"P": AxisRole.PARALLEL, "F": AxisRole.SEQUENTIAL}
```

Notes on what does *not* make a block opaque in our IR:

- **RMW slots** (`matmul.dst`, `activation_reduce.reduce_res`) are the
  reduction pattern, not arbitrary read-write overlap. Legal.
- **Multiple writers in the IR** (e.g. a memset + a matmul both writing
  `psum_prod`) don't make either block opaque — each individual block has
  one ISA leaf and is dominant within its own subtree. The cross-block
  ordering question is condition 5.
- **Data-dependent indexing** doesn't exist in the current op set
  (`axis_map` is always a static abstract→concrete mapping). If a future op
  introduces `kOpaque` semantics it would need a new `AxisRole`.

---

## 3. Same scope, target loop is not an ancestor of the block

You can't move a subtree under one of its own enclosing loops — that would
create a cycle in the tree.

```python
# BEFORE — canonical matmul nest:
for m in range(M_trips):           # nid=10
    for n in range(N_trips):       # nid=11
        for k in range(K_trips):   # nid=12
            nc_matmul(...)         # nid=13   <-- block root

# Proposed: ComputeAt(block_subtree_root=10, target_loop_nid=12)
#   "sink the matmul-M nest under its own K loop"
# ILLEGAL: target_loop_nid=12 is a descendant of block_subtree_root=10.
```

The check is structural: `target_loop_nid` must not be in
`tree.descendants(block_subtree_root)`, and `block_subtree_root` must not be in
`tree.descendants(target_loop_nid)` — the two must live on disjoint paths from
the root (i.e. they're under different root-children).

---

## 4. Block is not the kernel's output (ComputeAt only)

`ComputeAt` sinks a *producer*. The kernel's final-store leaf has no
consumers (no later leaf reads its output — it IS the output), so sinking it
has no meaningful target. `ReverseComputeAt` doesn't need this check (the
final store is a legal *consumer* to lift).

```python
# Canonical IR ends with:
hbm_out = NKIStore()(src=sbuf_prod)        # writes the kernel's return tensor

# Proposed: ComputeAt(block_subtree_root=<store nest>, target_loop=<anywhere>)
# ILLEGAL: the store writes `hbm_out`, which is the kernel's `return_name`.
# ReverseComputeAt of the store's nest (lifting it under, say, the
# tensor_copy's M-loop) is legal — covered by condition 5.
```

The check: in `ComputeAt`, every leaf descended from `block_subtree_root` whose
`writes` includes `ir.return_name` causes rejection.

---

## 5. Dependency preservation across the move

The directional rule. This is the heart of compute_at legality.

A leaf running at the kernel's root level (any sibling under the tree root,
not inside a loop) writes / reads its tensor *once* in pre-order before any
later root-sibling executes. So the question isn't "is the producer literally
inside the target loop?" — it's "after the move, will every consumer still
see the producer's output?"

The unified rule: pick the moved subtree's *new pre-order position* (under the
target). Every producer-consumer edge in the dependency graph that crosses the
move must remain a forward edge.

### 5a. ComputeAt — sinking a producer

Let `P` be the moved producer subtree (currently at root position `i_P`),
`T` be the target loop (currently at root position `i_T`). After the move,
`P`'s leaves execute *inside* every iteration of `T`'s loop nest, at the
chosen insertion slot among `T`'s children.

For every consumer `C` of `P` (every leaf with a RAW/WAW/WAR edge from any
leaf in `P`):

- If `C` is a descendant of `T` → ✅ (`C` runs in the same iteration that just
  produced what it reads, after `P`'s new insertion point).
- If `C`'s root-sibling has pre-order position `> i_T` (i.e. `C` runs *after*
  the entire target loop completes) → ✅ (`P` will have been re-executed
  `prod(target_trips)` times, the last write is still live).
- Otherwise (`C`'s root-sibling has position `≤ i_T` — `C` runs *before* the
  target loop or is in target's prefix siblings) → ❌.

```python
# BEFORE — three sibling root-children, indexed 0, 1, 2:
psum_prod = NKIMemset(value=0.0)()             # i=0, producer of psum_prod (block A)

for m in range(16):                            # i=1, RMW psum_prod via matmul
    for n in range(4):
        for k in range(16):
            nc_matmul(stationary=sbuf_lhs_T,
                      moving=sbuf_rhs,
                      dst=psum_prod)

for m in range(16):                            # i=2, reads psum_prod
    for n in range(4):
        tensor_copy(src=psum_prod, dst=sbuf_prod)

# Proposed: ComputeAt(block=A (memset), target_loop=<tensor_copy's m-loop>)
#   "sink memset under tensor_copy's m-loop" (target = i=2)
# Consumers of memset's psum_prod write:
#   - matmul nest (root-sibling i=1).  i=1 < target_root=2 → ❌
# ILLEGAL: matmul would now read uninitialised psum_prod (memset moved
#   to AFTER matmul).

# Proposed: ComputeAt(block=A (memset), target_loop=<matmul's n-loop>)
#   "sink memset under matmul's (m, n)" (target root-sibling = i=1)
# Consumers of memset's psum_prod write:
#   - matmul nest's nc_matmul leaf — descendant of target ✅
#   - tensor_copy nest (root-sibling i=2).  i=2 > target_root=1 → ✅
# LEGAL — but watch out: memset now runs once per (m, n), so each
# matmul's K-accumulation starts from zero correctly, and tensor_copy
# at i=2 reads the LAST (m=15, n=3) iteration's psum_prod, which is
# wrong semantics. The dependency check is necessary but not sufficient
# here — see the "full coverage" composition note at the bottom.
```

The check: for `ComputeAt`, for every consumer leaf `C` of any leaf in the
moved subtree, `C` must be either a descendant of `target_loop_nid` or live in
a root-sibling whose pre-order index is `> target_root_index`.

### 5b. ReverseComputeAt — lifting a consumer

Mirror. Let `C` be the moved consumer subtree (root position `i_C`), `T` the
target loop (root position `i_T`). After the move, `C`'s leaves execute inside
every iteration of `T`'s loop nest.

For every producer `P` of `C` (every leaf `C` reads or RMWs):

- If `P` is a descendant of `T` → ✅ (`P` writes its slice in the same
  iteration `C` then reads from).
- If `P`'s root-sibling has pre-order position `< i_T` (i.e. `P` runs *before*
  the target loop, writing the entire tensor once) → ✅ (every iteration of
  the lifted `C` reads already-written data).
- Otherwise → ❌.

```python
# BEFORE — same canonical layout as 5a.

# Proposed: ReverseComputeAt(block=<tensor_copy nest>,
#                            target_loop=<matmul's n-loop>)
#   "lift tensor_copy under matmul's (m, n) — i.e. PSUM hoist"
# tensor_copy reads psum_prod. Producers of psum_prod:
#   - memset (root-sibling i=0).  i=0 < target_root=1 → ✅ (writes whole
#     psum_prod once, before target loop).
#   - matmul (root-sibling i=1, descendant of target).  ✅
# LEGAL.

# AFTER:
psum_prod = NKIMemset(value=0.0)()             # i=0, unchanged

for m in range(16):                            # i=1, target loop
    for n in range(4):
        for k in range(16):
            nc_matmul(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
        tensor_copy(src=psum_prod, dst=sbuf_prod)   # lifted under (m, n)

# Note: this state is semantically valid — the K loop completes for each
# (m, n) before tensor_copy fires, so tensor_copy reads the correct
# matmul output for that tile. memset at root writes psum_prod once
# before any matmul iteration; after the first (m=0, n=0) iteration's
# matmul-K, psum_prod[0:128, 0:512] has been correctly RMW'd from the
# initial zero. Subsequent (m, n) iterations reuse the same psum_prod
# region; the per-(m, n) tile gets initialised by... actually nothing.
# This is the bug the next ReverseComputeAt fixes.

# Proposed: ReverseComputeAt(block=<memset>, target_loop=<matmul's n-loop>)
#   memset has NO reads (only writes psum_prod); zero producers → ✅ trivially.
# LEGAL.

# AFTER (the PSUM hoist):
for m in range(16):
    for n in range(4):
        memset(dst=psum_prod, value=0.0)
        for k in range(16):
            nc_matmul(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
        tensor_copy(src=psum_prod, dst=sbuf_prod)
```

The check: for `ReverseComputeAt`, for every producer leaf `P` of any leaf in
the moved subtree, `P` must be either a descendant of `target_loop_nid` or
live in a root-sibling whose pre-order index is `< target_root_index`.

---

## 6. There exists an insertion point under the target loop

After steps 1–5 establish that the moved subtree *can* go under the target,
this step picks *where* among the target's existing children. The constraint
is a single contiguous gap.

```python
# Before:
for m in range(16):                  # target loop
    for n in range(4):
        memset(...)                  # child 0   (producer of psum_prod)
        for k in range(16):          # child 1   (RMW psum_prod)
            nc_matmul(...)
        tensor_copy(src=psum_prod,   # child 2   (reads psum_prod)
                    dst=sbuf_prod)

# Proposed: ComputeAt(block=<some new producer of sbuf_prod>,
#                     target_loop=<the m-loop above>)
# Insertion point must be:
#   - AFTER all leaves that produce inputs of the moved block, and
#   - BEFORE all leaves that consume outputs of the moved block.
#
# If the new block reads tensor_copy's output (sbuf_prod) and writes
# something that no current child reads, the legal insertion range is
# [child 3, child 3] — only after tensor_copy.
#
# If instead the legal range collapsed to (child 1, child 1] (e.g. the
# moved block both reads matmul's output and feeds tensor_copy), the
# insertion point is exactly between children 1 and 2.
#
# ILLEGAL when the range is empty — e.g. the moved block must come
# AFTER tensor_copy AND BEFORE memset. No valid position; reject.
```

In TVM's terms, the search returns a position in `(last_producer_position,
first_consumer_position]`. For our IR, this maps to: scan the target's
children in order, find the largest index `lp` such that some descendant of
child `lp` is a producer of the moved subtree, and the smallest index `fc`
such that some descendant of child `fc` is a consumer. The legal range is
`(lp, fc]`. If `lp >= fc`, reject.

The option payload carries an explicit `index: int` field selecting where
among the legal range to insert (TVM defaults: `-1` = `fc`, the latest legal
slot; `-2` = `lp + 1`, the earliest). We adopt the same convention.

---

## Quick lookup table

| # | Rule | What it catches | Which transform |
|---|------|-----------------|-----------------|
| 1 | Stage pipeline (acyclic deps) | Cyclic producer-consumer chains | both (precondition) |
| 2 | Block is well-formed | Opaque/incomplete blocks | both |
| 3 | Target not ancestor of block | Self-referential moves | both |
| 4 | Block is not output | Sinking the kernel's final store | `ComputeAt` only |
| 5a | Every consumer is under target OR in a later root-sibling | Producer sinks past one of its readers | `ComputeAt` |
| 5b | Every producer is under target OR in an earlier root-sibling | Consumer lifts above one of its writers | `ReverseComputeAt` |
| 6 | Insertion gap exists | No legal position among target's children | both |

Conditions 5 and 6 together encode the dependency-preservation guarantee. The
others rule out structurally nonsensical moves.

## Composition with other transforms

`ComputeAt` is restricted to *full coverage* of any same-dim loop the moved
subtree carries. Partial-coverage cases must `Split` the moved subtree's loop
(or the target's enclosing loop) first, so the to-be-collapsed segment matches
the target's coverage exactly.

```python
# Moved tensor_copy has its own (m, n) nest:
for m in range(16):              # 16 trips, dim=d_M
    for n in range(4):           # 4 trips, dim=d_N
        tensor_copy(...)

# Target = matmul's m-loop, which covers M with trip=8 (post-Split factor 2,8):
for m_outer in range(2):
    for m_inner in range(8):     # <-- target_loop_nid
        for n in range(4):
            for k in range(16):
                nc_matmul(...)

# Direct ReverseComputeAt rejected — moved subtree has m-trip=16, target's
# enclosing m-coverage is 8. Not full coverage.
#
# Fix: Split the moved subtree's m-loop first, into (2, 8):
for m_outer in range(2):
    for m_inner in range(8):
        for n in range(4):
            tensor_copy(...)
#
# Now ReverseComputeAt(target_loop=<m_inner of matmul>) is legal — the
# moved subtree's m_inner exactly matches the target's m_inner coverage,
# and m_outer + n stay as the moved subtree's residual outer/inner loops.
```

This keeps `ComputeAt` mechanical: identify covered dims, drop those loops
wholesale, splice the rest under the target. No region-cover arithmetic.
