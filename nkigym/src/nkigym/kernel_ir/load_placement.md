## Load Placement in KernelIR

Load placement controls how many of a tensor-dim's tile iterations a single allocation covers. `per_tile` → one logical tile's worth; `per_block` → one block's worth; `full` → all blocks. Higher tiers grow the buffer, hoist the allocation and its DMA load outward, and reduce DMA frequency.

This doc describes the **IR representation**: the field, per-tier semantics, the per-group feasibility constraint against `group_dim_orders`, and how the renderer reads it. The transform that mutates the field lives in `transforms/load_placement.md`.

### Field

```python
tensor_placements: dict[tuple[str, str], str]
    # (tensor_name, dim_id) -> "per_tile" | "per_block" | "full"
```

One entry per `(tensor, dim)` where `dim` is in `tensor.dim_ids`. Same key space as `buffer_degrees`. The renderer branches on the literal string in `_compute_num_tiles` (`codegen/buffers.py`).

A tensor is relevant to every fusion group that contains an op producing or consuming it. A tier on `(t, d)` affects the buffer shape and the DMA placement depth in *every* group that touches `t`.

### Tier Semantics

Each dim `d` in a group's `dim_order` contributes two loops in the group's nest: `i_block_d` (outer) and `i_ltile_d` (inner, within its phase). A tier on `(t, d)` constrains whether these two loops sit above or below `t`'s allocation in every group where `t` appears:

| tier       | `i_block_d` vs alloc | `i_ltile_d` vs alloc | buffer slots on `d`               |
|------------|----------------------|----------------------|-----------------------------------|
| `per_tile` | above                | above                | `num_ptiles`                      |
| `per_block`| above                | below                | `num_ptiles × tpb`                |
| `full`     | below                | below                | `num_ptiles × tpb × num_blocks`   |

*above alloc* → loop encloses buffer → buffer is refilled each iteration.
*below alloc* → buffer encloses loop → one fill covers all iterations of that loop.

With `num_ptiles = max_op_tile / physical_tile_size`, `tpb = ltiles_per_block[d]`, `num_blocks = dim_size / (tpb × logical_tile_size)`.

### Per-Group Feasibility

Every fusion group emits its own complete loop nest (see `codegen/group_loops.py`). Within a group's nest, loops are emitted in **phase-grouped order**: all block loops first in the order listed in that group's `group_dim_orders` entry, then all tile loops in the same order. A dim's blocking status (`DimInfo.is_blocking`) only gates fusion legality — once a dim is in a group's `dim_order`, its loops emit identically regardless of blocking.

So within group `g`, a tensor `t` whose dims `(d_a, d_b)` both appear in `group_dim_orders[g] = [..., d_a, ..., d_b, ...]` is enclosed by:

```
for i_block_{d_a}:
  for i_block_{d_b}:
    for i_ltile_{d_a}:
      for i_ltile_{d_b}:
        ...<body>
```

**Feasibility rule (per group).** An allocation lives at *one* depth in each group's nest. Let $A_g$ be the loops the tensor's tiers require above alloc and $B_g$ the loops they require below — restricted to the dims in group `g`'s `dim_order`. The assignment is realizable in group `g` iff every loop in $A_g$ is emitted before every loop in $B_g$ under that group's phase-grouped order. The full assignment is feasible iff it is feasible in *every* group that touches `t`.

Concrete cases within a single group whose `dim_order` lists `[d_a, d_b]`:

| tiers `(d_a, d_b)`         | A (above)                                  | B (below)                                  | feasible? |
|----------------------------|--------------------------------------------|--------------------------------------------|-----------|
| `(per_tile, per_tile)`     | all four loops                             | —                                          | yes (alloc innermost) |
| `(per_block, per_block)`   | `block_da, block_db`                       | `tile_da, tile_db`                         | yes (alloc between phases) |
| `(full, full)`             | —                                          | all four loops                             | yes (alloc outermost) |
| `(per_block, per_tile)`    | `block_da, block_db, tile_db`              | `tile_da`                                  | no — `tile_da` precedes `tile_db` |
| `(per_tile, per_block)`    | `block_da, block_db, tile_da`              | `tile_db`                                  | no — `tile_db` follows `tile_da` |
| `(full, per_tile)`         | `block_db, tile_db`                        | `block_da, tile_da`                        | no — `block_da` precedes `block_db` |
| `(full, per_block)`        | `block_db`                                 | `block_da, tile_da, tile_db`               | no — `block_da` precedes `block_db` |
| `(per_tile, full)`         | `block_da, tile_da`                        | `block_db, tile_db`                        | no — `block_db` precedes `tile_da` |
| `(per_block, full)`        | `block_da`                                 | `block_db, tile_da, tile_db`               | no — `block_db` precedes `tile_da` |

The only feasible mixed-tier combinations are *uniform tiers across the dims sharing a phase within a single group*. Phase-grouped emission forces all block loops to cluster and all tile loops to cluster, so you can't interleave tiers between dims in the same phase.

A dim the tensor doesn't carry contributes no loops to that tensor's $A_g$ or $B_g$ — its tier doesn't exist. Other tensors' dim loops in the same group still emit around the alloc based on *their own* tiers; the feasibility rule is per-tensor.

**Reordering as an escape hatch.** If a specific assignment is desired but infeasible in some group, the structural fix is to reorder that group's `dim_order` to split the phase — a loop-reordering transform outside this doc's scope.

### Joint Constraint with `group_dim_orders`

1. A load-placement transform must reject tier changes that break the feasibility rule in *any* group where the tensor appears.
2. A loop-reorder transform on a group must reject reorderings that break an existing placement assignment on any tensor touched by that group.

Storage remains orthogonal — `tensor_placements` and `group_dim_orders` are independent dict/list fields — but *legality* is a cross-field check at transform time.

### Default

```python
tensor_placements[(tensor, dim_id)] = "per_tile"
    for tensor in da.tensors
    for dim_id in tensor.dim_ids
```

All-`per_tile` puts every loop above every allocation. $B_g$ is empty for every tensor in every group, feasibility is vacuous, the base IR always works.

### How the Renderer Uses It

**Buffer shape.** Tier picks `tpb_factor` and `blocks_factor` per dim; these plug into the shared num_tiles formula that also consumes `buffer_degrees`:

$$\text{num\_tiles}(t, d) = \text{num\_ptiles}(t, d) \times \text{tpb\_factor}(t, d) \times \text{blocks\_factor}(t, d) \times \text{buffer\_degrees}[(t, d)]$$

| tier        | `tpb_factor` | `blocks_factor` |
|-------------|--------------|-----------------|
| `per_tile`  | 1            | 1               |
| `per_block` | `tpb`        | 1               |
| `full`      | `tpb`        | `num_blocks`    |

**Allocation depth — derived, not stored.** The renderer walks each group's emission order and places the tensor's allocation at the boundary between its $A_g$ and $B_g$ within that group. For all-`per_tile`, this is the innermost body of the group's nest.

**Load position — derived.** An HBM input's `load_tensor_block` sits at the same depth as the buffer's allocation in each group that consumes it and fills all slots implied by the buffer shape via the gadget's internal par/free loops.

### What Load Placement Is Not

- **Not multi-buffering.** Multi-buffering rotates D slots at a given tier; placement picks the tier. A `per_block` buffer with `buffer_degrees = 2` has `2 × tpb × num_ptiles` slots.
- **Not ltiles_per_block.** `tpb` changes how many logical tiles pack into one block; placement picks which loops enclose the allocation. They compose: `per_block` buffer size scales with `tpb`.
- **Not allocation addressing.** The renderer chooses allocation *depth*; SBUF/PSUM byte address is a lower-level emitter concern.

### Attention Walkthrough

Default (all `per_tile`). From the attention example (see `codegen/loopnest.md` for the op graph):

- `d0` (16 blocks × 1 tile) — non-blocking.
- `d1` (1 block × 1 tile) — blocking (matmul K).
- `d2` (4 blocks × 1 tile) — blocking (softmax / matmul K).
- `d4` (1 block × 1 tile) — non-blocking.

Blocking status doesn't affect feasibility — only each group's `dim_order` does.

Every on-chip buffer's alloc sits at the innermost body of its group's nest. HBM tensors `Q(d0, d1)`, `K(d2, d1)`, `V(d2, d4)` get their `sbuf_*` staging buffers sized by `num_tiles = num_ptiles` on every axis and allocated at the innermost body of each consuming group.

Hoisting `K` to `per_tile → full` on `d2`: `tensor_placements[("K", "d2")] = "full"`. K appears in group 1 (K transpose) with `dim_order = [d1, d2]` and group 2 (QK matmul) with `dim_order = [d0, d1, d2]`.

- Group 1's nest under phase grouping: `block_d1, block_d2, tile_d1, tile_d2`. For K, $A_1 = \{block_{d1}, tile_{d1}\}$, $B_1 = \{block_{d2}, tile_{d2}\}$. $A_1$ loops at positions 0 and 2; $B_1$ at 1 and 3. `block_d2` at 1 precedes `tile_d1` at 2 — **infeasible in group 1** under the phase-grouped default.
- Group 2 has the same K-relevant dims in the same relative order, so it is independently infeasible for the same reason.

The assignment is rejected by the transform because it fails in at least one group. To make it feasible, loop-reordering must first reorder `group_dim_orders[1]` (and/or `[2]`) to split the phase, or multi-buffering must be used instead. This is exactly the cross-field coupling the feasibility rule catches.

Hoisting `Q` to `full` on `d1` (`tensor_placements[("Q", "d1")] = "full"`): Q appears in group 0 (Q transpose, `dim_order = [d0, d1]`) and group 2 (QK matmul, `dim_order = [d0, d1, d2]`). In both groups, Q has $A = \{$loops on d0$\}$ and $B = \{block_{d1}, tile_{d1}\}$. All non-d1 loops on Q are `per_tile`, so they stay in $A_g$; d1's loops move to $B_g$. Under phase grouping, the d0 loops precede the d1 loops in each relevant group — feasible. `sbuf_Q`'s shape is unchanged since `d1` has `num_ptiles = tpb = num_blocks = 1`, but the allocation rises up each group's nest. This is vacuously feasible and illustrates the separation between feasibility and buffer-size impact.

### Summary

The IR field is a flat `(tensor, dim) → tier` map with three legal values. The non-trivial part — the per-group feasibility constraint against each group's `dim_order` — lives in transform-time legality checks, not in the field's shape.
