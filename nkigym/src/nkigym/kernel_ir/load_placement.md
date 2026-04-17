## Load Placement in KernelIR

Load placement controls how many of a tensor-dim's tile iterations a single allocation covers. `per_tile` → one logical tile's worth; `per_block` → one block's worth; `full` → all blocks. Higher tiers grow the buffer, hoist the allocation and its DMA load outward, and reduce DMA frequency.

This doc describes the **IR representation**: the field, per-tier semantics, the feasibility constraint against `loop_order`, and how the renderer reads it. The transform that mutates the field lives in `transforms/load_placement.md`.

### Field

```python
tensor_placements: dict[tuple[str, str], str]
    # (tensor_name, dim_id) -> "per_tile" | "per_block" | "full"
```

One entry per `(tensor, dim)` where `dim` is in `tensor.dim_ids`. Same key space as `buffer_degrees`. The renderer branches on the literal string in `_compute_num_tiles` (`codegen/buffers.py`).

### Tier Semantics

Each dim `d` contributes two loops in the emitted kernel: `i_block_d` (outer) and `i_ltile_d` (inner, within its phase). A tier on `(t, d)` constrains whether these two loops sit above or below `t`'s allocation:

| tier       | `i_block_d` vs alloc | `i_ltile_d` vs alloc | buffer slots on `d`               |
|------------|----------------------|----------------------|-----------------------------------|
| `per_tile` | above                | above                | `num_ptiles`                      |
| `per_block`| above                | below                | `num_ptiles × tpb`                |
| `full`     | below                | below                | `num_ptiles × tpb × num_blocks`   |

*above alloc* → loop encloses buffer → buffer is refilled each iteration.
*below alloc* → buffer encloses loop → one fill covers all iterations of that loop.

With `num_ptiles = max_op_tile / physical_tile_size`, `tpb = ltiles_per_block[d]`, `num_blocks = dim_size / (tpb × logical_tile_size)`.

### Feasibility Against `loop_order`

The kernel emits loops in a **fixed phase order** (see `kernel_ir/data_parallel.py`, `kernel_ir/reduction.py`): within any scope (top level, or a fusion group's reduction region), *all* block loops come first in the order listed in `loop_order`, then *all* tile loops in the same order. So a tensor whose dims sit in the DP phase is enclosed by loops emitted in this pattern:

```
for i_block_{dp[0]}:
  for i_block_{dp[1]}:
    ...
    for i_ltile_{dp[0]}:
      for i_ltile_{dp[1]}:
        ...<body>
```

Reduction phases inside a fusion group follow the same block-then-tile pattern over that group's reduction dim sublist.

**Feasibility rule.** An allocation lives at *one* depth. Let $A$ be the loops the tensor's tiers require above alloc and $B$ the loops they require below. The assignment is realizable iff every loop in $A$ is emitted before every loop in $B$ under the phase-grouped order.

Concrete cases for a tensor on `(d_a, d_b)` both DP, with `loop_order` listing DP dims as `[d_a, d_b]`:

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

The only feasible mixed-tier combinations are *uniform tiers across the dims sharing a phase*. Phase-grouped emission forces all the block loops to cluster and all the tile loops to cluster, so you can't interleave tiers between dims in the same phase.

**Reordering as an escape hatch.** If a specific assignment is desired but infeasible, the only structural fix is splitting the phase — that would require loop-reordering to emit `block_da, tile_da, block_db, tile_db` instead of phase-grouped, which is a different transform outside this doc's scope.

**DP vs reduction dims.** DP loops always enclose reduction loops. A tensor's DP-dim tiers and reduction-dim tiers are therefore independent of each other for feasibility — the constraint only couples dims within the same phase. A tensor on `(dp, red)` with `(per_tile, full)` is always feasible: the DP tile loop is above alloc, the reduction loops are below, the alloc sits at the top of the innermost DP body.

### Joint Constraint with `loop_order`

1. A load-placement transform must reject tier changes that break the feasibility rule under the current `loop_order`.
2. A loop-reorder transform must reject reorderings that break an existing placement assignment.

Storage remains orthogonal — `tensor_placements` and `loop_order` are independent dicts/lists — but *legality* is a cross-field check at transform time.

### Default

```python
tensor_placements[(tensor, dim_id)] = "per_tile"
    for tensor in da.tensors
    for dim_id in tensor.dim_ids
```

All-`per_tile` puts every loop above every allocation. $B$ is empty for every tensor, feasibility is vacuous, the base IR always works.

### How the Renderer Uses It

**Buffer shape.** Tier picks `tpb_factor` and `blocks_factor` per dim; these plug into the shared num_tiles formula that also consumes `buffer_degrees`:

$$\text{num\_tiles}(t, d) = \text{num\_ptiles}(t, d) \times \text{tpb\_factor}(t, d) \times \text{blocks\_factor}(t, d) \times \text{buffer\_degrees}[(t, d)]$$

| tier        | `tpb_factor` | `blocks_factor` |
|-------------|--------------|-----------------|
| `per_tile`  | 1            | 1               |
| `per_block` | `tpb`        | 1               |
| `full`      | `tpb`        | `num_blocks`    |

**Allocation depth — derived, not stored.** The renderer walks the emission order and places each tensor's allocation at the boundary between its $A$ and $B$. For all-`per_tile`, this is the innermost body of the enclosing loop (innermost DP body for pure-DP tensors, innermost reduction body of the producing group for reduction-dim tensors).

**Load position — derived.** An HBM input's `load_tensor_block` sits at the same depth as the buffer's allocation and fills all slots implied by the buffer shape via the gadget's internal par/free loops.

### What Load Placement Is Not

- **Not multi-buffering.** Multi-buffering rotates D slots at a given tier; placement picks the tier. A `per_block` buffer with `buffer_degrees = 2` has `2 × tpb × num_ptiles` slots.
- **Not ltiles_per_block.** `tpb` changes how many logical tiles pack into one block; placement picks which loops enclose the allocation. They compose: `per_block` buffer size scales with `tpb`.
- **Not allocation addressing.** The renderer chooses allocation *depth*; SBUF/PSUM byte address is a lower-level emitter concern.

### Attention Walkthrough

Default (all `per_tile`). In the attention example's default lowering:

- DP dims: `d0` (16 blocks × 1 tile), `d4` (1 block × 1 tile).
- Reduction dims per group: `d1` (1 block × 1 tile), `d2` (4 blocks × 1 tile).

Every on-chip buffer's alloc sits at the innermost body of its enclosing loops. HBM tensors `Q (d0, d1)`, `K (d2, d1)`, `V (d2, d4)` get their `sbuf_*` staging allocated at the innermost body of the consuming group's reduction nest, and `load_tensor_block` fires there. Q alone is loaded 16 × 1 = 16 times across the `(i_block_d0, i_block_d1)` loops of group 0.

Hoisting `K` to `per_tile → full` on `d2`: `tensor_placements[("K", "d2")] = "full"`. Group 1's reduction nest orders `[d1, d2]` (loop_order sublist `["d1", "d2"]`), so under phase grouping the emitted loops are `block_d1, block_d2, tile_d1, tile_d2`. For K, $A = \{block_{d1}, tile_{d1}\}$, $B = \{block_{d2}, tile_{d2}\}$. $A$ loops are at positions 0 and 2; $B$ loops at 1 and 3. `block_d2` at 1 precedes `tile_d1` at 2 — **infeasible under the phase-grouped default**.

To make it feasible, loop-reordering must first reorder group 1's sublist to split the phase, or multi-buffering must be used instead. This is exactly the cross-field coupling the feasibility rule catches.

Hoisting `Q` to `full` on `d1` (which only needs $B = \{block_{d1}, tile_{d1}\}$, $A = \emptyset$): alloc sits above group 0's whole reduction nest. `sbuf_Q` grows to `(128, 1, 1, 128)` already since `d1` has `num_ptiles = tpb = num_blocks = 1` — shape is unchanged, but the allocation rises up the nest. This is vacuously feasible and illustrates the separation between feasibility and buffer-size impact.

### Summary

The IR field is a flat `(tensor, dim) → tier` map with three legal values. The non-trivial part — the joint constraint with `loop_order` — lives in transform-time legality checks, not in the field's shape.
