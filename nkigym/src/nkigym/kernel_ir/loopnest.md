## Data-Parallel Loops

After the header, `render_ir` emits loops for the data-parallel dimensions. A dimension is data-parallel when `DimInfo.is_data_parallel` is True — it appears in the kernel's return tensor. Every other dimension is a reduction dimension.

Each dimension contributes 3 loops: block, tile, and physical tile. All three are always emitted, even when trip count is 1. Trip counts come from `DimInfo` and `tiles_per_block`:

| Loop | Variable | Trip count |
|---|---|---|
| Block | `i_block_d{id}` | `dim_size / (tiles_per_block * logical_tile_size)` |
| Tile | `i_tile_d{id}` | `tiles_per_block` |
| Physical tile | `i_ptile_d{id}` | `num_physical_tiles_per_logical_tile` |

Loops are grouped by phase — all block loops outermost, then all tile loops, then all physical tile loops. Within each phase, data-parallel dimensions are sorted by dimension ID (fixed ordering — DP loops wrap all groups, so `loop_order` does not apply here). Block loops define the data boundary (DMA loads happen here), tile loops iterate within a block, and physical tile loops handle sub-tile iteration when ops have different tile size limits on the same dimension.

### Example: Attention DP Loops

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`, `tiles_per_block = 1`:

| Dim | dim_size | logical_tile_size | physical_tile_size | DP/reduction | block | tile | ptile |
|---|---|---|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | DP | 16 | 1 | 1 |
| d1 | 128 | 128 | 128 | reduction | — | — | — |
| d2 | 2048 | 512 | 128 | reduction | — | — | — |
| d4 | 128 | 128 | 128 | DP | 1 | 1 | 1 |

d1 and d2 are reduction dimensions — not emitted here. d0 and d4 are data-parallel:

```python
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_tile_d0 in range(1):
            for i_tile_d4 in range(1):
                for i_ptile_d0 in range(1):
                    for i_ptile_d4 in range(1):
                        ...
```

## Reduction Loops

Inside the innermost DP loop, the `...` placeholder is replaced by reduction content. The compute graph is a DAG, so the reduction region is a sequence of **sibling blocks** (one per fusion group), not a single deep nest. Each group runs its reduction to completion before the next group starts.

**Group ordering.** Lift `op_graph.edges` to group level: for each edge `(producer, consumer, tensor, role)`, find the producer's group and the consumer's group; if they differ, add a directed edge between groups. Topologically sort; ties broken by minimum `op_idx` in each group.

**Per-group reduction dims.** For each op in a group, collect all input and output tensor names from `op_graph.op_tensors[op_idx]`. Union their `dim_ids` (from `dim_analysis.tensors`) across all ops in the group, subtract the data-parallel dims. The remainder is the group's reduction dims.

**Per-group loop structure.** Same phase-grouped pattern and trip count formulas as the DP loops, applied to the group's reduction dims. Within each phase, reduction dims are ordered by `loop_order[group_idx]` (filtered to reduction dims, preserving relative order). `tiles_per_block` is read from `ir.tiles_per_block[(op_idx, dim_id)]` using the first op in the group.

Groups with no reduction dims emit no loops — the body sits directly at the DP indentation level.

### Example: Attention Reduction Loops

Op graph (11 ops):

```
[0] transpose Q ──→ [2] matmul QK ──→ [3] affine_select ──→ [4] tensor_scalar ─┬→ [5] tensor_reduce ─┐
[1] transpose K ──↗                                                              └→ [6] act_reduce ←───┘
                                                                                      ├→ [7] activation ──→ [10] tensor_scalar
                                                                                      └→ [8] transpose ──→ [9] matmul SV ──↗
```

Eleven fusion groups (initial): `[[0], [1], ..., [10]]`.

| Group | Op | Reduction dims |
|---|---|---|
| 0 | nc_transpose (Q) | {d1} |
| 1 | nc_transpose (K) | {d1, d2} |
| 2 | nc_matmul (QK) | {d1, d2} |
| 3 | affine_select (mask) | {d2} |
| 4 | tensor_scalar (scale) | {d2} |
| 5 | tensor_reduce (max) | {d2} |
| 6 | activation_reduce (exp+sum) | {d2} |
| 7 | activation (reciprocal) | (none) |
| 8 | nc_transpose (exp_S) | {d2} |
| 9 | nc_matmul (SV) | {d2} |
| 10 | tensor_scalar (scale output) | (none) |

**Group-level DAG:** `0→2`, `1→2`, `2→3`, `3→4`, `4→5`, `4→6`, `5→6`, `6→7`, `6→8`, `8→9`, `7→10`, `9→10`. Topological order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

Reduction dim trip counts:

| Dim | dim_size | logical_tile_size | physical_tile_size | block | tile | ptile |
|---|---|---|---|---|---|---|
| d1 | 128 | 128 | 128 | 1 | 1 | 1 |
| d2 | 2048 | 512 | 128 | 4 | 1 | 4 |

Inside the innermost DP loop, the eleven groups emit as sibling blocks. Groups 7 and 10 have no reduction dims — no loops emitted:

```python
"""inside innermost DP loop"""

# Group 0: nc_transpose Q [reduction: d1]
for i_block_d1 in range(1):
    for i_tile_d1 in range(1):
        for i_ptile_d1 in range(1):
            ...

# Group 1: nc_transpose K [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ptile_d1 in range(1):
                    for i_ptile_d2 in range(4):
                        ...

# Group 2: nc_matmul QK [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ptile_d1 in range(1):
                    for i_ptile_d2 in range(4):
                        ...

# Group 3: affine_select [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 4: tensor_scalar scale [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 5: tensor_reduce max [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 6: activation_reduce exp+sum [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 7: activation reciprocal [reduction: (none)]
...

# Group 8: nc_transpose exp_S [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 9: nc_matmul SV [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ptile_d2 in range(4):
            ...

# Group 10: tensor_scalar scale output [reduction: (none)]
...
```

This is the default lowering — each group gets its own independent reduction loops. The reference kernel is the result of applying transforms (loop fusion, online softmax) on top of this baseline.
