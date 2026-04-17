## Data-Parallel Loops

After the header, `render_ir` emits loops for the data-parallel dimensions. A dimension is data-parallel when `DimInfo.is_data_parallel` is True — it appears in the kernel's return tensor. Every other dimension is a reduction dimension.

Each dimension contributes **2 loops at the kernel level**: block and logical tile. Both are always emitted, even when trip count is 1. Trip counts come from `DimInfo` and `ltiles_per_block`:

| Loop | Variable | Trip count |
|---|---|---|
| Block | `i_block_d{id}` | `dim_size / (ltiles_per_block * logical_tile_size)` |
| Logical tile | `i_ltile_d{id}` | `ltiles_per_block` |

**Physical-tile iteration is per-op, not part of the kernel nest.** Each NKIOp knows its own op tile size on each dim (`da.op_tile_sizes[op_idx][dim_id]`) and the dim's `physical_tile_size`. How many physical tiles make up one logical tile varies by op:

- If the op covers the full logical tile in one ISA call (e.g. `nc_matmul` with N=512 on a dim whose logical tile is 512), the op does no inner iteration.
- If the op's tile is smaller than the logical tile (e.g. `nc_transpose` with F=128 on a 512-logical dim), the op internally iterates `num_ptiles_per_ltile = logical_tile_size / op_tile_size` times.

This physical-tile packing is encapsulated inside the op's emission. Following the pattern we already use for DMA (`load_tensor_block`, `store_tensor_block`), a per-op gadget should hide the inner ptile loop so the kernel source stays at two visible levels per dim.

Loops are grouped by phase — all block loops outermost, then all logical tile loops. Within each phase, dimension order is taken from `loop_order`. `loop_order` is a single flat list for the whole kernel: top-level string entries are DP dimensions in outer-to-inner order, and each nested `list[str]` is one fusion group's reduction dim IDs in outer-to-inner order, positional on `fusion_groups`. An empty sublist marks a group with no reduction dims. For attention with one fusion group per op, the default is `["d0", "d4", ["d1"], ["d1", "d2"], ["d2"], ["d2"], ["d2"], ["d2"], [], ["d2"], ["d2"], []]` — DP dims `d0, d4` on the outside, one reduction sublist per fusion group. Block loops define the data boundary (DMA loads happen here); logical tile loops iterate within a block.

### Example: Attention DP Loops

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`, `ltiles_per_block = 1`:

| Dim | dim_size | logical_tile_size | physical_tile_size | DP/reduction | block | tile |
|---|---|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | DP | 16 | 1 |
| d1 | 128 | 128 | 128 | reduction | — | — |
| d2 | 2048 | 512 | 128 | reduction | — | — |
| d4 | 128 | 128 | 128 | DP | 1 | 1 |

d1 and d2 are reduction dimensions — not emitted here. d0 and d4 are data-parallel:

```python
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_ltile_d0 in range(1):
            for i_ltile_d4 in range(1):
                ...
```

## Reduction Loops

Inside the innermost DP loop, the `...` placeholder is replaced by reduction content. The compute graph is a DAG, so the reduction region is a sequence of **sibling blocks** (one per fusion group), not a single deep nest. Each group runs its reduction to completion before the next group starts.

**Group ordering.** Lift `op_graph.edges` to group level: for each edge `(producer, consumer, tensor, role)`, find the producer's group and the consumer's group; if they differ, add a directed edge between groups. Topologically sort; ties broken by minimum `op_idx` in each group.

**Per-group reduction dims.** For each op in a group, collect all input and output tensor names from `op_graph.op_tensors[op_idx]`. Union their `dim_ids` (from `dim_analysis.tensors`) across all ops in the group, subtract the data-parallel dims. The remainder is the group's reduction dims.

**Per-group loop structure.** Same 2-phase block + logical-tile pattern and trip-count formulas as the DP loops, applied to the group's reduction dims. The group's reduction dims come from the nested sublist in `loop_order` at index `num_dp_entries + group_idx` (i.e. positional on `fusion_groups`). The sublist is filtered to the dims this group actually touches, preserving its relative order. `ltiles_per_block` is per-dimension (`ir.ltiles_per_block[dim_id]`) — the same block structure applies to every op and tensor on that dim, matching the reference attention kernel where K (tile 512) and V (tile 128) differ in per-op tile size on seqlen_k but share the same block iteration on that dim. Physical-tile packing within a logical tile is handled inside each op's gadget.

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

| Dim | dim_size | logical_tile_size | physical_tile_size | block | tile |
|---|---|---|---|---|---|
| d1 | 128 | 128 | 128 | 1 | 1 |
| d2 | 2048 | 512 | 128 | 4 | 1 |

Inside the innermost DP loop, the eleven groups emit as sibling blocks. Groups 7 and 10 have no reduction dims — no loops emitted:

```python
"""inside innermost DP loop"""

# Group 0: nc_transpose Q [reduction: d1]
for i_block_d1 in range(1):
    for i_ltile_d1 in range(1):
        ...

# Group 1: nc_transpose K [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                ...

# Group 2: nc_matmul QK [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                ...

# Group 3: affine_select [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 4: tensor_scalar scale [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 5: tensor_reduce max [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 6: activation_reduce exp+sum [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 7: activation reciprocal [reduction: (none)]
...

# Group 8: nc_transpose exp_S [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 9: nc_matmul SV [reduction: d2]
for i_block_d2 in range(4):
    for i_ltile_d2 in range(1):
        ...

# Group 10: tensor_scalar scale output [reduction: (none)]
...
```

Inside each innermost `i_ltile_*` body, the op's own gadget handles physical-tile packing on its dim(s). `nc_matmul` (groups 2, 9) covers the full d2 logical tile of 512 in a single ISA call — no ptile iteration. `nc_transpose` on d2 (groups 1, 8) only covers 128 per call, so the transpose gadget internally loops `num_ptiles_per_ltile = 4` times. Vector-engine ops on d2 (groups 3–6) behave the same way as transpose — the gadget packs 4 physical tiles per logical tile.

This is the default lowering — each group gets its own independent reduction loops. The reference kernel is the result of applying transforms (loop fusion, online softmax) on top of this baseline.
