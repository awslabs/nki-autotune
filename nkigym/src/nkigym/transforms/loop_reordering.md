# Loop Reordering

*Single-loop-nest transform — operates on one fusion group's loop nest. Fusing loop nests is handled by online fusion and loop fusion.*

Loop reordering changes how a fusion group iterates over its dimensions. Two orthogonal mechanisms:

1. **Dimension permutation** reorders the dimension iteration order.
2. **Dimension interleaving** splits a dimension's block and tile loops across different nesting levels.

Both preserve correctness. Each candidate modifies a single fusion group.

## Dimension Permutation

All permutations of a group's dimensions are valid. `nc_matmul` accumulates into PSUM indexed by output position — the iteration order doesn't affect the final result.

**What varies across orderings:**

- **PSUM sizing and writeback.** PSUM holds all output tile positions that iterate inside the reduction dim's loop. Output dims outside the reduction dim are processed one at a time — their PSUM results are written back before advancing. Moving the reduction dim deeper in the nest shrinks PSUM but increases writeback frequency.
- **DMA load positions.** At single placement, each load sits at its innermost relevant dimension. Reordering changes which loads end up outside inner loops, affecting reuse.
- **HBM access pattern.** The innermost-varying dim determines DMA contiguity.
- **Load placement interaction.** Reordering changes which loops are outer, affecting which hoists are profitable.

### Example

Matmul from load_placement.md: `lhs_T(K=d0, M=d1) × rhs(K=d0, N=d2) → result(d1, d2)` with d0 (K, tile=128, 16 tiles), d1 (M, tile=128, 16 tiles), d2 (N, tile=512, 4 tiles). lhs_T depends on (d0, d1). rhs depends on (d0, d2). Result indexed by (d1, d2).

SBUF staging buffers are the same for all 6 orders (single-tile at single placement):

```python
sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 1, 1, 512), buffer=nl.sbuf)
```

**Order (d0, d1, d2)** — K outermost, both output dims inside. psum_output (128, 16, 4, 512). lhs_T at d1 (reused 4× across d2), rhs at d2:

```python
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_d0 in range(16):
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

**Order (d0, d2, d1)** — K outermost, both output dims inside. psum_output (128, 16, 4, 512). rhs at d2 (reused 16× across d1), lhs_T at d1:

```python
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_d0 in range(16):
    for i_d2 in range(4):
        load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
        for i_d1 in range(16):
            load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

**Order (d1, d0, d2)** — d1 outside K, d2 inside. psum_output (128, 1, 4, 512). lhs_T at d0 (reused 4× across d2), rhs at d2:

```python
psum_output = nl.ndarray((128, 1, 4, 512), dtype=nl.float32, buffer=nl.psum)
for i_d1 in range(16):
    nisa.memset(dst=psum_output[0:128, 0, 0:4, 0:512], value=0.0)
    for i_d0 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, 0, i_d2, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
    save_tensor_block(dst=output, src=psum_output, par_ofs=i_d1*128, free_ofs=0)
```

**Order (d1, d2, d0)** — both output dims outside K. psum_output (128, 1, 1, 512). Both loads at d0 (innermost), no reuse:

```python
psum_output = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
for i_d1 in range(16):
    for i_d2 in range(4):
        nisa.memset(dst=psum_output[0:128, 0, 0, 0:512], value=0.0)
        for i_d0 in range(16):
            load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, 0, 0, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
        save_tensor_block(dst=output, src=psum_output, par_ofs=i_d1*128, free_ofs=i_d2*512)
```

**Order (d2, d0, d1)** — d2 outside K, d1 inside. psum_output (128, 16, 1, 512). rhs at d0 (reused 16× across d1), lhs_T at d1:

```python
psum_output = nl.ndarray((128, 16, 1, 512), dtype=nl.float32, buffer=nl.psum)
for i_d2 in range(4):
    nisa.memset(dst=psum_output[0:128, 0:16, 0, 0:512], value=0.0)
    for i_d0 in range(16):
        load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
        for i_d1 in range(16):
            load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, 0, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
    save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=i_d2*512)
```

**Order (d2, d1, d0)** — both output dims outside K. psum_output (128, 1, 1, 512). Both loads at d0 (innermost), no reuse:

```python
psum_output = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
for i_d2 in range(4):
    for i_d1 in range(16):
        nisa.memset(dst=psum_output[0:128, 0, 0, 0:512], value=0.0)
        for i_d0 in range(16):
            load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, 0, 0, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
        save_tensor_block(dst=output, src=psum_output, par_ofs=i_d1*128, free_ofs=i_d2*512)
```

### Summary (single placement)

| Order | lhs_T reuse | rhs reuse | psum_output | writeback |
|---|---|---|---|---|
| (d0, d1, d2) | 4× across d2 | 1× | (128, 16, 4, 512) | after all loops |
| (d0, d2, d1) | 1× | 16× across d1 | (128, 16, 4, 512) | after all loops |
| (d1, d0, d2) | 4× across d2 | 1× | (128, 1, 4, 512) | per d1 |
| (d1, d2, d0) | 1× | 1× | (128, 1, 1, 512) | per (d1, d2) |
| (d2, d0, d1) | 1× | 16× across d1 | (128, 16, 1, 512) | per d2 |
| (d2, d1, d0) | 1× | 1× | (128, 1, 1, 512) | per (d2, d1) |

The PSUM shape reflects which output dims iterate inside d0 (K). Dims inside d0 contribute their full tile count; dims outside contribute 1. Orders with K outermost use more PSUM but write back once; orders with K innermost use minimal PSUM but write back per output-tile.

Orders pair by reuse pattern: {(d0,d1,d2), (d1,d0,d2)} both get 4× lhs_T reuse; {(d0,d2,d1), (d2,d0,d1)} both get 16× rhs reuse; {(d1,d2,d0), (d2,d1,d0)} get no reuse. Within each pair, the K position differs — the order with K outermost uses larger PSUM but writes back once, while the order with K deeper uses less PSUM at the cost of more frequent writebacks.

## Dimension Interleaving

When tiles_per_block splits a dimension into multiple blocks, its block and tile loops can be separated with other dimensions' loops in between. This creates a **section** structure: each block iteration processes a chunk of the dimension across all enclosed iterations, with partial buffers for save/reload between sections.

### Mechanism

1. **Shrink `psum_output`.** Output dims that move between block and tile loops reduce to 1 — PSUM holds one tile position at a time.
2. **Allocate `sbuf_output`.** An SBUF buffer matching the pre-shrink PSUM shape (all output tile positions), zero-initialized via `nisa.memset`. This buffer persists partial sums across block iterations.
3. **Split the loops.** The dim's `i_block` loop stays at its position. The remaining phase loops (`i_psum_batch`, `i_tile`, `i_ig`) move deeper by `depth` positions past subsequent dims' loops.
4. **Move `tensor_copy(psum → sbuf)`.** The writeback moves from inside `save_tensor_block` to after the tile loop, saving each tile's accumulated result into `sbuf_output`.
5. **Insert `tensor_copy(sbuf → psum)`.** Before the tile loop: reload the partial sum from `sbuf_output` into `psum_output`. First reload gets zeros; subsequent reloads continue accumulation from the previous block.
6. **Update `save_tensor_block`.** src changes from `psum_output` to `sbuf_output` — data is already in SBUF after the per-iteration saves.

### Requirements

- **`num_blocks > 1`** on the interleaved dim. With 1 block, the block loop is `range(1)` and splitting is a no-op. The tiles_per_block transform creates multi-block structure.
- **Accumulating semantics.** Every op reducing over this dim must use additive accumulation. `nc_matmul` accumulates via `+=` — valid. `tensor_reduce(op="max")` is NOT additive — requires online fusion to convert to accumulating form first.

### Interleave depths

For the matmul with order `(d0, d1, d2)` and `tiles_per_block_d0 = 4` (`num_blocks_d0 = 4`):

| Depth | Loop nest | PSUM size | `sbuf_output` |
|---|---|---|---|
| 0 (default) | `d0_blk, d0_tile → d1 → d2` | (128, 16, 4, 512) | none |
| 1 | `d0_blk → d1 → d0_tile → d2` | (128, 1, 4, 512) | (128, 16, 4, 512) |
| 2 | `d0_blk → d1 → d2 → d0_tile` | (128, 1, 1, 512) | (128, 16, 4, 512) |

Deeper interleaving reduces PSUM usage (the reload/save narrows the accumulation window) but adds save/reload overhead. The `sbuf_output` size is the same at all non-zero depths — it must cover all output positions that iterate between block and tile. What changes is the PSUM size per tile and which operand loads can be placed between the split loops for cross-iteration reuse.

### Example

Same matmul, now with `tiles_per_block_d0 = 4` (4 blocks of 4 tiles each). Order `(d0, d1, d2)`. Staging buffers remain single-tile with per-tile loads to show the loop structure explicitly; in rendered code, tiles_per_block also grows buffers and coalesces loads to block level (see tiles_per_block.md).

**Before** — d0 not interleaved (depth 0):

```python
sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 1, 1, 512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_block_d0 in range(4):
    for i_tile_d0 in range(4):
        for i_d1 in range(16):
            load_tensor_block(dst=sbuf_lhs_T, src=lhs_T,
                par_ofs=(i_block_d0*4 + i_tile_d0)*128, free_ofs=i_d1*128)
            for i_d2 in range(4):
                load_tensor_block(dst=sbuf_rhs, src=rhs,
                    par_ofs=(i_block_d0*4 + i_tile_d0)*128, free_ofs=i_d2*512)
                nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

**After** — d0 interleaved at depth 2 (`interleave_depth = {"d0": 2}`):

```python
sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 1, 1, 512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
sbuf_output = nl.ndarray((128, 16, 4, 512), dtype=output.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_block_d0 in range(4):
    for i_d1 in range(16):
        for i_d2 in range(4):
            nisa.tensor_copy(dst=psum_output[0:128, 0, 0, 0:512],
                src=sbuf_output[0:128, i_d1, i_d2, 0:512])
            for i_tile_d0 in range(4):
                load_tensor_block(dst=sbuf_lhs_T, src=lhs_T,
                    par_ofs=(i_block_d0*4 + i_tile_d0)*128, free_ofs=i_d1*128)
                load_tensor_block(dst=sbuf_rhs, src=rhs,
                    par_ofs=(i_block_d0*4 + i_tile_d0)*128, free_ofs=i_d2*512)
                nisa.nc_matmul(dst=psum_output[0:128, 0, 0, 0:512],
                    stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
            nisa.tensor_copy(dst=sbuf_output[0:128, i_d1, i_d2, 0:512],
                src=psum_output[0:128, 0, 0, 0:512])
save_tensor_block(dst=output, src=sbuf_output, par_ofs=0, free_ofs=0)
```

Mechanical steps applied to the before:

1. **Shrink `psum_output`**: dims between block and tile (d1, d2) reduce to 1 → (128, 1, 1, 512)
2. **Allocate `sbuf_output`**: shape matches pre-shrink PSUM (128, 16, 4, 512), zero-initialized via `memset`
3. **Split loops**: `i_tile_d0` moves past 2 subsequent dims (d1, d2)
4. **Move `tensor_copy(psum → sbuf)`**: from inside `save_tensor_block` to after `i_tile_d0` loop
5. **Insert `tensor_copy(sbuf → psum)`**: reload before `i_tile_d0` loop
6. **Update `save_tensor_block`**: src changes from `psum_output` to `sbuf_output`

Operands loaded between `i_block_d0` and `i_tile_d0` persist across all 64 enclosed (d1 × d2) iterations per section. Load placement exploits this by hoisting loads into that gap.

**Fused groups.** In a fused group, a dim may be output for some ops and reduction for others. When interleaved, only the accumulating ops need partial buffer save/reload. Non-accumulating ops produce complete results per tile iteration, consumed within the same iteration by downstream ops.

## Representation and Candidates

```python
group_dim_orders: list[tuple[str, ...]]
interleave_depth: list[dict[str, int]]
```

`group_dim_orders` — per fusion group, ordered tuple of all dim IDs. Permutation modifies this ordering.

`interleave_depth` — per fusion group, maps dim ID → depth (number of subsequent dims between block and tile loops). Depth 0 = block and tile adjacent (default). Max depth for a dim at position $p$ is $n - p - 1$.

```python
class LoopReorder(Transform):
    NAME = "loop_reorder"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        for gidx, dim_order in enumerate(ir.group_dim_orders):
            """Dimension permutations (all dims)"""
            if len(dim_order) > 1:
                for perm in permutations(dim_order):
                    if perm != dim_order:
                        new_orders = list(ir.group_dim_orders)
                        new_orders[gidx] = perm
                        results.append(replace(ir, group_dim_orders=new_orders))
            """Dimension interleaving (accumulating dims with num_blocks > 1)"""
            for dim_id in dim_order:
                if not ir.is_accumulating(dim_id, gidx):
                    continue
                unified = ir.ctx.dim_sizes[dim_id] // ir.ctx.dim_tiles[dim_id]
                if unified // ir.tiles_per_block.get(dim_id, 1) <= 1:
                    continue
                pos = dim_order.index(dim_id)
                max_depth = len(dim_order) - pos - 1
                current = ir.interleave_depth[gidx].get(dim_id, 0)
                for depth in range(max_depth + 1):
                    if depth != current:
                        new_depths = [dict(d) for d in ir.interleave_depth]
                        new_depths[gidx][dim_id] = depth
                        results.append(replace(ir, interleave_depth=new_depths))
        return results
```
