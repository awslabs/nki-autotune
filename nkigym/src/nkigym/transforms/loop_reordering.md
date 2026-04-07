# Loop Reordering

Loop reordering changes a fusion group's dimension iteration order to improve data reuse and DMA efficiency. Two capabilities compose the search space: **dimension permutation** reorders non-blocking output dims, and **dimension interleaving** hoists a blocking dim's block loop above some or all output dims while keeping its tile loop inside. Both preserve correctness; each candidate changes a single group.

## Dimension Permutation

Non-blocking output dimensions in `group_dim_orders` (the per-group ordered list of non-blocking output dim IDs) can be freely permuted. The transform produces one candidate for each non-identity permutation per group.

**All permutations are valid.** Three structural properties guarantee this without per-candidate checking:

1. **Same-dimension phase ordering.** Permutation reorders dimensions, not phases (`i_block` → `i_tile` → `i_ig`) within a dimension.

2. **Reduction dims innermost.** PSUM accumulator lifecycle (memset → K loop → writeback) requires reduction dims enclosed by output dims. `group_dim_orders` never contains blocking dims.

3. **Fused groups: non-blocking invariant.** All dims in `group_dim_orders` are non-blocking for every op. In the fused double matmul: d1, d2 excluded as reduction dims, giving `group_dim_orders = (d0, d4)`.

**What varies across orderings:**

- **Data reuse.** Operands not depending on the innermost dim are loop-invariant there, loaded once and reused. Ordering `(d2, d0)` loads each K_t block once and reuses across 16 d0 blocks; `(d0, d2)` gives Q_t reuse instead.
- **HBM access pattern.** The innermost-varying dim determines DMA contiguity. Aligning it with row-major layout improves bandwidth.
- **Interaction with load placement.** Reordering changes which loops are outer, changing which hoists are profitable and at what SBUF cost.

**Before** — default order `(d0, d2)`, output dims only (d1 is reduction, always innermost):

```python
""" Op 2: Q_t x K_t -> S(d0, d2), default order: (d0, d2) """
for i_block_d0 in nl.affine_range(16):                          # d0 outermost
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(4):                   # d2 inner
            for i_tile_d2 in nl.affine_range(1):
                psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for i_block_d1 in nl.affine_range(1):           # d1 reduction
                    for i_tile_d1 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

**After** — reordered to `(d2, d0)`:

```python
""" Op 2: Q_t x K_t -> S(d0, d2), reordered: (d2, d0) """
for i_block_d2 in nl.affine_range(4):                           # d2 outermost
    for i_tile_d2 in nl.affine_range(1):
        for i_block_d0 in nl.affine_range(16):                  # d0 inner
            for i_tile_d0 in nl.affine_range(1):
                psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for i_block_d1 in nl.affine_range(1):           # d1 reduction
                    for i_tile_d1 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

With `d2` outermost, each K_t tile is reused across all 16 d0 iterations — matching the reference attention kernel's standalone matmul phases.

## Dimension Interleaving

Dimension permutation only reorders non-blocking dims. **Dimension interleaving** goes further: it hoists a blocking dim's block loop above some or all output dims while keeping its tile loop inside, splitting the dimension's loops across different nesting levels.

**Sequential** (default — blocking dim fully inside output dims):

```python
for d0:                            # output dim
    psum_acc = memset(0)
    for d2_block:                  # blocking dim — entirely inside d0
        for d2_tile:
            [accumulate into psum_acc]
    writeback psum_acc
```

**Interleaved** (d2 block loop hoisted above d0, tile loop stays inside):

```python
for d2_block:                      # blocking dim block — hoisted above d0
    for d0:                        # output dim — inside d2_block
        psum_acc = reload(partial[d0])
        for d2_tile:               # blocking dim tile — stays inside d0
            [accumulate into psum_acc]
        save(psum_acc → partial[d0])
```

**Why valid for accumulating dims.** `nc_matmul` uses `+=` semantics, adding to whatever PSUM holds. Reloading a partial sum via `tensor_copy` lets accumulation continue across blocks — the result is identical because addition is associative ($K = B \times T$ iterations split into $B$ blocks of $T$).

Pre-zeroing the SBUF partial buffer avoids conditional logic within `nl.affine_range` — the first reload loads zero (equivalent to memset), subsequent reloads load the running partial sum. When the partial buffer exceeds SBUF capacity, it can go through HBM instead (DMA store/reload each iteration). The reference attention kernel uses HBM for output accumulation between sections, while keeping small per-Q-group accumulators (running_max, running_sum) in SBUF.

```python
sbuf_partial = nl.ndarray((128, 16, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_partial[...], value=0.0)

for i_block_d2 in nl.affine_range(num_blocks_d2):               # section
    for i_block_d0 in nl.affine_range(16):                       # Q-group
        psum_acc = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_copy(dst=psum_acc[0:128, 0:128],             # reload partial
            src=sbuf_partial[0:128, i_block_d0, 0, 0:128])
        for i_tile_d2 in nl.affine_range(tiles_per_block_d2):    # d2 tile
            nisa.nc_matmul(dst=psum_acc[...], ...)               # accumulates via +=
        nisa.tensor_copy(dst=sbuf_partial[0:128, i_block_d0, 0, 0:128],
            src=psum_acc[0:128, 0:128])                          # save partial

for i_block_d0 in nl.affine_range(16):                           # writeback
    nisa.dma_copy(dst=output[...], src=sbuf_partial[0:128, i_block_d0, 0, 0:128])
```

**Requirements:**

- `tiles_per_block > 1` on the interleaved dim — otherwise there is no block/tile split to separate
- The dim must be **accumulating**: the op reducing over this dim adds each iteration's contribution to a running total via `+=` semantics (`nc_matmul` accumulates into PSUM). Non-accumulating blocking dims (reduce_max, reduce_sum) need online fusion to convert to accumulating form before interleaving applies

**Benefits and costs.** Data loaded between block and tile loops persists across all enclosed output-dim iterations, enabling cross-output-dim reuse via load placement. In the fused double matmul, K/V between d2's block and tile loops are reused across 16 d0 Q-groups per section (16x DMA reduction). The cost is extra `tensor_copy` per (block, output-dim) for save/reload, plus a partial buffer with $\prod_{i \geq k} \text{num\_blocks}(d_i)$ slots for a level-$k$ interleave.

**Example.** Fused double matmul `[[0,1,2,3,4]]` with `tiles_per_block_d2 = 2`. Interleaving d2 above d0 produces the section pattern:

```python
""" Fused double matmul — d2 interleaved above d0 (section pattern) """
sbuf_partial = nl.ndarray((128, 16, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_partial[...], value=0.0)

for i_block_d2 in nl.affine_range(2):                            """ section """
    """ K, V loaded here — reused across all 16 d0 iterations """
    for i_block_d0 in nl.affine_range(16):                       """ Q-group """
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d4 in nl.affine_range(1):
                for i_tile_d4 in nl.affine_range(1):
                    psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                    nisa.tensor_copy(dst=psum_output[0:128, 0:128],
                        src=sbuf_partial[0:128, i_block_d0, 0, 0:128])
                    for i_tile_d2 in nl.affine_range(2):         """ d2 tile (inside d0) """
                        psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                        nisa.memset(dst=psum_S[0:128, 0:512], value=0.0)
                        for i_block_d1 in nl.affine_range(1):
                            for i_tile_d1 in nl.affine_range(1):
                                """ Ops 0-1: transpose Q, K """
                                """ Op 2: matmul Q_t @ K_t → accumulate psum_S """
                        """ Ops 3-4: transpose S → S_t, matmul S_t @ V → accumulate psum_output """
                    nisa.tensor_copy(dst=sbuf_partial[0:128, i_block_d0, 0, 0:128],
                        src=psum_output[0:128, 0:128])

for i_block_d0 in nl.affine_range(16):
    nisa.dma_copy(dst=output[...], src=sbuf_partial[0:128, i_block_d0, 0, 0:128])
```

This matches the reference attention kernel's structure: `for section_idx` wraps `for grp_i`, K/V pre-loaded per section, Q loaded per group. Online fusion extends this by adding running statistics (max, sum) as persistent SBUF state updated in-place each section, plus rescaling corrections — interleaving provides the section loop structure, online fusion fills in the math.

## Representation and Candidates

```python
group_dim_orders: list[tuple[str, ...]]
interleave_levels: list[dict[str, int]]
```

`group_dim_orders` — per fusion group, the ordered tuple of non-blocking output dim IDs. Dimension permutation modifies this ordering; the initial ordering comes from dimension analysis.

`interleave_levels` — per fusion group, maps blocking dim ID → level $k$ in `group_dim_orders`. Level $k$ places the block loop above $d_k$ (enclosing $d_k$ through $d_{n-1}$). Level 0 = above all output dims. Level $n$ = below all (default, no interleaving). The tile loop stays inside all output dims regardless.

```python
class LoopReorder(Transform):
    NAME = "loop_reorder"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        for gidx, dim_order in enumerate(ir.group_dim_orders):
            """ Dimension permutations (non-blocking output dims) """
            if len(dim_order) > 1:
                for perm in permutations(dim_order):
                    if perm != dim_order:
                        new_orders = list(ir.group_dim_orders)
                        new_orders[gidx] = perm
                        results.append(replace(ir, group_dim_orders=new_orders))
            """ Dimension interleaving (blocking dims with tiles_per_block > 1) """
            for dim_id in ir.group_blocking_dims[gidx]:
                if ir.tiles_per_block.get(dim_id, 1) <= 1:
                    continue
                n = len(dim_order)
                current = ir.interleave_levels[gidx].get(dim_id, n)
                for level in range(n + 1):
                    if level != current:
                        new_levels = [dict(d) for d in ir.interleave_levels]
                        new_levels[gidx][dim_id] = level
                        results.append(replace(ir, interleave_levels=new_levels))
        return results
```
