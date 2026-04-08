# Loop Reordering

Loop reordering changes a fusion group's dimension iteration order to improve data reuse and DMA efficiency. Two capabilities compose the search space: **dimension permutation** reorders non-blocking output dims, and **dimension interleaving** hoists a blocking dim's block loop above some or all output dims while keeping its tile loop inside. Both preserve correctness; each candidate changes a single group.

## Dimension Permutation

Non-blocking output dimensions in `group_dim_orders` (the per-group ordered list of non-blocking output dim IDs) can be freely permuted. The transform produces one candidate for each non-identity permutation per group.

**All permutations are valid.** Three structural properties guarantee this without per-candidate checking:

1. **Same-dimension phase ordering.** Permutation reorders dimensions, not phases (`i_block` → `i_tile` → `i_ig`) within a dimension.

2. **Reduction dims innermost.** PSUM accumulator lifecycle (memset → K loop → writeback) requires reduction dims enclosed by output dims. `group_dim_orders` never contains blocking dims.

3. **Fused groups: non-blocking invariant.** All dims in `group_dim_orders` are non-blocking for every op. In the fused double matmul: d1, d2 excluded as reduction dims, giving `group_dim_orders = (d0, d4)`.

**What varies across orderings:**

- **Data reuse.** An operand is loop-invariant in any dim it doesn't depend on. Placing invariant dims innermost maximizes reuse — the operand is loaded once and persists across all inner iterations. Each ordering favors operands whose irrelevant dims land innermost.
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

With `d2` outermost, any operand that depends on d2 but not d0 is loop-invariant across d0's 16 iterations — a reuse opportunity that load placement (§5.2) exploits by hoisting those loads above the d0 loop.

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

Pre-zeroing the SBUF partial buffer avoids conditional logic within `nl.affine_range` — the first reload loads zero (equivalent to memset), subsequent reloads load the running partial sum.

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

- **`num_blocks > 1`** on the interleaved dim — with only 1 block the block loop has range(1) and interleaving is a no-op. $\texttt{num\_blocks} = \texttt{unified\_tiles} / \texttt{tiles\_per\_block}$ where $\texttt{unified\_tiles} = \texttt{dim\_size} / \texttt{max\_tile\_size}$ (§5.4). Increasing `tiles_per_block` reduces `num_blocks` (fewer, larger sections) and better amortizes save/reload overhead, but is not a prerequisite — any `num_blocks > 1` enables interleaving
- The dim must be **accumulating**: every op that reduces over this dim uses additive accumulation (`+=` semantics). `nc_matmul` accumulates into PSUM — valid. `tensor_reduce(op="add")` sums partial results — valid ($\sum A + \sum B = \sum(A \cup B)$). `tensor_reduce(op="max")` is NOT additive: $\max(A \cup B) \neq \max(A) + \max(B)$. Non-accumulating reductions require online fusion (§6) to convert to accumulating form before interleaving applies

**Benefits and costs.** Operands loaded between block and tile loops persist across all enclosed output-dim iterations ($d_k$ through $d_{n-1}$), enabling cross-output-dim reuse via load placement (§5.2). The reuse factor equals the product of trips across enclosed output dims. The cost is extra `tensor_copy` per (block-iteration, enclosed-output-tile) for save/reload of the partial accumulator, plus a partial buffer with $\prod_{i \geq k} \texttt{unified\_tiles}(d_i) \cdot \texttt{interleave}(d_i)$ sub-tile slots for a level-$k$ interleave, following the standard 4D buffer layout (§2).

**Example.** Fused double matmul `[[0,1,2,3,4]]` with `tiles_per_block_d2 = 2`. Interleaving d2 above d0 produces the section pattern:

```python
""" Fused double matmul — d2 interleaved above d0 (section pattern) """
sbuf_partial = nl.ndarray((128, 16, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_partial[...], value=0.0)

for i_block_d2 in nl.affine_range(2):                            """ d2 block (hoisted above d0) """
    for i_block_d0 in nl.affine_range(16):                       """ d0 (enclosed by d2 block) """
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
            """ Dimension interleaving (accumulating blocking dims with num_blocks > 1) """
            for dim_id in ir.group_blocking_dims[gidx]:
                if not ir.is_accumulating(dim_id, gidx):
                    continue
                unified = ir.ctx.dim_sizes[dim_id] // ir.ctx.dim_tiles[dim_id]
                if unified // ir.tiles_per_block.get(dim_id, 1) <= 1:
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
