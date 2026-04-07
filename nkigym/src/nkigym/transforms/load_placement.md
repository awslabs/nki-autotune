# Load Placement

Each HBM input's DMA load sits at a position in the fusion group's loop nest. Moving a load **up** (hoisting) reduces DMA frequency — the data persists across iterations of the loops below it — at the cost of a larger SBUF buffer. Moving it **down** (sinking) shrinks the buffer but re-executes the DMA more often. In the base IR, all loads start at the innermost position.

## Placement Positions

Given a fusion group's dim order $(d_0, d_1, \ldots, d_{n-1})$ from outermost to innermost, each dimension contributes a block loop (`i_block_d`) and a tile loop (`i_tile_d`) to the nest. A DMA load's position determines, for each relevant dimension $d$, how many tiles the buffer holds:

| Load position relative to $d$ | Buffer sub-tiles on $d$ | Name |
|---|---|---|
| Above $d$'s block loop | $\texttt{unified\_tiles}(d)$ | **full** |
| Between $d$'s block and tile loops | $\texttt{tiles\_per\_block}(d) \times \texttt{interleave}(d)$ | **per-block** |
| Inside $d$'s tile loop | 1 | **single** (default) |

In the base IR ($\texttt{tiles\_per\_block} = 1$), per-block buffers $\texttt{interleave}(d)$ sub-tiles — equal to single when $\texttt{interleave}(d) = 1$, distinct otherwise. In a non-interleaved loop nest, per-block is always dominated by single (same DMA, larger buffer). Dimensions the tensor doesn't depend on never affect buffer sizing.

## DMA and Buffer Tradeoff

Let $T$ be the set of dims the tensor depends on. The total DMA calls at level $k$ is:

$$\text{total\_dma}(k) = \underbrace{\prod_{d_j \in T} \text{trips}(d_j)}_{\text{relevant total (constant)}} \times \underbrace{\prod_{\substack{i < k \\ d_i \notin T}} \text{trips}(d_i)}_{\text{irrelevant waste}}$$

The first factor counts the tensor's unique tiles — this is constant across levels. The second factor counts redundant reloads from irrelevant dims above the load. Hoisting past an irrelevant dim reduces the waste factor; hoisting past a relevant dim leaves waste unchanged.

The buffer size (in tiles) at level $k$ is:

$$\text{buffer\_tiles}(k) = \prod_{\substack{i \geq k \\ d_i \in T}} \text{trips}(d_i)$$

Each relevant dim below the load contributes its full trip count to the buffer; irrelevant dims never affect buffer sizing. At level $n$ (default), $\text{buffer\_tiles} = 1$ (degree-1). Hoisting past a relevant dim multiplies the buffer by that dim's trips; hoisting past an irrelevant dim leaves the buffer unchanged.

Together, the two formulas fully characterize the (DMA, buffer) tradeoff at every level. Since $\text{buffer\_tiles}$ is non-decreasing as $k$ decreases (hoisting only adds relevant dims below) and $\text{total\_dma}$ is non-increasing as $k$ decreases (hoisting only removes irrelevant waste above), the Pareto frontier is a chain from $(high\ DMA,\ small\ buffer)$ at level $n$ to $(low\ DMA,\ large\ buffer)$ at level $0$.

The **Pareto-optimal levels** are found by scanning from level $n$ downward in buffer tiers. Dims with $\text{trips} = 1$ are skipped (no-ops in both formulas). For each non-trivial dim encountered:

- **Irrelevant** ($d_k \notin T$): **free hoist** — DMA drops by $\text{trips}(d_k)$, buffer unchanged. Always apply.
- **Relevant** ($d_k \in T$): **tier crossing** — buffer grows (single to per-block to full), DMA unchanged. Starts a new buffer tier.

## Algorithm

1. Starting at level $n$, greedily apply free hoists until blocked by a relevant dim or reaching level $0$. The resulting position is the **effective baseline**: minimum DMA at the default buffer size.
2. At each blocking relevant dim $d$, cross it in up to two tiers:
   - **Per-block tier**: buffer grows to $\texttt{tiles\_per\_block}(d) \times \texttt{interleave}(d)$ on $d$. The load moves from inside $d$'s tile loop to between $d$'s block and tile. Apply free hoists. If DMA decreased, emit as candidate.
   - **Full tier**: buffer grows to $\texttt{unified\_tiles}(d)$ on $d$. The load moves above $d$'s block loop. Apply free hoists. If DMA decreased compared to the previous tier, emit as candidate.
3. Repeat until reaching level $0$.

In a standard (non-interleaved) loop nest, per-block on a relevant dim does not reduce DMA waste — no irrelevant loops sit between a dimension's block and tile — so per-block is always dominated by single. Per-block becomes Pareto-optimal when dim interleaving places irrelevant dims between a relevant dim's block and tile loops. In the flash attention pattern, $d_2$'s block wraps $d_0$: K/V at per-block on $d_2$ escape $d_0$'s irrelevant iterations — DMA drops, buffer grows to $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ sub-tiles, a **combined tradeoff**. Full on $d_2$ offers no further DMA reduction (same waste factor), so per-block dominates full. The algorithm generalizes: scan the actual loop structure from innermost to outermost, compute (DMA, buffer) at each position, and emit Pareto-optimal candidates.

Free hoists are always applied (strictly dominant). Tradeoff candidates — at most one per buffer tier — are the search space evaluated on hardware. The total search space across tensors is the Cartesian product of each tensor's tradeoff candidates (plus the effective baseline). When irrelevant dims sit above relevant dims, they can only be escaped by crossing the relevant dim first (a tradeoff). Loop reordering can place irrelevant dims below relevant ones for non-blocking dims, converting tradeoffs into free hoists. For blocking dims, dim interleaving enables per-block escaping — a combined tradeoff, not a free hoist.

## Fused Kernel Example

After full fusion, the single group `[[0,1,2,3,4]]` has dim order $(d_0, d_4, d_2, d_1)$ — output dims outer, reduction dims inner. All three HBM inputs start at level 4 (default, 256 DMA calls each). Their dim dependencies determine the optimal placement:

| Input | Depends on | Irrelevant loops | Optimal level | DMA | Buffer change |
|---|---|---|---|---|---|
| Q | $d_0, d_1$ | $d_2, d_4$ | 2 (above $d_2$) | 256 to 16 | none |
| K | $d_2, d_1$ | $d_0, d_4$ | 0 (above all) | 256 to 16 | 1 to 16 tiles ($d_2$) |
| V | $d_2, d_4$ | $d_0, d_1$ | 0 (above all) | 256 to 16 | 1 to 16 tiles ($d_2$) |

Applying the Pareto algorithm to each tensor (dim order $(d_0, d_4, d_2, d_1)$, skipping trivial dims $d_4$ and $d_1$ with trips = 1):

- **Q** ($T = \{d_0, d_1\}$): scan from 4, $d_2$ (irrel, 16 trips) free hoist, $d_0$ (rel) blocks. Effective baseline = level 2. Cross $d_0$: no free hoists above, DMA unchanged, discard. **Result**: level 2 only (free hoist, no tradeoffs).
- **K** ($T = \{d_2, d_1\}$): scan from 4, $d_2$ (rel, 16 trips) blocks. Baseline = level 4. Cross $d_2$: $d_0$ (irrel, 16 trips) free hoist to level 0. DMA dropped $16\times$, valid tradeoff. **Result**: {4, 0}.
- **V** ($T = \{d_2, d_4\}$): same structure as K. **Result**: {4, 0}.

Total search: $1 \times 2 \times 2 = 4$ combinations (base IR). With dim interleaving ($d_2$'s block wrapping $d_0$), K/V gain a per-block tier: buffer holds $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ sub-tiles on $d_2$ ($\texttt{interleave}(d_2) = 4$ in the base IR; $T \times \texttt{interleave}(d_2)$ after tiles-per-block sets $\texttt{tiles\_per\_block}(d_2) = T$). At per-block, K/V escape $d_0$'s $16\times$ redundancy — DMA drops from 256 to $\texttt{unified\_tiles}(d_2) = 16$ (each unique sub-tile loaded once). Full on $d_2$ achieves the same DMA but with a larger buffer (all 16 sub-tiles), so per-block dominates full. K and V Pareto sets become $\{\text{single}, \text{per-block}\}$. The [reference attention kernel](/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py) uses per-block: K/V loaded per-section ($\texttt{tiles\_per\_block}$ tiles between $d_2$'s block and tile loops, into `bufs.k_sb` / `bufs.v_sb`), Q loaded per-group (single on $d_0$), achieving full K/V reuse across Q groups within each section. Interleaving also affects Q: $d_2$'s block loop sits above $d_0$, so Q can only free-hoist past $d_2$'s tile loop — Q's baseline DMA rises from 16 to $16 \times \texttt{num\_blocks}(d_2)$. However, Q gains a tradeoff candidate: cross $d_0$ (full tier, buffer grows to $\texttt{unified\_tiles}(d_0)$), then free-hoist past $d_2$'s block — DMA drops back to 16. Q's Pareto set with interleaving becomes $\{\text{baseline}\ (DMA{=}16 \times \texttt{num\_blocks}(d_2),\ buffer{=}1),\ \text{full-}d_0\ (DMA{=}16,\ buffer{=}\texttt{unified\_tiles}(d_0))\}$. The [reference kernel](/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py) chooses the baseline (per-group Q into `bufs.q_sb` with `num_free_tiles=[2]` for double-buffering) — smaller buffer leaves more SBUF for K/V, and staging multi-buffering hides the extra DMA latency.

## V Hoisting Example

Standalone double matmul Op 4: dim order $(d_0, d_4, d_2)$, V depends on $(d_2, d_4)$. Pareto scan: $d_2$ (relevant, 16 trips) blocks at baseline (DMA = 256, buffer = 1). Per-block on $d_2$: buffer grows to $\texttt{interleave}(d_2) = 4$, DMA unchanged (no irrelevant dims between $d_2$'s block and tile in a standard nest), dominated. Full on $d_2$: buffer grows to 16, free hoist past $d_0$ (irrel), DMA = 16, emitted. Result: $\{3, 0\}$. With interleaving ($d_2$ block wraps $d_0$), per-block escapes $d_0$ — DMA = 16, buffer = $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ — dominates full. Result: $\{\text{single}, \text{per-block}\}$ (flash attention section pattern).

**Before** — `sbuf_V` at level 3 (inside all loops), 256 DMA for 16 unique tiles:

```python
""" Op 4: nisa.nc_matmul -- S_t(K=d2, M=d0) x V(K=d2, N=d4) -> output(d0, d4) """
psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
sbuf_output = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(16):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d4 in nl.affine_range(1):
            for i_tile_d4 in nl.affine_range(1):
                for i_interleave_group_d0 in nl.affine_range(1):
                    for i_interleave_group_d4 in nl.affine_range(1):
                        nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
                        for i_block_d2 in nl.affine_range(4):
                            for i_tile_d2 in nl.affine_range(1):
                                for i_interleave_group_d2 in nl.affine_range(4):
                                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],       # degree-1: 1 tile
                                        src=V[i_block_d2*512+i_interleave_group_d2*128:i_block_d2*512+i_interleave_group_d2*128+128,
                                             i_block_d4*128+i_interleave_group_d4*128:i_block_d4*128+i_interleave_group_d4*128+128])
                                    nisa.nc_matmul(dst=psum_output[0:128, 0:128],
                                        stationary=sbuf_S_t[0:128, i_block_d2*4+i_interleave_group_d2, i_block_d0+i_interleave_group_d0, 0:128],
                                        moving=sbuf_V[0:128, 0:128])
                        nisa.tensor_copy(dst=sbuf_output[0:128, 0:128], src=psum_output[0:128, 0:128])
                        nisa.dma_copy(dst=hbm_output[...], src=sbuf_output[0:128, 0:128])
```

**After** — `sbuf_V` hoisted to level 0. Pre-load loop fills full-range buffer. Buffer grows from `(128, 128)` to `(128, 16, 1, 128)` (16 d2 tiles); DMA drops to 16:

```python
""" Op 4: nisa.nc_matmul -- S_t(K=d2, M=d0) x V(K=d2, N=d4) -> output(d0, d4) """
sbuf_V = nl.ndarray((128, 16, 1, 128), dtype=V.dtype, buffer=nl.sbuf)  # full-range on d2: 16 tiles
for i_block_d2 in nl.affine_range(4):
    for i_tile_d2 in nl.affine_range(1):
        for i_interleave_group_d2 in nl.affine_range(4):
            nisa.dma_copy(dst=sbuf_V[0:128, i_block_d2*4+i_interleave_group_d2, 0, 0:128],
                src=V[i_block_d2*512+i_interleave_group_d2*128:i_block_d2*512+i_interleave_group_d2*128+128, 0:128])
psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
sbuf_output = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(16):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d4 in nl.affine_range(1):
            for i_tile_d4 in nl.affine_range(1):
                for i_interleave_group_d0 in nl.affine_range(1):
                    for i_interleave_group_d4 in nl.affine_range(1):
                        nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
                        for i_block_d2 in nl.affine_range(4):
                            for i_tile_d2 in nl.affine_range(1):
                                for i_interleave_group_d2 in nl.affine_range(4):
                                    nisa.nc_matmul(dst=psum_output[0:128, 0:128],
                                        stationary=sbuf_S_t[0:128, i_block_d2*4+i_interleave_group_d2, i_block_d0+i_interleave_group_d0, 0:128],
                                        moving=sbuf_V[0:128, i_block_d2*4+i_interleave_group_d2, 0, 0:128])
                        nisa.tensor_copy(dst=sbuf_output[0:128, 0:128], src=psum_output[0:128, 0:128])
                        nisa.dma_copy(dst=hbm_output[...], src=sbuf_output[0:128, 0:128])
```

The pre-load loop fills all 16 slots of `sbuf_V`. The compute loop indexes into the pre-loaded buffer by d2 position instead of issuing per-iteration DMAs.

The [reference kernel](/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py) implements per-block placement with dim interleaving. The section loop ($d_2$ block) wraps the Q-group loop ($d_0$), with $d_2$'s tile loop inside — placing $d_0$ between $d_2$'s block and tile for load purposes. K and V are loaded per-block on $d_2$ (between section and group loops into `bufs.k_sb`, `bufs.v_sb` — buffers holding $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ sub-tiles), achieving full reuse across Q groups within each section. Q loads per-group (single on $d_0$, no cross-group reuse). This combination — interleaving to place the irrelevant dim below the blocking dim, then per-block hoisting — produces the flash attention tiling.

## Representation

```python
load_placements: dict[str, dict[str, int]]
```

Maps each HBM tensor to its per-dimension buffered sub-tile count. For each relevant dimension $d$: 1 (single, default), $\texttt{tiles\_per\_block}(d) \times \texttt{interleave}(d)$ (per-block), or $\texttt{unified\_tiles}(d)$ (full). Absent dimensions default to 1. The total buffer is $\prod_{d \in T} \texttt{sub\_tiles}(d)$ raw slots per the tensor layout formula. The transform applies free hoists unconditionally and generates one candidate per Pareto-optimal tier per tensor; the total search space is the Cartesian product across tensors.
