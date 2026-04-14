## Data Layout Transforms

The current search space treats data layout as fixed — each tensor's partition and free axes are determined by the math function and never change. A data layout transform pass would manipulate `nc_transpose` ops to improve memory access patterns and engine utilization. The key moves are:

1. **Insert dummy transpose pairs** — add a transpose before and after any tensor access point (the pair is a no-op).
2. **Cancel adjacent transposes** — two consecutive transposes on the same tensor annihilate.
3. **Move transposes** — slide a transpose earlier or later in the graph, past compatible ops, to find a more profitable placement.
4. **Merge transpose with DMA** — when a transpose is adjacent to an HBM load/store, replace the `nc_transpose` + `nisa.dma_copy` pair with a single transposing DMA (`nisa.dma_copy` with transposed source/destination layout), eliminating the Tensor Engine transpose entirely.
