# Development Notes

## Issues

### transpose-reuse

**Status:** open

**Title:** Duplicate transposes when lowering matmul with data reuse

**Description:**
`np.matmul(lhs, rhs)` lowers to `nisa.nc_matmul(nc_transpose(lhs), rhs)`. When LHS is reused across multiple matmuls, load reuse optimization merges the loads but transposes are still duplicated.

**Example:**

Before:
```
lhs_0=load(); matmul(lhs_0,rhs_0); matmul(lhs_0,rhs_1)
```

After lowering:
```
lhs_0=dma_copy(); lhsT_0=nc_transpose(lhs_0); nc_matmul(lhsT_0,rhs_0); lhsT_1=nc_transpose(lhs_0); nc_matmul(lhsT_1,rhs_1)
```

**Solution Reference:**
- Branch: `main`
- Module: `compute_graph`
- Approach: Users write nisa-like operators (Matmul with lhs_transposed flag). Graph-based `insert_tile_transpose()` adds TileTranspose nodes. `tensor_producer` dict naturally deduplicates since each tensor has one producer.
- Key files:
  - `compute_graph/graph.py:insert_tile_transpose`
  - `compute_graph/node/compute.py:Matmul,TileTranspose`
