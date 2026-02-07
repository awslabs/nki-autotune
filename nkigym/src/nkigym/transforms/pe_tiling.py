"""PE Column Tiling transform for tiled compute graphs.

Combines adjacent subgraphs to execute in parallel on different PE column
tiles within a single merged subgraph. Phase 1 of the tiling pass produces
minimum 128-wide tiles on all axes. PE Column Tiling merges groups of these
128-tiles into larger tiles (up to 512 on free dimensions) that map onto
hardware PE column tile positions.

Operator constraints on maximum tile size per dimension:

    nc_matmul   M (stationary free)  128
    nc_matmul   K (contraction)      128
    nc_matmul   N (moving free)      512
    nc_transpose                     128 x 128

Whether to apply this transform (and at what aggregation factor) should be
decided by search rather than hardcoded, because:

- Larger tiles consume proportionally more SBUF, which can prevent
  pipelining and double-buffering in complex fused kernels.
- nc_transpose is capped at 128x128, so subgraphs mixing matmul and
  transpose have conflicting tile size preferences.
- Smaller tiles skip more fully-masked regions in causal attention.
- Autotuning has empirically found cases where a smaller free dimension
  tile outperforms the maximum.

Follows the analyze/transform pattern from transforms.base.Transform:
  analyze()   - identify groups of adjacent 128-tiles on free dimensions
                that can be merged, respecting per-operator constraints
  transform() - rewrite tile slice boundaries in the AST to merge adjacent
                subgraphs into PE column-tiled subgraphs
"""
