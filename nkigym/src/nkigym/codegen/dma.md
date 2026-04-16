## DMA

Data moves through three memory levels: HBM → SBUF → PSUM (loads) and PSUM → SBUF → HBM (stores). One universal rule governs all store-direction transfers:

**Store rule: move data when the source is valid.** This applies identically at every memory boundary:
- **PSUM → SBUF** (`tensor_copy`): after the blocking dimension's accumulation loop completes — the PSUM accumulator is final.
- **SBUF → HBM** (`nl.store`): after all reduction groups finish for a DP tile — the output tile is final.

A non-blocking op (nc_transpose, single nc_matmul without accumulation) produces a valid PSUM result immediately, so `tensor_copy` follows right after. A blocking op (nc_matmul accumulating over K tiles) produces a valid result only after the full accumulation loop, so `tensor_copy` goes after the loop. Same principle, different granularity.

### Gadgets

All multi-tile transfers use helper gadgets from `nkigym.dma.gadgets` to avoid inline loop nests in generated code. Three gadgets:

- **`load_tensor_block(dst, src, par_ofs, free_ofs)`** — HBM → SBUF. Iterates over all tile slots in a 4D (or 2D) on-chip buffer and copies each tile from HBM via `nisa.dma_copy`.
- **`stage_tensor_block(dst, src)`** — PSUM → SBUF. Iterates over all tile slots and issues `nisa.tensor_copy` for each. Both buffers must have the same shape.
- **`store_tensor_block(dst, src, par_ofs, free_ofs)`** — SBUF → HBM. Iterates over all tile slots in an SBUF buffer and copies each tile to HBM via `nisa.dma_copy`.

All three support 4D buffers `(physical_tile_size_P, num_tiles_P, num_tiles_F, physical_tile_size_F)` for 2D tensors and 2D buffers `(physical_tile_size_P, num_tiles_P)` for 1D tensors. The HBM offset is computed from loop variables: `offset = i_block * (tiles_per_block * logical_tile_size) + i_tile * logical_tile_size + i_ptile * physical_tile_size`.

### Loads

Each HBM input tensor needs a `load_tensor_block` into its SBUF buffer before any op can consume it. A tensor's load position depends on which dimensions it carries:

- A tensor with only DP dims (no reduction dims) is loaded once per DP tile — at the top of the innermost DP loop, after buffer allocations, before reduction groups.
- A tensor with reduction dims is loaded inside the reduction group that consumes it, at the innermost position where all its dims have loop variables in scope.

In the default lowering (degree-1, `num_tiles = 1`), each load brings in one tile.

### Stores

PSUM → SBUF and SBUF → HBM both follow the store rule above.

**PSUM → SBUF.** Each PSUM tensor with an SBUF staging buffer gets a `stage_tensor_block(sbuf_{name}, psum_{name})` after its value is valid. Position is mechanically determined: after the blocking dimension's last inner loop for blocking ops, immediately after the ISA call for non-blocking ops.

**SBUF → HBM.** The return tensor is stored via `store_tensor_block` at the bottom of the innermost DP loop, after all reduction groups finish.

### Example: Attention

```python
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_tile_d0 in range(1):
            for i_tile_d4 in range(1):
                for i_ptile_d0 in range(1):
                    for i_ptile_d4 in range(1):
                        """buffer allocations..."""

                        # --- Reduction groups 0–10 ---
                        # Each group loads its HBM inputs via load_tensor_block
                        # at the innermost position where all dims are in scope.
                        # Group 0 loads Q inside its d1 loop.
                        # Group 1 loads K inside its d1,d2 loop.
                        # Group 9 loads V inside its d2 loop.
                        #
                        # PSUM→SBUF tensor_copy follows the store rule:
                        # Group 0 (nc_transpose Q): non-blocking, copy immediately.
                        # Group 2 (nc_matmul QK): blocking on d1, copy after d1 loop.
                        # Group 9 (nc_matmul SV): blocking on d2, copy after d2 loop.
                        """reduction loops with loads and tensor_copies..."""

                        # --- SBUF→HBM store: output tile ready ---
                        store_tensor_block(dst=output, src=sbuf_output, par_ofs=..., free_ofs=...)
```
