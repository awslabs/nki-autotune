## DMA *(not yet enabled in `render_ir`)*

Data moves through three memory levels: HBM → SBUF → PSUM (loads) and PSUM → SBUF → HBM (stores). One universal rule governs all store-direction transfers:

**Store rule: move data when the source is valid.** This applies identically at every memory boundary:
- **PSUM → SBUF** (`tensor_copy`): after the blocking dimension's accumulation loop completes — the PSUM accumulator is final.
- **SBUF → HBM** (`nl.store`): after the producing group's nest closes — the output tile is final.

A non-blocking op (nc_transpose, single nc_matmul without accumulation) produces a valid PSUM result immediately, so `tensor_copy` follows right after. A blocking op (nc_matmul accumulating over K tiles) produces a valid result only after the full accumulation loop, so `tensor_copy` goes after the loop. Same principle, different granularity.

### Gadgets

All multi-tile transfers use helper gadgets from `nkigym.dma.gadgets` to avoid inline loop nests in generated code. Three gadgets:

- **`load_tensor_block(dst, src, par_ofs, free_ofs)`** — HBM → SBUF. Iterates over all tile slots in a 4D (or 2D) on-chip buffer and copies each tile from HBM via `nisa.dma_copy`.
- **`stage_tensor_block(dst, src)`** — PSUM → SBUF. Iterates over all tile slots and issues `nisa.tensor_copy` for each. Both buffers must have the same shape.
- **`store_tensor_block(dst, src, par_ofs, free_ofs)`** — SBUF → HBM. Iterates over all tile slots in an SBUF buffer and copies each tile to HBM via `nisa.dma_copy`.

All three support 4D buffers `(physical_tile_size_P, num_tiles_P, num_tiles_F, physical_tile_size_F)` for 2D tensors and 2D buffers `(physical_tile_size_P, num_tiles_P)` for 1D tensors. The HBM offset is computed from the kernel-level loop variables: `offset = i_block * (ltiles_per_block * logical_tile_size) + i_ltile * logical_tile_size`. Physical-tile iteration within a logical tile is internal to the gadget — it walks the buffer's physical slots without needing a kernel-level `i_ptile_*` loop.

### Loads

Each HBM input tensor needs a `load_tensor_block` into its SBUF buffer before any op can consume it. A tensor's load position is determined by scope:

- A tensor is loaded inside the group that consumes it, at the innermost position where all the tensor's dims have loop variables in scope.
- If a kernel input tensor has no dims inside any group's nest, it is loaded at the top of the kernel body, before the group nests.

In the default lowering (degree-1, `num_tiles = 1`), each load brings in one tile.

### Stores

PSUM → SBUF and SBUF → HBM both follow the store rule above.

**PSUM → SBUF.** Each PSUM tensor with an SBUF staging buffer gets a `stage_tensor_block(sbuf_{name}, psum_{name})` after its value is valid. Position is mechanically determined: after the blocking dimension's last inner loop for blocking ops, immediately after the ISA call for non-blocking ops.

**SBUF → HBM.** The return tensor is stored via `store_tensor_block` after the producing group's nest closes. In topological order, this is the final group whose output feeds the return tensor.

### Example: Attention

Each group emits its own complete loop nest as a sibling block — no outer wrapper. A group's nest covers only the dims its ops touch; loads land inside the group where all their dims are in scope, and the final SBUF→HBM store follows the group that produces the return tensor.

```python
"""buffer allocations..."""

"""Group 1: nc_transpose (K -> K_t) [dims: d1, d2]"""
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                load_tensor_block(dst=sbuf_K, src=K, par_ofs=..., free_ofs=...)
                """nc_transpose -> psum_K_t; non-blocking: stage immediately"""
                stage_tensor_block(dst=sbuf_K_t, src=psum_K_t)

"""Group 2: nc_matmul (Q_t @ K_t -> S) [dims: d0, d1, d2]"""
for i_block_d0 in range(16):
    for i_block_d1 in range(1):
        for i_block_d2 in range(4):
            for i_ltile_d0 in range(1):
                for i_ltile_d1 in range(1):
                    """Q_t from group 0 in scope; K_t from group 1 in scope"""
                    for i_ltile_d2 in range(1):
                        """nc_matmul accumulates; blocking on d1"""
                        pass
                    """d1 loop closed: PSUM valid, stage"""
                    stage_tensor_block(dst=sbuf_S, src=psum_S)

"""... groups 3-9 ..."""

"""Group 10: tensor_scalar (attn * inv_sum -> output) [dims: d0, d4]"""
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_ltile_d0 in range(1):
            for i_ltile_d4 in range(1):
                """produces sbuf_output tile"""
                pass

"""Final group produced the return tensor; store after its nest closes"""
store_tensor_block(dst=output, src=sbuf_output, par_ofs=..., free_ofs=...)
```
