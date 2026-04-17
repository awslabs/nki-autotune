## DMA

Data moves through three memory levels: HBM → SBUF → PSUM (loads) and PSUM → SBUF → HBM (stores). One universal rule governs all store-direction transfers:

**Store rule: move data when the source is valid.** This applies identically at every memory boundary:
- **PSUM → SBUF** (`stage_tensor_block`): after the outermost blocking dim's *block* loop closes — every blocking dim's loops (block and tile, nested inside) have fully iterated, so the PSUM accumulator is final.
- **SBUF → HBM** (`store_tensor_block`): inside the producing group's innermost body — the output tile is final once the op writes it.

A non-blocking op (nc_transpose, single nc_matmul without accumulation) produces a valid PSUM result immediately, so `stage_tensor_block` follows right after at the innermost body. A blocking op (nc_matmul accumulating over K tiles) produces a valid result only after the full accumulation of that dim, so `stage_tensor_block` goes outside every loop touching a blocking dim. Same principle, different granularity.

### Gadgets

All multi-tile transfers use helper gadgets from `nkigym.dma.gadgets` to avoid inline loop nests in generated code. Three gadgets:

- **`load_tensor_block(dst, src, par_ofs, free_ofs)`** — HBM → SBUF. Iterates over all tile slots in a 4D (or 2D) on-chip buffer and copies each tile from HBM via `nisa.dma_copy`.
- **`stage_tensor_block(dst, src)`** — PSUM → SBUF. Iterates over all tile slots and issues `nisa.tensor_copy` for each. Both buffers must have the same shape.
- **`store_tensor_block(dst, src, par_ofs, free_ofs)`** — SBUF → HBM. Iterates over all tile slots in an SBUF buffer and copies each tile to HBM via `nisa.dma_copy`.

All three support 4D buffers `(physical_tile_size_P, num_tiles_P, num_tiles_F, physical_tile_size_F)` for 2D tensors and 2D buffers `(physical_tile_size_P, num_tiles_P)` for 1D tensors. The HBM offset is computed from the kernel-level loop variables: `offset = i_block * (ltiles_per_block * logical_tile_size) + i_ltile * logical_tile_size`. Physical-tile iteration within a logical tile is internal to the gadget — it walks the buffer's physical slots without needing a kernel-level `i_ptile_*` loop.

### Loads

Each HBM input tensor `t` in `dim_analysis.param_names` is loaded exactly once. The owning group is the one containing the earliest op (smallest `op_idx`) that reads `t`; this is derived from `op_graph.ops_touching(t)`. No new IR field — the position is fully determined by:

- `fusion_groups` — which group owns the load.
- `group_dim_orders[group_idx]` — the owning group's loop order.
- `tensor_placements[(t, d)]` — per-dim tier (`per_tile`, `per_block`, `full`), which also drives buffer shape (see `kernel_ir/load_placement.md`).

**Position rule.** Let `relevant = set(t.dim_ids) ∩ set(group_dim_orders[g])` — the dims of `t` that produce loop variables in `g`'s nest. The load is emitted at the innermost indent level where every `d ∈ relevant` has its tier's required loops in scope:

- `per_tile` — inside both `d`'s block and ltile loops (deepest).
- `per_block` — inside `d`'s block loop, outside its ltile loop.
- `full` — outside `d`'s block loop (hoisted).

Dims in `t.dim_ids` but not in `relevant` impose no constraint — the tensor is invariant over them.

If `relevant` is empty (tensor has no dims in any group's nest), the load is emitted at the top of the kernel body, before the group nests.

In the default lowering (all tiers `per_tile`, degree-1), each load brings in one tile at the innermost nest position.

### PSUM → SBUF Staging

Every PSUM tensor that a consumer (or the return tensor) reads from SBUF gets one `stage_tensor_block(sbuf_{name}, psum_{name})` call in the fusion group that produces it. The set of such tensors is `find_psum_tensors_needing_sbuf(ir)` in `codegen/buffers.py`. Position is fully determined by the producing op's blocking dims intersected with the group's `dim_order`:

- **Non-blocking producer** (no blocking dim appears in the group's `dim_order`) — stage at the innermost body (depth `2N`, where `N = len(group_dim_orders[group_idx])`). The PSUM result is complete after every ISA call.
- **Blocking producer** — let `i_min = min(pos(d) for d in blocking_dims ∩ dim_order)`, the outermost blocking dim's position in `dim_order`. Stage at depth `i_min` (after-plan) — the indent level right after that dim's *block* loop closes. Under phase-grouped emission (all block loops outermost, then all tile loops), being outside that block loop is strictly outside its tile loop too, so every blocking dim's loops (block and tile, at positions ≥ `i_min`) have fully iterated.

No new IR field — derivation uses `op_graph.producer_op`, per-op `BLOCKING_AXES`, and `dim_analysis.per_op_axis_maps`.

### SBUF → HBM Store

The return tensor is stored via `store_tensor_block` inside the producing group's innermost body. The group is the one containing `op_graph.producer_op(return_name)`. Emitting inside the innermost body keeps the group's loop variables in scope for the HBM offset expressions.

### Example: Attention

Each group emits its own complete loop nest as a sibling block — no outer wrapper. A group's nest covers only the dims its ops touch. Loads land at the depth dictated by `tensor_placements`; PSUM staging sits at the innermost body for non-blocking producers or after the outermost blocking dim's tile loop closes for blocking producers; the return tensor is stored inside the final producing group's innermost body.

```python
"""buffer allocations..."""

"""Group 1: nc_transpose (K -> K_t) [dims: d1, d2]"""
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                load_tensor_block(sbuf_K, K, i_block_d2 * 512 + ..., i_block_d1 * 128 + ...)
                """nc_transpose is non-blocking: stage at innermost body"""
                stage_tensor_block(sbuf_K_t, psum_K_t)
                pass

"""Group 2: nc_matmul (Q_t @ K_t -> S) [dims: d0, d1, d2]; blocking on d1 (position 1)"""
for i_block_d0 in range(16):
    for i_block_d1 in range(1):
        for i_block_d2 in range(4):
            for i_ltile_d0 in range(1):
                for i_ltile_d1 in range(1):
                    for i_ltile_d2 in range(1):
                        """nc_matmul accumulates across d1 tiles"""
                        pass
    """block_d1 closed: stage at depth i_min = 1, fires once per i_block_d0"""
    stage_tensor_block(sbuf_S, psum_S)

"""... groups 3-9 ..."""

"""Group 10: tensor_scalar (attn * inv_sum -> output) [dims: d0, d4]"""
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_ltile_d0 in range(1):
            for i_ltile_d4 in range(1):
                """return tensor's producer; store inside innermost body"""
                store_tensor_block(output, sbuf_output, i_block_d0 * 128 + ..., i_block_d4 * 128 + ...)
                pass
```
