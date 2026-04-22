## DMA

Data moves through three memory levels: HBM → SBUF → PSUM (loads) and PSUM → SBUF → HBM (stores). One universal rule governs all store-direction transfers:

**Store rule: move data when the source is valid.** This applies identically at every memory boundary:
- **PSUM → SBUF** (`stage_block`): after the outermost blocking dim's block loop closes — every blocking dim's loops (block and tile, nested inside) have fully iterated, so the PSUM accumulator is final.
- **SBUF → HBM** (`store_block`): inside the producing group's innermost body — the output tile is final once the op writes it.

A non-blocking op (nc_transpose, single nc_matmul without accumulation) produces a valid PSUM result immediately, so the stage follows right after at the innermost body. A blocking op (nc_matmul accumulating over K tiles) produces a valid result only after the full accumulation of that dim, so the stage goes outside every loop touching a blocking dim. Same principle, different granularity.

### Gadgets

SBUF buffers are nested Python lists `sbuf_X[NP_list][NF_list]` of 2D `nl.ndarray(phys_P, leaf_F)` leaves (see `tensor_buffers.md`). Three gadgets move data in and out; each Python-iterates per leaf and issues one ISA call per tile:

- **`load_block(sbuf, mem, p_start, p_count, f_start, f_count)`** — HBM → SBUF. Copies a `(p_count × phys_P, f_count × leaf_F)` slab of `mem` into the `[p_start:p_start + p_count][f_start:f_start + f_count]` sub-block of `sbuf` via `nisa.dma_copy`.
- **`stage_block(sbuf, mem, p_start, p_count, f_start, f_count)`** — PSUM → SBUF. Same sub-block semantics; per-leaf copy via `nisa.tensor_copy`. `mem` is the PSUM ndarray.
- **`store_block(mem, sbuf, p_start, p_count, f_start, f_count)`** — SBUF → HBM. Reads the same sub-block of `sbuf` and writes it to `mem` via `nisa.dma_copy`.

All three require `mem.shape == (p_count * phys_P, f_count * leaf_F)` and raise `ValueError` otherwise. NKI forbids a single DMA that spans multiple partition slots, so per-leaf iteration is obligatory; because each leaf is one 128-lane partition block, the inner ISA call is a genuine 2D memref access with no partition stride.

Per-ptile staging inside an op's ptile loop (where the PSUM holds just one physical tile) emits a direct `nisa.tensor_copy` into one `SbufBuffer.get_tile(...)` slice rather than a gadget call — the gadget is for multi-leaf bulk transfers.

### Loads

Each HBM input tensor ``t`` in ``context.param_names`` is loaded exactly once via an ``NKILoad`` node inserted at build time by ``insert_dma_nodes`` (``kernel_ir/graph/graph.py``). The renderer dispatches ``NKILoad`` to ``dma_load_line`` at the op's own emission slot inside its group. The load depth comes from ``tensor_placements[("sbuf", <sbuf_alias>, d)]`` on each of the Load output's dims:

- `per_tile` — inside both `d`'s block and ltile loops (deepest).
- `per_block` — inside `d`'s block loop, outside its ltile loop.
- `full` — outside `d`'s block loop (hoisted).

Dims outside the group's `dim_order` impose no constraint. If no in-group dim constrains the tensor, the load is emitted at the top of the group.

The sub-block bounds come from `SbufBuffer.range(AxisAccess, AxisAccess)` with each loop-var bound iff it's in scope at the emission depth. A factor whose loop is not yet in scope (outer to the current depth) spans its full count; a factor whose tier does not materialize it collapses to `"0"` with count 1.

### PSUM → SBUF Staging

Every PSUM tensor whose consumer (or the kernel return) reads from SBUF gets a staging sibling. Position:

- **Non-blocking producer** — stage at the innermost body (depth `2N`).
- **Blocking producer** — stage at depth `i_min` (after-plan), where `i_min` is the outermost blocking dim's position in the group's `dim_order`. Being outside that block loop is strictly outside its tile loop too, so every blocking dim has fully iterated.

Ops with non-blocking ptile dims own their own per-ptile staging inside the op's ptile loop (`render_nki_ops._ptile_stage_lines`). Only ops whose ptile dims are all blocking (or which have none) go through the group-scope stage.

### SBUF → HBM Store

The return tensor is stored via `store_block` inside the producing group's innermost body. When the producer is a PSUM op, the store lands in the after-plan after the group-scope stage so the sequencing SBUF → HBM reads committed data.

### Example: Attention (`seq_q = seq_k = 512, d_k = d_v = 128`, default tiers)

```python
"""buffer allocations..."""
sbuf_K_t = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(1)]

"""Group 1: nc_transpose (K -> K_t) [dims: d1, d2]"""
for i_block_d1 in range(1):
    for i_block_d2 in range(1):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                load_block(sbuf_K, K[...], 0, 4, 0, 1)
                for i_ptile_d2 in range(4):
                    nisa.nc_transpose(psum_K_t[0:128, 0:128], sbuf_K[i_ptile_d2][0][0:128, 0:128])
                    """per-ptile stage: one (128, 128) tile into one phys-tile slice of the leaf"""
                    nisa.tensor_copy(sbuf_K_t[0][0][0:128, i_ptile_d2 * 128:i_ptile_d2 * 128 + 128],
                                     psum_K_t[0:128, 0:128])

"""Group 2: nc_matmul (Q_t @ K_t -> S) [dims: d0, d2, d1]; blocking on d1 (position 2)"""
for i_block_d0 in range(4):
    for i_block_d2 in range(1):
        nisa.memset(psum_S[0:128, 0:512], 0.0)
        for i_block_d1 in range(1):
            for i_ltile_d0 in range(1):
                for i_ltile_d2 in range(1):
                    for i_ltile_d1 in range(1):
                        nisa.nc_matmul(dst=psum_S[0:128, 0:512], ...)
        """block_d1 closed; depth = i_min; stage into one (128, 512) leaf"""
        stage_block(sbuf_S, psum_S, i_block_d0, 1, i_block_d2, 1)
```
