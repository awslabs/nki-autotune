# Load Placement

Each HBM input's DMA load sits at a position in the fusion group's loop nest. Moving a load **up** (hoisting) keeps data in SBUF across more loop iterations (fewer DMA calls, larger buffer). Moving it **down** (sinking) shrinks the buffer but re-executes the DMA more often. In the base IR, all loads start at the innermost position.

## Placement Positions

Each dimension contributes a block loop (`i_block_d`), a tile loop (`i_tile_d`), and an interleave group loop (`i_ig_d`) to the nest. For each dimension $d$ that the tensor depends on (**relevant dim**), the load position determines the buffer's sub-tile count on $d$:

| Load position relative to $d$ | Buffer sub-tiles on $d$ | Name |
|---|---|---|
| Above $d$'s block loop | $\texttt{unified\_tiles}(d) \times \texttt{interleave}(d)$ | **full** |
| Between $d$'s block and tile loops | $\texttt{tiles\_per\_block}(d) \times \texttt{interleave}(d)$ | **per-block** |
| Inside all of $d$'s loops | 1 | **single** (default) |

Dimensions the tensor doesn't depend on (**irrelevant dims**) never affect buffer sizing. The renderer automatically hoists loads past irrelevant dims — this always reduces redundant DMA without growing the buffer, so it is not a search candidate (always applied).

Per-block is only useful when dim interleaving places irrelevant dims between a relevant dim's block and tile loops — the load escapes those irrelevant iterations while using a smaller buffer than full. In a standard (non-interleaved) nest, no loops sit between block and tile, so per-block offers no benefit over single.

## Representation

```python
load_placements: dict[str, dict[str, int]]
```

Maps each HBM tensor to its per-dimension buffered sub-tile count. For each relevant dimension $d$: 1 (single, default), $\texttt{tiles\_per\_block}(d) \times \texttt{interleave}(d)$ (per-block), or $\texttt{unified\_tiles}(d) \times \texttt{interleave}(d)$ (full). Absent dimensions default to 1. The total buffer is $\prod_{d \in T} \texttt{sub\_tiles}(d)$ raw slots.

## Candidate Generation

Each candidate increases one tensor's sub-tile count on one relevant dimension from its current tier to a higher tier.

`_apply(ir, name, dim_id, sub_tiles)` returns a new `KernelIR` with `load_placements[name][dim_id] = sub_tiles`.

```python
class LoadPlacement(Transform):
    NAME = "load_placement"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results = []
        for gidx, group in enumerate(ir.fusion_groups):
            for tensor_name in ir.group_hbm_inputs(gidx):
                relevant_dims = ir.tensor_relevant_dims(tensor_name)
                for dim_id in relevant_dims:
                    current = ir.load_placements.get(tensor_name, {}).get(dim_id, 1)
                    tpb = ir.tiles_per_block.get(dim_id, 1)
                    interleave = ir.ctx.interleave(dim_id)
                    unified = ir.ctx.unified_tiles(dim_id)
                    per_block = tpb * interleave
                    full = unified * interleave
                    if per_block > current and per_block < full:
                        results.append(self._apply(ir, tensor_name, dim_id, per_block))
                    if full > current:
                        results.append(self._apply(ir, tensor_name, dim_id, full))
        return results
```

Per-block is only offered when strictly between current and full (avoids duplicating single or full).

## Application

When the renderer encounters `load_placements[tensor][dim] > 1`, it transforms the rendered code in three ways:

1. **Buffer grows.** The staging buffer's `num_tiles` axis on the relevant dimension increases from 1 to the specified sub-tile count.

2. **DMA moves to a pre-load loop.** Instead of loading one sub-tile per compute iteration, the renderer emits a dedicated loop at the target position (between block and tile for per-block, above block for full) that fills all buffer slots before the compute loop begins.

3. **Compute loop indexes into the pre-loaded buffer.** The per-iteration DMA is removed; the compute loop reads from the correct buffer slot using the existing loop variables.

## Example

Standalone Op 4 from the double matmul: `nc_matmul(S_t, V)` with dim order $(d_0, d_4, d_2)$. V depends on $(d_2, d_4)$.

**Before** — `sbuf_V` at single (inside all loops):

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
                                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],
                                        src=V[i_block_d2*512+i_interleave_group_d2*128:i_block_d2*512+i_interleave_group_d2*128+128,
                                             i_block_d4*128+i_interleave_group_d4*128:i_block_d4*128+i_interleave_group_d4*128+128])
                                    nisa.nc_matmul(dst=psum_output[0:128, 0:128],
                                        stationary=sbuf_S_t[0:128, i_block_d2*4+i_interleave_group_d2, i_block_d0+i_interleave_group_d0, 0:128],
                                        moving=sbuf_V[0:128, 0:128])
                        nisa.tensor_copy(dst=sbuf_output[0:128, 0:128], src=psum_output[0:128, 0:128])
                        nisa.dma_copy(dst=hbm_output[...], src=sbuf_output[0:128, 0:128])
```

**After** — `load_placements = {"V": {"d2": 16}}` (full on d2). Buffer grows to `(128, 16, 1, 128)`. Pre-load loop fills all 16 slots; compute loop indexes into them:

```python
""" Op 4: nisa.nc_matmul -- S_t(K=d2, M=d0) x V(K=d2, N=d4) -> output(d0, d4) """
sbuf_V = nl.ndarray((128, 16, 1, 128), dtype=V.dtype, buffer=nl.sbuf)  # full on d2: 16 sub-tiles
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

## Reference Kernel Mapping

The [reference kernel](/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py) combines load placement with dim interleaving ($d_2$ block above $d_0$, tile inside) to produce the flash attention section pattern:

| Tensor | Placement | Buffer | Reference |
|---|---|---|---|
| K | per-block on $d_2$ | $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ sub-tiles | `bufs.k_sb` loaded per-section |
| V | per-block on $d_2$ | $\texttt{tiles\_per\_block}(d_2) \times \texttt{interleave}(d_2)$ sub-tiles | `bufs.v_sb` loaded per-section |
| Q | single on $d_0$ (+ multi-buffer D=2) | 1 sub-tile (double-buffered separately) | `bufs.q_sb` loaded per-group |

K and V at per-block on $d_2$ sit between $d_2$'s block (section) and tile loops. With $d_0$ (irrelevant to K/V) interleaved between them, K/V are loaded once per section and reused across all Q groups. Q stays at single — free hoists past $d_2$'s inner loops are applied automatically by the renderer.
