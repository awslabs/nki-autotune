# Load Placement

## What It Does

Each HBM input's DMA load sits at a position in the fusion group's loop nest. Moving a load **up** (hoisting) pre-loads more sub-tiles into a larger SBUF buffer; moving it **down** (sinking) shrinks the buffer but re-executes the DMA more often. In the base IR, all loads start at the innermost position.

Each dimension contributes a block loop (`i_block_d`), a psum-batch loop (`i_psum_batch_d`), a tile loop (`i_tile_d`), and an interleave group loop (`i_ig_d`) to the nest (§2). For each dimension $d$ that the tensor depends on (**relevant dim**), the load position determines the buffer's sub-tile count on $d$:

| Load position relative to $d$ | Buffer sub-tiles on $d$ | Name |
|---|---|---|
| Above $d$'s block loop | $\texttt{unified\_tiles}(d) \times \texttt{interleave}(d)$ | **full** |
| Between $d$'s block and psum-batch loops | $\texttt{tpb\_hbm}(d) \times \texttt{interleave}(d)$ | **per-block** |
| Between $d$'s psum-batch and tile loops | $\texttt{tpb\_psum}(d) \times \texttt{interleave}(d)$ | **per-psum-batch** |
| Between $d$'s tile and ig loops | $\texttt{interleave}(d)$ | **per-tile** |
| Inside all of $d$'s loops | 1 | **single** (default) |

When a dimension has a non-single tier, the renderer:

1. **Grows the buffer.** The staging buffer's `num_tiles` axis on that dimension increases from 1 to the tier's derived sub-tile count.

2. **Moves the DMA.** The DMA moves to the target position (above the hoisted dimension's loop). A tile-by-tile load loop fills all buffer slots via a helper gadget (e.g. `load_tensor_block`), keeping the main loop nest unchanged.

3. **Removes the inline DMA.** The original per-iteration DMA inside the loop is removed; the compute reads from the correct buffer slot using the existing loop variable.

The loop nest itself never changes — no loops are added or split. Dimensions the tensor doesn't depend on (**irrelevant dims**) never affect buffer sizing. At single placement, the DMA is inside all loops including irrelevant ones; eliminating that redundancy is the primary purpose of load placement.

## Why

Each tier above single reduces DMA calls by pre-loading more sub-tiles into a larger buffer. Per-tile pre-loads one tile's interleave groups; per-psum-batch pre-loads one PSUM batch; per-block pre-loads an entire block; full pre-loads the whole dimension. The tradeoff is SBUF capacity — higher tiers use larger buffers, which may not fit.

Dimension interleaving (§5.3) changes which loops sit between a load's position and the compute, affecting how much reuse each tier provides. See §5.3 for details on how interleaving encloses output dims between a blocking dim's block and psum-batch loops, turning per-block loads into cross-output-dim reuse.

## Candidate Generation

```python
load_placements: dict[str, dict[str, Literal["per_tile", "per_psum_batch", "per_block", "full"]]]
```

Maps each HBM tensor to its per-dimension placement tier. Absent dimensions default to single. The renderer derives each tier's sub-tile count from current dimension parameters (per-tile → $\texttt{interleave}(d)$, per-psum-batch → $\texttt{tpb\_psum}(d) \times \texttt{interleave}(d)$, per-block → $\texttt{tpb\_hbm}(d) \times \texttt{interleave}(d)$, full → $\texttt{unified\_tiles}(d) \times \texttt{interleave}(d)$). Total buffer = $\prod_{d \in T} \texttt{sub\_tiles}(d)$ raw slots.

Storing tiers instead of raw counts keeps the representation valid when $\texttt{tpb\_hbm}$ or $\texttt{tpb\_psum}$ changes — "per_block" always means "between block and psum-batch loops", "per_psum_batch" always means "between psum-batch and tile loops", and derived buffer sizes adjust automatically.

Each candidate promotes one tensor on one relevant dimension from its current tier to a higher tier. `_apply(ir, name, dim_id, tier)` returns a new `KernelIR` with `load_placements[name][dim_id] = tier`.

```python
class LoadPlacement(Transform):
    NAME = "load_placement"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results = []
        for gidx, group in enumerate(ir.fusion_groups):
            for tensor_name in ir.group_hbm_inputs(gidx):
                relevant_dims = ir.tensor_relevant_dims(tensor_name)
                for dim_id in relevant_dims:
                    tier = ir.load_placements.get(tensor_name, {}).get(dim_id, "single")
                    interleave = ir.ctx.interleave(dim_id)
                    tpb_hbm = ir.tpb_hbm.get(dim_id, 1)
                    tpb_psum = ir.tpb_psum.get(dim_id, tpb_hbm)
                    unified = ir.ctx.unified_tiles(dim_id)
                    counts = {"single": 1, "per_tile": interleave,
                              "per_psum_batch": tpb_psum * interleave,
                              "per_block": tpb_hbm * interleave,
                              "full": unified * interleave}
                    current = counts[tier]
                    if counts["per_tile"] > current and counts["per_tile"] < counts["per_psum_batch"]:
                        results.append(self._apply(ir, tensor_name, dim_id, "per_tile"))
                    if counts["per_psum_batch"] > current and counts["per_psum_batch"] < counts["per_block"]:
                        results.append(self._apply(ir, tensor_name, dim_id, "per_psum_batch"))
                    if counts["per_block"] > current and counts["per_block"] < counts["full"]:
                        results.append(self._apply(ir, tensor_name, dim_id, "per_block"))
                    if counts["full"] > current:
                        results.append(self._apply(ir, tensor_name, dim_id, "full"))
        return results
```

Tier distinctness is checked via derived sub-tile counts — each tier is only offered when its count is strictly between the current count and the next higher tier's count (avoids duplicates). Per-tile is only distinct from single when $\texttt{interleave} > 1$. Per-psum-batch is only distinct from per-tile when $\texttt{tpb\_psum} > 1$, and from per-block when $\texttt{tpb\_hbm} > \texttt{tpb\_psum}$.

## Example

Matmul `lhs_T(K=d0, M=d1) × rhs(K=d0, N=d2) → result(d1, d2)` with d0: 16 blocks of 128, d1: 16 blocks of 128, d2: 4 blocks of 512. Loop order (d0, d1, d2). rhs depends on (d0, d2); lhs\_T depends on (d0, d1).

**Before** — `sbuf_rhs` at single on d2 (DMA inside d2 loop, buffer holds 1 d2 sub-tile):

```python
sbuf_lhs_T = nl.ndarray((128,1,1,128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128,1,1,512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512),buffer=nl.psum)
sbuf_output = nl.ndarray((128, 16, 4, 512),buffer=nl.sbuf)
for i_d0 in range(16):
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],stationary=sbuf_lhs_T[...],moving=sbuf_rhs[...])
nisa.tensor_copy(psum_output -> sbuf_output)
nisa.dma_copy(sbuf_output -> hbm_output)
```

**After** — `load_placements = {"rhs": {"d2": "full"}}`. `sbuf_rhs` grows from `(128,1,1,512)` to `(128,1,4,512)`. The rhs DMA moves above the d2 loop and loads all 4 d2 tiles tile-by-tile via a helper; the d2 loop is unchanged:
```python
sbuf_lhs_T = nl.ndarray((128,1,1,128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128,1,4,512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512),buffer=nl.psum)
sbuf_output = nl.ndarray((128, 16, 4, 512),buffer=nl.psum)
for i_d0 in range(16):
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=0)
        for i_d2 in range(4):
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],stationary=sbuf_lhs_T[...],moving=sbuf_rhs[0:128,0,i_d2,0:512])
nisa.tensor_copy(psum_output -> sbuf_output)
nisa.dma_copy(sbuf_output -> hbm_output)
```