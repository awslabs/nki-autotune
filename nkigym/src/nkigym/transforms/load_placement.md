# Load Placement

*Single-loop-nest transform — operates on one fusion group's loop nest. Fusing loop nests is handled by online fusion and loop fusion.*

## What It Does

Each HBM input's DMA load sits at a position in the fusion group's loop nest. Moving a load **up** (hoisting) pre-loads more sub-tiles into a larger SBUF buffer; moving it **down** (sinking) shrinks the buffer but re-executes the DMA more often. In the base IR, all loads start at the innermost position.

Each dimension contributes a block loop (`i_block_d`), a psum-batch loop (`i_psum_batch_d`), a tile loop (`i_tile_d`), and an interleave group loop (`i_ig_d`) to the nest. For DMA, the sub-block loops (psum-batch, tile, ig) are encapsulated inside the `load_tensor_block` / `save_tensor_block` gadgets from `nkigym.gadgets` — the gadgets iterate over all tile slots in the buffer internally. The main loop nest only provides block-level offsets to the gadget; the compute loops remain explicit.

For each dimension $d$ that the tensor depends on (**relevant dim**), the load position determines the buffer's sub-tile count on $d$:

| Load position relative to $d$ | Buffer sub-tiles on $d$ | Name |
|---|---|---|
| Above $d$'s block loop | $\texttt{unified\_tiles}(d) \times \texttt{interleave}(d)$ | **full** |
| Between $d$'s block and psum-batch loops | $\texttt{tpb\_hbm}(d) \times \texttt{interleave}(d)$ | **per-block** |
| Between $d$'s psum-batch and tile loops | $\texttt{tpb\_psum}(d) \times \texttt{interleave}(d)$ | **per-psum-batch** |
| Between $d$'s tile and ig loops | $\texttt{interleave}(d)$ | **per-tile** |
| Inside all of $d$'s loops | 1 | **single** (default) |

When a dimension has a non-single tier, the renderer:

1. **Grows the buffer.** The staging buffer's `num_tiles` axis on that dimension increases from 1 to the tier's derived sub-tile count.

2. **Moves the DMA.** The DMA call moves to the target position (above the hoisted dimension's loop). `load_tensor_block` fills all buffer slots via its internal loops (`for par_tid in range(num_tiles_p): for free_tid in range(num_tiles_f):`), keeping the main loop nest unchanged.

3. **Removes the inline DMA.** The original per-iteration DMA inside the loop is removed; the compute reads from the correct buffer slot using the existing loop variable.

The loop nest itself never changes — no loops are added or split. Dimensions the tensor doesn't depend on (**irrelevant dims**) never affect buffer sizing. Load placement always hoists a load past any irrelevant dims that sit between the load's current position and its target relevant-dim tier — repeating a load inside an irrelevant dim re-executes identical DMAs (the tensor doesn't depend on that dim), so hoisting past it is a pure benefit with no tradeoff. At single placement, the DMA sits at the innermost relevant dimension's loop.

## Why

Each tier above single reduces DMA calls by pre-loading more sub-tiles into a larger buffer. Per-tile pre-loads one tile's interleave groups; per-psum-batch pre-loads one PSUM batch; per-block pre-loads an entire block; full pre-loads the whole dimension. The tradeoff is SBUF capacity — higher tiers use larger buffers, which may not fit.

Dimension interleaving (see loop_reordering.md) changes which loops sit between a load's position and the compute, affecting reuse. Interleaving a dim places other dimensions' loops between its block and tile loops, so per-block loads persist across those enclosed iterations.

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

Matmul `lhs_T(K=d0, M=d1) × rhs(K=d0, N=d2) → result(d1, d2)` with d0 (K, tile=128, 16 tiles), d1 (M, tile=128, 16 tiles), d2 (N, tile=512, 4 tiles). Loop order (d0, d1, d2). rhs depends on (d0, d2); lhs\_T depends on (d0, d1).

**Before** — `sbuf_rhs` at single on d2 (DMA inside d2 loop, buffer holds 1 d2 sub-tile):

```python
sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 1, 1, 512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_d0 in range(16):
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=i_d2*512)
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, 0, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

**After** — `load_placements = {"rhs": {"d2": "full"}}`. `sbuf_rhs` grows from `(128, 1, 1, 512)` to `(128, 1, 4, 512)`. The rhs DMA moves above the d2 loop and loads all 4 d2 tiles tile-by-tile via a helper; the d2 loop is unchanged:
```python
sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 1, 4, 512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_d0 in range(16):
    load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*128, free_ofs=0)
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                stationary=sbuf_lhs_T[0:128, 0, 0, 0:128], moving=sbuf_rhs[0:128, 0, i_d2, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

In this example d1 is irrelevant to rhs, so the load is automatically hoisted above the d1 loop as well (no redundant re-execution inside an irrelevant dim). The rhs DMA now executes once per d0 iteration instead of 16× (once per d1 tile).