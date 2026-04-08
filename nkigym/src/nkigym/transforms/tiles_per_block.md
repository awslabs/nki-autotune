# Tiles Per Block

`tiles_per_block` groups consecutive unified tiles into blocks, splitting each dimension's flat iteration into a two-level loop nest (outer block, inner tile). From §2:

$$\texttt{unified\_tiles} = \frac{\texttt{dim\_size}}{\texttt{max\_tile\_size}} = \texttt{num\_blocks} \times \texttt{tiles\_per\_block}$$

Must divide `unified_tiles`. Default is 1. Setting to T creates a two-level nest:

```python
""" tiles_per_block = 1 (default) """
for i_block_d in nl.affine_range(unified_tiles):
    for i_tile_d in nl.affine_range(1):

""" tiles_per_block = T """
for i_block_d in nl.affine_range(unified_tiles // T):
    for i_tile_d in nl.affine_range(T):
```

Total iterations are unchanged ($\texttt{num\_blocks} \times T = \texttt{unified\_tiles}$). `tiles_per_block` itself does not change buffer sizes or DMA — it creates a block/tile boundary that two other transforms exploit:

- **Load placement** (§5.2) positions DMA loads between block and tile loops, loading data once per block and reusing across tile iterations within the block.
- **Dimension interleaving** (§5.3) hoists the block loop above output dims while keeping the tile loop inside, enabling the flash attention section pattern.

Neither is effective without `tiles_per_block > 1` — the tile loop is `range(1)`, leaving no intra-block iterations to reuse across.

## Interaction with Multi-Buffer (§5.5)

Tile-level multi-buffer degree D and `tiles_per_block` T share the `i_tile` variable. NKI affine loops lack modular arithmetic (`i_tile % D` not expressible), so only two configurations are valid:

- **D = 1**: single buffer slot, independent of `i_tile`. `tiles_per_block` is unconstrained.
- **D = T**: `i_tile` directly indexes D buffer slots. `tiles_per_block` must equal D.

Block-level PSUM multi-buffering shares `i_block` instead, requiring `num_blocks = D` and fixing `tiles_per_block = unified / D`. The candidate generation excludes dimensions where any buffer has degree > 1.

## Representation

```python
tiles_per_block: dict[str, int]
```

Maps dimension ID to tiles_per_block in `KernelIR`. Absent dimensions default to 1.

## Candidate Generation

```python
class TilesPerBlock(Transform):
    NAME = "tiles_per_block"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        constrained_dims = {
            dim_id
            for dim_degrees in ir.buffer_degrees.values()
            for dim_id, deg in dim_degrees.items()
            if deg > 1
        }
        dim_sizes = _collect_dim_sizes(ir.ctx)
        for dim_id, max_tile in ir.ctx.dim_tiles.items():
            if dim_id in constrained_dims:
                continue
            unified = dim_sizes[dim_id] // max_tile
            current = ir.tiles_per_block.get(dim_id, 1)
            for tpb in _divisors(unified):
                if tpb <= current:
                    continue
                new_tpb = dict(ir.tiles_per_block)
                new_tpb[dim_id] = tpb
                results.append(replace(ir, tiles_per_block=new_tpb))
        return results
```

Candidates only increase `tiles_per_block` — the search graph explores all valid divisors greater than the current value in a single step. Dimensions constrained by multi-buffer (any buffer with degree > 1) are excluded to avoid invalidating existing buffer indexing.

## Example: Fused Double Matmul

Fully fused `[[0,1,2,3,4]]`, dim order $(d_0, d_4, d_2, d_1)$, all inputs `(2048, 128)`. On d2: `unified_tiles = 4`, `interleave = 4`. Setting `tiles_per_block_d2 = 2` changes the d2 loops from 4 blocks × 1 tile to 2 blocks × 2 tiles:

```python
for i_block_d0 in nl.affine_range(16):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d4 in nl.affine_range(1):
            for i_tile_d4 in nl.affine_range(1):
                psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
                for i_block_d2 in nl.affine_range(2):              """ 2 blocks (was 4) """
                    for i_tile_d2 in nl.affine_range(2):           """ 2 tiles/block (was 1) """
                        psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                        nisa.memset(dst=psum_S[0:128, 0:512], value=0.0)
                        for i_block_d1 in nl.affine_range(1):
                            for i_tile_d1 in nl.affine_range(1):
                                """ Ops 0-2: transpose Q, K; matmul → psum_S """
                        sbuf_S = nl.ndarray((128, 1, 4, 128), ...)  """ degree-1 """
                        nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
                        for i_ig_d2 in nl.affine_range(4):
                            """ Ops 3-4: transpose S; matmul → psum_output """
                sbuf_output = nl.ndarray((128, 1, 1, 128), ...)
                nisa.tensor_copy(dst=sbuf_output[...], src=psum_output[...])
                nisa.dma_copy(dst=hbm_output[...], src=sbuf_output[...])
```

S stays degree-1 — no extra SBUF cost. The boundary between `i_block_d2` and `i_tile_d2` is where subsequent transforms act:

- **Load placement** (§5.2): K/V DMA can move between block and tile, pre-loading `tiles_per_block × interleave = 8` raw sub-tiles per block.
- **Dimension interleaving** (§5.3): `i_block_d2` can hoist above d0, wrapping Q groups — K/V loaded per block are reused across all 16 d0 iterations (16× DMA reduction). This produces the flash attention section pattern: `for section_idx` (d2 block) → `for grp_i` (d0) → K/V tiles within section (d2 tile).

## Reference Kernel Mapping

| Reference kernel | nkigym equivalent | Size |
|---|---|---|
| `_V_TILE_SZ = 128` | `min_tile_size` (transpose F limit) | 128 |
| `_K_TILE_SZ = 512` | `max_tile_size` (matmul N limit) | 512 |
| `interleave = 512/128 = 4` | `interleave` | 4 |
| `section_len = 8192` | `tiles_per_block × max_tile_size` | 8192 (T=16) |
| `num_sections` | `num_blocks` | `seq_k / section_len` |
| `num_k_tiles_per_section` | `tiles_per_block` | 16 |

`for section_idx` → `i_block_d2`, `for grp_i` → `i_block_d0` (one d0 tile = 128 tokens). K/V loaded between block and tile (§5.2). The full loop structure `d2_block → d0 → d2_tile` is dim interleaving (§5.3) — `tiles_per_block > 1` makes it effective (with `tiles_per_block = 1` each section is a single tile, maximizing save/reload overhead).
