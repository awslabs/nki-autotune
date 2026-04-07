# Tiles Per Block

`tiles_per_block` is a per-dimension parameter that groups unified tiles into blocks. From §2:

$$\texttt{unified\_tiles} = \frac{\texttt{dim\_size}}{\texttt{max\_tile\_size}} = \texttt{num\_blocks} \times \texttt{tiles\_per\_block}$$

Must divide `unified_tiles`. Default is 1. Increasing to T creates a two-level nest:

```python
""" tiles_per_block = 1 (default) """
for i_block_d in nl.affine_range(unified_tiles):
    for i_tile_d in nl.affine_range(1):

""" tiles_per_block = T """
for i_block_d in nl.affine_range(unified_tiles // T):
    for i_tile_d in nl.affine_range(T):
```

This creates a block/tile boundary where other transforms act: load placement (§5.2) positions DMA loads there, dim interleaving (§5.3) hoists the block loop above other dimensions. tiles_per_block itself does not change buffer sizes or DMA — the benefit comes from their combination.

## Interaction with Multi-Buffer (§5.5)

Multi-buffer degree D and `tiles_per_block` T share the `i_tile` variable. NKI affine loops lack modular arithmetic (`i_tile % D` not expressible), so only two configurations are valid:

- **D = 1**: intermediates index slot 0 regardless of `i_tile`. `tiles_per_block` is unconstrained.
- **D = T**: `i_tile` directly indexes buffer slots. `tiles_per_block` must equal D.

The candidate generation excludes dimensions where any intermediate has degree > 1.

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

## Example: Fused Double Matmul

Fully fused `[[0,1,2,3,4]]`, dim order $(d_0, d_4, d_2, d_1)$, all inputs `(2048, 128)`. On d2: `unified_tiles = 4`, `interleave = 4`. Setting `tiles_per_block_d2 = 2` (2 blocks × 2 tiles), combined with V's DMA placed between block and tile (§5.2):

```python
for i_block_d0 in nl.affine_range(16):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d4 in nl.affine_range(1):
            for i_tile_d4 in nl.affine_range(1):
                psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
                for i_block_d2 in nl.affine_range(2):              """ 2 blocks (was 4) """
                    """ V DMA here (§5.2): 2 unified = 8 raw V-tiles """
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

S stays degree-1 — no extra SBUF cost. V is loaded per d2 block, each tile iteration consuming its own slice. This is the within-section structure of the [reference attention kernel](/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py) (`_FLASH_ATTENTION_SECTION_LENGTH` = section size, K/V loaded per section via `_load_k_tile`/`_load_v_tile`). Cross-Q-group K/V reuse additionally requires dim interleaving (§5.3).

## Reference Kernel Mapping

| Reference kernel | nkigym equivalent | Size |
|---|---|---|
| `_V_TILE_SZ = 128` | `min_tile_size` (transpose F limit) | 128 |
| `_K_TILE_SZ = 512` | `max_tile_size` (matmul N limit) | 512 |
| `interleave = 512/128 = 4` | `interleave` | 4 |
| `_LARGE_TILE_SZ = 2048` | no equivalent (PSUM bank rotation) | — |
| `section_len = 8192` | `tiles_per_block × max_tile_size` | 8192 (T=16) |
| `num_sections` | `num_blocks` | `seq_k / section_len` |
| `num_k_tiles_per_section` | `tiles_per_block` | 16 |

`for section_idx` → `i_block_d2`, `for grp_i` → `i_block_d0` (one d0 tile = 128 tokens). K/V loaded between block and tile (§5.2). The full loop structure `d2_block → d0 → d2_tile` is dim interleaving (§5.3) — `tiles_per_block > 1` is its prerequisite (without the block/tile split there is nothing to hoist). nkigym does not model the reference kernel's internal `_LARGE_TILE_SZ = 2048` grouping.
