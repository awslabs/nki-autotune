# Tiles Per Block

*Single-loop-nest transform — operates on one fusion group's loop nest. Fusing loop nests is handled by online fusion and loop fusion.*

`ltiles_per_block` groups consecutive dimension tiles into blocks, changing the dimension's iteration granularity and staging buffer sizes.

$$\texttt{dimension\_tiles} = \frac{\texttt{dim\_size}}{\texttt{max\_tile\_size}} = \texttt{num\_blocks} \times \texttt{tiles\_per\_block}$$

Must divide `dimension_tiles`. Default is 1. Setting to T:

1. **Loop trip count.** The dimension's loop iterates `num_blocks` times instead of `dimension_tiles`.
2. **Staging buffers.** Buffers for tensors that depend on the dimension grow from 1 to T tiles on that dimension's axis. `load_tensor_block` has built-in loops over all tile slots — the buffer shape drives how many tiles are loaded per call.
3. **Compute iteration.** Single-tile ops (e.g. `nc_matmul`) iterate over the T tiles in the buffer within each block.

The transform applies independently to each dimension — any combination of per-dimension values is valid as long as each divides the dimension's `dimension_tiles`.

Two other transforms build on the block structure:

- **Load placement** hoists loads across other dimensions' loops for cross-dimension reuse. Orthogonal to ltiles_per_block, which determines the same-dimension buffer granularity.
- **Dimension interleaving** separates the block-level iteration from within-block processing with other dimensions' loops in between, enabling section-based processing. Requires $\texttt{num\_blocks} > 1$.

## Example

Matmul from loop_reordering.md: `lhs_T(K=d0, M=d1) × rhs(K=d0, N=d2) → result(d1, d2)` with d0 (K, tile=128, 16 dimension tiles), d1 (M, tile=128, 16 dimension tiles), d2 (N, tile=512, 4 dimension tiles). Order (d0, d1, d2). lhs_T depends on (d0, d1); rhs depends on (d0, d2).

**Before** — `ltiles_per_block_d0 = 1` (default, 16 blocks of 1 tile):

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

**After** — `ltiles_per_block = {"d0": 4}` (4 blocks of 4 tiles):

```python
sbuf_lhs_T = nl.ndarray((128, 4, 1, 128), buffer=nl.sbuf)
sbuf_rhs = nl.ndarray((128, 4, 1, 512), buffer=nl.sbuf)
psum_output = nl.ndarray((128, 16, 4, 512), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_output[0:128, 0:16, 0:4, 0:512], value=0.0)
for i_d0 in range(4):
    for i_d1 in range(16):
        load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*4*128, free_ofs=i_d1*128)
        for i_d2 in range(4):
            load_tensor_block(dst=sbuf_rhs, src=rhs, par_ofs=i_d0*4*128, free_ofs=i_d2*512)
            for i_k in range(4):
                nisa.nc_matmul(dst=psum_output[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs_T[0:128, i_k, 0, 0:128],
                    moving=sbuf_rhs[0:128, i_k, 0, 0:512])
save_tensor_block(dst=output, src=psum_output, par_ofs=0, free_ofs=0)
```

Changes:

- **Buffers grow on d0.** `sbuf_lhs_T` from (128, **1**, 1, 128) to (128, **4**, 1, 128). `sbuf_rhs` from (128, **1**, 1, 512) to (128, **4**, 1, 512). Both tensors depend on d0 (par axis), so their par tile count grows to `ltiles_per_block`.
- **Loop trip count.** d0 loop: `range(16)` → `range(4)`. Offset: `i_d0*128` → `i_d0*4*128` (block start).
- **Load granularity.** `load_tensor_block` reads the buffer shape and loads all 4 par tiles per call (built-in internal loop). One call per block replaces what was one call per tile.
- **Compute iteration.** `for i_k in range(4)` iterates over the 4 K tiles in the buffer. Each `nc_matmul` indexes `sbuf_lhs_T[0:128, i_k, 0, 0:128]` and `sbuf_rhs[0:128, i_k, 0, 0:512]`.
- **PSUM unchanged.** PSUM sizing depends on which output dims are inside K (loop reordering), not on ltiles_per_block.

### Valid values for d0

| T | num_blocks | Divisor of 16? |
|---|---|---|
| 1 | 16 | yes (default) |
| 2 | 8 | yes |
| 4 | 4 | yes |
| 8 | 2 | yes |
| 16 | 1 | yes |

$T = 16$ collapses to a single block — interleaving becomes a no-op (`num_blocks = 1`), and the buffer holds all 16 tiles (equivalent to full placement on d0). The search explores all divisors; each creates a different trade-off between block count and tiles per block.

## Representation and Candidates

```python
ltiles_per_block: dict[str, int]
```

Maps dimension ID to ltiles_per_block in `KernelIR`. Absent dimensions default to 1.

```python
class TilesPerBlock(Transform):
    NAME = "ltiles_per_block"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        constrained_dims = {
            dim_id
            for dim_degrees in ir.buffer_degrees.values()
            for dim_id, deg in dim_degrees.items()
            if deg > 1 and deg == ir.ltiles_per_block.get(dim_id, 1)
        }
        dim_sizes = _collect_dim_sizes(ir.ctx)
        for dim_id, max_tile in ir.ctx.dim_tiles.items():
            if dim_id in constrained_dims:
                continue
            unified = dim_sizes[dim_id] // max_tile
            current = ir.ltiles_per_block.get(dim_id, 1)
            for tpb in _divisors(unified):
                if tpb <= current:
                    continue
                new_tpb = dict(ir.ltiles_per_block)
                new_tpb[dim_id] = tpb
                results.append(replace(ir, ltiles_per_block=new_tpb))
        return results
```

Candidates only increase `ltiles_per_block` — all valid divisors greater than the current value are explored in a single step. Dimensions where tile-level multi-buffer has degree $D > 1$ are excluded — changing `ltiles_per_block` would break the buffer's tile-loop indexing (detected by $D = \texttt{tiles\_per\_block}[d]$, since tile-level multi-buffer sets both `buffer_degrees` and `ltiles_per_block` to $D$). Values are constrained to divisors of `dimension_tiles`, so some block sizes are unreachable for input sizes where `dimension_tiles` has few factors.
