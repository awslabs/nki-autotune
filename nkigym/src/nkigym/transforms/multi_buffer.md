# Multi-Buffer Transform

Multi-buffering increases a buffer from 1 to D slots per dimension, so consecutive loop iterations address different memory regions. The hardware overlaps one iteration's consumer with the next iteration's producer — pipelining compute or overlapping DMA with compute.

## Scope

Three categories of on-chip buffers:

- **Fusion intermediates** — SBUF buffers produced and consumed within a fused loop body. Default degree-1 from loop fusion; higher degrees pipeline producer/consumer across iterations.
- **DMA staging buffers** — HBM inputs and outputs. Higher degrees overlap DMA with compute.
- **PSUM accumulators** — matmul accumulation buffers. Higher degrees cycle through D PSUM banks, avoiding Tensor Engine / Vector Engine contention on the same bank.

All three use the same mechanism: D buffer slots indexed by a loop variable.

## Representation

`buffer_degrees` maps each buffer name to per-dimension degrees. Loop fusion and math transforms add entries at degree 1. The multi-buffer transform adds DMA staging and PSUM entries at degree D ≥ 2, or increases existing entries:

```python
buffer_degrees: dict[str, dict[str, int]]
buffer_degrees = {"S": {"d0": 1, "d2": 2}, "corr": {"d0": 1}, "Q": {"d0": 2}, "psum_S": {"d2": 4}}
```

## Buffer Sizing (SBUF)

$\texttt{num\_tiles} = \texttt{degree} \times \texttt{interleave}$. For S(d0, d2) with interleave_d0=1, interleave_d2=4:

| Degrees (d0, d2) | num_tiles_P | num_tiles_F | Buffer shape |
|---|---|---|---|
| (1, 1) | 1 | 4 | `(128, 1, 4, 128)` |
| (1, 2) | 1 | 8 | `(128, 1, 8, 128)` |
| (2, 1) | 2 | 4 | `(128, 2, 4, 128)` |

## Indexing

SBUF: `i_tile` directly indexes buffer slots:

```python
for i_block_d2 in nl.affine_range(num_blocks):
    for i_tile_d2 in nl.affine_range(D):
        idx = i_tile_d2 * interleave + i_ig
        sbuf_S[..., idx, ...]
```

PSUM: address = `(base_bank + i % D) × PSUM_BANK_SIZE`. Each accumulator gets D consecutive banks starting at base_bank. Concurrent accumulators use non-overlapping ranges; total D across all accumulators ≤ 8 (PSUM has 8 banks).

## Loop Constraints

NKI affine loops cannot express modular arithmetic, so the driving loop variable must range exactly 0..D-1.

**Tile-level** (SBUF buffers + PSUM on non-contraction dims): tiles_per_block = D. `i_tile` ranges 0..D-1, indexing D buffer slots.

- **D = 1**: tiles_per_block set independently by §5.4. Buffer indexes as `0 × interleave + i_ig`.
- **D > 1**: tiles_per_block = D. $\texttt{num\_blocks} = \texttt{unified\_tiles} / D$.

A dimension is controlled by either `buffer_degrees` (this transform) or `tiles_per_block` (§5.4), never both. When multiple SBUF buffers in a fused group share a tile loop, all must use the same D.

**Block-level** (PSUM on contraction dims): tiles accumulate into the same PSUM bank within a block — the bank can only change between blocks. `i_block` ranges 0..D-1, requiring num_blocks = D. Does NOT set tiles_per_block.

Which level to use for PSUM is deterministic: if the dimension is the matmul's contraction dimension (K dim), tiles accumulate per block → block-level. Otherwise each tile produces an independent result → tile-level. No annotation needed — the renderer derives the level from the op graph.

**Coexistence.** Tile-level and block-level can operate on the same dimension simultaneously when $\texttt{unified\_tiles} = D_\text{tile} \times D_\text{block}$. The two degrees are coupled: tile-level sets tiles_per_block = D_tile, which determines num_blocks = unified / D_tile = D_block. Increasing one decreases the other. In the reference kernel: d2 with unified_tiles=16, tiles_per_block=4 (D_tile=4), num_blocks=4 (D_block=4). MM1 PSUM rotates tile-level via `i_tile_d2` (banks 0–3); MM2 PSUM rotates block-level via `i_block_d2` (banks 4–7).

## Candidate Generation

Three sources. D must divide the relevant trip count: unified_tiles for tile-level, num_blocks for block-level.

```python
class MultiBuffer(Transform):
    NAME = "multi_buffer"
    MAX_DEGREE = 4

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results = []
        psum_names = {op.psum_name for op in ir.matmul_ops}
        """Source 1: increase existing SBUF entries (tile-level)"""
        for tensor_name, dim_degrees in ir.buffer_degrees.items():
            if tensor_name in psum_names:
                continue
            for dim_id, current in dim_degrees.items():
                unified = dim_size[dim_id] // max_tile_size[dim_id]
                for D in range(current + 1, self.MAX_DEGREE + 1):
                    if unified % D != 0:
                        continue
                    results.append(self._apply(ir, tensor_name, dim_id, D))
        """Source 2: HBM staging (tile-level)"""
        for name in (*ir.hbm_inputs, *ir.hbm_outputs):
            if name in ir.buffer_degrees:
                continue
            for dim_id in tensor_dims(name):
                unified = dim_size[dim_id] // max_tile_size[dim_id]
                for D in range(2, self.MAX_DEGREE + 1):
                    if unified % D != 0:
                        continue
                    results.append(self._apply(ir, name, dim_id, D))
        """Source 3: PSUM — tile-level on output dims, block-level on contraction dim"""
        for op in ir.matmul_ops:
            current_degrees = ir.buffer_degrees.get(op.psum_name, {})
            for dim_id in op.loop_dims:
                current = current_degrees.get(dim_id, 0)
                if dim_id == op.contraction_dim:
                    """block-level: D = num_blocks exactly (affine loops can't express i_block % D)"""
                    num_blocks = (dim_size[dim_id] // max_tile_size[dim_id]) // ir.tiles_per_block.get(dim_id, 1)
                    if 2 <= num_blocks <= self.MAX_DEGREE and num_blocks > current:
                        results.append(self._apply(ir, op.psum_name, dim_id, num_blocks))
                else:
                    """tile-level: D divides unified_tiles, sets tiles_per_block = D"""
                    unified = dim_size[dim_id] // max_tile_size[dim_id]
                    for D in range(max(2, current + 1), self.MAX_DEGREE + 1):
                        if unified % D != 0:
                            continue
                        results.append(self._apply(ir, op.psum_name, dim_id, D))
        return results
```

## Example

Fused matmul S → transpose S_t, intermediate S(d0, d2). Degree-2 along d2 — `buffer_degrees = {"S": {"d0": 1, "d2": 2}}`:

```python
sbuf_S = nl.ndarray((128, 1, 8, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(16):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(2):                            """ was 4, now 2 """
            for i_tile_d2 in nl.affine_range(2):                         """ was 1, now 2 (= D) """
                for i_ig_d2 in nl.affine_range(4):
                    sbuf_S[0:128, 0, i_tile_d2*4+i_ig_d2, 0:128]
                ...
```

## Reference Kernel Mapping

**DMA staging (tile-level on d0).** Q staging D=2 (DMA-compute overlap across Q groups). Output `mm2_sb` D=2 (writeback overlaps next group). K/V: full-range pre-load per section (§5.2, not multi-buffering).

**PSUM (d2).** MM1 D=4 tile-level banks 0–3 (d2 is non-contraction for MM1; each K-tile produces an independent S column). MM2 D=4 block-level banks 4–7 (d2 is contraction for MM2; 4 K-tiles accumulate per large tile). Coexistence: tiles_per_block_d2=4, num_blocks_d2=4, unified=16.

**Online fusion state (tile-level on d0).** Per-group temporary copies (correction factor, previous running max/sum) D=2. Persistent running state (`mm1_running_max`, `exp_running_sum`) is D=1 — accumulated in-place across all sections.

## Limitations

**Tile-level D < tiles_per_block.** The reference kernel's `ModularAllocator` handles `index % D` for buffers like `mm1_copy_sb` (D=2, T=4). Expressing this in nkigym would require splitting the tile loop: `for i_outer in range(T // D): for i_slot in range(D):` — a future codegen extension.

**Block-level D < num_blocks.** Same affine limitation: `i_block % D` requires splitting the block loop. Currently, block-level multi-buffering requires num_blocks = D exactly.
