# Multi-Buffer Transform

Multi-buffering increases a buffer from 1 to D slots so consecutive loop iterations address different memory regions. The hardware overlaps one iteration's consumer with the next iteration's producer — pipelining compute or overlapping DMA with compute.

Three categories of buffers:

- **DMA staging** (SBUF) — HBM inputs/outputs. Higher degrees overlap DMA with compute across block iterations.
- **Fusion intermediates** (SBUF) — buffers produced and consumed within a fused loop body. Higher degrees pipeline producer/consumer across block iterations.
- **PSUM accumulators** — matmul accumulation buffers. Higher degrees cycle through D PSUM banks, avoiding TE/VE contention on the same bank.

## SBUF Multi-Buffering

The buffer grows by `buffer_degree` rotating slots. Each slot holds one block's worth of tiles:

$$\texttt{num\_tiles} = \texttt{tpb\_hbm} \times \texttt{interleave} \times \texttt{buffer\_degree}$$

With degree 1, DMA overwrites the single slot each iteration. With degree D, each iteration loads into a distinct slot — DMA for block $k+1$ overlaps compute on block $k$. Compute indexes with $\texttt{i\_d} \times S + \texttt{i\_tile} \times \texttt{interleave} + \texttt{i\_ig}$ where $S = \texttt{tpb\_hbm} \times \texttt{interleave}$.

Before (buffer_degree=1, tpb_hbm=4, interleave=1, num_blocks=4):

```python
sbuf_lhs_T = nl.ndarray((128, 4, 1, 128), buffer=nl.sbuf)
for i_d0 in range(4):
    load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_d0*4*128, free_ofs=i_d1*128)
    for i_k in range(4):
        nisa.nc_matmul(..., stationary=sbuf_lhs_T[0:128, i_k, 0, 0:128], ...)
```

After (buffer_degree=4):

```python
sbuf_lhs_T = nl.ndarray((128, 16, 1, 128), buffer=nl.sbuf)
for i_d0 in range(4):
    load_tensor_block(dst=sbuf_lhs_T[0:128, i_d0*4:(i_d0+1)*4, 0:1, 0:128],
        src=lhs_T, par_ofs=i_d0*4*128, free_ofs=i_d1*128)
    for i_k in range(4):
        nisa.nc_matmul(..., stationary=sbuf_lhs_T[0:128, i_d0*4+i_k, 0, 0:128], ...)
```

Buffer grows from (128, **4**, 1, 128) to (128, **16**, 1, 128) — 4 rotating slots of 4 tiles. Previous blocks persist for DMA-compute overlap.

**Constraints:** buffer_degree must divide num_blocks (same modular arithmetic limitation as PSUM — each block maps to a unique slot only when D divides the trip count). Independent of tiles_per_block — they control different concerns (buffer_degree: software pipelining, tiles_per_block: DMA-to-compute ratio). Different buffers on the same dimension can have different buffer_degrees.

## PSUM Multi-Buffering

Without multi-buffering, the Tensor Engine (TE) and Vector Engine (VE) contend on the same PSUM bank — one must wait for the other. With D banks, TE writes to bank $k+1$ while VE reads bank $k$, overlapping the two engines.

PSUM has 8 banks of `PSUM_BANK_SIZE` (2048) free elements each. A multi-buffered accumulator occupies D consecutive banks. The driving loop variable (0..D-1) strides by `PSUM_BANK_SIZE` via `nl.ds`:

```python
psum_acc = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                       buffer=nl.psum, address=(0, base_bank * PSUM_BANK_SIZE))
for i in range(D):
    nisa.nc_matmul(dst=psum_acc[0:128, nl.ds(i * PSUM_BANK_SIZE, tile_free)], ...)
```

The driving loop operates at one of two levels:

| Level | D | Driving loop |
|---|---|---|
| Tile | tiles_per_block | `i_tile` |
| Block | num_blocks | `i_d` |

NKI `range` lacks modular arithmetic (`i % D`), so D must equal the driving loop's trip count.

The dimension being multi-buffered determines the accumulation pattern:

### Free-dim PSUM

The matmul's free dimension (e.g., N in C=A@B) — each tile produces an independent, complete result. No cross-tile accumulation needed.

**Tile level:** D = tiles_per_block. Matmul writes a full result per tile; `tensor_copy` saves to SBUF while the next tile starts on a different bank:

```python
psum_S = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                     buffer=nl.psum, address=(0, 0))
for i_tile_d2 in range(D):
    nisa.memset(dst=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)], value=0.0)
    nisa.nc_matmul(dst=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)], ...)
    nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)])
```

### Contraction-dim PSUM

The matmul's contraction dimension (e.g., K in C=A@B) — multiple tiles accumulate partials into one bank. Since PSUM banks rotate, partial results can't stay in PSUM across driving-loop iterations. Instead, each iteration drains its bank into a zero-initialized SBUF accumulator via `nisa.tensor_tensor(add)`.

**Block level:** D = num_blocks. All tiles x interleave groups within a block share one bank:

```python
psum_output = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                          buffer=nl.psum, address=(0, 4 * PSUM_BANK_SIZE))
sbuf_accum = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_accum[0:128, 0, 0, 0:128], value=0.0)
for i_d2 in range(D):
    nisa.memset(dst=psum_output[0:128, nl.ds(i_d2 * PSUM_BANK_SIZE, 128)], value=0.0)
    for i_tile_d2 in range(tiles_per_block):
        for i_ig_d2 in range(interleave):
            nisa.nc_matmul(dst=psum_output[0:128, nl.ds(i_d2 * PSUM_BANK_SIZE, 128)], ...)
    nisa.tensor_tensor(dst=sbuf_accum[0:128, 0, 0, 0:128],
                       data=sbuf_accum[0:128, 0, 0, 0:128],
                       data2=psum_output[0:128, nl.ds(i_d2 * PSUM_BANK_SIZE, 128)],
                       op=nl.add)
```

**Tile level:** Same pattern with `i_tile` driving bank rotation instead of `i_d`.

**Constraints:**
- Tile level: constrains tiles_per_block = D. Valid for both free and contraction dims.
- Block level: D = num_blocks. Requires tiles_per_block > 1 (from a prior transform) and num_blocks ≤ MAX_DEGREE.
- Tile and block levels are mutually exclusive for the same PSUM on the same dimension.
- Total D across all concurrent accumulators in a fusion group must be ≤ 8.

## Candidate Generation

Three sources, all skipping persistent state buffers (online fusion accumulators):

1. **Increase existing SBUF entries:** For each buffer already in `buffer_degrees`, try D from current+1 to MAX_DEGREE where D divides num_blocks.
2. **New HBM staging dims:** For each HBM input/output dimension not yet in `buffer_degrees`, try D from 2 to MAX_DEGREE where D divides num_blocks.
3. **PSUM:** For each matmul's non-partition dims, try tile-level (sets tiles_per_block = D) and block-level (D = num_blocks, requires existing tiles_per_block > 1).

`_apply(ir, name, dim_id, D)` sets `buffer_degrees` only. `_apply_psum_tile` also sets `psum_levels="tile"` and `tiles_per_block=D`. `_apply_psum_block` sets `psum_levels="block"` without constraining tiles_per_block.
