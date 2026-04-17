# Multi-Buffer Transform

*Single-loop-nest transform — operates on one fusion group's loop nest. Fusing loop nests is handled by online fusion and loop fusion.*

Multi-buffering increases a buffer from 1 to D slots so consecutive loop iterations address different memory regions. The hardware overlaps one iteration's consumer with the next iteration's producer — pipelining compute or overlapping DMA with compute.

Three categories of buffers:

- **DMA staging** (SBUF) — HBM inputs/outputs. Higher degrees overlap DMA with compute across block iterations.
- **Fusion intermediates** (SBUF) — buffers produced and consumed within a fused loop body. Higher degrees pipeline producer/consumer across block iterations.
- **PSUM accumulators** — matmul accumulation buffers. Higher degrees cycle through D PSUM banks, avoiding TE/VE contention on the same bank.

## SBUF Multi-Buffering

The buffer grows by `buffer_degree` rotating slots. Each slot holds one block's worth of tiles:

$$\texttt{num\_tiles} = \texttt{ltiles\_per\_block} \times \texttt{num\_ptiles\_per\_ltile} \times \texttt{buffer\_degree}$$

With degree 1, DMA overwrites the single slot each iteration. With degree D, each iteration loads into a distinct slot — DMA for block $k+1$ overlaps compute on block $k$. Compute indexes with $\texttt{i\_block} \times S + \texttt{i\_ltile} \times \texttt{num\_ptiles\_per\_ltile} + \texttt{i\_ptile}$ where $S = \texttt{ltiles\_per\_block} \times \texttt{num\_ptiles\_per\_ltile}$.

Before (buffer_degree=1, ltiles_per_block=4, num_ptiles_per_ltile=1, num_blocks=4). Only d0 shown; d1 loop encloses this fragment:

```python
sbuf_lhs_T = nl.ndarray((128, 4, 1, 128), buffer=nl.sbuf)
for i_block_d0 in range(4):
    load_tensor_block(dst=sbuf_lhs_T, src=lhs_T, par_ofs=i_block_d0*4*128, free_ofs=i_block_d1*128)
    for i_ltile_d0 in range(4):
        nisa.nc_matmul(..., stationary=sbuf_lhs_T[0:128, i_ltile_d0, 0, 0:128], ...)
```

After (buffer_degree=4):

```python
sbuf_lhs_T = nl.ndarray((128, 16, 1, 128), buffer=nl.sbuf)
for i_block_d0 in range(4):
    load_tensor_block(dst=sbuf_lhs_T[0:128, i_block_d0*4:(i_block_d0+1)*4, 0:1, 0:128],
        src=lhs_T, par_ofs=i_block_d0*4*128, free_ofs=i_block_d1*128)
    for i_ltile_d0 in range(4):
        nisa.nc_matmul(..., stationary=sbuf_lhs_T[0:128, i_block_d0*4+i_ltile_d0, 0, 0:128], ...)
```

Buffer grows from (128, **4**, 1, 128) to (128, **16**, 1, 128) — 4 rotating slots of 4 tiles. Previous blocks persist for DMA-compute overlap.

**Constraints:** buffer_degree must divide num_blocks (same modular arithmetic limitation as PSUM — each block maps to a unique slot only when D divides the trip count). Independent of ltiles_per_block — they control different concerns (buffer_degree: software pipelining, ltiles_per_block: DMA-to-compute ratio). Different buffers on the same dimension can have different buffer_degrees.

## PSUM Multi-Buffering

Without multi-buffering, the Tensor Engine (TE) and Vector Engine (VE) contend on the same PSUM bank — one must wait for the other. With D banks, TE writes to bank $k+1$ while VE reads bank $k$, overlapping the two engines.

PSUM has 8 banks of `PSUM_BANK_SIZE` (2048) free elements each. A multi-buffered accumulator occupies D consecutive banks. The driving loop variable (0..D-1) strides by `PSUM_BANK_SIZE` via `nl.ds`:

```python
psum_acc = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                       buffer=nl.psum, address=(0, base_bank * PSUM_BANK_SIZE))
for i in range(D):
    nisa.nc_matmul(dst=psum_acc[0:128, nl.ds(i * PSUM_BANK_SIZE, tile_free)], ...)
```

The driving loop operates at one of two levels within the group's nest:

| Level | D | Driving loop |
|---|---|---|
| Tile | ltiles_per_block | `i_ltile_d{id}` (innermost logical-tile loop on the dim) |
| Block | num_blocks | `i_block_d{id}` (outermost block loop on the dim) |

NKI `range` lacks modular arithmetic (`i % D`), so D must equal the driving loop's trip count.

The dimension being multi-buffered determines the accumulation pattern:

### Free-dim PSUM

The matmul's free dimension (e.g., N in C=A@B) — each tile produces an independent, complete result. No cross-tile accumulation needed.

**Tile level:** D = ltiles_per_block. Matmul writes a full result per tile; `tensor_copy` saves to SBUF while the next tile starts on a different bank:

```python
psum_S = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                     buffer=nl.psum, address=(0, 0))
for i_ltile_d2 in range(D):
    nisa.memset(dst=psum_S[0:128, nl.ds(i_ltile_d2 * PSUM_BANK_SIZE, 512)], value=0.0)
    nisa.nc_matmul(dst=psum_S[0:128, nl.ds(i_ltile_d2 * PSUM_BANK_SIZE, 512)], ...)
    nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[0:128, nl.ds(i_ltile_d2 * PSUM_BANK_SIZE, 512)])
```

### Contraction-dim PSUM

The matmul's contraction dimension (e.g., K in C=A@B) — multiple tiles accumulate partials into one bank. Since PSUM banks rotate, partial results can't stay in PSUM across driving-loop iterations. Instead, each iteration drains its bank into a zero-initialized SBUF accumulator via `nisa.tensor_tensor(add)`.

**Block level:** D = num_blocks. All logical tiles and physical tiles within a block share one bank:

```python
psum_output = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                          buffer=nl.psum, address=(0, 4 * PSUM_BANK_SIZE))
sbuf_accum = nl.ndarray((128, 1, 1, 128), dtype=output.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_accum[0:128, 0, 0, 0:128], value=0.0)
for i_block_d2 in range(D):
    nisa.memset(dst=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)], value=0.0)
    for i_ltile_d2 in range(ltiles_per_block):
        for i_ptile_d2 in range(num_ptiles_per_ltile):
            nisa.nc_matmul(dst=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)], ...)
    nisa.tensor_tensor(dst=sbuf_accum[0:128, 0, 0, 0:128],
                       data1=sbuf_accum[0:128, 0, 0, 0:128],
                       data2=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)],
                       op=nl.add)
```

**Tile level:** Same pattern with `i_ltile_d{id}` driving bank rotation instead of `i_block_d{id}`.

**Constraints:**
- Tile level: constrains ltiles_per_block = D. Valid for both free and contraction dims.
- Block level: D = num_blocks. Requires ltiles_per_block > 1 (from a prior transform) and num_blocks $\le$ 8 (the PSUM bank count, `PSUM_NUM_BANKS`).
- Tile and block levels are mutually exclusive for the same PSUM on the same dimension.
- Total D across all concurrent accumulators in a fusion group must be $\le$ 8 (the PSUM bank count).

## Candidate Generation

Three sources, all skipping persistent state buffers (online fusion accumulators). `MAX_DEGREE = 8` is the PSUM bank count — it caps both PSUM multi-buffering (hardware limit) and SBUF multi-buffering (chosen for consistency; SBUF has no equivalent bank constraint):

1. **Increase existing SBUF entries:** For each buffer already in `buffer_degrees` with degree $\ge 1$, try D from current+1 to MAX_DEGREE where D divides num_blocks on the dim.
2. **New HBM staging dims:** For each HBM input/output dimension not yet keyed in `buffer_degrees`, try D from 2 to MAX_DEGREE where D divides num_blocks on the dim.
3. **PSUM:** For each matmul's non-partition dims, try tile-level (sets `ltiles_per_block[dim] = D` and `buffer_degrees[(out, dim)] = D`) and block-level (D = num_blocks on the dim, requires existing `ltiles_per_block[dim] > 1`).

`_apply(ir, name, dim_id, D)` sets `buffer_degrees` only. `_apply_psum_tile` also sets `ltiles_per_block=D`; `_apply_psum_block` leaves `ltiles_per_block` unchanged. The renderer infers tile vs block level from the driving loop it selects for a PSUM accumulator — currently `buffer_degrees[(psum_tensor, dim_id)] == ltiles_per_block[dim_id]` signals tile level, otherwise block level. *(A dedicated `psum_levels: dict[tuple[str, str], str]` field is a proposed extension; today this is derived and not stored in `KernelIR`.)*
