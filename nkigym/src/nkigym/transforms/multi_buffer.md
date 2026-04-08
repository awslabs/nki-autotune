# Multi-Buffer Transform

Multi-buffering increases a buffer from 1 to D slots per dimension, so consecutive loop iterations address different memory regions. The hardware overlaps one iteration's consumer with the next iteration's producer — pipelining compute or overlapping DMA with compute.

## Scope

Three categories of on-chip buffers:

- **Fusion intermediates** — SBUF buffers produced and consumed within a fused loop body. Default degree-1 from loop fusion; higher degrees pipeline producer/consumer across iterations.
- **DMA staging buffers** — HBM inputs and outputs. Higher degrees overlap DMA with compute.
- **PSUM accumulators** — matmul accumulation buffers. Higher degrees cycle through D PSUM banks, avoiding Tensor Engine / Vector Engine contention on the same bank.

All three use the same mechanism: D buffer slots indexed by a loop variable.

## Representation

`buffer_degrees` maps each buffer name to per-dimension degrees. Loop fusion and math transforms add entries at degree 1. The multi-buffer transform adds DMA staging and PSUM entries at degree D ≥ 2, or increases existing entries. `psum_levels` stores the loop level for each PSUM entry ("tile" or "block"), since the degree alone does not determine which loop drives the bank rotation:

```python
buffer_degrees: dict[str, dict[str, int]]
psum_levels: dict[str, dict[str, str]]
buffer_degrees = {"S": {"d0": 1, "d2": 2}, "Q": {"d0": 2}, "psum_S": {"d2": 4}, "psum_output": {"d2": 4}}
psum_levels = {"psum_S": {"d2": "tile"}, "psum_output": {"d2": "block"}}
```

## Buffer Sizing

### SBUF

$\texttt{num\_tiles} = \texttt{degree} \times \texttt{interleave}$. For S(d0, d2) with interleave_d0=1, interleave_d2=4:

| Degrees (d0, d2) | num_tiles_P | num_tiles_F | Buffer shape |
|---|---|---|---|
| (1, 1) | 1 | 4 | `(128, 1, 4, 128)` |
| (1, 2) | 1 | 8 | `(128, 1, 8, 128)` |
| (2, 1) | 2 | 4 | `(128, 2, 4, 128)` |

### PSUM

PSUM has 8 banks of `PSUM_BANK_SIZE` (2048) free elements each. A multi-buffered accumulator occupies D consecutive banks starting at `base_bank`. The renderer assigns `base_bank` sequentially across concurrent accumulators (those whose lifetimes overlap within the same fusion group). Total D across all concurrent accumulators must be $\leq 8$.

## Indexing

**SBUF:** `i_tile` directly indexes buffer slots:

```python
for i_block_d2 in nl.affine_range(num_blocks):
    for i_tile_d2 in nl.affine_range(D):
        idx = i_tile_d2 * interleave + i_ig
        sbuf_S[..., idx, ...]
```

**PSUM:** Allocate a single PSUM tensor spanning D consecutive banks. The driving loop variable (0..D-1) strides by `PSUM_BANK_SIZE` to address each bank via `nl.ds`:

```python
psum_acc = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                       buffer=nl.psum, address=(0, base_bank * PSUM_BANK_SIZE))
for i in nl.affine_range(D):
    nisa.nc_matmul(dst=psum_acc[0:128, nl.ds(i * PSUM_BANK_SIZE, tile_free)], ...)
```

The driving loop variable is determined by the PSUM's level — see §Loop Levels below.

## PSUM Loop Levels

PSUM multi-buffering operates at two loop levels, depending on which loop variable indexes the D banks:

| Level | D | Driving loop | Free dim | Contraction dim |
|---|---|---|---|---|
| Tile | tiles_per_block | `i_tile` | Valid | Valid |
| Block | num_blocks | `i_block` | Valid | Valid |

NKI `nl.affine_range` loops lack modular arithmetic (`i % D`), so D must equal the driving loop's trip count — `i_tile` ranges 0..D-1 only when D = tiles_per_block, and `i_block` only when D = num_blocks.

Why not interleave level (`i_ig`)? The matmul's op_tile_size on the free dim (N=512) spans all interleave groups (each 128 elements) in one call. The matmul runs at `i_tile` granularity, outside the `i_ig` loop, so it cannot use `i_ig` for bank selection.

### Free-dim PSUM

Each iteration of the driving loop produces an independent result in its own bank. While the Tensor Engine writes bank $k+1$, the Vector Engine reads bank $k$ — overlapping TE and VE.

**Tile level** (primary pattern, matches reference kernel MM1): D = tiles_per_block. The matmul runs once per tile at its op_tile_size (512 free elements), writing a full result to one bank. After each tile, `tensor_copy` saves the result to SBUF while the next tile's matmul starts on a different bank. The consumer transpose then processes the saved SBUF data at `i_ig` granularity (128 elements per group). Sets tiles_per_block = D.

```python
psum_S = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                     buffer=nl.psum, address=(0, 0))
for i_tile_d2 in nl.affine_range(D):
    nisa.memset(dst=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)], value=0.0)
    nisa.nc_matmul(dst=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)], ...)
    nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 512)])
```

**Block level**: D = num_blocks. Banks rotate per block. Within each block, all tiles share one bank. Less useful for free dims — no TE/VE overlap within a block.

### Contraction-dim PSUM

Multiple tiles accumulate into the same output — each tile's partial products add to the running sum. Multi-buffering rotates the PSUM bank at the driving loop boundary, saving each partial result to an SBUF accumulator via `nisa.tensor_tensor(add)`. This pipelines the next iteration's matmul (TE) with the previous iteration's save (VE).

**Tile level**: D = tiles_per_block. Within each tile, interleave groups accumulate into the same bank. Between tiles, the bank rotates and the partial result is added to SBUF. Sets tiles_per_block = D.

```python
psum_output = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                          buffer=nl.psum, address=(0, 4 * PSUM_BANK_SIZE))
sbuf_accum = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_accum[0:128, 0, 0, 0:128], value=0.0)
for i_tile_d2 in nl.affine_range(D):
    nisa.memset(dst=psum_output[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 128)], value=0.0)
    for i_ig_d2 in nl.affine_range(interleave):
        nisa.nc_matmul(dst=psum_output[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 128)], ...)
    nisa.tensor_tensor(dst=sbuf_accum[0:128, 0, 0, 0:128],
                       data=sbuf_accum[0:128, 0, 0, 0:128],
                       data2=psum_output[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, 128)],
                       op=nl.add)
```

The zero-initialized SBUF accumulator avoids branching on `i_tile == 0` (which affine loops can't express) — the first add is `0 + result = result`. The SBUF accumulator is the matmul's output staging buffer — no extra allocation is needed.

**Block level** (matches reference kernel MM2): D = num_blocks. Within each block, ALL tiles and interleave groups accumulate into the same bank. Between blocks, the bank rotates and the partial result is added to SBUF. Does not constrain tiles_per_block.

```python
psum_output = nl.ndarray((128, D * PSUM_BANK_SIZE), dtype=nl.float32,
                          buffer=nl.psum, address=(0, 4 * PSUM_BANK_SIZE))
sbuf_accum = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_accum[0:128, 0, 0, 0:128], value=0.0)
for i_block_d2 in nl.affine_range(D):
    nisa.memset(dst=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)], value=0.0)
    for i_tile_d2 in nl.affine_range(tiles_per_block):
        for i_ig_d2 in nl.affine_range(interleave):
            nisa.nc_matmul(dst=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)], ...)
    nisa.tensor_tensor(dst=sbuf_accum[0:128, 0, 0, 0:128],
                       data=sbuf_accum[0:128, 0, 0, 0:128],
                       data2=psum_output[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, 128)],
                       op=nl.add)
```

Block-level accumulates more partial products per bank (all tiles × interleave groups within a block) and makes fewer saves (one per block vs one per tile). Requires tiles_per_block > 1 from a prior transform and D = num_blocks $\leq$ MAX_DEGREE.

### Bank Budget

All concurrent PSUM accumulators within a fusion group share 8 banks. In the reference attention kernel with d2 (seq_k):

- MM1 (free dim, tile level): D = 4 banks (banks 0–3)
- MM2 (contraction dim, block level): D = 4 banks (banks 4–7)
- Total: 8 banks (maximum)

Both are allocated with fixed bank addresses within the fusion group scope — the renderer assigns non-overlapping ranges, so both must fit within 8 banks.

## Loop Constraints

### SBUF (tile-level only)

- **D = 1**: tiles_per_block set independently by §5.4. Buffer indexes as `0 × interleave + i_ig`.
- **D > 1**: tiles_per_block = D. $\texttt{num\_blocks} = \texttt{unified\_tiles} / D$.

All SBUF buffers at D > 1 on the same dimension must share the same D (= tiles_per_block). Buffers at D = 1 don't constrain tiles_per_block. The tiles_per_block transform (§5.4) excludes dimensions where any buffer has D > 1; multi-buffer candidates on a dimension where tiles_per_block is already T > 1 are constrained to D = T.

### PSUM

- **Tile level**: D = tiles_per_block. Same mutual constraint with SBUF tile-level and tiles_per_block transform. Valid for both free and contraction dims.
- **Block level**: D = unified_tiles / tiles_per_block. Requires tiles_per_block > 1 from a prior transform, so D is determined and won't be invalidated by later changes. D must be $\leq$ MAX_DEGREE.

Tile-level and block-level are mutually exclusive for the same PSUM on the same dimension. The `psum_levels` field stores the level explicitly — the renderer reads it directly rather than inferring from D.

## Candidate Generation

Candidate generation requires `buffer_degrees` from prior transforms (loop fusion sets degree-1 entries for fusion intermediates) and respects existing `tiles_per_block` constraints.

Two SBUF sources and one PSUM source. Source 1 increases existing `buffer_degrees` entries (fusion intermediates, or HBM dims added by a prior multi-buffer application). Source 2 adds new dimensions for HBM inputs/outputs. Source 3 adds PSUM entries at tile or block level. For tile-level, D must divide unified_tiles (since tiles_per_block = D determines num_blocks = unified_tiles / D). Tile-level D sets tiles_per_block for the dimension, so all tile-level buffers sharing that dimension's tile loop must use the same D (or stay at D=1). If tiles_per_block is already set to T > 1, tile-level candidates are constrained to D = T. For block-level, tiles_per_block must already be > 1 (set by a prior transform); D = unified_tiles / tiles_per_block.

`_apply(ir, name, dim_id, D)` returns a new `KernelIR` with `buffer_degrees[name][dim_id] = D` and `tiles_per_block[dim_id] = D`. For PSUM, also sets `psum_levels[name][dim_id] = "tile"`. `_apply_block(ir, name, dim_id, D)` sets `buffer_degrees[name][dim_id] = D` and `psum_levels[name][dim_id] = "block"` without modifying `tiles_per_block`. Only called when tiles_per_block > 1 is already set.

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
                existing_tpb = ir.tiles_per_block.get(dim_id, 1)
                for D in range(current + 1, self.MAX_DEGREE + 1):
                    if unified % D != 0:
                        continue
                    if existing_tpb > 1 and D != existing_tpb:
                        continue
                    results.append(self._apply(ir, tensor_name, dim_id, D))
        """Source 2: HBM staging (tile-level) — add dims not yet in buffer_degrees"""
        for name in (*ir.hbm_inputs, *ir.hbm_outputs):
            existing_dims = ir.buffer_degrees.get(name, {})
            for dim_id in tensor_dims(name):
                if dim_id in existing_dims:
                    continue
                unified = dim_size[dim_id] // max_tile_size[dim_id]
                existing_tpb = ir.tiles_per_block.get(dim_id, 1)
                for D in range(2, self.MAX_DEGREE + 1):
                    if unified % D != 0:
                        continue
                    if existing_tpb > 1 and D != existing_tpb:
                        continue
                    results.append(self._apply(ir, name, dim_id, D))
        """Source 3: PSUM — tile-level and block-level on free or contraction dim.
        Skip partition dim (M): PSUM banks are free-axis addressed only."""
        for op in ir.matmul_ops:
            current_degrees = ir.buffer_degrees.get(op.psum_name, {})
            for dim_id in op.loop_dims:
                if dim_id == op.partition_dim:
                    continue
                current = current_degrees.get(dim_id, 0)
                """Tile-level: both free and contraction dims. Sets tiles_per_block = D."""
                unified = dim_size[dim_id] // max_tile_size[dim_id]
                existing_tpb = ir.tiles_per_block.get(dim_id, 1)
                for D in range(max(2, current + 1), self.MAX_DEGREE + 1):
                    if unified % D != 0:
                        continue
                    if existing_tpb > 1 and D != existing_tpb:
                        continue
                    results.append(self._apply(ir, op.psum_name, dim_id, D))
                """Block-level: D = num_blocks. Only when tiles_per_block > 1
                (set by prior transform), so D won't be invalidated later."""
                if existing_tpb > 1:
                    num_blocks = unified // existing_tpb
                    if num_blocks >= 2 and num_blocks <= self.MAX_DEGREE and num_blocks > current:
                        results.append(self._apply_block(ir, op.psum_name, dim_id, num_blocks))
        return results
```

Persistent state buffers — running accumulators from online fusion (e.g., `mm1_running_max`, `exp_running_sum`) that carry values across loop iterations via read-modify-write — must be excluded from all three sources. Multi-buffering assigns each iteration a separate slot, breaking the serial dependency these buffers require. The IR tracks persistent buffer names; candidate generation skips them.

The bank budget constraint (total PSUM banks $\leq 8$) is not checked during candidate generation — the compiler rejects infeasible allocations.

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

All mappings reference the attention CTE kernel (`attention_cte.py`). The section dimension (seq_k, our d2) has unified_tiles=16, tiles_per_block=4, interleave=4, num_blocks=4.

**DMA staging (tile-level on d0).**

| Buffer | D | Mechanism |
|---|---|---|
| `q_sb` | 2 | DMA-compute overlap across Q groups |
| `mm2_sb` | 2 | writeback overlaps next group's compute |
| `k_sb`, `v_sb` | full-range | pre-loaded per section (§5.2, not multi-buffering) |

**PSUM (d2).** MM1 and MM2 use different loop levels on the same dimension:

| Accumulator | Level | D | Banks | Why |
|---|---|---|---|---|
| `mm1_psum` | tile | 4 | 0–3 | d2 is MM1's free dim (N). Each K tile (512 tokens = 1 unified tile) produces an independent 128×512 result → `i_tile_d2` indexes banks |
| `mm2_psum` | block | 4 | 4–7 | d2 is MM2's contraction dim (K). Each large tile (2048 tokens = 1 block of 4 K tiles) accumulates a partial sum → `i_block_d2` indexes banks; partial result accumulated into `mm2_sb` via `tensor_tensor(add)` per block |

Reference code: `mm1_psum` address = `(k_tile_idx % 4) * PSUM_BANK_SIZE`; `mm2_psum` address = `(4 + large_tile_idx % 4) * PSUM_BANK_SIZE`. Mapping: k_tile_idx → `i_tile_d2` (512 tokens = max_tile_size), large_tile_idx → `i_block_d2` (2048 tokens = 4 tiles = 1 block). Both D=4 with trip_count=4, so `% 4` is identity — expressible in affine loops.

**Fusion intermediates and online fusion state (tile-level on d0).**

| Buffer | D | Role |
|---|---|---|
| `mm1_masked` | 2 | masked MM1 output, overlaps QK with exp across groups |
| `exp_sb` | 1–4 | exp output, varies by config |
| `prev_mm1_running_max` | 2 | temporary copy for correction factor |
| `prev_exp_running_sum` | 2 | temporary copy for flash attention rescaling |
| `flash_attn_correction_factor` | 2 | exp(prev_max − curr_max) |
| `mm1_running_max`, `exp_running_sum` | 1 | persistent across all sections, accumulated in-place |

## Limitations

**D < trip_count.** The reference kernel's `ModularAllocator` handles `index % D` via Python-level list indexing (e.g., `mm2_sb` D=2 with trip=num_grps, `mm1_copy_sb` D=2 with trip=4 K tiles). Our framework uses `nl.affine_range` where `i % D` is non-affine. To support D < trip_count, the loop would need splitting: `for i_outer in range(T // D): for i_slot in range(D):` — a future codegen extension.

Our framework CAN express tile-level D = tiles_per_block (e.g., D=2 with tiles_per_block=2), providing local pipelining within each block. What it CANNOT express is D < trip_count — rotating a small buffer (D=2) across many iterations (trip >> 2) of a single loop. The reference kernel's software pipelining across Q groups uses this D < trip_count pattern for all DMA staging and online fusion state buffers at D=2. Without the loop-split extension, these specific patterns are unreachable; the framework can still pipeline these buffers at tile-level (D = tiles_per_block), which provides per-block overlap rather than cross-loop overlap.

All reference kernel multi-buffering patterns where D = trip_count (both PSUM allocations: MM1 tile-level D=4=tiles_per_block, MM2 block-level D=4=num_blocks) are fully supported.
