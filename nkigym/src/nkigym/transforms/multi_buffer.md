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

Example from reference attention CTE kernel with d2 coexistence (unified_tiles=16, tiles_per_block=4, num_blocks=4):

| Accumulator | Level | D | Banks |
|---|---|---|---|
| psum_S (MM1) | tile-level | 4 | 0–3 |
| psum_output (MM2) | block-level | 4 | 4–7 |

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
for i_tile_d2 in nl.affine_range(D):
    nisa.nc_matmul(dst=psum_acc[0:128, nl.ds(i_tile_d2 * PSUM_BANK_SIZE, tile_free)], ...)
```

Tile-level uses `i_tile`; block-level uses `i_block`. Banks are free-axis addressed — only the matmul output's free dimension (N) and contraction dimension (K) support PSUM multi-buffering. The partition dimension (M) cannot rotate banks.

## Loop Constraints

NKI `nl.affine_range` loops produce symbolic values restricted to affine index expressions. Modular arithmetic (`i % D`) is non-affine, so the only way to index D buffer slots from a loop variable is to have a loop that ranges exactly 0..D-1.

**Tile-level** (SBUF buffers on any dim; PSUM on free dim N only): tiles_per_block = D. `i_tile` ranges 0..D-1, indexing D buffer slots.

- **D = 1**: tiles_per_block set independently by §5.4. Buffer indexes as `0 × interleave + i_ig`.
- **D > 1**: tiles_per_block = D. $\texttt{num\_blocks} = \texttt{unified\_tiles} / D$.

Tile-level D > 1 sets tiles_per_block = D. Both this transform and tiles_per_block (§5.4) can set the value — they must agree. The tiles_per_block transform excludes dimensions where any buffer has D > 1; multi-buffer candidates on a dimension where tiles_per_block is already T > 1 are constrained to D = T. Buffers at D=1 don't constrain tiles_per_block — they index as `0 × interleave + i_ig` regardless of the tile loop's trip count. All buffers at D > 1 on the same dimension must share the same D (= tiles_per_block).

**Block-level** (PSUM on contraction dims): tiles accumulate into the same PSUM bank within a block — the bank can only change between blocks. `i_block` ranges 0..D-1, requiring num_blocks = D. Does NOT set tiles_per_block.

Block-level changes the reduction pattern. Without it (D=1), a single PSUM bank accumulates the full contraction. With D > 1, each block produces a **partial** result in its own bank. The partial results are accumulated in an SBUF buffer via `nisa.tensor_tensor(add)`:

```python
nisa.memset(dst=sbuf_accum[0:128, 0:128], value=0.0)
for i_block_d2 in nl.affine_range(D):
    nisa.memset(dst=psum_acc[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, tile_free)], value=0.0)
    for i_tile_d2 in nl.affine_range(tiles_per_block):
        for i_ig_d2 in nl.affine_range(interleave):
            nisa.nc_matmul(dst=psum_acc[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, tile_free)], ...)
    nisa.tensor_tensor(dst=sbuf_accum, data=sbuf_accum,
                       data2=psum_acc[0:128, nl.ds(i_block_d2 * PSUM_BANK_SIZE, tile_free)], op=nl.add)
```

The zero-initialized SBUF accumulator avoids branching on `i_block == 0` (which affine loops can't express). The SBUF accumulator is the matmul's output staging buffer — no extra allocation is needed. The reference kernel uses this pattern for MM2 (`mm2_sb += mm2_psum_tile` per large tile).

Which level to use for PSUM is deterministic: if the dimension is the matmul's contraction dimension (K dim), tiles accumulate per block → block-level. Otherwise each tile produces an independent result → tile-level. No annotation needed — the renderer derives the level from the op graph.

**Coexistence.** Tile-level and block-level can both be active on the same dimension. D_block is not independently chosen — it equals num_blocks, which is determined by D_tile: $D_\text{block} = \texttt{unified\_tiles} / D_\text{tile}$. Both must be $\leq$ MAX_DEGREE for the combination to be reachable. In the reference kernel: d2 with unified_tiles=16, D_tile=4 (tiles_per_block=4), D_block=4 (num_blocks=4) — the only valid pair where both $\leq 4$. MM1 PSUM rotates tile-level via `i_tile_d2` (banks 0–3); MM2 PSUM rotates block-level via `i_block_d2` (banks 4–7).

## Candidate Generation

Three sources. For tile-level, D must divide unified_tiles (since tiles_per_block = D determines num_blocks = unified_tiles / D). For block-level, D equals num_blocks exactly — there is no choice of D; the value is determined by the current tiles_per_block. Tile-level D sets tiles_per_block for the dimension, so all tile-level buffers sharing that dimension's tile loop must use the same D (or stay at D=1). If tiles_per_block is already set to a value T > 1 on a dimension, tile-level candidates for that dimension are constrained to D = T.

`_apply(ir, name, dim_id, D)` returns a new `KernelIR` with `buffer_degrees[name][dim_id] = D`. For tile-level D > 1, it also sets `tiles_per_block[dim_id] = D`. For block-level (PSUM contraction dim), tiles_per_block is unchanged — `num_blocks = D` is already satisfied by the current tiles_per_block.

Tile-level D changes num_blocks to `unified / D`. If a block-level PSUM degree D_block > 1 already exists on the same dim, num_blocks must stay = D_block, constraining tile-level D to `unified / D_block`. All tile-level sources filter candidates that violate this via `_conflicts_block_psum(ir, dim_id, unified // D)`, which returns True if any matmul's contraction dim = dim_id has a PSUM degree D_block > 1 that differs from the proposed num_blocks.

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
                    if _conflicts_block_psum(ir, dim_id, unified // D):
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
                    if _conflicts_block_psum(ir, dim_id, unified // D):
                        continue
                    results.append(self._apply(ir, name, dim_id, D))
        """Source 3: PSUM — tile-level on free dim, block-level on contraction dim.
        Skip partition dim (M): PSUM banks are free-axis addressed only."""
        for op in ir.matmul_ops:
            current_degrees = ir.buffer_degrees.get(op.psum_name, {})
            for dim_id in op.loop_dims:
                if dim_id == op.partition_dim:
                    continue
                current = current_degrees.get(dim_id, 0)
                if dim_id == op.contraction_dim:
                    """block-level: D = num_blocks exactly (affine loops can't express i_block % D)"""
                    num_blocks = (dim_size[dim_id] // max_tile_size[dim_id]) // ir.tiles_per_block.get(dim_id, 1)
                    if 2 <= num_blocks <= self.MAX_DEGREE and num_blocks > current:
                        results.append(self._apply(ir, op.psum_name, dim_id, num_blocks))
                else:
                    """tile-level on free dim: D divides unified_tiles, sets tiles_per_block = D"""
                    unified = dim_size[dim_id] // max_tile_size[dim_id]
                    existing_tpb = ir.tiles_per_block.get(dim_id, 1)
                    for D in range(max(2, current + 1), self.MAX_DEGREE + 1):
                        if unified % D != 0:
                            continue
                        if existing_tpb > 1 and D != existing_tpb:
                            continue
                        if _conflicts_block_psum(ir, dim_id, unified // D):
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

All mappings reference the attention CTE kernel (`attention_cte.py`). The section dimension (seq_k, our d2) has unified_tiles=16, tiles_per_block=4, num_blocks=4.

**DMA staging (tile-level on d0).**

| Buffer | D | Mechanism |
|---|---|---|
| `q_sb` | 2 | DMA-compute overlap across Q groups |
| `mm2_sb` | 2 | writeback overlaps next group's compute |
| `k_sb`, `v_sb` | full-range | pre-loaded per section (§5.2, not multi-buffering) |

**PSUM (d2).** MM1 and MM2 share d2 via coexistence:

| Accumulator | Level | D | Banks | Why |
|---|---|---|---|---|
| `mm1_psum` | tile-level | 4 | 0–3 | d2 is the free dim (N) for MM1; each tile produces an independent S column, so rotating across `i_tile_d2` pipelines TE/VE |
| `mm2_psum` | block-level | 4 | 4–7 | d2 is contraction for MM2; tiles accumulate within a block, so rotating across `i_block_d2` pipelines consecutive blocks |

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

**D < trip_count (both levels).** The reference kernel's `ModularAllocator` handles `index % D` via Python-level list indexing (e.g., `mm1_copy_sb` D=2 with trip=4). Our framework uses `nl.affine_range` where `i % D` is non-affine. To support D < trip_count, the loop would need splitting: `for i_outer in range(T // D): for i_slot in range(D):` — a future codegen extension.

Impact at tile-level: the reference kernel's `mm1_copy_sb` and `mm1_affine_select_output` patterns (D=2 within 4 K-tiles per large tile) cannot currently be expressed.

Impact at block-level: for the reference configuration (unified_tiles=16, tiles_per_block=4), D_block = num_blocks = 4 $\leq$ MAX_DEGREE, so block-level PSUM rotation is reachable. For larger inputs (e.g., unified_tiles=32 $\rightarrow$ num_blocks=8), D_block exceeds MAX_DEGREE. The three-level loop split (outer $\times$ D-slot $\times$ inner-tile) would keep D_block within bounds regardless of input size — matching the reference kernel's section $\times$ large_tile $\times$ K-tile structure.

All reference kernel multi-buffering patterns at the tested configuration (unified_tiles=16) with D = trip_count are fully supported.
