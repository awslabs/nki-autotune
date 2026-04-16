## Tensor Buffers

Every tensor needs a buffer allocation. HBM inputs get an SBUF staging buffer (`sbuf_{name}`) for DMA loads — the load gadget copies tiles from HBM into this buffer, and ops consume from it. On-chip tensors (`isa_loc` is `"sbuf"` or `"psum"`) get their primary buffer as before, plus PSUM tensors get an SBUF staging buffer when a consumer requires it. The return tensor gets both an HBM allocation (in the kernel header, for the final store destination) and an on-chip buffer here (for the intermediate compute result).

**Placement rule: allocate at the top of the loop level where the buffer lives.** A buffer that is reused across iterations of a loop is allocated outside that loop. A buffer that is recycled each iteration is allocated at the top of the loop body. In the default lowering (degree-1, no load placement), every on-chip buffer holds one tile and is consumed within the innermost DP loop body, so all allocations go at the top of the innermost DP loop body, before any reduction loops.

The reference attention CTE kernel follows this same rule: persistent buffers (running_max, running_sum) are allocated before the section loop because they survive across sections. Per-section buffers (K/V SBUF, compute temps) are allocated at the top of the section loop body, with the allocator reset to a checkpoint each iteration so memory is reused.

**Buffer shape.** SBUF and PSUM buffers use different layouts:

**SBUF** buffers use a 4D layout `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)` for 2D tensors, or `(tile_size_P, num_tiles_P)` for 1D tensors. `tile_size` is the unified (global) tile size per dimension. `num_tiles` includes the interleave factor (see below). Multi-tile SBUF is a single `nl.ndarray` allocation:

```python
sbuf_{name} = nl.ndarray(
    ({dA_tile_size}, {dA_num_tiles}, {dB_num_tiles}, {dB_tile_size}),
    dtype=nl.{dtype},
    buffer=nl.sbuf,
)
```

**PSUM** buffers must be 2D — one tile per allocation, `(partition_tile, free_tile)`. When `num_tiles > 1`, PSUM uses a Python list of 2D tiles:

```python
psum_{name} = [
    nl.ndarray(({dA_tile_size}, {dB_tile_size}), dtype=nl.{dtype}, buffer=nl.psum)
    for _ in range({total_tiles})
]
```

where `total_tiles = num_tiles_P * num_tiles_F`. PSUM tiles are indexed by flat index `[i_p * num_tiles_F + i_f]`. When `total_tiles == 1`, it's a single `nl.ndarray` (no list).

**`num_tiles` derivation.** `num_tiles` per dimension has three contributing factors:

$$\text{num\_tiles} = \text{ig} \times \text{location\_factor} \times \text{buffer\_degree}$$

1. **Interleave (`ig`)** — `max(op_tile_sizes on this dim across ops touching this tensor) / di_tile_size`. Ensures the buffer holds enough tiles for the largest op's hardware tile. When `max_op_tile == di_tile`, ig = 1 (no interleave). When `max_op_tile > di_tile` (e.g. matmul N=512 on a partition-capped dim with di_tile=128), ig = 4.

2. **Location factor** — from `load_placements[(tensor_name, dim_id)]`:

| Tier | Factor |
|---|---|
| `"per_tile"` | 1 |
| `"per_block"` | `tiles_per_block` |
| `"full"` | `num_blocks × tiles_per_block` |

where `num_blocks = dim_size / (tiles_per_block × tile_size)`.

3. **`buffer_degrees[(group_idx, tensor_name, dim_id)]`** — multi-buffering degree (e.g. double-buffering = 2).

Both `load_placements` and `buffer_degrees` are initialized in `build_ir`: `load_placements` defaults to `"per_tile"`, `buffer_degrees` defaults to `1`. With these defaults and ig=1 (no partition cap triggers), `num_tiles = 1`. The renderer derives `num_tiles` purely from KernelIR fields — no hardcoded fallbacks.

**Tile sizes** come from `dim_analysis.dims[dim_id].tile_size`.

**PSUM staging.** A PSUM tensor gets an additional SBUF staging buffer (`sbuf_{name}`) when a consumer requires it. Each op declares per-operand memory requirements via `INPUT_LOCS` (e.g. `{"stationary": "sbuf", "moving": "sbuf"}` for nc_matmul). The renderer checks each consumer: if any operand reading this tensor has `INPUT_LOCS[role] == "sbuf"`, an SBUF staging buffer is emitted. The return tensor also needs staging (dma_copy to HBM reads from SBUF).

Currently all ops declare `INPUT_LOCS = "sbuf"` for all operands, so every PSUM tensor gets staging. The consumer-driven check is generic — if a future op accepts PSUM input directly, its tensors would skip staging automatically.

**Dtype.** PSUM buffers from ops with `PSUM_DTYPE` set (nc_matmul → float32) use that dtype. All SBUF buffers use the tensor's dtype.

**Buffer naming.** `sbuf_{tensor_name}` for SBUF, `psum_{tensor_name}` for PSUM.

### Example: Attention

With partition cap: d2 is capped to `tile_size=128` (appears as partition of K, exp_S_t). nc_matmul has `op_tile=512` on d2 → `ig = 512/128 = 4`. Buffers on d2 have `num_tiles = 4` on the d2 axis. All other dims have ig=1.

SBUF buffers (4D layout):

```python
sbuf_Q = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)         # (d0, d1) ig=1,1
sbuf_K = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)         # (d2, d1) ig=1,1
sbuf_V = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)         # (d2, d4) ig=1,1
sbuf_Q_t = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)       # (d1, d0) staging
sbuf_K_t = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)       # (d1, d2) staging, d2 ig=4
sbuf_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)         # (d0, d2) staging, d2 ig=4
sbuf_masked_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # (d0, d2) d2 ig=4
sbuf_scaled_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # (d0, d2) d2 ig=4
sbuf_neg_max = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
sbuf_exp_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)     # (d0, d2) d2 ig=4
sbuf_sum_exp = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
sbuf_inv_sum = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
sbuf_exp_S_t = nl.ndarray((128, 4, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)   # (d2, d0) staging, d2 ig=4
sbuf_attn = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)      # (d0, d4) staging
sbuf_output = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)    # (d0, d4)
```

PSUM buffers (2D tiles, lists when ig > 1):

```python
psum_Q_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)             # (d1, d0) single tile
psum_K_t = [nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(4)]  # (d1, d2) 4 tiles
psum_S = [nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum) for _ in range(4)]     # (d0, d2) 4 tiles
psum_exp_S_t = [nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(4)]  # (d2, d0) 4 tiles
psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)             # (d0, d4) single tile
```

PSUM is always 2D `(partition, free)` — one hardware tile per allocation. When ig > 1, a Python list holds the tiles. Ops index with `psum_K_t[i_ig]` (list index) then `[0:128, 0:128]` (tile slice). Ops needing the full ig range (e.g. nc_matmul reading 512 = 4×128) concatenate tiles via reshape.
