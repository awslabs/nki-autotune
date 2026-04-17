## NKI Ops Rendering *(not yet enabled in `render_ir`)*

Inside each reduction group's innermost loop, the renderer emits the actual ISA calls for the ops in that group. Each op uses `NKIOp.format_isa_call(dst_expr, operand_exprs)` to produce the `nisa.*` call string.

**Physical-tile packing is per op, hidden in gadgets.** When an op's own tile size is smaller than the dimension's logical tile, the op internally loops `num_ptiles_per_ltile = logical_tile_size / op_tile_size` times. This iteration happens inside a per-op helper (analogous to the DMA gadgets) and never appears as a kernel-level `i_ptile_*` loop. Ops whose tile size equals the logical tile (e.g. `nc_matmul` on a dim whose logical tile is 512) emit a single ISA call with no inner loop.

**Operand resolution.** For each operand, the renderer looks up the tensor name and resolves it to the appropriate buffer variable:
- HBM input → `sbuf_{name}` (the DMA load buffer)
- SBUF on-chip tensor → `sbuf_{name}`
- PSUM on-chip tensor consumed by an op requiring SBUF → `sbuf_{name}` (the staging buffer)
- PSUM on-chip tensor consumed by an op accepting PSUM → `psum_{name}`

**Destination resolution.** The op's output goes to `psum_{name}` or `sbuf_{name}` based on the op's own `ISA_LOC` class attribute.

**PSUM→SBUF staging.** After an op that writes to PSUM, if the tensor has an SBUF staging buffer, emit `stage_tensor_block(sbuf_{name}, psum_{name})`. Position follows the store rule: immediately for non-blocking ops, after the blocking axis loop for blocking ops.

**Memset.** Before a blocking op's reduction loop, emit `nisa.memset(psum_{name}, 0.0)` to zero the PSUM accumulator. Position: before the blocking dimension's outermost loop in the current `loop_order`.

**Tensor indexing.** Each operand is indexed into its buffer using the **op's own tile size** (from `op_tile_sizes[op_idx]`). The buffer uses physical tile sizes with `num_ptiles_per_ltile` folded into `num_tiles`. Ops slice the buffer and reshape to their own tile size:
- `op_tile == physical_tile`: one slot, trivial reshape
- `op_tile > physical_tile`: multi-slot `[0:physical_tile, 0:n, ...]`, reshape to `(op_tile_p, op_tile_f)`
- `op_tile < physical_tile`: sub-tile within one slot

**Memset** addresses the full buffer using physical tile sizes.

### Example: Attention nc_transpose (multiple physical tiles)

Group 8: nc_transpose on exp_S `(d0, d2)` → exp_S_t `(d2, d0)`. nc_transpose tile is 128×128, but d2 has `physical_tile_size=128` (partition-capped), and matmul wants 512 → `num_ptiles_per_ltile=4`. The buffer `psum_exp_S_t` has 4 PSUM tiles. The kernel-level code is a single gadget call; the 4-way ptile loop lives inside the gadget:

```python
"""kernel-level — inside the group's innermost logical-tile loop"""
nc_transpose_block(psum_exp_S_t, sbuf_exp_S)
stage_tensor_block(sbuf_exp_S_t, psum_exp_S_t)
```

### Example: Matmul Group (single physical tile)

```python
nisa.memset(psum_result[0:128, 0, 0, 0:512], 0.0)
for i_block_d0 in range(64):
    for i_ltile_d0 in range(1):
        load_tensor_block(sbuf_lhs_T, lhs_T, ...)
        load_tensor_block(sbuf_rhs, rhs, ...)
        nisa.nc_matmul(psum_result[0:128, 0, 0, 0:512],
                       sbuf_lhs_T[0:128, 0, 0, 0:128],
                       sbuf_rhs[0:128, 0, 0, 0:512])
stage_tensor_block(sbuf_result, psum_result)
```

`nc_matmul` covers the full 512 free dim in one call, so there is no gadget wrapper — the ISA call is emitted directly. Memset before the d0 loop, nc_matmul accumulates across all d0 iterations, `stage_tensor_block` after the loop.
