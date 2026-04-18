## NKI Ops Rendering

Inside each fusion group's loop nest, the renderer emits the ISA calls for the group's ops plus any `nisa.memset` required to zero PSUM accumulators before blocking-dim loops. Each op uses `NKIOp.format_isa_call(dst_expr, operand_exprs, scalar_kwargs)` to produce the `nisa.*` call string.

Placement mirrors the DMA planner shape: `render_nki_ops(ir, op_to_group)` returns `(before_plan, after_plan)` dicts keyed by `(group_idx, depth)`, consumed by `render_group_loops`'s `before_plan` / `after_plan` hooks (see `codegen/dma.md` for the shared depth convention).

### ISA Call Placement

Every op's ISA call is emitted at the **innermost body** of its group nest — depth `2 * N`, where `N = len(group_dim_orders[group_idx])`. The ISA call reads operands from the buffers allocated for them and writes to the op's destination buffer. Ops within a group fire in source-order.

### Memset Placement

A blocking op's PSUM accumulator must be zeroed before the outermost blocking dim's loop opens, so that successive iterations accumulate rather than overwrite. For a producer op with `BLOCKING_AXES` mapping to dims in the group's `dim_order`:

- Let `i_min = min(pos(d) for d in blocking_dims ∩ dim_order)` — the outermost blocking dim's position.
- Emit `nisa.memset(psum_{name}[...], 0.0)` at **before-plan depth `i_min`** — just inside any dim at position `< i_min`, before the outermost blocking dim's block loop opens.

This pairs symmetrically with the PSUM→SBUF staging rule (`after-plan depth i_min`, see `codegen/dma.md`): memset opens the window, staging closes it, with the full blocking iteration in between.

Non-blocking producers need no memset — each ISA call fully writes its output.

### Operand Resolution

For each operand, the renderer resolves the tensor name to a buffer variable based on the producing op's `ISA_LOC` and the consuming op's `INPUT_LOCS`:

- HBM input → `sbuf_{name}` (DMA load buffer)
- SBUF on-chip tensor → `sbuf_{name}`
- PSUM tensor consumed by op requiring SBUF → `sbuf_{name}` (staging buffer)
- PSUM tensor consumed by op accepting PSUM → `psum_{name}`

### Destination Resolution

The op's output goes to `psum_{name}` or `sbuf_{name}` based on its own `ISA_LOC` class attribute.

### Tensor Indexing

Each operand is indexed into its buffer using the op's own tile size (from `dim_analysis.op_tile_sizes[op_idx]`). Buffers use physical tile sizes with `num_ptiles_per_ltile` folded into `num_tiles`. Ops slice the buffer and reshape to their own tile size:

- `op_tile == physical_tile` → one slot, trivial reshape
- `op_tile > physical_tile` → multi-slot `[0:physical_tile, 0:n, ...]`, reshape to `(op_tile_p, op_tile_f)`
- `op_tile < physical_tile` → sub-tile within one slot

Memset addresses the full buffer using physical tile sizes (no reshape).

### Example: Attention Group 2 (matmul, blocking on d1)

Group 2: `nc_matmul(Q_t, K_t) → S`. dim_order = `[d0, d1, d2]`, blocking dim d1 at position 1. N = 3.

- Memset at before-plan depth `i_min = 1` — inside `block_d0`, just before `block_d1` opens.
- ISA call at before-plan depth `2N = 6` — innermost body.
- Stage at after-plan depth `i_min = 1` — after `block_d1` closes.

```python
# Group 2: nc_matmul [dims: d0, d1, d2]
for i_block_d0 in range(16):
    nisa.memset(psum_S[0:128, 0, 0:4, 0:128], 0.0)
    for i_block_d1 in range(1):
        for i_block_d2 in range(4):
            for i_ltile_d0 in range(1):
                for i_ltile_d1 in range(1):
                    for i_ltile_d2 in range(1):
                        nisa.nc_matmul(psum_S[...], sbuf_Q_t[...], sbuf_K_t[...])
    stage_tensor_block(sbuf_S, psum_S)
```

### Example: Attention Group 0 (nc_transpose, non-blocking)

Group 0: `nc_transpose(Q) → Q_t`. dim_order = `[d0, d1]`. No blocking dims → no memset.

- ISA call at before-plan depth `2N = 4` — innermost body.
- Stage at before-plan depth `4` — immediately follows ISA (PSUM valid per-iteration).

```python
# Group 0: nc_transpose [dims: d0, d1]
for i_block_d0 in range(16):
    for i_block_d1 in range(1):
        for i_ltile_d0 in range(1):
            for i_ltile_d1 in range(1):
                nisa.nc_transpose(psum_Q_t[...], sbuf_Q[...])
                stage_tensor_block(sbuf_Q_t, psum_Q_t)
```
