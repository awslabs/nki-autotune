## Per-Group Loop Nests

After the header and buffer allocations, `render_ir` emits one loop nest per fusion group as a sibling block. Each group's nest iterates over *its own* `group_dim_orders` entry — the complete set of dims any op in the group touches.

There is **no DP-vs-reduction split** at the render level. A dim's blocking status (`DimInfo.is_blocking`) only affects fusion legality, not which loops exist or where they sit. Each dim in a group's `dim_order` contributes two loops:

| Loop | Variable | Trip count |
|---|---|---|
| Block | `i_block_d{id}` | `dim_size / (ltiles_per_block * logical_tile_size)` |
| Logical tile | `i_ltile_d{id}` | `ltiles_per_block` |

Both are always emitted, even when trip count is 1. Trip counts come from `DimInfo` and `ltiles_per_block`.

**Physical-tile iteration is per-op, not part of the kernel nest.** Each NKIOp knows its own op tile size on each dim (`da.op_tile_sizes[op_idx][dim_id]`) and the dim's `physical_tile_size`. Physical-tile packing is encapsulated inside the op's emission. Following the pattern we already use for DMA (`load_tensor_block`, `store_tensor_block`), a per-op gadget hides the inner ptile loop so the kernel source stays at two visible levels per dim.

Loops within one group are grouped by phase — all block loops outermost, then all logical tile loops. Dimension order within each phase is taken from `group_dim_orders[group_idx]`. Groups are emitted in topological order of the group-level DAG; ties broken by minimum `op_idx`.

### Default IR: singleton groups

`build_ir` produces `fusion_groups = [[0], [1], ..., [n-1]]`. Each group's `dim_order` is every dim that op touches. No dim sits outside a group that doesn't depend on it — an op only loops over its own dims.

This is maximally unfused. Loop fusion merges groups; the rejection sampler redraws each group's `dim_order` on every call. Both operate on `fusion_groups`, not on the renderer.

### Example: Attention default

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`, `ltiles_per_block = 1`:

| Dim | dim_size | logical_tile_size | physical_tile_size | blocking? | block | tile |
|---|---|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | no | 16 | 1 |
| d1 | 128 | 128 | 128 | yes | 1 | 1 |
| d2 | 2048 | 512 | 128 | yes | 4 | 1 |
| d4 | 128 | 128 | 128 | no | 1 | 1 |

Op graph (11 ops):

```
[0] transpose Q ──→ [2] matmul QK ──→ [3] affine_select ──→ [4] tensor_scalar ─┬→ [5] tensor_reduce ─┐
[1] transpose K ──↗                                                              └→ [6] act_reduce ←───┘
                                                                                      ├→ [7] activation ──→ [10] tensor_scalar
                                                                                      └→ [8] transpose ──→ [9] matmul SV ──↗
```

Default `fusion_groups = [[0], [1], …, [10]]`. Default `group_dim_orders`, sorted per group:

| Group | Op | Dim order |
|---|---|---|
| 0 | nc_transpose (Q) | `[d0, d1]` |
| 1 | nc_transpose (K) | `[d1, d2]` |
| 2 | nc_matmul (QK) | `[d0, d1, d2]` |
| 3 | affine_select | `[d0, d2]` |
| 4 | tensor_scalar | `[d0, d2]` |
| 5 | tensor_reduce (max) | `[d0, d2]` |
| 6 | activation_reduce | `[d0, d2]` |
| 7 | activation (reciprocal) | `[d0]` |
| 8 | nc_transpose (exp_S) | `[d0, d2]` |
| 9 | nc_matmul (SV) | `[d0, d2, d4]` |
| 10 | tensor_scalar (scale) | `[d0, d4]` |

Each group emits its own nest. Group 1 (K transpose) loops only over `d1, d2` — no spurious `d0` loop. Group 7 loops only over `d0`:

```python
# Group 1: nc_transpose [dims: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_ltile_d1 in range(1):
            for i_ltile_d2 in range(1):
                ...

# Group 7: activation [dims: d0]
for i_block_d0 in range(16):
    for i_ltile_d0 in range(1):
        ...
```

Compare to the old DP-outermost layout, which wrapped group 1's body in `for i_block_d0 in range(16): for i_block_d4 in range(1): for i_ltile_d0 in range(1): for i_ltile_d4 in range(1):` — running K transpose 16× for no reason, because `d0` was forced outermost even for ops that don't touch it.

Inside each innermost `i_ltile_*` body, the op's own gadget handles physical-tile packing on its dim(s). `nc_matmul` (groups 2, 9) covers the full d2 logical tile of 512 in a single ISA call — no ptile iteration. `nc_transpose` on d2 (groups 1, 8) only covers 128 per call, so the transpose gadget internally loops `num_ptiles_per_ltile = 4` times. Vector-engine ops on d2 (groups 3–6) behave the same way as transpose.

This is the default lowering — each op singleton, each group's nest tight to its own dims. The reference attention_cte kernel is the result of online fusion (math preprocessing) followed by loop fusion and a sampler draw over `dim_order` + `tensor_placements` on top of this baseline.
