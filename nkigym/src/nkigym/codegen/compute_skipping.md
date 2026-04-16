## Compute Skipping

The renderer emits `if` guards inside reduction loops to skip tile computations guaranteed to produce masked-out values. This is a render-time optimization -- the skip condition is deterministic given the `affine_select` pattern, so there is no search space.

### How It Works

An `affine_select` with `on_false_value = -inf` and `cmp_op = "greater_equal"` encodes a position-dependent mask. The affine value at global element position `(p, f)` is:

$$\text{affine}(p, f) = \text{offset} + p \times \text{cm} + f \times \text{step}$$

where `cm` is `channel_multiplier` and `step` is the first element of the single `[step, count]` pair in `pattern`. Elements where `affine(p, f) >= 0` keep their input; elements where `affine(p, f) < 0` become `-inf`. After softmax, `-inf` positions contribute zero to all downstream results.

The renderer lifts this predicate to tile granularity. The guard sits inside the ig loops, so each evaluation covers one `min_tile_size_P x min_tile_size_F` region. Let `tp = min_tile_size_P`, `tf = min_tile_size_F`. A tile covering `p in [p_start, p_start + tp)` and `f in [f_start, f_start + tf)` is entirely masked when even the **most favorable corner** (maximizing `affine`) evaluates to `< 0`:

| `cm` sign | `step` sign | Most favorable corner |
|---|---|---|
| `>= 0` | `>= 0` | `(p_start + tp - 1, f_start + tf - 1)` |
| `>= 0` | `< 0` | `(p_start + tp - 1, f_start)` |
| `< 0` | `>= 0` | `(p_start, f_start + tf - 1)` |
| `< 0` | `< 0` | `(p_start, f_start)` |

For the standard causal mask (`cm=1, step=-1, offset=0`), the most favorable corner is `(p_start + tp - 1, f_start)`, giving `affine = p_start + tp - 1 - f_start`. The tile is fully masked when `affine < 0`:

$$\text{skip when } p\_start + tp \leq f\_start$$

The emitted guard is the negation: `if p_start + tp > f_start:`.

### Detection

The renderer scans `op_graph.op_classes` for `affine_select` ops satisfying:

1. `on_false_value` is `-inf`.
2. `cmp_op` is `"greater_equal"`.
3. `pattern` has exactly one `[step, count]` pair.

These fields are in `op_graph.op_all_kwargs[op_idx]` as source strings (e.g., `pattern` is `"[[-1, K.shape[0]]]"`). Since we generate one kernel per fixed input shape, the renderer evaluates these against `dim_analysis` to get concrete integers.

The `affine_select`'s axis map (`per_op_axis_maps[op_idx]`) identifies which concrete dimensions are P and F -- the two dimensions the guard depends on.

### Guard Placement

The guard goes inside the innermost loop whose variable appears in the skip condition. For the causal mask, the condition depends on d0 (P) and d2 (F) loop variables, so the guard sits inside whichever of d0's or d2's ig loop is innermost. The guard does not change the loop structure: memset stays outside, loops still iterate the full range, only the body is conditionally executed.

In the attention example, P maps to d0 (DP) and F maps to d2 (reduction):

```python
for i_block_d0 in range(16):                               # d0 DP
    ...
        nisa.memset(psum_S[...], 0.0)
        for i_block_d1 in range(1):                         # d1 reduction
            for i_block_d2 in range(4):                     # d2 reduction
                for i_tile_d1 in range(1):
                    for i_tile_d2 in range(1):
                        for i_ig_d1 in range(1):
                            for i_ig_d2 in range(4):        # F dimension ig loop
                                p_start = i_block_d0 * 128
                                f_start = i_block_d2 * 512 + i_ig_d2 * 128
                                if p_start + 128 > f_start: # guard
                                    nisa.dma_copy(...)
                                    nisa.nc_matmul(...)
                                    nisa.affine_select(...)
                                    nisa.tensor_scalar(...)
```

Offset formulas are the same as DMA indexing:

$$p\_start = i\_block \times tpb \times tile\_size + i\_tile \times tile\_size + i\_ig \times min\_tile\_size$$

### Scope

The guard covers all ops inside the F dimension's loop body — any op whose dimensions include F. Ops without F sit outside these loops and always execute.

### What the Renderer Reads

No new KernelIR fields. All information comes from existing structures:

| Information | Source |
|---|---|
| Which ops are `affine_select` | `op_graph.op_classes[op_idx]` |
| Pattern, cm, offset, cmp_op, on_false_value | `op_graph.op_all_kwargs[op_idx]` |
| Which dims are P and F | `dim_analysis.per_op_axis_maps[op_idx]` |
| Tile sizes and min_tile_sizes | `dim_analysis.dims[dim_id]` |

### Example: Attention

`seq_q = seq_k = 2048`, `d_k = d_v = 128`. Causal mask: `affine_select(S, [[-1, 2048]], 1, -inf, cmp_op="greater_equal")`. d0 min_tile = 128, d2 min_tile = 128.

Guard: `if i_block_d0 * 128 + 128 > i_block_d2 * 512 + i_ig_d2 * 128`.

| `i_block_d0` | Q rows | Iterations executed (of 16) |
|---|---|---|
| 0 | 0-127 | 1 |
| 1 | 128-255 | 2 |
| 3 | 384-511 | 4 |
| 7 | 896-1023 | 8 |
| 15 | 1920-2047 | 16 |

Each Q group executes `i_block_d0 + 1` iterations. Total: `1 + 2 + ... + 16 = 136` of 256, skipping 47%.
