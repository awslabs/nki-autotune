## Dimension Analysis

A forward pass over all ops produces three results per dimension: an ID, a tile size, and a data-parallel classification.

### 1. Dimension ID Assignment

Each op maps its abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from prior ops) provide the mapping; operands without dim_ids get fresh IDs allocated. When two operands of the same op share an abstract axis (e.g., matmul's K axis appears in both stationary and moving), their concrete dims are **unified** — the later operand's dim is renamed to match the earlier one, propagating backward through all tensors that share the old ID.

Only axes the tensor actually has get mapped. When an op collapses a dimension (e.g. `tensor_reduce` reduces F, producing a 1D output), the reduced axis is gone — downstream ops consuming that 1D tensor see fewer dims, not a spurious size-1 dim. For example, `sum_exp: (d0,)` fed to `activation(data=sum_exp)` maps only P=d0; the F axis declared in `OPERAND_AXES` is skipped because the tensor has no second dimension.

### 2. Dimension Tile Size

Each op's tile size per dimension comes from hardware tile limits of the PE array. The Trainium PE array operates on tiles of (128, 128) or (128, 512) depending on the axis role. ISA ops consume these hardware tiles — e.g., `nc_matmul` uses (K=128, M=128, N=512), `nc_transpose` uses (P=128, F=128).

**Partition axis constraint.** On-chip buffers have a physical partition dimension capped at 128. If a dimension ever appears as the first (partition) axis of any tensor, its tile size is capped at 128 regardless of op tile limits. This is a hardware constraint in addition to the per-op tile limits.

For each dimension `d{i}`:

```python
max_op_tile = max(op_tile_size[op, d{i}] for all ops touching d{i})
is_ever_partition = any(tensor.dim_ids[0] == d{i} for tensor in all tensors with d{i})
d{i}_tile_size = min(max_op_tile, 128) if is_ever_partition else max_op_tile
d{i}_tile_size = min(d{i}_tile_size, dim_size)
d{i}_min_tile_size = min(op_tile_size[op, d{i}] for all ops touching d{i})
d{i}_min_tile_size = min(d{i}_min_tile_size, dim_size)
```

Per-op behavior depends on how its hardware tile compares to the unified tile:

| Case | Condition | Behavior |
|---|---|---|
| Sub-tile | `op_tile < di_tile` | Op processes a sub-region within one buffer tile. Iterates `di_tile / op_tile` times per tile, indexing via reshape. |
| Exact | `op_tile == di_tile` | One-to-one. Op processes one buffer tile directly. |
| Multi-tile | `op_tile > di_tile` | Op consumes `op_tile / di_tile` buffer tiles. Reshapes them into a contiguous operand. |

The buffer's `num_tiles` on a dimension includes the interleave factor from the largest op: `ig = max(op_tile_size[op, d{i}]) / d{i}_tile_size`. This ensures the buffer holds enough slots for the largest op's tile. `ig ≥ 1` always, since `d{i}_tile_size ≤ max(op_tile_size)` (it's capped, not raised).

### 3. Data-Parallel Classification

A dimension is **data-parallel** if it appears in the kernel's return tensor — iterating over it produces independent partial results. All other dimensions are **reduction** — the full iteration must complete before the result is valid.

## Attention Example

```python
def attention(Q, K, V, scale):
    Q_t = GymTranspose()(Q)                    # op0
    K_t = GymTranspose()(K)                    # op1
    S = GymMatmul()(Q_t, K_t)                  # op2
    masked_S = GymAffineSelect()(...)          # op3
    scaled_S = GymTensorScalar()(masked_S, op0="multiply", operand0=scale)  # op4
    neg_max_S = GymTensorReduce()(op="max", data=scaled_S, axis=1, negate=True)  # op5
    exp_S, sum_exp = GymActivationReduce()(op="exp", data=scaled_S, reduce_op="add", bias=neg_max_S)  # op6
    inv_sum = GymActivation()(op="reciprocal", data=sum_exp)  # op7
    exp_S_t = GymTranspose()(exp_S)            # op8
    attn = GymMatmul()(exp_S_t, V)             # op9
    output = GymTensorScalar()(attn, op0="multiply", operand0=inv_sum)  # op10
    return output
```

Inputs: `Q: (seq_q, d_k)`, `K: (seq_k, d_k)`, `V: (seq_k, d_v)`.

### Step 1 — ID assignment

Each row shows the op, the action taken, and the dim_ids of **all tensors after that op**. Unification propagates backward — when d3→d1 at op2, K and K_t are retroactively updated.

| Op | Action | Q | Q_t | K | K_t | S | masked_S | scaled_S | neg_max_S | exp_S | sum_exp | inv_sum | exp_S_t | V | attn | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | transpose(Q): alloc d0,d1 | (d0,d1) | (d1,d0) | | | | | | | | | | | | | |
| 1 | transpose(K): alloc d2,d3 | (d0,d1) | (d1,d0) | (d2,d3) | (d3,d2) | | | | | | | | | | | |
| 2 | matmul(Q_t,K_t): **d3→d1** | (d0,d1) | (d1,d0) | (d2,**d1**) | (**d1**,d2) | (d0,d2) | | | | | | | | | | |
| 3 | affine_select(S) | " | " | " | " | (d0,d2) | (d0,d2) | | | | | | | | | |
| 4 | tensor_scalar(masked_S) | " | " | " | " | " | " | (d0,d2) | | | | | | | | |
| 5 | tensor_reduce(scaled_S) | " | " | " | " | " | " | " | (d0,) | | | | | | | |
| 6 | act_reduce(scaled_S) | " | " | " | " | " | " | " | " | (d0,d2) | (d0,) | | | | | |
| 7 | activation(sum_exp) | " | " | " | " | " | " | " | " | " | " | (d0,) | | | | |
| 8 | transpose(exp_S) | " | " | " | " | " | " | " | " | " | " | " | (d2,d0) | | | |
| 9 | matmul(exp_S_t,V): alloc d3,d4, **d3→d2** | " | " | " | " | " | " | " | " | " | " | " | " | (**d2**,d4) | (d0,d4) | |
| 10 | tensor_scalar(attn) | " | " | " | " | " | " | " | " | " | " | " | " | " | " | (d0,d4) |

`"` = unchanged from prior row. Two unification events: d3→d1 at op2 (shared `d_k`), d3→d2 at op9 (shared `seq_k`). Final dimensions: d0 (seq_q), d1 (d_k), d2 (seq_k), d4 (d_v).

### Step 2 — Tile sizes

Hardware tile limits per op (P is always 128):

| Op | P | F | Notes |
|---|---|---|---|
| nc_matmul | K=128, M=128 | N=512 | PE array systolic tile |
| nc_transpose | 128 | 128 | Tensor Engine transpose |
| tensor_scalar, activation, activation_reduce, affine_select | 128 | SBUF capacity | Bounded by SBUF partition size, not a fixed tile |
| tensor_reduce | 128 | SBUF capacity | Reduces free axis; same SBUF bound |

For dimension tile size, elementwise/reduce ops don't impose a binding F constraint — their SBUF limit is far above the matmul/transpose tiles. The binding constraints come from `nc_matmul` N=512 and `nc_transpose` P/F=128.

The partition axis constraint additionally affects d2 and d4: d2 appears as the partition axis of K `(d2, d1)`, exp_S_t `(d2, d0)`, etc.; d4 appears as partition of V `(d2, d4)` — no, V has d2 first. Actually d4 only appears as the free axis of V and output, never as partition. So only d2 is capped.

| Dimension | Semantic | Size | Binding ops | Partition? | d{i}_tile_size | d{i}_min_tile_size |
|---|---|---|---|---|---|---|
| d0 | seq_q | 4096 | transpose(P=128), matmul(M=128) | Yes (Q, S, ...) | 128 | 128 |
| d1 | d_k | 128 | transpose(F=128), matmul(K=128) | Yes (Q_t, K_t) | 128 | 128 |
| d2 | seq_k | 4096 | transpose(P=128), matmul(N=512) | Yes (K, exp_S_t) | 128 | 128 |
| d4 | d_v | 128 | matmul(N=512) | No | 512 | 512 |

d2: `max(op_tiles) = 512` from matmul N, but d2 appears as partition of K `(d2, d1)` and exp_S_t `(d2, d0)` → capped to 128. The buffer holds 128-element d2 tiles. nc_matmul needs 512 elements of d2 per call (N=512), so it consumes `ig = 512/128 = 4` buffer tiles and reshapes them into a contiguous 512-wide operand. Transpose and VE ops use 128 per call — no reshape needed.

d4: never partition → `d4_tile_size = 512` from matmul N. But `dim_size = 128 < 512` → clamped to 128. So `d4_tile_size = 128 = d4_min_tile_size`, no interleave.

### Step 3 — Data-parallel classification

Return tensor `output` has dims (d0, d4). These are **data-parallel**. Dimensions d1 and d2 are **reduction** — d1 is the first matmul's accumulation axis, d2 is the second matmul's accumulation axis and the reduce_max/reduce_sum axis.

