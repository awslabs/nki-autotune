## Dimension Analysis

A forward pass over all ops produces three results per dimension: an ID, a tile size, and a **blocking vs non-blocking** classification. It also records per-tensor **logical shape information**: `(name, dim_ids, shape, dtype)`. Memory location (`hbm`/`sbuf`/`psum`) is **not** a dim-analysis concern — code-generation consults each op's `ISA_LOC` directly at its own call site.

### 1. Dimension ID Assignment

Each op maps its abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from prior ops) provide the mapping; operands without dim_ids get fresh IDs allocated. When two operands of the same op share an abstract axis (e.g., matmul's K axis appears in both stationary and moving), their concrete dims are **unified** — the later operand's dim is renamed to match the earlier one, propagating backward through all tensors that share the old ID.

Only axes the tensor actually has get mapped. When an op collapses a dimension (e.g. `tensor_reduce` reduces F, producing a 1D output), the reduced axis is gone — downstream ops consuming that 1D tensor see fewer dims, not a spurious size-1 dim. For example, `sum_exp: (d0,)` fed to `activation(data=sum_exp)` maps only P=d0; the F axis declared in `OPERAND_AXES` is skipped because the tensor has no second dimension.

### 2. Dimension Tile Sizes

Each op declares hardware tile limits per abstract axis via `TILE_LIMITS`. These come from the PE array geometry — e.g., `nc_matmul` uses (K=128, M=128, N=512), `nc_transpose` uses (P=128, F=128), vector-engine ops use (P=128, F=512).

Two tile sizes are derived per dimension:

**Logical tile size** — the iteration granularity. Each loop step processes one logical tile's worth of data. Computed as the max of all op tile limits on the dimension, clamped to dim_size:

$$\text{logical\_tile\_size} = \min(\max(\text{op\_tile\_sizes}),\ \text{dim\_size})$$

**Physical tile size** — the buffer granularity. Buffers are allocated in physical-tile-sized slots. Computed as the min of all op tile limits on the dimension. If the dimension ever appears as the first (partition) axis of any tensor, the partition hardware limit (128) is included:

$$\text{physical\_tile\_size} = \min(\text{op\_tile\_sizes},\ [128\ \text{if partition}])$$

The **number of physical tiles per logical tile** `num_ptiles_per_ltile = logical_tile_size / physical_tile_size` is how many physical slots make up one logical tile in the buffer.

Within one logical tile, each op:
- Iterates `logical_tile_size / op_tile_size` times (the physical tile sub-loop packs multiple op invocations)
- Consumes `op_tile_size / physical_tile_size` physical buffer slots per invocation

### 3. Blocking Classification

A dimension is **blocking** if any op touching it has it listed in `BLOCKING_AXES` — the op accumulates over the dim and its output is only valid after the dim's full iteration completes. All other dimensions are **non-blocking** — each iteration produces a complete per-iteration result.

The classification drives loop-fusion legality. A blocking shared dim between two groups constrains ordering; a non-blocking shared dim is freely permutable. Online fusion can flip a blocking dim to non-blocking by injecting running state + correction factors, which enlarges the legal fusion space downstream.

This replaces the older data-parallel-vs-reduction split, which tied classification to the kernel's return tensor. Classifying by `BLOCKING_AXES` is a per-op property of the actual computation, independent of which dims happen to appear in the return, and it generalizes cleanly once online fusion converts reductions into non-blocking running state.

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

For dimension tile sizes, elementwise/reduce ops don't impose a binding F constraint — their SBUF limit is far above the matmul/transpose tiles. The binding constraints come from `nc_matmul` N=512 and `nc_transpose` P/F=128.

| Dimension | Semantic | Size | Binding ops | Partition? | logical_tile_size | physical_tile_size | num_ptiles_per_ltile |
|---|---|---|---|---|---|---|---|
| d0 | seq_q | 4096 | transpose(P=128), matmul(M=128) | Yes | 128 | 128 | 1 |
| d1 | d_k | 128 | transpose(F=128), matmul(K=128) | Yes | 128 | 128 | 1 |
| d2 | seq_k | 4096 | transpose(P=128), matmul(N=512) | Yes | 512 | 128 | 4 |
| d4 | d_v | 128 | matmul(N=512) | No | 128 | 128 | 1 |

d0: logical_tile_size = `min(max(128, 128), 4096) = 128`. physical_tile_size = `min(128, 128, 128) = 128`. num_ptiles_per_ltile = 1.

d1: logical_tile_size = `min(max(128, 128), 128) = 128`. physical_tile_size = `min(128, 128, 128) = 128`. num_ptiles_per_ltile = 1.

d2: logical_tile_size = `min(max(128, 512), 4096) = 512`. physical_tile_size = `min(128, 512) = 128`, partition cap applied → `min(128, 128) = 128`. num_ptiles_per_ltile = 512/128 = 4. Per op:
- nc_matmul: `512/512 = 1` call per logical tile, `512/128 = 4` physical slots per call → 1 × 4 = 4.
- transpose/VE: `512/128 = 4` calls per logical tile, `128/128 = 1` physical slot per call → 4 × 1 = 4.

d4: logical_tile_size = `min(max(512), 128) = 128` (clamped to dim_size). physical_tile_size = `min(512, 128) = 128`, not partition → no cap. num_ptiles_per_ltile = 1.

### Step 3 — Blocking classification

Per-op `BLOCKING_AXES`: `nc_matmul` has `{K}`, `tensor_reduce` and `activation_reduce` have `{F}`, all others empty. Mapping abstract → concrete:

- Op 2 (matmul): K → d1 → **d1 blocking**.
- Op 5 (reduce_max): F → d2 → **d2 blocking**.
- Op 6 (activation_reduce): F → d2 → **d2 blocking** (already known).
- Op 9 (matmul): K → d2 → **d2 blocking** (already known).

Final classification: **d1, d2 blocking**; **d0, d4 non-blocking**. Matches the old DP/reduction result for this example — but the new rule derives it from `BLOCKING_AXES`, not from the return tensor, so it remains correct after online fusion flips specific dims.

