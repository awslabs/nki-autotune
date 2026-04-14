## Dimension Analysis

A forward pass over all ops produces three results per dimension: an ID, a tile size, and a data-parallel classification.

### 1. Dimension ID Assignment

Each op maps its abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from prior ops) provide the mapping; operands without dim_ids get fresh IDs allocated. When two operands of the same op share an abstract axis (e.g., matmul's K axis appears in both stationary and moving), their concrete dims are **unified** — the later operand's dim is renamed to match the earlier one, propagating backward through all tensors that share the old ID.

### 2. Dimension Tile Size

Each op's tile size per dimension comes from ISA hardware limits on its abstract axis (e.g., `nc_matmul` N=512, `nc_transpose` P=128).

```python
unified_tile_size[d] = max(op_tile_size[op, d] for all ops touching d)
min_tile_size[d]     = min(op_tile_size[op, d] for all ops touching d)
```

Per-op interleave count: `num_ig[op, d] = unified_tile_size[d] / op_tile_size[op, d]`. Ops with smaller tile sizes iterate multiple times per unified tile.

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

| Op | Operation | Input dim_ids | Action | Output dim_ids |
|---|---|---|---|---|
| 0 | transpose(Q) | Q: none → allocate (d0, d1) | — | Q_t: (d1, d0) |
| 1 | transpose(K) | K: none → allocate (d2, d3) | — | K_t: (d3, d2) |
| 2 | matmul(Q_t, K_t) | Q_t: (d1, d0), K_t: (d3, d2) | K axis: d1 vs d3 → **unify d3→d1** | S: (d0, d2) |
| 3 | affine_select(S) | S: (d0, d2) | — | masked_S: (d0, d2) |
| 4 | tensor_scalar(masked_S) | masked_S: (d0, d2) | — | scaled_S: (d0, d2) |
| 5 | tensor_reduce(scaled_S) | scaled_S: (d0, d2) | reduces F=d2 | neg_max_S: (d0,) |
| 6 | activation_reduce(scaled_S) | scaled_S: (d0, d2) | — | exp_S: (d0, d2), sum_exp: (d0,) |
| 7 | activation(sum_exp) | sum_exp: (d0,) | — | inv_sum: (d0,) |
| 8 | transpose(exp_S) | exp_S: (d0, d2) | — | exp_S_t: (d2, d0) |
| 9 | matmul(exp_S_t, V) | exp_S_t: (d2, d0), V: none → allocate (d4, d5) | K axis: d2 vs d4 → **unify d4→d2** | attn: (d0, d5) |
| 10 | tensor_scalar(attn) | attn: (d0, d5) | — | output: (d0, d5) |

Two unification events: d3→d1 at op2 (shared `d_k`), d4→d2 at op9 (shared `seq_k`).

### Step 2 — Tile sizes

ISA tile limits per abstract axis: `nc_transpose` P=128, F=128; `nc_matmul` K=128, M=128, N=512; elementwise ops P=128, F=512; `tensor_reduce` P=128, F=512.

| Dimension | Semantic | Size | Ops touching it | unified | min |
|---|---|---|---|---|---|
| d0 | seq_q | 4096 | transpose(P), matmul(M), elementwise(P), transpose(P), matmul(M), elementwise(P) | 128 | 128 |
| d1 | d_k | 128 | transpose(F), matmul(K) | 128 | 128 |
| d2 | seq_k | 4096 | transpose(P), matmul(N), elementwise(F), reduce(F), transpose(F), matmul(K) | 512 | 128 |
| d5 | d_v | 128 | matmul(N), elementwise(F) | 512 | 512 |

d2 has the largest spread: `nc_transpose` P=128 vs `nc_matmul` N=512 → unified=512, giving interleave=4 for transpose ops on d2.

### Step 3 — Data-parallel classification

Return tensor `output` has dims (d0, d5). These are **data-parallel**. Dimensions d1 and d2 are **reduction** — d1 is the first matmul's accumulation axis, d2 is the second matmul's accumulation axis and the reduce_max/reduce_sum axis.

## Data Model

```python
@dataclass
class DimInfo:
    """Per-dimension global info, computed once by build_ir."""
    dim_size: int
    unified_tile_size: int
    min_tile_size: int


@dataclass
class TensorInfo:
    """Per-tensor info, computed once by build_ir."""
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    isa_loc: str
    """
    Where the tensor lives or is produced:
      "hbm"  — kernel input, lives in HBM
      "psum" — ISA writes output to PSUM (nc_transpose, nc_matmul)
      "sbuf" — ISA writes directly to SBUF (tensor_scalar, activation, etc.)
    """


@dataclass
class OpDimInfo:
    """Per-op per-dimension tiling info, derived from hardware limits."""
    op_tile_size: int
    num_ig: int           # unified_tile_size / op_tile_size
    tiles_per_ig: int     # op_tile_size / min_tile_size
```
