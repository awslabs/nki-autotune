---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-25"
geometry: margin=1in
---

# NKIGym Guide

**NKIGym** takes a user function written with `nkigym.*` ops and explores execution strategies on Trainium hardware. Four IR levels:

| Level | Representation | Section |
|---|---|---|
| 1. Math Function | User-written `nkigym.*` ops | §1 |
| 2. Parsed Ops + Analysis | Op calls, dimension unification, pass assignment | §2 |
| 3. Schedule | Loop order, blocking, placements | §3 |
| 4. NKI Kernel | Rendered NKI source code | §4 |

**Running example**: naive single-head attention `softmax(Q @ K^T) @ V`. Inputs use standard ML layout `(seq, hidden)`:

- `q: float16[512, 128]` — `(seq_q, d_k)`
- `k: float16[512, 128]` — `(seq_k, d_k)`
- `v: float16[512, 1024]` — `(seq_k, d_v)`
- output: `float16[512, 1024]` — `(seq_q, d_v)`

---

## 1. Math Function

```python
def attention(q, k, v):
    q_t = nkigym.transpose(q)
    k_t = nkigym.transpose(k)
    scores = nkigym.nc_matmul(q_t, k_t)
    max_s = nkigym.tensor_reduce(scores, op=np.max)
    shifted = nkigym.tensor_scalar(scores, max_s, op0=np.subtract)
    exp_s = nkigym.activation(shifted, op=np.exp)
    sum_exp = nkigym.tensor_reduce(exp_s, op=np.add)
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    attn = nkigym.tensor_scalar(exp_s, inv_sum, op0=np.multiply)
    attn_t = nkigym.transpose(attn)
    output = nkigym.nc_matmul(attn_t, v)
    return output
```

Ops use return-value style. Every `nkigym.*` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented.

**`nc_matmul(stationary, moving)`** computes `stationary^T @ moving`. Both inputs have K (contraction) at dim 0.

- `nc_matmul(q_t, k_t)`: q_t`(d_k, seq_q)`.T @ k_t`(d_k, seq_k)` = scores`(seq_q, seq_k)`. This is Q @ K^T.
- `nc_matmul(attn_t, v)`: attn_t`(seq_k, seq_q)`.T @ v`(seq_k, d_v)` = output`(seq_q, d_v)`. This is attn @ V.

**`transpose(q)`** and **`transpose(k)`** put the contraction dim `d_k` at position 0 for the first matmul. **`transpose(attn)`** swaps `(seq_q, seq_k)` → `(seq_k, seq_q)` for the second matmul. All three are real hardware operations (`nisa.nc_transpose`), not free views.

**Softmax normalization** uses multiply-by-reciprocal (NKI ISA has no divide): `nisa.activation(op=nl.reciprocal)` computes `1/x`, then `tensor_scalar` multiplies `exp_s * (1/sum_exp)`.

---

## 2. Parsed Ops + Analysis

### 2.1 Parse

`parse_body()` extracts one `_OpCall` per `nkigym.*` call. Each has: `stmt_type` (op class from registry), `input_vars`, `config_kwargs`, `output_var`. Post-parse reclassification converts `NKIActivation` → `NKIActivation1D` when the input is 1D.

Running example produces 11 op calls:

| # | stmt_type | input_vars | output_var | config_kwargs |
|---|---|---|---|---|
| 0 | `NKITranspose` | `(q,)` | `q_t` | |
| 1 | `NKITranspose` | `(k,)` | `k_t` | |
| 2 | `NKIMatmul` | `(q_t, k_t)` | `scores` | |
| 3 | `NKITensorReduce` | `(scores,)` | `max_s` | `(("op", np.max),)` |
| 4 | `NKITensorScalar` | `(scores, max_s)` | `shifted` | `(("op0", np.subtract),)` |
| 5 | `NKIActivation` | `(shifted,)` | `exp_s` | `(("op", np.exp),)` |
| 6 | `NKITensorReduce` | `(exp_s,)` | `sum_exp` | `(("op", np.add),)` |
| 7 | `NKIActivation1D` | `(sum_exp,)` | `inv_sum` | `(("op", "reciprocal"),)` |
| 8 | `NKITensorScalar` | `(exp_s, inv_sum)` | `attn` | `(("op0", np.multiply),)` |
| 9 | `NKITranspose` | `(attn,)` | `attn_t` | |
| 10 | `NKIMatmul` | `(attn_t, v)` | `output` | |

Parse does NOT add infrastructure ops (DMA loads, PSUM→SBUF staging, DMA stores). Those are generated during rendering (§4).

### 2.2 Analysis

`analyze_dims()` takes the 11 op calls + parameter shapes and:

1. **Assigns dimension IDs**: `q → (d0, d1)`, `k → (d2, d3)`, `v → (d4, d5)`.
2. **Unifies dimensions** across ops using `OPERAND_AXES` axis labels:
   - Ops 0–1 transpose: q`(d0, d1)` → q_t`(d1, d0)`. k`(d2, d3)` → k_t`(d3, d2)`.
   - Op 2 matmul1: K=d1 unifies d3→d1. Output scores → (d0, d2).
   - Op 9 transpose: attn`(d0, d2)` → attn_t`(d2, d0)`.
   - Op 10 matmul2: K=d2 unifies d4→d2. Output → (d0, d5).
3. **Computes tile sizes**: `tile_size = max(all TILE_LIMITS)`, capped at 128 when the dim appears at position 0 (partition axis) of any tensor.
4. **Classifies dimensions**: dims in the return variable → parallel; rest → reduction.

| Dim | Size | Partition axis? | tile_size | count | Type |
|---|---|---|---|---|---|
| d0 (seq_q) | 512 | yes | 128 | 4 | parallel |
| d1 (d_k) | 128 | yes (q_t, k_t) | 128 | 1 | reduction |
| d2 (seq_k) | 512 | yes (attn_t, v) | 128 | 4 | reduction |
| d5 (d_v) | 1024 | no | 512 | 2 | parallel |

d2 gets `max(N:512, K:128) = 512` from TILE_LIMITS, but the **partition-axis cap** reduces it to 128 because d2 appears at position 0 in `attn_t(d2, d0)` and `v(d2, d5)`. d1 has 1 tile — its loop is `affine_range(1)` (no-op), so it is omitted from the enumeration analysis below.

### 2.3 Pass Assignment

`assign_passes()` walks the op calls and identifies **barrier ops** — ops with cross-tile reduction (`has_reduction` = True). Each barrier increments a per-dimension pass counter.

| Barrier | Op | Reduces over | Pass |
|---|---|---|---|
| Op 2 | `NKIMatmul` (scores) | d1 | d1 pass 0 |
| Op 3 | `NKITensorReduce` (max_s) | d2 | d2 pass 0 |
| Op 6 | `NKITensorReduce` (sum_exp) | d2 | d2 pass 1 |
| Op 10 | `NKIMatmul` (output) | d2 | d2 pass 2 |

Non-barrier ops are classified by position relative to barriers:

| Classification | Ops | Description |
|---|---|---|
| **pre_compute** for (d1, 0) | 0 (transpose q), 1 (transpose k) | Transpose inputs before matmul1 |
| **pre_compute** for (d2, 1) | 4 (subtract), 5 (exp) | 2D ops before d2 pass 1 barrier |
| **inter_pass** after (d2, 1) | 7 (reciprocal) | 1D op between d2 passes 1 and 2 |
| **pre_compute** for (d2, 2) | 8 (multiply), 9 (transpose) | 2D ops before d2 pass 2 barrier |

**Inter-pass ops** are 1D column-vector operations that execute once after a pass completes (not recomputed per d2 tile). The reciprocal of `sum_exp` runs between passes 1 and 2.

---

## 3. Schedule

A `Schedule` is an immutable, hashable NamedTuple:

```python
class Schedule(NamedTuple):
    loop_order: tuple[tuple[str, int], ...]
    dim_schedules: tuple[DimSchedule, ...]
    op_placements: tuple[int, ...]
```

### 3.1 Loop Order

`loop_order` is a sequence of items `(dim_id, pass_index)`. Parallel dims use `pass_index=0`; reduction passes use `0, 1, 2, ...`.

Walking left to right, parallel dims accumulate into the **nesting context**. Reduction passes inherit the current context but don't extend it. The context determines which dims wrap a pass's loop.

**Default loop_order** for the running example:

```
((d0, 0), (d5, 0), (d2, 0), (d2, 1), (d2, 2))
```

| Pos | Item | Context | Meaning |
|---|---|---|---|
| 0 | (d0, 0) | {d0} | Outer parallel loop: seq_q |
| 1 | (d5, 0) | {d0, d5} | Inner parallel loop: d_v |
| 2 | (d2, 0) | {d0, d5} | Pass 0: sequential d2 loop (running max) |
| 3 | (d2, 1) | {d0, d5} | Pass 1: sequential d2 loop (running sum) |
| 4 | (d2, 2) | {d0, d5} | Pass 2: affine d2 loop (matmul2 accumulation) |

### 3.2 Blocking and Placements

**`DimSchedule`**: per-dim `(dim_id, tile_size, tiles_per_block)`. Default: `tpb=1`.

**`op_placements`**: per-input-param level from 0 to `num_dependent_dims`. Controls load buffer sizing via the **sentinel mechanism**:

- Level 0: all dims outside → load all tiles once (largest buffer, max reuse)
- Level N (natural): all dims active → reload each iteration (smallest buffer)

Buffer shape: `(tile_size_par, num_tiles_par, num_tiles_free_0, ..., tile_size_free_0, ...)`. `num_tiles` = `total_tiles` (outside) or `tiles_per_block` (active).

### 3.3 Enumeration

The schedule space is the cross-product of three independent axes:

**Axis 1 — Loop Orders**: permutations of the 5 items where d2 passes maintain relative order. $C(5,2) \times 2! = 20$ valid orderings.

**Axis 2 — Op Placements**: per-load level from 0 to `num_dependent_dims`. Load q has 1 effective dim (d0), load k has 1 (d2), load v has 2 (d5, d2). $2 \times 2 \times 3 = 12$ combinations.

**Axis 3 — Blocking**: divisors of total tile count per dim. d0: {1,2,4}, d5: {1,2}, d2: {1,2,4}. $3 \times 2 \times 3 = 18$ combinations.

$$\text{Full space} = 20 \times 12 \times 18 = 4320 \text{ candidates}$$

Validation (§3.4) prunes to feasible schedules. Every survivor is rendered, compiled, and benchmarked.

### 3.4 Validation

1. **SBUF partition**: dim 0 of each buffer ≤ 128.
2. **PSUM partition**: accumulator dim 0 ≤ 128.
3. **PSUM free-dim**: accumulator free-dim ≤ `ACC_FREE_DIM_LIMIT` (2048).
4. **Blocking**: `tiles_per_block` divides total tiles.

### 3.5 Schedule Transformations (Design Direction)

Exhaustive enumeration works for the attention example (4320 candidates) but won't scale to workloads with more dimensions. Schedule-to-schedule transformations enable local search:

| Transform | What changes | Performance impact |
|---|---|---|
| **Loop interchange** | Swap two adjacent items in `loop_order` | Data reuse: outer dims reuse more. 2–10x. |
| **Pass relocation** | Move a parallel dim relative to reduction passes | Eliminates redundant iterations. 2–5x. |
| **Blocking change** | Change `tiles_per_block` for one dim | DMA/compute overlap. 1.5–3x. |
| **Placement change** | Raise/lower one load's level | Memory traffic vs SBUF pressure. 1.2–2x. |

**Completeness**: adjacent swaps generate any permutation, blocking/placement each reach any valid value independently. The space is connected — any schedule reachable from any other. Constraint: same-dim passes must maintain relative order (swapping (d2,0)↔(d2,1) is invalid).

**Example — pass relocation**: Move d5 from position 1 to between passes 1 and 2:

```
Default:   ((d0, 0), (d5, 0), (d2, 0), (d2, 1), (d2, 2))
Relocated: ((d0, 0), (d2, 0), (d2, 1), (d5, 0), (d2, 2))
```

Default: passes 0–2 all nest inside the d5 loop, but passes 0–1 (softmax) don't depend on d5 — the d5 iterations are redundant work. Relocated: passes 0–1 run with context {d0} only — 2x fewer iterations (4 d0 × 4 d2 = 16 per pass, vs 4 d0 × 2 d5 × 4 d2 = 32). Pass 2 retains context {d0, d5} for the matmul2 accumulation that produces the (seq_q, d_v) output.

**What matters most for attention:**

- **Pass relocation of d5**: softmax passes (0, 1) don't depend on d5 — relocating d5 after them eliminates redundant iterations and reduces q/k DMA traffic in those passes.
- **d2 blocking**: larger blocks reduce matmul initiation overhead; the NKI compiler can interleave DMA, Tensor Engine, Vector Engine, and Scalar Engine across more tiles.
- **Load-v placement**: hoisting v (level 0) loads all v tiles once into SBUF. For attention, v is 512×1024 fp16 = 1 MB (fits in 24 MB SBUF). Eliminates redundant DMA when d2 tiles are reused.

---

## 4. NKI Kernel

The render produces NKI source from (analysis + schedule + op calls). Infrastructure (loads, staging, stores) is generated here, not during parsing.

### 4.1 Rendered Structure (Default Schedule)

```python
@nki.jit
def attention(q, k, v):
    hbm_tensor_0 = nl.ndarray((512, 1024), dtype=q.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(4):                   # d0: seq_q tiles
        for i_1 in nl.affine_range(2):               # d5: d_v tiles
            """ Pass 0: running max over d2 """
            running_max = nl.ndarray(..., buffer=nl.psum)
            for i_2 in nl.sequential_range(4):       # d2 tiles
                # load q tile → nc_transpose → tensor_copy to SBUF
                # load k tile → nc_transpose → tensor_copy to SBUF
                # matmul1 → stage to SBUF → tensor_reduce(max) into running_max

            """ Pass 1: running sum (uses completed max) """
            running_sum = nl.ndarray(..., buffer=nl.psum)
            for i_3 in nl.sequential_range(4):       # d2 tiles
                # load q, k → transpose each → matmul1 → stage
                # → subtract(running_max) → exp → reduce(add)
            # inter-pass: reciprocal(running_sum) in SBUF

            """ Pass 2: matmul2 accumulation """
            psum_acc = nl.ndarray(..., buffer=nl.psum)
            for i_4 in nl.affine_range(4):           # d2 tiles (K for matmul2)
                # load q, k → transpose each → matmul1 → stage
                # → subtract → exp → multiply(reciprocal)
                # → transpose → load v tile → matmul2 into psum_acc
            # stage psum_acc → SBUF, DMA store to hbm_tensor_0
    return hbm_tensor_0
```

**Per-pass recomputation**: transpose + matmul1 + softmax intermediates recomputed in every d2 pass because their d2-dimension outputs don't persist in SBUF across passes. The final transpose (attn → attn_t) appears only in pass 2. Inter-pass scalars (running_max, running_sum, reciprocal) persist between passes.

---

## 5. Op Registry

Ops are auto-discovered via `NKIOp.__init_subclass__`. Each subclass declares `OPERAND_AXES`, `OUTPUT_AXES`, `TILE_LIMITS`.

### 5.1 Compute Ops

| nkigym Op | ISA Call | Semantics | OPERAND_AXES | OUTPUT_AXES | TILE_LIMITS |
|---|---|---|---|---|---|
| `NKIMatmul` | `nisa.nc_matmul` | dst += stationary.T @ moving | stationary:(K,M), moving:(K,N) | (M,N) | K:128, M:128, N:512 |
| `NKITranspose` | `nisa.nc_transpose` + `nisa.tensor_copy` | Swap P↔F via PE array | data:(P,F) | (F,P) | P:128 |
| `NKITensorReduce` | `nisa.tensor_reduce` | Reduce free axis | data:(P,F) | (P,) | P:128 |
| `NKITensorScalar` | `nisa.tensor_scalar` | Binary op: 2D tile × 1D column | data:(P,F), operand0:(P,) | (P,F) | P:128 |
| `NKITensorScalarConst` | `nisa.tensor_scalar` | Compound op with literal operands | data:(P,) | (P,) | P:128 |
| `NKIActivation` | `nisa.activation` | Element-wise unary (2D) | data:(P,F) | (P,F) | P:128 |
| `NKIActivation1D` | `nisa.activation` | Element-wise unary (1D) | data:(P,) | (P,) | P:128 |
| `NKIActivationReduce` | `nisa.activation` (with reduce) | Fused activation + free-axis reduce | data:(P,F) | (P,) | P:128 |
| `NKIAdd` | `nisa.tensor_tensor` | Element-wise add | data1:(P,F), data2:(P,F) | (P,F) | — |
| `NKIMultiply` | `nisa.tensor_tensor` | Element-wise multiply | data1:(P,F), data2:(P,F) | (P,F) | — |

### 5.2 Infrastructure Ops

| nkigym Op | ISA Call | Semantics |
|---|---|---|
| `NKIDmaCopy` | `nisa.dma_copy` | HBM ↔ SBUF transfer |
| `NKITensorCopy` | `nisa.tensor_copy` | PSUM → SBUF staging |

### 5.3 Available NKI Operations

Each `nkigym.*` op maps to a real `nisa.*` call. The `nl.*` constants used as operation parameters:

| Category | `nl.*` constants | Parameter | ISA calls |
|---|---|---|---|
| Binary | `nl.add`, `nl.subtract`, `nl.multiply`, `nl.maximum` | `op0`/`op1`, `op` | `nisa.tensor_scalar`, `nisa.tensor_tensor` |
| Unary | `nl.exp`, `nl.tanh`, `nl.square`, `nl.rsqrt`, `nl.reciprocal` | `op` | `nisa.activation` |
| Reduce | `nl.add`, `nl.max` | `op` | `nisa.tensor_reduce` |

NKI ISA has **no divide**. Division is multiply-by-reciprocal: `nisa.activation(op=nl.reciprocal)` computes `1/x`, then `nisa.tensor_scalar` multiplies.

### 5.4 Classification

`has_reduction(op)` returns True when any input axis label is absent from `OUTPUT_AXES`. Both `NKIMatmul` (K ∉ output) and `NKITensorReduce` (F ∉ output) are reduction ops. Adding a new op requires: op class in `ops/`, import in `ops/__init__.py`, simulation in `simulate/simulate.py`.

---

## 6. Design Rules

**Separation of concerns**: the schedule captures *how* to execute independently of *what* to execute (from analysis). The NKI kernel is a rendering output.

**All orderings are correct**: the only constraint is that same-dim reduction passes maintain relative order. Hardware validation filters infeasible schedules.

**Combinatorial enumeration**: full space = loop orders × placements × blocking. Each schedule is an independent grid point.

**Partition-axis cap**: `tile_size = max(TILE_LIMITS)`, capped at 128 when the dim appears at position 0 of any tensor. Resolves conflicts where a dim serves as both free axis (N:512) and partition axis (K:128).

**Cross-tile reduction**: matmul accumulates in PSUM across K tiles. `tensor_reduce` accumulates in SBUF via read-modify-write.

**Sequential passes**: when a full reduction must complete before the next op starts, the dim appears multiple times in `loop_order` as `(dim_id, pass_0), (dim_id, pass_1), ...`.

**Per-pass recomputation**: d2-dimension intermediates recomputed per pass (SBUF holds one tile at a time). 1D inter-pass results persist between passes.

**IO is explicit**: `nisa.dma_copy` for HBM↔SBUF, `nisa.tensor_copy` for PSUM→SBUF. All ISA calls use `dst=`, no return values.

**Naming**: `{memspace}_tensor_N` per memory space. Input params keep user names. PSUM always `nl.float32`.
