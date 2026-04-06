---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-25"
geometry: margin=1in
---

# NKIGym Guide

**Running example**: causal single-head attention `softmax(mask(scale * Q @ K^T)) @ V`. Inputs use standard ML layout `(seq, hidden)`:

- `q: float16[4096, 128]` — `(seq_q, d_k)`
- `k: float16[4096, 128]` — `(seq_k, d_k)`
- `v: float16[4096, 128]` — `(seq_k, d_v)`
- output: `float16[4096, 128]` — `(seq_q, d_v)`

---

## 1. Logical Computation

The math function defines what to compute — no loads/stores, no tiling, no memory hierarchy. Each `nkigym.*` call maps 1:1 to a real `nisa.*` ISA instruction, but the math function only specifies the logical data flow.

```python
def attention(Q, K, V, scale):
    Q_t = nkigym.nc_transpose(Q)
    K_t = nkigym.nc_transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    masked_S = nkigym.affine_select(
        pattern=[[-1, S.shape[1]]], channel_multiplier=1,
        on_true_tile=S, on_false_value=-np.inf,
        cmp_op="greater_equal")
    scaled_S = nkigym.tensor_scalar(masked_S, op0="multiply",
                                     operand0=scale)
    neg_max_S = nkigym.tensor_reduce(op="max", data=scaled_S,
                                      axis=1, negate=True)
    exp_S, sum_exp = nkigym.activation_reduce(
        op="exp", data=scaled_S, reduce_op="add",
        bias=neg_max_S)
    inv_sum = nkigym.activation(op="reciprocal", data=sum_exp)
    exp_S_t = nkigym.nc_transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, op0="multiply",
                                   operand0=inv_sum)
    return output
```

### 1.1 NKI Gym Operators

Every `nkigym.*` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented. At the logical level, an op only declares its axis semantics and a CPU simulation. No hardware constraints, tiling, or memory placement.

```python
class NKIOp:
    """Logical NKI operator — axis semantics and CPU simulation only."""
    NAME: str
    OPERAND_AXES: dict[str, tuple[str, ...]]
    OUTPUT_AXES: dict[str, tuple[str, str, ...]]

    @abstractmethod
    def __call__(self, **kwargs):
        """CPU simulation using Numpy in default float64.
        Takes input arrays + config, returns output array(s).
        """
```

#### Operator Subclasses

```python
class NKIMatmul(NKIOp):
    """stationary(K, M).T @ moving(K, N) → output(M, N).
    K is the contraction (accumulation) axis.
    """

    NAME = "nc_matmul"
    OPERAND_AXES = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES = {"output": ("M", "N")}

    def __call__(self, stationary, moving):
        return stationary.T @ moving


class NKITranspose(NKIOp):
    """data(P, F) → output(F, P). Real hardware op, not a free view."""

    NAME = "nc_transpose"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("F", "P")}

    def __call__(self, data):
        return data.T


class NKITensorScalar(NKIOp):
    """(data <op0> operand0) <op1> operand1 → output(P, F).
    operand0/operand1: scalar constant or (P,) column vector,
    broadcast across the free axis.
    reverse0/reverse1 swap operand order for non-commutative ops.
    """

    NAME = "tensor_scalar"
    OPERAND_AXES = {"data": ("P", "F"), "operand0": ("P",), "operand1": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, data, op0, operand0, reverse0=False,
                 op1=None, operand1=None, reverse1=False):
        ops = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}
        b = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        result = ops[op0](b, data) if reverse0 else ops[op0](data, b)
        if op1 is not None:
            c = operand1[..., np.newaxis] if isinstance(operand1, np.ndarray) else operand1
            result = ops[op1](c, result) if reverse1 else ops[op1](result, c)
        return result


class NKIAffineSelect(NKIOp):
    """Position-predicated element select.
    affine_value = offset + p * channel_multiplier + Σ(idx_i * step_i).
    Compares affine_value to 0; selects on_true_tile or on_false_value.
    pattern: list of [step, count] pairs describing free-axis layout.
    """

    NAME = "affine_select"
    OPERAND_AXES = {"on_true_tile": ("P", "F")}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, pattern, channel_multiplier, on_true_tile,
                 on_false_value, cmp_op="equal", offset=0):
        P = on_true_tile.shape[0]
        F = int(np.prod([n for _, n in pattern]))
        p_idx = np.arange(P)[:, np.newaxis]
        f_vals = np.array([0])
        for step, count in pattern:
            f_vals = (f_vals[:, np.newaxis] + np.arange(count) * step).ravel()
        affine = offset + p_idx * channel_multiplier + f_vals[np.newaxis, :]
        cmps = {"greater_equal": np.greater_equal, "equal": np.equal}
        mask = cmps[cmp_op](affine, 0)
        return np.where(mask, on_true_tile.reshape(P, F), on_false_value)


class NKITensorReduce(NKIOp):
    """Reduce along specified axis with optional negation.
    data(P, F) → output(P,).
    axis must be trailing free dimension(s); partition axis cannot be reduced.
    """

    NAME = "tensor_reduce"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P",)}

    def __call__(self, op, data, axis, negate=False, keepdims=False):
        reduce = {"max": np.max, "add": np.sum}
        result = reduce[op](data, axis=axis, keepdims=keepdims)
        if negate:
            result = -result
        return result


class NKIActivationReduce(NKIOp):
    """op(data * scale + bias) → output(P, F), and simultaneously
    reduce_op(output) along free axis → reduce_res(P,).
    reduce_op only supports "add".
    """

    NAME = "activation_reduce"
    OPERAND_AXES = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F"), "reduce_res": ("P",)}

    def __call__(self, op, data, reduce_op, bias=None, scale=1.0):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
        }
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        elem = fns[op](data * s + b)
        red = {"add": np.sum}[reduce_op](elem, axis=1)
        return elem, red


class NKIActivation(NKIOp):
    """output = op(data * scale + bias).
    Applies unary activation element-wise.
    """

    NAME = "activation"
    OPERAND_AXES = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F")}

    def __call__(self, op, data, bias=None, scale=1.0):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
            "rsqrt": lambda x: 1.0 / np.sqrt(x),
        }
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        return fns[op](data * s + b)
```

### 1.2 CPU Simulation

The math function is plain Python — each `nkigym.*` call dispatches to `NKIOp.__call__()` which executes the op with numpy at float64 precision. No parsing or IR needed; just call the function directly:

```python
q = np.random.randn(4096, 128)
k = np.random.randn(4096, 128)
v = np.random.randn(4096, 128)
output = attention(q, k, v, scale=1.0 / np.sqrt(128))
```

The result is the **reference output** — a float64 array that any correctly rendered and compiled NKI kernel must match (within hardware precision tolerance).

---

## 2. Initial Eager Mode Kernel

The renderer takes a math function (§1) and generates an NKI kernel: one independent loop nest per op, concatenated in math function order. Each op's `render()` emits its own complete loop nest. Within-op PSUM accumulators and DMA staging buffers are **degree 1** (one physical tile, reused every iteration). Inter-op intermediate SBUF buffers are **full-range** so that subsequent ops' independent loop nests can read from arbitrary tile positions. This is correct by construction but naive in performance: full-range intermediates far exceed hardware SBUF capacity, and no loops are shared between ops. Variant transforms (§3 programmatic, §4 math) optimize from this baseline.

### 2.1 Hardware Tile Limits

Each ISA instruction imposes per-tensor limits on tile dimensions. The renderer stores these in a lookup table keyed by op name:

| Op | Axes | Limits |
|---|---|---|
| nc_matmul | K (accumulation), M (partition), N (free) | K ≤ 128, M ≤ 128, N ≤ 512 |
| nc_transpose | P (partition), F (free) | P ≤ 128, F ≤ 128 |

These are the only ops implemented so far. Other ops (tensor_scalar, affine_select, tensor_reduce, activation_reduce, activation) follow the same partition ≤ 128 pattern with large SBUF-based free limits — they will be added as needed.

The limits are **per-tensor constraints** — a single tensor allocation must satisfy them. Whether all tensors fit simultaneously is the compiler's problem; if compilation fails due to total SBUF pressure, that variant is discarded.

### 2.2 Tile Unification and Interleave Layout

A single dimension (e.g., d2 = seq_k) may appear across multiple ops in different roles — as matmul's N (limit 512) and transpose's F (limit 128). The renderer's analysis pass computes two global quantities per dimension:

- **`dim_tiles[d]`** (unified tile) = max of all ISA limits across ops that touch dimension d, capped at the actual dimension size.
- **`dim_min_tiles[d]`** (min tile) = min of all ISA limits across ops that touch dimension d, capped at the actual dimension size.

The ratio `interleave_groups = dim_tiles[d] // dim_min_tiles[d]` is the **interleave count**. All on-chip buffers for that dimension store each tile as `(interleave_groups, min_tile)` rather than a flat `(unified_tile,)`. This is the **interleave layout**.

**Hardware constraint.** Trainium DMA cannot operate on tensors with more than 5 meaningful (non-singleton) dimensions. This limits on-chip buffer shapes to low-dimensional layouts:

- **Intra-op buffers** (staging, PSUM accumulators, temp): **2D** — `(op_tile_P, op_tile_F)`. One physical tile, reused every iteration.
- **Inter-op buffers** (SBUF intermediates between ops): **4D** — `(tile_P, total_tiles_P, total_tiles_F, tile_F)`. The tile sizes are `min_tile` for each dimension; `total_tiles = dim_size / min_tile`, which flattens blocks × interleave into a single axis.

Indexing into an inter-op buffer at block `i_block` and interleave group `i_ig`: `tile_index = i_block * intlv + i_ig`.

**Why interleave?** Different ops need different tile sizes for the same dimension. Rather than reshaping the buffer at every op boundary, we store the data at the smallest common granularity (`min_tile`) and let each op consume however many slots it needs. An op with a large tile limit (e.g., matmul N = 512) reads 4 consecutive slots at once and reshapes `(4, 128)` → `(1, 512)`. An op with a small limit (e.g., transpose F = 128) iterates one slot at a time via sub-loops.

**Chunk+reshape protocol.** Each op computes three quantities per dimension it touches:

- `chunks = unified_tile // op_tile` — how many sub-loop iterations per unified tile.
- `gpi = op_tile // min_tile` — slots consumed per chunk iteration (groups-per-iteration).
- The op loops `chunks` times, each iteration consuming `gpi` interleave slots. When `gpi > 1`, the op reshapes the full 4D inter-op buffer to merge `total_tiles_F` and `tile_F` into one dimension, then slices at `op_tile` granularity. When `gpi = 1`, no reshape is needed; slicing one slot directly yields `(tile_P, 1, 1, tile_F)`.

### 2.3 Renderer Pipeline

The renderer (`render()`) works in three stages:

**Stage 1 — AST inspection.** Parse the math function source to extract each `NKIOp()(kwargs)` call, its operand-to-variable mapping, and the output variable name. Unassigned op calls are rejected — the variable name becomes the tensor name in the generated kernel.

**Stage 2 — Dimension assignment and tile unification.** A single forward pass over all ops:

1. For each op, build a *local axis map* from abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from input specs or prior ops) provide the mapping; fresh IDs are allocated for new dimensions. This is scoped per op invocation, so two transposes on different tensors get different dimension IDs.
2. Create the output tensor for each op. If the output is consumed by a later op (detected by scanning all operand maps), it is marked as inter-op SBUF; otherwise it is the final HBM output.
3. Unify tile sizes: for each concrete dimension, compute `dim_tiles` (max across all ISA limits) and `dim_min_tiles` (min), both capped at the actual dimension size.

**Stage 3 — Per-op code emission.** For each op in source order, call `op.render(ctx, operand_map, output_name, is_final)`. Each op's render method emits a self-contained block of NKI source lines: buffer allocations, loop nests, DMA loads, ISA calls with chunk sub-loops, writeback, and optional DMA store. The renderer indents each block and concatenates them into the final kernel function.

### 2.4 Per-Op Render Structure

Each `NKIOp` subclass implements its own `render()` method. While the details vary by op, they follow a common pattern:

1. **Buffer allocation.** Allocate intra-op buffers as 2D `(op_tile_P, op_tile_F)`: a PSUM buffer for the ISA output (matmul accumulates in fp32 PSUM; transpose writes to PSUM then copies to SBUF), SBUF staging buffers for any HBM inputs. The output SBUF buffer is either inter-op 4D `(tile_P, total_tiles_P, total_tiles_F, tile_F)` for intermediates, or intra-op 2D for the final kernel output.

2. **Loop nest.** Nested loops over each output dimension, three levels deep per dimension: `i_block` (over `num_blocks`), `i_tile` (over `tiles_per_block`, always 1 in the baseline), and `i_ig` (over `chunks`, i.e., `unified // op_tile`). All three loops are always emitted even when their trip count is 1. For ops with a reduction dimension (matmul's K), the reduction loops are nested inside the output loops.

3. **DMA loads.** For each HBM operand, emit `nisa.dma_copy` from HBM into the degree-1 staging buffer. The HBM slice is computed from the block variable × unified tile + chunk variable × op_tile.

4. **ISA call with chunk reshape.** Emit the ISA instruction (e.g., `nisa.nc_matmul`, `nisa.nc_transpose`) reading from either the 2D intra-op staging buffer (HBM inputs) or the 4D inter-op buffer (prior op outputs). When an operand's `gpi > 1`, the read expression includes a `.reshape()` on the inter-op buffer to merge `total_tiles_F` and `tile_F` before slicing at `op_tile` granularity.

5. **Writeback.** Copy the result from PSUM to the output SBUF buffer (`nisa.tensor_copy`). For the final output, also emit `nisa.dma_copy` from SBUF to HBM. For inter-op intermediates, the SBUF write goes to the tile-indexed position so later ops can read it.

**Matmul specifics.** `nisa.nc_matmul` computes `stationary(K, M).T @ moving(K, N) → output(M, N)`, accumulating into PSUM in fp32. The PSUM accumulator is memset to 0 before the K reduction loop. After all K iterations, `nisa.tensor_copy` moves the result to SBUF. When the moving operand's N interleave has `gpi > 1` (e.g., d2 with interleave 4 and matmul N = 512), the read expression reshapes the 4D inter-op buffer to merge `total_tiles_F` and `tile_F`, then slices at the matmul's N tile size: `sbuf_tensor.reshape(merged_shape)[slice]`.

**Transpose specifics.** `nisa.nc_transpose` operates at a fixed 128 × 128 tile size. It reads from SBUF, writes to a 2D PSUM temp buffer `(128, 128)`, then `nisa.tensor_copy` moves the result to the output SBUF. The output inter-op buffer swaps P and F: `(tile_F, total_tiles_F, total_tiles_P, tile_P)` since the transpose flips partition and free axes. When a dimension's unified tile exceeds 128, the chunk sub-loop iterates over 128-element slices.

### 2.5 Attention Example: Op and Dimension Summary

The running example produces 11 back-to-back loop nests — one per op — communicating through full-range SBUF intermediate buffers:

| # | Op | Output dims | Consumed dim | Init |
|---|---|---|---|---|
| 0 | nc_transpose(Q) → Q_t | d1, d0 | — | — |
| 1 | nc_transpose(K) → K_t | d1, d2 | — | — |
| 2 | nc_matmul(Q_t, K_t) → S | d0, d2 | d1 | 0 |
| 3 | affine_select(S) → masked_S | d0, d2 | — | — |
| 4 | tensor_scalar(masked_S, scale) → scaled_S | d0, d2 | — | — |
| 5 | tensor_reduce(scaled_S, max) → neg_max_S | d0 | d2 | $-\infty$ |
| 6 | activation_reduce(scaled_S, neg_max_S) → (exp_S, sum_exp) | (d0, d2), (d0) | —, d2 | —, 0 |
| 7 | activation(sum_exp, reciprocal) → inv_sum | d0 | — | — |
| 8 | nc_transpose(exp_S) → exp_S_t | d2, d0 | — | — |
| 9 | nc_matmul(exp_S_t, V) → attn | d0, d5 | d2 | 0 |
| 10 | tensor_scalar(attn, inv_sum) → output | d0, d5 | — | — |

Dimensions after unification:

| Dim | Semantic | dim_size | unified | min_tile | interleave | total_tiles | chunks (matmul) | chunks (transpose) |
|---|---|---|---|---|---|---|---|---|
| d0 | seq_q | 4096 | 128 | 128 | 1 | 32 | 1 | 1 |
| d1 | d_k | 128 | 128 | 128 | 1 | 1 | 1 | 1 |
| d2 | seq_k | 4096 | 512 | 128 | 4 | 32 | 1 (N=512) | 4 (F=128) |
| d5 | d_v | 128 | 128 | 128 | 1 | 1 | 1 | — |

Only d2 has a non-trivial interleave factor: matmul wants N = 512, transpose wants F = 128, giving `unified = 512`, `min_tile = 128`, `interleave = 4`, `total_tiles = 32`. Matmul consumes all 4 slots per block in one chunk (`gpi = 4`, `chunks = 1`), reshaping `(4, 128)` → `(1, 512)`. Transpose iterates 4 chunks of 1 slot each (`gpi = 1`, `chunks = 4`), processing one 128-element slice per sub-loop iteration.

### 2.6 Buffer Degree and Shape Dimensions

Two parameters control memory footprint and compute reuse:

- **Buffer degree (full-range vs degree-1).** A full-range inter-op buffer stores all tile positions: 4D with `total_tiles = dim_size / min_tile` per dimension. A degree-1 intra-op buffer stores one physical tile: 2D `(op_tile_P, op_tile_F)`, reused every iteration. Fusion (§3.1) converts inter-op buffers to intra-op when producer and consumer share a loop.

- **`tiles_per_block`** controls arithmetic intensity — how many compute iterations share one loaded block of data before the next DMA. The initial baseline sets `tiles_per_block = 1` everywhere; §3.4 explores increasing it.

The initial baseline uses 2D intra-op for all within-op buffers and 4D inter-op for all intermediates. The total inter-op SBUF allocation far exceeds hardware capacity. Variant transforms (§3) shrink inter-op buffers by fusing loops, adjusting buffer degree, and reordering dimensions.

**Naming conventions:**
- **Loop variables**: `i_block_d{id}`, `i_tile_d{id}`, `i_ig_d{id}` (interleave group).
- **Tensor buffers**: `sbuf_{name}` for SBUF, `psum_{name}` for PSUM. DMA staging buffers reuse the input parameter name (e.g., `sbuf_Q`).

## 3. Programmatic Transforms

Programmatic transforms are mechanical rearrangements of the loop nest that do not change the math — they only change how and when ops execute. All transforms must respect the topological computation order: every op's inputs must be computed before the op runs. The math function (§1) defines one valid topological order, and the eager mode variant follows it exactly. Transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

Four transforms generate the search space.

### 3.1 Loop Fusion

Adjacent ops whose loops iterate over the same dimensions can share a single loop nest instead of having separate nests. Fusion eliminates redundant loop overhead and — critically — keeps intermediate tiles on-chip between producer and consumer, avoiding HBM round-trips. A dimension is **blocking** when the full reduction over that dimension must complete before a valid result exists (e.g., matmul accumulation, reduce_max). A dimension is **non-blocking** when partial results are already correct, just incomplete. Fusion is legal across any non-blocking boundary. It is forbidden across a blocking boundary for the blocking dimension: the reduction must complete before the consumer starts.

**Before** — Ops 3 and 4 each have their own loop nest; `sbuf_masked_S` is a full-range buffer:

```python
""" Op 3: nisa.affine_select -- S(d0, d2) -> masked_S(d0, d2) """
sbuf_masked_S = nl.ndarray((128, 32, 32, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                nisa.affine_select(dst=sbuf_masked_S[...], on_true_tile=sbuf_S[...], ...)

""" Op 4: nisa.tensor_scalar -- masked_S(d0, d2) x scale -> scaled_S(d0, d2) """
sbuf_scaled_S = nl.ndarray((128, 32, 32, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                nisa.tensor_scalar(dst=sbuf_scaled_S[...], data=sbuf_masked_S[...], op0=nl.multiply, operand0=scale)
```

**After** — single fused loop nest; `sbuf_masked_S` shrinks to per-tile:

```python
""" Ops 3+4 (fused): affine_select + tensor_scalar -> scaled_S(d0, d2) """
sbuf_scaled_S = nl.ndarray((128, 32, 32, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                sbuf_masked_S = nl.ndarray((128, 512), dtype=Q.dtype, buffer=nl.sbuf)  # intra-op 2D, not full-range
                nisa.affine_select(dst=sbuf_masked_S[...], on_true_tile=sbuf_S[...], ...)
                nisa.tensor_scalar(dst=sbuf_scaled_S[...], data=sbuf_masked_S[...], op0=nl.multiply, operand0=scale)
```

### 3.2 Load Placement

A DMA load can be **hoisted** (moved above a loop) or **sunk** (moved below a loop) from its current position in the loop nest. Hoisting means the loaded data persists across all iterations of that loop, amortizing DMA latency at the cost of a larger SBUF buffer. Sinking means the buffer is smaller (only one tile at a time) but the load executes more frequently. The slot between block and tile loops (§2.2) provides the placement points at each nesting level.

In the greedy baseline (§2.3), all loads start at the innermost position. After loop fusion and reordering, loads may need to move in either direction to balance SBUF pressure against DMA frequency.

**Before** — `sbuf_V` loaded inside the d2 reduction loop, reloaded every iteration:

```python
""" Op 9: nisa.nc_matmul -- exp_S_t x V -> attn(d0, d5), accumulate over d2 """
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                for i_block_d2 in nl.affine_range(8):
                    for i_tile_d2 in nl.affine_range(1):
                        for i_ig_d2 in nl.affine_range(4):
                            sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)  # 2D intra-op, inside d2
                            nisa.dma_copy(dst=sbuf_V[...], src=V[...])
                            nisa.nc_matmul(dst=psum_attn[...], stationary=sbuf_exp_S_t[...], moving=sbuf_V[...])
                nisa.tensor_copy(dst=sbuf_attn[...], src=psum_attn[...])
```

**After** — `sbuf_V` hoisted outside d2, loaded once per (d0, d5) tile; buffer grows from 2D intra-op to 4D:

```python
""" Op 9: nisa.nc_matmul -- exp_S_t x V -> attn(d0, d5), accumulate over d2 """
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                sbuf_V = nl.ndarray((128, 32, 1, 128), dtype=V.dtype, buffer=nl.sbuf)  # 4D, hoisted: all d2 tiles
                nisa.dma_copy(dst=sbuf_V[...], src=V[...])
                psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                for i_block_d2 in nl.affine_range(8):
                    for i_tile_d2 in nl.affine_range(1):
                        for i_ig_d2 in nl.affine_range(4):
                            nisa.nc_matmul(dst=psum_attn[...], stationary=sbuf_exp_S_t[...], moving=sbuf_V[...])
                nisa.tensor_copy(dst=sbuf_attn[...], src=psum_attn[...])
```

### 3.3 Loop Reordering

The order of dimension loops can be permuted, subject to: (1) same-dimension phase ordering (block before tile), (2) cross-dimension data dependencies (if phase B consumes a tensor produced by phase A of a different dimension, A must precede B), and (3) in-place buffer reuse constraints (a parallel dim not in the buffer's dimensions cannot wrap phases that write the buffer).

**Before** — default order d0 > d2 > d1:

```python
""" Op 2: nisa.nc_matmul -- Q_t x K_t -> S(d0, d2), default order: d0 > d2 > d1 """
for i_block_d0 in nl.affine_range(32):                          # d0 outermost
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):                   # d2 middle
            for i_tile_d2 in nl.affine_range(1):
                psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for i_block_d1 in nl.affine_range(1):           # d1 innermost (reduction)
                    for i_tile_d1 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

**After** — reordered to d2 > d0 > d1:

```python
""" Op 2: nisa.nc_matmul -- Q_t x K_t -> S(d0, d2), reordered: d2 > d0 > d1 """
for i_block_d2 in nl.affine_range(8):                           # d2 outermost
    for i_tile_d2 in nl.affine_range(1):
        for i_block_d0 in nl.affine_range(32):                  # d0 middle
            for i_tile_d0 in nl.affine_range(1):
                psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for i_block_d1 in nl.affine_range(1):           # d1 innermost (reduction)
                    for i_tile_d1 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

### 3.4 Tiles Per Block

`tiles_per_block` is a per-dimension parameter that controls the block size — how many tiles are grouped before the tile loop iterates. Larger `tiles_per_block` means larger SBUF buffers but fewer DMA round-trips: the loaded block is reused across `tiles_per_block` compute iterations. Must divide `total_tiles`.

**Before** — `tiles_per_block=1`: 8 blocks, 1 tile each:

```python
""" Op 2 (d2 dimension): tiles_per_block=1 -- 8 blocks x 1 tile """
for i_block_d2 in nl.affine_range(8):                           # 8 blocks
    for i_tile_d2 in nl.affine_range(1):                        # 1 tile per block
        psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        for i_block_d1 in nl.affine_range(1):
            for i_tile_d1 in nl.affine_range(1):
                nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
        nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

**After** — `tiles_per_block=4`: 2 blocks, 4 tiles each; DMA loads placed between block and tile loops:

```python
""" Op 2 (d2 dimension): tiles_per_block=4 -- 2 blocks x 4 tiles """
for i_block_d2 in nl.affine_range(2):                           # 2 blocks (8 / 4)
    """ DMA loads placed here -- load 4 tiles at once """
    for i_tile_d2 in nl.affine_range(4):                        # 4 tiles per block
        psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        for i_block_d1 in nl.affine_range(1):
            for i_tile_d1 in nl.affine_range(1):
                nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
        nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

## 4. Math Transforms

Math transforms restructure the algorithm itself — changing the computation to break blocking dependencies. Programmatic transforms (§3) cannot fuse loops across blocking dependencies; math transforms eliminate those dependencies so that programmatic transforms can then optimize the resulting loop nest.

### 4.1 Online Fusion

![Math function DAG — Op 7 and Op 9 are topologically independent, converging only at Op 10](diagrams/math_function_dag.png)

A **blocking dependency** exists when a consumer op requires the producer to complete its full reduction loop before the consumer can start. To identify blocking pairs, first fuse any elementwise ops between two reductions into the second reduction's body — elementwise ops don't introduce new blocking barriers. In the source-level attention pipeline:

```
tensor_reduce(max) → tensor_scalar(subtract) → activation(exp) → tensor_reduce(add)
```

The subtract and exp are elementwise. They belong to the accumulation body of the second reduction, not to a separate phase blocked by the first reduction. Fusing them into the second reduction yields the IR-level `activation_reduce` (Op 6), which computes `exp(data + bias)` and accumulates with `add` in a single instruction. The same principle applies throughout the pipeline: Ops 3-4 (affine_select, tensor_scalar) are elementwise between Op 2 (matmul) and Op 5 (reduce_max); Op 8 (transpose) is non-blocking between Op 6 (activation_reduce) and Op 9 (matmul). Op 7 (reciprocal) is topologically independent from Op 9 (§1 DAG). Op 10 (multiply) depends on Op 9's output but runs after the d2 reduction completes, outside the fused loop.

After fusing elementwise ops into their consuming reductions, three blocking pairs remain:

| Producer | Consumer | Blocking dim |
|---|---|---|
| Op 2: matmul accumulates S over d1 | Op 5: tensor_reduce(max) reduces over d2 (with Ops 3-4 fused in) | d1 |
| Op 5: tensor_reduce(max) over d2 → neg_max_S | Op 6: activation_reduce uses neg_max_S as bias | d2 |
| Op 6: activation_reduce produces exp_S using neg_max_S | Op 9: matmul accumulates exp_S_t @ V over d2 (with Op 8 fused in) | d2 |

Some of these match the **X + Accumulation** pattern from the [online fusion paper draft](/home/ubuntu/online_fusion/main.pdf), which enables **online fusion** — a math-level transformation that eliminates the blocking barrier.

#### 4.1.1 The X + Accumulation Pattern

**Standard X + Accumulation** (Algorithm 2 in the paper draft). Two sequential loops over the same blocking dimension with $K$ tiles:

```
Input: tiles V_0[1..K], V_1[1..K]
Initialize O_0 depending on the X operator
Initialize O_1 = 0

Loop 1 (X):
  for k = 1 to K:
    O_0_k = f_X(O_0_{k-1}, V_0_k)

Loop 2 (Accumulation):
  for k = 1 to K:
    B_k = g_B(O_0_K) * h_B(V_1_k)
    O_1_k = O_1_{k-1} + B_k

Output: O_1_K
```

| Component | Role |
|---|---|
| $f_X$ | X recurrence function — updates the running reduction each iteration |
| $\mathbf{O_0}_K$ | Complete X output — Loop 2 blocks on this |
| $g_B(\mathbf{O_0}_K)$ | Bias scale — the part of the bias that depends on the X output |
| $h_B(\mathbf{V_1}_k)$ | Bias input — the part of the bias that depends on per-tile input |
| $\mathbf{B}_k = g_B(\mathbf{O_0}_K) \cdot h_B(\mathbf{V_1}_k)$ | Separable bias — multiplicatively decomposes into X-dependent and input-dependent parts |

The blocking barrier exists because Loop 2 needs the complete $\mathbf{O_0}_K$ before it can compute any $\mathbf{B}_k$.

**Fused X + Accumulation** (Algorithm 4 in the paper draft). Online fusion replaces $\mathbf{O_0}_K$ with the partial $\mathbf{O_0}_k$ available at each iteration, introducing a **scale coefficient** $s_k$ to correct the running accumulator:

```
Input: tiles V_0[1..K], V_1[1..K]
Initialize O_0 depending on the X operator
Initialize ~O_1 = 0

for k = 1 to K:                          # single fused loop
  O_0_k = f_X(O_0_{k-1}, V_0_k)         # X step
  s_k = g_B(O_0_k) / g_B(O_0_{k-1})     # scale coefficient
  B_k = g_B(O_0_k) * h_B(V_1_k)         # bias with partial X output
  ~O_1_k = s_k * ~O_1_{k-1} + B_k       # rescale + accumulate

Output: ~O_1_K = O_1_K
```

The scale coefficient $s_k = g_B(\mathbf{O_0}_k) / g_B(\mathbf{O_0}_{k-1})$ corrects the running accumulator each time the X output evolves. The final $\tilde{\mathbf{O_1}}_K = \mathbf{O_1}_K$ — intermediate values differ but the final output is exact.

This transformation requires three properties of the bias and accumulation:
1. **Separability**: $\mathbf{B}_k = g_B(\mathbf{O_0}) \cdot h_B(\mathbf{V_1}_k)$ — the bias decomposes multiplicatively.
2. **Associativity**: the sum can be split and recombined in any order.
3. **Linearity**: a scalar factor can be extracted: $\sum(c \cdot x_k) = c \cdot \sum(x_k)$.

When multiple online fusions share the same X reduction, they share the same scale coefficient $s_k$ — the X step runs once per iteration and all accumulators rescale together.

We classify each blocking pair:

| Producer → Consumer | Blocking dim | Handled by | Why |
|---|---|---|---|
| Op 2 (matmul over d1) → Op 5 (reduce max over d2) | d1 | **Not fusable** | Different blocking dims (d1 vs d2); no X+Accumulation pattern |
| Op 5 (reduce max over d2) → Op 6 (exp + sum over d2) | d2 | **Online fusion** | X+Accumulation pattern (§4.1.2) |
| Op 6 (exp_S depends on max) → Op 9 (matmul over d2) | d2 | **Online fusion** | Same X (running max), same $s_k$ (§4.1.3) |

Online fusion breaks the Op 5→6 and Op 6→9 barriers, pulling Ops 5, 6, 8, 9 into a single d2 loop. Op 7 is topologically independent from Op 9 (§1 DAG). Op 10 depends on Op 9's output but runs after the d2 reduction completes. Neither is part of the online fusion transforms. The result is the flash attention algorithm.

#### 4.1.2 Online Fusion: Op 5→6

**Before.** Ops 5 and 6 run in two separate d2 loops. Op 5 reduces max over the full d2 range to produce `neg_max_S`. Op 6 uses `neg_max_S` as a bias — it cannot start until Op 5 completes:

```python
""" Op 5: tensor_reduce max over d2 -> neg_max_S """
psum_partial_max = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.tensor_reduce(dst=psum_partial_max[0:128, i_block_d2:i_block_d2+1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)
nisa.tensor_reduce(dst=sbuf_neg_max_S[0:128, i_block_d0:i_block_d0+1],
    data=psum_partial_max[0:128, 0:8], op=nl.maximum, axis=1, negate=True)

""" Op 6: activation_reduce exp+sum over d2 -> exp_S, sum_exp """
psum_partial_sum = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
sbuf_exp_S_reshd = sbuf_exp_S.reshape((128, 32, 4096))
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 2 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.activation_reduce(dst=sbuf_exp_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.exp, bias=sbuf_neg_max_S[0:128, i_block_d0:i_block_d0+1],
            reduce_op=nl.add, reduce_res=psum_partial_sum[0:128, i_block_d2:i_block_d2+1])
nisa.tensor_reduce(dst=sbuf_sum_exp[0:128, i_block_d0:i_block_d0+1], data=psum_partial_sum[0:128, 0:8], op=nl.add, axis=1)
```

**Pattern match.** Map each component of the Standard X + Accumulation pattern (§4.1.1) to attention ops:

| Algorithm 2 Component | Attention Mapping |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $\mathbf{V_0}_k$ (X input) | `sbuf_scaled_S_reshd[..., i_block_d2*512:(i_block_d2+1)*512]` — tile $k$ of scaled S |
| $f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ | $\max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ — running max (Op 5) |
| $\mathbf{O_0}_K$ | `neg_max_S` — complete row-max of S (negated for use as bias) |
| $\mathbf{V_1}_k$ (Accumulation input) | Same `sbuf_scaled_S` tile — Op 6 reads the same data as Op 5 |
| $g_B(\mathbf{O_0}_K)$ | $e^{-m}$ where $m = \mathbf{O_0}_K$ — the X-dependent part of the bias |
| $h_B(\mathbf{V_1}_k)$ | $\text{rowsum}(e^{\mathbf{V_1}_k})$ — the input-dependent part |
| $\mathbf{B}_k = g_B \cdot h_B$ | $\text{rowsum}(e^{\mathbf{V_1}_k - m})$ — what `activation_reduce` computes per tile |
| $\mathbf{O_1}_k = \mathbf{O_1}_{k-1} + \mathbf{B}_k$ | `psum_sum_exp` accumulation (Op 6) |

**Verify separability.** The bias function $f^B(m, \mathbf{V}_k) = \text{rowsum}(e^{\mathbf{V}_k - m})$ decomposes as:

$$f^B(m, \mathbf{V}_k) = e^{-m} \cdot \text{rowsum}(e^{\mathbf{V}_k}) = g_B(m) \cdot h_B(\mathbf{V}_k)$$

This is multiplicatively separable. The accumulation (sum) is associative and linear with respect to scalar factors.

**Derive scale coefficient.** From Algorithm 4, $s_k = g_B(\mathbf{O_0}_k) / g_B(\mathbf{O_0}_{k-1})$:

$$s_k = \frac{e^{-m_k}}{e^{-m_{k-1}}} = e^{m_{k-1} - m_k}$$

where $m_k$ is the running max after tile $k$. This is the correction factor: when the running max grows from $m_{k-1}$ to $m_k$, all previously accumulated terms must be scaled down by $e^{m_{k-1} - m_k}$.

**Mechanical derivation.** Apply Algorithm 4 step-by-step in each iteration of the fused loop:

1. **X step**: $\mathbf{O_0}_k = \max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ — per-tile max → update running max
2. **Scale**: $s_k = e^{m_{k-1} - m_k}$ — compute from old and new running max
3. **Rescale**: $\tilde{\mathbf{O_1}}_k \mathrel{*}= s_k$ — multiply running sum by scale
4. **Bias + accumulate**: $\mathbf{B}_k = \text{rowsum}(e^{\mathbf{V_1}_k - m_k})$, then $\tilde{\mathbf{O_1}}_k \mathrel{+}= \mathbf{B}_k$

**After.** d2 loops 1+2 → one fused loop:

```python
""" Ops 5+6 fused: running max + exp + sum in one d2 loop """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0:1], value=0.0)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
sbuf_exp_S_reshd = sbuf_exp_S.reshape((128, 32, 4096))

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1+2 """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max """
        psum_tile_max = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0:1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)

        psum_max_pair = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0:1], src=sbuf_running_max[0:128, 0:1])
        nisa.tensor_copy(dst=psum_max_pair[0:128, 1:2], src=psum_tile_max[0:128, 0:1])
        sbuf_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sbuf_new_max[0:128, 0:1],
            data=psum_max_pair[0:128, 0:2], op=nl.maximum, axis=1)

        """ Scale: s_k = exp(m_{k-1} - m_k), then rescale running sum """
        sbuf_max_diff = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_max_diff[0:128, 0:1],
            data=sbuf_running_max[0:128, 0:1],
            op0=nl.subtract, operand0=sbuf_new_max[0:128, 0:1])
        sbuf_max_scale = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0:1],
            data=sbuf_max_diff[0:128, 0:1], op=nl.exp)
        nisa.tensor_scalar(dst=psum_running_sum[0:128, 0:1],
            data=psum_running_sum[0:128, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1])
        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0:1],
            src=sbuf_new_max[0:128, 0:1])

        """ Bias + accumulate: exp(V - m_k) with rowsum into running sum """
        sbuf_neg_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_max[0:128, 0:1],
            data=sbuf_new_max[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.activation(dst=sbuf_exp_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.exp, data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            bias=sbuf_neg_max[0:128, 0:1],
            reduce_op=nl.add, reduce_res=psum_running_sum[0:128, 0:1],
            reduce_cmd=nisa.reduce_cmd.reduce)
```

Note: The naive kernel (§2.3) uses `nisa.activation_reduce` with per-block PSUM slots — each slot is written once, so the reset is harmless. Online fusion changes the math: instead of separate slots + cross-block combine, there's a running accumulator rescaled each iteration. The transform rewrites `activation_reduce` → `nisa.activation(..., reduce_cmd=reduce)` which adds to the existing accumulator without resetting.

#### 4.1.3 Online Fusion: Ops 6, 8, 9

Op 9 (matmul) accumulates `exp_S_t @ V` over d2. Each tile's `exp_S` depends on the running max — the same X output as §4.1.2. This is a second application of online fusion with the same X reduction and the same scale coefficient $s_k = e^{m_{k-1} - m_k}$.

First, prepare two d2 loops. Op 8 (transpose) is elementwise — fold it into Op 9's d2 loop body.

**Before.** Two d2 loops after elementwise prep:

```python
""" Fused Ops 5+6: d2 loop (from §4.1.2) """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0:1], value=0.0)
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A """
    for i_tile_d2 in nl.affine_range(1):
        """ ... X step + scale + bias + accumulate from §4.1.2 ... """

""" Fused Ops 8+9: d2 loop """
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        for i_block_d2 in nl.affine_range(8):                                  """ d2 loop B """
            for i_tile_d2 in nl.affine_range(1):
                for i_ig_d2 in nl.affine_range(4):                             """ chunk sub-loop: d2 interleave """
                    sbuf_exp_S_t = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                    nisa.nc_transpose(dst=sbuf_exp_S_t[0:128, 0:128],
                        src=sbuf_exp_S[0:128, i_block_d0:i_block_d0+1, i_block_d2*4+i_ig_d2:i_block_d2*4+i_ig_d2+1, 0:128])
                    sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],
                        src=V[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128,
                             i_block_d5*128:i_block_d5*128+128])
                    nisa.nc_matmul(dst=psum_attn[0:128, 0:128],
                        stationary=sbuf_exp_S_t[0:128, 0:128],
                        moving=sbuf_V[0:128, 0:128])
```

**Pattern match.** Same Algorithm 2 structure as §4.1.2, with a different accumulation body:

| Algorithm 2 Component | Attention Mapping (Ops 6, 8, 9) |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ | Same as §4.1.2 — $\max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ |
| $g_B(\mathbf{O_0}_K)$ | Same $e^{-m}$ — `exp_S` uses the same max |
| $h_B(\mathbf{V_1}_k)$ | $\text{transpose}(\exp(\mathbf{V_1}_k))^T \cdot \mathbf{V}_k$ — matmul body per tile |
| $\mathbf{B}_k = g_B \cdot h_B$ | $\exp(\mathbf{S}_k - m_K)^T @ \mathbf{V}_k$ — one tile's matmul contribution |
| $s_k$ | Same $e^{m_{k-1} - m_k}$ — shared with the Op 6 accumulator |
| $\tilde{\mathbf{O_1}}_k$ | `psum_attn` — rescaled by $s_k$ then accumulated with $\mathbf{B}_k$ |

Since both online fusions share the same X (running max) and same $g_B(m) = e^{-m}$, they share the same $s_k$. Both `psum_running_sum` and `psum_attn` get rescaled by the same `sbuf_max_scale` each iteration — the X step runs once, then all accumulators rescale together.

**After.** Online fusion merges loops A+B. Both accumulators rescale by the same $s_k$:

```python
""" Ops 5+6+8+9 fused: one d2 loop """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0:1], value=0.0)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=psum_attn[0:128, 0:128], value=0.0)

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A+B """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max (same as §4.1.2) """
        psum_tile_max = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0:1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)
        psum_max_pair = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0:1], src=sbuf_running_max[0:128, 0:1])
        nisa.tensor_copy(dst=psum_max_pair[0:128, 1:2], src=psum_tile_max[0:128, 0:1])
        sbuf_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sbuf_new_max[0:128, 0:1],
            data=psum_max_pair[0:128, 0:2], op=nl.maximum, axis=1)

        """ Scale: s_k = exp(m_{k-1} - m_k) """
        sbuf_max_diff = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_max_diff[0:128, 0:1],
            data=sbuf_running_max[0:128, 0:1],
            op0=nl.subtract, operand0=sbuf_new_max[0:128, 0:1])
        sbuf_max_scale = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0:1],
            data=sbuf_max_diff[0:128, 0:1], op=nl.exp)

        """ Rescale BOTH accumulators by s_k """
        nisa.tensor_scalar(dst=psum_running_sum[0:128, 0:1],
            data=psum_running_sum[0:128, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1])
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                nisa.tensor_scalar(dst=psum_attn[0:128, 0:128],
                    data=psum_attn[0:128, 0:128],
                    op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1])

        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0:1],
            src=sbuf_new_max[0:128, 0:1])

        """ Accumulator 1: exp + sum (Op 6 body) """
        sbuf_neg_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_max[0:128, 0:1],
            data=sbuf_new_max[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        sbuf_exp_S = nl.ndarray((128, 512), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_exp_S[0:128, 0:512],
            op=nl.exp, data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            bias=sbuf_neg_max[0:128, 0:1],
            reduce_op=nl.add, reduce_res=psum_running_sum[0:128, 0:1],
            reduce_cmd=nisa.reduce_cmd.reduce)

        """ Accumulator 2: transpose + matmul (Ops 8+9 body) """
        for i_ig_d2 in nl.affine_range(4):                                    """ chunk sub-loop: d2 interleave """
            sbuf_exp_S_t = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
            nisa.nc_transpose(dst=sbuf_exp_S_t[0:128, 0:128],
                src=sbuf_exp_S[0:128, i_ig_d2*128:(i_ig_d2+1)*128])
            for i_block_d5 in nl.affine_range(1):
                for i_tile_d5 in nl.affine_range(1):
                    sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],
                        src=V[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128,
                             i_block_d5*128:i_block_d5*128+128])
                    nisa.nc_matmul(dst=psum_attn[0:128, 0:128],
                        stationary=sbuf_exp_S_t[0:128, 0:128],
                        moving=sbuf_V[0:128, 0:128])
```

`sbuf_exp_S` shrinks to per-tile since it is produced and consumed within the same d2 iteration. Op 7 is topologically independent from Op 9 (§1 DAG). Op 10 depends on Op 9's output but runs after the d2 reduction completes. Neither is part of this transform.

## 5. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel ([`attention_cte.py`](/home/ubuntu/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py)) already uses `nl.ds` for this purpose.

- **Data layout transforms.** The current search space treats data layout as fixed — each tensor's partition and free axes are determined by the math function and never change. A data layout transform pass would manipulate `nc_transpose` ops to improve memory access patterns and engine utilization. The key moves are: (1) **insert dummy transpose pairs** — add a transpose before and after any tensor access point (the pair is a no-op); (2) **cancel adjacent transposes** — two consecutive transposes on the same tensor annihilate; (3) **move transposes** — slide a transpose earlier or later in the graph, past compatible ops, to find a more profitable placement; and (4) **merge transpose with DMA** — when a transpose is adjacent to an HBM load/store, replace the `nc_transpose` + `nisa.dma_copy` pair with a single transposing DMA (`nisa.dma_copy` with transposed source/destination layout), eliminating the Tensor Engine transpose entirely.
