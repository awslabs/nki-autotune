---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-25"
geometry: margin=1in
---

# NKIGym Guide

**Running example**: causal single-head attention `softmax(mask(scale * Q @ K^T)) @ V`. Inputs use standard ML layout `(seq, hidden)`:

- `q: float16[512, 128]` — `(seq_q, d_k)`
- `k: float16[512, 128]` — `(seq_k, d_k)`
- `v: float16[512, 1024]` — `(seq_k, d_v)`
- output: `float16[512, 1024]` — `(seq_q, d_v)`

---

## 1. Math Function
The top-level IR math function only defines the math computations. There are no load/store operators or hardware tiling, and it ignores the memory hierarchy.
```python
def attention(Q, K, V, scale):
    Q_t = nkigym.nc_transpose(Q)
    K_t = nkigym.nc_transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    scaled_S = nkigym.tensor_scalar(S, op0="multiply", operand0=scale)
    masked_S = nkigym.affine_select(scaled_S, cmp_op="greater_equal",
                                     on_false_value=-np.inf,
                                     channel_multiplier=1, step=-1)
    max_S = nkigym.tensor_reduce(masked_S, op="max")
    shifted_S = nkigym.tensor_scalar(masked_S, max_S, op0="subtract")
    exp_S = nkigym.activation(shifted_S, op="exp")
    sum_exp = nkigym.tensor_reduce(exp_S, op="add")
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    exp_S_t = nkigym.nc_transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")
    return output
```

## 2. NKI Gym Operators

Every `nkigym.*` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented.

```python
class Tensor:
    """
    Represents a N-dimensional NKI tensor in the shape of:
    [TILE_SIZES[AXES[0]],
        NUM_BLOCKS[AXES[0]], ..., NUM_BLOCKS[AXES[N-1]],
        NUM_TILES[AXES[0]], ..., NUM_TILES[AXES[N-1]],
        TILE_SIZES[AXES[1]], ..., TILE_SIZES[AXES[N-1]]
    ]
    """
    NAME: str
    AXES: tuple[str, ...]
    TILE_SIZES: dict[str, int]
    NUM_BLOCKS: dict[str, int]
    NUM_TILES: dict[str, int]
    LOCATION: str # SBUF, PSUM

class NKIOp:
    """
    Represents a NKI operator semantics
    Suuports multi outputs
    """
    NAME: str
    OPERAND_AXES: dict[str, tuple[str, ...]]
    OUTPUT_AXES: dict[str, tuple[str, str, ...]]
    MAX_TILE_SIZES: dict[str, int]
    ENGINE: str

    @abstract
    def __call__(self, **kwargs):
        """
        CPU simulation using Numpy in default float64.
        Takes input arrays + config, returns output array(s).
        """

    @abstract
    def render(self, ctx: "RenderContext"):
        """
        Render into NKI kernel source codes.
        If any operand or output tile dimension in ctx exceeds
        this op's MAX_TILE_SIZES, emit an in-place sub-loop that
        iterates in chunks of MAX_TILE_SIZES to cover the full tile.
        """

class RenderContext:
    """
    Everything an NKIOp.render() needs to emit NKI source.
    Built by the renderer for each op call site.
    """
    outputs: dict[str, Tensor]
    operands: dict[str, Tensor]
    config_kwargs: dict[str, Any]
    tile_idx: dict[str, str]  # dim_id → loop var expr (e.g. "i_0")
    tile_start: dict[str, str]  # dim_id → element offset expr (e.g. "i_0 * 128")
```

### 2.1 NKI Operator Subclasses
```python
class NKIMatmul(NKIOp):
    """stationary.T @ moving (Tensor Engine)

    stationary(K, M).T @ moving(K, N) → output(M, N).
    Accumulates into PSUM in fp32 regardless of input dtype.
    PSUM allocation is the renderer's responsibility (before the
    reduction loop); this op only emits the nc_matmul call.

    Sub-loop: if K exceeds K:128, emits a sub-loop that splits the
    contraction axis into 128-element chunks, each accumulating into
    the same PSUM output via hardware accumulation.
    """

    NAME = "nc_matmul"
    OPERAND_AXES = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES = {"output": ("M", "N")}
    MAX_TILE_SIZES = {"K": 128, "M": 128, "N": 512}
    ENGINE = "TensorEngine"

    def __call__(self, stationary, moving):
        return stationary.T @ moving

    def render(self, ctx):
        dst = ctx.outputs["output"]
        stat = ctx.operands["stationary"]
        mov = ctx.operands["moving"]
        k_size = stat.TILE_SIZES[stat.AXES[0]]
        k_max = self.MAX_TILE_SIZES["K"]
        if k_size > k_max:
            lines = [f"for i_k_sub in nl.affine_range({k_size // k_max}):"]
            stat_slice = f"{stat.NAME}[i_k_sub*{k_max}:(i_k_sub+1)*{k_max}, ...]"
            mov_slice = f"{mov.NAME}[i_k_sub*{k_max}:(i_k_sub+1)*{k_max}, ...]"
            lines.append(
                f"  nisa.nc_matmul({dst.NAME}, {stat_slice}, {mov_slice})"
            )
            return "\n".join(lines)
        return f"nisa.nc_matmul({dst.NAME}, {stat.NAME}, {mov.NAME})"


class NKITranspose(NKIOp):
    """Swap partition and free dims (Tensor Engine or Vector Engine)

    data(P, F) → output(F, P).
    Tensor Engine path: SBUF→PSUM, both dims ≤ 128.
    Vector Engine path: SBUF/PSUM→SBUF/PSUM, both dims ≤ 32.
    Real hardware op, not a free view.

    Sub-loop: if P or F exceeds 128, emits a sub-loop that
    transposes (128, 128) chunks, writing each to the corresponding
    slice of the output.
    """

    NAME = "nc_transpose"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("F", "P")}
    MAX_TILE_SIZES = {"P": 128, "F": 128}
    ENGINE = "TensorEngine"

    def __call__(self, data):
        return data.T

    def render(self, ctx):
        dst = ctx.outputs["output"]
        src = ctx.operands["data"]
        p_size = src.TILE_SIZES[src.AXES[0]]
        f_size = src.TILE_SIZES[src.AXES[1]]
        p_max = self.MAX_TILE_SIZES["P"]
        f_max = self.MAX_TILE_SIZES["F"]
        if p_size > p_max or f_size > f_max:
            p_subs = p_size // p_max
            f_subs = f_size // f_max
            lines = []
            if p_subs > 1:
                lines.append(f"for i_p_sub in nl.affine_range({p_subs}):")
            if f_subs > 1:
                lines.append(f"  for i_f_sub in nl.affine_range({f_subs}):")
            indent = "  " * (int(p_subs > 1) + int(f_subs > 1))
            src_p = f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}" if p_subs > 1 else f"0:{p_size}"
            src_f = f"i_f_sub*{f_max}:(i_f_sub+1)*{f_max}" if f_subs > 1 else f"0:{f_size}"
            dst_p = f"i_f_sub*{f_max}:(i_f_sub+1)*{f_max}" if f_subs > 1 else f"0:{f_size}"
            dst_f = f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}" if p_subs > 1 else f"0:{p_size}"
            lines.append(
                f"{indent}nisa.nc_transpose("
                f"{dst.NAME}[{dst_p}, {dst_f}], "
                f"{src.NAME}[{src_p}, {src_f}])"
            )
            return "\n".join(lines)
        return f"nisa.nc_transpose({dst.NAME}, {src.NAME})"


class NKITensorScalar(NKIOp):
    """Element-wise binary op with broadcast (Vector/Scalar/GpSimd)

    data(P, F) <op0> operand0 → output(P, F).
    operand0 can be a scalar constant or a (P,) column vector;
    it broadcasts across the free axis.
    ISA supports compound (data <op0> operand0) <op1> operand1;
    nkigym decomposes compound ops into separate calls.
    """

    NAME = "tensor_scalar"
    OPERAND_AXES = {"data": ("P", "F"), "operand0": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F")}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "VectorEngine"

    def __call__(self, data, operand0, op0):
        ops = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}
        if isinstance(operand0, np.ndarray):
            operand0 = operand0[..., np.newaxis]
        return ops[op0](data, operand0)

    def render(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op0"]
        op0 = ctx.operands.get("operand0")
        if op0 is not None:
            op0_expr = op0.NAME
        else:
            op0_expr = ctx.config_kwargs["operand0"]
        return (
            f"nisa.tensor_scalar(dst={dst.NAME}, data={data.NAME}, "
            f"operand0={op0_expr}, op0=nl.{op_name})"
        )


class NKIAffineSelect(NKIOp):
    """Position-predicated element select (GpSimd Engine)

    Generates affine_value = offset + p*channel_multiplier + f*step
    per element, compares to 0. Selects data or on_false_value.
    Renderer computes offset from tile_start at render time.
    """

    NAME = "affine_select"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P", "F")}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "GpSimd"

    def __call__(self, data, cmp_op, on_false_value,
                 channel_multiplier, step):
        P, F = data.shape
        p_idx = np.arange(P)[:, np.newaxis]
        f_idx = np.arange(F)[np.newaxis, :]
        cmps = {"greater_equal": np.greater_equal}
        mask = cmps[cmp_op](
            p_idx * channel_multiplier + f_idx * step, 0
        )
        return np.where(mask, data, on_false_value)

    def render(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        kw = ctx.config_kwargs
        p_dim, f_dim = data.AXES[0], data.AXES[1]
        offset = (
            f"{ctx.tile_start[p_dim]} * {kw['channel_multiplier']}"
            f" + {ctx.tile_start[f_dim]} * {kw['step']}"
        )
        return (
            f"nisa.affine_select(dst={dst.NAME}, "
            f"pattern=[[{kw['step']}, {data.TILE_SIZES[f_dim]}]], "
            f"offset={offset}, "
            f"channel_multiplier={kw['channel_multiplier']}, "
            f"cmp_op=nl.{kw['cmp_op']}, "
            f"on_true_tile={data.NAME}, "
            f"on_false_value={kw['on_false_value']})"
        )


class NKITensorReduce(NKIOp):
    """Reduce free axis (Vector Engine)

    data(P, F) → output(P,).
    Collapses free dimension with the specified reduction op.
    Both input and output can be SBUF or PSUM.
    """

    NAME = "tensor_reduce"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P",)}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "VectorEngine"

    _NKI_REDUCE_OPS = {"max": "maximum", "add": "add"}

    def __call__(self, data, op):
        reduce = {"max": np.max, "add": np.sum}
        return reduce[op](data, axis=1)

    def render(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["op"]]
        return f"nisa.tensor_reduce(dst={dst.NAME}, data={data.NAME}, op=nl.{op_name}, axis=1)"


class NKIActivation(NKIOp):
    """Element-wise unary activation (Scalar Engine)

    output = op(data). Works on both 2D (P, F) and 1D (P,) tiles.
    ISA computes op(data * scale + bias) with optional reduction;
    nkigym decomposes scale/bias into separate tensor_scalar ops,
    so this op models only the unary activation.
    Input/output can be SBUF or PSUM.
    """

    NAME = "activation"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P", "F")}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "ScalarEngine"

    def __call__(self, data, op):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
            "rsqrt": lambda x: 1.0 / np.sqrt(x),
        }
        return fns[op](data)

    def render(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
        return f"nisa.activation(dst={dst.NAME}, data={data.NAME}, op=nl.{op_name})"
```

## 3. Simulation

The math function is plain Python — each `nkigym.*` call dispatches to `NKIOp.__call__()` which executes the op with numpy at float64 precision. No parsing or IR needed; just call the function directly:

```python
q = np.random.randn(512, 128)
k = np.random.randn(512, 128)
v = np.random.randn(512, 1024)
output = attention(q, k, v, scale=1.0 / np.sqrt(128))
```

The result is the **reference output** — a float64 array that any correctly rendered and compiled NKI kernel must match (within hardware precision tolerance).

---

## 4. Parse

AST parsing extracts the computation graph from the math function source. `ast.parse(inspect.getsource(fn))` walks the function body and produces an ordered list of op calls:

```python
class OpCall(NamedTuple):
    op_type: type[NKIOp]
    input_vars: tuple[str, ...]
    output_var: str
    config_kwargs: dict[str, Any]
```

For the running example, parsing `attention` produces:

| # | op_type | input_vars | output_var | config_kwargs |
|---|---|---|---|---|
| 0 | NKITranspose | (Q,) | Q_t | {} |
| 1 | NKITranspose | (K,) | K_t | {} |
| 2 | NKIMatmul | (Q_t, K_t) | S | {} |
| 3 | NKITensorScalar | (S,) | scaled_S | {op0: "multiply", operand0: scale} |
| 4 | NKIAffineSelect | (scaled_S,) | masked_S | {cmp_op: "greater_equal", ...} |
| 5 | NKITensorReduce | (masked_S,) | max_S | {op: "max"} |
| 6 | NKITensorScalar | (masked_S, max_S) | shifted_S | {op0: "subtract"} |
| 7 | NKIActivation | (shifted_S,) | exp_S | {op: "exp"} |
| 8 | NKITensorReduce | (exp_S,) | sum_exp | {op: "add"} |
| 9 | NKIActivation | (sum_exp,) | inv_sum | {op: "reciprocal"} |
| 10 | NKITranspose | (exp_S,) | exp_S_t | {} |
| 11 | NKIMatmul | (exp_S_t, V) | attn | {} |
| 12 | NKITensorScalar | (attn, inv_sum) | output | {op0: "multiply"} |

The parser also identifies the function's input parameters (`Q, K, V, scale`) and the return variable (`output`). Intermediate tensor names (`Q_t`, `S`, `masked_S`, etc.) are extracted from the AST assignment targets, preserving the user's naming from the math function. This `list[OpCall]` is the shared input for dimension analysis (§5.1.1) and render (§6) — the render uses these names in the generated kernel for readability.

---

## 5. Kernel Variants

Starting from the math function, we mechanically generate an **initial eager mode variant** (§5.1) — correct by construction but with terrible performance. We then identify **kernel constraints** (§5.2) — blocking relationships and topological order — that restrict how the variant can be transformed. Finally, we define **variant transforms** (§5.3) — fusion, load hoisting, loop reordering, tiles-per-block — that must respect these constraints. Any combination that does so produces a valid kernel variant; the full cross-product is the **search space** (§5.4).

### 5.1 Initial Eager Mode Variant

#### 5.1.1 Mechanical Lowering from Math Function

The lowering from math function to loop nest is fully mechanical. It proceeds in three steps: (1) derive dimensions by walking the op call list, (2) determine tile sizes from hardware limits, and (3) emit one independent loop nest per operator.

**Step 1 — Dimension analysis.** Each input parameter gets fresh dimension IDs. Walking the op calls in order, we match each input variable's dimensions to the op's `OPERAND_AXES` labels. When two operands share an axis label, their corresponding dimension IDs are **unified** (merged into one).

| # | Math function call | Dimension assignment |
|---|---|---|
| 0 | `Q_t = nkigym.nc_transpose(Q)` | Q(d0,d1) → Q_t(d1,d0) |
| 1 | `K_t = nkigym.nc_transpose(K)` | K(d2,d3) → K_t(d3,d2) |
| 2 | `S = nkigym.nc_matmul(Q_t, K_t)` | shared K → **unify** d3→d1; S(d0,d2) |
| 3–9 | scalar/reduce/activation ops | propagate d0, d2 |
| 10 | `exp_S_t = nkigym.nc_transpose(exp_S)` | exp_S_t(d2,d0) |
| 11 | `attn = nkigym.nc_matmul(exp_S_t, V)` | shared K → **unify** d4→d2; attn(d0,d5) |
| 12 | `output = nkigym.tensor_scalar(attn, inv_sum)` | output(d0,d5) |

After the walk, 4 unique dimensions remain: **d0** (seq_q), **d1** (d_k), **d2** (seq_k), **d5** (d_v).

**Step 2 — Tile sizes.** Each dimension's tile size is `min(max(all MAX_TILE_SIZES for this dim), input_size)`. Ops with limits smaller than the tile size emit in-place sub-loops.

| Dim | Limits collected from ops | tile_size |
|---|---|---|
| d0 | P:128, M:128, F:128 | 128 |
| d1 | F:128, K:128 | 128 |
| d2 | P:128, N:512, F:128, K:128 | 512 |
| d5 | N:512 | 512 |

**Step 3 — Emit loop nests.** Each op gets its own loop nest, derived from its tensor dimensions:
- **Parallel loops**: one double loop (block + tile) per dimension in the op's output tensor.
- **Reduction loop**: if the op consumes a dimension (present in operand axes but absent from output axes), nest an inner double loop. Initialize the accumulator before the reduction loop.

Each dimension produces a **double loop**:

```
for block_iter in range(total_tiles // tiles_per_block):   # block loop
    for tile_iter in range(tiles_per_block):                # tile loop
        [ops execute here, one tile at a time]
```

Hardware computes on a single tile at a time — this is fixed. The double loop structure creates a slot between the block and tile loops where DMA loads can be placed (§5.1.2). The renderer **always emits both loops** regardless of `tiles_per_block` value.

| # | Op | Output dims | Consumed dim | Init |
|---|---|---|---|---|
| 0 | nc_transpose(Q) → Q_t | d0, d1 | — | — |
| 1 | nc_transpose(K) → K_t | d1, d2 | — | — |
| 2 | nc_matmul(Q_t, K_t) → S | d0, d2 | d1 | 0 |
| 3 | tensor_scalar(S, scale) → scaled_S | d0, d2 | — | — |
| 4 | affine_select(scaled_S) → masked_S | d0, d2 | — | — |
| 5 | tensor_reduce(masked_S, max) → max_S | d0 | d2 | $-\infty$ |
| 6 | tensor_scalar(masked_S, max_S) → shifted_S | d0, d2 | — | — |
| 7 | activation(shifted_S, exp) → exp_S | d0, d2 | — | — |
| 8 | tensor_reduce(exp_S, add) → sum_exp | d0 | d2 | 0 |
| 9 | activation(sum_exp, reciprocal) → inv_sum | d0 | — | — |
| 10 | nc_transpose(exp_S) → exp_S_t | d0, d2 | — | — |
| 11 | nc_matmul(exp_S_t, V) → attn | d0, d5 | d2 | 0 |
| 12 | tensor_scalar(attn, inv_sum) → output | d0, d5 | — | — |

The dimension analysis also produces an `Analysis` — the complete tiling and data-flow metadata consumed by the rest of the kernel variant pipeline and the renderer (§6):

```python
class Analysis(NamedTuple):
    dims: dict[str, Dim]
    dependency_groups: tuple[tuple[str, ...], ...]
    var_axes: dict[str, tuple[str, ...]]
    return_var: str

class Dim(NamedTuple):
    dim_id: str
    tile_size: int
    input_size: int
```

#### 5.1.2 NKI Kernel with Greedy DMA Placement

The eager loop nest (§5.1.1) operates on abstract tensors. To produce a concrete NKI kernel, we assign each tensor to a memory space and insert explicit DMA where needed. Only two kinds of `nisa.dma_copy` appear:

- **Input loads** (blue): `Q`, `K`, `V` are loaded from HBM to SBUF before the op that first uses them. Each load is placed at the **innermost loop level** — the greediest (most frequent) placement.
- **Output store** (orange): the final `output` tensor is stored from SBUF to HBM after the last op.

Each intermediate tensor (`Q_t`, `S`, `scaled_S`, …) is allocated as a **full-range** `nl.ndarray` buffer in SBUF right before the op that produces it. Their shapes span the complete tensor dimensions so they persist across operators — each op writes tile-indexed slices into its output buffer; subsequent ops read from the same buffer at the appropriate tile positions. This makes the data flow between separate loop nests explicit and mathematically correct, though the total SBUF allocation far exceeds hardware capacity (the running example needs ~260 MB of SBUF for intermediates alone, vs 24 MB available). Variant transforms (§5.3) shrink these buffers by fusing loop nests (eliminating intermediates entirely), hoisting loads, and reordering dimensions.

```python
@nki.jit
def attention_kernel(Q, K, V):
    """ Q: (4096, 128)  K: (4096, 128)  V: (4096, 128) """
    """ d0: 128x32  d1: 128x1  d2: 512x8  d5: 128x1  (tile_size x num_blocks) """
    output = nl.ndarray((4096, 128), dtype=Q.dtype, buffer=nl.shared_hbm)

    """ Op 0: nisa.nc_transpose -- Q(d0, d1) -> Q_t(d1, d0) """
    sbuf_Q_t = nl.ndarray((128, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d1 in nl.affine_range(1):
                for t_d1 in nl.affine_range(1):
                    sbuf_Q = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)              # DMA load Q
                    nisa.dma_copy(dst=sbuf_Q[0:128, 0:128], src=Q[i_d0*128:i_d0*128+128, i_d1*128:i_d1*128+128])
                    nisa.nc_transpose(dst=sbuf_Q_t[i_d1*128:i_d1*128+128, i_d0*128:i_d0*128+128], src=sbuf_Q[0:128, 0:128])

    """ Op 1: nisa.nc_transpose -- K(d2, d1) -> K_t(d1, d2) """
    sbuf_K_t = nl.ndarray((128, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d1 in nl.affine_range(1):
        for t_d1 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    sbuf_K = nl.ndarray((512, 128), dtype=K.dtype, buffer=nl.sbuf)              # DMA load K
                    nisa.dma_copy(dst=sbuf_K[0:512, 0:128], src=K[i_d2*512:i_d2*512+512, i_d1*128:i_d1*128+128])
                    nisa.nc_transpose(dst=sbuf_K_t[i_d1*128:i_d1*128+128, i_d2*512:i_d2*512+512], src=sbuf_K[0:512, 0:128])

    """ Op 2: nisa.nc_matmul -- Q_t(d1, d0) x K_t(d1, d2) -> S(d0, d2), accumulate over d1 """
    sbuf_S = nl.ndarray((4096, 4096), dtype=nl.float32, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    for i_d1 in nl.affine_range(1):
                        for t_d1 in nl.affine_range(1):
                            nisa.nc_matmul(dst=psum_S[0:128, 0:512], stationary=sbuf_Q_t[i_d1*128:i_d1*128+128, i_d0*128:i_d0*128+128], moving=sbuf_K_t[i_d1*128:i_d1*128+128, i_d2*512:i_d2*512+512])
                    nisa.tensor_copy(dst=sbuf_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], src=psum_S[0:128, 0:512])

    """ Op 3: nisa.tensor_scalar -- S(d0, d2) x scale -> scaled_S(d0, d2) """
    sbuf_scaled_S = nl.ndarray((4096, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.tensor_scalar(dst=sbuf_scaled_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], data=sbuf_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], op0=nl.multiply, operand0=scale)

    """ Op 4: nisa.affine_select -- scaled_S(d0, d2) -> masked_S(d0, d2) """
    sbuf_masked_S = nl.ndarray((4096, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.affine_select(dst=sbuf_masked_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], pred=sbuf_scaled_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512])

    """ Op 5: nisa.tensor_reduce -- masked_S(d0, d2) -> max_S(d0), reduce max over d2 """
    sbuf_max_S = nl.ndarray((4096, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            psum_max_S = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(dst=psum_max_S[0:128, 0:1], value=nl.neg_inf)
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.tensor_reduce(dst=psum_max_S[0:128, 0:1], data=sbuf_masked_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], op=nl.max, axis=1)
            nisa.tensor_copy(dst=sbuf_max_S[i_d0*128:i_d0*128+128, 0:1], src=psum_max_S[0:128, 0:1])

    """ Op 6: nisa.tensor_scalar -- masked_S(d0, d2) - max_S(d0) -> shifted_S(d0, d2) """
    sbuf_shifted_S = nl.ndarray((4096, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.tensor_scalar(dst=sbuf_shifted_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], data=sbuf_masked_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], op0=nl.subtract, operand0=sbuf_max_S[i_d0*128:i_d0*128+128, 0:1])

    """ Op 7: nisa.activation -- exp(shifted_S(d0, d2)) -> exp_S(d0, d2) """
    sbuf_exp_S = nl.ndarray((4096, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.activation(dst=sbuf_exp_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], data=sbuf_shifted_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], op=nl.exp)

    """ Op 8: nisa.tensor_reduce -- exp_S(d0, d2) -> sum_exp(d0), reduce add over d2 """
    sbuf_sum_exp = nl.ndarray((4096, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            psum_sum_exp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.tensor_reduce(dst=psum_sum_exp[0:128, 0:1], data=sbuf_exp_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512], op=nl.add, axis=1)
            nisa.tensor_copy(dst=sbuf_sum_exp[i_d0*128:i_d0*128+128, 0:1], src=psum_sum_exp[0:128, 0:1])

    """ Op 9: nisa.activation -- reciprocal(sum_exp(d0)) -> inv_sum(d0) """
    sbuf_inv_sum = nl.ndarray((4096, 1), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            nisa.activation(dst=sbuf_inv_sum[i_d0*128:i_d0*128+128, 0:1], data=sbuf_sum_exp[i_d0*128:i_d0*128+128, 0:1], op=nl.reciprocal)

    """ Op 10: nisa.nc_transpose -- exp_S(d0, d2) -> exp_S_t(d2, d0) """
    sbuf_exp_S_t = nl.ndarray((4096, 4096), dtype=Q.dtype, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d2 in nl.affine_range(8):
                for t_d2 in nl.affine_range(1):
                    nisa.nc_transpose(dst=sbuf_exp_S_t[i_d2*512:i_d2*512+512, i_d0*128:i_d0*128+128], src=sbuf_exp_S[i_d0*128:i_d0*128+128, i_d2*512:i_d2*512+512])

    """ Op 11: nisa.nc_matmul -- exp_S_t(d2, d0) x V(d2, d5) -> attn(d0, d5), accumulate over d2 """
    sbuf_attn = nl.ndarray((4096, 128), dtype=nl.float32, buffer=nl.sbuf)
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d5 in nl.affine_range(1):
                for t_d5 in nl.affine_range(1):
                    psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                    for i_d2 in nl.affine_range(8):
                        for t_d2 in nl.affine_range(1):
                            sbuf_V = nl.ndarray((512, 128), dtype=V.dtype, buffer=nl.sbuf)      # DMA load V
                            nisa.dma_copy(dst=sbuf_V[0:512, 0:128], src=V[i_d2*512:i_d2*512+512, i_d5*128:i_d5*128+128])
                            nisa.nc_matmul(dst=psum_attn[0:128, 0:128], stationary=sbuf_exp_S_t[i_d2*512:i_d2*512+512, i_d0*128:i_d0*128+128], moving=sbuf_V[0:512, 0:128])
                    nisa.tensor_copy(dst=sbuf_attn[i_d0*128:i_d0*128+128, i_d5*128:i_d5*128+128], src=psum_attn[0:128, 0:128])

    """ Op 12: nisa.tensor_scalar -- attn(d0, d5) x inv_sum(d0) -> output(d0, d5) """
    for i_d0 in nl.affine_range(32):
        for t_d0 in nl.affine_range(1):
            for i_d5 in nl.affine_range(1):
                for t_d5 in nl.affine_range(1):
                    sbuf_output = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=sbuf_output[0:128, 0:128], data=sbuf_attn[i_d0*128:i_d0*128+128, i_d5*128:i_d5*128+128], op0=nl.multiply, operand0=sbuf_inv_sum[i_d0*128:i_d0*128+128, 0:1])
                    nisa.dma_copy(dst=output[i_d0*128:i_d0*128+128, i_d5*128:i_d5*128+128], src=sbuf_output[0:128, 0:128])  # DMA store output

    return output
```

### 5.2 Kernel Constraints

#### 5.2.1 Blocking Producer-Loop → Consumer

A **blocking relationship** exists when a consumer op requires the producer to complete its full reduction loop before the consumer can start. In the eager mode loop nest, these are the 4 reduction loops whose outputs are consumed by subsequent ops:

| Producer | Consumer | Blocking dim | Why |
|---|---|---|---|
| Op 2: `S[i_d0, i_d2] += nc_matmul(...)` over d1 | Op 3: `tensor_scalar(S[i_d0, i_d2], ...)` | d1 | S needs full d1 accumulation |
| Op 5: `max_S[i_d0] = max(...)` over d2 | Op 6: `tensor_scalar(masked_S[...], max_S[i_d0], ...)` | d2 | max_S needs full d2 reduction |
| Op 8: `sum_exp[i_d0] += tensor_reduce(...)` over d2 | Op 9: `activation(sum_exp[i_d0], ...)` | d2 | sum_exp needs full d2 reduction |
| Op 11: `attn[i_d0, i_d5] += nc_matmul(...)` over d2 | Op 12: `tensor_scalar(attn[...], ...)` | d2 | attn needs full d2 accumulation |

![Op 2 accumulates S over d1, blocking Op 3 which reads sbuf_S](diagrams/blocking_op2_op3.png)
*Op 2 must finish accumulating S over the full d1 reduction before Op 3 can read sbuf_S.*

![Op 5 reduces max_S over d2, blocking Op 6 which reads sbuf_max_S](diagrams/blocking_op5_op6.png)
*Op 5 must finish reducing max over the full d2 range before Op 6 can read sbuf_max_S.*

![Op 8 reduces sum_exp over d2, blocking Op 9 which reads sbuf_sum_exp](diagrams/blocking_op8_op9.png)
*Op 8 must finish reducing sum over the full d2 range before Op 9 can read sbuf_sum_exp.*

![Op 11 accumulates attn over d2, blocking Op 12 which reads sbuf_attn](diagrams/blocking_op11_op12.png)
*Op 11 must finish accumulating attn over the full d2 reduction before Op 12 can read sbuf_attn.*

#### 5.2.2 Topological Computation Order

Ops must execute in an order consistent with the data-flow graph: every op's inputs must be computed before the op runs. The math function (§1) defines one valid topological order, and the eager mode variant follows it exactly. Variant transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

### 5.3 Variant Transforms

Given the initial eager mode variant and its constraints (§5.2), a **kernel variant** is any legal rearrangement of the loop nest. Four transforms generate the search space.

#### 5.3.1 Loop Fusion

Adjacent ops whose loops iterate over the same dimensions can share a single loop nest instead of having separate nests. Fusion eliminates redundant loop overhead and — critically — keeps intermediate tiles on-chip between producer and consumer, avoiding HBM round-trips. Fusion is legal across any non-blocking boundary. It is forbidden across a blocking boundary for the blocking dimension: the reduction must complete before the consumer starts.

![Loop fusion example: Ops 3+4 (tensor_scalar + affine_select) share the same (d0, d2) loops. Before: separate loop nests with redundant store/load of scaled_S (gray). After: single loop nest, scaled_S stays on-chip.](diagrams/transform_fusion.png)

#### 5.3.2 Load Hoisting

A DMA load can be moved from its greedy placement (immediately before the op) to a higher or lower point in the loop nest. Hoisting a load above a loop means the loaded data persists across all iterations of that loop, amortizing DMA latency at the cost of a larger SBUF buffer. Sinking a load deeper reduces buffer size at the cost of more frequent reloads. The slot between block and tile loops (§5.1.1) provides the placement points at each nesting level.

In the greedy baseline, all loads start at the innermost position, so the initial direction is always hoisting outward. After loop fusion and reordering, loads may need to move in either direction to balance SBUF pressure against DMA frequency.

![Load hoisting example: Q_t in Op 2 (matmul). Before: Q_t loaded inside the d1 loop, reloaded every (d2, d1) iteration (gray). After: Q_t hoisted outside the d2 loop, loaded once per d0 tile (green). Trades larger SBUF buffer for fewer DMA operations.](diagrams/transform_load_hoist.png)

#### 5.3.3 Loop Reordering

The order of dimension loops can be permuted, subject to: (1) same-dimension phase ordering (block before tile), (2) cross-dimension data dependencies (if phase B consumes a tensor produced by phase A of a different dimension, A must precede B), and (3) in-place buffer reuse constraints (a parallel dim not in the buffer's dimensions cannot wrap phases that write the buffer).

![Loop reordering transform: Op 2 matmul loop order d0 > d2 > d1 reordered to d2 > d0 > d1](diagrams/transform_reorder.png)

#### 5.3.4 Tiles Per Block

`tiles_per_block` is a per-dimension parameter that controls the block size — how many tiles are grouped before the tile loop iterates. Larger `tiles_per_block` means larger SBUF buffers but fewer DMA round-trips: the loaded block is reused across `tiles_per_block` compute iterations. Must divide `total_tiles`.

![Tiles per block transform: Op 2 d2 dimension from 8 blocks x 1 tile to 2 blocks x 4 tiles](diagrams/transform_tpb.png)

### 5.4 Search Space

A `Schedule` is an immutable, hashable descriptor that captures *how* to execute the math function — the specific combination of variant transforms applied to the initial eager mode variant.

```python
class Loop(NamedTuple):
    dim_id: str            # global dimension ID (e.g. "d0")
    tile_size: int         # hardware tile size from §5.1.1
    tiles_per_block: int   # block size — §5.3.4

class Schedule(NamedTuple):
    loop_order: tuple[tuple[Loop, tuple[int, int]], ...]  # (Loop, (level, order))
    load_placements: tuple[tuple[str, int], ...]          # (tensor_name, relative_pos)
```

Each `loop_order` entry pairs a `Loop` with a `(level, order)` tuple. `level` is the nesting depth — different levels nest. `order` is sequential execution order within the same level — same level, different order means back-to-back loops.

The search space is the cross-product of three independent axes:

**Axis 1 — Loop order**: permutations of loop items (parallel dims + reduction dim phases), filtered by three ordering constraints: (1) same-dim phase ordering, (2) cross-dim data dependencies, and (3) buffer-reuse constraints. The valid count depends on the specific dependency structure and is computed by enumeration.

**Axis 2 — Tiles per block**: for each dimension, `tiles_per_block` can be any divisor of `total_tiles` (= dim_size / tile_size). This axis is a simple cross-product over divisor sets.

**Axis 3 — Load placements**: for each input tensor, `relative_pos` determines placement depth in the loop nest (0 = outermost, more = deeper). Deeper placement means more frequent reloads but smaller SBUF buffers.

The full space is the cross-product of all three axes. Hardware validation then prunes infeasible candidates — larger `tiles_per_block` grows SBUF buffers, which must stay within capacity (24 MB). Every survivor is rendered, compiled, and benchmarked.

**Running example** with typical Llama-style shapes: `q[4096, 128], k[4096, 128], v[4096, 128]`:

| Dim | Size | tile_size | total_tiles | Valid tiles_per_block (divisors) | Count |
|---|---|---|---|---|---|
| d0 (seq_q) | 4096 | 128 | 32 | {1, 2, 4, 8, 16, 32} | 6 |
| d1 (d_k) | 128 | 128 | 1 | {1} | 1 |
| d2 (seq_k) | 4096 | 512 | 8 | {1, 2, 4, 8} | 4 |
| d5 (d_v) | 128 | 128 | 1 | {1} | 1 |

- **Loop orders**: 18 valid permutations (after all three constraint filters)
- **Tiles per block**: 6 × 1 × 4 × 1 = 24 combinations
- **Load placements**: 3 × 3 × 3 = 27 combinations (Q: 3 slots, K: 3 slots, V: 3 slots)
- **Full cross-product**: 18 × 24 × 27 = 11,664 candidates (before hardware validation)

## 6. Render (Specialization)

Rendering is a **specialization** step: it takes the math function (§1), the op call list (§4), dimension analysis (§5.1.1), a schedule (§5.4), and **concrete input shapes and dtypes** to produce an NKI kernel tailored to those specific inputs. Each unique combination of input shapes and dtypes requires its own tuned kernel. Infrastructure ops (DMA loads, PSUM→SBUF staging, DMA stores) are generated by the renderer — they do not appear in the math function.

### 6.1 Render Pipeline

```
render(op_calls: list[OpCall], analysis: Analysis, schedule: Schedule, input_shapes, dtypes) → NKI kernel source
```

- **`op_calls`**: ordered list of op calls from the parser (§4)
- **`analysis`**: dimension analysis result — dims, var_axes, return_var (§5.1.1)
- **`schedule`**: loop order and load placements (§5.4)
- **`input_shapes`**: concrete sizes for each input parameter
- **`dtypes`**: data type for each input parameter

1. **Emit preamble**: imports (`nki`, `nki.language as nl`, `nki.isa as nisa`), `@nki.jit` decorator, function signature with input params, shape and dtype assertions for each input, HBM output allocation.

**Tensor naming**: input/output tensors use the same names as the math function (`q`, `k`, `v`, `output`). Intermediate tensors are named `{location}_{var_name}` where `location` is `sbuf` or `psum` and `var_name` is the variable name from the parsed math function (§4). Examples: `sbuf_Q_t`, `psum_S`, `sbuf_masked_S`.
2. **Emit loop nest**: walk the `loop_order` from the `Schedule`. Each entry's `(level, order)` determines nesting: entries at different levels produce nested loops (`nl.affine_range`), entries at the same level with different orders produce back-to-back sequential loops. Each `Loop` emits a block loop (`nl.affine_range(total_tiles // tiles_per_block)`) and tile loop (`nl.affine_range(tiles_per_block)`).
3. **Emit DMA loads**: for each input tensor, place the `nisa.dma_copy(HBM→SBUF)` at the `relative_pos` specified by `schedule.load_placements` (§5.4). Allocate an SBUF buffer with shape determined by `Tensor.NUM_TILES`.
4. **Emit PSUM accumulators**: before each dependent-dim loop, emit `nl.ndarray(..., buffer=nl.psum)` followed by `nisa.memset(buf, value=...)`. Always this two-step sequence — never `nl.zeros`.
   - **`nc_matmul`**: `nl.ndarray` + `nisa.memset(buf, value=0.0)`. Hardware accumulates — each matmul call adds its partial product to the same PSUM address, so a single buffer suffices.
   - **`tensor_reduce`**: when the reduction axis spans $T$ tiles in the loop, `nl.ndarray` a PSUM partial-results buffer with $T$ columns (one per tile), then `nisa.memset` with identity value (`0.0` for `add`, `-3.4e38` for `max`). Inside the loop, each tile's reduction result writes to its own column. After the loop, a second `nisa.tensor_reduce` collapses all columns into the final scalar result. This gather-then-reduce pattern is necessary because `tensor_reduce` overwrites its destination — unlike `nc_matmul`, it does not accumulate.
5. **Emit compute ops**: each op is emitted as soon as all its operands are available — not deferred to the innermost level. For each op, build a `RenderContext` (§2) and call `NKIOp.render(ctx)`. Vector/Scalar/GpSimd ops (tensor_scalar, activation, affine_select) operate on SBUF; the renderer stages PSUM→SBUF before these ops and SBUF→PSUM after if needed.
6. **Emit staging**: when an op needs an operand in SBUF but the data is currently in PSUM, emit `nisa.tensor_copy(PSUM→SBUF)` to stage it. This is triggered on demand — not at a fixed point in the pipeline.
7. **Emit DMA store**: `nisa.dma_copy(SBUF→HBM)` writes the final output tile back.

### 6.2 RenderContext Construction

For each op call site, the renderer builds a `RenderContext`:

- `outputs`: output `Tensor`(s) with axes, tile sizes, location (SBUF or PSUM), and the rendered buffer name (§6.1 step 1)
- `operands`: input `Tensor`(s) with their current location (SBUF or PSUM) and slice expressions for the current tile
- `config_kwargs`: pass-through from the math function call (e.g., `op0=nl.multiply`, `cmp_op=nl.greater_equal`)
- `tile_idx`: maps each dim_id to the current loop variable (e.g., `{"d0": "i_0", "d2": "i_2"}`)
- `tile_start`: maps each dim_id to the element offset expression (e.g., `{"d0": "i_0 * 128", "d2": "i_2 * 512"}`)

Each `NKIOp.render(ctx)` returns one or more NKI source lines using `nisa.*` calls.

### 6.3 Example Output

Generic attention with symbolic shapes `q[S_q, D_k]`, `k[S_k, D_k]`, `v[S_k, D_v]` and all `tiles_per_block=1` with `relative_pos=max`. Tile sizes from §5.1.1: $t_0 = 128$ (d0), $t_1 = 128$ (d1), $t_2 = 512$ (d2), $t_5 = \min(512, D_v)$ (d5). Total tiles per dim: $\text{total\_tiles}_i = S_i / t_i$.

The actual generated code always emits the double loop per §5.1.1 (e.g., `for i_d0_outer in nl.affine_range(total // 1)` + `for i_d0_inner in nl.affine_range(1)`). Below, the inner loops are elided for readability since `tiles_per_block=1` makes them single-iteration.

```python
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def attention_kernel(q, k, v, scale):
    """
    q: (d0=S_q, d1=D_k) — S_q divisible by t_0, D_k divisible by t_1
    k: (d2=S_k, d1=D_k) — S_k divisible by t_2, d1 unified with q
    v: (d2=S_k, d5=D_v) — d2 unified with k, D_v divisible by t_5
    output: (d0=S_q, d5=D_v)
    """

    d0_tileotal_tiles = S_q // t_0
    d1_tileotal_tiles = D_k // t_1
    d2_tileotal_tiles = S_k // t_2
    d5_tileotal_tiles = D_v // t_5

    output_hbm = nl.ndarray((S_q, D_v), dtype=q.dtype, buffer=nl.shared_hbm)

    for i_d0 in nl.affine_range(d0_tileotal_tiles):
      d0_off = i_d0 * t_0
      for i_d5 in nl.affine_range(d5_tileotal_tiles):
        d5_off = i_d5 * t_5

        """ psum_S — axes=(d0, d2): holds all d2 tiles for matmul1 accumulation.
        d0: 1 tile (inside loop), d2: d2_tileotal_tiles (accumulator spans full dim). """
        psum_S = nl.ndarray(
            (t_0, 1, 1, d2_tileotal_tiles, 1, t_2),
            dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_S, value=0.0)

        for i_d1 in nl.affine_range(d1_tileotal_tiles):
          d1_off = i_d1 * t_1

          """ sbuf_Q — axes=(d0, d1): 1 tile each (tiles_per_block=1 for both). """
          sbuf_Q = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_1),
              dtype=q.dtype, buffer=nl.sbuf)
          nisa.dma_copy(
              dst=sbuf_Q[0:t_0, 0, 0, 0, 0, 0:t_1],
              src=q[d0_off:d0_off+t_0, d1_off:d1_off+t_1])

          psum_Q_t = nl.ndarray(
              (t_1, 1, 1, 1, 1, t_0),
              dtype=nl.float32, buffer=nl.psum)
          nisa.nc_transpose(
              dst=psum_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
              data=sbuf_Q[0:t_0, 0, 0, 0, 0, 0:t_1])
          sbuf_Q_t = nl.ndarray(
              (t_1, 1, 1, 1, 1, t_0),
              dtype=q.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(
              dst=sbuf_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
              src=psum_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0])

          for i_d2 in nl.affine_range(d2_tileotal_tiles):
            d2_off = i_d2 * t_2

            """ Sub-loop: d2 tile_size (512) exceeds nc_transpose P:128.
            Each sub-iteration transposes a (128, t_1) chunk of K
            and feeds it to nc_matmul at the corresponding output slice. """
            for i_d2_sub in nl.affine_range(t_2 // 128):

              sbuf_K_sub = nl.ndarray(
                  (128, 1, 1, 1, 1, t_1),
                  dtype=k.dtype, buffer=nl.sbuf)
              nisa.dma_copy(
                  dst=sbuf_K_sub[0:128, 0, 0, 0, 0, 0:t_1],
                  src=k[d2_off + i_d2_sub * 128
                        : d2_off + (i_d2_sub + 1) * 128,
                        d1_off:d1_off+t_1])

              psum_K_t_sub = nl.ndarray(
                  (t_1, 1, 1, 1, 1, 128),
                  dtype=nl.float32, buffer=nl.psum)
              nisa.nc_transpose(
                  dst=psum_K_t_sub[0:t_1, 0, 0, 0, 0, 0:128],
                  data=sbuf_K_sub[0:128, 0, 0, 0, 0, 0:t_1])
              sbuf_K_t_sub = nl.ndarray(
                  (t_1, 1, 1, 1, 1, 128),
                  dtype=k.dtype, buffer=nl.sbuf)
              nisa.tensor_copy(
                  dst=sbuf_K_t_sub[0:t_1, 0, 0, 0, 0, 0:128],
                  src=psum_K_t_sub[0:t_1, 0, 0, 0, 0, 0:128])

              nisa.nc_matmul(
                  dst=psum_S[0:t_0, 0, 0, i_d2, 0,
                             i_d2_sub * 128 : (i_d2_sub + 1) * 128],
                  stationary=sbuf_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
                  moving=sbuf_K_t_sub[0:t_1, 0, 0, 0, 0, 0:128])

        """ Gather-then-reduce for max: one column per d2 tile. """
        psum_partial_max = nl.ndarray(
            (t_0, 1, d2_tileotal_tiles), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_partial_max, value=-3.4028235e38)

        for i_d2 in nl.affine_range(d2_tileotal_tiles):
          d2_off = i_d2 * t_2
          sbuf_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_copy(
              dst=sbuf_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              src=psum_S[0:t_0, 0, 0, i_d2, 0, 0:t_2])
          sbuf_scaled_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_scalar(
              dst=sbuf_scaled_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              data=sbuf_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              op0=nl.multiply, operand0=scale)
          sbuf_masked_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.affine_select(
              dst=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              pattern=[[-1, t_2]],
              offset=d0_off - d2_off,
              channel_multiplier=1,
              on_true_tile=sbuf_scaled_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              on_false_value=-3.4028235e38,
              cmp_op=nl.greater_equal)
          nisa.tensor_reduce(
              dst=psum_partial_max[0:t_0, 0, i_d2],
              op=nl.maximum,
              data=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              axis=(1,))
          nisa.tensor_copy(
              dst=psum_S[0:t_0, 0, 0, i_d2, 0, 0:t_2],
              src=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2])

        psum_max_S = nl.ndarray(
            (t_0, 1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(
            psum_max_S[0:t_0, 0, 0:1],
            nl.maximum, psum_partial_max[0:t_0, 0, 0:d2_tileotal_tiles], 1)
        sbuf_max_S = nl.ndarray(
            (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=sbuf_max_S[0:t_0, 0, 0:1],
            src=psum_max_S[0:t_0, 0, 0:1])

        """ Gather-then-reduce for sum: one column per d2 tile. """
        psum_partial_sum = nl.ndarray(
            (t_0, 1, d2_tileotal_tiles), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_partial_sum, value=0.0)

        for i_d2 in nl.affine_range(d2_tileotal_tiles):
          sbuf_masked_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_copy(
              dst=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              src=psum_S[0:t_0, 0, 0, i_d2, 0, 0:t_2])
          sbuf_shifted_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_scalar(
              dst=sbuf_shifted_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              data=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              op0=nl.subtract,
              operand0=sbuf_max_S[0:t_0, 0, 0:1])
          sbuf_exp_S = nl.ndarray(
              (t_0, 1, 1, 1, 1, t_2),
              dtype=nl.float32, buffer=nl.sbuf)
          nisa.activation(
              dst=sbuf_exp_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              op=nl.exp,
              data=sbuf_shifted_S[0:t_0, 0, 0, 0, 0, 0:t_2])
          nisa.tensor_reduce(
              dst=psum_partial_sum[0:t_0, 0, i_d2],
              op=nl.add,
              data=sbuf_exp_S[0:t_0, 0, 0, 0, 0, 0:t_2],
              axis=(1,))
          nisa.tensor_copy(
              dst=psum_S[0:t_0, 0, 0, i_d2, 0, 0:t_2],
              src=sbuf_exp_S[0:t_0, 0, 0, 0, 0, 0:t_2])

        psum_sum_exp = nl.ndarray(
            (t_0, 1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(
            psum_sum_exp[0:t_0, 0, 0:1],
            nl.add, psum_partial_sum[0:t_0, 0, 0:d2_tileotal_tiles], 1)
        sbuf_sum_exp = nl.ndarray(
            (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=sbuf_sum_exp[0:t_0, 0, 0:1],
            src=psum_sum_exp[0:t_0, 0, 0:1])

        sbuf_inv_sum = nl.ndarray(
            (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=sbuf_inv_sum[0:t_0, 0, 0:1],
            op=nl.reciprocal,
            data=sbuf_sum_exp[0:t_0, 0, 0:1])

        """ psum_attn — axes=(d0, d5): matmul2 accumulator. """
        psum_attn = nl.ndarray(
            (t_0, 1, 1, 1, 1, t_5),
            dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_attn, value=0.0)

        for i_d2 in nl.affine_range(d2_tileotal_tiles):
          d2_off = i_d2 * t_2

          """ Sub-loop: d2 tile_size (512) exceeds nc_transpose P:128 and F:128.
          Each sub-iteration transposes a (t_0, 128) chunk of exp_S,
          loads the corresponding (128, t_5) chunk of V,
          and accumulates into psum_attn. """
          for i_d2_sub in nl.affine_range(t_2 // 128):

            sbuf_exp_S_sub = nl.ndarray(
                (t_0, 1, 1, 1, 1, 128),
                dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=sbuf_exp_S_sub[0:t_0, 0, 0, 0, 0, 0:128],
                src=psum_S[0:t_0, 0, 0, i_d2, 0,
                           i_d2_sub * 128 : (i_d2_sub + 1) * 128])

            psum_exp_S_t_sub = nl.ndarray(
                (128, 1, 1, 1, 1, t_0),
                dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=psum_exp_S_t_sub[0:128, 0, 0, 0, 0, 0:t_0],
                data=sbuf_exp_S_sub[0:t_0, 0, 0, 0, 0, 0:128])
            sbuf_exp_S_t_sub = nl.ndarray(
                (128, 1, 1, 1, 1, t_0),
                dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=sbuf_exp_S_t_sub[0:128, 0, 0, 0, 0, 0:t_0],
                src=psum_exp_S_t_sub[0:128, 0, 0, 0, 0, 0:t_0])

            sbuf_V_sub = nl.ndarray(
                (128, 1, 1, 1, 1, t_5),
                dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=sbuf_V_sub[0:128, 0, 0, 0, 0, 0:t_5],
                src=v[d2_off + i_d2_sub * 128
                      : d2_off + (i_d2_sub + 1) * 128,
                      d5_off:d5_off+t_5])

            nisa.nc_matmul(
                dst=psum_attn[0:t_0, 0, 0, 0, 0, 0:t_5],
                stationary=sbuf_exp_S_t_sub[0:128, 0, 0, 0, 0, 0:t_0],
                moving=sbuf_V_sub[0:128, 0, 0, 0, 0, 0:t_5])

        sbuf_attn = nl.ndarray(
            (t_0, 1, 1, 1, 1, t_5),
            dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=sbuf_attn[0:t_0, 0, 0, 0, 0, 0:t_5],
            src=psum_attn[0:t_0, 0, 0, 0, 0, 0:t_5])
        sbuf_output = nl.ndarray(
            (t_0, 1, 1, 1, 1, t_5),
            dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sbuf_output[0:t_0, 0, 0, 0, 0, 0:t_5],
            data=sbuf_attn[0:t_0, 0, 0, 0, 0, 0:t_5],
            op0=nl.multiply,
            operand0=sbuf_inv_sum[0:t_0, 0, 0:1])
        nisa.dma_copy(
            dst=output_hbm[d0_off:d0_off+t_0,
                           d5_off:d5_off+t_5],
            src=sbuf_output[0:t_0, 0, 0, 0, 0, 0:t_5])

    return output_hbm
```

Each tensor's shape comment shows its axes and the symbolic shape `(tile_par, nb_0, nb_1, nt_0, nt_1, tile_free)`. The nb and nt values for each axis come from the load placement (§5.4):
- **after compute loop**: nb=1, nt=1 (inside both load and compute loops)
- **between load/compute**: nb=1, nt=tiles_per_block (inside load loop, before compute loop)
- **before load loop**: nb=total_tiles/tiles_per_block, nt=tiles_per_block (outside all loops for that dim)

---

## 7. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel (`attention_cte.py`) already uses `nl.ds` for this purpose.
