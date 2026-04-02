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

## 1. Math Function
The top-level IR math function only defines the math computations. There are no load/store operators or hardware tiling, and it ignores the memory hierarchy.
```python
def attention(Q, K, V, scale):
    Q_t = nkigym.nc_transpose(Q)
    K_t = nkigym.nc_transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    masked_S = nkigym.affine_select(S, cmp_op="greater_equal",
                                     on_false_value=-np.inf,
                                     channel_multiplier=1, step=-1)
    scaled_S = nkigym.tensor_scalar(masked_S, op0="multiply",
                                     operand0=scale)
    neg_max_S = nkigym.tensor_reduce(scaled_S, reduce_op="max",
                          negate=True)
    exp_S, sum_exp = nkigym.activation_reduce(scaled_S, neg_max_S,
                          op="exp", reduce_op="add")
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    exp_S_t = nkigym.nc_transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")
    return output
```

![Math function DAG — Op 7 and Op 9 are topologically independent, converging only at Op 10](diagrams/math_function_dag.png)

## 2. NKI Gym Operators

Every `nkigym.*` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented.

```python
class Tensor:
    """
    Represents a N-dimensional NKI tensor in the shape of:
    [tile_size[AXES[0]],
        num_blocks[AXES[0]], ..., num_blocks[AXES[N-1]],
        tiles_per_block[AXES[0]], ..., tiles_per_block[AXES[N-1]],
        tile_size[AXES[1]], ..., tile_size[AXES[N-1]]
    ]
    """
    NAME: str
    AXES: tuple[str, ...]
    tile_size: dict[str, int]
    num_blocks: dict[str, int]
    tiles_per_block: dict[str, int]
    LOCATION: str # SBUF, PSUM

class NKIOp:
    """
    Represents a NKI operator semantics
    Supports multi outputs
    """
    NAME: str
    OPERAND_AXES: dict[str, tuple[str, ...]]
    OUTPUT_AXES: dict[str, tuple[str, str, ...]]
    MAX_TILE_SIZES: dict[str, int]
    ENGINE: str

    @abstractmethod
    def __call__(self, **kwargs):
        """
        CPU simulation using Numpy in default float64.
        Takes input arrays + config, returns output array(s).
        """

    def render(self, ctx: "RenderContext") -> str:
        """
        Emit a complete loop nest for this op.

        1. Allocate full-range output buffer in SBUF.
        2. Parallel double loops (block + tile) for each
           dimension in OUTPUT_AXES.
        3. If consumed dims exist: allocate PSUM accumulator
           (tile-sized), memset to init value, then emit
           reduction double loops.
        4. The ISA call (subclass render_isa()).
        5. If consumed dims exist: tensor_copy PSUM → SBUF
           output buffer after the reduction loop closes.

        Each dimension produces a double loop:
          for i_block in range(num_blocks):
              for i_tile in range(tiles_per_block):
                  [body]

        Both loops are always emitted regardless of
        tiles_per_block value. The slot between block
        and tile loops is where DMA loads are placed
        (§3.2).
        """
        lines = []
        out = next(iter(ctx.outputs.values()))
        consumed = self._consumed_dims(ctx)  # dims reduced over (not in output)

        """Step 1: output buffer (full-range)"""
        lines.append(f"{out.NAME} = nl.ndarray({out.shape}, ...)")

        """Step 2: parallel double loops for output dims"""
        for dim_id in out.AXES:
            lines.append(
                f"for i_block_{dim_id} in "
                f"nl.affine_range({out.num_blocks[dim_id]}):"
            )
            lines.append(
                f"  for i_tile_{dim_id} in "
                f"nl.affine_range({out.tiles_per_block[dim_id]}):"
            )

        """Step 3: reduction dims (if any)"""
        for dim_id in consumed:
            lines.append(
                f"    psum_acc = nl.ndarray(..., buffer=nl.psum)"
            )
            init = self._init_value(dim_id)  # e.g. 0 for sum, -inf for max
            lines.append(f"    nisa.memset(psum_acc, {init})")
            op = ctx.operand_with_dim(dim_id)  # operand spanning this dim
            lines.append(
                f"    for i_block_{dim_id} in "
                f"nl.affine_range({op.num_blocks[dim_id]}):"
            )
            lines.append(
                f"      for i_tile_{dim_id} in "
                f"nl.affine_range({op.tiles_per_block[dim_id]}):"
            )

        """Step 4: ISA call (subclass)"""
        lines.append(self.render_isa(ctx))

        """Step 5: copy PSUM → SBUF after reduction"""
        if consumed:
            lines.append(
                f"    nisa.tensor_copy({out.NAME}, psum_acc)"
            )
        return "\n".join(lines)

    @abstractmethod
    def render_isa(self, ctx: "RenderContext") -> str:
        """Emit the ISA call inside the innermost loop.
        If any tile dim exceeds MAX_TILE_SIZES, emit
        an in-place sub-loop that iterates in chunks.
        """

class RenderContext:
    """
    Everything an NKIOp.render() needs to emit NKI source.
    Built for each op call site during kernel generation.
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
    Sub-loop: if K exceeds K:128, render() emits a sub-loop that splits the
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

    def render_isa(self, ctx):
        dst = ctx.outputs["output"]
        stat = ctx.operands["stationary"]
        mov = ctx.operands["moving"]
        k_size = stat.tile_size[stat.AXES[0]]
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

    Sub-loop: if P or F exceeds 128, render() emits a
    sub-loop that transposes (128, 128) chunks, writing each
    to the corresponding slice of the output.
    """

    NAME = "nc_transpose"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("F", "P")}
    MAX_TILE_SIZES = {"P": 128, "F": 128}
    ENGINE = "TensorEngine"

    def __call__(self, data):
        return data.T

    def render_isa(self, ctx):
        dst = ctx.outputs["output"]
        src = ctx.operands["data"]
        p_size = src.tile_size[src.AXES[0]]
        f_size = src.tile_size[src.AXES[1]]
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

    def render_isa(self, ctx):
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

    def render_isa(self, ctx):
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
            f"pattern=[[{kw['step']}, {data.tile_size[f_dim]}]], "
            f"offset={offset}, "
            f"channel_multiplier={kw['channel_multiplier']}, "
            f"cmp_op=nl.{kw['cmp_op']}, "
            f"on_true_tile={data.NAME}, "
            f"on_false_value={kw['on_false_value']})"
        )


class NKITensorReduce(NKIOp):
    """Reduce free axis with optional negation (Vector Engine)

    data(P, F) → output(P,).
    Collapses free dimension with the specified reduction op.
    When negate=True, the output is negated after reduction.
    Both input and output can be SBUF or PSUM.
    """

    NAME = "tensor_reduce"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("P",)}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "VectorEngine"

    _NKI_REDUCE_OPS = {"max": "maximum", "add": "add"}

    def __call__(self, data, op, negate=False):
        reduce = {"max": np.max, "add": np.sum}
        result = reduce[op](data, axis=1)
        if negate:
            result = -result
        return result

    def render_isa(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["op"]]
        negate = ctx.config_kwargs.get("negate", False)
        negate_str = ", negate=True" if negate else ""
        return f"nisa.tensor_reduce(dst={dst.NAME}, data={data.NAME}, op=nl.{op_name}, axis=1{negate_str})"


class NKIActivationReduce(NKIOp):
    """Element-wise activation with simultaneous free-axis reduction (Scalar Engine)

    op(data + bias) → output(P, F), and simultaneously
    reduce_op(output) along free axis → reduce_res(P,).
    Dual-output: produces both the activation result and the
    reduction result in a single ISA call.
    bias is a (P,) column vector broadcast across the free axis
    (typically the negated row-wise max for numerically stable exp).
    """

    NAME = "activation_reduce"
    OPERAND_AXES = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES = {"output": ("P", "F"), "reduce_res": ("P",)}
    MAX_TILE_SIZES = {"P": 128}
    ENGINE = "ScalarEngine"

    _NKI_REDUCE_OPS = {"max": "maximum", "add": "add"}

    def __call__(self, data, bias, op, reduce_op):
        fns = {
            "exp": np.exp, "tanh": np.tanh, "square": np.square,
            "reciprocal": lambda x: 1.0 / x,
        }
        reduce = {"max": np.max, "add": np.sum}
        elem_result = fns[op](data + bias[..., np.newaxis])
        reduce_result = reduce[reduce_op](elem_result, axis=1)
        return elem_result, reduce_result

    def render_isa(self, ctx):
        dst = ctx.outputs["output"]
        red = ctx.outputs["reduce_res"]
        data = ctx.operands["data"]
        bias = ctx.operands["bias"]
        op_name = ctx.config_kwargs["op"]
        reduce_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["reduce_op"]]
        return (
            f"nisa.activation_reduce(dst={dst.NAME}, data={data.NAME}, "
            f"op=nl.{op_name}, bias={bias.NAME}, "
            f"reduce_op=nl.{reduce_name}, reduce_res={red.NAME})"
        )


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

    def render_isa(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
        return f"nisa.activation(dst={dst.NAME}, data={data.NAME}, op=nl.{op_name})"
```

### 2.2 CPU Simulation

The math function is plain Python — each `nkigym.*` call dispatches to `NKIOp.__call__()` which executes the op with numpy at float64 precision. No parsing or IR needed; just call the function directly:

```python
q = np.random.randn(4096, 128)
k = np.random.randn(4096, 128)
v = np.random.randn(4096, 128)
output = attention(q, k, v, scale=1.0 / np.sqrt(128))
```

The result is the **reference output** — a float64 array that any correctly rendered and compiled NKI kernel must match (within hardware precision tolerance).

---

## 3. Initial Eager Mode Kernel

The initial eager mode kernel chains `NKIOp.render()` calls — one per op in math function order. Each op emits its own independent loop nest; the kernel is their concatenation. This is correct by construction but naive in performance: every intermediate is a full-range buffer, and no loops are shared between ops. Variant transforms (§4 programmatic, §5 math) optimize from this baseline.

### 3.1 Eager Loop Nest: Chaining `NKIOp.render()`

The eager loop nest is a simple orchestrator. Before rendering, an analysis pass walks the math function to build `Tensor` and `RenderContext` objects for each op — assigning dimension IDs (unifying shared axes), computing tile sizes from `MAX_TILE_SIZES`, and setting `tiles_per_block=1` everywhere. Then the generator concatenates each op's `render()` output in math function order:

```python
for op, ctx in zip(math_function.ops, render_contexts):
    kernel_lines += op.render(ctx)
```

Each `render()` call emits its own complete, independent loop nest (§2.1). The result is 11 back-to-back loop nests — one per op — communicating through full-range intermediate buffers. The running example produces:

| # | Op | Output dims | Consumed dim | Init |
|---|---|---|---|---|
| 0 | nc_transpose(Q) → Q_t | d1, d0 | — | — |
| 1 | nc_transpose(K) → K_t | d1, d2 | — | — |
| 2 | nc_matmul(Q_t, K_t) → S | d0, d2 | d1 | 0 |
| 3 | affine_select(S) → masked_S | d0, d2 | — | — |
| 4 | tensor_scalar(masked_S, scale, multiply) → scaled_S | d0, d2 | — | — |
| 5 | tensor_reduce(scaled_S, max, negate) → neg_max_S | d0 | d2 | $-\infty$ |
| 6 | activation_reduce(scaled_S, neg_max_S, exp, add) → (exp_S, sum_exp) | (d0, d2), (d0) | —, d2 | —, 0 |
| 7 | activation(sum_exp, reciprocal) → inv_sum | d0 | — | — |
| 8 | nc_transpose(exp_S) → exp_S_t | d2, d0 | — | — |
| 9 | nc_matmul(exp_S_t, V) → attn | d0, d5 | d2 | 0 |
| 10 | tensor_scalar(attn, inv_sum) → output | d0, d5 | — | — |

Dimensions after unification: **d0** (seq_q, 128×32), **d1** (d_k, 128×1), **d2** (seq_k, 512×8), **d5** (d_v, 128×1) — format is tile_size × num_blocks.

**Naming conventions:**
- **Loop variables**: `i_block_d{id}` / `i_tile_d{id}` (e.g., `i_block_d0`, `i_tile_d2`).
- **Tensor buffers**: `{location}_{name}` (e.g., `sbuf_Q_t`, `psum_S`). DMA staging buffers use the input parameter name (e.g., `sbuf_Q`).

### 3.2 NKI Kernel with Greedy DMA Placement

The eager loop nest (§3.1) operates on abstract tensors. To produce a concrete NKI kernel, we assign each tensor to a memory space and insert explicit DMA where needed. Only two kinds of `nisa.dma_copy` appear:

- **Input loads**: `Q`, `K`, `V` are loaded from HBM to SBUF before the op that first uses them. Each load is placed at the **innermost loop level** — the greediest (most frequent) placement. Marked with `# DMA load` comments in the kernel.
- **Output store**: the final `output` tensor is stored from SBUF to HBM after the last op. Marked with `# DMA store` comments in the kernel.

Each intermediate tensor (`Q_t`, `S`, `scaled_S`, …) is allocated as a **full-range** `nl.ndarray` buffer in SBUF right before the op that produces it. Their shapes span the complete tensor dimensions so they persist across operators — each op writes tile-indexed slices into its output buffer; subsequent ops read from the same buffer at the appropriate tile positions. This makes the data flow between separate loop nests explicit and mathematically correct, though the total SBUF allocation far exceeds hardware capacity (the running example needs ~260 MB of SBUF for intermediates alone, vs 24 MB available). Variant transforms (§4) shrink these buffers by fusing loop nests (eliminating intermediates entirely), hoisting loads, and reordering dimensions.

```python
@nki.jit
def attention_kernel(Q, K, V):
    """ Q: (4096, 128)  K: (4096, 128)  V: (4096, 128) """
    """ d0: 128x32  d1: 128x1  d2: 512x8  d5: 128x1  (tile_size x num_blocks) """
    output = nl.ndarray((4096, 128), dtype=Q.dtype, buffer=nl.shared_hbm)

    """ Op 0: nisa.nc_transpose -- Q(d0, d1) -> Q_t(d1, d0) """
    sbuf_Q_t = nl.ndarray((128, 1, 32, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d1 in nl.affine_range(1):
                for i_tile_d1 in nl.affine_range(1):
                    sbuf_Q = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)              # DMA load Q
                    nisa.dma_copy(dst=sbuf_Q[0:128, 0, 0, 0, 0, 0:128], src=Q[i_block_d0*128:i_block_d0*128+128, i_block_d1*128:i_block_d1*128+128])
                    nisa.nc_transpose(dst=sbuf_Q_t[0:128, i_block_d1, i_block_d0, i_tile_d1, i_tile_d0, 0:128], src=sbuf_Q[0:128, 0, 0, 0, 0, 0:128])

    """ Op 1: nisa.nc_transpose -- K(d2, d1) -> K_t(d1, d2) """
    sbuf_K_t = nl.ndarray((128, 1, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d1 in nl.affine_range(1):
        for i_tile_d1 in nl.affine_range(1):
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    for i_p_sub in nl.affine_range(4):                                                       # sub-loop: P=512 > MAX_P=128
                        sbuf_K = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=K.dtype, buffer=nl.sbuf)          # DMA load K (128-row chunk)
                        nisa.dma_copy(dst=sbuf_K[0:128, 0, 0, 0, 0, 0:128], src=K[i_block_d2*512+i_p_sub*128:i_block_d2*512+i_p_sub*128+128, i_block_d1*128:i_block_d1*128+128])
                        nisa.nc_transpose(dst=sbuf_K_t[0:128, i_block_d1, i_block_d2, i_tile_d1, i_tile_d2, i_p_sub*128:i_p_sub*128+128], src=sbuf_K[0:128, 0, 0, 0, 0, 0:128])

    """ Op 2: nisa.nc_matmul -- Q_t(d1, d0) x K_t(d1, d2) -> S(d0, d2), accumulate over d1 """
    sbuf_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=nl.float32, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    psum_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    for i_block_d1 in nl.affine_range(1):
                        for i_tile_d1 in nl.affine_range(1):
                            nisa.nc_matmul(dst=psum_S[0:128, 0, 0, 0, 0, 0:512], stationary=sbuf_Q_t[0:128, i_block_d1, i_block_d0, i_tile_d1, i_tile_d0, 0:128], moving=sbuf_K_t[0:128, i_block_d1, i_block_d2, i_tile_d1, i_tile_d2, 0:512])
                    nisa.tensor_copy(dst=sbuf_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], src=psum_S[0:128, 0, 0, 0, 0, 0:512])

    """ Op 3: nisa.affine_select -- S(d0, d2) -> masked_S(d0, d2) """
    sbuf_masked_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    nisa.affine_select(dst=sbuf_masked_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], pred=sbuf_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512])

    """ Op 4: nisa.tensor_scalar -- masked_S(d0, d2) x scale -> scaled_S(d0, d2) """
    sbuf_scaled_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    nisa.tensor_scalar(dst=sbuf_scaled_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], data=sbuf_masked_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], op0=nl.multiply, operand0=scale)

    """ Op 5: nisa.tensor_reduce -- max(scaled_S(d0, d2)) negate -> neg_max_S(d0), reduce over d2 """
    sbuf_neg_max_S = nl.ndarray((128, 32, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            psum_partial_max = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    nisa.tensor_reduce(dst=psum_partial_max[0:128, i_block_d2:i_block_d2+1], data=sbuf_scaled_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], op=nl.maximum, axis=1)
            nisa.tensor_reduce(dst=sbuf_neg_max_S[0:128, i_block_d0, 0:1], data=psum_partial_max[0:128, 0:8], op=nl.maximum, axis=1, negate=True)

    """ Op 6: nisa.activation_reduce -- exp(scaled_S(d0, d2) + neg_max_S(d0)) -> exp_S(d0, d2) + add -> sum_exp(d0), reduce over d2 """
    sbuf_exp_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
    sbuf_sum_exp = nl.ndarray((128, 32, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            psum_sum_exp = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(dst=psum_sum_exp[0:128, 0, 0:1], value=0.0)
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    nisa.activation_reduce(dst=sbuf_exp_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], data=sbuf_scaled_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, 0:512], op=nl.exp, bias=sbuf_neg_max_S[0:128, i_block_d0, 0:1], reduce_op=nl.add, reduce_res=psum_sum_exp[0:128, 0, 0:1])
            nisa.tensor_copy(dst=sbuf_sum_exp[0:128, i_block_d0, 0:1], src=psum_sum_exp[0:128, 0, 0:1])

    """ Op 7: nisa.activation -- reciprocal(sum_exp(d0)) -> inv_sum(d0) """
    sbuf_inv_sum = nl.ndarray((128, 32, 1), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            nisa.activation(dst=sbuf_inv_sum[0:128, i_block_d0, 0:1], data=sbuf_sum_exp[0:128, i_block_d0, 0:1], op=nl.reciprocal)

    """ Op 8: nisa.nc_transpose -- exp_S(d0, d2) -> exp_S_t(d2, d0) """
    sbuf_exp_S_t = nl.ndarray((512, 8, 32, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d2 in nl.affine_range(8):
                for i_tile_d2 in nl.affine_range(1):
                    for i_f_sub in nl.affine_range(4):                                                       # sub-loop: F=512 > MAX_F=128
                        nisa.nc_transpose(dst=sbuf_exp_S_t[i_f_sub*128:i_f_sub*128+128, i_block_d2, i_block_d0, i_tile_d2, i_tile_d0, 0:128], src=sbuf_exp_S[0:128, i_block_d0, i_block_d2, i_tile_d0, i_tile_d2, i_f_sub*128:i_f_sub*128+128])

    """ Op 9: nisa.nc_matmul -- exp_S_t(d2, d0) x V(d2, d5) -> attn(d0, d5), accumulate over d2 """
    sbuf_attn = nl.ndarray((128, 32, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d5 in nl.affine_range(1):
                for i_tile_d5 in nl.affine_range(1):
                    psum_attn = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
                    for i_block_d2 in nl.affine_range(8):
                        for i_tile_d2 in nl.affine_range(1):
                            sbuf_V = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)      # DMA load V
                            nisa.dma_copy(dst=sbuf_V[0:512, 0, 0, 0, 0, 0:128], src=V[i_block_d2*512:i_block_d2*512+512, i_block_d5*128:i_block_d5*128+128])
                            for i_k_sub in nl.affine_range(4):                                               # sub-loop: K=512 > MAX_K=128
                                nisa.nc_matmul(dst=psum_attn[0:128, 0, 0, 0, 0, 0:128], stationary=sbuf_exp_S_t[i_k_sub*128:i_k_sub*128+128, i_block_d2, i_block_d0, i_tile_d2, i_tile_d0, 0:128], moving=sbuf_V[i_k_sub*128:i_k_sub*128+128, 0, 0, 0, 0, 0:128])
                    nisa.tensor_copy(dst=sbuf_attn[0:128, i_block_d0, i_block_d5, i_tile_d0, i_tile_d5, 0:128], src=psum_attn[0:128, 0, 0, 0, 0, 0:128])

    """ Op 10: nisa.tensor_scalar -- attn(d0, d5) x inv_sum(d0) -> output(d0, d5) """
    for i_block_d0 in nl.affine_range(32):
        for i_tile_d0 in nl.affine_range(1):
            for i_block_d5 in nl.affine_range(1):
                for i_tile_d5 in nl.affine_range(1):
                    sbuf_output = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=sbuf_output[0:128, 0, 0, 0, 0, 0:128], data=sbuf_attn[0:128, i_block_d0, i_block_d5, i_tile_d0, i_tile_d5, 0:128], op0=nl.multiply, operand0=sbuf_inv_sum[0:128, i_block_d0, 0:1])
                    nisa.dma_copy(dst=output[i_block_d0*128:i_block_d0*128+128, i_block_d5*128:i_block_d5*128+128], src=sbuf_output[0:128, 0, 0, 0, 0, 0:128])  # DMA store output

    return output
```

## 4. Programmatic Transforms

Programmatic transforms are mechanical rearrangements of the loop nest that do not change the math — they only change how and when ops execute. All transforms must respect the topological computation order: every op's inputs must be computed before the op runs. The math function (§1) defines one valid topological order, and the eager mode variant follows it exactly. Transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

Four transforms generate the search space.

### 4.1 Loop Fusion

Adjacent ops whose loops iterate over the same dimensions can share a single loop nest instead of having separate nests. Fusion eliminates redundant loop overhead and — critically — keeps intermediate tiles on-chip between producer and consumer, avoiding HBM round-trips. A dimension is **blocking** when the full reduction over that dimension must complete before a valid result exists (e.g., matmul accumulation, reduce_max). A dimension is **non-blocking** when partial results are already correct, just incomplete. Fusion is legal across any non-blocking boundary. It is forbidden across a blocking boundary for the blocking dimension: the reduction must complete before the consumer starts.

**Before** — Ops 3 and 4 each have their own loop nest; `sbuf_masked_S` is a full-range buffer:

```python
""" Op 3: nisa.affine_select -- S(d0, d2) -> masked_S(d0, d2) """
sbuf_masked_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                nisa.affine_select(dst=sbuf_masked_S[...], pred=sbuf_S[...])

""" Op 4: nisa.tensor_scalar -- masked_S(d0, d2) x scale -> scaled_S(d0, d2) """
sbuf_scaled_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                nisa.tensor_scalar(dst=sbuf_scaled_S[...], data=sbuf_masked_S[...], op0=nl.multiply, operand0=scale)
```

**After** — single fused loop nest; `sbuf_masked_S` shrinks to per-tile:

```python
""" Ops 3+4 (fused): affine_select + tensor_scalar -> scaled_S(d0, d2) """
sbuf_scaled_S = nl.ndarray((128, 32, 8, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):
            for i_tile_d2 in nl.affine_range(1):
                sbuf_masked_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)  # per-tile, not full-range
                nisa.affine_select(dst=sbuf_masked_S[...], pred=sbuf_S[...])
                nisa.tensor_scalar(dst=sbuf_scaled_S[...], data=sbuf_masked_S[...], op0=nl.multiply, operand0=scale)
```

### 4.2 Load Placement

A DMA load can be **hoisted** (moved above a loop) or **sunk** (moved below a loop) from its current position in the loop nest. Hoisting means the loaded data persists across all iterations of that loop, amortizing DMA latency at the cost of a larger SBUF buffer. Sinking means the buffer is smaller (only one tile at a time) but the load executes more frequently. The slot between block and tile loops (§3.1) provides the placement points at each nesting level.

In the greedy baseline (§3.2), all loads start at the innermost position. After loop fusion and reordering, loads may need to move in either direction to balance SBUF pressure against DMA frequency.

**Before** — `sbuf_V` loaded inside the d2 reduction loop, reloaded every iteration:

```python
""" Op 9: nisa.nc_matmul -- exp_S_t x V -> attn(d0, d5), accumulate over d2 """
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                psum_attn = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
                for i_block_d2 in nl.affine_range(8):
                    for i_tile_d2 in nl.affine_range(1):
                        sbuf_V = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)  # inside d2
                        nisa.dma_copy(dst=sbuf_V[...], src=V[...])
                        nisa.nc_matmul(dst=psum_attn[...], stationary=sbuf_exp_S_t[...], moving=sbuf_V[...])
                nisa.tensor_copy(dst=sbuf_attn[...], src=psum_attn[...])
```

**After** — `sbuf_V` hoisted outside d2, loaded once per (d0, d5) tile; buffer grows to hold all d2 blocks:

```python
""" Op 9: nisa.nc_matmul -- exp_S_t x V -> attn(d0, d5), accumulate over d2 """
for i_block_d0 in nl.affine_range(32):
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                sbuf_V = nl.ndarray((512, 8, 1, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)  # floated up, all d2 blocks
                nisa.dma_copy(dst=sbuf_V[...], src=V[...])
                psum_attn = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
                for i_block_d2 in nl.affine_range(8):
                    for i_tile_d2 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_attn[...], stationary=sbuf_exp_S_t[...], moving=sbuf_V[...])
                nisa.tensor_copy(dst=sbuf_attn[...], src=psum_attn[...])
```

### 4.3 Loop Reordering

The order of dimension loops can be permuted, subject to: (1) same-dimension phase ordering (block before tile), (2) cross-dimension data dependencies (if phase B consumes a tensor produced by phase A of a different dimension, A must precede B), and (3) in-place buffer reuse constraints (a parallel dim not in the buffer's dimensions cannot wrap phases that write the buffer).

**Before** — default order d0 > d2 > d1:

```python
""" Op 2: nisa.nc_matmul -- Q_t x K_t -> S(d0, d2), default order: d0 > d2 > d1 """
for i_block_d0 in nl.affine_range(32):                          # d0 outermost
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(8):                   # d2 middle
            for i_tile_d2 in nl.affine_range(1):
                psum_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
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
                psum_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
                for i_block_d1 in nl.affine_range(1):           # d1 innermost (reduction)
                    for i_tile_d1 in nl.affine_range(1):
                        nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

### 4.4 Tiles Per Block

`tiles_per_block` is a per-dimension parameter that controls the block size — how many tiles are grouped before the tile loop iterates. Larger `tiles_per_block` means larger SBUF buffers but fewer DMA round-trips: the loaded block is reused across `tiles_per_block` compute iterations. Must divide `total_tiles`.

**Before** — `tiles_per_block=1`: 8 blocks, 1 tile each:

```python
""" Op 2 (d2 dimension): tiles_per_block=1 -- 8 blocks x 1 tile """
for i_block_d2 in nl.affine_range(8):                           # 8 blocks
    for i_tile_d2 in nl.affine_range(1):                        # 1 tile per block
        psum_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
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
        psum_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
        for i_block_d1 in nl.affine_range(1):
            for i_tile_d1 in nl.affine_range(1):
                nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
        nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
```

## 5. Math Transforms

Math transforms restructure the algorithm itself — changing the computation to break blocking dependencies. Programmatic transforms (§4) cannot fuse loops across blocking dependencies; math transforms eliminate those dependencies so that programmatic transforms can then optimize the resulting loop nest.

### 5.1 Online Fusion

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

#### 5.1.1 The X + Accumulation Pattern

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
| Op 5 (reduce max over d2) → Op 6 (exp + sum over d2) | d2 | **Online fusion** | X+Accumulation pattern (§5.1.2) |
| Op 6 (exp_S depends on max) → Op 9 (matmul over d2) | d2 | **Online fusion** | Same X (running max), same $s_k$ (§5.1.3) |

Online fusion breaks the Op 5→6 and Op 6→9 barriers, pulling Ops 5, 6, 8, 9 into a single d2 loop. Op 7 is topologically independent from Op 9 (§1 DAG). Op 10 depends on Op 9's output but runs after the d2 reduction completes. Neither is part of the online fusion transforms. The result is the flash attention algorithm.

#### 5.1.2 Online Fusion: Op 5→6

**Before.** Ops 5 and 6 run in two separate d2 loops. Op 5 reduces max over the full d2 range to produce `neg_max_S`. Op 6 uses `neg_max_S` as a bias — it cannot start until Op 5 completes:

```python
""" Op 5: tensor_reduce max over d2 -> neg_max_S """
psum_partial_max = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.tensor_reduce(dst=psum_partial_max[0:128, i_block_d2:i_block_d2+1],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.maximum, axis=1)
nisa.tensor_reduce(dst=sbuf_neg_max_S[0:128, 0, 0:1],
    data=psum_partial_max[0:128, 0:8], op=nl.maximum, axis=1, negate=True)

""" Op 6: activation_reduce exp+sum over d2 -> exp_S, sum_exp """
psum_sum_exp = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_sum_exp[0:128, 0, 0:1], value=0.0)
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 2 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.activation_reduce(dst=sbuf_exp_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.exp, bias=sbuf_neg_max_S[0:128, 0, 0:1],
            reduce_op=nl.add, reduce_res=psum_sum_exp[0:128, 0, 0:1])
nisa.tensor_copy(dst=sbuf_sum_exp[0:128, 0, 0:1], src=psum_sum_exp[0:128, 0, 0:1])
```

**Pattern match.** Map each component of the Standard X + Accumulation pattern (§5.1.1) to attention ops:

| Algorithm 2 Component | Attention Mapping |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $\mathbf{V_0}_k$ (X input) | `sbuf_scaled_S[..., i_block_d2, ..., 0:512]` — tile $k$ of scaled S |
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
sbuf_running_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0, 0:1], value=0.0)

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1+2 """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max """
        psum_tile_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0, 0:1],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.maximum, axis=1)

        psum_max_pair = nl.ndarray((128, 1, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0, 0:1], src=sbuf_running_max[0:128, 0, 0:1])
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0, 1:2], src=psum_tile_max[0:128, 0, 0:1])
        sbuf_new_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sbuf_new_max[0:128, 0, 0:1],
            data=psum_max_pair[0:128, 0, 0:2], op=nl.maximum, axis=1)

        """ Scale: s_k = exp(m_{k-1} - m_k), then rescale running sum """
        sbuf_max_diff = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_max_diff[0:128, 0, 0:1],
            data=sbuf_running_max[0:128, 0, 0:1],
            op0=nl.subtract, operand0=sbuf_new_max[0:128, 0, 0:1])
        sbuf_max_scale = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0, 0:1],
            data=sbuf_max_diff[0:128, 0, 0:1], op=nl.exp)
        nisa.tensor_scalar(dst=psum_running_sum[0:128, 0, 0:1],
            data=psum_running_sum[0:128, 0, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0, 0:1])
        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0, 0:1],
            src=sbuf_new_max[0:128, 0, 0:1])

        """ Bias + accumulate: exp(V - m_k) with rowsum into running sum """
        sbuf_neg_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_max[0:128, 0, 0:1],
            data=sbuf_new_max[0:128, 0, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.activation_reduce(dst=sbuf_exp_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.exp, bias=sbuf_neg_max[0:128, 0, 0:1],
            reduce_op=nl.add, reduce_res=psum_running_sum[0:128, 0, 0:1])
```

#### 5.1.3 Online Fusion: Ops 6, 8, 9

Op 9 (matmul) accumulates `exp_S_t @ V` over d2. Each tile's `exp_S` depends on the running max — the same X output as §5.1.2. This is a second application of online fusion with the same X reduction and the same scale coefficient $s_k = e^{m_{k-1} - m_k}$.

First, prepare two d2 loops. Op 8 (transpose) is elementwise — fold it into Op 9's d2 loop body.

**Before.** Two d2 loops after elementwise prep:

```python
""" Fused Ops 5+6: d2 loop (from §5.1.2) """
sbuf_running_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0, 0:1], value=0.0)
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A """
    for i_tile_d2 in nl.affine_range(1):
        """ ... X step + scale + bias + accumulate from §5.1.2 ... """

""" Fused Ops 8+9: d2 loop """
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
        for i_block_d2 in nl.affine_range(8):                                  """ d2 loop B """
            for i_tile_d2 in nl.affine_range(1):
                sbuf_exp_S_t = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
                for i_f_sub in nl.affine_range(4):                              """ sub-loop: F=512 > MAX_F=128 """
                    nisa.nc_transpose(dst=sbuf_exp_S_t[i_f_sub*128:i_f_sub*128+128, 0, 0, 0, 0, 0:128],
                        src=sbuf_exp_S[0:128, 0, i_block_d2, 0, i_tile_d2, i_f_sub*128:i_f_sub*128+128])
                sbuf_V = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_V[0:512, 0, 0, 0, 0, 0:128],
                    src=V[i_block_d2*512:i_block_d2*512+512, i_block_d5*128:i_block_d5*128+128])
                for i_k_sub in nl.affine_range(4):                              """ sub-loop: K=512 > MAX_K=128 """
                    nisa.nc_matmul(dst=psum_attn[0:128, 0, 0, 0, 0, 0:128],
                        stationary=sbuf_exp_S_t[i_k_sub*128:i_k_sub*128+128, 0, 0, 0, 0, 0:128],
                        moving=sbuf_V[i_k_sub*128:i_k_sub*128+128, 0, 0, 0, 0, 0:128])
```

**Pattern match.** Same Algorithm 2 structure as §5.1.2, with a different accumulation body:

| Algorithm 2 Component | Attention Mapping (Ops 6, 8, 9) |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ | Same as §5.1.2 — $\max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ |
| $g_B(\mathbf{O_0}_K)$ | Same $e^{-m}$ — `exp_S` uses the same max |
| $h_B(\mathbf{V_1}_k)$ | $\text{transpose}(\exp(\mathbf{V_1}_k))^T \cdot \mathbf{V}_k$ — matmul body per tile |
| $\mathbf{B}_k = g_B \cdot h_B$ | $\exp(\mathbf{S}_k - m_K)^T @ \mathbf{V}_k$ — one tile's matmul contribution |
| $s_k$ | Same $e^{m_{k-1} - m_k}$ — shared with the Op 6 accumulator |
| $\tilde{\mathbf{O_1}}_k$ | `psum_attn` — rescaled by $s_k$ then accumulated with $\mathbf{B}_k$ |

Since both online fusions share the same X (running max) and same $g_B(m) = e^{-m}$, they share the same $s_k$. Both `psum_running_sum` and `psum_attn` get rescaled by the same `sbuf_max_scale` each iteration — the X step runs once, then all accumulators rescale together.

**After.** Online fusion merges loops A+B. Both accumulators rescale by the same $s_k$:

```python
""" Ops 5+6+8+9 fused: one d2 loop """
sbuf_running_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0, 0:1], value=-np.inf)
psum_running_sum = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
nisa.memset(dst=psum_running_sum[0:128, 0, 0:1], value=0.0)
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 1, 1, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=psum_attn[0:128, 0, 0, 0, 0, 0:128], value=0.0)

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A+B """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max (same as §5.1.2) """
        psum_tile_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0, 0:1],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.maximum, axis=1)
        psum_max_pair = nl.ndarray((128, 1, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0, 0:1], src=sbuf_running_max[0:128, 0, 0:1])
        nisa.tensor_copy(dst=psum_max_pair[0:128, 0, 1:2], src=psum_tile_max[0:128, 0, 0:1])
        sbuf_new_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sbuf_new_max[0:128, 0, 0:1],
            data=psum_max_pair[0:128, 0, 0:2], op=nl.maximum, axis=1)

        """ Scale: s_k = exp(m_{k-1} - m_k) """
        sbuf_max_diff = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_max_diff[0:128, 0, 0:1],
            data=sbuf_running_max[0:128, 0, 0:1],
            op0=nl.subtract, operand0=sbuf_new_max[0:128, 0, 0:1])
        sbuf_max_scale = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0, 0:1],
            data=sbuf_max_diff[0:128, 0, 0:1], op=nl.exp)

        """ Rescale BOTH accumulators by s_k """
        nisa.tensor_scalar(dst=psum_running_sum[0:128, 0, 0:1],
            data=psum_running_sum[0:128, 0, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0, 0:1])
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                nisa.tensor_scalar(dst=psum_attn[0:128, 0, 0, 0, 0, 0:128],
                    data=psum_attn[0:128, 0, 0, 0, 0, 0:128],
                    op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0, 0:1])

        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0, 0:1],
            src=sbuf_new_max[0:128, 0, 0:1])

        """ Accumulator 1: exp + sum (Op 6 body) """
        sbuf_neg_max = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_max[0:128, 0, 0:1],
            data=sbuf_new_max[0:128, 0, 0:1], op0=nl.multiply, operand0=-1.0)
        sbuf_exp_S = nl.ndarray((128, 1, 1, 1, 1, 512), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sbuf_exp_S[0:128, 0, 0, 0, 0, 0:512],
            data=sbuf_scaled_S[0:128, 0, i_block_d2, 0, i_tile_d2, 0:512],
            op=nl.exp, bias=sbuf_neg_max[0:128, 0, 0:1],
            reduce_op=nl.add, reduce_res=psum_running_sum[0:128, 0, 0:1])

        """ Accumulator 2: transpose + matmul (Ops 8+9 body) """
        sbuf_exp_S_t = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
        for i_f_sub in nl.affine_range(4):                                     """ sub-loop: F=512 > MAX_F=128 """
            nisa.nc_transpose(dst=sbuf_exp_S_t[i_f_sub*128:i_f_sub*128+128, 0, 0, 0, 0, 0:128],
                src=sbuf_exp_S[0:128, 0, 0, 0, 0, i_f_sub*128:i_f_sub*128+128])
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                sbuf_V = nl.ndarray((512, 1, 1, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_V[0:512, 0, 0, 0, 0, 0:128],
                    src=V[i_block_d2*512:i_block_d2*512+512, i_block_d5*128:i_block_d5*128+128])
                for i_k_sub in nl.affine_range(4):                             """ sub-loop: K=512 > MAX_K=128 """
                    nisa.nc_matmul(dst=psum_attn[0:128, 0, 0, 0, 0, 0:128],
                        stationary=sbuf_exp_S_t[i_k_sub*128:i_k_sub*128+128, 0, 0, 0, 0, 0:128],
                        moving=sbuf_V[i_k_sub*128:i_k_sub*128+128, 0, 0, 0, 0, 0:128])
```

`sbuf_exp_S` shrinks to per-tile since it is produced and consumed within the same d2 iteration. Op 7 is topologically independent from Op 9 (§1 DAG). Op 10 depends on Op 9's output but runs after the d2 reduction completes. Neither is part of this transform.

## 6. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel ([`attention_cte.py`](/home/ubuntu/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py)) already uses `nl.ds` for this purpose.

- **Data layout transforms.** The current search space treats data layout as fixed — each tensor's partition and free axes are determined by the math function and never change. A data layout transform pass would manipulate `nc_transpose` ops to improve memory access patterns and engine utilization. The key moves are: (1) **insert dummy transpose pairs** — add a transpose before and after any tensor access point (the pair is a no-op); (2) **cancel adjacent transposes** — two consecutive transposes on the same tensor annihilate; (3) **move transposes** — slide a transpose earlier or later in the graph, past compatible ops, to find a more profitable placement; and (4) **merge transpose with DMA** — when a transpose is adjacent to an HBM load/store, replace the `nc_transpose` + `nisa.dma_copy` pair with a single transposing DMA (`nisa.dma_copy` with transposed source/destination layout), eliminating the Tensor Engine transpose entirely.
