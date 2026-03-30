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

    def __call__(self, data, op):
        reduce = {"max": np.max, "add": np.sum}
        return reduce[op](data, axis=1)

    def render(self, ctx):
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
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

The parser also identifies the function's input parameters (`Q, K, V, scale`) and the return variable (`output`). Intermediate tensor names (`Q_t`, `S`, `masked_S`, etc.) are extracted from the AST assignment targets, preserving the user's naming from the math function. This `list[OpCall]` is the shared input for dimension analysis (§5) and render (§7) — the render uses these names in the generated kernel for readability.

---

## 5. Dimension Analysis

To tile and schedule a math function for hardware, we need to know what dimensions exist, how they relate across ops, and which are parallel vs reduction. This is derived mechanically from `NKIOp.OPERAND_AXES` and `OUTPUT_AXES`.

### 5.1 Dimension Assignment and Unification

Each input parameter gets fresh dimension IDs. Walking the op calls in order, we match each input variable's dimensions to the op's `OPERAND_AXES` labels. When two operands share an axis label, their corresponding dimension IDs are **unified** (merged into one). When a shared axis label maps to dimension IDs that are already assigned and equal, we **assert** consistency — no merge is needed, but we verify the math function is well-formed. If they are already assigned to *different* dimensions, that is a **dimension conflict error**.

Three outcomes when operands share an axis label:
1. **Unify**: one or both dimensions are fresh → merge into one ID
2. **Assert**: both already assigned to the same ID → consistent, no action
3. **Error**: both already assigned to different IDs → math function is malformed

For the running example:

**Initial assignment:**
- `Q → (d0, d1)` — (seq_q, d_k)
- `K → (d2, d3)` — (seq_k, d_k)
- `V → (d4, d5)` — (seq_k, d_v)

**Unification walk:**

| # | Math function call | Dimension assignment |
|---|---|---|
| 0 | `Q_t = nkigym.nc_transpose(Q)` | Q(d0,d1) → Q_t(d1,d0) |
| 1 | `K_t = nkigym.nc_transpose(K)` | K(d2,d3) → K_t(d3,d2) |
| 2 | `S = nkigym.nc_matmul(Q_t, K_t)` | stationary Q_t(d1,d0): K=d1, M=d0; moving K_t(d3,d2): K=d3, N=d2; shared K → **unify** d3→d1; S(d0,d2) |
| 3 | `scaled_S = nkigym.tensor_scalar(S, op0="multiply", operand0=scale)` | scaled_S(d0,d2) |
| 4 | `masked_S = nkigym.affine_select(scaled_S, ...)` | masked_S(d0,d2) |
| 5 | `max_S = nkigym.tensor_reduce(masked_S, op="max")` | max_S(d0) |
| 6 | `shifted_S = nkigym.tensor_scalar(masked_S, max_S, op0="subtract")` | P axis: d0 from data, d0 from operand0 → **assert** d0=d0 ✓; shifted_S(d0,d2) |
| 7 | `exp_S = nkigym.activation(shifted_S, op="exp")` | exp_S(d0,d2) |
| 8 | `sum_exp = nkigym.tensor_reduce(exp_S, op="add")` | sum_exp(d0) |
| 9 | `inv_sum = nkigym.activation(sum_exp, op="reciprocal")` | inv_sum(d0) |
| 10 | `exp_S_t = nkigym.nc_transpose(exp_S)` | exp_S_t(d2,d0) |
| 11 | `attn = nkigym.nc_matmul(exp_S_t, V)` | stationary exp_S_t(d2,d0): K=d2, M=d0; moving V(d4,d5): K=d4, N=d5; shared K → **unify** d4→d2; attn(d0,d5) |
| 12 | `output = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")` | P axis: d0 from data, d0 from operand0 → **assert** d0=d0 ✓; output(d0,d5) |

After the walk, 4 unique dimensions remain: **d0, d1, d2, d5**.

### 5.2 Parallel vs Reduction

An axis label present in `OPERAND_AXES` but absent from `OUTPUT_AXES` is consumed by the op — it's a **reduction axis**. For example, `NKIMatmul` has K in both operands but not in the output; `NKITensorReduce` has F in the input but not the output.

```python
def has_reduction(op: NKIOp) -> bool:
    input_labels = {l for axes in op.OPERAND_AXES.values() for l in axes}
    output_labels = {l for axes in op.OUTPUT_AXES.values() for l in axes}
    return bool(input_labels - output_labels)
```

A dimension's **global classification** depends on whether it appears in the final output:

- **Parallel**: dimension is in the return variable's axes → independent tiles, can execute in any order
- **Reduction**: dimension is not in the return variable → must be accumulated across tiles

```python
def classify_dims(dims, return_var_axes):
    return {d: "parallel" if d in return_var_axes else "reduction" for d in dims}
```

For the running example, `output` has axes `(d0, d5)`:

| Dim | Type | Why |
|---|---|---|
| d0 | parallel | in output(d0, d5) |
| d1 | reduction | consumed by matmul K axis |
| d2 | reduction | consumed by matmul K axis and tensor_reduce F axis |
| d5 | parallel | in output(d0, d5) |

### 5.3 Tile Sizes

Each dimension's tile size is the **maximum** across all `MAX_TILE_SIZES` entries, capped at the actual dimension size:

1. Collect all `MAX_TILE_SIZES` entries for a dimension across all ops where it appears.
2. `tile_size = min(max(all limits for this dim), dimension input size)`.

Ops whose limit for a dimension is smaller than the tile size emit **in-place sub-loops**: `sub_iters = tile_size / op_limit` iterations per tile. For example, d2 has tile_size 512 (from `nc_matmul` N:512). Ops with d2 limits of 128 emit `512 / 128 = 4` sub-iterations:
- **transpose K** (`nc_transpose` P:128, F:128): 4 sub-transposes of 128-element chunks, grouped to feed one `nc_matmul` call
- **transpose exp_S** + **nc_matmul₂** (K:128): 4 sub-transposes of exp_S feed 4 sub-matmul accumulations into the same PSUM output

| Dim | Limits collected from ops | tile_size |
|---|---|---|
| d0 | P:128 (transpose, tensor_scalar, affine_select, tensor_reduce, activation), M:128 (nc_matmul), F:128 (transpose) | 128 |
| d1 | F:128 (transpose Q, transpose K), K:128 (nc_matmul) | 128 |
| d2 | P:128 (transpose K), N:512 (nc_matmul₁), F:128 (transpose exp_S), K:128 (nc_matmul₂) | 512 |
| d5 | N:512 (nc_matmul₂) | 512 |

### 5.4 Analysis Result

The dimension analysis produces an `Analysis` — the complete tiling and data-flow metadata consumed by the schedule enumerator (§6) and renderer (§7):

```python
class Analysis(NamedTuple):
    dims: dict[str, Dim]
    var_axes: dict[str, tuple[str, ...]]
    return_var: str

class Dim(NamedTuple):
    dim_id: str
    classification: str  # "parallel" or "reduction"
    tile_size: int
    input_size: int
```

- **`dims`**: all unique dimensions after unification, each with its classification (§5.2) and tile size (§5.3).
- **`var_axes`**: maps every variable name (inputs, intermediates, output) to its ordered tuple of dimension IDs. E.g., `{"Q": ("d0", "d1"), "S": ("d0", "d2"), "output": ("d0", "d5"), ...}`.
- **`return_var`**: the name of the function's return variable (e.g., `"output"`).

For the running example:

| Field | Value |
|---|---|
| `dims` | `{"d0": Dim("d0", "parallel", 128, 512), "d1": Dim("d1", "reduction", 128, 128), "d2": Dim("d2", "reduction", 512, 512), "d5": Dim("d5", "parallel", 512, 1024)}` |
| `var_axes` | `{"Q": ("d0","d1"), "K": ("d2","d1"), "V": ("d2","d5"), "Q_t": ("d1","d0"), "K_t": ("d1","d2"), "S": ("d0","d2"), ...}` |
| `return_var` | `"output"` |

## 6. Schedule

A `Schedule` is an immutable, hashable descriptor that captures *how* to execute the math function — independently of *what* to compute (which comes from §1–5).

```python
class Loop(NamedTuple):
    dim_id: str
    tiles_per_block: int

class Schedule(NamedTuple):
    loop_order: tuple[tuple[Loop, tuple[int, int]], ...]  # ((Loop, (level, order)), ...)
    load_placements: tuple[tuple[str, int], ...]  # ((tensor_name, relative_pos), ...)
```

### 6.1 Loop Order

`loop_order` is a tuple of `(Loop, (level, order))` pairs. The two integers fully determine nesting and execution order:

- **`level`**: nesting level. Different levels mean nested loops (outer vs inner).
- **`order`**: execution order within that level. Same level, different order means sequential (back-to-back) loops.

Each `Loop` represents a **two-level loop** — a block loop and a tile loop:

```
for block in range(num_blocks):                 # block loop: num_blocks = total_tiles / tiles_per_block
    for tile in range(tiles_per_block):          # tile loop: iterate within the block
```

Parallel dims each increase the level. Reduction dims that require multiple sequential sweeps (e.g., running max, running sum, matmul accumulation) share the same level but have distinct order indices — they run back-to-back, not nested.

`tiles_per_block` controls blocking — how many tiles are grouped into one block. Must divide the total tile count. Larger `tiles_per_block` means larger SBUF buffers (holding more tiles at once) but fewer DMA round-trips — the loaded block is reused across all tile iterations within it.

**Example loops** for the running example:

| Loop (dim_id, tpb) | (level, order) | Meaning |
|---|---|---|
| (d0, 1) | (0, 0) | Outer parallel loop: seq_q |
| (d5, 1) | (1, 0) | Parallel loop: d_v |
| (d1, 1) | (2, 0) | d_k reduction (matmul1 K axis) |
| (d2, 1) | (2, 1) | seq_k reduction (running max) |
| (d2, 1) | (2, 2) | seq_k reduction (running sum) — needs completed max |
| (d2, 1) | (2, 3) | seq_k reduction (matmul2 K axis) — needs completed inv_sum |

d2 appears 3 times because the math function has 3 barrier ops that fully reduce over d2: `tensor_reduce(max)`, `tensor_reduce(sum)`, and `nc_matmul` (matmul2). Each barrier requires a complete sweep — the next cannot start until the previous reduction is finished across all d2 tiles.

### 6.2 Load Placements

`load_placements` maps each input tensor name to an integer `relative_pos` from 0 to `num_dependent_dims`. The `relative_pos` controls how deep into the loop nest the DMA load is placed, relative to the tensor's dependent dims. Load placement is the **only** allocation with a real search space — hoisting a load outward amortizes DMA latency across more tiles at the cost of larger SBUF buffers. All other on-chip allocations are either **deterministic** (accumulation targets like matmul/reduce outputs must be outside the loop they accumulate across, dictated by the math data flow) or **trivially minimum-scope** (SSA intermediates and cross-pass buffers have no DMA cost to amortize, so they always go at the innermost possible scope to conserve SBUF/PSUM capacity).

`num_dependent_dims` is the number of unique loop dimensions that appear in the tensor's axes (after unification), ordered by their position in `loop_order`. Each `relative_pos` moves the load point one dependent loop deeper. Since each Loop is a block loop + tile loop, "outside" means before the block loop, and "inside" means between the block and tile loops:

- **relative_pos 0**: outside all dependent loops (before their block loops) → `NUM_TILES = total_tiles` for all dependent axes
- **relative_pos k**: inside the first k dependent loops (between their block and tile loops) → `NUM_TILES = tiles_per_block` for those k axes, `total_tiles` for the rest
- **relative_pos N**: inside all dependent loops → `NUM_TILES = tiles_per_block` for all dependent axes

To hold just 1 tile for a dimension, set `tiles_per_block=1` on the Loop and place the tensor inside — no separate case needed.

A tensor with N dependent dims has N+1 valid `relative_pos` values.

For the running example:

| Tensor | Axes (after unification) | Dependent dims (loop order) | Valid relative_pos |
|---|---|---|---|
| Q | (d0, d1) | d0, d1 | 0, 1, 2 |
| K | (d2, d1) | d1, d2 | 0, 1, 2 |
| V | (d2, d5) | d5, d2 | 0, 1, 2 |

### 6.3 Enumeration

The schedule space is the cross-product of three independent axes:

**Axis 1 — Loop order**: permutations of the $L$ loop entries. If a dimension appears $r$ times (multiple reduction sweeps), those $r$ entries must maintain their relative order. The constraint reduces the space from $L!$ to $L! / \prod r_i!$ where $r_i$ is the repeat count of each dimension.

**Axis 2 — Blocking**: for each unique dimension, `tiles_per_block` can be any divisor of `total_tiles` (= dim_size / tile_size). The number of choices per dimension is the divisor count of its total tiles.

**Axis 3 — Load placements**: for each input tensor with $N$ dependent dims, `relative_pos` ranges from 0 to $N$, giving $N+1$ choices.

$$\text{Full space} = \text{loop orders} \times \prod_{\text{dims}} |\text{divisors}(total\_tiles_i)| \times \prod_{\text{tensors}} (N_j + 1)$$

Hardware validation prunes infeasible schedules. In the `Tensor` layout, `TILE_SIZES[AXES[0]]` is the partition dimension (≤ 128 by hardware), while `NUM_TILES` for all axes — including tiles_per_block and num_blocks — expand into the free dimension. Larger blocking grows the free dim, which must stay within SBUF capacity (24 MB) and PSUM accumulator limits. Every survivor is rendered, compiled, and benchmarked.

**Running example** with typical Llama-style single-head attention shapes: `q[4096, 128], k[4096, 128], v[4096, 128]`:

| Dim | Size | tile_size | total_tiles | Valid tpb (divisors) | Count |
|---|---|---|---|---|---|
| d0 (seq_q) | 4096 | 128 | 32 | {1, 2, 4, 8, 16, 32} | 6 |
| d1 (d_k) | 128 | 128 | 1 | {1} | 1 |
| d2 (seq_k) | 4096 | 512 | 8 | {1, 2, 4, 8} | 4 |
| d5 (d_v) | 128 | 128 | 1 | {1} | 1 |

- **Loop orders**: $6! / 3! = 120$ (4 unique dims after unification: d0, d1, d2, d5; d2 appears 3× for its 3 barrier ops → 6 entries, d2s maintain relative order)
- **Blocking**: $6 \times 1 \times 4 \times 1 = 24$
- **Load placements**: $3 \times 3 \times 3 = 27$ (Q, K, V each have 2 dependent dims → 3 valid `relative_pos` values)

$$\text{Full space} = 120 \times 24 \times 27 = 77{,}760 \text{ candidates}$$

## 7. Render (Specialization)

Rendering is a **specialization** step: it takes the math function (§1), the op call list (§4), dimension analysis (§5), a schedule (§6), and **concrete input shapes and dtypes** to produce an NKI kernel tailored to those specific inputs. Each unique combination of input shapes and dtypes requires its own tuned kernel. Infrastructure ops (DMA loads, PSUM→SBUF staging, DMA stores) are generated by the renderer — they do not appear in the math function.

### 7.1 Render Pipeline

```
render(op_calls: list[OpCall], analysis: Analysis, schedule: Schedule, input_shapes, dtypes) → NKI kernel source
```

- **`op_calls`**: ordered list of op calls from the parser (§4)
- **`analysis`**: dimension analysis result — dims, var_axes, return_var (§5.4)
- **`schedule`**: loop order and load placements (§6)
- **`input_shapes`**: concrete sizes for each input parameter
- **`dtypes`**: data type for each input parameter

1. **Emit preamble**: imports (`nki`, `nki.language as nl`, `nki.isa as nisa`), `@nki.jit` decorator, function signature with input params, shape and dtype assertions for each input, HBM output allocation.

**Tensor naming**: input/output tensors use the same names as the math function (`q`, `k`, `v`, `output`). Intermediate tensors are named `{location}_{var_name}` where `location` is `sbuf` or `psum` and `var_name` is the variable name from the parsed math function (§4). Examples: `sbuf_Q_t`, `psum_S`, `sbuf_masked_S`.
2. **Emit loop nest**: walk `schedule.loop_order` — each Loop emits a block loop (`nl.affine_range(num_blocks)`) and a tile loop (`nl.affine_range(tiles_per_block)`). `level` determines nesting; same-level loops run sequentially by `order`.
3. **Emit DMA loads**: for each input tensor, place the `nisa.dma_copy(HBM→SBUF)` at the `relative_pos` specified by `schedule.load_placements`. Allocate an SBUF buffer with shape determined by `Tensor.NUM_TILES` (§6.2).
4. **Emit PSUM accumulators**: before each reduction loop, emit `nl.ndarray(..., buffer=nl.psum)` followed by `nisa.memset(buf, value=...)`. Always this two-step sequence — never `nl.zeros`.
   - **`nc_matmul`**: `nl.ndarray` + `nisa.memset(buf, value=0.0)`. Hardware accumulates — each matmul call adds its partial product to the same PSUM address, so a single buffer suffices.
   - **`tensor_reduce`**: when the reduction axis spans $T$ tiles in the loop, `nl.ndarray` a PSUM partial-results buffer with $T$ columns (one per tile), then `nisa.memset` with identity value (`0.0` for `add`, `-3.4e38` for `max`). Inside the loop, each tile's reduction result writes to its own column. After the loop, a second `nisa.tensor_reduce` collapses all columns into the final scalar result. This gather-then-reduce pattern is necessary because `tensor_reduce` overwrites its destination — unlike `nc_matmul`, it does not accumulate.
5. **Emit compute ops**: each op is emitted as soon as all its operands are available — not deferred to the innermost level. For each op, build a `RenderContext` (§2) and call `NKIOp.render(ctx)`. Vector/Scalar/GpSimd ops (tensor_scalar, activation, affine_select) operate on SBUF; the renderer stages PSUM→SBUF before these ops and SBUF→PSUM after if needed.
6. **Emit staging**: when an op needs an operand in SBUF but the data is currently in PSUM, emit `nisa.tensor_copy(PSUM→SBUF)` to stage it. This is triggered on demand — not at a fixed point in the pipeline.
7. **Emit DMA store**: `nisa.dma_copy(SBUF→HBM)` writes the final output tile back.

### 7.2 RenderContext Construction

For each op call site, the renderer builds a `RenderContext`:

- `outputs`: output `Tensor`(s) with axes, tile sizes, location (SBUF or PSUM), and the rendered buffer name (§7.1 step 1)
- `operands`: input `Tensor`(s) with their current location (SBUF or PSUM) and slice expressions for the current tile
- `config_kwargs`: pass-through from the math function call (e.g., `op0=nl.multiply`, `cmp_op=nl.greater_equal`)
- `tile_idx`: maps each dim_id to the current loop variable (e.g., `{"d0": "i_0", "d2": "i_2"}`)
- `tile_start`: maps each dim_id to the element offset expression (e.g., `{"d0": "i_0 * 128", "d2": "i_2 * 512"}`)

Each `NKIOp.render(ctx)` returns one or more NKI source lines using `nisa.*` calls.

### 7.3 Example Output

Generic attention with symbolic shapes `q[S_q, D_k]`, `k[S_k, D_k]`, `v[S_k, D_v]` and all loads at `relative_pos=max`. Tile sizes from §5.3: $t_0 = 128$ (d0), $t_1 = 128$ (d1), $t_2 = 512$ (d2), $t_5 = \min(512, D_v)$ (d5). Block counts: $\text{num\_blocks}_i = S_i / (t_i \cdot \text{tiles\_per\_block}_i)$.

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

    d0_block_size = d0_tiles_per_block * t_0
    d1_block_size = d1_tiles_per_block * t_1
    d2_block_size = d2_tiles_per_block * t_2
    d5_block_size = d5_tiles_per_block * t_5

    output_hbm = nl.ndarray((S_q, D_v), dtype=q.dtype, buffer=nl.shared_hbm)

    for i_d0_blk in nl.affine_range(d0_num_blocks):
      d0_blk_off = i_d0_blk * d0_block_size
      for i_d0_tile in nl.affine_range(d0_tiles_per_block):
        d0_off = d0_blk_off + i_d0_tile * t_0
        for i_d5_blk in nl.affine_range(d5_num_blocks):
          d5_blk_off = i_d5_blk * d5_block_size
          for i_d5_tile in nl.affine_range(d5_tiles_per_block):
            d5_off = d5_blk_off + i_d5_tile * t_5

            # psum_S — axes=(d0, d2): d0 after tile, d2 before blk
            psum_S = nl.ndarray(
                (t_0, 1, d2_num_blocks, 1, d2_tiles_per_block, t_2),
                dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_S, value=0.0)

            for i_d1_blk in nl.affine_range(d1_num_blocks):
              d1_blk_off = i_d1_blk * d1_block_size

              # sbuf_Q — axes=(d0, d1): d0 after tile, d1 between blk/tile
              sbuf_Q = nl.ndarray(
                  (t_0, 1, 1, 1, d1_tiles_per_block, t_1),
                  dtype=q.dtype, buffer=nl.sbuf)
              nisa.dma_copy(
                  dst=sbuf_Q[0:t_0, 0, 0, 0, 0:d1_tiles_per_block, 0:t_1],
                  src=q[d0_off:d0_off+t_0,
                        d1_blk_off:d1_blk_off+d1_block_size])

              for i_d1_tile in nl.affine_range(d1_tiles_per_block):
                d1_off = d1_blk_off + i_d1_tile * t_1

                # psum_Q_t — axes=(d1, d0): d1 after tile, d0 after tile
                psum_Q_t = nl.ndarray(
                    (t_1, 1, 1, 1, 1, t_0),
                    dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=psum_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
                    data=sbuf_Q[0:t_0, 0, 0, 0, i_d1_tile, 0:t_1])

                # sbuf_Q_t — copy transpose to SBUF for matmul input
                sbuf_Q_t = nl.ndarray(
                    (t_1, 1, 1, 1, 1, t_0),
                    dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=sbuf_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
                    src=psum_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0])

                for i_d2_blk in nl.affine_range(d2_num_blocks):
                  for i_d2_tile in nl.affine_range(d2_tiles_per_block):
                    d2_off = (i_d2_blk * d2_tiles_per_block + i_d2_tile) * t_2

                    """ nc_transpose sub-loop: d2 partition exceeds P:128.
                    Each sub-iteration transposes a (128, t_1) chunk of K
                    and feeds it to nc_matmul at the corresponding output slice.
                    Emitted by NKITranspose.render() and NKIMatmul.render(). """
                    for i_d2_sub in nl.affine_range(t_2 // 128):

                      # sbuf_K_sub — load one (128, t_1) chunk of K
                      sbuf_K_sub = nl.ndarray(
                          (128, 1, 1, 1, 1, t_1),
                          dtype=k.dtype, buffer=nl.sbuf)
                      nisa.dma_copy(
                          dst=sbuf_K_sub[0:128, 0, 0, 0, 0, 0:t_1],
                          src=k[d2_off + i_d2_sub * 128
                                : d2_off + (i_d2_sub + 1) * 128,
                                d1_off:d1_off+t_1])

                      # transpose K chunk: (128, t_1) → (t_1, 128)
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

                      # Q_t @ K_t_sub -> S slice: contract d1, output (d0, d2)
                      nisa.nc_matmul(
                          dst=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile,
                                     i_d2_sub * 128 : (i_d2_sub + 1) * 128],
                          stationary=sbuf_Q_t[0:t_1, 0, 0, 0, 0, 0:t_0],
                          moving=sbuf_K_t_sub[0:t_1, 0, 0, 0, 0, 0:128])

            # psum_partial_max — one column per d2 tile for gather-then-reduce
            d2_total_tiles = d2_num_blocks * d2_tiles_per_block
            psum_partial_max = nl.ndarray(
                (t_0, 1, d2_total_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_partial_max, value=-3.4028235e38)

            for i_d2_blk in nl.affine_range(d2_num_blocks):
              for i_d2_tile in nl.affine_range(d2_tiles_per_block):
                d2_col = i_d2_blk * d2_tiles_per_block + i_d2_tile
                d2_off = d2_col * t_2
                sbuf_S = nl.ndarray(
                    (t_0, 1, 1, 1, 1, t_2),
                    dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=sbuf_S[0:t_0, 0, 0, 0, 0, 0:t_2],
                    src=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile, 0:t_2])
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
                    dst=psum_partial_max[0:t_0, 0, d2_col],
                    op=nl.maximum,
                    data=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
                    axis=(1,))
                nisa.tensor_copy(
                    dst=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile, 0:t_2],
                    src=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2])

            # collapse all partial maxes into a single column
            psum_max_S = nl.ndarray(
                (t_0, 1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.tensor_reduce(
                psum_max_S[0:t_0, 0, 0:1],
                nl.maximum, psum_partial_max[0:t_0, 0, 0:d2_total_tiles], 1)
            sbuf_max_S = nl.ndarray(
                (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=sbuf_max_S[0:t_0, 0, 0:1],
                src=psum_max_S[0:t_0, 0, 0:1])

            # psum_partial_sum — one column per d2 tile for gather-then-reduce
            psum_partial_sum = nl.ndarray(
                (t_0, 1, d2_total_tiles), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_partial_sum, value=0.0)

            for i_d2_blk in nl.affine_range(d2_num_blocks):
              for i_d2_tile in nl.affine_range(d2_tiles_per_block):
                d2_col = i_d2_blk * d2_tiles_per_block + i_d2_tile
                sbuf_masked_S = nl.ndarray(
                    (t_0, 1, 1, 1, 1, t_2),
                    dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=sbuf_masked_S[0:t_0, 0, 0, 0, 0, 0:t_2],
                    src=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile, 0:t_2])
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
                    dst=psum_partial_sum[0:t_0, 0, d2_col],
                    op=nl.add,
                    data=sbuf_exp_S[0:t_0, 0, 0, 0, 0, 0:t_2],
                    axis=(1,))
                nisa.tensor_copy(
                    dst=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile, 0:t_2],
                    src=sbuf_exp_S[0:t_0, 0, 0, 0, 0, 0:t_2])

            # collapse all partial sums into a single column
            psum_sum_exp = nl.ndarray(
                (t_0, 1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.tensor_reduce(
                psum_sum_exp[0:t_0, 0, 0:1],
                nl.add, psum_partial_sum[0:t_0, 0, 0:d2_total_tiles], 1)

            # sbuf_sum_exp — copy sum to SBUF for reciprocal
            sbuf_sum_exp = nl.ndarray(
                (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=sbuf_sum_exp[0:t_0, 0, 0:1],
                src=psum_sum_exp[0:t_0, 0, 0:1])

            # sbuf_inv_sum — axes=(d0,): d0 after tile
            sbuf_inv_sum = nl.ndarray(
                (t_0, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=sbuf_inv_sum[0:t_0, 0, 0:1],
                op=nl.reciprocal,
                data=sbuf_sum_exp[0:t_0, 0, 0:1])

            # psum_attn — axes=(d0, d5): d0 after tile, d5 after tile
            psum_attn = nl.ndarray(
                (t_0, 1, 1, 1, 1, t_5),
                dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_attn, value=0.0)

            for i_d2_blk in nl.affine_range(d2_num_blocks):
              for i_d2_tile in nl.affine_range(d2_tiles_per_block):
                d2_off = (i_d2_blk * d2_tiles_per_block + i_d2_tile) * t_2

                """ nc_transpose sub-loop: d2 exceeds P:128 and F:128.
                Each sub-iteration transposes a (t_0, 128) chunk of exp_S,
                loads the corresponding (128, t_5) chunk of V,
                and accumulates into psum_attn.
                Emitted by NKITranspose.render() and NKIMatmul.render(). """
                for i_d2_sub in nl.affine_range(t_2 // 128):

                  # copy exp_S sub-tile from psum_S to SBUF
                  sbuf_exp_S_sub = nl.ndarray(
                      (t_0, 1, 1, 1, 1, 128),
                      dtype=nl.float32, buffer=nl.sbuf)
                  nisa.tensor_copy(
                      dst=sbuf_exp_S_sub[0:t_0, 0, 0, 0, 0, 0:128],
                      src=psum_S[0:t_0, 0, i_d2_blk, 0, i_d2_tile,
                                 i_d2_sub * 128 : (i_d2_sub + 1) * 128])

                  # transpose: (d0=t_0, d2_sub=128) → (d2_sub=128, d0=t_0)
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

                  # load V sub-tile: (128, t_5)
                  sbuf_V_sub = nl.ndarray(
                      (128, 1, 1, 1, 1, t_5),
                      dtype=v.dtype, buffer=nl.sbuf)
                  nisa.dma_copy(
                      dst=sbuf_V_sub[0:128, 0, 0, 0, 0, 0:t_5],
                      src=v[d2_off + i_d2_sub * 128
                            : d2_off + (i_d2_sub + 1) * 128,
                            d5_off:d5_off+t_5])

                  # exp_S_t_sub @ V_sub -> attn: accumulates over d2 chunks
                  nisa.nc_matmul(
                      dst=psum_attn[0:t_0, 0, 0, 0, 0, 0:t_5],
                      stationary=sbuf_exp_S_t_sub[0:128, 0, 0, 0, 0, 0:t_0],
                      moving=sbuf_V_sub[0:128, 0, 0, 0, 0, 0:t_5])

            # sbuf_attn — axes=(d0, d5): d0 after tile, d5 after tile
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

Each tensor's shape comment shows its axes, the symbolic shape `(tile_par, nb_0, nb_1, nt_0, nt_1, tile_free)`, and which placement rule determined each nb/nt value:
- **after tile**: nb=1, nt=1 (inside both block and tile loops)
- **between blk/tile**: nb=1, nt=tiles_per_block (inside block loop, before tile loop)
- **before blk**: nb=num_blocks, nt=tiles_per_block (outside all loops for that dim)

---

## 8. Related Work

NKIGym sits at the intersection of auto-scheduling, tile-level kernel generation, and hardware-specific compilation. This section surveys the landscape and identifies where NKIGym's design diverges from prior systems.

### 8.1 Auto-Scheduling Frameworks

**TVM ecosystem.** AutoTVM (Chen et al., NeurIPS 2018) searches over schedule knobs within expert-written templates using a learned XGBoost cost model. Ansor (Zheng et al., OSDI 2020) eliminates templates by auto-generating sketches from the computation DAG and navigating the space with evolutionary search + learned cost models. TensorIR/MetaSchedule (Feng et al., ASPLOS 2023) generalizes Ansor with first-class tensorized instruction support. All operate on scalar loop nests derived from tensor expressions, with search spaces in the millions requiring learned cost models.

**Halide.** Separates algorithm from schedule (Ragan-Kelley et al., 2013). Adams et al. (2019) introduced beam search with a neural cost model trained on randomly generated programs. Established the algorithm/schedule split that NKIGym inherits, but targets image processing pipelines on CPUs/GPUs at the loop level.

**Key difference from NKIGym:** These systems reason about scalar loop nests and target mainstream hardware (GPUs, CPUs). NKIGym reasons about ISA-level tile operations on a fixed accelerator memory hierarchy, yielding a search space small enough (tens of thousands) for exhaustive evaluation without learned cost models.

### 8.2 Tile-Level Kernel Generators

**Triton** (Tillet et al., 2019; OpenAI) provides a Python DSL for tile-level GPU programming. Users write tile operations (`tl.load`, `tl.dot`, `tl.exp`); the compiler lowers to PTX. `triton.autotune` benchmarks user-enumerated configurations — no automatic space generation.

**ThunderKittens** (Spector et al., 2024) is a C++ template library with warp-level tile abstractions for H100 GPUs. Achieves FlashAttention-level performance through composable primitives, but requires fully manual scheduling.

**CUTLASS/CuTe** (NVIDIA) provides composable GEMM building blocks with CuTe's layout algebra for rigorous memory layout reasoning. Pre-defined configurations; no automatic search.

**Key difference from NKIGym:** These systems provide tile abstractions but no automatic schedule search or dimension analysis. NKIGym automates both — users write only the math function; tiling structure, loop ordering, and data movement are derived and searched automatically.

### 8.3 Polyhedral and Constraint-Based Compilation

**Pluto/PPCG** (Bondhugula et al., 2008; Verdoolaege et al., 2013) use ILP to find optimal affine schedules for static loop nests. **Tiramisu** (Baghdadi et al., CGO 2019) layers a 4-level IR over polyhedral analysis with manual or learned scheduling.

**AKG** (Zhao et al., PLDI 2024) is the closest polyhedral system to NKIGym: it generates kernels for Huawei Ascend NPUs by composing ISA-level primitives with polyhedral scheduling and ILP-based optimization. It targets a custom DL accelerator ISA, but uses fundamentally different machinery — polyhedral dependence analysis over scalar iteration domains vs. NKIGym's op-level dimension unification.

### 8.4 Exhaustive and Superoptimization Approaches

**Roller** (Zhu et al., OSDI 2022) constrains tile shapes to memory-aligned "rTiles," enabling near-exhaustive constructive search that completes in seconds. Operates at the loop/tile level for GPUs.

**Mirage** (Wu & Jia et al., OSDI 2025) is the closest system in philosophy: it exhaustively enumerates a structured 3-level program space (kernel graph, thread-block graph, shared-memory tensor graph) with algebraic equivalence pruning. Discovered novel GPU kernels competitive with FlashAttention without user scheduling. Key distinction: Mirage searches over *program structure* (what ops to compose), while NKIGym searches over *schedule structure* (how to execute a fixed op sequence).

**Equality saturation** (Yang et al., MLSys 2021) uses e-graphs to explore exponentially many equivalent tensor program rewrites efficiently. Operates at the graph rewrite level, orthogonal to NKIGym's tile-level scheduling.

### 8.5 Hardware-Specific Accelerator DSLs

**Exo** (Ikarashi et al., PLDI 2022) delegates hardware-specific scheduling to user-written rewrite rules, enabling custom accelerator support without modifying the compiler. Philosophically aligned with NKI's explicit-control model but requires manual scheduling.

No published autotuning framework targets NKI or AWS Trainium. Other accelerator DSLs — Cerebras CSL (PE-level C), Graphcore Poplar (tile/vertex C++), SambaNova SambaFlow (high-level PyTorch) — similarly lack automatic schedule search.

### 8.6 Graph-Level Optimizers

XLA, TorchInductor, and TVM Relay operate at the operator fusion level — choosing which ops to fuse and how to lay out data across ops. These are orthogonal to NKIGym, which optimizes *within* a fused kernel.

### 8.7 Where NKIGym Diverges

The table below positions NKIGym against the closest systems on two axes: abstraction level and search strategy.

| System | Abstraction | Search | Target |
|---|---|---|---|
| Ansor | Scalar loop nests | Learned cost model + evolutionary | GPU, CPU |
| Triton | Tile-level DSL | User-enumerated grid | GPU |
| AKG | ISA-level (polyhedral) | ILP + heuristic | Huawei Ascend |
| Mirage | Multi-level program IR | Exhaustive + verification | NVIDIA GPU |
| Roller | Loop/tile (rTile) | Constructive/analytical | GPU, CPU |
| **NKIGym** | **ISA-level (op-level IR)** | **Exhaustive enumeration** | **AWS Trainium** |

Five design choices distinguish NKIGym from the landscape:

1. **Op-level IR with 1:1 ISA mapping.** Each `NKIOp` maps to exactly one `nisa.*` instruction. This is lower than Triton (many-to-many mapping from tile ops to PTX) and more structured than raw NKI (no abstraction over ISA calls). The IR sits between "compiler lowers for you" and "you write assembly."

2. **Unification-based dimension analysis.** NKIGym derives tiling structure by propagating dimension identities through `OPERAND_AXES`/`OUTPUT_AXES` declarations across the op graph (§5.1). This replaces both polyhedral dependence analysis (which reasons about scalar index expressions) and TVM's tensor expression analysis (which analyzes lambda bodies). The formulation is simpler and sufficient because the ISA ops have fixed, declared axis semantics.

3. **Barrier-driven multi-pass reductions.** A reduction dimension can appear multiple times in the loop order when sequential barrier ops each require a complete sweep (§6.1: d2 appears 3× for max, sum, and matmul2 in attention). The `(level, order)` addressing scheme with same-level sequential loops captures this naturally. Prior auto-schedulers model reductions as single loops.

4. **Load placement as the sole non-deterministic allocation.** NKIGym identifies that DMA load hoisting is the only allocation with a real search space — accumulation targets are deterministic (dictated by data flow) and intermediates are trivially minimum-scope (§6.2). This decomposition, specific to the SBUF/PSUM/HBM hierarchy, eliminates an entire class of scheduling decisions from the search.

5. **Tractable exhaustive search.** The structured decomposition into three independent axes — loop order ($L!/\prod r_i!$ permutations), blocking (divisor counts), and load placement ($N+1$ choices per tensor) — keeps the space at tens of thousands of candidates. This enables brute-force evaluation without learned cost models, evolutionary search, or sampling heuristics. The tradeoff is hardware-specificity: the decomposition assumes Trainium's fixed memory hierarchy.

---

## 9. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel (`attention_cte.py`) already uses `nl.ds` for this purpose.
