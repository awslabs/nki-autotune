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

The parser also identifies the function's input parameters (`Q, K, V, scale`) and the return variable (`output`). Intermediate tensor names (`Q_t`, `S`, `masked_S`, etc.) are extracted from the AST assignment targets, preserving the user's naming from the math function. This `list[OpCall]` is the shared input for dimension analysis (§5) and render (§7) — the render uses these names in the generated kernel for readability.

---

## 5. Dimension Analysis

To tile and schedule a math function for hardware, we need to know what dimensions exist, how they relate across ops, and which are independent vs dependent. This is derived mechanically from `NKIOp.OPERAND_AXES` and `OUTPUT_AXES`.

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

### 5.2 Dimension Dependencies

Dimensions form **dependency groups** based on data flow. An axis label present in `OPERAND_AXES` but absent from `OUTPUT_AXES` is consumed by that op — all tiles along that dimension must be processed before the op's output is complete. Dimensions that share this property (consumed by ops in the same data-flow chain) form a dependency group.

```python
def is_consumed_by(op: NKIOp) -> set[str]:
    input_labels = {l for axes in op.OPERAND_AXES.values() for l in axes}
    output_labels = {l for axes in op.OUTPUT_AXES.values() for l in axes}
    return input_labels - output_labels
```

A dimension's grouping depends on whether it appears in the final output's axes:

- **Independent**: dimension is in the return variable's axes → forms a dependency group by itself. Tiles can execute in any order.
- **Dependent**: dimension is not in the return variable's axes → shares a dependency group with other non-output dimensions. All tiles across the group must be swept before the output tile is ready.

```python
def dependency_groups(dims, return_var_axes):
    independent = [(d,) for d in dims if d in return_var_axes]
    dependent = tuple(d for d in dims if d not in return_var_axes)
    groups = independent + ([dependent] if dependent else [])
    return tuple(groups)
```

For the running example, `output` has axes `(d0, d5)`:

| Dim | Group | Why |
|---|---|---|
| d0 | {d0} — independent | in output(d0, d5) |
| d1 | {d1, d2} — dependent | consumed by matmul K axis |
| d2 | {d1, d2} — dependent | consumed by matmul K axis and tensor_reduce F axis |
| d5 | {d5} — independent | in output(d0, d5) |

Independent dimensions (d0, d5) each form a group of size 1 — their tiles are self-contained. Dependent dimensions (d1, d2) form a shared group — all tiles across both must be swept before the output tile is ready. As a group, {d1, d2} is independent from {d0} and {d5}.

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

The dimension analysis produces an `Analysis` — the complete tiling and data-flow metadata consumed by the loop nest enumerator (§6) and renderer (§7):

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

- **`dims`**: all unique dimensions after unification, each with its tile size (§5.3).
- **`dependency_groups`**: dimension groupings from §5.2. Each tuple is a group of dimension IDs that are data-dependent on each other. Independent dimensions appear as singleton tuples.
- **`var_axes`**: maps every variable name (inputs, intermediates, output) to its ordered tuple of dimension IDs. E.g., `{"Q": ("d0", "d1"), "S": ("d0", "d2"), "output": ("d0", "d5"), ...}`.
- **`return_var`**: the name of the function's return variable (e.g., `"output"`).

For the running example:

| Field | Value |
|---|---|
| `dims` | `{"d0": Dim("d0", 128, 512), "d1": Dim("d1", 128, 128), "d2": Dim("d2", 512, 512), "d5": Dim("d5", 512, 1024)}` |
| `dependency_groups` | `(("d0",), ("d5",), ("d1", "d2"))` |
| `var_axes` | `{"Q": ("d0","d1"), "K": ("d2","d1"), "V": ("d2","d5"), "Q_t": ("d1","d0"), "K_t": ("d1","d2"), "S": ("d0","d2"), ...}` |
| `return_var` | `"output"` |

## 6. Kernel Spec

A `Schedule` is an immutable, hashable descriptor that captures *how* to execute the math function — independently of *what* to compute (which comes from §1–5).

```python
class Loop(NamedTuple):
    dim_id: str            # global dimension ID from analysis (e.g. "d0")
    tile_size: int         # hardware tile size from §5.3
    tiles_per_block: int   # block size — §6.3

class Schedule(NamedTuple):
    loop_order: tuple[tuple[Loop, tuple[int, int]], ...]  # (Loop, (level, order)) — §6.2
    load_placements: tuple[tuple[str, int], ...]          # (tensor_name, relative_pos) — §6.4
```

Each `loop_order` entry pairs a `Loop` with a `(level, order)` tuple. `level` is the nesting depth — different levels nest. `order` is sequential execution order within the same level — same level, different order means back-to-back loops (§6.2).

### 6.1 Single Dimension Loop

Each dimension produces a double loop — a **block loop** and a **tile loop**:

```
for block_iter in range(total_tiles // tiles_per_block):   # block loop
    for tile_iter in range(tiles_per_block):                # tile loop
        [ops execute here, one tile at a time]
```

Hardware computes on a single tile at a time — this is fixed. The double loop structure creates a slot between the block and tile loops where DMA loads (§6.4) can be placed. `tiles_per_block` controls the block size (§6.3).

For code style consistency, the renderer **always emits both loops** regardless of `tiles_per_block` value or whether DMA loads are placed. When `tiles_per_block = 1`, the tile loop runs 1 iteration. When `tiles_per_block = total_tiles`, the block loop runs 1 iteration. Neither case collapses to a single loop — the double structure is always explicit.

### 6.2 Eager Mode Loop Nest

The **eager mode** is the baseline loop nest: each operator from the math function runs to completion before the next begins. Every intermediate tensor is fully materialized across all its tiles. No ops share loops — 13 ops produce 13 sequential loop nests.

#### Deriving an op's loop nest

Each op's loop structure follows mechanically from its tensor dimensions (§5.1) and consumed axes (§5.2):

1. **Parallel loops**: one double loop (§6.1) per dimension in the op's output tensor.
2. **Reduction loop**: if the op consumes a dimension (present in operand axes but absent from output axes), nest an inner double loop for that dimension. Initialize the accumulator before the reduction loop.

For the running example, dimensions: d0 (seq_q), d1 (d_k), d2 (seq_k), d5 (d_v).

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

#### Running example

The full eager mode loop nest for attention. Each dimension has a double loop (block + tile) per §6.1. `dX_num_blocks` and `dX_tiles_per_block` are the block count and tiles-per-block for dimension X. Ops execute sequentially — each runs over all its tiles before the next begins.

![Eager mode loop nest for attention. Outer boxes group non-blocking consecutive ops. Inner boxes highlight reduction loops. Dashed arrows labeled BLOCKS mark the 4 blocking boundaries where the consumer requires the producer's complete output. Plain arrows connect non-blocking transitions where each tile is independently valid.](diagrams/eager_loopnest.png)

```python
""" Op 0: nc_transpose(Q) → Q_t(d1, d0) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d1_block in range(d1_num_blocks):
            for i_d1_tile in range(d1_tiles_per_block):
                i_d1 = i_d1_block * d1_tiles_per_block + i_d1_tile
                Q_t[i_d1, i_d0] = nc_transpose(Q[i_d0, i_d1])

""" Op 1: nc_transpose(K) → K_t(d1, d2) """
for i_d1_block in range(d1_num_blocks):
    for i_d1_tile in range(d1_tiles_per_block):
        i_d1 = i_d1_block * d1_tiles_per_block + i_d1_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                K_t[i_d1, i_d2] = nc_transpose(K[i_d2, i_d1])

""" Op 2: nc_matmul(Q_t, K_t) → S(d0, d2), accumulate over d1 """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                S[i_d0, i_d2] = 0
                for i_d1_block in range(d1_num_blocks):
                    for i_d1_tile in range(d1_tiles_per_block):
                        i_d1 = i_d1_block * d1_tiles_per_block + i_d1_tile
                        S[i_d0, i_d2] += nc_matmul(Q_t[i_d1, i_d0], K_t[i_d1, i_d2])

""" Op 3: tensor_scalar(S, multiply, scale) → scaled_S(d0, d2) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                scaled_S[i_d0, i_d2] = tensor_scalar(S[i_d0, i_d2], multiply, scale)

""" Op 4: affine_select(scaled_S) → masked_S(d0, d2) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                masked_S[i_d0, i_d2] = affine_select(scaled_S[i_d0, i_d2])

""" Op 5: tensor_reduce(masked_S, max) → max_S(d0), reduce over d2 """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        max_S[i_d0] = -inf
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                max_S[i_d0] = max(max_S[i_d0], tensor_reduce(masked_S[i_d0, i_d2]))

""" Op 6: tensor_scalar(masked_S, max_S, subtract) → shifted_S(d0, d2) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                shifted_S[i_d0, i_d2] = tensor_scalar(masked_S[i_d0, i_d2], max_S[i_d0], subtract)

""" Op 7: activation(shifted_S, exp) → exp_S(d0, d2) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                exp_S[i_d0, i_d2] = activation(shifted_S[i_d0, i_d2], exp)

""" Op 8: tensor_reduce(exp_S, add) → sum_exp(d0), reduce over d2 """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        sum_exp[i_d0] = 0
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                sum_exp[i_d0] += tensor_reduce(exp_S[i_d0, i_d2])

""" Op 9: activation(sum_exp, reciprocal) → inv_sum(d0) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        inv_sum[i_d0] = activation(sum_exp[i_d0], reciprocal)

""" Op 10: nc_transpose(exp_S) → exp_S_t(d2, d0) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d2_block in range(d2_num_blocks):
            for i_d2_tile in range(d2_tiles_per_block):
                i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                exp_S_t[i_d2, i_d0] = nc_transpose(exp_S[i_d0, i_d2])

""" Op 11: nc_matmul(exp_S_t, V) → attn(d0, d5), accumulate over d2 """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d5_block in range(d5_num_blocks):
            for i_d5_tile in range(d5_tiles_per_block):
                i_d5 = i_d5_block * d5_tiles_per_block + i_d5_tile
                attn[i_d0, i_d5] = 0
                for i_d2_block in range(d2_num_blocks):
                    for i_d2_tile in range(d2_tiles_per_block):
                        i_d2 = i_d2_block * d2_tiles_per_block + i_d2_tile
                        attn[i_d0, i_d5] += nc_matmul(exp_S_t[i_d2, i_d0], V[i_d2, i_d5])

""" Op 12: tensor_scalar(attn, inv_sum, multiply) → output(d0, d5) """
for i_d0_block in range(d0_num_blocks):
    for i_d0_tile in range(d0_tiles_per_block):
        i_d0 = i_d0_block * d0_tiles_per_block + i_d0_tile
        for i_d5_block in range(d5_num_blocks):
            for i_d5_tile in range(d5_tiles_per_block):
                i_d5 = i_d5_block * d5_tiles_per_block + i_d5_tile
                output[i_d0, i_d5] = tensor_scalar(attn[i_d0, i_d5], inv_sum[i_d0], multiply)
```

Every intermediate is fully materialized. The execution order matches the math function exactly: each op sees completed inputs, correct by construction.

### 6.3 Tiles Per Block

`tiles_per_block` is a per-dimension parameter that controls the block size — how many tiles are grouped before the tile loop iterates over them. Larger `tiles_per_block` means larger SBUF buffers but fewer DMA round-trips: the loaded block is reused across `tiles_per_block` compute iterations. Must divide `total_tiles`.

With `tiles_per_block > 1`, the block/tile split becomes meaningful:

```
for i_d0_blocklock in range(32 // 4):            # block: 8 iterations
  [DMA load: 4 tiles of Q along d0]
  for i_d0_tileile in range(4):                 # tile: 4 iterations per block
    ...
```

When no DMA load is placed on a dimension (e.g., reduction phases that operate on on-chip intermediates), the double loop still appears — no loads fire between them, but the structure is preserved.

**Enumeration.** For each unique dimension, `tiles_per_block` can be any divisor of `total_tiles` (= `dim_size / tile_size`). The total number of combinations is the product of divisor counts across all dimensions:

$$\prod_{\text{dims}} |\text{divisors}(total\_tiles_i)|$$

For the running example: $3 \times 1 \times 1 \times 2 = 6$ (d0: 4 tiles → 3 divisors, d1: 1 tile → 1, d2: 1 tile → 1, d5: 2 tiles → 2).

### 6.4 Load Placements

`load_placements` maps each input tensor to an integer `relative_pos` — how deep into the loop nest the DMA load is placed. The N dependent dimensions of a tensor create N+1 placement slots:

- **Slot 0**: outside all dependent loops (before the outermost dependent dim's block loop)
- **Slot k**: between the k-th dependent dim's block and tile loops
- **Slot N**: between the N-th (innermost) dependent dim's block and tile loops

Placing a load at slot k means it is automatically inside the block loops of dims 1..k (since they are enclosing). The load fires once per block iteration of the k-th dim. Deeper placement → more frequent reloads but smaller SBUF buffers.

Each slot determines the DMA buffer size along every dependent axis:

- **Dims 1..k** (load is inside their block loop): `NUM_TILES = tiles_per_block` — only the current block is in SBUF
- **Dims k+1..N** (load is outside their loop): `NUM_TILES = total_tiles` — the full extent is loaded

Load placement is the **only** allocation with a real search space — hoisting a load outward amortizes DMA latency across more compute iterations at the cost of larger SBUF buffers. All other on-chip allocations are either **deterministic** (accumulation targets like matmul/reduce outputs must be outside the loop they accumulate across, dictated by the math data flow) or **trivially minimum-scope** (SSA intermediates and cross-pass buffers have no DMA cost to amortize, so they always go at the innermost possible scope to conserve SBUF/PSUM capacity).

For the running example:

| Tensor | Axes (after unification) | Dependent dims (loop order) | Valid relative_pos |
|---|---|---|---|
| Q | (d0, d1) | d0, d1 | 0, 1, 2 |
| K | (d2, d1) | d1, d2 | 0, 1, 2 |
| V | (d2, d5) | d5, d2 | 0, 1, 2 |

### 6.5 Enumeration

The search space is the cross-product of three independent axes:

**Axis 1 — Loop order** (§6.2): permutations of items (parallel dims + reduction dim phases) where ordering chains (same-dim phases + cross-dim data dependencies) maintain sequence. Count = $n! / \prod_i c_i!$ where $c_i$ are chain lengths.

**Axis 2 — Tiles per block**: for each unique dimension, `tiles_per_block` can be any divisor of `total_tiles` (= dim_size / tile_size). The number of choices per dimension is the divisor count of its total tiles.

**Axis 3 — Load placements**: for each input tensor, `relative_pos` determines placement depth. The number of valid slots depends on the tensor's dependent dims and their nesting structure (§6.4).

$$\text{Full space} = \text{loop orders} \times \prod_{\text{dims}} |\text{divisors}(total\_tiles_i)| \times \prod_{\text{tensors}} \text{valid\_slots}_j$$

Hardware validation prunes infeasible candidates. In the `Tensor` layout, `TILE_SIZES[AXES[0]]` is the partition dimension (≤ 128 by hardware), while `NUM_TILES` for all axes expand into the free dimension. Larger `tiles_per_block` grows the free dim, which must stay within SBUF capacity (24 MB) and PSUM accumulator limits. Every survivor is rendered, compiled, and benchmarked.

**Running example** with typical Llama-style shapes: `q[4096, 128], k[4096, 128], v[4096, 128]`:

| Dim | Size | tile_size | total_tiles | Valid tiles_per_block (divisors) | Count |
|---|---|---|---|---|---|
| d0 (seq_q) | 4096 | 128 | 32 | {1, 2, 4, 8, 16, 32} | 6 |
| d1 (d_k) | 128 | 128 | 1 | {1} | 1 |
| d2 (seq_k) | 4096 | 512 | 8 | {1, 2, 4, 8} | 4 |
| d5 (d_v) | 128 | 128 | 1 | {1} | 1 |

- **Loop orders**: 18 (from 30 chain-valid, after buffer-reuse pruning)
- **Tiles per block**: $6 \times 1 \times 4 \times 1 = 24$
- **Load placements**: $3 \times 3 \times 3 = 27$ (Q: 3 slots, K: 3 slots, V: 3 slots)

$$\text{Full space} = 18 \times 24 \times 27 = 11{,}664 \text{ candidates}$$

## 7. Render (Specialization)

Rendering is a **specialization** step: it takes the math function (§1), the op call list (§4), dimension analysis (§5), a loop nest (§6), and **concrete input shapes and dtypes** to produce an NKI kernel tailored to those specific inputs. Each unique combination of input shapes and dtypes requires its own tuned kernel. Infrastructure ops (DMA loads, PSUM→SBUF staging, DMA stores) are generated by the renderer — they do not appear in the math function.

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
2. **Emit loop nest**: walk the `loop_order` from the `Schedule`. Each entry's `(level, order)` determines nesting: entries at different levels produce nested loops (`nl.affine_range`), entries at the same level with different orders produce back-to-back sequential loops. Each `Loop` emits a block loop (`nl.affine_range(total_tiles // tiles_per_block)`) and tile loop (`nl.affine_range(tiles_per_block)`).
3. **Emit DMA loads**: for each input tensor, place the `nisa.dma_copy(HBM→SBUF)` at the `relative_pos` specified by `schedule.load_placements` (§6.4). Allocate an SBUF buffer with shape determined by `Tensor.NUM_TILES`.
4. **Emit PSUM accumulators**: before each dependent-dim loop, emit `nl.ndarray(..., buffer=nl.psum)` followed by `nisa.memset(buf, value=...)`. Always this two-step sequence — never `nl.zeros`.
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

Generic attention with symbolic shapes `q[S_q, D_k]`, `k[S_k, D_k]`, `v[S_k, D_v]` and all `tiles_per_block=1` with `relative_pos=max`. Tile sizes from §5.3: $t_0 = 128$ (d0), $t_1 = 128$ (d1), $t_2 = 512$ (d2), $t_5 = \min(512, D_v)$ (d5). Total tiles per dim: $\text{total\_tiles}_i = S_i / t_i$.

The actual generated code always emits the double loop per §6.1 (e.g., `for i_d0_outer in nl.affine_range(total // 1)` + `for i_d0_inner in nl.affine_range(1)`). Below, the inner loops are elided for readability since `tiles_per_block=1` makes them single-iteration.

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

Each tensor's shape comment shows its axes and the symbolic shape `(tile_par, nb_0, nb_1, nt_0, nt_1, tile_free)`. The nb and nt values for each axis come from the load placement (§6.2):
- **after compute loop**: nb=1, nt=1 (inside both load and compute loops)
- **between load/compute**: nb=1, nt=tiles_per_block (inside load loop, before compute loop)
- **before load loop**: nb=total_tiles/tiles_per_block, nt=tiles_per_block (outside all loops for that dim)

---

## 8. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel (`attention_cte.py`) already uses `nl.ds` for this purpose.
