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
    Q_t = nkigym.transpose(Q)
    K_t = nkigym.transpose(K)
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
    exp_S_t = nkigym.transpose(exp_S)
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
    [TILE_SIZES[AXES[0]], NUM_TILES[AXES[0]]
        NUM_TILES[AXES[1]], ..., NUM_TILES[AXES[N-1]],
        TILE_SIZES[AXES[1]], ..., TILE_SIZES[AXES[N-1]]
    ]
    """
    NAME: str
    AXES: tuple[str, ...]
    TILE_SIZES: dict[str, int]
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
        Render into NKI kernel source codes
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
        return f"nisa.nc_matmul(dst={dst.NAME}, stationary={stat.NAME}, moving={mov.NAME})"


class NKITranspose(NKIOp):
    """Swap partition and free dims (Tensor Engine or Vector Engine)

    data(P, F) → output(F, P).
    Tensor Engine path: SBUF→PSUM, both dims ≤ 128.
    Vector Engine path: SBUF/PSUM→SBUF/PSUM, both dims ≤ 32.
    Real hardware op, not a free view.
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
        return f"nisa.nc_transpose(dst={dst.NAME}, data={src.NAME})"


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
        return f"nisa.tensor_reduce(dst={dst.NAME}, data={data.NAME}, op=nl.{op_name})"


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
| 0 | `Q_t = nkigym.transpose(Q)` | Q(d0,d1) → Q_t(d1,d0) |
| 1 | `K_t = nkigym.transpose(K)` | K(d2,d3) → K_t(d3,d2) |
| 2 | `S = nkigym.nc_matmul(Q_t, K_t)` | stationary Q_t(d1,d0): K=d1, M=d0; moving K_t(d3,d2): K=d3, N=d2; shared K → **unify** d3→d1; S(d0,d2) |
| 3 | `scaled_S = nkigym.tensor_scalar(S, op0="multiply", operand0=scale)` | scaled_S(d0,d2) |
| 4 | `masked_S = nkigym.affine_select(scaled_S, ...)` | masked_S(d0,d2) |
| 5 | `max_S = nkigym.tensor_reduce(masked_S, op="max")` | max_S(d0) |
| 6 | `shifted_S = nkigym.tensor_scalar(masked_S, max_S, op0="subtract")` | P axis: d0 from data, d0 from operand0 → **assert** d0=d0 ✓; shifted_S(d0,d2) |
| 7 | `exp_S = nkigym.activation(shifted_S, op="exp")` | exp_S(d0,d2) |
| 8 | `sum_exp = nkigym.tensor_reduce(exp_S, op="add")` | sum_exp(d0) |
| 9 | `inv_sum = nkigym.activation(sum_exp, op="reciprocal")` | inv_sum(d0) |
| 10 | `exp_S_t = nkigym.transpose(exp_S)` | exp_S_t(d2,d0) |
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

Each dimension's tile size is always the maximum across all `MAX_TILE_SIZES` entries from ops that use it:

1. Collect all `MAX_TILE_SIZES` entries for a dimension across all ops where it appears.
2. `tile_size = max(all limits for this dim)`.

When an op's limit for a dimension is smaller than the tile size, the renderer emits an in-place loop that packs multiple op invocations to cover the full tile. For example, d2 has tile_size 512 (from nc_matmul N:512), but transpose has F:128 — so the renderer emits 4 transpose calls per tile to cover 512 elements.

| Dim | Limits collected from ops | tile_size |
|---|---|---|
| d0 | P:128 (transpose, tensor_scalar, affine_select, tensor_reduce, activation), M:128 (nc_matmul), F:128 (transpose) | 128 |
| d1 | F:128 (transpose Q, transpose K), K:128 (nc_matmul) | 128 |
| d2 | P:128 (transpose K), N:512 (nc_matmul₁), F:128 (transpose exp_S), K:128 (nc_matmul₂) | 512 |
| d5 | N:512 (nc_matmul₂) | 512 |

## 6. Schedule

A `Schedule` is an immutable, hashable descriptor that captures *how* to execute the math function — independently of *what* to compute (which comes from §1–5).

```python
class Loop(NamedTuple):
    dim_id: str
    tile_size: int
    tiles_per_block: int

class Schedule(NamedTuple):
    loop_order: dict[Loop, int]  # Loop → depth; dict order = execution order
    load_placements: dict[str, int]  # tensor name → level (0..num_dependent_dims)
```

### 6.1 Loop Order

`loop_order` maps each `Loop` to a `depth` integer. Dict iteration order defines execution order. Each `Loop` represents a **two-level loop** — a block loop and a tile loop:

```
for block in range(num_blocks):                 # block loop: num_blocks = total_tiles / tiles_per_block
    for tile in range(tiles_per_block):          # tile loop: iterate within the block
```

`depth` is the nesting level — loops at the same depth are siblings (sequential), loops at increasing depth are nested.

Parallel dims each increase the depth by 1. Reduction dims that require multiple sequential loops (e.g., running max, running sum, matmul accumulation) repeat at the **same depth** — they are back-to-back sibling loops, not nested.

`tiles_per_block` controls blocking — how many tiles are grouped into one block. Must divide the total tile count. Larger `tiles_per_block` means larger SBUF buffers (holding more tiles at once) but fewer DMA round-trips — the loaded block is reused across all tile iterations within it.

**Example loops** for the running example:

| Pos | Loop (dim_id, tile_size, tpb) | depth | Meaning |
|---|---|---|---|
| 0 | (d0, 128, 1) | 0 | Outer parallel loop: seq_q |
| 1 | (d5, 512, 1) | 1 | Parallel loop: d_v |
| 2 | (d1, 128, 1) | 2 | d_k reduction (matmul1 K axis) |
| 3 | (d2, 512, 1) | 2 | seq_k reduction (running max) |
| 4 | (d2, 512, 1) | 2 | seq_k reduction (running sum) — needs completed max from pos 3 |
| 5 | (d2, 512, 1) | 2 | seq_k reduction (matmul2 K axis) — needs completed inv_sum from pos 4 |

d2 appears 3 times because the math function has 3 barrier ops that fully reduce over d2: `tensor_reduce(max)`, `tensor_reduce(sum)`, and `nc_matmul` (matmul2). Each barrier requires a complete sweep — the next cannot start until the previous reduction is finished across all d2 tiles.

### 6.2 Load Placements

`load_placements` maps each input tensor name to an integer level from 0 to `num_dependent_dims`. The level controls how deep into the loop nest the DMA load is placed, relative to the tensor's dependent dims. Compute and store placements are not free parameters — compute runs immediately when operands are available, stores run immediately when results are ready.

`num_dependent_dims` is the number of unique loop dimensions that appear in the tensor's axes (after unification), ordered by their position in `loops`. Each level moves the load point one dependent loop deeper. Since each Loop is a block loop + tile loop, "outside" means before the block loop, and "inside" means between the block and tile loops:

- **Level 0**: outside all dependent loops (before their block loops) → `NUM_TILES = total_tiles` for all dependent axes
- **Level k**: inside the first k dependent loops (between their block and tile loops) → `NUM_TILES = tiles_per_block` for those k axes, `total_tiles` for the rest
- **Level N**: inside all dependent loops → `NUM_TILES = tiles_per_block` for all dependent axes

To hold just 1 tile for a dimension, set `tiles_per_block=1` on the Loop and place the tensor inside — no separate case needed.

A tensor with N dependent dims has N+1 relative levels.

For the running example:

| Tensor | Axes (after unification) | Dependent dims (loop order) | Relative levels |
|---|---|---|---|
| Q | (d0, d1) | d0, d1 | 0, 1, 2 |
| K | (d2, d1) | d1, d2 | 0, 1, 2 |
| V | (d2, d5) | d5, d2 | 0, 1, 2 |

### 6.3 Enumeration

The schedule space is the cross-product of three independent axes:

**Axis 1 — Loop order**: permutations of the $L$ loop entries. If a dimension appears $r$ times (multiple reduction sweeps), those $r$ entries must maintain their relative order. The constraint reduces the space from $L!$ to $L! / \prod r_i!$ where $r_i$ is the repeat count of each dimension.

**Axis 2 — Blocking**: for each unique dimension, `tiles_per_block` can be any divisor of `total_tiles` (= dim_size / tile_size). The number of choices per dimension is the divisor count of its total tiles.

**Axis 3 — Load placements**: for each input tensor with $N$ dependent dims, the level ranges from 0 to $N$, giving $N+1$ choices.

$$\text{Full space} = \text{loop orders} \times \prod_{\text{dims}} |\text{divisors}(total\_tiles_i)| \times \prod_{\text{tensors}} (N_j + 1)$$

Hardware validation prunes infeasible schedules. In the `Tensor` layout, `TILE_SIZES[AXES[0]]` is the partition dimension (≤ 128 by hardware), while `NUM_TILES` for all axes — including tiles_per_block and num_blocks — expand into the free dimension. Larger blocking grows the free dim, which must stay within SBUF capacity (24 MB) and PSUM accumulator limits. Every survivor is rendered, compiled, and benchmarked.

**Running example** with typical Llama-style single-head attention shapes: `q[4096, 128], k[4096, 128], v[4096, 128]`:

| Dim | Size | tile_size | total_tiles | Valid tpb (divisors) | Count |
|---|---|---|---|---|---|
| d0 (seq_q) | 4096 | 128 | 32 | {1, 2, 4, 8, 16, 32} | 6 |
| d1 (d_k) | 128 | 128 | 1 | {1} | 1 |
| d2 (seq_k) | 4096 | 512 | 8 | {1, 2, 4, 8} | 4 |
| d5 (d_v) | 128 | 512 | 1 | {1} | 1 |

- **Loop orders**: $6! / 3! = 120$ (4 unique dims after unification: d0, d1, d2, d5; d2 appears 3× for its 3 barrier ops → 6 entries, d2s maintain relative order)
- **Blocking**: $6 \times 1 \times 4 \times 1 = 24$
- **Load placements**: $3 \times 3 \times 3 = 27$ (Q, K, V each have 2 dependent dims → 3 levels)

$$\text{Full space} = 120 \times 24 \times 27 = 77{,}760 \text{ candidates}$$

## 7. Render (Specialization)

Rendering is a **specialization** step: it takes the math function (§1), the op call list (§4), dimension analysis (§5), a schedule (§6), and **concrete input shapes and dtypes** to produce an NKI kernel tailored to those specific inputs. Each unique combination of input shapes and dtypes requires its own tuned kernel. Infrastructure ops (DMA loads, PSUM→SBUF staging, DMA stores) are generated by the renderer — they do not appear in the math function.

### 7.1 Render Pipeline

```
render(math_function, analysis, schedule, input_shapes, dtypes) → NKI kernel source
```

1. **Emit preamble**: imports (`nki`, `nki.language as nl`, `nki.isa as nisa`), `@nki.jit` decorator, function signature with input params, shape and dtype assertions for each input, HBM output allocation.

**Tensor naming**: input/output tensors use the same names as the math function (`q`, `k`, `v`, `output`). Intermediate tensors are named `{location}_{var_name}` where `location` is `sbuf` or `psum` and `var_name` is the variable name from the parsed math function (§4). Examples: `sbuf_Q_t`, `psum_S`, `sbuf_masked_S`.
2. **Emit loop nest**: walk `schedule.loop_order` — each Loop emits a block loop (`nl.affine_range(num_blocks)`) and a tile loop (`nl.affine_range(tiles_per_block)`). Depth determines nesting; same-depth loops are sequential siblings.
3. **Emit DMA loads**: for each input tensor, place the `nisa.dma_copy(HBM→SBUF)` at the level specified by `schedule.load_placements`. Allocate an SBUF buffer with shape determined by `Tensor.NUM_TILES` (§6.2).
4. **Emit compute ops**: at the innermost loop level, walk the math function's op calls. For each op, build a `RenderContext` (§2) and call `NKIOp.render(ctx)`. Reduction ops (matmul, tensor_reduce) accumulate into PSUM.
5. **Emit staging**: after reduction loops complete, `nisa.tensor_copy(PSUM→SBUF)` stages the accumulator for subsequent ops or store.
6. **Emit post-reduction ops**: ops that consume reduction results (e.g., multiply by inv_sum) render after staging, operating on SBUF data.
7. **Emit DMA store**: `nisa.dma_copy(SBUF→HBM)` writes the final output tile back.

### 7.2 RenderContext Construction

For each op call site, the renderer builds a `RenderContext`:

- `outputs`: output `Tensor` with axes, tile sizes, and buffer name from the render's naming scheme
- `operands`: input `Tensor`(s) with SBUF buffer references and slice expressions for the current tile
- `config_kwargs`: pass-through from the math function call (e.g., `op0="multiply"`, `cmp_op="greater_equal"`)
- `tile_idx`: maps each dim_id to the current loop variable (e.g., `{"d0": "i_0", "d2": "i_2"}`)
- `tile_start`: maps each dim_id to the element offset expression (e.g., `{"d0": "i_0 * 128", "d2": "i_2 * 512"}`)

Each `NKIOp.render(ctx)` returns an NKI source line using `nisa.*` calls with `dst=` syntax.

### 7.3 Example Output

For the Llama-style example (`q[4096, 128], k[4096, 128], v[4096, 128]`) with the example schedule from §6.1 (all `tiles_per_block=1`, all loads at level 0):

```python
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def attention(q, k, v, scale):
    assert q.shape == (4096, 128), f"Expected q shape (4096, 128), got {q.shape}"
    assert k.shape == (4096, 128), f"Expected k shape (4096, 128), got {k.shape}"
    assert v.shape == (4096, 128), f"Expected v shape (4096, 128), got {v.shape}"
    assert q.dtype == nl.float16
    assert k.dtype == nl.float16
    assert v.dtype == nl.float16

    output_hbm = nl.ndarray((4096, 128), dtype=nl.float16, buffer=nl.shared_hbm)

    """ Load all tiles (level 0 placement) """
    sbuf_q = nl.ndarray((...), dtype=q.dtype, buffer=nl.sbuf)
    sbuf_k = nl.ndarray((...), dtype=k.dtype, buffer=nl.sbuf)
    sbuf_v = nl.ndarray((...), dtype=v.dtype, buffer=nl.sbuf)
    # nisa.dma_copy for each tile of q, k, v

    for i_d0 in nl.affine_range(32):                # d0 block loop (32 blocks)
        for i_d5 in nl.affine_range(1):             # d5 block loop (1 block)
            for i_d1 in nl.affine_range(1):         # d1 block loop (1 block)
                """ d1 reduction: matmul1 """
                # NKITranspose.render → nisa.nc_transpose (Q)
                # NKITranspose.render → nisa.nc_transpose (K)
                # NKIMatmul.render   → nisa.nc_matmul (Q_t @ K_t → S)

            """ d2 reduction 1: running max """
            for i_d2 in nl.affine_range(8):         # d2 block loop (8 blocks)
                # NKITensorScalar.render → nisa.tensor_scalar (scale)
                # NKIAffineSelect.render → nisa.affine_select (causal mask)
                # NKITensorReduce.render → nisa.tensor_reduce (max)

            """ d2 reduction 2: running sum """
            for i_d2 in nl.affine_range(8):
                # recompute: scale, mask
                # NKITensorScalar.render → nisa.tensor_scalar (subtract max)
                # NKIActivation.render   → nisa.activation (exp)
                # NKITensorReduce.render → nisa.tensor_reduce (sum)
            # NKIActivation.render → nisa.activation (reciprocal)

            """ d2 reduction 3: matmul2 """
            for i_d2 in nl.affine_range(8):
                # recompute: scale, mask, subtract, exp, transpose
                # NKIMatmul.render → nisa.nc_matmul (exp_S_t @ V → attn)

            # nisa.tensor_copy (PSUM → SBUF)
            # NKITensorScalar.render → nisa.tensor_scalar (multiply inv_sum)
            # nisa.dma_copy (SBUF → HBM)

    return output_hbm
```

Each `nisa.*` call is produced by the corresponding `NKIOp.render(ctx)` — the renderer constructs the `RenderContext` and delegates to the op subclass.
