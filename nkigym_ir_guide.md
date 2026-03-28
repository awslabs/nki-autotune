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

## 4. Dimension Analysis

To tile and schedule a math function for hardware, we need to know what dimensions exist, how they relate across ops, and which are parallel vs reduction. This is derived mechanically from `NKIOp.OPERAND_AXES` and `OUTPUT_AXES`.

### 4.1 Dimension Assignment and Unification

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

### 4.2 Parallel vs Reduction

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

### 4.3 Tile Sizes

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

## 5. Schedule

A `Schedule` is an immutable, hashable descriptor that captures *how* to execute the math function — independently of *what* to compute (which comes from §1–4).

```python
class Loop(NamedTuple):
    dim_id: str
    tile_size: int
    tiles_per_block: int

class Schedule(NamedTuple):
    loop_order: dict[Loop, int]  # Loop → depth; dict order = execution order
    load_placements: dict[str, int]  # tensor name → level (0..num_dependent_dims)
```

### 5.1 Loop Order

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

### 5.2 Load Placements

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

<!-- ---

## 2. Parsed Ops + Analysis

### 2.2 Analysis

`analyze_dims()` takes the 13 op calls + parameter shapes and:

1. **Assigns dimension IDs**: `q → (d0, d1)`, `k → (d2, d3)`, `v → (d4, d5)`.
2. **Unifies dimensions** across ops using `OPERAND_AXES` axis labels:
   - Ops 0–1 transpose: q`(d0, d1)` → q_t`(d1, d0)`. k`(d2, d3)` → k_t`(d3, d2)`.
   - Op 2 matmul1: K=d1 unifies d3→d1. Output scores → (d0, d2).
   - Ops 3–4 (scale, mask): shape-preserving, no new unifications.
   - Op 10 transpose: exp_s`(d0, d2)` → exp_s_t`(d2, d0)`.
   - Op 11 matmul2: K=d2 unifies d4→d2. Output → (d0, d5).
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
| Op 5 | `NKITensorReduce` (max_s) | d2 | d2 pass 0 |
| Op 8 | `NKITensorReduce` (sum_exp) | d2 | d2 pass 1 |
| Op 11 | `NKIMatmul` (attn) | d2 | d2 pass 2 |

Non-barrier ops are classified by position relative to barriers:

| Classification | Ops | Description |
|---|---|---|
| **pre_compute** for (d1, 0) | 0 (transpose q), 1 (transpose k) | Transpose inputs before matmul1 |
| **pre_compute** for (d2, 0) | 3 (scale), 4 (affine_select) | Scale and mask before d2 pass 0 barrier |
| **pre_compute** for (d2, 1) | 6 (subtract), 7 (exp) | 2D ops before d2 pass 1 barrier |
| **inter_pass** after (d2, 1) | 9 (reciprocal) | 1D op between d2 passes 1 and 2 |
| **pre_compute** for (d2, 2) | 10 (transpose) | Transpose exp_s before matmul2 |
| **post_compute** after (d2, 2) | 12 (multiply) | Normalize output by inv_sum |

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
def attention(q, k, v, scale):
    hbm_tensor_0 = nl.ndarray((512, 1024), dtype=q.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(4):                   # d0: seq_q tiles
        for i_1 in nl.affine_range(2):               # d5: d_v tiles
            """ Pass 0: running max over d2 """
            running_max = nl.ndarray(..., buffer=nl.psum)
            for i_2 in nl.sequential_range(4):       # d2 tiles
                # load q tile → nc_transpose → tensor_copy to SBUF
                # load k tile → nc_transpose → tensor_copy to SBUF
                # matmul1 → stage to SBUF
                # → tensor_scalar(multiply, scale) → tensor_copy to SBUF
                # → affine_select(causal) → tensor_reduce(max) into running_max

            """ Pass 1: running sum (uses completed max) """
            running_sum = nl.ndarray(..., buffer=nl.psum)
            for i_3 in nl.sequential_range(4):       # d2 tiles
                # load q, k → transpose each → matmul1 → stage
                # → scale → affine_select(causal)
                # → subtract(running_max) → exp → reduce(add)
            # inter-pass: reciprocal(running_sum) in SBUF

            """ Pass 2: matmul2 accumulation """
            psum_acc = nl.ndarray(..., buffer=nl.psum)
            for i_4 in nl.affine_range(4):           # d2 tiles (K for matmul2)
                # load q, k → transpose each → matmul1 → stage
                # → scale → affine_select(causal)
                # → subtract → exp → transpose
                # → load v tile → matmul2 into psum_acc
            # tensor_scalar(multiply, reciprocal) → stage psum_acc → SBUF
            # DMA store to hbm_tensor_0
    return hbm_tensor_0
```

**Per-pass recomputation**: transpose + matmul1 + scale + mask + softmax intermediates recomputed in every d2 pass because their d2-dimension outputs don't persist in SBUF across passes. Inter-pass scalars (running_max, running_sum, reciprocal) persist between passes.

## 6. Design Rules

**Separation of concerns**: the schedule captures *how* to execute independently of *what* to execute (from analysis). The NKI kernel is a rendering output.

**All orderings are correct**: the only constraint is that same-dim reduction passes maintain relative order. Hardware validation filters infeasible schedules.

**Combinatorial enumeration**: full space = loop orders × placements × blocking. Each schedule is an independent grid point.

**Partition-axis cap**: `tile_size = max(TILE_LIMITS)`, capped at 128 when the dim appears at position 0 of any tensor. Resolves conflicts where a dim serves as both free axis (N:512) and partition axis (K:128).

**Cross-tile reduction**: matmul accumulates in PSUM across K tiles. `tensor_reduce` accumulates in SBUF via read-modify-write.

**Sequential passes**: when a full reduction must complete before the next op starts, the dim appears multiple times in `loop_order` as `(dim_id, pass_0), (dim_id, pass_1), ...`.

**Per-pass recomputation**: d2-dimension intermediates recomputed per pass (SBUF holds one tile at a time). 1D inter-pass results persist between passes.

**IO is explicit**: `nisa.dma_copy` for HBM↔SBUF, `nisa.tensor_copy` for PSUM→SBUF. All ISA calls use `dst=`, no return values.

**Naming**: `{memspace}_tensor_N` per memory space. Input params keep user names. PSUM always `nl.float32`. -->
