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

## 2. Tensor Layout

Every on-chip tensor uses a uniform **4D layout**: `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)`. This is the central design decision — loop structure, buffer sizing, and all transforms operate on this single shape.

**Hardware constraint.** Trainium DMA cannot operate on tensors with more than 5 meaningful (non-singleton) dimensions, so 4D is the maximum usable rank for a two-axis (P, F) tensor.

**Memory hierarchy.** Trainium has three memory levels: HBM (off-chip) → SBUF (on-chip scratchpad) → PSUM (accumulator registers). Each boundary between adjacent levels introduces a block/tile tiling parameter that controls arithmetic intensity — how many tiles are processed before results move to the next level. Two ops produce results in PSUM: `nc_matmul` (accumulates partial products) and `nc_transpose` (writes transposed tile to a PSUM bank). All other ops work entirely in SBUF.

**HBM↔SBUF boundary.** Active when an op reads from or writes to HBM. Introduces `tpb_hbm` (tunable) and `num_blocks_hbm` (derived). DMA loads `tpb_hbm` unified tiles per block; the `load_tensor_block` / `save_tensor_block` gadgets from `nkigym.gadgets` handle the tile-by-tile DMA internally.

**SBUF↔PSUM boundary.** Active when an op produces results in PSUM that a subsequent consumer needs in SBUF. Introduces `tpb_sbuf` (tunable) and `num_blocks_sbuf` (derived). Controls how many tiles pile up in PSUM before `tensor_copy` spills to SBUF.

The two boundaries nest — PSUM batching happens within each HBM block:

$$\texttt{dim\_size} = \texttt{num\_blocks\_hbm} \times \texttt{tpb\_hbm} \times \texttt{interleave\_tile\_size}$$

$$\texttt{num\_blocks\_sbuf} \times \texttt{tpb\_sbuf} = \texttt{tpb\_hbm} \quad \text{(by construction)}$$

**Interleave.** Different ops may have different tile size limits on the same dimension (e.g., `nc_matmul` N=512 vs `nc_transpose` F=128). `interleave_tile_size` = max of all op tile sizes on the dimension. Ops with smaller tile sizes iterate `tiles_per_interleave_group = interleave_tile_size / op_tile_size` times per unified tile.

This gives up to four nested loops per dimension. When only the HBM↔SBUF boundary is active (3 loops):

| Loop | Trip count | Controls |
|---|---|---|
| `i_block_hbm` | `num_blocks_hbm` | DMA block iterations |
| `i_tpb_hbm` | `tpb_hbm` | Unified tiles within an HBM block |
| `i_ig` | `tiles_per_interleave_group` | Per-op sub-tile iteration |

When both boundaries are active, `i_tpb_hbm` splits into two PSUM loops (`num_blocks_sbuf × tpb_sbuf = tpb_hbm`), giving 4 loops:

| Loop | Trip count | Controls |
|---|---|---|
| `i_block_hbm` | `num_blocks_hbm` | DMA block iterations |
| `i_block_sbuf` | `num_blocks_sbuf` | PSUM spill batches within an HBM block |
| `i_tpb_sbuf` | `tpb_sbuf` | Unified tiles processed before PSUM spill |
| `i_ig` | `tiles_per_interleave_group` | Per-op sub-tile iteration |

**Component classification:**

- **`interleave_tile_size`** (hardware) — max of all hardware tile size limits across ops on the dimension. This is the unified tile size — one iteration of the block/tile loops processes `interleave_tile_size` elements.
- **`tiles_per_interleave_group`** (hardware, per-op) — `interleave_tile_size / op_tile_size`. The buffer stores at the smallest op tile size; ops with larger tiles consume multiple buffer slots per iteration.
- **`tpb_hbm`** (tunable) — unified tiles per HBM block. Data is loaded from HBM once per block via `load_tensor_block`, then reused across compute iterations. Controls arithmetic intensity and SBUF capacity.
- **`tpb_sbuf`** (tunable, where applicable) — unified tiles per PSUM spill. Controls how many tiles pile up in PSUM before `tensor_copy` spills to SBUF. Must divide `tpb_hbm`. Defaults to `tpb_hbm` (one spill per block).
- **`num_blocks_hbm`** (derived) — `dim_size / (tpb_hbm × interleave_tile_size)`.
- **`num_blocks_sbuf`** (derived) — `tpb_hbm / tpb_sbuf`.

### Example: N dimension with transpose + matmul
```python
def matmul(lhs_T, rhs_T):
    # lhs_T[d0, d1]
    # rhs_T[d2, d0]
    rhs = nkigym.nc_transpose(rhs_T) # [d0, d2]
    result = nkigym.nc_matmul(lhs_T, rhs) # [d1, d2]
    return result
```
Consider how to produce the loops for `d2` dimension for the `nc_transpose` operator.
**Constraints**
- `num_blocks_sbuf` * `tpb_sbuf` = `tpb_hbm` by construction.
- `nc_transpose` on N has `nc_transpose_tile_size` = 128.
- `nc_matmul` on N has `nc_matmul_tile_size` = 512.
- Assume `d2` = 8192.

**Derived**
- `interleave_tile_size` = max(`nc_transpose_tile_size`, `nc_matmul_tile_size`) = 512
- `tiles_per_interleave_group` = `interleave_tile_size` / `nc_transpose_tile_size` = 4

**Active Memory Boundaries**
- Because `nc_transpose` reads from HBM tensors, the HBM-SBUF boundary is active --> need to control `tpb_hbm`.
- Because `nc_transpose` produces `rhs` in PSUM, and the subsequent consumer `nc_matmul` needs `rhs` in SBUF, the SBUF-PSUM boundary is also active --> need to control `tpb_sbuf`.

`num_blocks_hbm` * `tpb_hbm` * `interleave_tile_size` = `d2` = 8192.
Therefore, `num_blocks_hbm` * `tpb_hbm` = `d2` / `interleave_tile_size` = 16.

**Valid Tuning Options**

`tpb_hbm` must divide 16. `tpb_sbuf` must divide `tpb_hbm`.

| `tpb_hbm` | `tpb_sbuf` | `num_blocks_hbm` | `num_blocks_sbuf` |
|---|---|---|---|
| 1 | 1 | 16 | 1 |
| 2 | 1 | 8 | 2 |
| 2 | 2 | 8 | 1 |
| 4 | 1 | 4 | 4 |
| 4 | 2 | 4 | 2 |
| 4 | 4 | 4 | 1 |
| 8 | 1 | 2 | 8 |
| 8 | 2 | 2 | 4 |
| 8 | 4 | 2 | 2 |
| 8 | 8 | 2 | 1 |
| 16 | 1 | 1 | 16 |
| 16 | 2 | 1 | 8 |
| 16 | 4 | 1 | 4 |
| 16 | 8 | 1 | 2 |
| 16 | 16 | 1 | 1 |

Example: `tpb_hbm = 4, tpb_sbuf = 2` → `num_blocks_hbm = 4, num_blocks_sbuf = 2`. The loop nest for `nc_transpose` on `d2`:

```
for i_block_hbm in range(4):                    # 4 HBM blocks
    load_tensor_block(...)                       # DMA: 4 unified tiles (16 buffer slots of 128)
    for i_block_sbuf in range(2):                # 2 PSUM batches per block
        for i_tpb_sbuf in range(2):              # 2 unified tiles per batch
            for i_ig in range(4):                # 4 × 128 = 512 per unified tile
                nc_transpose(one 128-wide tile)
        tensor_copy(PSUM → SBUF)                 # spill after each batch
```

**Buffer sizing.** The buffer's `num_tiles` on a dimension = `tpb_hbm × tiles_per_interleave_group` per block (here 4 × 4 = 16 slots of 128 elements). A tensor's placement in the loop nest determines how many loop levels fall outside vs inside the buffer — shifting up grows the buffer, shifting down shrinks it. Full-range placement: `num_tiles = dim_size / min_op_tile_size`; inside all loops: `num_tiles = 1`.

## 3. Dimension Analysis

A forward pass over all ops assigns concrete dimension IDs and computes the hardware-determined components (`tile_size` and `interleave`, §2) for each dimension. No tensor starts with dimension IDs — they are discovered automatically from op structure.

For each op, build a local axis map from abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from prior ops) provide the mapping; operands without dim_ids get fresh IDs allocated. When two operands of the same op share an abstract axis (e.g., matmul's K axis appears in both stationary and moving), their concrete dims are **unified** — the later operand's dim is renamed to match the earlier one, propagating backward through all tensors that share the old ID. This is how shared dimensions are discovered without any manual labeling.

Example: double matmul `(Q @ K.T) @ V`:

```python
Q_t = NKITranspose()(data=Q)
K_t = NKITranspose()(data=K)
S = NKIMatmul()(stationary=Q_t, moving=K_t)
S_t = NKITranspose()(data=S)
output = NKIMatmul()(stationary=S_t, moving=V)
```

Walk through the forward pass. Inputs Q, K, V have no dim_ids — the pass discovers them:

1. **`nc_transpose(Q)`** — Q has no dim_ids. Allocate P=d0, F=d1. Assign **Q → (d0, d1)**. `OUTPUT_AXES = ("F", "P")` → **Q_t gets (d1, d0)**.

2. **`nc_transpose(K)`** — K has no dim_ids. Allocate P=d2, F=d3. Assign **K → (d2, d3)**. Output → **K_t gets (d3, d2)**.

3. **`nc_matmul(Q_t, K_t)`** — Q_t has (d1, d0) → K=d1, M=d0. K_t has (d3, d2) → K=d3, N=d2. Abstract axis K maps to both d1 and d3 — **unify d3→d1**. This renames K's dim_ids from (d2, d3) to (d2, d1), and K_t from (d3, d2) to (d1, d2). Now K=d1 is consistent across both operands. `OUTPUT_AXES = ("M", "N")` → **S gets (d0, d2)**.

4. **`nc_transpose(S)`** — S has (d0, d2) → P=d0, F=d2. Output → **S_t gets (d2, d0)**.

5. **`nc_matmul(S_t, V)`** — S_t has (d2, d0) → K=d2, M=d0. V has no dim_ids. K=d2 already in local map, allocate N=d4. Assign **V → (d2, d4)**. Output → **output gets (d0, d4)**.

## 4. Initial KernelIR

`KernelIR` is the intermediate representation that carries all state needed to render and transform a kernel. `build_ir(func, input_specs)` creates the base IR from a math function — it runs the dimension analysis (§3), computes tile sizes and interleave factors, then wraps everything in a `KernelIR` with each op in its own singleton fusion group (no transforms applied).

```python
@dataclass
class KernelIR:
    ops: list[tuple[NKIOp, dict[str, str], str]]
    per_op_maps: list[dict[str, str]]
    ctx: RenderContext
    return_name: str
    func_name: str
    param_names: list[str]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    fusion_groups: list[list[int]]
    buffer_degrees: dict[str, int]
```

`build_ir` performs three steps: (1) parse the math function to discover ops and their operand maps via AST inspection, (2) run dimension analysis — assign dim IDs, unify shared dimensions, create output tensors (§3), and (3) compute per-dimension tile sizes and interleave factors (§4.1–4.2). The result is a `KernelIR` ready for rendering (§4.3) or transforms (§5).

### 4.1 Determine Dimension Tile Size

Each ISA instruction imposes per-tensor limits on tile dimensions:

| Op | Dimensions | Hardware Tile Size Limit |
|---|---|---|
| nc_matmul | K (accumulation), M (partition), N (free) | K ≤ 128, M ≤ 128, N ≤ 512 |
| nc_transpose | P (partition), F (free) | P ≤ 128, F ≤ 128 |

These are the only ops implemented so far. Other ops (tensor_scalar, affine_select, tensor_reduce, activation_reduce, activation) follow the same partition ≤ 128 pattern with large SBUF-based free limits — they will be added as needed.

We assume `dim_size` is always a multiple of the smallest hardware limit across ops on that dimension. For each concrete dimension, collect all hardware limits from the table above, then compute:
- `max_tile_size = min(max(hw_limits), dim_size)`.
- `min_tile_size = min(hw_limits, dim_size)`.

Inter-op buffers store the dimension as `(max_tile_size / min_tile_size, min_tile_size)` — data is laid out at the smallest granularity so that every op can read at its own tile size.

### 4.2 Interleaving Groups

Different ops may have different tile size limits on the same dimension. For example, `nc_matmul` can process N up to 512 elements, while `nc_transpose` is fixed at 128. The buffer stores data at `min_tile_size` granularity — the smallest op limit on that dimension. Ops that work at a larger tile size consume multiple buffer slots per iteration, grouped together as one **interleave group**. This lets all ops share a single buffer layout despite different tile size requirements.

Each op computes a per-op tile size for each dimension it touches: `op_tile_size = min(max_tile_size, hw_limit)`. From this, two quantities determine how the op reads from the interleaved buffer:
- `num_interleave_groups = max_tile_size / op_tile_size` — sub-loop iterations per unified tile (the `i_interleave_group_d{id}` loop).
- `tiles_per_interleave_group` = `op_tile_size / min_tile_size` — buffer slots consumed per interleave group iteration.

Per-op reshape:
```python
for i_interleave_group_d{id} in range(max_tile_size/op_tile_size):
    (op_tile_size/min_tile_size, min_tile_size).reshape((1, op_tile_size))
```

### 4.3 Rendering NKI Source from KernelIR

`render_ir(ir)` converts a `KernelIR` to a complete NKI kernel source string. For the base IR, each op sits in its own singleton fusion group (`[[0], [1], ...]`), so each gets its own self-contained code block: buffer allocations, loop nest, DMA loads, ISA call, and writeback. The renderer iterates over fusion groups in order and concatenates their output.

When transforms modify the `KernelIR` (changing fusion groups, loop order, etc.), `render_ir` produces the transformed kernel — fused groups emit shared loop nests, reordered dimensions change loop nesting, and adjusted `tpb_hbm` changes block/tile loop bounds. The same `render_ir` handles both base and transformed IRs.

### 4.4 Example: Double Matmul

**Naming conventions:** loop variables `i_block_d{id}`, `i_tile_d{id}`, `i_interleave_group_d{id}`; buffers `sbuf_{name}` (SBUF), `psum_{name}` (PSUM).

Double matmul `(Q @ K.T) @ V` with `Q: (2048, 128)`, `K: (2048, 128)`, `V: (2048, 128)`:

```python
Q_t = NKITranspose()(data=Q)
K_t = NKITranspose()(data=K)
S = NKIMatmul()(stationary=Q_t, moving=K_t)
S_t = NKITranspose()(data=S)
output = NKIMatmul()(stationary=S_t, moving=V)
```

**Dimension analysis** (§3):

1. **`nc_transpose(Q)`** — P=d0, F=d1. **Q → (d0, d1)**. **Q_t → (d1, d0)**.
2. **`nc_transpose(K)`** — P=d2, F=d3. **K → (d2, d3)**. **K_t → (d3, d2)**.
3. **`nc_matmul(Q_t, K_t)`** — K=d1, M=d0, K=d3 → **unify d3→d1**. N=d2. **S → (d0, d2)**.
4. **`nc_transpose(S)`** — P=d0, F=d2. **S_t → (d2, d0)**.
5. **`nc_matmul(S_t, V)`** — K=d2, M=d0. V has no dim_ids; K=d2 in map, allocate N=d4. **V → (d2, d4)**. **output → (d0, d4)**.

Dimensions: d0=seq_q (2048), d1=d_k (128), d2=seq_k (2048), d4=d_v (128).

**Tile sizes** (§4.1). d2 is touched by matmul as N (limit 512) and by transpose as F (limit 128), producing different hardware limits:

| Dim | dim_size | max_tile_size | min_tile_size |
|---|---|---|---|
| d0 (seq_q) | 2048 | 128 | 128 |
| d1 (d_k) | 128 | 128 | 128 |
| d2 (seq_k) | 2048 | 512 | 128 |
| d4 (d_v) | 128 | 128 | 128 |

**Interleave on d2** (§4.2). Buffer stores `(4, 128)`. Per-op breakdown:

| Op on d2 | hw_limit | op_tile_size | num_interleave_groups | tiles_per_interleave_group |
|---|---|---|---|---|
| nc_matmul | 512 | 512 | 1 | 4 |
| nc_transpose | 128 | 128 | 4 | 1 |

Matmul reshapes `(4, 128)` → `(1, 512)` and processes in 1 group. Transpose reads one `(1, 128)` slot per group across 4 groups.

**Transpose codegen.** Fixed 128 × 128 tile. Reads from SBUF, writes to a degree-1 PSUM temp `(128, 1, 1, 128)`, then `nisa.tensor_copy` to the output SBUF. The output buffer swaps P and F axes.

**Matmul codegen.** Accumulates into PSUM in fp32 (memset to 0 before the K loop). After all K iterations, `nisa.tensor_copy` moves the result to SBUF.

**PSUM spill scheduling.** Both `nc_matmul` and `nc_transpose` produce results in PSUM and require `nisa.tensor_copy` to move them to SBUF (§2). For `nc_transpose`, each tile produces an independent PSUM result that is spilled immediately. For `nc_matmul` accumulating across a reduction dimension, multiple partial products pile up in the same PSUM tile — `tpb_psum` (§2) controls how many iterations accumulate before the spill. In the reference attention kernel, `tpb_psum = 4` on d2 gives a "large tile" of 4 × 512 = 2048 positions (16 V-tile matmuls) per PSUM batch, with 4 batches per section.

## 5. Programmatic Transforms

Programmatic transforms are mechanical rearrangements of the loop nest that do not change the math — they only change how and when ops execute. All transforms must respect the topological computation order: every op's inputs must be computed before the op runs. The math function (§1) defines one valid topological order, and the initial KernelIR (§4) follows it exactly. Transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

Transforms operate on the `KernelIR` (§4). Each transform produces a new `KernelIR` with modified transform state (e.g., different `fusion_groups`) while sharing immutable state (`ops`, `per_op_maps`, `ctx`).

Each transform subclass implements `candidates(ir) → list[KernelIR]`, returning every possible single-step application of that transform. Each candidate is a clone with the relevant state modified.

```python
class Transform(ABC):
    NAME: ClassVar[str] = ""

    @abstractmethod
    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        ...
```

Transforms compose: applying loop fusion to the base IR produces a set of variants; applying loop reordering to any of those produces further variants. The search (§7) explores this space by randomly walking the graph of transform applications.

Four transforms generate the search space.

### 5.1 Loop Fusion

See [loop_fusion.md](nkigym/src/nkigym/transforms/loop_fusion.md).

### 5.2 Load Placement

See [load_placement.md](nkigym/src/nkigym/transforms/load_placement.md).

### 5.3 Loop Reordering

See [loop_reordering.md](nkigym/src/nkigym/transforms/loop_reordering.md).

### 5.4 Tiles Per Block

See [tiles_per_block.md](nkigym/src/nkigym/transforms/tiles_per_block.md).

### 5.5 Multi-Buffer

See [multi_buffer.md](nkigym/src/nkigym/transforms/multi_buffer.md).

## 6. Math Transforms

See [online_fusion.md](nkigym/src/nkigym/transforms/online_fusion.md).

## 7. Search Interface

The search finds high-performing kernel variants by randomly exploring the transform graph and benchmarking each variant on hardware.

### 7.1 Transform Graph

The transform graph is a directed graph where **nodes** are kernel variants (`KernelIR` + rendered source) and **edges** are transform applications. Different transform paths can converge on the same node (same rendered source), so the graph is not necessarily a tree or DAG. It starts with one node — the base kernel from `build_ir()` — and grows by randomly picking a node, picking an applicable transform, and applying it to produce a new child node.

```
Node 0 (base)  ──[loop_fusion(ops 2,3)]──>  Node 1
Node 0         ──[loop_reorder(d2>d0)]──>    Node 2
Node 1         ──[loop_reorder(d2>d0)]──>    Node 3
Node 2         ──[loop_fusion(ops 2,3)]──>   Node 3  (deduplicated)
```

Expansion runs until the graph reaches `num_variants` nodes or no node has applicable transforms left. Duplicate kernels (same rendered source) are rejected, so different transform paths that converge on the same kernel collapse into one node.

```python
class TransformGraph:
    nodes: list[SearchNode]   """SearchNode = (ir, source)"""
    edges: list[SearchEdge]   """SearchEdge = (transform_name, parent_idx, child_idx)"""

    def expand(self, num_variants, render_fn, rng): ...
```

### 7.2 remote_search

`remote_search` wraps `remote_profile` with transform graph expansion:

```python
ir = build_ir(double_matmul_nkigym, input_specs)

results = remote_search(
    initial_kernel=ir,
    golden_source=golden_source,
    golden_func_name="double_matmul_numpy",
    hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
    cache_dir="/home/ubuntu/cache/double_matmul_search",
    num_variants=100,
    atol=0.5,
    rtol=0.1,
    warmup=10,
    iters=100,
)
```

Internally:

1. Render the base IR to get the initial kernel source (node 0).
2. Build a `TransformGraph` with all registered transforms (default: `[LoopFusion()]`).
3. Randomly expand to `num_variants` nodes.
4. Render each node's IR to source.
5. Submit every variant as a `KernelJob` to `remote_profile`.
6. Return the `ProfileOutput` with timing and correctness results for all variants.

## 8. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel ([`attention_cte.py`](/home/ubuntu/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py)) already uses `nl.ds` for this purpose.

- **Data layout transforms.** The current search space treats data layout as fixed — each tensor's partition and free axes are determined by the math function and never change. A data layout transform pass would manipulate `nc_transpose` ops to improve memory access patterns and engine utilization. The key moves are: (1) **insert dummy transpose pairs** — add a transpose before and after any tensor access point (the pair is a no-op); (2) **cancel adjacent transposes** — two consecutive transposes on the same tensor annihilate; (3) **move transposes** — slide a transpose earlier or later in the graph, past compatible ops, to find a more profitable placement; and (4) **merge transpose with DMA** — when a transpose is adjacent to an HBM load/store, replace the `nc_transpose` + `nisa.dma_copy` pair with a single transposing DMA (`nisa.dma_copy` with transposed source/destination layout), eliminating the Tensor Engine transpose entirely.
