---
title: "NKIGym Design: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-25"
geometry: margin=1in
---

# NKIGym Design

## 1. Logical Computation
**Running example**: causal single-head attention `softmax(mask(scale * Q @ K^T)) @ V`. Inputs use standard ML layout `(seq, hidden)`:

- `q: float16[4096, 128]` — `(seq_q, d_k)`
- `k: float16[4096, 128]` — `(seq_k, d_k)`
- `v: float16[4096, 128]` — `(seq_k, d_v)`
- output: `float16[4096, 128]` — `(seq_q, d_v)`

The math function defines what to compute — no loads/stores, no tiling, no memory hierarchy. Each `GymXXX` call maps 1:1 to a real `nisa.*` ISA instruction, but the math function only specifies the logical data flow.

```python
def attention(Q, K, V, scale):
    Q_t = GymTranspose()(Q)
    K_t = GymTranspose()(K)
    S = GymMatmul()(Q_t, K_t)
    masked_S = GymAffineSelect()(
        pattern=[[-1, S.shape[1]]], channel_multiplier=1,
        on_true_tile=S, on_false_value=-np.inf,
        cmp_op="greater_equal")
    scaled_S = GymTensorScalar()(masked_S, op0="multiply",
                                     operand0=scale)
    neg_max_S = GymTensorReduce()(op="max", data=scaled_S,
                                      axis=1, negate=True)
    exp_S, sum_exp = GymActivationReduce()(
        op="exp", data=scaled_S, reduce_op="add",
        bias=neg_max_S)
    inv_sum = GymActivation()(op="reciprocal", data=sum_exp)
    exp_S_t = nkigym.nc_transpose()(exp_S)
    attn = GymMatmul()(exp_S_t, V)
    output = GymTensorScalar()(attn, op0="multiply",operand0=inv_sum)
    return output
```

---

## 2 NKIGym Operators

Every `GymXXX` op mirrors a real `nisa.*` or `nl.*` ISA operation — no ops are invented. At the logical level, an op only declares its axis semantics and a CPU simulation. No hardware constraints, tiling, or memory placement.

```python
class GymOp:
    """Logical NKIGym operator — axis semantics and CPU simulation only."""
    NAME: str
    OPERAND_AXES: dict[str, tuple[str, ...]]
    OUTPUT_AXES: dict[str, tuple[str, str, ...]]

    @abstractmethod
    def __call__(self, **kwargs):
        """CPU simulation using Numpy in default float64.
        Takes input arrays + config, returns output array(s).
        """
```

### 2.1 Operator Subclasses

```python
class GymMatmul(GymOp):
    """stationary(K, M).T @ moving(K, N) → output(M, N).
    K is the contraction (accumulation) axis.
    """

    NAME = "nc_matmul"
    OPERAND_AXES = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES = {"output": ("M", "N")}

    def __call__(self, stationary, moving):
        return stationary.T @ moving


class GymTranspose(GymOp):
    """data(P, F) → output(F, P). Real hardware op, not a free view."""

    NAME = "nc_transpose"
    OPERAND_AXES = {"data": ("P", "F")}
    OUTPUT_AXES = {"output": ("F", "P")}

    def __call__(self, data):
        return data.T


class GymTensorScalar(GymOp):
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


class GymAffineSelect(GymOp):
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


class GymTensorReduce(GymOp):
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


class GymActivationReduce(GymOp):
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


class GymActivation(GymOp):
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

### 2.2 CPU Simulation

The math function is plain Python — each `GymXXX` call dispatches to `GymOp.__call__()` which executes the op with numpy at float64 precision. No parsing or IR needed; just call the function directly:

```python
q = np.random.randn(4096, 128)
k = np.random.randn(4096, 128)
v = np.random.randn(4096, 128)
output = attention(q, k, v, scale=1.0 / np.sqrt(128))
```

The result is the **reference output** — a float64 array that any correctly rendered and compiled NKI kernel must match (within hardware precision tolerance).

---

## 3. Dimension Analysis

A forward pass over all ops assigns concrete dimension IDs and computes the hardware-determined components (`op{int}_d{int}_unified_tile_size` and `d{int}_unified_tile_size`) for each dimension. No tensor starts with dimension IDs — they are discovered automatically from op structure.

For each op, build a local axis map from abstract axes (K, M, N or P, F) to concrete dimension IDs (d0, d1, ...). Operands that already carry dim_ids (from prior ops) provide the mapping; operands without dim_ids get fresh IDs allocated. When two operands of the same op share an abstract axis (e.g., matmul's K axis appears in both stationary and moving), their concrete dims are **unified** — the later operand's dim is renamed to match the earlier one, propagating backward through all tensors that share the old ID. This is how shared dimensions are discovered without any manual labeling.

### Example

```python
Q_t = GymTranspose()(Q)
K_t = GymTranspose()(K)
S = GymMatmul()(Q_t, K_t)
```

Walk through the forward pass. Inputs Q, K, V have no dim_ids — the pass discovers them:

1. **`GymTranspose(Q)`** — Q has no dim_ids. Allocate P=d0, F=d1. Assign **Q → (d0, d1)**. `OUTPUT_AXES = ("F", "P")` → **Q_t gets (d1, d0)**.

2. **`GymTranspose(K)`** — K has no dim_ids. Allocate P=d2, F=d3. Assign **K → (d2, d3)**. Output → **K_t gets (d3, d2)**.

3. **`GymMatmul(Q_t, K_t)`** — Q_t has (d1, d0) → K=d1, M=d0. K_t has (d3, d2) → K=d3, N=d2. Abstract axis K maps to both d1 and d3 — **unify d3→d1**. This renames K's dim_ids from (d2, d3) to (d2, d1), and K_t from (d3, d2) to (d1, d2). Now K=d1 is consistent across both operands. `OUTPUT_AXES = ("M", "N")` → **S gets (d0, d2)**.

---

## 4. Tensor Tiling

Every on-chip tensor uses a uniform **4D layout**: `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)`. This is the central design decision — loop structure, buffer sizing, and all transforms operate on this single shape.

**Hardware constraint.** Trainium DMA cannot operate on tensors with more than 5 meaningful (non-singleton) dimensions.

### 4.1 Memory Hierarchy and Boundaries

Trainium has three memory levels: HBM (off-chip) → SBUF (on-chip scratchpad) → PSUM (accumulator registers). Each boundary between adjacent levels introduces a block/tile tiling parameter that controls arithmetic intensity. Two ops produce results in PSUM: `nc_matmul` and `nc_transpose`.

Each op decides which boundary it triggers by examining its own inputs — an entirely localized decision. An op triggers at most one boundary:

- **HBM↔SBUF boundary.** Triggered when the op's input lives in HBM. Introduces `tpb_hbm` (tunable) and `num_blocks_hbm` (derived). DMA loads `tpb_hbm` unified tiles per block; the `load_tensor_block` / `save_tensor_block` gadgets from `nkigym.gadgets` handle the tile-by-tile DMA internally.

- **SBUF↔PSUM boundary.** Triggered when the op's input lives in PSUM but the op needs it in SBUF. The op inserts a `tensor_copy(PSUM → SBUF)` before consuming. Introduces `tpb_sbuf` (tunable) and `num_blocks_sbuf` (derived).

### 4.2 Interleave
Different ops may have different tile size limits on the same dimension (e.g., `nc_matmul` N=512 vs `nc_transpose` F=128). `d{int}_unified_tile_size` = max of all op tile sizes and is per-dimension across all ops. Ops with smaller tile sizes iterate `tiles_per_unified_tile = d{int}_unified_tile_size / d{int}_op{int}_tile_size` times per unified tile.

### 4.3 Dimension Decomposition
When the HBM-SBUF boundary is active on the tensor for an operator:
$$\texttt{dim\_size} = \texttt{num\_blocks\_hbm} \times \texttt{tpb\_hbm} \times \texttt{d\{int\}\_tile\_size}$$

When the SBUF-PSUM boundary is active on the tensor for an operator:
$$\texttt{dim\_size} = \texttt{num\_blocks\_sbuf} \times \texttt{tpb\_sbuf} \times \texttt{d\{int\}\_tile\_size}$$

In both cases, `d{int}_unified_tile_size` is further broken down into `tiles_per_unified_tile` * `d{int}_op{int}_tile_size`.

### 4.4 Per-Op Loop Nest
For each dimension of an op's tensor operands, either P or F:
- `d{int}_unified_tile_size` (hardware, per-dimension) — max of all hardware tile size limits across ops on the dimension. One iteration of block/tile loops processes `d{int}_unified_tile_size` elements.
- `d{int}_op{int}_tile_size` (hardware, per-dimension per-op) - hardware tile size for a particular dimension for an op.
- `tiles_per_unified_tile` (hardware, per-dimension per-op) — `d{int}_unified_tile_size / d{int}_op{int}_tile_size`. The buffer stores at the smallest op tile size; ops with larger tiles consume multiple buffer slots per iteration.
- `tpb_hbm` (tunable, where applicable) — unified tiles per HBM block. Data is loaded from HBM once per block via `load_tensor_block`, then reused across compute iterations.
- `tpb_sbuf` (tunable, where applicable) — unified tiles per PSUM→SBUF staging batch. When an op finds its input in PSUM, it stages `tpb_sbuf` tiles to SBUF via `tensor_copy` before consuming them.
- `num_blocks_hbm` (derived) — `dim_size / (tpb_hbm × d{int}_unified_tile_size)`.
- `num_blocks_sbuf` (derived) — `dim_size / (tpb_sbuf × d{int}_unified_tile_size)`.

Each op contributes 3 loops per dimension: block, tile, and interleave. Loops are grouped by **phase** — all block loops outermost, then all tile loops, then all interleave loops. Within each phase, dimensions follow the loop order. This per-phase grouping means block loops define the data boundary, the load happens once per block combination, and tile/ig loops iterate within the loaded data.

| Phase | Loop Variable | Trip count | Controls |
|---|---|---|---|
| Block | `i_block_d{id}` | `num_blocks` | Data boundary — DMA loads here |
| Tile | `i_tile_d{id}` | `tpb` | Tiles within a block |
| Interleave | `i_ig_d{id}` | `tiles_per_unified_tile` | Per-op sub-tile iteration |

### Example

```python
Q_t = GymTranspose()(Q)
K_t = GymTranspose()(K)
S = GymMatmul()(Q_t, K_t)
```
Using the dimension assignments from §3 and ISA tile limits (`nc_transpose`: P=128, F=128; `nc_matmul`: K=128, M=128, N=512). Suppose d0=4096, d1=256, d2=4096.

```python
dim_size = {"d0": 4096, "d1": 256, "d2": 4096}

"""Unified tile size: max of all op tile sizes on each dimension."""
unified = {
    "d0": max(128, 128),  # transpose P, matmul M → 128
    "d1": max(128, 128),  # transpose F, matmul K → 128
    "d2": max(128, 512),  # transpose P, matmul N → 512
}

"""Per-op tile size: hardware limit for this op on this dimension."""
op_tile = {
    (0, "d0"): 128,  # transpose P
    (0, "d1"): 128,  # transpose F
    (1, "d2"): 128,  # transpose P
    (1, "d1"): 128,  # transpose F
    (2, "d1"): 128,  # matmul K
    (2, "d0"): 128,  # matmul M
    (2, "d2"): 512,  # matmul N
}

interleave = {k: unified[k[1]] // v for k, v in op_tile.items()}
# (1,"d2") → 512//128 = **4**, all others → 1

"""Boundary type per op — determined by where the input lives."""
boundary = {0: "hbm", 1: "hbm", 2: "sbuf"}
# Q,K in HBM → HBM↔SBUF; Q_t,K_t in PSUM → SBUF↔PSUM

"""Tunable tiles-per-block (one per boundary type)."""
tpb = {0: tpb_hbm, 1: tpb_hbm, 2: tpb_sbuf}

num_blocks = {
    k: dim_size[k[1]] // (tpb[k[0]] * unified[k[1]])
    for k in op_tile
}
```

**Per-op loop nests (per-phase grouping).** Block phase → tile phase → ig phase. Within each phase, dimensions follow the loop order.

- **op0** `transpose(Q)` [HBM↔SBUF], order (d0, d1):
  blocks: `i_block_d0` × `i_block_d1` → load
  tiles: `i_tile_d0` × `i_tile_d1`
  ig: `i_ig_d0`(1) × `i_ig_d1`(1)

- **op1** `transpose(K)` [HBM↔SBUF], order (d2, d1):
  blocks: `i_block_d2` × `i_block_d1` → load
  tiles: `i_tile_d2` × `i_tile_d1`
  ig: `i_ig_d2`(**4**) × `i_ig_d1`(1)

- **op2** `matmul` [SBUF↔PSUM], order (d0, d1, d2):
  blocks: `i_block_d0` × `i_block_d1` × `i_block_d2` → load/stage
  tiles: `i_tile_d0` × `i_tile_d1` × `i_tile_d2`
  ig: `i_ig_d0`(1) × `i_ig_d1`(1) × `i_ig_d2`(1)

The interleave trip of **4** in op1/d2 is the key asymmetry: the matmul forces `unified["d2"]`=512, but transpose handles only 128 elements on P per iteration.

## 5. KernelIR and Lowering

The pipeline: **math function →** `build_ir` **→ KernelIR → online fusion (§7) → KernelIR → programmatic transforms (§6) → KernelIR →** `render_ir` **→ NKI source → test + profile.** `KernelIR` is the structured representation that all transforms operate on, avoiding repeated AST parsing of NKI source for every variant. `build_ir(func, input_specs)` constructs the initial IR once (dimension analysis §3, tiling §4, default transform state). Online fusion (§7) greedily applies all detected math-level optimizations, producing a single KernelIR with blocking barriers eliminated. Programmatic transforms (§6) then clone and modify the transform state to produce variant candidates — all within `KernelIR`, no source generation yet. `render_ir(ir)` mechanically lowers any `KernelIR` — initial or transformed — to NKI source.

### 5.1 Data Model

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


@dataclass
class OpInfo:
    """Node in the computation DAG."""
    op_type: str
    operands: dict[str, str]    # role → tensor_name
    output: str                 # output tensor name
    dim_map: dict[str, str]     # abstract_axis → dim_id
    per_dim: dict[str, OpDimInfo]
    predecessors: list[int]     # indices of ops that produce this op's inputs


@dataclass
class OpGraph:
    """Computation DAG — ops in topological order with explicit edges."""
    nodes: list[OpInfo]         # topological order, indexed by op_idx
    tensor_producers: dict[str, int]
    """
    tensor_name → op_idx that produces it.
    Absent for kernel inputs (HBM).
    Inverse of OpInfo.output — together with predecessors,
    gives both forward and backward traversal of the DAG.
    """


@dataclass
class KernelIR:
    """Complete representation for lowering to NKI source."""
    func_name: str
    param_names: list[str]
    return_name: str
    dims: dict[str, DimInfo]
    tensors: dict[str, TensorInfo]
    op_graph: OpGraph

    """Transform state (mutable — modified by §6 transforms)."""
    fusion_groups: list[list[int]]
    tiles_per_block: dict[tuple[int, str], int]
    buffer_degrees: dict[tuple[int, str, str], int]
    loop_order: list[list[str]]
    load_placements: dict[tuple[str, str], str]
```

**Immutable state** — produced by `build_ir`, shared across all transform variants:
- `dims`, `tensors`, `op_graph`: full results of §3 dimension analysis and §4 tiling. The `OpGraph` stores ops in topological order with explicit predecessor edges and a `tensor_producers` map for reverse lookup.
- Signature fields: `func_name`, `param_names`, `return_name`.

**Transform state** — starts at defaults, modified by transforms (§6):
- `fusion_groups`: initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: initially `1` for all `(op_idx, dim_id)` pairs.
- `buffer_degrees`: initially `1` for all `(group_idx, tensor_name, dim_id)` triples. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — non-blocking dims first, blocking (accumulation) dims last.
- `load_placements`: initially absent (all loads at default position). `(tensor_name, dim_id) → tier` where tier is `"per_tile"`, `"per_block"`, or `"full"`. See §6.2.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the transform state above:
- **memset**: before the accumulation (K) dimension's outermost loop in the current `loop_order`.
- **save / tensor_copy(psum→sbuf)**: after the accumulation dimension's last inner loop.
- **tensor_copy (interleave reload/save)**: around the tile loop when an accumulating dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.

### 5.2 Generic Lowering

`render_ir(ir)` mechanically converts any `KernelIR` — initial or transformed — to NKI source. The same renderer handles all variants by reading the transform state and applying dependency-based statement placement.

**Kernel frame.** Emit `@nki.jit` decorator, `def func_name(params):`, and the HBM output `nl.ndarray(..., buffer=nl.shared_hbm)` for the return tensor.

**Per-group codegen.** For each fusion group in topological order:

1. **Loop nest.** Each dimension contributes 3 loops (block, tile, ig). Nesting follows `loop_order`. All loops always explicit, even when trip count is 1.

   | Loop | Variable | Trip count |
   |---|---|---|
   | Block | `i_block_d{id}` | `dim_size / (tpb × unified_tile_size)` |
   | Tile | `i_tile_d{id}` | `tiles_per_block[(op, d)]` |
   | Interleave | `i_ig_d{id}` | `num_ig` from `OpDimInfo` |

2. **Buffers.** Sizes derived from `tiles_per_block`, `buffer_degrees`, `load_placements`, and dim info:
   - *HBM load buffer* (SBUF): size per relevant dimension from `load_placements` tier — absent = 1 tile, `"per_tile"` = ig tiles, `"per_block"` = tpb × ig tiles, `"full"` = all tiles.
   - *PSUM temp*: degree-1 shape `(op_tile_P, 1, 1, op_tile_F)`.
   - *Cross-group SBUF*: full-range buffer for intermediate tensors consumed by later groups.
   - *Interleave sbuf_output*: when an accumulating dim's block and tile are split by other dims, persists partial sums across block iterations.

3. **DMA load.** Position from `load_placements`. `load_tensor_block` gadget handles tile-by-tile DMA internally.

4. **PSUM init / staging.** Positions derived from `loop_order` + accumulation dimensions (RAW/WAW):
   - `nc_matmul`: `memset` before K's outermost loop. After K's last iteration: `save_tensor_block` (return tensor) or `tensor_copy` (intermediates).
   - `nc_transpose`: `tensor_copy(psum→sbuf)` immediately after each ISA call.
   - Block-tile split on accumulating dim: `tensor_copy` reload before tile loop, save after.

5. **ISA call.** Innermost loop level. Buffer tile index: $idx = i_{block} \times (tpb \times num\_ig) + i_{tile} \times num\_ig + i_{ig}$. Interleave reshape `(tiles_per_ig, min_tile)` → `(1, op_tile)` is a free view.

6. **DMA store.** `save_tensor_block` after K's last iteration. Handles PSUM→SBUF staging internally.

### 5.3 Initial KernelIR Conventions

`build_ir` produces the initial KernelIR with these defaults — the starting point before any transforms (§6):

- **`tiles_per_block = 1`** for all (op, dim). Block loops carry the full tile count; tile loops are `range(1)`.
- **`load_placements = {}`** (absent). All loads after the block phase, before tile phase.
- **`buffer_degrees = 1`** for all (group, tensor, dim). Single-buffered.
- **`loop_order`**: per-phase grouping (§4.4) — blocks outermost, tiles middle, igs inner. Non-blocking dims first, blocking dims last.
- **`fusion_groups = [[0], [1], ...]`** — each op in its own group. Intermediate tensors fully materialized in cross-group SBUF buffers.

### 5.4 Initial KernelIR Example

Transpose + matmul with interleave asymmetry on d2:

```python
def matmul(lhs_T, rhs_T):
    rhs = GymTranspose()(data=rhs_T)
    result = GymMatmul()(stationary=lhs_T, moving=rhs)
    return result
```

Input specs: `lhs_T: float16[128, 2048]` (d0, d1), `rhs_T: float16[8192, 128]` (d2, d0).

Dimensions from §3: d0=128, d1=2048, d2=8192. Tiling from §4:

| Dim | dim_size | unified | min |
|---|---|---|---|
| d0 | 128 | 128 | 128 |
| d1 | 2048 | 128 | 128 |
| d2 | 8192 | 512 | 128 |

**KernelIR** (initial, all `tiles_per_block = 1`):

```python
ir = KernelIR(
    func_name="matmul",
    param_names=["lhs_T", "rhs_T"],
    return_name="result",
    dims={
        "d0": DimInfo(128, 128, 128),
        "d1": DimInfo(2048, 128, 128),
        "d2": DimInfo(8192, 512, 128),
    },
    tensors={
        "lhs_T":  TensorInfo(("d0", "d1"), (128, 2048), "float16", "hbm"),
        "rhs_T":  TensorInfo(("d2", "d0"), (8192, 128), "float16", "hbm"),
        "rhs":    TensorInfo(("d0", "d2"), (128, 8192), "float16", "psum"),
        "result": TensorInfo(("d1", "d2"), (2048, 8192), "float16", "psum"),
    },
    op_graph=OpGraph(
        nodes=[
            OpInfo("nc_transpose", {"data": "rhs_T"}, "rhs",
                   {"P": "d2", "F": "d0"},
                   {"d2": OpDimInfo(128, 4, 1), "d0": OpDimInfo(128, 1, 1)},
                   predecessors=[]),
            OpInfo("nc_matmul", {"stationary": "lhs_T", "moving": "rhs"}, "result",
                   {"K": "d0", "M": "d1", "N": "d2"},
                   {"d0": OpDimInfo(128, 1, 1), "d1": OpDimInfo(128, 1, 1),
                    "d2": OpDimInfo(512, 1, 4)},
                   predecessors=[0]),
        ],
        tensor_producers={"rhs": 0, "result": 1},
    ),
    fusion_groups=[[0], [1]],
    tiles_per_block={
        (0, "d2"): 1, (0, "d0"): 1,
        (1, "d0"): 1, (1, "d1"): 1, (1, "d2"): 1,
    },
    buffer_degrees={
        (0, "rhs", "d0"): 1, (0, "rhs", "d2"): 1,
        (1, "result", "d1"): 1, (1, "result", "d2"): 1,
    },
    loop_order=[["d2", "d0"], ["d1", "d2", "d0"]],
    load_placements={},
)
```

**Derived loop trip counts** from §4.4 formulas:

Op 0 (transpose), loop order d2 → d0:

| Dim | num_blocks | tpb | ig |
|---|---|---|---|
| d2 | 16 | 1 | 4 |
| d0 | 1 | 1 | 1 |

Op 1 (matmul), loop order d1 → d2 → d0(K):

| Dim | num_blocks | tpb | ig |
|---|---|---|---|
| d1 | 16 | 1 | 1 |
| d2 | 16 | 1 | 1 |
| d0 (K) | 1 | 1 | 1 |

**Buffer sizing.** Each buffer follows the 4D layout `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)`:

| Buffer | Role | Shape | Size |
|---|---|---|---|
| `sbuf_rhs_T` | HBM load for rhs_T | `(128, 4, 1, 128)` | P=d2: tpb(1)×ig(4)=4 tiles. F=d0: 1 tile. |
| `psum_rhs_temp` | PSUM temp for transpose | `(128, 1, 1, 128)` | Degree-1. One 128×128 transpose tile. |
| `sbuf_rhs` | Cross-op output for rhs | `(128, 1, 64, 128)` | Full-range at min tile size per §4. P=d0: 1 tile. F=d2: 8192/128=64 tiles. Matmul reshapes `(4, 128)` → `(1, 512)` (free view, no copy). |
| `sbuf_lhs_T` | HBM load for lhs_T | `(128, 1, 1, 128)` | K=d0: 1 tile. M=d1: 1 tile. |
| `psum_result` | PSUM accum for matmul | `(128, 1, 1, 512)` | Degree-1. One 128×512 matmul tile (fp32). `save_tensor_block` handles PSUM→SBUF→HBM internally. |

**Lowered NKI kernel** — one self-contained loop nest per op:

```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def matmul(lhs_T, rhs_T):
    result = nl.ndarray((2048, 8192), dtype=nl.float16, buffer=nl.shared_hbm)

    """ === Op 0: nc_transpose(rhs_T) → rhs === """
    sbuf_rhs_T = nl.ndarray((128, 4, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_rhs_temp = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.psum)
    sbuf_rhs = nl.ndarray((128, 1, 64, 128), dtype=nl.float16, buffer=nl.sbuf)
    sbuf_rhs_op1 = sbuf_rhs.reshape((128, 1, 16, 512))

    for i_block_d2 in range(16):
        for i_block_d0 in range(1):
            load_tensor_block(sbuf_rhs_T, rhs_T,
                              par_ofs=i_block_d2 * 512, free_ofs=i_block_d0 * 128)
            for i_tile_d2 in range(1):
                for i_tile_d0 in range(1):
                    for i_ig_d2 in range(4):
                        for i_ig_d0 in range(1):
                            ld2 = i_tile_d2 * 4 + i_ig_d2
                            td0 = i_tile_d0 * 1 + i_ig_d0
                            gd2 = i_block_d2 * 4 + ld2
                            nisa.nc_transpose(psum_rhs_temp[0:128, td0, 0, 0:128],
                                              sbuf_rhs_T[0:128, ld2, td0, 0:128])
                            nisa.tensor_copy(sbuf_rhs[0:128, td0, gd2, 0:128],
                                             psum_rhs_temp[0:128, td0, 0, 0:128])

    """ === Op 1: nc_matmul(lhs_T, rhs) → result === """
    sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_result = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)

    for i_block_d1 in range(16):
        for i_block_d2 in range(16):
            for i_block_d0 in range(1):
                load_tensor_block(sbuf_lhs_T, lhs_T,
                                  par_ofs=i_block_d0 * 128, free_ofs=i_block_d1 * 128)
                nisa.memset(psum_result[0:128, 0, 0, 0:512], value=0.0)
                for i_tile_d1 in range(1):
                    for i_tile_d2 in range(1):
                        for i_tile_d0 in range(1):
                            for i_ig_d1 in range(1):
                                for i_ig_d2 in range(1):
                                    for i_ig_d0 in range(1):
                                        td0 = i_tile_d0 * 1 + i_ig_d0
                                        td1 = i_tile_d1 * 1 + i_ig_d1
                                        d2_ut = i_block_d2 + i_tile_d2
                                        nisa.nc_matmul(
                                            psum_result[0:128, td1, i_ig_d2, 0:512],
                                            sbuf_lhs_T[0:128, td0, td1, 0:128],
                                            sbuf_rhs_op1[0:128, td0, d2_ut, 0:512])
                save_tensor_block(result, psum_result,
                                  par_ofs=i_block_d1 * 128, free_ofs=i_block_d2 * 512)

    return result
```

## 6. Programmatic Transforms

Programmatic transforms are mechanical rearrangements of the loop nest that do not change the math — they only change how and when ops execute. All transforms must respect the topological computation order: every op's inputs must be computed before the op runs. The math function (§1) defines one valid topological order, and the initial KernelIR (§5) follows it exactly. Transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

Transforms operate on the `KernelIR` (§5). Each transform produces a new `KernelIR` with modified transform state (e.g., different `fusion_groups`) while sharing immutable state (`dims`, `tensors`, `op_graph`).

Each transform subclass implements `candidates(ir) → list[KernelIR]`, returning every possible single-step application of that transform. Each candidate is a clone with the relevant state modified.

```python
class Transform(ABC):
    NAME: ClassVar[str] = ""

    @abstractmethod
    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        ...
```

Transforms compose: applying loop fusion to the base IR produces a set of variants; applying loop reordering to any of those produces further variants. The search (§8) explores this space by randomly walking the graph of transform applications.

Four transforms generate the search space.

### 6.1 Loop Fusion

See [loop_fusion.md](nkigym/src/nkigym/transforms/loop_fusion.md).

### 6.2 Load Placement

See [load_placement.md](nkigym/src/nkigym/transforms/load_placement.md).

### 6.3 Loop Reordering

See [loop_reordering.md](nkigym/src/nkigym/transforms/loop_reordering.md).

### 6.4 Tiles Per Block

See [tiles_per_block.md](nkigym/src/nkigym/transforms/tiles_per_block.md).

### 6.5 Multi-Buffer

See [multi_buffer.md](nkigym/src/nkigym/transforms/multi_buffer.md).

## 7. Online Fusion

Greedy math-level preprocessing — not part of the programmatic search space. Detects all X + Accumulation patterns, applies tile-level fusion to each, and produces a single KernelIR with blocking barriers eliminated. Programmatic transforms (§6) then operate on this already-fused IR. Block-level granularity emerges from programmatic transforms: tiles_per_block + dimension interleaving naturally create the section structure where corrections happen once per block.

See [online_fusion.md](nkigym/src/nkigym/online_fusion/online_fusion.md).

## 8. Search Interface

The search finds high-performing kernel variants by randomly exploring the transform graph and benchmarking each variant on hardware.

### 8.1 Transform Graph

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

### 8.2 remote_search

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

## 9. Future Work

- **Boundary handling for non-divisible input shapes.** The current guide assumes all input dimensions are exact multiples of their tile sizes. Real workloads often have ragged last tiles (e.g., a sequence length of 1000 with tile size 128 leaves a remainder of 104). Supporting this requires `nl.ds(offset, size)` dynamic slicing to emit variable-length last tiles, with masking or predication where needed. The reference attention kernel ([`attention_cte.py`](/home/ubuntu/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py)) already uses `nl.ds` for this purpose.

- **Data layout transforms.** The current search space treats data layout as fixed — each tensor's partition and free axes are determined by the math function and never change. A data layout transform pass would manipulate `nc_transpose` ops to improve memory access patterns and engine utilization. The key moves are: (1) **insert dummy transpose pairs** — add a transpose before and after any tensor access point (the pair is a no-op); (2) **cancel adjacent transposes** — two consecutive transposes on the same tensor annihilate; (3) **move transposes** — slide a transpose earlier or later in the graph, past compatible ops, to find a more profitable placement; and (4) **merge transpose with DMA** — when a transpose is adjacent to an HBM load/store, replace the `nc_transpose` + `nisa.dma_copy` pair with a single transposing DMA (`nisa.dma_copy` with transposed source/destination layout), eliminating the Tensor Engine transpose entirely.
