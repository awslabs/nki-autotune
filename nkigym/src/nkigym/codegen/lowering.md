## 1. Tensor Tiling

Every on-chip tensor uses a uniform **4D layout**: `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)`. This is the central design decision — loop structure, buffer sizing, and all transforms operate on this single shape.

**Hardware constraint.** Trainium DMA cannot operate on tensors with more than 5 meaningful (non-singleton) dimensions.

### 1.1 Memory Hierarchy and Boundaries

Trainium has three memory levels: HBM (off-chip) → SBUF (on-chip scratchpad) → PSUM (accumulator registers). Each boundary between adjacent levels introduces a block/tile tiling parameter that controls arithmetic intensity. Two ops produce results in PSUM: `nc_matmul` and `nc_transpose`.

Each op decides which boundary it triggers by examining its own inputs — an entirely localized decision. An op triggers at most one boundary:

- **HBM↔SBUF boundary.** Triggered when the op's input lives in HBM. Introduces `tpb_hbm` (tunable) and `num_blocks_hbm` (derived). DMA loads `tpb_hbm` unified tiles per block; the `load_tensor_block` / `save_tensor_block` gadgets from `nkigym.gadgets` handle the tile-by-tile DMA internally.

- **SBUF↔PSUM boundary.** Triggered when the op's input lives in PSUM but the op needs it in SBUF. The op inserts a `tensor_copy(PSUM → SBUF)` before consuming. Introduces `tpb_sbuf` (tunable) and `num_blocks_sbuf` (derived).

### 1.2 Interleave
Different ops may have different tile size limits on the same dimension (e.g., `nc_matmul` N=512 vs `nc_transpose` F=128). `d{int}_unified_tile_size` = max of all op tile sizes and is per-dimension across all ops. Ops with smaller tile sizes iterate `tiles_per_unified_tile = d{int}_unified_tile_size / d{int}_op{int}_tile_size` times per unified tile.

### 1.3 Dimension Decomposition
When the HBM-SBUF boundary is active on the tensor for an operator:
$$\texttt{dim\_size} = \texttt{num\_blocks\_hbm} \times \texttt{tpb\_hbm} \times \texttt{d\{int\}\_tile\_size}$$

When the SBUF-PSUM boundary is active on the tensor for an operator:
$$\texttt{dim\_size} = \texttt{num\_blocks\_sbuf} \times \texttt{tpb\_sbuf} \times \texttt{d\{int\}\_tile\_size}$$

In both cases, `d{int}_unified_tile_size` is further broken down into `tiles_per_unified_tile` * `d{int}_op{int}_tile_size`.

### 1.4 Per-Op Loop Nest
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

### 1.2 Tiling Example

```python
Q_t = GymTranspose()(Q)
K_t = GymTranspose()(K)
S = GymMatmul()(Q_t, K_t)
```
Using ISA tile limits (`nc_transpose`: P=128, F=128; `nc_matmul`: K=128, M=128, N=512). Suppose d0=4096, d1=256, d2=4096.

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

## 2. KernelIR and Lowering

The pipeline: **math function →** `build_ir` **→ KernelIR → online fusion → KernelIR → programmatic transforms → KernelIR →** `render_ir` **→ NKI source → test + profile.** `KernelIR` is the structured representation that all transforms operate on, avoiding repeated AST parsing of NKI source for every variant. `build_ir(func, input_specs)` constructs the initial IR once (dimension analysis, tiling, default transform state). Online fusion greedily applies all detected math-level optimizations, producing a single KernelIR with blocking barriers eliminated. Programmatic transforms then clone and modify the transform state to produce variant candidates — all within `KernelIR`, no source generation yet. `render_ir(ir)` mechanically lowers any `KernelIR` — initial or transformed — to NKI source.

### 2.1 Computation DAG

```python
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
```

**Immutable state** — produced by `build_ir`, shared across all transform variants:
- `dims`, `tensors`, `op_graph`: full results of dimension analysis and tiling. The `OpGraph` stores ops in topological order with explicit predecessor edges and a `tensor_producers` map for reverse lookup.
- Signature fields: `func_name`, `param_names`, `return_name`.

### 2.2 Generic Lowering

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
   - *PSUM accum* (matmul): holds all output tiles live between memset and save. Dims inside the K dimension's outermost loop contribute their full tile count; dims outside contribute 1. Shape depends on `loop_order`.
   - *Cross-group SBUF*: full-range buffer for intermediate tensors consumed by later groups.
   - *Interleave sbuf_output*: when an accumulating dim's block and tile are split by other dims, persists partial sums across block iterations.

3. **DMA load.** Position from `load_placements`. `load_tensor_block` gadget handles tile-by-tile DMA internally.

4. **PSUM init / staging.** Positions derived from `loop_order` + accumulation dimensions (RAW/WAW):
   - `nc_matmul`: `memset` before K's outermost loop. After K's last iteration: `save_tensor_block` (return tensor) or `tensor_copy` (intermediates).
   - `nc_transpose`: `tensor_copy(psum→sbuf)` immediately after each ISA call.
   - Block-tile split on accumulating dim: `tensor_copy` reload before tile loop, save after.

5. **ISA call.** Innermost loop level. Buffer tile index: $idx = i_{block} \times (tpb \times num\_ig) + i_{tile} \times num\_ig + i_{ig}$. Interleave reshape `(tiles_per_ig, min_tile)` → `(1, op_tile)` is a free view.

6. **DMA store.** `save_tensor_block` after K's last iteration. Handles PSUM→SBUF staging internally.

### 2.3 Initial KernelIR Conventions

`build_ir` produces the initial KernelIR with these defaults — the most naive, mechanical lowering with no optimization choices. Every field starts at its simplest value; transforms improve from here.

- **`tiles_per_block = 1`** for all (op, dim). Block loops carry the full tile count; tile loops are `range(1)`.
- **`load_placements = {}`** (absent). All loads after the block phase, before tile phase.
- **`buffer_degrees = 1`** for all (group, tensor, dim). Single-buffered.
- **`loop_order`**: per-phase grouping (§1.1) — blocks outermost, tiles middle, igs inner. Dimensions in order of first appearance (d0, d1, d2, ...).
- **`fusion_groups = [[0], [1], ...]`** — each op in its own group. Intermediate tensors fully materialized in cross-group SBUF buffers.

The initial `loop_order` uses dimension ID order because it is a neutral starting point that makes no assumptions about which dimension benefits from being outermost or innermost. Loop reordering transforms explore the space of valid orderings from here. When an accumulation dimension (K) appears early in the numbering, it ends up outermost, and the PSUM accumulator must hold all inner-dimension output tiles between memset and save. This can produce buffers too large for hardware PSUM — that is expected. The initial IR is not required to fit on hardware; transforms shrink buffers by reordering loops (moving K innermost) or adjusting `tiles_per_block`.

### 2.4 Initial KernelIR Examples

#### Single matmul

```python
def matmul(lhs_T, rhs):
    result = GymMatmul()(stationary=lhs_T, moving=rhs)
    return result
```

Input specs: `lhs_T: float16[8192, 8192]` (d0, d1), `rhs: float16[8192, 8192]` (d0, d2). Only one op, so unified = op tile size with no interleave asymmetry.

| Dim | dim_size | unified | min |
|---|---|---|---|
| d0 | 8192 | 128 | 128 |
| d1 | 8192 | 128 | 128 |
| d2 | 8192 | 512 | 512 |

**KernelIR** (initial, all `tiles_per_block = 1`):

```python
ir = KernelIR(
    func_name="matmul",
    param_names=["lhs_T", "rhs"],
    return_name="result",
    dims={
        "d0": DimInfo(8192, 128, 128),
        "d1": DimInfo(8192, 128, 128),
        "d2": DimInfo(8192, 512, 512),
    },
    tensors={
        "lhs_T":  TensorInfo(("d0", "d1"), (8192, 8192), "float16", "hbm"),
        "rhs":    TensorInfo(("d0", "d2"), (8192, 8192), "float16", "hbm"),
        "result": TensorInfo(("d1", "d2"), (8192, 8192), "float16", "psum"),
    },
    op_graph=OpGraph(
        nodes=[
            OpInfo("nc_matmul", {"stationary": "lhs_T", "moving": "rhs"}, "result",
                   {"K": "d0", "M": "d1", "N": "d2"},
                   {"d0": OpDimInfo(128, 1, 1), "d1": OpDimInfo(128, 1, 1),
                    "d2": OpDimInfo(512, 1, 1)},
                   predecessors=[]),
        ],
        tensor_producers={"result": 0},
    ),
    fusion_groups=[[0]],
    tiles_per_block={
        (0, "d0"): 1, (0, "d1"): 1, (0, "d2"): 1,
    },
    buffer_degrees={
        (0, "result", "d1"): 1, (0, "result", "d2"): 1,
    },
    loop_order=[["d0", "d1", "d2"]],
    load_placements={},
)
```

**Derived loop trip counts** — loop order d0(K) → d1 → d2:

| Dim | num_blocks | tpb | ig |
|---|---|---|---|
| d0 (K) | 64 | 1 | 1 |
| d1 | 64 | 1 | 1 |
| d2 | 16 | 1 | 1 |

**Buffer sizing.** All buffers use 4D layout `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)`.

- `sbuf_lhs_T` — HBM load for lhs_T(K=d0, M=d1). `load_placements` absent → buffer holds 1 unified tile per dim. Tile sizes at min granularity: `min_d0=128`, `min_d1=128`. Num tiles: `unified_d0/min_d0 = 128/128 = 1`, `unified_d1/min_d1 = 128/128 = 1`. Shape: **`(128, 1, 1, 128)`**.
- `sbuf_rhs` — HBM load for rhs(K=d0, N=d2). Same rule. `min_d0=128`, `min_d2=512`. Num tiles: `128/128 = 1`, `512/512 = 1`. Shape: **`(128, 1, 1, 512)`**.
- `psum_result` — PSUM accum for result(M=d1, N=d2). K=d0 is outermost in `loop_order` — all of d1 and d2 are inside K's loop, so both contribute full tile count. Op tile sizes: `op_M=128`, `op_N=512`. Num tiles: `8192/128 = 64`, `8192/512 = 16`. fp32. Shape: **`(128, 64, 16, 512)`**.

**Lowered NKI kernel:**

```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def matmul_kernel(lhs_T, rhs):
    assert lhs_T.shape == (8192, 8192)
    assert rhs.shape == (8192, 8192)
    result = nl.ndarray((8192, 8192), dtype=nl.float16, buffer=nl.shared_hbm)

    """ === Op 0: nc_matmul(lhs_T, rhs) → result === """
    sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=nl.float16, buffer=nl.sbuf)
    psum_result = nl.ndarray((128, 64, 16, 512), dtype=nl.float32, buffer=nl.psum)

    nisa.memset(psum_result[0:128, 0:64, 0:16, 0:512], value=0.0)
    for i_block_d0 in range(64):
        for i_block_d1 in range(64):
            for i_block_d2 in range(16):
                load_tensor_block(sbuf_lhs_T, lhs_T,
                                  par_ofs=i_block_d0 * 128, free_ofs=i_block_d1 * 128)
                load_tensor_block(sbuf_rhs, rhs,
                                  par_ofs=i_block_d0 * 128, free_ofs=i_block_d2 * 512)
                for i_tile_d0 in range(1):
                    for i_tile_d1 in range(1):
                        for i_tile_d2 in range(1):
                            for i_ig_d0 in range(1):
                                for i_ig_d1 in range(1):
                                    for i_ig_d2 in range(1):
                                        nisa.nc_matmul(
                                            psum_result[0:128, i_block_d1, i_block_d2, 0:512],
                                            sbuf_lhs_T[0:128, 0, 0, 0:128],
                                            sbuf_rhs[0:128, 0, 0, 0:512])
    save_tensor_block(result, psum_result, par_ofs=0, free_ofs=0)

    return result
```

Both HBM operands are reloaded inside the innermost block loop — lhs_T does not depend on d2 but is reloaded 16× per (d0, d1) combination. This is expected: the naive lowering places all loads at the block-tile boundary without hoisting. The load placement transform eliminates this redundancy.

#### Transpose + matmul with interleave asymmetry on d2

```python
def matmul(lhs_T, rhs_T):
    rhs = GymTranspose()(data=rhs_T)
    result = GymMatmul()(stationary=lhs_T, moving=rhs)
    return result
```

Input specs: `lhs_T: float16[8192, 8192]` (d0, d1), `rhs_T: float16[8192, 8192]` (d2, d0).

Dimensions: d0=8192, d1=8192, d2=8192. Tiling:

| Dim | dim_size | unified | min |
|---|---|---|---|
| d0 | 8192 | 128 | 128 |
| d1 | 8192 | 128 | 128 |
| d2 | 8192 | 512 | 128 |

**KernelIR** (initial, all `tiles_per_block = 1`):

```python
ir = KernelIR(
    func_name="matmul",
    param_names=["lhs_T", "rhs_T"],
    return_name="result",
    dims={
        "d0": DimInfo(8192, 128, 128),
        "d1": DimInfo(8192, 128, 128),
        "d2": DimInfo(8192, 512, 128),
    },
    tensors={
        "lhs_T":  TensorInfo(("d0", "d1"), (8192, 8192), "float16", "hbm"),
        "rhs_T":  TensorInfo(("d2", "d0"), (8192, 8192), "float16", "hbm"),
        "rhs":    TensorInfo(("d0", "d2"), (8192, 8192), "float16", "psum"),
        "result": TensorInfo(("d1", "d2"), (8192, 8192), "float16", "psum"),
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
    loop_order=[["d0", "d2"], ["d0", "d1", "d2"]],
    load_placements={},
)
```

**Derived loop trip counts** from §1.1 formulas:

Op 0 (transpose), loop order d0 → d2:

| Dim | num_blocks | tpb | ig |
|---|---|---|---|
| d0 | 64 | 1 | 1 |
| d2 | 16 | 1 | 4 |

Op 1 (matmul), loop order d0(K) → d1 → d2:

| Dim | num_blocks | tpb | ig |
|---|---|---|---|
| d0 (K) | 64 | 1 | 1 |
| d1 | 64 | 1 | 1 |
| d2 | 16 | 1 | 1 |

**Buffer sizing.** All buffers use 4D layout `(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)` at min tile size. PSUM buffers use op tile size.

Op 0 (transpose):

- `sbuf_rhs_T` — HBM load for rhs_T(P=d2, F=d0). `load_placements` absent → 1 unified tile per dim. `min_d2=128`, `min_d0=128`. Num tiles: `unified_d2/min_d2 = 512/128 = 4`, `unified_d0/min_d0 = 128/128 = 1`. Shape: **`(128, 4, 1, 128)`**. The 4 tiles span one unified tile of d2 (512 elements) at the 128-element min granularity.
- `psum_rhs_temp` — PSUM temp for transpose. Degree-1 at op tile: `op_P=128`, `op_F=128`. Shape: **`(128, 1, 1, 128)`**.
- `sbuf_rhs` — Cross-group SBUF for rhs(d0, d2). Full-range at min tile: `min_d0=128`, `min_d2=128`. Num tiles: `8192/128 = 64`, `8192/128 = 64`. Shape: **`(128, 64, 64, 128)`**. Matmul consumes this via reshape `(128, 64, 16, 512)` — free view, no copy.

Op 1 (matmul):

- `sbuf_lhs_T` — HBM load for lhs_T(K=d0, M=d1). Same derivation as single matmul. Shape: **`(128, 1, 1, 128)`**.
- `psum_result` — PSUM accum for result(M=d1, N=d2). K=d0 outermost — d1 and d2 both inside K's loop. `op_M=128`, `op_N=512`. Num tiles: `8192/128 = 64`, `8192/512 = 16`. fp32. Shape: **`(128, 64, 16, 512)`**. `save_tensor_block` handles PSUM→SBUF→HBM internally.

**Lowered NKI kernel** — one self-contained loop nest per op:

```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def matmul(lhs_T, rhs_T):
    result = nl.ndarray((8192, 8192), dtype=nl.float16, buffer=nl.shared_hbm)

    """ === Op 0: nc_transpose(rhs_T) → rhs === """
    sbuf_rhs_T = nl.ndarray((128, 4, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_rhs_temp = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.psum)
    sbuf_rhs = nl.ndarray((128, 64, 64, 128), dtype=nl.float16, buffer=nl.sbuf)
    sbuf_rhs_op1 = sbuf_rhs.reshape((128, 64, 16, 512))

    for i_block_d0 in range(64):
        for i_block_d2 in range(16):
            load_tensor_block(sbuf_rhs_T, rhs_T,
                              par_ofs=i_block_d2 * 512, free_ofs=i_block_d0 * 128)
            for i_tile_d0 in range(1):
                for i_tile_d2 in range(1):
                    for i_ig_d0 in range(1):
                        for i_ig_d2 in range(4):
                            ld2 = i_tile_d2 * 4 + i_ig_d2
                            td0 = i_tile_d0 * 1 + i_ig_d0
                            gd2 = i_block_d2 * 4 + ld2
                            gd0 = i_block_d0 * 1 + td0
                            nisa.nc_transpose(psum_rhs_temp[0:128, 0, 0, 0:128],
                                              sbuf_rhs_T[0:128, ld2, td0, 0:128])
                            nisa.tensor_copy(sbuf_rhs[0:128, gd0, gd2, 0:128],
                                             psum_rhs_temp[0:128, 0, 0, 0:128])

    """ === Op 1: nc_matmul(lhs_T, rhs) → result === """
    sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_result = nl.ndarray((128, 64, 16, 512), dtype=nl.float32, buffer=nl.psum)

    nisa.memset(psum_result[0:128, 0:64, 0:16, 0:512], value=0.0)
    for i_block_d0 in range(64):
        for i_block_d1 in range(64):
            for i_block_d2 in range(16):
                load_tensor_block(sbuf_lhs_T, lhs_T,
                                  par_ofs=i_block_d0 * 128, free_ofs=i_block_d1 * 128)
                for i_tile_d0 in range(1):
                    for i_tile_d1 in range(1):
                        for i_tile_d2 in range(1):
                            for i_ig_d0 in range(1):
                                for i_ig_d1 in range(1):
                                    for i_ig_d2 in range(1):
                                        td0 = i_tile_d0 * 1 + i_ig_d0
                                        td1 = i_tile_d1 * 1 + i_ig_d1
                                        d2_ut = i_block_d2 + i_tile_d2
                                        gd0 = i_block_d0 * 1 + td0
                                        gd1 = i_block_d1 * 1 + td1
                                        nisa.nc_matmul(
                                            psum_result[0:128, gd1, d2_ut, 0:512],
                                            sbuf_lhs_T[0:128, td0, td1, 0:128],
                                            sbuf_rhs_op1[0:128, gd0, d2_ut, 0:512])
    save_tensor_block(result, psum_result, par_ofs=0, free_ofs=0)

    return result
```
