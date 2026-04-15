## KernelIR

`KernelIR` structurally represents a kernel. `render_ir` mechanically lowers it to NKI source code.

```python
@dataclass
class KernelIR:
    dim_analysis: DimAnalysis
    op_graph: OpGraph

    fusion_groups: list[list[int]]
    tiles_per_block: dict[tuple[int, str], int]
    buffer_degrees: dict[tuple[int, str, str], int]
    loop_order: list[list[str]]
    load_placements: dict[tuple[str, str], str]
```

**`dim_analysis`** — from `analyze_dims`. Contains:
- `func_name`, `param_names`, `return_name` — kernel signature.
- `dims: dict[str, DimInfo]` — per-dimension `(dim_size, tile_size, min_tile_size, is_data_parallel)`.
- `tensors: dict[str, TensorInfo]` — per-tensor `(dim_ids, shape, dtype, isa_loc)`.

**`op_graph`** — from `build_op_graph`. Contains:
- `nodes: list[str]` — `op_idx -> op_type` (e.g. `"nc_matmul"`).
- `edges: list[tuple[int, int, str, str]]` — `(producer, consumer, tensor, role)`. Only inter-op tensors — kernel inputs with no producer op are absent.
- `op_tensors: list[tuple[dict[str, str], list[str]]]` — per-op `(inputs, outputs)`. `inputs` maps `role -> tensor_name` (including kernel inputs with no producer). `outputs` lists output tensor names.

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: `(op_idx, dim_id) -> int`. Initially `1` for all pairs.
- `buffer_degrees`: `(group_idx, tensor_name, dim_id) -> int`. Initially `1`. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — all dimension IDs in priority order. The renderer filters to the relevant subset (reduction dims for section 3). Initially `sorted(da.dims)` for every group.
- `load_placements`: `(tensor_name, dim_id) -> tier`. Initially absent. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the current `loop_order`.
- **save / tensor_copy(psum→sbuf)**: after the blocking dimension's last inner loop.
- **tensor_copy (interleave reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.

## 1. Kernel Header

`render_ir` emits a fixed preamble before any loop nests or buffers. All fields come directly from KernelIR — no heuristics.

**Imports.** Always the same three lines:

```python
import nki
import nki.isa as nisa
import nki.language as nl
```

**Decorator and signature.** `@nki.jit` decorator, then `def {func_name}({param_names}):` where `func_name` and `param_names` are read from KernelIR.

**Input shape assertions.** For each parameter in `param_names`, emit `assert {param}.shape == {shape}` using the shape from `tensors[param]`.

**Output HBM allocation.** For the return tensor `return_name`, emit:

```python
{return_name} = nl.ndarray({shape}, dtype=nl.{dtype}, buffer=nl.shared_hbm)
```

Shape and dtype come from `tensors[return_name]`. The output is always allocated in `nl.shared_hbm`. At the end of the function body, emit `return {return_name}`.

## 2. Data-Parallel Loops

After the header, `render_ir` emits loops for the data-parallel dimensions. A dimension is data-parallel when `DimInfo.is_data_parallel` is True — it appears in the kernel's return tensor. Every other dimension is a reduction dimension.

Each dimension contributes 3 loops: block, tile, and interleave. All three are always emitted, even when trip count is 1. Trip counts come from `DimInfo` and `tiles_per_block`:

| Loop | Variable | Trip count |
|---|---|---|
| Block | `i_block_d{id}` | `dim_size / (tiles_per_block * tile_size)` |
| Tile | `i_tile_d{id}` | `tiles_per_block` |
| Interleave | `i_ig_d{id}` | `tile_size / min_tile_size` |

Loops are grouped by phase — all block loops outermost, then all tile loops, then all interleave loops. Within each phase, data-parallel dimensions are sorted by dimension ID (fixed ordering — DP loops wrap all groups, so `loop_order` does not apply here). Block loops define the data boundary (DMA loads happen here), tile loops iterate within a block, and interleave loops handle sub-tile iteration when ops have different tile size limits on the same dimension.

### 2.1 Example: Attention

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`, `tiles_per_block = 1`:

| Dim | dim_size | tile_size | min_tile_size | DP/reduction | block | tile | ig |
|---|---|---|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | DP | 16 | 1 | 1 |
| d1 | 128 | 128 | 128 | reduction | — | — | — |
| d2 | 2048 | 512 | 128 | reduction | — | — | — |
| d4 | 128 | 128 | 128 | DP | 1 | 1 | 1 |

d1 and d2 are reduction dimensions — not emitted here. d0 and d4 are data-parallel:

```python
for i_block_d0 in range(16):
    for i_block_d4 in range(1):
        for i_tile_d0 in range(1):
            for i_tile_d4 in range(1):
                for i_ig_d0 in range(1):
                    for i_ig_d4 in range(1):
                        ...
```

## 3. Reduction Loops

Inside the innermost DP loop, the `...` placeholder is replaced by reduction content. The compute graph is a DAG, so the reduction region is a sequence of **sibling blocks** (one per fusion group), not a single deep nest. Each group runs its reduction to completion before the next group starts.

**Group ordering.** Lift `op_graph.edges` to group level: for each edge `(producer, consumer, tensor, role)`, find the producer's group and the consumer's group; if they differ, add a directed edge between groups. Topologically sort; ties broken by minimum `op_idx` in each group.

**Per-group reduction dims.** For each op in a group, collect all input and output tensor names from `op_graph.op_tensors[op_idx]`. Union their `dim_ids` (from `dim_analysis.tensors`) across all ops in the group, subtract the data-parallel dims. The remainder is the group's reduction dims.

**Per-group loop structure.** Same phase-grouped pattern and trip count formulas as section 2, applied to the group's reduction dims. Within each phase, reduction dims are ordered by `loop_order[group_idx]` (filtered to reduction dims, preserving relative order). `tiles_per_block` is read from `ir.tiles_per_block[(op_idx, dim_id)]` using the first op in the group.

Groups with no reduction dims emit no loops — the body sits directly at the DP indentation level.

### 3.1 Example: Attention

Continuing from section 2.1. Op graph (11 ops):

```
[0] transpose Q ──→ [2] matmul QK ──→ [3] affine_select ──→ [4] tensor_scalar ─┬→ [5] tensor_reduce ─┐
[1] transpose K ──↗                                                              └→ [6] act_reduce ←───┘
                                                                                      ├→ [7] activation ──→ [10] tensor_scalar
                                                                                      └→ [8] transpose ──→ [9] matmul SV ──↗
```

Eleven fusion groups (initial): `[[0], [1], ..., [10]]`.

| Group | Op | Reduction dims |
|---|---|---|
| 0 | nc_transpose (Q) | {d1} |
| 1 | nc_transpose (K) | {d1, d2} |
| 2 | nc_matmul (QK) | {d1, d2} |
| 3 | affine_select (mask) | {d2} |
| 4 | tensor_scalar (scale) | {d2} |
| 5 | tensor_reduce (max) | {d2} |
| 6 | activation_reduce (exp+sum) | {d2} |
| 7 | activation (reciprocal) | (none) |
| 8 | nc_transpose (exp_S) | {d2} |
| 9 | nc_matmul (SV) | {d2} |
| 10 | tensor_scalar (scale output) | (none) |

**Group-level DAG:** `0→2`, `1→2`, `2→3`, `3→4`, `4→5`, `4→6`, `5→6`, `6→7`, `6→8`, `8→9`, `7→10`, `9→10`. Topological order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

Reduction dim trip counts:

| Dim | dim_size | tile_size | min_tile_size | block | tile | ig |
|---|---|---|---|---|---|---|
| d1 | 128 | 128 | 128 | 1 | 1 | 1 |
| d2 | 2048 | 512 | 128 | 4 | 1 | 4 |

Inside the innermost DP loop, the eleven groups emit as sibling blocks. Groups 7 and 10 have no reduction dims — no loops emitted:

```python
"""inside innermost DP loop"""

# Group 0: nc_transpose Q [reduction: d1]
for i_block_d1 in range(1):
    for i_tile_d1 in range(1):
        for i_ig_d1 in range(1):
            ...

# Group 1: nc_transpose K [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ig_d1 in range(1):
                    for i_ig_d2 in range(4):
                        ...

# Group 2: nc_matmul QK [reduction: d1, d2]
for i_block_d1 in range(1):
    for i_block_d2 in range(4):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ig_d1 in range(1):
                    for i_ig_d2 in range(4):
                        ...

# Group 3: affine_select [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 4: tensor_scalar scale [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 5: tensor_reduce max [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 6: activation_reduce exp+sum [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 7: activation reciprocal [reduction: (none)]
...

# Group 8: nc_transpose exp_S [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 9: nc_matmul SV [reduction: d2]
for i_block_d2 in range(4):
    for i_tile_d2 in range(1):
        for i_ig_d2 in range(4):
            ...

# Group 10: tensor_scalar scale output [reduction: (none)]
...
```

This is the default lowering — each group gets its own independent reduction loops. The reference kernel (section 5) is the result of applying transforms (loop fusion, online softmax) on top of this baseline.

## 4. Tensor Buffers

Every on-chip tensor (`isa_loc` is `"sbuf"` or `"psum"`) needs a buffer allocation. HBM tensors (kernel inputs) are excluded — they are parameters, not on-chip buffers. The return tensor gets both an HBM allocation (section 1, for the final store destination) and an on-chip buffer here (for the intermediate compute result, e.g. PSUM accumulator).

**Placement rule: allocate at the top of the loop level where the buffer lives.** A buffer that is reused across iterations of a loop is allocated outside that loop. A buffer that is recycled each iteration is allocated at the top of the loop body. In the default lowering (degree-1, no load placement), every on-chip buffer holds one tile and is consumed within the innermost DP loop body, so all allocations go at the top of the innermost DP loop body, before any reduction loops.

The reference attention CTE kernel follows this same rule: persistent buffers (running_max, running_sum) are allocated before the section loop because they survive across sections. Per-section buffers (K/V SBUF, compute temps) are allocated at the top of the section loop body, with the allocator reset to a checkpoint each iteration so memory is reused.

**Buffer shape.** Every on-chip buffer uses a uniform layout with `(tile_size, num_tiles)` per dimension. A 2D tensor with `dim_ids = (dA, dB)` gets a 4D buffer:

```python
{loc}_{name} = nl.ndarray(
    ({dA_tile_size}, {dA_num_tiles}, {dB_num_tiles}, {dB_tile_size}),
    dtype=nl.{dtype},
    buffer=nl.{isa_loc},
)
```

A 1D tensor with `dim_ids = (dA,)` gets a 2D buffer: `({dA_tile_size}, {dA_num_tiles})`.

**`num_tiles` formula.** For each dimension, `num_tiles` is the product of four factors:

$$\text{num\_tiles} = \text{num\_blocks} \times \text{tiles\_per\_block} \times \text{interleave} \times \text{buffer\_degree}$$

where `num_blocks = dim_size / (tiles_per_block * tile_size)`, `interleave = tile_size / min_tile_size`, and `buffer_degree` comes from `ir.buffer_degrees[(group_idx, tensor_name, dim_id)]`. In the default lowering (`tiles_per_block = 1`, `buffer_degree = 1`, no load hoisting), `num_tiles = 1` for every dimension — the buffer holds exactly one tile.

Transforms grow `num_tiles` by changing these factors: `tiles_per_block` increases the tile-loop range, `buffer_degrees` adds multi-buffering capacity (e.g. double-buffering = 2), and `load_placements` hoists buffers to cover larger tile ranges.

**Tile sizes** come from `dim_analysis.dims[dim_id].tile_size`.

**PSUM staging.** A PSUM tensor gets an additional SBUF staging buffer (`sbuf_{name}`) when a consumer requires it. Each op declares per-operand memory requirements via `INPUT_LOCS` (e.g. `{"stationary": "sbuf", "moving": "sbuf"}` for nc_matmul). The renderer checks each consumer: if any operand reading this tensor has `INPUT_LOCS[role] == "sbuf"`, an SBUF staging buffer is emitted. The return tensor also needs staging (dma_copy to HBM reads from SBUF).

Currently all ops declare `INPUT_LOCS = "sbuf"` for all operands, so every PSUM tensor gets staging. The consumer-driven check is generic — if a future op accepts PSUM input directly, its tensors would skip staging automatically.

**Dtype.** PSUM buffers from ops with `PSUM_DTYPE` set (nc_matmul → float32) use that dtype. All SBUF buffers use the tensor's dtype.

**Buffer naming.** `sbuf_{tensor_name}` for SBUF, `psum_{tensor_name}` for PSUM.

### 4.1 Example: Attention

On-chip tensors (Q, K, V excluded — HBM inputs). All `num_tiles = 1` in the default lowering:

```python
psum_Q_t = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.psum)       # (d1, d0)
sbuf_Q_t = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)       # staging: nc_matmul reads sbuf
psum_K_t = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.psum)       # (d1, d2)
sbuf_K_t = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)       # staging: nc_matmul reads sbuf
psum_S = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)          # (d0, d2)
sbuf_S = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)         # staging: affine_select reads sbuf
sbuf_masked_S = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)  # (d0, d2)
sbuf_scaled_S = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)  # (d0, d2)
sbuf_neg_max = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
sbuf_exp_S = nl.ndarray((128, 1, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)     # (d0, d2)
sbuf_sum_exp = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
sbuf_inv_sum = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)           # (d0)
psum_exp_S_t = nl.ndarray((512, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.psum)   # (d2, d0)
sbuf_exp_S_t = nl.ndarray((512, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)   # staging: nc_matmul reads sbuf
psum_attn = nl.ndarray((128, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)       # (d0, d4)
sbuf_attn = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)      # staging: tensor_scalar reads sbuf
sbuf_output = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)    # (d0, d4)
```

## 5. Reference Kernel

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`:

| Dim | dim_size | tile_size | min_tile_size | DP/reduction |
|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | DP |
| d1 | 128 | 128 | 128 | reduction |
| d2 | 2048 | 512 | 128 | reduction |
| d4 | 128 | 128 | 128 | DP |

Op graph (11 ops, DAG with diamond at ops 4→5/6):

```
[0] transpose Q ──→ [2] matmul QK ──→ [3] affine_select ──→ [4] tensor_scalar ─┬→ [5] tensor_reduce ─┐
[1] transpose K ──↗                                                              └→ [6] act_reduce ←───┘
                                                                                      ├→ [7] activation ──→ [10] tensor_scalar
                                                                                      └→ [8] transpose ──→ [9] matmul SV ──↗
```

The reference attention CTE kernel organizes this DAG into sequential phases inside a d2 section loop, with d0 iteration inside. Translated to our dimension names (reference variable names in parentheses):

```python
for i_d2_section in range(1):                     # (section_idx) d2 block groups; >1 when seq_k > 8192
    nl.load K[i_d2_section]                       #   all d2 tiles in section → SBUF
    nl.load V[i_d2_section]                       #   all d2 tiles in section → SBUF

    for i_d0 in range(16):                        # (grp_i) DP: d0 tiles, 128 rows each
        nl.load Q[i_d0]                           #   one d0 tile → SBUF
        nisa.nc_transpose Q → Q_t                 #   [0] d0×d1 → d1×d0

        # --- Phase: QK + mask + scale + max (ops 2,3,4,5 fused) ---
        # (reference: _qk_and_max_impl)
        nisa.memset partial_max
        for i_d2 in range(4):                     # (k_tile_idx) d2 tiles, 512 each
            nisa.nc_matmul Q_t, K_t → S           #   [2] d1 reduced, one d2 tile
            nisa.affine_select S → masked_S        #   [3] causal mask
            nisa.tensor_scalar masked_S → scaled_S #   [4] scale + partial max

        # (reference: _update_max_impl)
        nisa.tensor_reduce partial_max → neg_max   # [5] max across all d2 tiles

        # --- Phase: exp + sum + transpose (ops 6,8 fused) ---
        # (reference: _exp_impl)
        nisa.memset partial_sum
        for i_d2 in range(4):                     # (exp_tile_idx) d2 tiles, 512 each
            nisa.activation_reduce → exp_S, sum_exp #  [6] exp(scaled_S - neg_max) + sum
            nisa.nc_transpose exp_S → exp_S_t       #  [8] d0×d2 → d2×d0

        # --- Phase: PV matmul (op 9 alone) ---
        # (reference: _pv_impl)
        nisa.memset pv_accum
        for i_d2 in range(4):                     # (mm2_grp×mm2_i) d2 tiles, 512 each
            nisa.nc_matmul exp_S_t, V → attn       #  [9] d2 accumulated
        nisa.tensor_copy psum → sbuf

        # --- Phase: write-back (ops 7,10 — no d2 loop) ---
        # (reference: _write_back_impl)
        nisa.activation sum_exp → inv_sum          # [7] reciprocal
        nisa.tensor_scalar attn, inv_sum → output  # [10] scale
        nl.store output → HBM
```

**Structure.** The d2 section loop is outermost — K and V are loaded once per section, then all d0 tiles (Q groups) are processed against that section's K/V. Within each d0 iteration, phases are sequential siblings: each d2-dependent phase has its own `i_d2` tile loop. Ops within a phase share one d2 loop: QK+mask+scale+max (ops 2–5) fused, exp+transpose (ops 6+8) fused, PV matmul (op 9) alone. Write-back (ops 7, 10) has no d2 loop.

**Blocking-axis boundaries frame each phase.** `memset` before the d2 loop zeros the accumulator; `tensor_reduce`/`tensor_copy` after finalizes the reduction. d1 is trivial (1 tile) — consumed entirely within each `nc_matmul` call, no explicit loop.

**The diamond** (op 4 → ops 5,6) is sequenced: max (op 5) completes before exp+sum (op 6) starts, since op 6 consumes neg_max. Both paths reconverge at op 10.