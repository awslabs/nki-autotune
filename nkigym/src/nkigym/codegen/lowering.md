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
- `edges: list[tuple[int, int, str, str]]` — `(producer, consumer, tensor, role)`.

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: `(op_idx, dim_id) -> int`. Initially `1` for all pairs.
- `buffer_degrees`: `(group_idx, tensor_name, dim_id) -> int`. Initially `1`. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — dimension ordering within each phase.
- `load_placements`: `(tensor_name, dim_id) -> tier`. Initially absent. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the accumulation (K) dimension's outermost loop in the current `loop_order`.
- **save / tensor_copy(psum→sbuf)**: after the accumulation dimension's last inner loop.
- **tensor_copy (interleave reload/save)**: around the tile loop when an accumulating dim's block and tile are split by other dims' loops.
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

Loops are grouped by phase — all block loops outermost, then all tile loops, then all interleave loops. Within each phase, data-parallel dimensions are ordered by dimension ID. The block loops define the data boundary (DMA loads happen here), tile loops iterate within a block, and interleave loops handle sub-tile iteration when ops have different tile size limits on the same dimension.

### 2.1 Example

Single matmul: `result = GymMatmul()(stationary=lhs_T, moving=rhs)` where lhs_T is (d0, d1) and rhs is (d0, d2). `result` has dims (d1, d2). With initial `tiles_per_block = 1`:

| Dim | dim_size | tile_size | min_tile_size | is_data_parallel | block | tile | ig |
|---|---|---|---|---|---|---|---|
| d0 | 8192 | 128 | 128 | False | — | — | — |
| d1 | 8192 | 128 | 128 | True | 64 | 1 | 1 |
| d2 | 8192 | 512 | 512 | True | 16 | 1 | 1 |

d0 is a reduction dimension — it is not emitted here. d1 and d2 are data-parallel:

```python
for i_block_d1 in range(64):
    for i_block_d2 in range(16):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ig_d1 in range(1):
                    for i_ig_d2 in range(1):
                        ...
```

Transpose + matmul: `rhs = GymTranspose()(data=rhs_T)`, `result = GymMatmul()(stationary=lhs_T, moving=rhs)` where lhs_T is (d0, d1) and rhs_T is (d2, d0). Same return tensor dims (d1, d2), but the interleave asymmetry on d2 (tile_size=512 from matmul N, min_tile_size=128 from transpose P) changes the ig trip count:

| Dim | dim_size | tile_size | min_tile_size | is_data_parallel | block | tile | ig |
|---|---|---|---|---|---|---|---|
| d1 | 8192 | 128 | 128 | True | 64 | 1 | 1 |
| d2 | 8192 | 512 | 128 | True | 16 | 1 | 4 |

```python
for i_block_d1 in range(64):
    for i_block_d2 in range(16):
        for i_tile_d1 in range(1):
            for i_tile_d2 in range(1):
                for i_ig_d1 in range(1):
                    for i_ig_d2 in range(4):
                        ...
```