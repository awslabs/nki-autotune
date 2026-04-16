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
- `per_op_axis_maps: list[dict[str, str]]` — per-op mapping from abstract axis names to concrete dim IDs.
- `op_tile_sizes: list[dict[str, int]]` — per-op tile sizes mapped to concrete dim IDs.

**`op_graph`** — from `build_op_graph`. Contains:
- `op_classes: list[type[NKIOp]]` — single source of truth for all per-op attributes (NAME, BLOCKING_AXES, PSUM_DTYPE, INPUT_LOCS, ISA_LOC, format_isa_call).
- `edges: list[tuple[int, int, str, str]]` — `(producer, consumer, tensor, role)`. Only inter-op tensors — kernel inputs with no producer op are absent.
- `op_tensors: list[tuple[dict[str, str], list[str]]]` — per-op `(inputs, outputs)`. `inputs` maps `role -> tensor_name` (including kernel inputs with no producer). `outputs` lists output tensor names.
- `op_all_kwargs: list[dict[str, str]]` — per-op `{kwarg_name: source_string}` for all kwargs (tensors and scalars).

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: `(op_idx, dim_id) -> int`. Initially `1` for all pairs.
- `buffer_degrees`: `(group_idx, tensor_name, dim_id) -> int`. Initially `1`. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — all dimension IDs in priority order. The renderer filters to the relevant subset (reduction dims for section 3 of lowering.md). Initially `sorted(da.dims)` for every group.
- `load_placements`: `(tensor_name, dim_id) -> tier`. Initially `"per_tile"` for all pairs. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the current `loop_order`.
- **tensor_copy(psum→sbuf) and store(sbuf→hbm)**: when the source is valid — after the blocking dimension's last inner loop for PSUM→SBUF, after all reduction groups for SBUF→HBM. Same rule at both memory boundaries.
- **tensor_copy (interleave reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.
