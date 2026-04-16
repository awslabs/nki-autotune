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

**`dim_analysis`** — from `analyze_dims`. Provides dimension IDs, logical/physical tile sizes, interleave factors, data-parallel classification, tensor metadata, and per-op axis/tile mappings. The renderer reads these to compute loop trip counts and buffer shapes.

**`op_graph`** — from `build_op_graph`. Provides op classes, inter-op edges, per-op tensor roles, and per-op kwargs. The renderer reads these to determine op ordering, memory boundaries, and ISA call rendering.

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: `(op_idx, dim_id) -> int`. Initially `1` for all pairs.
- `buffer_degrees`: `(group_idx, tensor_name, dim_id) -> int`. Initially `1`. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — all dimension IDs in priority order. The renderer filters to the relevant subset per fusion group (e.g. reduction dims only for single-op groups). Initially `sorted(da.dims)` for every group.
- `load_placements`: `(tensor_name, dim_id) -> tier`. Initially `"per_tile"` for all pairs. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the current `loop_order`.
- **tensor_copy(psum→sbuf) and store(sbuf→hbm)**: when the source is valid — after the blocking dimension's last inner loop for PSUM→SBUF, after all reduction groups for SBUF→HBM. Same rule at both memory boundaries.
- **tensor_copy (interleave reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.
