## KernelIR

`KernelIR` structurally represents a kernel. `render_ir` mechanically lowers it to NKI source code.

```python
@dataclass
class KernelIR:
    dim_analysis: DimAnalysis
    op_graph: OpGraph

    fusion_groups: list[list[int]]
    ltiles_per_block: dict[str, int]
    buffer_degrees: dict[tuple[str, str], int]
    loop_order: list[str | list[str]]
    tensor_placements: dict[tuple[str, str], str]
```

**`dim_analysis`** — from `analyze_dims`. Provides dimension IDs, logical/physical tile sizes, num_ptiles_per_ltile, data-parallel classification, tensor metadata, and per-op axis/tile mappings. The renderer reads these to compute loop trip counts and buffer shapes.

**`op_graph`** — from `build_op_graph`. Provides op classes, inter-op edges, per-op tensor roles, and per-op kwargs. The renderer reads these to determine op ordering, memory boundaries, and ISA call rendering.

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — each op in its own group.
- `ltiles_per_block`: `dim_id -> int`. Per-dimension tiling factor — every op and tensor on a given dim shares the same block structure. Initially `1` for every dim in `da.dims`.
- `buffer_degrees`: `(tensor_name, dim_id) -> int`. Initially `1` for every `(tensor, dim)` pair where `dim` is in the tensor's `dim_ids`. Multi-buffering degree along that axis — see `multi_buffer.md`.
- `loop_order`: single flat list for the whole kernel nest. Top-level string entries are DP dim IDs in outer-to-inner order; each nested `list[str]` is one fusion group's reduction dim IDs in outer-to-inner order, positional on `fusion_groups`. An empty sublist marks a group with no reduction dims. Example: for a kernel with two DP dims `d0, d4` and two fusion groups whose reduction nests are `for d1` and `for d1: for d2`, `loop_order = ["d0", "d4", ["d1"], ["d1", "d2"]]` — decoded as `for d0: for d4: { for d1 { ... }; for d1: for d2 { ... } }`. Initially DP dims `sorted` on top, then one sorted reduction sublist per fusion group.
- `tensor_placements`: `(tensor_name, dim_id) -> tier`. Initially `"per_tile"` for all pairs. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the current `loop_order`.
- **tensor_copy(psum→sbuf) and store(sbuf→hbm)**: when the source is valid — after the blocking dimension's last inner loop for PSUM→SBUF, after all reduction groups for SBUF→HBM. Same rule at both memory boundaries.
- **tensor_copy (reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.
