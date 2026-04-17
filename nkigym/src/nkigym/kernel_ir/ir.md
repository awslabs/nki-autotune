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
    group_dim_orders: list[list[str]]
    tensor_placements: dict[tuple[str, str], str]
```

**`dim_analysis`** — from `analyze_dims`. Provides dimension IDs, logical/physical tile sizes, num_ptiles_per_ltile, **blocking vs non-blocking** classification, tensor metadata, and per-op axis/tile mappings. The renderer reads these to compute loop trip counts and buffer shapes.

**`op_graph`** — from `build_op_graph`. Provides op classes, inter-op edges, per-op tensor roles, and per-op kwargs. The renderer reads these to determine op ordering, memory boundaries, and ISA call rendering.

**Rendering parameters** — `render_ir` reads these to determine loop structure, buffer sizes, and DMA placement:
- `fusion_groups`: which ops share a loop nest. Initially `[[0], [1], ...]` — one singleton group per op. Loop fusion merges groups later.
- `ltiles_per_block`: `dim_id -> int`. Per-dimension tiling factor — every op and tensor on a given dim shares the same block structure. Initially `1` for every dim in `da.dims`.
- `buffer_degrees`: `(tensor_name, dim_id) -> int`. Initially `1` for every `(tensor, dim)` pair where `dim` is in the tensor's `dim_ids`. Multi-buffering degree along that axis — see `multi_buffer.md`.
- `group_dim_orders`: one complete dim_order per fusion group, positional on `fusion_groups`. Each entry is the outer-to-inner dim list for that group's nest, covering every dim any op in the group touches. Every dim is a loop — no DP/reduction split. Example: for attention's default (11 singleton groups), `group_dim_orders[1] = ["d1", "d2"]` — K transpose's group loops only over its own dims, so no spurious outer loops on dims its ops don't touch.
- `tensor_placements`: `(tensor_name, dim_id) -> tier`. Initially `"per_tile"` for all pairs. Tier is `"per_tile"`, `"per_block"`, or `"full"`.

**Why no DP-vs-reduction split.** A dim's category (blocking / non-blocking, via `DimInfo.is_blocking`) matters for fusion *legality*, not loop *structure*. The old `loop_order` forced DP dims outermost and reduction dims per-group — which incidentally inflated iteration counts for ops that don't touch every DP dim (e.g., K transpose ran 16× inside the d0 DP loop despite having no d0 dependence). Per-group `dim_order` lets each group loop over exactly the dims its ops touch; loop fusion and loop reordering later expose arbitrary cross-group sharing.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the current `group_dim_orders` entry.
- **tensor_copy(psum→sbuf) and store(sbuf→hbm)**: when the source is valid — after the blocking dimension's last inner loop for PSUM→SBUF, at group-nest close for SBUF→HBM.
- **tensor_copy (reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.
