### KernelIR

```python
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

**Transform state** — starts at defaults, modified by transforms (§6):
- `fusion_groups`: initially `[[0], [1], ...]` — each op in its own group.
- `tiles_per_block`: initially `1` for all `(op_idx, dim_id)` pairs.
- `buffer_degrees`: initially `1` for all `(group_idx, tensor_name, dim_id)` triples. Each tensor's buffer is independent per fusion group — the same tensor loaded in two groups can have different degrees.
- `loop_order`: per group — dimension ordering within each phase.
- `load_placements`: initially absent (all loads at default position). `(tensor_name, dim_id) → tier` where tier is `"per_tile"`, `"per_block"`, or `"full"`. See §6.2.

**Renderer-derived positions** — NOT stored in KernelIR, mechanically derived by `render_ir` from the transform state above:
- **memset**: before the accumulation (K) dimension's outermost loop in the current `loop_order`.
- **save / tensor_copy(psum→sbuf)**: after the accumulation dimension's last inner loop.
- **tensor_copy (interleave reload/save)**: around the tile loop when an accumulating dim's block and tile are split by other dims' loops.
- These are deterministic given the loop structure — exactly one correct position for each, no ambiguity.
