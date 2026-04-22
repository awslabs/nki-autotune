## KernelContext + KernelIR

A kernel is represented by two independent pieces with a single source of truth each — codegen, sampling, and validation take both as arguments.

```python
@dataclass
class KernelContext:
    """Kernel-wide globals."""
    dim_analysis: DimAnalysis
    ltiles_per_block: dict[str, int]
    required_merges: list[frozenset[int]]

@dataclass
class KernelIR:
    """Trivial pass-through bundle of (context, graph). No state of its own."""
    context: KernelContext
    graph: OpGraph
```

**`KernelContext`** — globals. Anything that describes the kernel as a whole and is referenced by multiple fusion groups or by codegen that spans groups.
- `dim_analysis`: dim IDs, sizes, tile sizes, `DimRole`, and the logical-tensor catalog (`{tensor_name: TensorInfo}`). Tensor metadata is keyed by name and crosses group boundaries; splitting it per group would turn every cross-group lookup into a search.
- `ltiles_per_block`: `dim_id -> int`. A property of the dim, not of any group — every group touching a d-carrying tensor must agree on `num_blocks × ltiles_per_block` for `d`, because a consumer's `i_block_d` / `i_ltile_d` loops index the slots its producer filled.
- `required_merges`: op-index clusters that must end up in a single fusion group. Populated by IR-level rewrites that need specific groupings downstream; empty by default.

**`OpGraph`** — structure + per-group codegen state. Documented fully in `../graph/op_graph.md`. In short: `op_classes` / `edges` / `op_tensors` / `op_all_kwargs` describe the DAG; `groups: list[FusionGroup]` carries per-group `dim_order` / `buffer_degrees` / `tensor_placements`.

**`KernelIR`** — a zero-behavior passthrough. Holds no state. Attribute access is forwarded: `ir.dim_analysis` → `ir.context.dim_analysis`, `ir.ltiles_per_block` → `ir.context.ltiles_per_block`, `ir.op_graph` → `ir.graph`, `ir.fusion_groups` → `ir.graph.groups`. Entry points that accept a single argument can take `ir`; entry points that want explicit `(context, graph)` unpack as `ir.context, ir.graph`.

**Per-group codegen state** lives on each `FusionGroup` in `graph.groups`:
- `dim_order`: outer-to-inner dim list for this group's nest.
- `buffer_degrees`: `(buffer_kind, tensor, dim) -> int`. Multi-buffering degree along that axis — see `../sampler/multi_buffer.md`.
- `tensor_placements`: `(buffer_kind, tensor, dim) -> tier`. Tier is `"per_tile"` / `"per_block"` / `"full"` — see `../sampler/load_placement.md`.

**Why no DP-vs-reduction split in `dim_order`.** A dim's role (`PARALLEL` / `SERIAL` / `ACCUMULATION`, via `DimInfo.role`) matters for fusion legality and accumulation semantics, not loop structure. Each group loops over exactly the dims its ops touch; loop fusion and loop reordering expose arbitrary cross-group sharing through the merge sampler.

**Renderer-derived positions** — NOT stored in the IR, mechanically derived by `render_ir` from the fields above:
- **memset**: before the blocking dimension's outermost loop in the group's `dim_order`.
- **tensor_copy (psum→sbuf) and store (sbuf→hbm)**: when the source is valid — after the blocking dimension's last inner loop for PSUM→SBUF, at group-nest close for SBUF→HBM.
- **tensor_copy (reload/save)**: around the tile loop when a blocking dim's block and tile are split by other dims' loops.
- Deterministic given the loop structure — exactly one correct position for each, no ambiguity.
