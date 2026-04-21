Code structure:
kernel_ir/ir.py: KernelIR(context=KernelContext, graph=KernelGraph)
kernel_ir/context/context.py: KernelContext(dimensions, logical tensors, ltiles_per_block)
kernel_ir/graph/graph.py: KernelGraph(list[FusionGroup], edges=xxx)
kernel_ir/graph/fusion_group.py: FusionGroup(list[NKIOp], group-specific codegen info like group dim orders)
kernel_ir/rewrites:
1. FusionGroup([NKILoad])+FusionGroup([NKITranspose])=FusionGroup([NKIDMATranspose])
2. FusionGroup([X])+FusionGroup([accumulator])=FusionGroup([NKIOnlineFusionChain])
3. FusionGroup([op A])+FusionGroup([op B])=FusionGroup([A, B]), where applicable e.g. for elementwise ops
ops: various NKIOp, including NKIOnlineFusionChain(NKIOp), NKILoad, NKIStore
codegen: generate NKI kernel based on KernelIR

```python
# kernel_ir/context/context.py
@dataclass
class KernelContext:
    func_name: str
    param_names: list[str]
    return_name: str
    dimensions: dict[str, DimInfo]
    logical_tensors: dict[str, TensorInfo]
    ltiles_per_block: dict[str, int]
    op_inputs: dict[NKIOp, dict[str, str]]
    op_outputs: dict[NKIOp, list[str]]
    op_kwargs: dict[NKIOp, dict[str, str]]
    op_axis_map: dict[NKIOp, dict[str, str]]
    op_tile_sizes: dict[NKIOp, dict[str, int]]
    op_blocking_dims: dict[NKIOp, set[str]]

# kernel_ir/graph/fusion_group.py
@dataclass
class FusionGroup:
    ops: list[NKIOp]
    dim_order: list[str]
    buffer_degrees: dict[tuple[str, str, str], int]
    tensor_placements: dict[tuple[str, str, str], str]

# kernel_ir/graph/graph.py
@dataclass
class KernelGraph:
    groups: list[FusionGroup]
    edges: list[tuple[int, int, str, str]]   # (producer_gi, consumer_gi, tensor, role)

# kernel_ir/ir.py
@dataclass
class KernelIR:
    context: KernelContext
    graph: KernelGraph
```