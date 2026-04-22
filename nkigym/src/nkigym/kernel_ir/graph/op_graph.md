## Graph Analysis

The computation DAG tracks producer-consumer dependencies between ops. Each op reads input tensors and writes an output tensor; when op B reads the tensor that op A writes, B depends on A. Every downstream decision — memory boundary selection, fusion legality, loop ordering constraints, cross-group buffer allocation — reduces to a query on this graph.

```python
@dataclass
class OpGraph:
    """Computation DAG."""
    op_classes: list[type[NKIOp]]                 # op_idx → NKIOp subclass
    edges: list[tuple[int, int, str, str]]        # (producer, consumer, tensor, role)
    op_tensors: list[tuple[dict[str, str], list[str]]]
                                                  # per-op (role → input tensor, [output tensors])
    op_all_kwargs: list[dict[str, str]]           # per-op raw kwarg source strings
```

- **`op_classes`** carry the NKIOp subclass — graph queries like fusion legality and blocking analysis read class-level attributes (`NAME`, `BLOCKING_AXES`, `ISA_LOC`, `INPUT_LOCS`, `PSUM_DTYPE`, `format_isa_call`).
- **`edges`** carry the tensor name and the role it fills at the consumer — knowing *how* an op consumes a tensor (e.g., matmul's stationary vs moving) matters for dimension relationship analysis across edges. Only inter-op tensors appear; kernel inputs (no producing op) are absent.
- **`op_tensors`** record each op's input-role-to-tensor map (including kernel inputs, which have no producer edge) and its output tensor names.
- **`op_all_kwargs`** record the raw source string of every kwarg (tensor or scalar) so `format_isa_call` can inline scalar parameters.

**Helpers.** `OpGraph.producer_op(name)` returns the op index that produces `name`, or `None` if `name` is a kernel input. Codegen uses this (plus `param_names`) to determine each tensor's memory location instead of caching a per-tensor `loc` field.
