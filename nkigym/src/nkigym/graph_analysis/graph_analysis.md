## Graph Analysis

The computation DAG tracks producer-consumer dependencies between ops. Each op reads input tensors and writes an output tensor; when op B reads the tensor that op A writes, B depends on A. Every downstream decision — memory boundary selection, fusion legality, loop ordering constraints, cross-group buffer allocation — reduces to a query on this graph.

```python
@dataclass
class OpGraph:
    """Computation DAG."""
    nodes: list[str]                              # op_idx → op_type
    edges: list[tuple[int, int, str, str]]        # (producer, consumer, tensor, role)
```

- **Nodes** carry the op type — graph queries like fusion legality and blocking analysis depend on op semantics.
- **Edges** carry the tensor name and the role it fills at the consumer — knowing *how* an op consumes a tensor (e.g., matmul's stationary vs moving) matters for dimension relationship analysis across edges.
