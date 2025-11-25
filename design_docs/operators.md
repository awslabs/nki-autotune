# Operator Design

## Operator Representation

Operators use **BufferNode** base class with semantic-based API:

```python
Matmul(
    dest="output",
    lhs="lhs_norm",
    rhs="rhs",
    lhs_transposed=False
)
# Semantic axes: lhs=[M,K], rhs=[K,N], dest=[M,N]
```

Check [NKI ISA documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.isa.html) for operator semantics.

## BufferNode Architecture

### Core Components

- **semantic_to_axes**: Maps semantic names to axis lists (e.g., `{"lhs": ["M", "K"]}`)
- **semantic_to_name**: Maps semantic names to tensor variable names (e.g., `{"lhs": "lhs_norm"}`)
- **axis_sizes**: Dictionary tracking axis specialization (e.g., `{"M": 256}`)

### Specialization

Operators specialize via shape propagation:
```python
operator.specialize(semantic_name="lhs", shape=(256, 1024))
# Maps axes M=256, K=1024
```

Validation ensures:
- All input/output semantics have axes and names
- Axis sizes are consistent across operators
- Constant axes (numeric strings like "1", "128") auto-populate

## Design Principles

### 1. Axis Order Defines Indexing

- `["M", "K"]` means tensor is indexed as `A[M, K]`
- Axis names are symbolic until specialized

### 2. Constant Axes for Reductions

Reduced dimensions use size 1 as string constant `"1"`:
```python
semantic_to_axes = {
    "data": ["P", "F"],
    "reduce_res": ["P", "1"]  # F axis reduced
}
```

## Implemented Operators

**On-chip**: TensorScalar, Activation, Transpose, TileTranspose, Matmul, Allocate
**HBM**: Load, Store
