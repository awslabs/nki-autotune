# Operator Design

## Operator Representation

Operators in the graph IR are defined with the following structure:

```python
operator = {
    "op_code": "nisa.nc_matmul",
    "inputs": ["A", "B"],
    "outputs": ["C"],
    "tensor_axes": {
        "A": ["K", "M"],
        "B": ["K", "N"],
        "C": ["M", "N"]
    }
}
```
Check [NKI ISA documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.isa.html) for the proper operator semantics.

## Design Principles

### 1. Dimension Names Define Indexing

- **Order matters**: `["K", "M"]` means the tensor is indexed as `A[K, M]`
- Dimension names serve as both the axis label and the index variable
- Example: `"A": ["K", "M"]` means tensor A is indexed by dimensions K and M in that order

### 2. Automatic Reduction/Parallel Classification

Index semantics are inferred from tensor dimensions:
- **Parallel dimensions**: Appear in output tensor
- **Reduction dimensions**: Appear only in inputs, not in output

For the matmul example:
- Output C has dimensions `["M", "N"]` → M, N are parallel
- K appears in inputs A, B but not in output C → K is reduction
- Computation: `C[M, N] = sum_K(A[K, M] * B[K, N])`

## Key Design Features

### Constant Dimension for Reductions

Reduced dimensions are kept with size 1 using the string constant `1`:
- `tensor_dims = {"reduce_result": ["M", "1"]}`  # M kept, last dim reduced to 1
- Only `str` or `str 1` allowed in dimension specifications

### Dimension Tracking
- Graph resolves correspondence during specialization

### Operator Categories

**On-chip Operators**: TensorScalar, Activation, Transpose, Matmul, and other nisa computation operators. `nl.ndarray` Allocate operator.
**HBM Operators**: Load, Store