# Operand Coalesce

Finds pairs of the same operation where all operands are identical except
one, and that single differing operand can be combined along an adjacent
dimension. When found, the two operations are coalesced into a single
wider operation.

The algorithm:

1. Group operations by type.
2. For each pair of the same type, compare operands.
3. If all operands match except one, check whether the differing operand
   pair is adjacent and within hardware limits.
4. Verify data dependencies — coalescing must not change the computation.
5. If so, report a coalesce opportunity.

Like all transforms, operand coalesce has two steps:

1. **`analyze(func)`** — inspect the IR and return a list of coalesce groups.
2. **`transform(func, group)`** — apply a single group, returning a new callable.

## Examples

### Tile loads

Two loads from the same source tensor with the same partition slice and
adjacent free slices. The differing operand is the free slice.

**Before:**

```python
tensor_1 = b[0:128, 0:128]
tensor_4 = b[0:128, 128:256]
```

**After** (`tensor_4` absorbed into `tensor_1`):

```python
tensor_1 = b[0:128, 0:256]
```

### `nc_matmul`

Two matmul calls with the same LHS. The differing operand is the RHS,
which consists of adjacent slices along N (moving free dimension).
The output stores are merged accordingly.

**Before:**

```python
tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 0:128])
output[0:128, 0:128] = tensor_2

tensor_5 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 128:256])
output[0:128, 128:256] = tensor_5
```

**After:**

```python
tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
output[0:128, 0:256] = tensor_2
```

### `tensor_tensor`

Two element-wise binary ops (`data1 <op> data2`) with the same operator
and one shared operand. The differing operand consists of adjacent slices.
Both inputs must match in partition size and elements per partition.

**Before:**

```python
tensor_2 = nkigym.tensor_tensor(tensor_0[0:128, 0:128], tensor_1, op=np.add)
output[0:128, 0:128] = tensor_2

tensor_4 = nkigym.tensor_tensor(tensor_0[0:128, 128:256], tensor_1, op=np.add)
output[0:128, 128:256] = tensor_4
```

**After:**

```python
tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)
output[0:128, 0:256] = tensor_2
```

### `activation`

Two activation calls with the same function, scale, and bias, applied to
adjacent slices of the same input. The Scalar Engine processes each
partition independently, so widening the free dimension is valid as long
as the result fits in SBUF.

**Before:**

```python
tensor_1 = nkigym.activation(tensor_0[0:128, 0:128], op=nl.relu)
output[0:128, 0:128] = tensor_1

tensor_3 = nkigym.activation(tensor_0[0:128, 128:256], op=nl.relu)
output[0:128, 128:256] = tensor_3
```

**After:**

```python
tensor_1 = nkigym.activation(tensor_0, op=nl.relu)
output[0:128, 0:256] = tensor_1
```

### `tensor_scalar`

Two tensor-scalar ops with the same operator(s) and scalar/vector
operand(s), applied to adjacent slices of the input data. The operands
broadcast as `(P, 1)` vectors along the free dimension, so widening the
data slice does not change their value.

**Before:**

```python
tensor_1 = nkigym.tensor_scalar(tensor_0[0:128, 0:128], op0=np.multiply, operand0=2.0)
output[0:128, 0:128] = tensor_1

tensor_3 = nkigym.tensor_scalar(tensor_0[0:128, 128:256], op0=np.multiply, operand0=2.0)
output[0:128, 128:256] = tensor_3
```

**After:**

```python
tensor_1 = nkigym.tensor_scalar(tensor_0, op0=np.multiply, operand0=2.0)
output[0:128, 0:256] = tensor_1
```
## Operator tile size constraints (Trn2 / NeuronCore-v3)

| Operator | Dimension | Max size |
|---|---|---|
| `nc_matmul` | M (stationary free) | 128 |
| `nc_matmul` | K (contraction) | 128 |
| `nc_matmul` | N (moving free) | 512 |
| `nc_transpose` | Tensor Engine | 128 x 128 |
| `activation` | partition dim | 128 |
| `activation` | free dim | SBUF partition size |
| `tensor_reduce` | partition dim | 128 |
| `tensor_reduce` | free dim | SBUF partition size |
| `tensor_scalar` | operands | broadcast as (P, 1) vectors |
| `tensor_tensor` | inputs | must match partition size and elements per partition |
