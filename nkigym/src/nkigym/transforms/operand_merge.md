# Operand Merge

Finds pairs of the same operation where only a single operand differs, and that single differing operand can be combined along an adjacent
dimension. When found, the two operations are merged into a single wider operation.

The algorithm:

1. Group operations by type.
2. For each pair of the same type, compare operands.
3. If all operands match except one, check whether the differing operand
   pair is adjacent and within hardware limits for that op.
4. Verify data dependencies — merging must not change the computation.
5. If so, report a merge opportunity.

Like all transforms, operand merge has two steps:

1. **`analyze(func)`** — inspect the IR and return a list of merge pairs.
2. **`transform(func, group)`** — apply a single pair, returning a new callable.

## Examples

### Loads

Two loads from the same source tensor that differ on exactly one
dimension with adjacent ranges. The merged first dimension (partition)
must be <= 128 (Trn DMA tile constraint).

**Before:**

```python
tensor_1 = b[0:128, 0:128]
tensor_4 = b[0:128, 128:256]
```

**After** (`tensor_4` absorbed into `tensor_1`):

```python
tensor_1 = b[0:128, 0:256]
```

Merging on the first dimension is also valid when the result fits
within 128:

**Before:**

```python
tensor_0 = a[0:64, 0:128]
tensor_3 = a[64:128, 0:128]
```

**After** (`tensor_3` absorbed into `tensor_0`):

```python
tensor_0 = a[0:128, 0:128]
```

If the merged first dimension would exceed 128, it is not an
option:

```python
tensor_0 = a[0:128, 0:128]
tensor_3 = a[128:256, 0:128]
```

Merged dim 0 would be 0:256 (size 256 > 128) — **rejected**.

### `nc_matmul`

Two matmul calls (`nc_matmul(lhs[K,M], rhs[K,N])`) that share one
operand and differ on the other with adjacent slices. Either operand
can be the differing one, subject to tile limits (N <= 512, M <= 128).

**RHS merge** — same LHS, adjacent RHS slices along N:

**Before:**

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
output[0:128, 0:128] = tensor_2

tensor_5 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 128:256])
output[0:128, 128:256] = tensor_5
```

**After** (merged N = 256, `tensor_5` absorbed into `tensor_2`):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:256])
output[0:128, 0:128] = tensor_2[0:128, 0:128]
output[0:128, 128:256] = tensor_2[0:128, 128:256]
```

**LHS merge** — same RHS, adjacent LHS slices along M:

**Before:**

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:64], tensor_1[0:128, 0:128])
output[0:64, 0:128] = tensor_2

tensor_4 = nkigym.nc_matmul(tensor_0[0:128, 64:128], tensor_1[0:128, 0:128])
output[64:128, 0:128] = tensor_4
```

**After** (merged M = 128, `tensor_4` absorbed into `tensor_2`):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
output[0:64, 0:128] = tensor_2[0:64, 0:128]
output[64:128, 0:128] = tensor_2[64:128, 0:128]
```

If the merged dimension would exceed the limit, it is not an option:

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:256])
tensor_5 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 256:512])
```

Merged N would be 0:512 (size 512) — **accepted** (at limit).

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:384])
tensor_5 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 384:640])
```

Merged N would be 0:640 (size 640 > 512) — **rejected**.

### Stores

Two stores to the same output tensor that differ on exactly one
dimension with adjacent ranges. The merged first dimension (partition)
must be <= 128, same as loads.

**Before:**

```python
output[0:128, 0:128] = tensor_2[0:128, 0:128]
output[0:128, 128:256] = tensor_2[0:128, 128:256]
```

**After:**

```python
output[0:128, 0:256] = tensor_2[0:128, 0:256]
```

Merging on the first dimension:

**Before:**

```python
output[0:64, 0:128] = tensor_2[0:64, 0:128]
output[64:128, 0:128] = tensor_2[64:128, 0:128]
```

**After:**

```python
output[0:128, 0:128] = tensor_2[0:128, 0:128]
```

If the merged first dimension would exceed 128, it is not an option:

```python
output[0:128, 0:128] = tensor_2[0:128, 0:128]
output[128:256, 0:128] = tensor_2[128:256, 0:128]
```

Merged dim 0 would be 0:256 (size 256 > 128) — **rejected**.

## Operator tile size constraints (Trn2 / NeuronCore-v3)

| Operator | Dimension | Max size |
|---|---|---|
| `load` | partition dim (dim 0) | 128 |
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
