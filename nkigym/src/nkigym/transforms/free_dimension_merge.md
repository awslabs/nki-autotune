# Free Dimension Merge

Merges the free dimension of adjacent tensor tiles into a single larger tile.
For example, two (128, 128) tiles can be merged into one (128, 256) tile along
the free axis.

For `nc_matmul`, the moving free dimension (N) supports up to 512 elements,
so up to 4 adjacent 128-wide N tiles can be merged into one.

## Merge operation

Two adjacent rhs tiles are combined into one wider tile. The merged tile
adopts the name of the first tensor.

**Before** (N tiled at 128):

```python
tensor_1 = b[0:128, 0:128]
tensor_4 = b[0:128, 128:256]
```

**After** (N merged to 256, `tensor_4` absorbed into `tensor_1`):

```python
tensor_1 = b[0:128, 0:256]
```

## Full example

A (128, 128) @ (128, 256) matmul tiled at 128 produces 2 subgraphs
(1 M-tile x 2 N-tiles):

**Before** (2 subgraphs):

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 256), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5

    return output
```

### Step 1: Analysis

Identify tiles that can be merged in their free dimension. `tensor_1` and
`tensor_4` load from the same source tensor `b` with the same partition
slice `[0:128, :]` and adjacent free dimension slices (`0:128`, `128:256`):

```
merge group: (tensor_1, tensor_4)  ->  b[0:128, 0:256]
```

### Step 2: Transform

Apply the merge for `(tensor_1, tensor_4)`. The two loads are replaced by a
single wider load under `tensor_1`. Downstream `nc_matmul` calls that
previously consumed `tensor_1` or `tensor_4` now slice into the merged tile:

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 256), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:256]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_1[0:128, 128:256])
    output[0:128, 128:256] = tensor_5

    return output
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
