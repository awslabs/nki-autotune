# Data Reuse

Deduplicates identical tensor loads across subgraphs. When two variables load
the same source tensor with the same slice, one is removed and all references
are rewritten to the surviving variable.

Like all transforms, data reuse has two steps:

1. **`analyze(func)`** — inspect the IR and return a list of reuse pairs.
2. **`transform(func, pair)`** — apply a single pair, returning a new callable.

## Merge operation

Two variables with identical loads are collapsed into one. The merged variable
adopts the name of the earlier variable.

**Before** (duplicate loads of `a[0:128, 0:128]`):

```python
tensor_0 = a[0:128, 0:128]
tensor_3 = a[0:128, 0:128]
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
```

**After** (`tensor_3` removed, references rewritten to `tensor_0`):

```python
tensor_0 = a[0:128, 0:128]
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
tensor_5 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_4[0:128, 0:128])
```

## Full example

A (256, 128) @ (128, 128) matmul tiled at 128 produces 2 subgraphs
(2 M-tiles x 1 N-tile). Both subgraphs load the same `b` slice:

**Before** (2 subgraphs):

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 128), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[128:256, 0:128] = tensor_5[0:128, 0:128]

    return output
```

### Step 1: Analysis

Identify variables that load the same source tensor with the same slice.
`tensor_1` and `tensor_4` both load `b[0:128, 0:128]`:

```
reuse pair: (tensor_1, tensor_4)
```

### Step 2: Transform

Remove `tensor_4`'s assignment and rewrite all references to `tensor_1`:

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 128), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    tensor_3 = a[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_1[0:128, 0:128])
    output[128:256, 0:128] = tensor_5[0:128, 0:128]

    return output
```
