# Data Reuse

Deduplicates identical tensor loads across subgraphs. When two variables load
the same source tensor with the same slice, one is removed and all references
are rewritten to the surviving variable.

## Merge operation

Two variables with identical loads are collapsed into one. The merged variable
adopts the name of the first tensor.

**Before** (duplicate loads of `a[0:128, 0:128]`):

```python
tensor_0 = a[0:128, 0:128]
tensor_3 = a[0:128, 0:128]
```

**After** (`tensor_3` removed, references rewritten to `tensor_0`):

```python
tensor_0 = a[0:128, 0:128]
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
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_4)
    output[128:256, 0:128] = tensor_5

    return output
```

### Step 1: Analysis

Identify variables that load the same source tensor with the same slice.
`tensor_1` and `tensor_4` both load `b[0:128, 0:128]`:

```
reuse group: (tensor_1, tensor_4)
```

### Step 2: Transform

Remove `tensor_4`'s assignment and rewrite all references to `tensor_1`:

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 128), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3, tensor_1)
    output[128:256, 0:128] = tensor_5

    return output
```
