# Transforms

Atomic transforms that rewrite tiled compute graphs. Each transform is
independent; the autotuner searches over combinations to find the best kernel.

| Transform | What it does | Doc |
|---|---|---|
| Free dimension merge | Merges adjacent rhs loads into a wider tile | [free_dimension_merge.md](free_dimension_merge.md) |
| Data reuse | Deduplicates identical tensor loads | [data_reuse.md](data_reuse.md) |
| Matmul coalesce | Combines adjacent `nc_matmul` calls into one wider call | [matmul_coalesce.md](matmul_coalesce.md) |

## Example: all three transforms applied

A (128, 128) @ (128, 256) matmul tiled at 128 produces 2 subgraphs:

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

After [free dimension merge](free_dimension_merge.md):

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

After data reuse (`tensor_3` duplicates `tensor_0`):

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 256), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:256]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2

    tensor_5 = nkigym.nc_matmul(tensor_0, tensor_1[0:128, 128:256])
    output[0:128, 128:256] = tensor_5

    return output
```

After [matmul coalesce](matmul_coalesce.md):

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 256), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:256]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:256] = tensor_2

    return output
```
