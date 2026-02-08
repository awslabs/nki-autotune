# Matmul Coalesce

Combines two `nc_matmul` calls that share the same lhs and operate on
adjacent slices of the same rhs into a single wider `nc_matmul`.

## Example

Two `nc_matmul` calls share the same lhs (`tensor_0`) and operate on adjacent
slices of the same rhs (`tensor_1`):

**Before:**

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

**After** matmul coalesce (single wider `nc_matmul`):

```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 256), dtype=np.float32)

    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:256]
    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)
    output[0:128, 0:256] = tensor_2

    return output
```
