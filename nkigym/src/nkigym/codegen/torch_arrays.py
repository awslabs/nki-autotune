"""Pure array helpers shared by generated Torch ABI adapters."""

from typing import Any, cast

import numpy as np
import torch

from nkigym.profile.types import _physical_numpy_dtype

ArrayResult = np.ndarray | tuple[np.ndarray, ...]


def _cast_output(array: np.ndarray, name: str) -> np.ndarray:
    """Cast one output through its physical ABI dtype."""
    value = np.asarray(array).astype(_physical_numpy_dtype(name), copy=False)
    return value.astype(np.float32, copy=False) if "float" in name else value


def _cast_output_dtypes(result: ArrayResult, output_dtypes: tuple[str, ...]) -> ArrayResult:
    """Cast adapted outputs to the dtypes allocated by the generated kernel."""
    arrays = result if isinstance(result, tuple) else (result,)
    casted = tuple(_cast_output(array, name) for array, name in zip(arrays, output_dtypes, strict=True))
    return casted[0] if len(casted) == 1 else casted


def as_numpy(value: object) -> np.ndarray:
    """Convert one Torch or array-like value to a CPU NumPy array."""
    tensor: Any = cast(torch.Tensor, value).detach().cpu() if isinstance(value, torch.Tensor) else value
    return np.asarray(tensor.float() if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16 else tensor)


def flatten_output_array(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Flatten one higher-rank output according to the generated matrix ABI."""
    if array.shape[-1] == 1:
        result = array.reshape(-1)
    elif array.ndim == 4 and array.shape[0] == 128:
        result = array.transpose(3, 1, 0, 2).reshape(array.shape[3], -1)
    elif array.ndim == 4 and np.prod(array.shape[1:-1]) == 1:
        result = array.reshape(1, -1)
    else:
        result = array.reshape(-1, array.shape[-1])
    return result.T if result.shape != target_shape and result.T.shape == target_shape else result


def logical_output_shape(
    output_shapes: tuple[tuple[int, ...], ...], output_groups: tuple[int, ...], index: int
) -> tuple[int, ...]:
    """Return one logical output shape before physical segmentation."""
    size = output_groups[index]
    shapes = output_shapes[sum(output_groups[:index]) :][:size]
    return (shapes[0][0], sum(shape[-1] for shape in shapes)) if size > 1 else shapes[0]
