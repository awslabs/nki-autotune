"""Pure array helpers shared by generated Torch ABI adapters."""

from typing import Any, cast

import numpy as np
import torch


def as_numpy(value: object) -> np.ndarray:
    """Convert one Torch or array-like value to a CPU NumPy array."""
    tensor: Any = cast(torch.Tensor, value).detach().cpu() if isinstance(value, torch.Tensor) else value
    return np.asarray(tensor.float() if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16 else tensor)


def flatten_output_array(array: np.ndarray) -> np.ndarray:
    """Flatten one higher-rank output according to the generated matrix ABI."""
    if array.shape[-1] == 1:
        return array.reshape(-1)
    if array.ndim == 4 and array.shape[0] == 128:
        return array.transpose(3, 1, 0, 2).reshape(array.shape[3], -1)
    if array.ndim == 4 and np.prod(array.shape[1:-1]) == 1:
        return array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1])


def logical_output_shape(
    output_shapes: tuple[tuple[int, ...], ...], output_groups: tuple[int, ...], index: int
) -> tuple[int, ...]:
    """Return one logical output shape before physical segmentation."""
    size = output_groups[index]
    shapes = output_shapes[sum(output_groups[:index]) :][:size]
    return (shapes[0][0], sum(shape[-1] for shape in shapes)) if size > 1 else shapes[0]
