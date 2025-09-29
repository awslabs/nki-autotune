from typing import List, Tuple

import numpy as np


class InTensor:
    def __init__(self, name: str, axes: Tuple[str, ...], blocking_axis: str, tensor: np.ndarray) -> None:
        assert blocking_axis in axes, f"Blocking axis {blocking_axis} is missing in {axes}."
        assert np.ndim(tensor) == len(
            axes
        ), f"Number of axes mismatch. Axes has {len(axes)}, tensor has {np.ndim(tensor)}."
        self.name = name
        self.axes = axes
        self.blocking_axis = blocking_axis
        self.parallel_axes: List[str] = []
        for axis in axes:
            if axis != blocking_axis:
                self.parallel_axes.append(axis)
        self.tensor = tensor

    def read(self, start: int, size: int) -> np.ndarray:
        slices = []
        for axis in self.axes:
            if axis == self.blocking_axis:
                slices.append(slice(start, start + size))
            else:
                slices.append(slice(None))
        return self.tensor[tuple(slices)]

    @property
    def blocking_size(self) -> int:
        blocking_axis_index = self.axes.index(self.blocking_axis)
        blocking_size = self.tensor.shape[blocking_axis_index]
        return blocking_size

    @property
    def parallel_shape(self) -> Tuple[int, ...]:
        """Returns the shape of parallel axes (all axes except blocking axis)."""
        shape = []
        for i, axis in enumerate(self.axes):
            if axis != self.blocking_axis:
                shape.append(self.tensor.shape[i])
        return tuple(shape)

    @property
    def full_shape(self) -> Tuple[int, ...]:
        """Returns the full shape of the tensor."""
        return self.tensor.shape

    def __repr__(self) -> str:
        return f"InTensor({self.name}: {self.axes}{self.tensor.shape}, blocking={self.blocking_axis})"
