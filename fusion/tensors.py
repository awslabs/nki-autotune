"""Tensor wrapper for fusion operations."""

import numpy as np


class Tensor:
    """Numpy array wrapper with fusion-specific helpers."""

    def __init__(self, axes: list[str], data: np.ndarray):
        self.axes = axes
        self.data = data

    def get_axis_slice(self, axis: str, start: int, size: int) -> "Tensor":
        axis_index = self.axes.index(axis)
        slices = [slice(None)] * len(self.data.shape)
        slices[axis_index] = slice(start, start + size)
        sliced_data = self.data[tuple(slices)]
        return Tensor(axes=self.axes, data=sliced_data)

    def get_axis_size(self, axis: str) -> int:
        axis_index = self.axes.index(axis)
        size = self.data.shape[axis_index]
        return size

    def get_parallel_shape(self, fusion_axis: str) -> list[int]:
        parallel_shape = []
        for axis, size in zip(self.axes, self.data.shape):
            if axis != fusion_axis:
                parallel_shape.append(size)
        return parallel_shape

    def get_parallel_axes(self, fusion_axis: str) -> list[str]:
        parallel_axes = []
        for axis in self.axes:
            if axis != fusion_axis:
                parallel_axes.append(axis)
        return parallel_axes

    def __repr__(self) -> str:
        return f"Tensor(axes={self.axes}, shape={self.data.shape}, dtype={self.data.dtype})"
