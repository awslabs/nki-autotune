"""Tensor wrapper for fusion operations."""

from typing import List

import numpy as np


class Tensor:
    """Numpy array wrapper with fusion-specific helpers."""

    def __init__(self, name: str, axes: List[str], data: np.ndarray):
        self.name = name
        self.axes = axes
        self.data = data

    # def get_fusion_axis_slice

    def get_axis_size(self, axis: str) -> int:
        axis_index = self.axes.index(axis)
        size = self.data.shape[axis_index]
        return size

    def get_parallel_shape(self, fusion_axis: str) -> List[int]:
        parallel_shape = []
        for axis, size in zip(self.axes, self.data.shape):
            if axis != fusion_axis:
                parallel_shape.append(size)
        return parallel_shape

    def get_parallel_axes(self, fusion_axis: str) -> List[str]:
        parallel_axes = []
        for axis in self.axes:
            if axis != fusion_axis:
                parallel_axes.append(axis)
        return parallel_axes

    def __repr__(self) -> str:
        return f"Tensor(name={self.name}, axes={self.axes}, shape={self.data.shape}, dtype={self.data.dtype})"
