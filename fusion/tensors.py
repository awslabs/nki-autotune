from typing import Optional, Tuple

import numpy as np


class Tensor:
    def __init__(self, name: str, axes: Tuple[str, ...], data: np.ndarray, fusion_axis: Optional[str] = None) -> None:
        if fusion_axis:
            assert fusion_axis in axes, f"Fusion axis {fusion_axis} is missing in {axes}."
        assert np.ndim(data) == len(axes), f"Number of axes mismatch. Axes has {len(axes)}, data has {np.ndim(data)}."
        self.name = name
        self.axes = axes
        self.fusion_axis = fusion_axis
        self.data = data

    def get_fusion_slice(self, start: int, size: int) -> "Tensor":
        slices = []
        for axis in self.axes:
            if axis == self.fusion_axis:
                slices.append(slice(start, start + size))
            else:
                slices.append(slice(None))
        data_slice = self.data[tuple(slices)]
        tensor_slice = Tensor(name=f"{self.name}_slice", axes=self.axes, data=data_slice, fusion_axis=self.fusion_axis)
        return tensor_slice

    @property
    def parallel_axes(self) -> Tuple[str]:
        parallel_axes = []
        for axis in self.axes:
            if axis != self.fusion_axis:
                parallel_axes.append(axis)
        return tuple(parallel_axes)

    @property
    def fusion_size(self) -> int:
        fusion_axis_index = self.axes.index(self.fusion_axis)
        fusion_size = self.data.shape[fusion_axis_index]
        return fusion_size

    @property
    def parallel_shape(self) -> Tuple[int, ...]:
        """Returns the shape of parallel axes (all axes except fusion axis)."""
        shape = []
        for i, axis in enumerate(self.axes):
            if axis != self.fusion_axis:
                shape.append(self.data.shape[i])
        return tuple(shape)

    @property
    def full_shape(self) -> Tuple[int, ...]:
        """Returns the full shape of the data."""
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(name={self.name}, axes={self.axes}, shape={self.full_shape}, fusion_axis={self.fusion_axis}, data={self.data})"
