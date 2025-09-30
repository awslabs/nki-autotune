from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from fusion.tensors import Tensor


class Operator(ABC):
    def __init__(self, name: str, input_tensors: Tuple[str, ...]) -> None:
        self.name = name
        self.input_tensors = input_tensors
        self.step_count = 0

    @abstractmethod
    def step(
        self, step_count: int, block_size: int, input_tensors: Tuple[Tensor, ...], prev_result: Optional[Tensor] = None
    ):
        raise NotImplementedError("Step for the base class is not implemented.")

    @abstractmethod
    def initialize_result(self, input_tensors: Tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError("initialize_result for the base class is not implemented.")


class SumSquaresFx(Operator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__(name="Sum of Squares", input_tensors=(input_tensor,))

    def initialize_result(self, input_tensors: Tuple[Tensor, ...]) -> Tensor:
        assert len(input_tensors) == 1, f"SumSquaresFx expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_result = np.zeros(shape=input_tensor.parallel_shape, dtype=np.float32)
        tensor = Tensor(name="prev_sum_squares", axes=input_tensor.parallel_axes, data=init_result)
        return tensor

    def step(self, step_count: int, block_size: int, input_tensors: Tuple[Tensor, ...], prev_sum: Tensor) -> None:
        assert len(input_tensors) == 1, f"SumSquaresFx expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        data_slice = input_tensor.get_fusion_slice(start=step_count * block_size, size=block_size)
        squared = np.square(data_slice)
        fusion_axis_index = input_tensor.axes.index(input_tensor.fusion_axis)
        sum_squared = np.sum(squared, axis=fusion_axis_index)
        prev_sum.data = prev_sum.data + sum_squared
