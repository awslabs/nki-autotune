from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from fusion.tensors import Tensor


class Operator(ABC):
    def __init__(self, name: str, input_tensors: Tuple[str, ...]) -> None:
        self.name = name
        self.input_tensors = input_tensors

    @abstractmethod
    def step(self, input_tensors: Tuple[Tensor, ...]):
        raise NotImplementedError("Step for the base class is not implemented.")

    @abstractmethod
    def get_initial_value(self, input_tensors: Tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError("get_initial_value for the base class is not implemented.")

    def get_input_tensors(self, all_input_tensors: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        input_tensors: List[Tensor] = []
        for tensor_name in self.input_tensors:
            tensor = all_input_tensors[tensor_name]
            input_tensors.append(tensor)
        input_tensors_tuple = tuple(input_tensors)
        return input_tensors_tuple


class SumSquaresFx(Operator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__(name="Sum of Squares", input_tensors=(input_tensor,))

    def get_initial_value(self, input_tensors: Tuple[Tensor, ...]) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"SumSquaresFx get_initial_value expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_result = np.zeros(shape=input_tensor.parallel_shape, dtype=np.float32)
        tensor = Tensor(name="prev_sum_squares", axes=input_tensor.parallel_axes, data=init_result)
        return tensor

    def step(self, input_tensors: Tuple[Tensor, ...]) -> None:
        assert (
            len(input_tensors) == 2
        ), f"SumSquaresFx step expects two input tensors: prev_sum and input tensor, received {len(input_tensors)}"
        prev_sum, input_tensor = input_tensors
        squared = np.square(input_tensor.data)
        fusion_axis_index = input_tensor.axes.index(input_tensor.fusion_axis)
        sum_squared = np.sum(squared, axis=fusion_axis_index)
        prev_sum.data = prev_sum.data + sum_squared


class RMSNormGb(Operator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__(name="RMS Normalization", input_tensors=(input_tensor,))

    def get_initial_value(self, input_tensors: Tuple[Tensor, ...]):
        raise NotImplementedError("RMSNormGb has no initial value.")

    def step(self, input_tensors: Tuple[Tensor, ...]):
        assert len(input_tensors) == 1, f"RMSNormGb step expects one tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        print(input_tensor)
