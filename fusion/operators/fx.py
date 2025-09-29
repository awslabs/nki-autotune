from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from fusion.tensors import InTensor


class FxOperator(ABC):
    def __init__(self, name: str, input_tensors: Tuple[str, ...]) -> None:
        self.name = name
        self.input_tensors = input_tensors
        self.step_count = 0

    @abstractmethod
    def step(self, input_tensors: Tuple[InTensor, ...]):
        raise NotImplementedError("Step for the base class is not implemented.")

    @abstractmethod
    def initialize_result(self, input_tensors: Tuple[InTensor, ...]) -> np.ndarray:
        raise NotImplementedError("initialize_result for the base class is not implemented.")


class SumSquaresFx(FxOperator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__(name="Sum of Squares", input_tensors=(input_tensor,))

    def initialize_result(self, input_tensors: Tuple[InTensor, ...]) -> np.ndarray:
        assert len(input_tensors) == 1, f"SumSquaresFx expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_result = np.zeros(shape=input_tensor.parallel_shape, dtype=np.float32)
        return init_result

    def step(self, input_tensors: Tuple[InTensor, ...]):
        print(f"SumSquaresFx input_tensors = {input_tensors}")
