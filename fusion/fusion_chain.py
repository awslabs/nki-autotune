from typing import Dict, Tuple

import numpy as np

from fusion.operators.fx import FxOperator, SumSquaresFx
from fusion.tensors import InTensor


class FusionChain:
    """
    Main fusion chain orchestrator using IR-guided fusion strategy.

    This class builds a chain of operations, analyzes their dependencies,
    and generates an optimized kernel using online fusion.
    """

    def __init__(self, name: str, input_tensors: Tuple[InTensor, ...], fx: FxOperator):
        self.name = name
        self.input_tensors: Dict[str, InTensor] = {}
        for intensor in input_tensors:
            self.input_tensors[intensor.name] = intensor
        if not all([input_tensors[0].blocking_size == intensor.blocking_size for intensor in input_tensors]):
            raise NotImplementedError("Different blocking axis sizes are not supported")
        self.fx = fx
        self.gbs = []
        self.hbs = []

    def add_operator(self, gb, hb):
        self.gbs.append(gb)
        self.hbs.append(hb)

    def initialze_outputs(self):
        fx_inputs = tuple([self.input_tensors[tensor_name] for tensor_name in self.fx.input_tensors])
        prev_fx_result = self.fx.initialize_result(input_tensors=fx_inputs)
        print(prev_fx_result.shape, prev_fx_result)
        for gb, hb in zip(self.gbs, self.hbs):
            print(gb, hb)

    def step(self):
        pass


def sum_of_squares(prev_sum: np.ndarray, input_tensors: Tuple[np.ndarray, ...], blocking_axis_idx: int) -> np.ndarray:
    input_tensor = input_tensors[0]
    squared = np.square(input_tensor)
    sum_of_squares = np.sum(squared, axis=blocking_axis_idx)
    new_sum = prev_sum + sum_of_squares
    return new_sum


if __name__ == "__main__":
    M = 1024
    N = 512
    K = 2048
    data_type = np.float32
    lhs = InTensor(
        name="lhs", blocking_axis="K", axes=("M", "K"), tensor=np.random.normal(size=(M, K)).astype(data_type)
    )
    rhs = InTensor(
        name="rhs", blocking_axis="K", axes=("K", "N"), tensor=np.random.normal(size=(K, N)).astype(data_type)
    )

    chain = FusionChain(name="RMSNorm+Matmul", input_tensors=(lhs, rhs), fx=SumSquaresFx("lhs"))
    chain.initialze_outputs()
