#!/usr/bin/env python3
from typing import List

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import Operator
from fusion.tensors import Tensor


def find_single_diff(list1: List[str], list2: List[str]):
    """
    Check if exactly one string is missing between two lists.
    Returns the missing string if found, otherwise raises an error.
    """
    set1 = set(list1)
    set2 = set(list2)

    # Find all differences between the two sets
    diff = set1.symmetric_difference(set2)

    if len(diff) == 1:
        return diff.pop()
    else:
        raise ValueError(f"Expected exactly 1 difference, found {len(diff)}")


class SumSquares(Operator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__([input_tensor])

    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert (
            len(input_tensors) == 2
        ), f"SumSquares forward expects prev_output, input_tensor, received {len(input_tensors)} tensors"
        prev_output, input_tensor = input_tensors
        sum_axis = find_single_diff(prev_output.axes, input_tensor.axes)
        block_sum_squares = np.sum(np.square(input_tensor.data), axis=input_tensor.axes.index(sum_axis))
        output_tensor.data = prev_output.data + block_sum_squares

    def initialize_output(self, input_tensors: List[Tensor], fusion_axis: str) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"SumSquares initialize output expects one input tensor to do init, received {len(input_tensors)}"
        tensor = input_tensors[0]
        init_sum_shape = tensor.get_parallel_shape(fusion_axis)
        init_sum_parallel_axes = tensor.get_parallel_axes(fusion_axis)
        init_sum = np.zeros(shape=init_sum_shape, dtype=np.float32)
        init_tensor = Tensor(name=f"prev_sum_squares", axes=init_sum_parallel_axes, data=init_sum)
        return init_tensor


class NormFactor(Operator):
    def __init__(self, input_tensor: str, epsilon: float, num_elements: int) -> None:
        super().__init__([input_tensor])
        self.epsilon = epsilon
        self.num_elements = num_elements

    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert len(input_tensors) == 1, f"NormFactor forward expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        norm_factor = 1 / np.sqrt(input_tensor.data / self.num_elements + self.epsilon)
        output_tensor.data = norm_factor

    def initialize_output(self, input_tensors: List[Tensor]) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"NormFactor initialize_output expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_norm = np.zeros(shape=input_tensor.data.shape, dtype=np.float32)
        init_tensor = Tensor(name=f"prev_rmsnorm_factor", axes=input_tensor.axes, data=init_norm)
        return init_tensor


class Matmul(Operator):
    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__(input_tensors)

    def forward(self, inputs: List[Tensor], next_output: Tensor) -> None:
        pass

    def initialize_output(self, inputs: List[Tensor]) -> Tensor:
        pass


class RMSNormMatmulFusion(FusionChain):
    def __init__(self, fx: Operator, gbs: List[Operator], hbs: List[Operator]):
        super().__init__(fx, gbs, hbs)


def test_rmsnorm_matmul_fusion():
    """
    Test RMSNorm + MatMul fusion: rmsnorm(lhs) @ rhs

    Example dimensions:
    - lhs: (512, 1024) - sequence length x hidden dimension
    - rhs: (1024, 128) - hidden dimension x output dimension
    - Output: (512, 128)
    """
    seq_len = 512
    hidden_dim = 1024
    output_dim = 128
    epsilon = 1e-6

    lhs = Tensor(name="LHS", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim).astype(np.float32))
    rhs = Tensor(name="RHS", axes=["hidden", "output"], data=np.random.randn(hidden_dim, output_dim).astype(np.float32))

    sum_squares_op = SumSquares("LHS")
    rms_factor_op = NormFactor("curr_O1", epsilon, hidden_dim)
    matmul_op = Matmul(["LHS", "RHS"])
    fusion = RMSNormMatmulFusion(fx=sum_squares_op, gbs=[rms_factor_op], hbs=[matmul_op])
    result_fused = fusion.execute(fusion_axis="hidden", fusion_step_size=128, input_tensors=[lhs, rhs])
    result_standard = fusion.execute(fusion_axis="hidden", fusion_step_size=hidden_dim, input_tensors=[lhs, rhs])
    check_correctness(result_standard.data, result_fused.data, 1e-4, 1e-4, verbose=True)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
