#!/usr/bin/env python3
from typing import List, Optional

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import FxOperator, GbOperator, HbOperator
from fusion.tensors import Tensor


class SumSquares(FxOperator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__([input_tensor])

    def forward(
        self, prev_output: Optional[Tensor], input_tensors: List[Tensor], curr_output: Tensor, reduction_axis: str
    ) -> None:
        assert (
            len(input_tensors) == 1
        ), f"SumSquares forward expects input_tensor, received {len(input_tensors)} tensors"
        input_tensor = input_tensors[0]
        block_sum_squares = np.sum(np.square(input_tensor.data), axis=input_tensor.axes.index(reduction_axis))
        if prev_output:
            curr_output.data = prev_output.data + block_sum_squares
        else:
            curr_output.data = block_sum_squares

    def initialize_output(self, input_tensors: List[Tensor], fusion_axis: str, output_tensor_name: str) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"SumSquares initialize output expects one input tensor to do init, received {len(input_tensors)}"
        tensor = input_tensors[0]
        init_sum_shape = tensor.get_parallel_shape(fusion_axis)
        init_sum_parallel_axes = tensor.get_parallel_axes(fusion_axis)
        init_sum = np.zeros(shape=init_sum_shape, dtype=np.float32)
        init_tensor = Tensor(name=output_tensor_name, axes=init_sum_parallel_axes, data=init_sum)
        return init_tensor


class NormFactor(GbOperator):
    def __init__(self, epsilon: float, num_elements: int) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.num_elements = num_elements

    def forward(self, dependent_output: Tensor, output_tensor: Tensor) -> None:
        norm_factor = 1 / np.sqrt(dependent_output.data / self.num_elements + self.epsilon)
        output_tensor.data = norm_factor

    def initialize_output(self, dependent_output: Tensor, output_tensor_name: str) -> Tensor:
        init_norm = np.zeros(shape=dependent_output.data.shape, dtype=np.float32)
        init_tensor = Tensor(name=output_tensor_name, axes=dependent_output.axes, data=init_norm)
        return init_tensor


class Matmul(HbOperator):
    def __init__(self, input_tensors: List[str]) -> None:
        super().__init__(input_tensors)

    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert len(input_tensors) == 2, f"Matmul forward expects LHS, RHS tensors, received {len(input_tensors)}"
        lhs, rhs = input_tensors
        output_tensor.data = np.matmul(lhs.data, rhs.data)

    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str, fusion_axis: str) -> Tensor:
        assert (
            len(input_tensors) == 2
        ), f"Matmul initialize_output expects LHS, RHS tensors, received {len(input_tensors)}"
        lhs, rhs = input_tensors
        output_shape = lhs.get_parallel_shape(fusion_axis) + rhs.get_parallel_shape(fusion_axis)
        output_axes = lhs.get_parallel_axes(fusion_axis) + rhs.get_parallel_axes(fusion_axis)
        init_matmul = np.zeros(shape=output_shape, dtype=lhs.data.dtype)
        init_tensor = Tensor(name=output_tensor_name, axes=output_axes, data=init_matmul)
        return init_tensor


def rmsnorm_matmul_golden(lhs: Tensor, rhs: Tensor, epsilon: float) -> np.ndarray:
    x = lhs.data
    weight = rhs.data

    # Explicit intermediate steps
    squares = x**2  # Explicit squares computation
    sum_of_squares = np.sum(squares, axis=-1, keepdims=True)  # Explicit sum of squares
    square_mean = sum_of_squares / x.shape[-1]  # Convert sum to mean

    rms = np.sqrt(square_mean + epsilon)
    x_normalized = x / rms
    result = np.matmul(x_normalized, weight)
    return result


def test_rmsnorm_matmul_fusion():
    """
    Test RMSNorm + MatMul fusion: rmsnorm(lhs) @ rhs

    Example dimensions:
    - lhs: (512, 1024) - sequence length x hidden dimension
    - rhs: (1024, 128) - hidden dimension x output dimension
    - Output: (512, 128)
    """
    seq_len = 128
    hidden_dim = 1024
    output_dim = 256
    epsilon = 1e-6
    atol = 1e-4
    rtol = 1e-4

    lhs = Tensor(name="LHS", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim).astype(np.float32))
    rhs = Tensor(name="RHS", axes=["hidden", "output"], data=np.random.randn(hidden_dim, output_dim).astype(np.float32))

    sum_squares_op = SumSquares("LHS")
    rms_factor_op = NormFactor(epsilon, hidden_dim)
    matmul_op = Matmul(["LHS", "RHS"])
    fusion = FusionChain(fx=sum_squares_op, gbs=[rms_factor_op], hbs=[matmul_op])
    result_fused = fusion.execute(fusion_axis="hidden", fusion_step_size=256, input_tensors=[lhs, rhs])
    # result_standard = fusion.execute(fusion_axis="hidden", fusion_step_size=hidden_dim, input_tensors=[lhs, rhs])
    # check_correctness(result_standard.data, result_fused.data, atol, rtol, verbose=True)

    golden = rmsnorm_matmul_golden(lhs, rhs, epsilon)
    # check_correctness(golden, result_standard.data, atol, rtol, verbose=True)
    check_correctness(golden, result_fused.data, atol, rtol, verbose=True)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
