#!/usr/bin/env python3
from typing import List, Optional

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import BiasOperator, FxOperator, ScaleOperator
from fusion.tensors import Tensor


class SumSquares(FxOperator):
    def __init__(self, input_tensors: List[str], sum_axis: str) -> None:
        super().__init__(input_tensors)
        self.sum_axis = sum_axis

    def forward(self, output_old: Optional[Tensor], input_tensors: List[Tensor], output_new: Tensor) -> None:
        assert (
            len(input_tensors) == 1
        ), f"SumSquares forward expects input_tensor, received {len(input_tensors)} tensors"
        input_tensor = input_tensors[0]
        sum_squares = np.sum(np.square(input_tensor.data), axis=input_tensor.axes.index(self.sum_axis))
        if output_old:
            output_new.data = output_old.data + sum_squares
        else:
            output_new.data = sum_squares

    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"SumSquares initialize output expects one input tensor, received {len(input_tensors)}"
        tensor = input_tensors[0]
        init_sum_shape = tensor.get_parallel_shape(self.sum_axis)
        init_sum_parallel_axes = tensor.get_parallel_axes(self.sum_axis)
        init_sum = np.zeros(shape=init_sum_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=init_sum_parallel_axes, data=init_sum)
        return init_tensor


class RMSNormMatmul(BiasOperator):
    def __init__(self, input_tensors: List[str], epsilon: float, num_elements: int, matmul_axis: str) -> None:
        super().__init__(input_tensors)
        self.epsilon = epsilon
        self.num_elements = num_elements
        self.matmul_axis = matmul_axis

    def forward(self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert (
            len(input_tensors) == 2
        ), f"RMSNormMatmul forward expects two input tensors, received {len(input_tensors)}"
        sum_squares = outputs_new[-1]
        lhs, rhs = input_tensors

        mat_prod = np.matmul(lhs.data, rhs.data)
        norm_factor = np.sqrt(sum_squares.data / self.num_elements + self.epsilon)
        output_tensor.data = mat_prod / norm_factor[:, None]

    def initialize_output(
        self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        assert (
            len(input_tensors) == 2
        ), f"RMSNormMatmul initialize_output expects LHS, RHS tensors, received {len(input_tensors)}."
        lhs, rhs = input_tensors
        output_shape = lhs.get_parallel_shape(self.matmul_axis) + rhs.get_parallel_shape(self.matmul_axis)
        output_axes = lhs.get_parallel_axes(self.matmul_axis) + rhs.get_parallel_axes(self.matmul_axis)
        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=output_axes, data=init_data)
        return init_tensor


class Scale(ScaleOperator):
    def __init__(self, epsilon: float, num_elements: int) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.num_elements = num_elements

    def forward(self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor: Tensor) -> None:
        output_old = outputs_old[-1]
        output_new = outputs_new[-1]
        output_tensor.data = np.sqrt(output_old.data / self.num_elements + self.epsilon) / np.sqrt(
            output_new.data / self.num_elements + self.epsilon
        )

    def initialize_output(
        self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        output_old = outputs_old[-1]
        init_tensor = Tensor(name=output_tensor_name, axes=output_old.axes, data=np.zeros(shape=output_old.data.shape))
        return init_tensor


def rmsnorm_matmul_golden(lhs: Tensor, rhs: Tensor, epsilon: float) -> np.ndarray:
    x = lhs.data
    weight = rhs.data

    # Explicit intermediate steps
    squares = x**2  # Explicit squares computation
    sum_of_squares = np.sum(squares, axis=-1, keepdims=False)  # Explicit sum of squares
    square_mean = sum_of_squares / x.shape[-1]  # Convert sum to mean

    rms = np.sqrt(square_mean + epsilon)
    x_normalized = x / rms[:, None]
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
    atol = 1e-5
    rtol = 1e-5

    lhs = Tensor(name="LHS", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    rhs = Tensor(name="RHS", axes=["hidden", "output"], data=np.random.randn(hidden_dim, output_dim))

    sum_squares_op = SumSquares(["LHS"], sum_axis="hidden")
    rms_matmul_op = RMSNormMatmul(["LHS", "RHS"], epsilon, hidden_dim, matmul_axis="hidden")
    scale_op = Scale(epsilon, hidden_dim)
    fusion = FusionChain(fx=sum_squares_op, bias_ops=[rms_matmul_op], scale_ops=[scale_op])
    result_fused = fusion.execute(fusion_axis="hidden", fusion_step_size=256, input_tensors=[lhs, rhs])
    result_standard = fusion.execute(fusion_axis="hidden", fusion_step_size=hidden_dim, input_tensors=[lhs, rhs])
    golden = rmsnorm_matmul_golden(lhs, rhs, epsilon)
    check_correctness(golden, result_standard.data, atol, rtol, verbose=True)
    check_correctness(golden, result_fused.data, atol, rtol, verbose=True)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
