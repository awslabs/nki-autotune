#!/usr/bin/env python3

import numpy as np

from .fusion_chain import FusionChain
from .operators import Operator
from .tensors import Tensor


class SumSquares(Operator):
    """Computes incremental sum of squares along a given axis for RMSNorm."""

    def __init__(self, input_tensors: list[str], sum_axis: str) -> None:
        """Initialize SumSquares operator.

        Args:
            input_tensors: Names of input tensors.
            sum_axis: Axis along which to sum squares.
        """
        super().__init__(input_tensors)
        self.sum_axis = sum_axis

    def forward(self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]) -> None:
        """Accumulate sum of squared elements from the input tensor.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Single input tensor to square and sum.
        """
        assert (
            len(input_tensors) == 1
        ), f"SumSquares forward expects input_tensor, received {len(input_tensors)} tensors"
        input_tensor = input_tensors[0]
        sum_squares = np.sum(np.square(input_tensor.data), axis=input_tensor.axes.index(self.sum_axis))
        output_new = intermediate_tensors[f"O_{self.operator_index}_new"]
        if f"O_{self.operator_index}_old" in intermediate_tensors:
            output_old = intermediate_tensors[f"O_{self.operator_index}_old"]
            output_new.data = output_old.data + sum_squares
        else:
            output_new.data = sum_squares

    def initialize_output(
        self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]
    ) -> dict[str, Tensor]:
        """Allocate zero-initialized accumulator for sum of squares.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Single input tensor used to determine output shape.

        Returns:
            Dictionary mapping output name to initialized Tensor.
        """
        assert (
            len(input_tensors) == 1
        ), f"SumSquares initialize output expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_sum_shape = input_tensor.get_parallel_shape(self.sum_axis)
        init_sum_parallel_axes = input_tensor.get_parallel_axes(self.sum_axis)
        init_sum = np.zeros(shape=init_sum_shape)
        init_tensor = Tensor(axes=init_sum_parallel_axes, data=init_sum)
        return {f"O_{self.operator_index}_new": init_tensor}


class RMSNormMatmul(Operator):
    """Fused RMSNorm normalization followed by matrix multiplication."""

    def __init__(self, input_tensors: list[str], epsilon: float, num_elements: int, matmul_axis: str) -> None:
        """Initialize RMSNormMatmul operator.

        Args:
            input_tensors: Names of input tensors (LHS and RHS).
            epsilon: Small constant for numerical stability in normalization.
            num_elements: Number of elements used to compute the mean of squares.
            matmul_axis: Axis contracted in the matrix multiplication.
        """
        super().__init__(input_tensors)
        self.epsilon = epsilon
        self.num_elements = num_elements
        self.matmul_axis = matmul_axis

    def forward(self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]) -> None:
        """Compute normalized matmul: (lhs / rms_norm) @ rhs.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Two tensors: LHS (to normalize) and RHS (weight).
        """
        assert (
            len(input_tensors) == 2
        ), f"RMSNormMatmul forward expects two input tensors, received {len(input_tensors)}"
        lhs, rhs = input_tensors
        sum_squares = intermediate_tensors[f"O_{self.operator_index - 1}_new"]

        mat_prod = np.matmul(lhs.data, rhs.data)
        norm_factor = np.sqrt(sum_squares.data / self.num_elements + self.epsilon)
        intermediate_tensors[f"bias_{self.operator_index}"].data = mat_prod / norm_factor[:, None]

    def initialize_output(
        self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]
    ) -> dict[str, Tensor]:
        """Allocate zero-initialized output tensor for fused matmul result.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Two tensors (LHS, RHS) used to determine output shape.

        Returns:
            Dictionary mapping bias name to initialized Tensor.
        """
        assert (
            len(input_tensors) == 2
        ), f"RMSNormMatmul initialize_output expects LHS, RHS tensors, received {len(input_tensors)}."
        lhs, rhs = input_tensors
        output_shape = lhs.get_parallel_shape(self.matmul_axis) + rhs.get_parallel_shape(self.matmul_axis)
        output_axes = lhs.get_parallel_axes(self.matmul_axis) + rhs.get_parallel_axes(self.matmul_axis)
        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(axes=output_axes, data=init_data)
        return {f"bias_{self.operator_index}": init_tensor}


class Scale(Operator):
    """Computes RMSNorm correction scale when partial sum of squares changes."""

    def __init__(self, epsilon: float, num_elements: int) -> None:
        """Initialize Scale operator.

        Args:
            epsilon: Small constant for numerical stability.
            num_elements: Number of elements used to compute the mean of squares.
        """
        super().__init__(input_tensors=[])
        self.epsilon = epsilon
        self.num_elements = num_elements

    def forward(self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]) -> None:
        """Compute ratio of old to new RMS normalization factors.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Unused (scale derived from intermediate state).
        """
        output_old = intermediate_tensors[f"O_{self.operator_index-1}_old"]
        output_new = intermediate_tensors[f"O_{self.operator_index-1}_new"]
        scale_tensor = intermediate_tensors[f"scale_{self.operator_index}"]
        scale_tensor.data = np.sqrt(output_old.data / self.num_elements + self.epsilon) / np.sqrt(
            output_new.data / self.num_elements + self.epsilon
        )

    def initialize_output(
        self, intermediate_tensors: dict[str, Tensor], input_tensors: list[Tensor]
    ) -> dict[str, Tensor]:
        """Allocate zero-initialized scale tensor.

        Args:
            intermediate_tensors: Shared tensor state across operators.
            input_tensors: Unused.

        Returns:
            Dictionary mapping scale name to initialized Tensor.
        """
        output_old = intermediate_tensors[f"O_{self.operator_index-1}_old"]
        init_tensor = Tensor(axes=output_old.axes, data=np.zeros(shape=output_old.data.shape))
        return {f"scale_{self.operator_index}": init_tensor}


def rmsnorm_matmul_golden(lhs: Tensor, rhs: Tensor, epsilon: float) -> np.ndarray:
    """Compute golden reference for RMSNorm + MatMul: rmsnorm(lhs) @ rhs.

    Args:
        lhs: Input tensor to normalize.
        rhs: Weight tensor for the matrix multiplication.
        epsilon: Small constant for numerical stability.

    Returns:
        Result of normalized lhs multiplied by rhs.
    """
    x = lhs.data
    weight = rhs.data

    squares = x**2
    sum_of_squares = np.sum(squares, axis=-1, keepdims=False)
    square_mean = sum_of_squares / x.shape[-1]

    rms = np.sqrt(square_mean + epsilon)
    x_normalized = x / rms[:, None]
    result = np.matmul(x_normalized, weight)
    return result


def test_rmsnorm_matmul_fusion() -> None:
    """Test RMSNorm + MatMul fusion: rmsnorm(lhs) @ rhs.

    Verifies that the fused online computation matches the standard
    RMSNorm golden reference for both full and tiled step sizes.
    """
    seq_len = 128
    hidden_dim = 1024
    output_dim = 256
    epsilon = 1e-6
    atol = 1e-5
    rtol = 1e-5

    lhs = Tensor(axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    rhs = Tensor(axes=["hidden", "output"], data=np.random.randn(hidden_dim, output_dim))
    input_tensors = {"LHS": lhs, "RHS": rhs}

    fusion = FusionChain(
        fx=SumSquares(["LHS"], sum_axis="hidden"),
        bias_ops=[RMSNormMatmul(["LHS", "RHS"], epsilon, hidden_dim, matmul_axis="hidden")],
        scale_ops=[Scale(epsilon, hidden_dim)],
    )
    result_fused = fusion.execute(fusion_axis="hidden", fusion_step_size=256, input_tensors=input_tensors, verbose=True)
    result_standard = fusion.execute(fusion_axis="hidden", fusion_step_size=hidden_dim, input_tensors=input_tensors)
    golden = rmsnorm_matmul_golden(lhs, rhs, epsilon)
    np.testing.assert_allclose(result_standard.data, golden, atol=atol, rtol=rtol)
    np.testing.assert_allclose(result_fused.data, golden, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
