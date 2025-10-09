#!/usr/bin/env python3
from typing import List, Optional

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import BiasOperator, FxOperator, ScaleOperator
from fusion.tensors import Tensor


class Rowmax(FxOperator):
    """
    Computes incremental row maximum for online fusion.
    O_0_k = max(O_0_{k-1}, rowmax(P_k))
    """

    def __init__(self, input_tensors: List[str], reduction_axis: str) -> None:
        super().__init__(input_tensors)
        self.reduction_axis = reduction_axis

    def forward(self, output_old: Optional[Tensor], input_tensors: List[Tensor], output_new: Tensor) -> None:
        assert len(input_tensors) == 1, f"Rowmax forward expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        row_max = np.max(input_tensor.data, axis=input_tensor.axes.index(self.reduction_axis))

        if output_old:
            output_new.data = np.maximum(output_old.data, row_max)
        else:
            output_new.data = row_max

    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str) -> Tensor:
        assert len(input_tensors) == 1, f"Rowmax expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_shape = input_tensor.get_parallel_shape(self.reduction_axis)
        init_parallel_axes = input_tensor.get_parallel_axes(self.reduction_axis)
        init_data = np.zeros(shape=init_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=init_parallel_axes, data=init_data)
        return init_tensor


class SumExpBias(BiasOperator):
    """
    Computes bias term for sum of exponentials: rowsum(exp(P_k - O_0_k))
    Used in O_1 accumulation for softmax denominator.
    """

    def __init__(self, input_tensors: List[str], reduction_axis: str) -> None:
        super().__init__(input_tensors)
        self.reduction_axis = reduction_axis

    def forward(self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert len(input_tensors) == 1, f"SumExpBias expects 1 input tensor, received {len(input_tensors)}"
        assert len(outputs_new) >= 1, f"SumExpBias needs O_0 (rowmax), received {len(outputs_new)} outputs"

        P_slice = input_tensors[0]
        # FIXME: needs manual tracking to know which outputs to use
        O_0_rowmax = outputs_new[0]  # Current rowmax O_0_k

        # Broadcast rowmax for subtraction: P_k - O_0_k
        rowmax_broadcasted = O_0_rowmax.data[:, np.newaxis]
        exp_normalized = np.exp(P_slice.data - rowmax_broadcasted)
        sum_exp = np.sum(exp_normalized, axis=P_slice.axes.index(self.reduction_axis))

        output_tensor.data = sum_exp

    def initialize_output(
        self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        assert len(input_tensors) == 1, f"SumExpBias expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        output_shape = input_tensor.get_parallel_shape(self.reduction_axis)
        output_axes = input_tensor.get_parallel_axes(self.reduction_axis)
        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=output_axes, data=init_data)
        return init_tensor


class SumExpScale(ScaleOperator):
    """
    Computes scaling factor for sum_exp accumulation: exp(O_0_{k-1} - O_0_k)
    This corrects the previous partial sum when rowmax changes.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor: Tensor) -> None:
        # FIXME: needs manual tracking to know which outputs to use
        O_0_old = outputs_old[0]  # Previous rowmax O_0_{k-1}
        O_0_new = outputs_new[0]  # Current rowmax O_0_k

        # Scale factor: exp(O_0_old - O_0_new)
        output_tensor.data = np.exp(O_0_old.data - O_0_new.data)

    def initialize_output(
        self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        O_0_old = outputs_old[0]
        init_tensor = Tensor(name=output_tensor_name, axes=O_0_old.axes, data=np.zeros(shape=O_0_old.data.shape))
        return init_tensor


class AttentionOutputBias(BiasOperator):
    """
    Computes attention output bias: (exp(P_k - O_0_k) / O_1_k) @ V_k
    This is the weighted sum with the value matrix.
    """

    def __init__(self, input_tensors: List[str], P_reduction_axis: str, matmul_axis: str) -> None:
        super().__init__(input_tensors)
        self.P_reduction_axis = P_reduction_axis
        self.matmul_axis = matmul_axis

    def forward(self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor: Tensor) -> None:
        assert len(input_tensors) == 2, f"AttentionOutputBias expects P and V tensors, received {len(input_tensors)}"
        assert len(outputs_new) >= 2, f"AttentionOutputBias needs O_0 and O_1, received {len(outputs_new)}"

        P_slice, V_slice = input_tensors
        O_0_rowmax, O_1_sum_exp = outputs_new

        # Compute softmax weights: exp(P_k - O_0_k) / O_1_k
        rowmax_broadcasted = O_0_rowmax.data[:, np.newaxis]
        """
        FIXME: exp_normalized computation can be cached.
        Need to allow operators to return multiple tensors and save in intermedaites.
        """
        exp_normalized = np.exp(P_slice.data - rowmax_broadcasted)

        sum_exp_broadcasted = O_1_sum_exp.data[:, np.newaxis]
        softmax_weights = exp_normalized / sum_exp_broadcasted

        # Matrix multiplication: softmax_weights @ V_k
        attention_output = np.matmul(softmax_weights, V_slice.data)
        output_tensor.data = attention_output

    def initialize_output(
        self, outputs_new: List[Tensor], input_tensors: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        assert len(input_tensors) == 2, f"AttentionOutputBias expects P and V, received {len(input_tensors)}"

        P_slice = input_tensors[0]
        V_slice = input_tensors[1]

        # Output shape: [seq_len, hidden_dim]
        output_shape = P_slice.get_parallel_shape(self.matmul_axis) + V_slice.get_parallel_shape(self.matmul_axis)
        output_axes = P_slice.get_parallel_axes(self.matmul_axis) + V_slice.get_parallel_axes(self.matmul_axis)

        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=output_axes, data=init_data)
        return init_tensor


class AttentionOutputScale(ScaleOperator):
    """
    Computes scaling factor for attention output: exp(O_0_{k-1} - O_0_k) * O_1_{k-1} / O_1_k
    This accounts for both rowmax change and normalization factor change.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor: Tensor) -> None:
        O_0_old = outputs_old[0]  # Previous rowmax
        O_0_new = outputs_new[0]  # Current rowmax
        O_1_old = outputs_old[1]  # Previous sum_exp
        O_1_new = outputs_new[1]  # Current sum_exp

        # Scale factor: exp(O_0_old - O_0_new) * O_1_old / O_1_new
        output_tensor.data = np.exp(O_0_old.data - O_0_new.data) * (O_1_old.data / O_1_new.data)

    def initialize_output(
        self, outputs_old: List[Tensor], outputs_new: List[Tensor], output_tensor_name: str
    ) -> Tensor:
        O_0_old = outputs_old[0]
        init_tensor = Tensor(name=output_tensor_name, axes=O_0_old.axes, data=np.zeros(shape=O_0_old.data.shape))
        return init_tensor


def flash_attention_golden(Q: Tensor, K: Tensor, V: Tensor) -> np.ndarray:
    """
    Standard attention computation: softmax(Q @ K^T) @ V

    Args:
        Q: Query tensor [seq_len, hidden_dim]
        K: Key tensor [seq_len, hidden_dim]
        V: Value tensor [seq_len, hidden_dim]

    Returns:
        Attention output [seq_len, hidden_dim]
    """
    # Compute attention scores: Q @ K^T
    P = np.matmul(Q.data, K.data.T)

    # Apply softmax
    rowmax = np.max(P, axis=-1, keepdims=True)
    exp_normalized = np.exp(P - rowmax)
    sum_exp = np.sum(exp_normalized, axis=-1, keepdims=True)
    softmax_weights = exp_normalized / sum_exp

    # Compute attention output: softmax @ V
    attention_output = np.matmul(softmax_weights, V.data)

    return attention_output


def test_flash_attention_fusion():
    """
    Test Flash Attention fusion: softmax(Q @ K^T) @ V

    Example dimensions:
    - Q: (seq_len, hidden_dim) - Query
    - K: (seq_len, hidden_dim) - Key
    - V: (seq_len, hidden_dim) - Value
    - P = Q @ K^T: (seq_len, seq_len) - Attention scores
    - Output: (seq_len, hidden_dim) - Attention output
    """
    seq_len = 128
    hidden_dim = 64
    atol = 1e-5
    rtol = 1e-5

    # Create input tensors
    Q = Tensor(name="Q", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    K = Tensor(name="K", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    V = Tensor(name="V", axes=["fusion", "hidden"], data=np.random.randn(seq_len, hidden_dim))

    # Precompute P = Q @ K^T
    P_data = np.matmul(Q.data, K.data.T)
    P = Tensor(name="P", axes=["seq", "fusion"], data=P_data)

    # Golden reference
    golden = flash_attention_golden(Q, K, V)

    # Create fusion chain operators
    rowmax_op = Rowmax(input_tensors=["P"], reduction_axis="fusion")
    sum_exp_bias_op = SumExpBias(input_tensors=["P"], reduction_axis="fusion")
    sum_exp_scale_op = SumExpScale()
    attention_bias_op = AttentionOutputBias(input_tensors=["P", "V"], P_reduction_axis="fusion", matmul_axis="fusion")
    attention_scale_op = AttentionOutputScale()

    # Create fusion chain
    fusion = FusionChain(
        fx=rowmax_op, bias_ops=[sum_exp_bias_op, attention_bias_op], scale_ops=[sum_exp_scale_op, attention_scale_op]
    )
    result_standard = fusion.execute(fusion_axis="fusion", fusion_step_size=seq_len, input_tensors=[P, V])
    result_fused = fusion.execute(fusion_axis="fusion", fusion_step_size=64, input_tensors=[P, V])
    check_correctness(golden, result_standard.data, atol, rtol, verbose=True)
    check_correctness(golden, result_fused.data, atol, rtol, verbose=True)


if __name__ == "__main__":
    test_flash_attention_fusion()
