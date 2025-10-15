#!/usr/bin/env python3
from typing import Dict, List

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import Operator
from fusion.tensors import Tensor


class Rowmax(Operator):
    """
    Computes incremental row maximum for online fusion.
    O_0_k = max(O_0_{k-1}, rowmax(P_k))
    """

    def __init__(self, input_tensors: List[str], reduction_axis: str) -> None:
        super().__init__(input_tensors)
        self.reduction_axis = reduction_axis

    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        assert len(input_tensors) == 1, f"Rowmax forward expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        row_max = np.max(input_tensor.data, axis=input_tensor.axes.index(self.reduction_axis))

        output_new = intermediate_tensors[f"O_{self.operator_index}_new"]
        if f"O_{self.operator_index}_old" in intermediate_tensors:
            output_old = intermediate_tensors[f"O_{self.operator_index}_old"]
            output_new.data = np.maximum(output_old.data, row_max)
        else:
            output_new.data = row_max

    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        assert len(input_tensors) == 1, f"Rowmax expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_shape = input_tensor.get_parallel_shape(self.reduction_axis)
        init_parallel_axes = input_tensor.get_parallel_axes(self.reduction_axis)
        init_data = np.zeros(shape=init_shape)
        init_tensor = Tensor(axes=init_parallel_axes, data=init_data)
        return {f"O_{self.operator_index}_new": init_tensor}


class SumExpBias(Operator):
    """
    Computes bias term for sum of exponentials: rowsum(exp(P_k - O_0_k))
    Also caches exp(P_k - O_0_k) for reuse in attention output computation.
    """

    def __init__(self, input_tensors: List[str], reduction_axis: str) -> None:
        super().__init__(input_tensors)
        self.reduction_axis = reduction_axis

    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        assert len(input_tensors) == 1, f"SumExpBias expects 1 input tensor, received {len(input_tensors)}"

        P_slice = input_tensors[0]
        O_0_rowmax = intermediate_tensors[f"O_{self.operator_index - 1}_new"]  # Current rowmax O_0_k

        # Broadcast rowmax for subtraction: P_k - O_0_k
        rowmax_broadcasted = O_0_rowmax.data[:, np.newaxis]
        exp_normalized = np.exp(P_slice.data - rowmax_broadcasted)

        # Cache exp_normalized for reuse in attention output
        intermediate_tensors[f"exp_normalized_{self.operator_index}"] = Tensor(axes=P_slice.axes, data=exp_normalized)

        # Compute sum of exponentials
        sum_exp = np.sum(exp_normalized, axis=P_slice.axes.index(self.reduction_axis))
        intermediate_tensors[f"bias_{self.operator_index}"].data = sum_exp

    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        assert len(input_tensors) == 1, f"SumExpBias expects 1 input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]

        # Initialize bias output
        output_shape = input_tensor.get_parallel_shape(self.reduction_axis)
        output_axes = input_tensor.get_parallel_axes(self.reduction_axis)
        init_data = np.zeros(shape=output_shape)
        bias_tensor = Tensor(axes=output_axes, data=init_data)

        # Initialize cache for exp_normalized
        exp_tensor = Tensor(axes=input_tensor.axes, data=np.zeros(shape=input_tensor.data.shape))

        return {f"bias_{self.operator_index}": bias_tensor, f"exp_normalized_{self.operator_index}": exp_tensor}


class SumExpScale(Operator):
    """
    Computes scaling factor for sum_exp accumulation: exp(O_0_{k-1} - O_0_k)
    This corrects the previous partial sum when rowmax changes.
    """

    def __init__(self) -> None:
        super().__init__(input_tensors=[])

    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        O_0_old = intermediate_tensors[f"O_{self.operator_index - 1}_old"]  # Previous rowmax O_0_{k-1}
        O_0_new = intermediate_tensors[f"O_{self.operator_index - 1}_new"]  # Current rowmax O_0_k

        # Scale factor: exp(O_0_old - O_0_new)
        scale_tensor = intermediate_tensors[f"scale_{self.operator_index}"]
        scale_tensor.data = np.exp(O_0_old.data - O_0_new.data)

    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        O_0_old = intermediate_tensors[f"O_{self.operator_index - 1}_old"]
        init_tensor = Tensor(axes=O_0_old.axes, data=np.zeros(shape=O_0_old.data.shape))
        return {f"scale_{self.operator_index}": init_tensor}


class AttentionOutputBias(Operator):
    """
    Computes attention output bias: (exp(P_k - O_0_k) / O_1_k) @ V_k
    Uses cached exp_normalized to avoid recomputation.
    """

    def __init__(self, input_tensors: List[str], P_reduction_axis: str, matmul_axis: str) -> None:
        super().__init__(input_tensors)
        self.P_reduction_axis = P_reduction_axis
        self.matmul_axis = matmul_axis

    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        assert len(input_tensors) == 2, f"AttentionOutputBias expects P and V tensors, received {len(input_tensors)}"

        P_slice, V_slice = input_tensors
        O_1_sum_exp = intermediate_tensors[f"O_{self.operator_index - 1}_new"]  # Current sum_exp

        # Retrieve cached exp_normalized from SumExpBias (operator_index 1)
        exp_normalized = intermediate_tensors["exp_normalized_1"].data

        # Compute softmax weights using cached exp_normalized
        sum_exp_broadcasted = O_1_sum_exp.data[:, np.newaxis]
        softmax_weights = exp_normalized / sum_exp_broadcasted

        # Matrix multiplication: softmax_weights @ V_k
        attention_output = np.matmul(softmax_weights, V_slice.data)
        intermediate_tensors[f"bias_{self.operator_index}"].data = attention_output

    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        assert len(input_tensors) == 2, f"AttentionOutputBias expects P and V, received {len(input_tensors)}"

        P_slice = input_tensors[0]
        V_slice = input_tensors[1]

        # Output shape: [seq_len, hidden_dim]
        output_shape = P_slice.get_parallel_shape(self.matmul_axis) + V_slice.get_parallel_shape(self.matmul_axis)
        output_axes = P_slice.get_parallel_axes(self.matmul_axis) + V_slice.get_parallel_axes(self.matmul_axis)

        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(axes=output_axes, data=init_data)
        return {f"bias_{self.operator_index}": init_tensor}


class AttentionOutputScale(Operator):
    """
    Computes scaling factor for attention output: exp(O_0_{k-1} - O_0_k) * O_1_{k-1} / O_1_k
    This accounts for both rowmax change and normalization factor change.
    """

    def __init__(self) -> None:
        super().__init__(input_tensors=[])

    def forward(self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]) -> None:
        O_0_old = intermediate_tensors[f"O_0_old"]  # Previous rowmax
        O_0_new = intermediate_tensors[f"O_0_new"]  # Current rowmax
        O_1_old = intermediate_tensors[f"O_1_old"]  # Previous sum_exp
        O_1_new = intermediate_tensors[f"O_1_new"]  # Current sum_exp

        # Scale factor: exp(O_0_old - O_0_new) * O_1_old / O_1_new
        scale_tensor = intermediate_tensors[f"scale_{self.operator_index}"]
        # Broadcasting for element-wise operations
        scale_tensor.data = np.exp(O_0_old.data - O_0_new.data)[:, np.newaxis] * (
            O_1_old.data[:, np.newaxis] / O_1_new.data[:, np.newaxis]
        )

    def initialize_output(
        self, intermediate_tensors: Dict[str, Tensor], input_tensors: List[Tensor]
    ) -> Dict[str, Tensor]:
        # Get shape from O_2 (attention output)
        attention_output = intermediate_tensors[f"bias_{self.operator_index}"]
        init_tensor = Tensor(axes=attention_output.axes, data=np.zeros(shape=attention_output.data.shape))
        return {f"scale_{self.operator_index}": init_tensor}


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
    Q = Tensor(axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    K = Tensor(axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim))
    V = Tensor(axes=["fusion", "hidden"], data=np.random.randn(seq_len, hidden_dim))

    # Precompute P = Q @ K^T
    P_data = np.matmul(Q.data, K.data.T)
    P = Tensor(axes=["seq", "fusion"], data=P_data)
    input_tensors = {"P": P, "V": V}

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

    # Test with full step size (standard execution)
    result_standard = fusion.execute(fusion_axis="fusion", fusion_step_size=seq_len, input_tensors=input_tensors)
    check_correctness(golden, result_standard.data, atol, rtol, verbose=True)

    # Test with smaller step size (fused execution)
    result_fused = fusion.execute(fusion_axis="fusion", fusion_step_size=32, input_tensors=input_tensors, verbose=True)
    check_correctness(golden, result_fused.data, atol, rtol, verbose=True)


if __name__ == "__main__":
    test_flash_attention_fusion()
