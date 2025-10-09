#!/usr/bin/env python3
from typing import List, Optional

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.fusion_chain import FusionChain
from fusion.operators import FxOperator, GbOperator, HbOperator
from fusion.tensors import Tensor


class Rowmax(FxOperator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__([input_tensor])

    def forward(
        self, prev_output: Optional[Tensor], input_tensors: List[Tensor], curr_output: Tensor, reduction_axis: str
    ) -> None:
        assert len(input_tensors) == 1, f"Rowmax forward expects input_tensor, received {len(input_tensors)} tensors"
        input_tensor = input_tensors[0]
        row_max = np.max(input_tensor.data, axis=input_tensor.axes.index(reduction_axis))
        if prev_output:
            curr_output.data = np.maximum(prev_output.data, row_max)
        else:
            curr_output.data = row_max

    def initialize_output(self, input_tensors: List[Tensor], fusion_axis: str, output_tensor_name: str) -> Tensor:
        assert (
            len(input_tensors) == 1
        ), f"Rowmax initialize output expects one input tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        init_shape = input_tensor.get_parallel_shape(fusion_axis)
        init_parallel_axes = input_tensor.get_parallel_axes(fusion_axis)
        init_data = np.zeros(shape=init_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=init_parallel_axes, data=init_data)
        return init_tensor


class SumExp(HbOperator):
    def __init__(self, input_tensor: str) -> None:
        super().__init__([input_tensor])

    def forward(self, input_tensors: List[Tensor], output_tensor: Tensor, fusion_axis: str) -> None:
        assert len(input_tensors) == 1, f"SumExp forward expects one tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        output_tensor.data = np.sum(np.exp(input_tensor.data), axis=input_tensor.axes.index(fusion_axis))

    def initialize_output(self, input_tensors: List[Tensor], output_tensor_name: str, fusion_axis: str) -> Tensor:
        assert len(input_tensors) == 1, f"SumExp forward expects one tensor, received {len(input_tensors)}"
        input_tensor = input_tensors[0]
        output_shape = input_tensor.get_parallel_shape(fusion_axis)
        output_axes = input_tensor.get_parallel_axes(fusion_axis)
        init_data = np.zeros(shape=output_shape)
        init_tensor = Tensor(name=output_tensor_name, axes=output_axes, data=init_data)
        return init_tensor


class SumExpGb(GbOperator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, dependent_output: Tensor, output_tensor: Tensor) -> None:
        output_tensor.data = 1 / np.exp(dependent_output.data)

    def initialize_output(self, dependent_output: Tensor, output_tensor_name: str) -> Tensor:
        init_data = np.zeros(shape=dependent_output.data.shape)
        init_tensor = Tensor(name=output_tensor_name, axes=dependent_output.axes, data=init_data)
        return init_tensor


def softmax_matmul_golden(P: Tensor, V: Tensor) -> np.ndarray:
    p_mat = P.data
    v_mat = V.data

    rowmax = np.max(p_mat, axis=-1)
    sum_exp = np.sum(np.exp(p_mat - rowmax), axis=-1)
    return sum_exp


def test_softmax_matmul_fusion():
    seq_len = 1024
    hidden_dim = 512
    atol = 1e-4
    rtol = 1e-4
    P = Tensor(name="P", axes=["seq", "fusion"], data=np.random.randn(seq_len, seq_len))
    V = Tensor(name="V", axes=["fusion", "seq"], data=np.random.randn(seq_len, hidden_dim))
    input_tensors = [P, V]

    golden = softmax_matmul_golden(P, V)

    rowmax_op = Rowmax("P")
    sum_exp_op = SumExp("P")
    sum_exp_gb_op = SumExpGb()
    fusion = FusionChain(fx=rowmax_op, gbs=[sum_exp_gb_op], hbs=[sum_exp_op])
    result_standard = fusion.execute(fusion_axis="fusion", fusion_step_size=seq_len, input_tensors=input_tensors)
    check_correctness(golden, result_standard.data, atol, rtol, verbose=True)
    result_fused = fusion.execute(fusion_axis="fusion", fusion_step_size=256, input_tensors=input_tensors)
    check_correctness(golden, result_fused.data, atol, rtol, verbose=True)


if __name__ == "__main__":
    test_softmax_matmul_fusion()
