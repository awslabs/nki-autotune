#!/usr/bin/env python3
from typing import List

import numpy as np

from fusion.fusion_chain import FusionChain
from fusion.metrics import check_correctness
from fusion.operators import Operator
from fusion.tensors import Tensor


class SumSquares(Operator):
    def forward(self, inputs: List[Tensor], next_output: Tensor) -> None:
        pass

    def initialize_output(self, inputs: List[Tensor]) -> Tensor:
        pass


class RMSNormFactor(Operator):
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs: List[Tensor], next_output: Tensor) -> None:
        pass

    def initialize_output(self, inputs: List[Tensor]) -> Tensor:
        pass


class Matmul(Operator):
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

    np.random.seed(42)
    lhs = Tensor(name="LHS", axes=["seq", "hidden"], data=np.random.randn(seq_len, hidden_dim).astype(np.float32))
    rhs = Tensor(name="RHS", axes=["hidden", "output"], data=np.random.randn(hidden_dim, output_dim).astype(np.float32))

    sum_squares_op = SumSquares()
    rms_factor_op = RMSNormFactor(epsilon)
    matmul_op = Matmul()
    fusion = RMSNormMatmulFusion(fx=sum_squares_op, gbs=[rms_factor_op], hbs=[matmul_op])
    result_fused = fusion.execute(block_size=128, input_tensors=[lhs, rhs])
    result_standard = fusion.execute(block_size=hidden_dim, input_tensors=[lhs, rhs])

    check_correctness(result_standard.data, result_fused.data, 1e-4, 1e-4)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
