import math
from typing import Dict, Tuple

import numpy as np

from autotune.core.metrics import check_correctness
from fusion.operators import Operator, SumSquaresFx
from fusion.tensors import Tensor


class FusionChain:
    """
    Main fusion chain orchestrator using IR-guided fusion strategy.

    This class builds a chain of operations, analyzes their dependencies,
    and generates an optimized kernel using online fusion.
    """

    def __init__(self, name: str, input_tensors: Tuple[Tensor, ...], fx: Operator):
        self.name = name
        self.input_tensors: Dict[str, Tensor] = {}
        fusion_sizes = []
        for intensor in input_tensors:
            self.input_tensors[intensor.name] = intensor
            if intensor.fusion_axis:
                fusion_sizes.append(intensor.fusion_size)
        assert fusion_sizes, f"Input tensors must have fusion axis."
        if not all([x == fusion_sizes[0] for x in fusion_sizes]):
            raise NotImplementedError(f"Different fusion sizes is not supported.")
        self.fusion_size = fusion_sizes[0]
        self.fx = fx
        self.gbs = []
        self.hbs = []

    def add_operator(self, gb, hb):
        self.gbs.append(gb)
        self.hbs.append(hb)

    def run(self, block_size: int):
        num_fusion_steps = math.ceil(self.fusion_size / block_size)
        fx_inputs_tensors = tuple([self.input_tensors[tensor_name] for tensor_name in self.fx.input_tensors])
        prev_x = self.fx.initialize_result(fx_inputs_tensors)
        for fusion_step in range(num_fusion_steps):
            self.fx.step(fusion_step, block_size, fx_inputs_tensors, prev_x)
        return prev_x.data


if __name__ == "__main__":
    M = 1024
    N = 512
    K = 2048
    data_type = np.float32
    lhs = Tensor(name="lhs", axes=("M", "K"), fusion_axis="K", data=np.random.normal(size=(M, K)).astype(data_type))
    rhs = Tensor(name="rhs", axes=("K", "N"), fusion_axis="K", data=np.random.normal(size=(K, N)).astype(data_type))

    chain = FusionChain(name="RMSNorm+Matmul", input_tensors=(lhs, rhs), fx=SumSquaresFx("lhs"))
    chain_result = chain.run(block_size=128)

    golden = np.sum(np.square(lhs.data), axis=-1)
    check_correctness(desired=golden, actual=chain_result, atol=1e-5, rtol=1e-5)
