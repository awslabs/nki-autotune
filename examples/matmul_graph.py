import logging

import neuronxcc.nki.language as nl
import numpy as np

from compute_graph.compute_ops import Activation, Matmul, TensorScalar
from compute_graph.graph import ComputeGraph

logging.basicConfig(
    level=logging.DEBUG,
    filename="/fsx/weittang/kernelgen_cache/debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s\n%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def test_graph_gen() -> None:
    """Test data reuse graph transformation with a single merge."""
    M = 256
    K = 1024
    N = 512

    epsilon = 1e-6

    rmsnorm_matmul_graph = ComputeGraph(
        operators=[
            Activation(dest="lhs_square", op=np.square, data="lhs", reduce_op=np.add, reduce_res="lhs_sum_square"),
            TensorScalar(
                dest="rmsnorm_factor",
                data="lhs_sum_square",
                op0=np.multiply,
                operand0=1 / K,
                op1=np.add,
                operand1=epsilon,
            ),
            Activation(dest="rmsnorm_factor", op=nl.rsqrt, data="rmsnorm_factor"),
            TensorScalar(dest="lhs_norm", data="lhs", op0=np.multiply, operand0="rmsnorm_factor"),
            Matmul(dest="output", lhs="lhs_norm", rhs="rhs", lhs_transposed=False),
        ],
        input_shapes={"lhs": (M, K), "rhs": (K, N)},
        output="output",
    )


if __name__ == "__main__":
    test_graph_gen()
