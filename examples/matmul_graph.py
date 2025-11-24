import neuronxcc.nki.language as nl
import numpy as np

from compute_graph.buffer_ops import Activation, Matmul, TensorScalar
from compute_graph.graph import ComputeGraph


def matmul_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.matmul(lhs, rhs)
    return result


def test_graph_gen() -> None:
    """Test data reuse graph transformation with a single merge."""
    TILE_M = 128
    TILE_N = 512
    TILE_K = 512

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
            Matmul(dest="output", lhs="lhs_norm", rhs="rhs"),
        ]
    )
    rmsnorm_matmul_graph.specialize(inputs={"lhs": (M, K), "rhs": (K, N)}, output="output")


if __name__ == "__main__":
    test_graph_gen()
