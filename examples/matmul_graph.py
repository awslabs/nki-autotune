import logging
import os

import neuronxcc.nki.language as nl
import numpy as np

from compute_graph.compute_ops import Activation, Matmul, TensorScalar
from compute_graph.graph import ComputeGraph
from compute_graph.visualize import save_graph, setup_logging

cache_root = os.environ.get("NKI_CACHE_ROOT", "/fsx/weittang/kernelgen_cache")
setup_logging(f"{cache_root}/debug.log")
logger = logging.getLogger(__name__)


def test_graph_gen() -> None:
    """Test data reuse graph transformation with a single merge."""
    M = 256
    K = 1024
    N = 128

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
    save_graph(rmsnorm_matmul_graph, output_file=f"{cache_root}/graph.png", title="RMSNorm + Matmul")


if __name__ == "__main__":
    test_graph_gen()
