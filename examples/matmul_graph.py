import numpy as np

from compute_graph.graph import ComputeGraph
from compute_graph.operators import ActivationReduce, TensorScalar
from compute_graph.tensors import Axis
from compute_graph.visualize import save_graph


def matmul_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.matmul(lhs, rhs)
    return result


def test_graph_gen() -> None:
    """Test data reuse graph transformation with a single merge."""
    TILE_M = 128
    TILE_N = 512
    TILE_K = 128

    M = 256
    K = 999
    N = 512

    lhs = Axis(M, TILE_M, "parallel"), Axis(K, TILE_K, "reduction")
    rhs = Axis(K, TILE_K, "reduction"), Axis(N, TILE_N, "parallel")
    sum_squares = Axis(M, TILE_M, "parallel"), Axis(1, 1, "parallel")
    output = Axis(M, TILE_M, "parallel"), Axis(N, TILE_N, "parallel")
    epsilon = 1e-6
    rmsnorm_matmul_graph = ComputeGraph(
        input_tensors={"lhs": lhs, "rhs": rhs},
        operators=[
            ActivationReduce(op="square", src="lhs", dest="sum_squares", reduce_op="sum", reduction_axis=1),
            TensorScalar(op="mult", src="sum_squares", dest="sum_squares", scalar=1 / K, op1="add", scalar1=epsilon),
        ],
        output_tensors={"sum_squares": sum_squares},
    )
    save_graph(rmsnorm_matmul_graph, output_file="rmsnorm_matmul.png", title="RMSNorm + Matmul ComputeGraph")


if __name__ == "__main__":
    test_graph_gen()
