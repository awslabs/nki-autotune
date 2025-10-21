import numpy as np

from compute_graph.graph import InitialGraphGenerator
from compute_graph.operators import Operator, Workload
from compute_graph.visualize import save_graph_as_dot


def matmul_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.matmul(lhs, rhs)
    return result


def test_graph_generation() -> None:
    """Test initial graph generation for matmul workload."""
    TILE_M = 128
    TILE_N = 512

    M = 2048
    K = 128
    N = 1024

    workload = Workload(
        input_tensors={"lhs": (M, K), "rhs": (K, N)},
        operators=[Operator("matmul", inputs=["lhs", "rhs"], params={})],
        parallel_axes=[("lhs", 0, TILE_M), ("rhs", 1, TILE_N)],
        output_tensor="output",
    )

    generator = InitialGraphGenerator(workload)
    graph = generator.generate()

    save_graph_as_dot(graph, output_file="matmul_graph.png", title="Matmul Compute Graph")


if __name__ == "__main__":
    test_graph_generation()
