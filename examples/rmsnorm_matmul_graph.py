import numpy as np

from compute_graph.graph import InitialGraphGenerator
from compute_graph.operators import Operator, Workload
from compute_graph.visualize import save_graph_as_dot


def rmsnorm_matmul_golden(lhs, rhs, epsilon: float) -> np.ndarray:
    squares = lhs**2
    sum_of_squares = np.sum(squares, axis=-1, keepdims=False)
    square_mean = sum_of_squares / lhs.shape[-1]

    rms = np.sqrt(square_mean + epsilon)
    lhs_normalized = lhs / rms[:, None]
    result = np.matmul(lhs_normalized, rhs)
    return result


def test_graph_generation() -> None:
    """Test initial graph generation for rmsnorm-matmul workload."""
    TILE_M = 128
    TILE_N = 512

    seq_len = 2048
    hidden_dim = 128
    output_dim = 1024

    workload = Workload(
        input_tensors={"lhs": (seq_len, hidden_dim), "rhs": (hidden_dim, output_dim)},
        operators=[
            Operator("square", inputs=["lhs"], params={}),
            Operator("sum", inputs=["squares"], params={"axis": 1}),
            Operator("rms_norm", inputs=["lhs", "sum_squares"], params={"epsilon": 1e-6, "N": hidden_dim}),
            Operator("matmul", inputs=["normalized", "rhs"], params={}),
        ],
        parallel_axes=[("lhs", 0, TILE_M), ("rhs", 1, TILE_N)],
        output_tensor="output",
    )

    generator = InitialGraphGenerator(workload)
    graph = generator.generate()

    save_graph_as_dot(
        graph,
        output_file="rmsnorm_matmul_graph.png",
        title="RMSNorm-Matmul Compute Graph",
        metadata={
            "parallel_axes": [("lhs", 0, TILE_M), ("rhs", 1, TILE_N)],
            "input_tensors": {"lhs": (seq_len, hidden_dim), "rhs": (hidden_dim, output_dim)},
        },
    )


if __name__ == "__main__":
    test_graph_generation()
