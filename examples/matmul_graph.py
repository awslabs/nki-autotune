import numpy as np

from compute_graph.graph import ComputeGraph
from compute_graph.operators import Operator
from compute_graph.visualize import save_graph


def matmul_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.matmul(lhs, rhs)
    return result


def test_data_reuse_transformation() -> None:
    """Test data reuse graph transformation with a single merge."""
    TILE_M = 128
    TILE_N = 512

    M = 256
    K = 128
    N = 1024

    graph = ComputeGraph(
        input_tensors={"lhs": (M, K), "rhs": (K, N)},
        parallel_axes=[("lhs", 0, TILE_M), ("rhs", 1, TILE_N)],
        operators=[Operator("rmsnorm", src=["lhs"], dest="O1"), Operator("matmul", src=["O1", "rhs"], dest="O2")],
        output_tensors=["O2"],
    )
    save_graph(graph, output_file="matmul_initial.png", title="Initial Matmul ComputeGraph")

    seq_len = 256
    hidden = 512
    attention_graph = ComputeGraph(
        input_tensors={"Q": (seq_len, hidden), "K": (seq_len, hidden), "V": (seq_len, hidden)},
        parallel_axes=[("Q", 0, TILE_M), ("V", 1, TILE_N)],
        operators=[
            Operator("transpose", src=["K"], dest="K_T"),
            Operator("matmul", src=["Q", "K_T"], dest="O1"),
            Operator("softmax", src=["O1"], dest="O2"),
            Operator("matmul", src=["O2", "V"], dest="O3"),
        ],
        output_tensors=["O3"],
    )
    save_graph(attention_graph, output_file="attention.png", title="Attention ComputeGraph")

    # merge_opportunities = analyze_data_reuse_opportunities(graph)
    # print(f"\n=== Data Reuse Analysis ===")
    # print(f"Found {len(merge_opportunities)} merge opportunities")
    # print(merge_opportunities)

    # selected_merge = max(merge_opportunities, key=len)
    # print(f"\n=== Selected Merge for Demo ===")
    # print(f"Applying merge for {len(selected_merge)} load nodes: {selected_merge}")

    # graph = apply_data_reuse_merge(graph, selected_merge)

    # print("\n=== After Data Reuse Transformation ===")
    # print(f"Total nodes: {graph.number_of_nodes()}")
    # print(f"Total edges: {graph.number_of_edges()}")
    # print(f"Nodes eliminated: {len(selected_merge) - 1}")

    # save_graph_as_dot(graph, output_file="matmul_after_data_reuse.png", title="After Data Reuse (Single Merge)")


if __name__ == "__main__":
    test_data_reuse_transformation()
