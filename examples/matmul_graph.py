import numpy as np

from compute_graph.graph import ComputeGraph
from compute_graph.operators import Operator
from compute_graph.visualize import save_graph_as_dot


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
        operators=[Operator("matmul", inputs=["lhs", "rhs"], outputs=["O_1"])],
        output_tensors=["O_1"],
    )
    print(graph)
    print(graph.nodes)
    print(graph.edges)

    save_graph_as_dot(graph, output_file="matmul_initial.png", title="Initial Matmul ComputeGraph")

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
