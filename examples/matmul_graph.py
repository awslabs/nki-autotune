import neuronxcc.nki.language as nl
import numpy as np

from compute_graph.codegen import NKICodegen
from compute_graph.graph import ComputeGraph
from compute_graph.operators import Activation, Matmul, TensorScalar
from compute_graph.tensors import HBMTensor
from compute_graph.visualize import save_graph


def matmul_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.matmul(lhs, rhs)
    return result


def test_graph_gen() -> None:
    """Test data reuse graph transformation with a single merge."""
    TILE_M = 128
    TILE_N = 512
    TILE_K = 999

    M = 256
    K = 999
    N = 512

    lhs = HBMTensor("lhs", (M, K), (TILE_M, TILE_K), ("parallel", "reduction"))
    rhs = HBMTensor("rhs", (K, N), (TILE_K, TILE_N), ("reduction", "parallel"))
    output = HBMTensor("output", (M, N), (TILE_M, TILE_N), ("parallel", "parallel"))
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
            Matmul(dest="output", stationary="lhs_norm", moving="rhs"),
        ]
    )
    rmsnorm_matmul_graph.specialize(inputs=[lhs, rhs], outputs=[output])
    save_graph(rmsnorm_matmul_graph, output_file="rmsnorm_matmul.png", title="RMSNorm + Matmul ComputeGraph")

    codegen = NKICodegen(rmsnorm_matmul_graph)
    kernel_code = codegen.generate_kernel("rmsnorm_matmul_kernel")
    output_file = "generated_kernels/rmsnorm_matmul_kernel.py"
    with open(output_file, "w") as f:
        f.write(kernel_code)


if __name__ == "__main__":
    test_graph_gen()
