import logging
import os

import numpy as np

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.metrics import check_correctness
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE
from compute_graph.codegen import NKICodegen
from compute_graph.compute_ops import Activation, Matmul, TensorScalar
from compute_graph.graph import ComputeGraph
from compute_graph.visualize import save_graph, setup_logging

cache_root = os.environ.get("NKI_CACHE_ROOT", "/fsx/weittang/kernelgen_cache")
setup_logging(f"{cache_root}/debug.log")
logger = logging.getLogger(__name__)

RMSNORM_EPSILON = 1e-6


def rmsnorm_gemm_correctness(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_outputs: OUTPUT_TENSORS_DTYPE
) -> None:
    """Postprocessing function to verify RMSNorm + GEMM correctness.

    Computes golden reference: output = RMSNorm(lhs) @ rhs
    where RMSNorm(x) = x / sqrt(sum(x^2) / K + epsilon)
    """
    lhs, rhs = input_tensors.values()
    K = lhs.shape[-1]

    lhs_square = np.square(lhs)
    lhs_sum_square = np.sum(lhs_square, axis=-1, keepdims=True)
    rmsnorm_factor = 1.0 / np.sqrt(lhs_sum_square / K + RMSNORM_EPSILON)
    lhs_norm = lhs * rmsnorm_factor
    golden = np.matmul(lhs_norm, rhs).astype(np.float32)

    nki_out = kernel_outputs[0].astype(np.float32)
    check_correctness(golden, nki_out, atol=1e-4, rtol=1e-2)


def generate_kernel(M: int, K: int, N: int) -> tuple[str, str, dict[str, tuple[int, ...]]]:
    """Generate RMSNorm + Matmul kernel code.

    Args:
        M: Number of rows in LHS matrix
        K: Number of columns in LHS / rows in RHS matrix
        N: Number of columns in RHS matrix

    Returns:
        Tuple of (kernel_path, kernel_name, input_tensor_shapes)
    """
    input_tensor_shapes = {"lhs_hbm": (M, K), "rhs_hbm": (K, N)}

    rmsnorm_matmul_graph = ComputeGraph(
        operators=[
            Activation(dest="lhs_square", op="np.square", data="lhs", reduce_op="np.add", reduce_res="lhs_sum_square"),
            TensorScalar(
                dest="rmsnorm_factor",
                data="lhs_sum_square",
                op0="np.multiply",
                operand0=1 / K,
                op1="np.add",
                operand1=RMSNORM_EPSILON,
            ),
            Activation(dest="rmsnorm_factor", op="nl.rsqrt", data="rmsnorm_factor"),
            TensorScalar(dest="lhs_norm", data="lhs", op0="np.multiply", operand0="rmsnorm_factor"),
            Matmul(dest="output", lhs="lhs_norm", rhs="rhs", lhs_transposed=False),
        ],
        input_shapes={"lhs": (M, K), "rhs": (K, N)},
        output="output",
    )
    save_graph(rmsnorm_matmul_graph, output_dir=f"{cache_root}", title="RMSNorm + Matmul")
    kernel_name = "rmsnorm_matmul_kernel"
    kernel_path = f"{cache_root}/{kernel_name}.py"
    codegen = NKICodegen(rmsnorm_matmul_graph)
    codegen.generate_kernel(kernel_name, kernel_path)

    return kernel_path, kernel_name, input_tensor_shapes


def run_benchmark(
    kernel_path: str, kernel_name: str, input_tensor_shapes: dict[str, tuple[int, ...]], mac_count: int
) -> None:
    """Benchmark the generated kernel.

    Args:
        kernel_path: Path to the generated kernel file
        kernel_name: Name of the kernel function
        input_tensor_shapes: Dict mapping input names to shapes
        mac_count: Number of multiply-accumulate operations for MFU calculation
    """
    jobs = ProfileJobs(cache_root_dir=cache_root, target_instance_family="trn2")
    jobs.add_job(
        kernel=(kernel_path, kernel_name),
        input_tensor_shapes=input_tensor_shapes,
        data_type=np.float32,
        kernel_kwargs={},
        compiler_flags="--auto-cast=none --internal-tensorizer-opt-level=nki",
        postprocessing=rmsnorm_gemm_correctness,
        mac_count=mac_count,
    )
    benchmark = Benchmark(jobs=jobs)
    benchmark()


if __name__ == "__main__":
    M, K, N = 256, 128, 128
    kernel_path, kernel_name, input_tensor_shapes = generate_kernel(M, K, N)
    run_benchmark(kernel_path, kernel_name, input_tensor_shapes, mac_count=M * N * K)
