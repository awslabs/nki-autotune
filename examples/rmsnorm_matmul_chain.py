import argparse
import os

import neuronxcc.nki.language as nl
import numpy as np

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.metrics import check_correctness
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE
from chain.axes import sample_axes_configs


def rmsnorm_matmul_golden(lhs, rhs, epsilon: float) -> np.ndarray:
    squares = lhs**2
    sum_of_squares = np.sum(squares, axis=-1, keepdims=False)
    square_mean = sum_of_squares / lhs.shape[-1]

    rms = np.sqrt(square_mean + epsilon)
    lhs_normalized = lhs / rms[:, None]
    result = np.matmul(lhs_normalized, rhs)
    return result


class ChainCorrectness:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-5, 1e-2
        lhs, rhs = input_tensors
        golden = nl.static_cast(rmsnorm_matmul_golden(lhs, rhs, self.epsilon), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensors[0], data_type)
        check_correctness(golden, nki_out_tensor, atol, rtol)


def run_rmsnorm_matmul_fusion_benchmark(cache_dir: str) -> None:
    """
    Run rmsnorm-matmul fusion benchmarks using autotune infrastructure.

    Args:
        cache_dir: Root directory for the benchmark cache
    """
    current_file = os.path.abspath(__file__)
    nki_fusion_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(nki_fusion_dir)

    seq_len = 256
    hidden_dim = 1024
    output_dim = 512
    data_type = np.float32

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128

    input_tensor_shapes = {"lhs": (seq_len, hidden_dim), "rhs": (hidden_dim, output_dim)}
    parallel_axes = [("lhs", 0, TILE_M), ("rhs", 1, TILE_N)]
    axes_config = sample_axes_configs(input_tensor_shapes, parallel_axes)

    kernel = (f"{project_root}/chain/chain.py", "chain_wrapper")
    postprocessing = ChainCorrectness(epsilon=1e-6)

    all_jobs = ProfileJobs(cache_root_dir=cache_dir, target_instance_family="trn2")
    all_jobs.add_job(
        kernel=kernel,
        input_tensor_shapes=input_tensor_shapes,
        data_type=data_type,
        kernel_kwargs={"parallel_axes": axes_config},
        compiler_flags="--auto-cast=none --internal-tensorizer-opt-level=nki",
        postprocessing=postprocessing,
        mac_count=0,
    )
    tuner = Benchmark(jobs=all_jobs)
    tuner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rmsnorm-matmul fusion benchmarks")
    parser.add_argument("--cache-dir", type=str, required=True, help="Root directory for the benchmark cache")
    args = parser.parse_args()

    run_rmsnorm_matmul_fusion_benchmark(args.cache_dir)
