# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations, product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import BASELINE_CACHE_DIR, TUNED_CACHE_DIR
from autotune.cache.parameter_importance import analyze_and_visualize
from autotune.cache.visualize import plot_pe_vs_k_comparison
from autotune.kernels.matmul import MatMulCompatibility, matmul_main
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs


def matmul_xt_op(x_t, y):
    """Matrix multiplication with transposed first operand"""
    x = np.transpose(x_t, (1, 0))
    return np.matmul(x, y)


def get_autotune_jobs(M: int, N: int, K: int) -> ProfileJobs:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for NUM_BLOCK_M,
                NUM_BLOCK_N, and NUM_BLOCK_K.
    """
    size_options = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = size_options
    NUM_BLOCK_N_options = size_options
    NUM_BLOCK_K_options = size_options
    BUFFER_M_options = NUM_BLOCK_M_options
    BUFFER_N_options = NUM_BLOCK_N_options
    BUFFER_K_options = NUM_BLOCK_K_options
    loop_orders = ["".join(p) for p in permutations("MNK")]
    params = list(
        product(
            NUM_BLOCK_M_options,
            NUM_BLOCK_N_options,
            NUM_BLOCK_K_options,
            BUFFER_M_options,
            BUFFER_N_options,
            BUFFER_K_options,
            loop_orders,
        )
    )
    lhsT = np.zeros((K, M), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order in params:
        config = {
            "NUM_BLOCK_M": NUM_BLOCK_M,
            "NUM_BLOCK_N": NUM_BLOCK_N,
            "NUM_BLOCK_K": NUM_BLOCK_K,
            "BUFFER_M": BUFFER_M,
            "BUFFER_N": BUFFER_N,
            "BUFFER_K": BUFFER_K,
            "loop_order": loop_order,
        }
        jobs.add_job(kernel=matmul_main, kernel_args=(lhsT, rhs), pruning_func=MatMulCompatibility, **config)
    return jobs


def profile():
    mn_shapes = [2048, 4096, 8192]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    MNK = list(product([2048], [8192], [2048]))
    for M, N, K in MNK:
        lhsT = np.zeros((K, M), dtype=bfloat16)
        rhs = np.zeros((K, N), dtype=bfloat16)
        jobs = ProfileJobs()
        jobs.add_job(kernel=matmul_xt_op, kernel_args=(lhsT, rhs), pruning_func=MatMulCompatibility)
        baseline_tuner = Benchmark(jobs=jobs, cache_dir=f"{BASELINE_CACHE_DIR}/GEMM/M{M}-N{N}-K{K}")
        baseline_tuner()

        jobs = get_autotune_jobs(M, N, K)
        jobs = jobs.sample(4)
        tuner = Benchmark(jobs=jobs, cache_dir=f"{TUNED_CACHE_DIR}/GEMM/M{M}-N{N}-K{K}")
        tuner()


if __name__ == "__main__":
    profile()
    plot_pe_vs_k_comparison(tuned_dir=f"{TUNED_CACHE_DIR}/GEMM", baseline_dir=f"{BASELINE_CACHE_DIR}/GEMM")
    analyze_and_visualize(f"{TUNED_CACHE_DIR}/GEMM")
