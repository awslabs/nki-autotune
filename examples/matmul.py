# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations, product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.visualize import plot_metrics_vs_k_comparison
from autotune.core.utils import GEMMCompatibility
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs


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
    loop_orders = ["".join(p) for p in permutations("MNK")]
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options, loop_orders))
    lhsT = np.zeros((K, M), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, loop_order in params:
        config = {
            "NUM_BLOCK_M": NUM_BLOCK_M,
            "NUM_BLOCK_N": NUM_BLOCK_N,
            "NUM_BLOCK_K": NUM_BLOCK_K,
            "loop_order": loop_order,
        }
        jobs.add_job(
            kernel_name="matmul_main",
            input_tensors=(lhsT, rhs),
            preprocessing=GEMMCompatibility(transposed_lhs=True),
            **config,
        )
    return jobs


def profile(workload_name: str, M: int, N: int, K: int):
    lhsT = np.zeros((K, M), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    jobs.add_job(
        kernel_name="matmul_xt_op", input_tensors=(lhsT, rhs), preprocessing=GEMMCompatibility(transposed_lhs=True)
    )
    cache_dir = get_cache_dir(workload_name, "baseline", M=M, N=N, K=K)
    baseline_tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    baseline_tuner()

    jobs = get_autotune_jobs(M, N, K)
    jobs.sample(100)
    cache_dir = get_cache_dir(workload_name, "tuned", M=M, N=N, K=K)
    tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    tuner()


if __name__ == "__main__":
    workload_name = "GEMM"
    mn_shapes = [2048]
    k_shapes = [1024, 2048]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    for M, N, K in MNK:
        profile(workload_name, M, N, K)
        plot_metrics_vs_k_comparison(workload_name)
    plot_metrics_vs_k_comparison(workload_name)
