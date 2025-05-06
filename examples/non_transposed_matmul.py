# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.parameter_importance import analyze_and_visualize
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
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options))
    lhs = np.zeros((M, K), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K in params:
        config = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}
        jobs.add_job(
            kernel_name="non_transposed_matmul",
            kernel_args=(lhs, rhs),
            filter=GEMMCompatibility(transposed_lhs=False),
            **config,
        )
    return jobs


def profile(workload_name: str, M: int, N: int, K: int):
    lhs = np.zeros((M, K), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    jobs.add_job(kernel_name="matmul_op", kernel_args=(lhs, rhs), filter=GEMMCompatibility(transposed_lhs=False))
    cache_dir = get_cache_dir(workload_name, "baseline", M=M, N=N, K=K)
    baseline_tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    baseline_tuner()

    jobs = get_autotune_jobs(M, N, K)
    jobs.sample(100)
    cache_dir = get_cache_dir(workload_name, "tuned", M=M, N=N, K=K)
    tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    tuner()


if __name__ == "__main__":
    workload_name = "non_transposed_GEMM"
    mn_shapes = [2048]
    k_shapes = [1024, 2048]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    for M, N, K in MNK:
        profile(workload_name, M, N, K)
        plot_metrics_vs_k_comparison(workload_name)
    plot_metrics_vs_k_comparison(workload_name)
    analyze_and_visualize(workload_name)
