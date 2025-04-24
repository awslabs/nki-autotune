# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.parameter_importance import analyze_and_visualize
from autotune.cache.visualize import plot_metrics_vs_k_comparison
from autotune.kernels.utils import GEMMCompatibility
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs


def get_autotune_jobs(M: int, N: int, K: int) -> ProfileJobs:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        List[Dict]: A list of dictionaries, each containing configuration parameters for
        NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N
    """
    sizes = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = sizes
    NUM_BLOCK_N_options = sizes
    BUFFER_M_options = NUM_BLOCK_M_options
    BUFFER_N_options = NUM_BLOCK_N_options
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, BUFFER_M_options, BUFFER_N_options))

    batch = 1
    lhs = np.zeros((batch, M, K), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)

    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N in params:
        config = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "BUFFER_M": BUFFER_M, "BUFFER_N": BUFFER_N}
        jobs.add_job(
            kernel_name="blocked_fused_rms_norm_linear",
            kernel_args=(lhs, rhs),
            filter=GEMMCompatibility(transposed_lhs=False),
            **config,
        )
    return jobs


def get_baseline_jobs(M: int, N: int, K: int) -> ProfileJobs:
    batch = 1
    lhs = np.zeros((batch, M, K), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    jobs.add_job(
        kernel_name="stack_allocated_fused_rms_norm_qkv",
        kernel_args=(lhs, rhs),
        filter=GEMMCompatibility(transposed_lhs=False),
    )
    jobs.add_job(
        kernel_name="rmsnorm_linear_op", kernel_args=(lhs, rhs), filter=GEMMCompatibility(transposed_lhs=False)
    )
    return jobs


def profile(workload_name: str, M: int, N: int, K: int):
    baseline_jobs = get_baseline_jobs(M, N, K)
    cache_dir = get_cache_dir(workload_name, "baseline", M=M, N=N, K=K)
    baseline_tuner = Benchmark(jobs=baseline_jobs, cache_dir=cache_dir)
    baseline_tuner()

    jobs = get_autotune_jobs(M, N, K)
    jobs.sample(100)
    cache_dir = get_cache_dir(workload_name, "tuned", M=M, N=N, K=K)
    tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    tuner()


if __name__ == "__main__":
    workload_name = "fused_rmsnorm_GEMM"
    mn_shapes = [2048, 4096, 8192]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    for M, N, K in MNK:
        profile(workload_name, M, N, K)
        plot_metrics_vs_k_comparison(workload_name)
        break
    plot_metrics_vs_k_comparison(workload_name)
    analyze_and_visualize(workload_name)
