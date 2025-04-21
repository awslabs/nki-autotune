# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.baseline.np_baselines import rmsnorm_linear_op
from autotune.cache.directories import BASELINE_CACHE_DIR, TUNED_CACHE_DIR
from autotune.cache.parameter_importance import analyze_and_visualize
from autotune.cache.visualize import plot_pe_vs_k_comparison
from autotune.kernels.rmsnorm_linear import blocked_fused_rms_norm_linear, stack_allocated_fused_rms_norm_qkv
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
            kernel=blocked_fused_rms_norm_linear,
            kernel_args=(lhs, rhs),
            pruning_func=GEMMCompatibility(transposed_lhs=False),
            **config,
        )
    return jobs


def get_baseline_jobs(M: int, N: int, K: int) -> ProfileJobs:
    batch = 1
    lhs = np.zeros((batch, M, K), dtype=bfloat16)
    rhs = np.zeros((K, N), dtype=bfloat16)
    jobs = ProfileJobs()
    jobs.add_job(kernel=stack_allocated_fused_rms_norm_qkv, kernel_args=(lhs, rhs))
    jobs.add_job(kernel=rmsnorm_linear_op, kernel_args=(lhs, rhs))
    return jobs


def profile(workload_name: str):
    MNK = list(product([2048], [512], [8192]))
    for M, N, K in MNK:
        baseline_jobs = get_baseline_jobs(M, N, K)
        baseline_tuner = Benchmark(jobs=baseline_jobs, cache_dir=f"{BASELINE_CACHE_DIR}/{workload_name}/M{M}-N{N}-K{K}")
        baseline_tuner()

        jobs = get_autotune_jobs(M, N, K)
        sampled_jobs = jobs.sample(100)
        tuner = Benchmark(jobs=sampled_jobs, cache_dir=f"{TUNED_CACHE_DIR}/{workload_name}/M{M}-N{N}-K{K}")
        tuner()


if __name__ == "__main__":
    workload_name = "fused_rmsnorm_GEMM"
    profile(workload_name)
    plot_pe_vs_k_comparison(
        tuned_dir=f"{TUNED_CACHE_DIR}/{workload_name}", baseline_dir=f"{BASELINE_CACHE_DIR}/{workload_name}"
    )
    analyze_and_visualize(f"{TUNED_CACHE_DIR}/{workload_name}")
