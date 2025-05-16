# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.visualize import plot_metrics_vs_k_comparison
from autotune.core.utils import GEMMCompatibility
from autotune.golden.gemm import GEMMCorrectness
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs


def run_autotune_jobs(workload_name: str, M: int, N: int, K: int):
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.
    """
    size_options = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = size_options
    NUM_BLOCK_N_options = size_options
    NUM_BLOCK_K_options = size_options
    templates = ["legacy_MKN"]
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options, templates))
    lhs = np.random.random_sample((M, K)).astype(bfloat16)
    rhs = np.random.random_sample((K, N)).astype(bfloat16)
    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, template in params:
        jobs.add_job(
            kernel_name="non_transposed_matmul",
            input_tensors=(lhs, rhs),
            kernel_kwargs={
                "NUM_BLOCK_M": NUM_BLOCK_M,
                "NUM_BLOCK_N": NUM_BLOCK_N,
                "NUM_BLOCK_K": NUM_BLOCK_K,
                "template": template,
            },
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=GEMMCompatibility(transposed_lhs=False),
            postprocessing=GEMMCorrectness(transposed_lhs=False),
        )
    jobs.sample(100)
    cache_dir = get_cache_dir(workload_name, "tuned", M=M, N=N, K=K)
    tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    tuner()


def profile_baseline(workload_name: str, M: int, N: int, K: int):
    data_type = bfloat16
    lhs = np.random.random_sample((M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    jobs = ProfileJobs()
    jobs.add_job(
        kernel_name="matmul_op",
        input_tensors=(lhs, rhs),
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        preprocessing=GEMMCompatibility(transposed_lhs=False),
        postprocessing=GEMMCorrectness(transposed_lhs=False),
    )
    cache_dir = get_cache_dir(workload_name, "baseline", M=M, N=N, K=K)
    baseline_tuner = Benchmark(jobs=jobs, cache_dir=cache_dir)
    baseline_tuner()


if __name__ == "__main__":
    workload_name = "non_transposed_GEMM"
    mn_shapes = [1024, 2048, 4096, 8192]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    mn_shapes = [1024]
    k_shapes = [1024]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    for M, N, K in MNK:
        profile_baseline(workload_name, M, N, K)
        run_autotune_jobs(workload_name, M, N, K)
        plot_metrics_vs_k_comparison(workload_name)
    plot_metrics_vs_k_comparison(workload_name)
