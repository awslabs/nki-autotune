# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.visualize import plot_metric
from autotune.core.golden import GEMMCorrectness
from autotune.core.utils import GEMMCompatibility
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
    templates = ["MN", "MKN", "MNK", "legacy_MKN"]
    data_type = bfloat16
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options, templates))
    lhs = np.random.normal(size=(M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    if data_type == np.dtype("float32"):
        postprocessing = GEMMCorrectness(transposed_lhs=False)
    else:
        postprocessing = None
    jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, template in params:
        jobs.add_job(
            kernel=("autotune/core/lhs_rhs.py", "gemm_main"),
            input_tensors=(lhs, rhs),
            kernel_kwargs={
                "NUM_BLOCK_M": NUM_BLOCK_M,
                "NUM_BLOCK_N": NUM_BLOCK_N,
                "NUM_BLOCK_K": NUM_BLOCK_K,
                "template": template,
            },
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=GEMMCompatibility(transposed_lhs=False),
            postprocessing=postprocessing,
        )
    jobs.sample(100)
    cache_dir = get_cache_dir(workload_name, "tuned", M=M, N=N, K=K)


def add_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    data_type = bfloat16
    lhsT = np.random.normal(size=(K, M)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    if data_type == np.dtype("float32"):
        postprocessing = GEMMCorrectness(transposed_lhs=False)
    else:
        postprocessing = None
    jobs.add_job(
        kernel=("autotune/core/golden.py", "gemm_cpu_golden"),
        input_tensors=(lhsT, rhs),
        kernel_kwargs={"transposed_lhs": True},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        preprocessing=GEMMCompatibility(transposed_lhs=False),
        postprocessing=postprocessing,
    )
    cache_dir = get_cache_dir(workload_name, "baseline", M=M, N=N, K=K)


if __name__ == "__main__":
    mn_shapes = [2048]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        add_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs)
    tuner()
    plot_metric("min_ms", ["lhsT_rhs_GEMM"])
    plot_metric("mfu_estimated_percent", ["lhsT_rhs_GEMM"])
