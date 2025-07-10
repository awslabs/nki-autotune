# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.visualize import plot_metric
from autotune.core.benchmark import Benchmark
from autotune.core.golden import GEMMCorrectness
from autotune.core.job import ProfileJobs
from autotune.modules.matmul import GEMMCompatibility


def add_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    data_type = bfloat16
    lhsT = np.random.normal(size=(K, M)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    if data_type == np.dtype("float32"):
        postprocessing = GEMMCorrectness(transposed_lhs=True)
    else:
        postprocessing = None
    for trial in range(10):
        jobs.add_job(
            kernel=("autotune/core/golden.py", "lhsT_rhs_gemm_np"),
            input_tensors=(lhsT, rhs),
            kernel_kwargs={"transposed_lhs": True},
            compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
            preprocessing=GEMMCompatibility(transposed_lhs=True),
            postprocessing=postprocessing,
        )

    size_options = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = size_options
    NUM_BLOCK_N_options = size_options
    NUM_BLOCK_K_options = size_options
    templates = ["MN", "MKN", "MNK", "legacy_MKN"]
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options, templates))
    autotune_jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, template in params:
        autotune_jobs.add_job(
            kernel=("autotune/core/lhsT_rhs.py", "lhsT_rhs_gemm"),
            input_tensors=(lhsT, rhs),
            kernel_kwargs={
                "NUM_BLOCK_M": NUM_BLOCK_M,
                "NUM_BLOCK_N": NUM_BLOCK_N,
                "NUM_BLOCK_K": NUM_BLOCK_K,
                "template": template,
            },
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=GEMMCompatibility(transposed_lhs=True),
            postprocessing=postprocessing,
        )
    autotune_jobs.sample(100)
    jobs.extend(autotune_jobs)


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    mn_shapes = [2048]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        add_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    tuner()
    plot_metric(cache_root_dir, "min_ms", ["lhsT_rhs_gemm_np", "lhsT_rhs_GEMM"])
    plot_metric(cache_root_dir, "mfu_estimated_percent", ["lhsT_rhs_gemm_np", "lhsT_rhs_GEMM"])
