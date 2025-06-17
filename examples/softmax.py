# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from itertools import product

import numpy as np

from autotune.cache.visualize import plot_metric
from autotune.core.utils import GEMMCompatibility
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_library.softmax import softmax_gemm_correctness_postprocessing


def create_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    batch = 1
    data_type = np.float32
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)

    if data_type == np.dtype("float32"):
        postprocessing = softmax_gemm_correctness_postprocessing
    else:
        postprocessing = None
    jobs.add_job(
        kernel=("kernel_library/softmax.py", "softmax_gemm_np"),
        input_tensors=(lhs, rhs),
        kernel_kwargs={},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        preprocessing=GEMMCompatibility(transposed_lhs=False),
        postprocessing=postprocessing,
    )
    jobs.add_job(
        kernel=("kernel_library/softmax.py", "online_softmax_gemm_np"),
        input_tensors=(lhs, rhs),
        kernel_kwargs={},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        preprocessing=GEMMCompatibility(transposed_lhs=False),
        postprocessing=postprocessing,
    )

    # sizes = [1, 2, 4, 8, 16]
    # NUM_BLOCK_M_options = sizes
    # NUM_BLOCK_N_options = sizes
    # NUM_BLOCK_K_options = sizes
    # params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options))
    # autotune_jobs = ProfileJobs()
    # for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K in params:
    #     autotune_jobs.add_job(
    #         kernel=("kernel_library/softmax.py", "online_softmax_linear_MKN"),
    #         input_tensors=(lhs, rhs),
    #         kernel_kwargs={"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K},
    #         compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
    #         preprocessing=GEMMCompatibility(transposed_lhs=False),
    #         postprocessing=postprocessing,
    #     )
    # autotune_jobs.sample(100)
    # jobs.extend(autotune_jobs)


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    mn_shapes = [1024]
    k_shapes = [1024, 2048]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        create_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    tuner()
    kernels = ["softmax_gemm_np", "online_softmax_gemm_np", "online_softmax_linear_MKN"]
    plot_metric(cache_root_dir, "min_ms", kernels)
    plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
