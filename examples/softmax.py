# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.visualize import plot_metric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_library.softmax import softmax_gemm_correctness_postprocessing


def create_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    batch = 1
    data_type = bfloat16
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)

    if data_type == np.dtype("float32"):
        postprocessing = softmax_gemm_correctness_postprocessing
    else:
        postprocessing = None
    jobs.add_job(
        kernel=("kernel_library/softmax.py", "softmax_gemm_np"),
        input_tensor_shapes={"lhs": lhs.shape, "rhs": rhs.shape},
        kernel_kwargs={},
        compiler_flags="--auto-cast=none --model-type=transformer",
        postprocessing=postprocessing,
        data_type=data_type,
        mac_count=batch * M * K * N,
    )

    sizes = range(17)
    NUM_BLOCK_M_options = [1]
    NUM_BLOCK_N_options = [1]
    NUM_BLOCK_K_options = [10]
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options))
    autotune_jobs = ProfileJobs(cache_root_dir=jobs.cache_root_dir, target_instance_family=jobs.target_instance_family)
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K in params:
        autotune_jobs.add_job(
            kernel=("kernel_library/softmax.py", "online_softmax_linear_MKN"),
            input_tensor_shapes={"lhs": lhs.shape, "rhs": rhs.shape},
            kernel_kwargs={"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K},
            compiler_flags="--auto-cast=none --internal-tensorizer-opt-level=nki",
            postprocessing=postprocessing,
            data_type=data_type,
            mac_count=batch * M * K * N,
        )
    # Limit to 100 jobs if there are too many
    if len(autotune_jobs.jobs) > 100:
        import random

        sampled_indices = random.sample(range(len(autotune_jobs.jobs)), 100)
        sampled_jobs = ProfileJobs(
            cache_root_dir=jobs.cache_root_dir, target_instance_family=jobs.target_instance_family
        )
        for idx in sampled_indices:
            job = autotune_jobs.jobs[idx]
            sampled_jobs.jobs[len(sampled_jobs.jobs)] = job
        jobs.extend(sampled_jobs)
    else:
        jobs.extend(autotune_jobs)


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    mn_shapes = [10240]
    k_shapes = [128]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs(cache_root_dir=cache_root_dir, target_instance_family="trn1")
    for M, N, K in MNK:
        create_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs)
    tuner()
    kernels = ["softmax_gemm_np", "online_softmax_linear_MKN"]
    plot_metric(cache_root_dir, "min_ms", kernels)
    plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
