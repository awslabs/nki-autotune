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
from kernel_library.rmsnorm_linear_golden import rmsnorm_correctness_postprocessing


def create_jobs(jobs: ProfileJobs, M: int, N: int, K: int, data_type: str):
    batch = 1
    if data_type == "float32":
        data_type = np.float32
        postprocessing = rmsnorm_correctness_postprocessing
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    jobs.add_job(
        kernel=("kernel_library/rmsnorm_linear_golden.py", "rmsnorm_matmul_golden"),
        input_tensor_shapes=[lhs.shape, rhs.shape],
        kernel_kwargs={"eps": 1e-6},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        postprocessing=postprocessing,
        data_type=data_type,
    )

    sizes = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = sizes
    NUM_BLOCK_N_options = sizes
    NUM_BLOCK_K_options = sizes
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options))
    autotune_jobs = ProfileJobs()
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K in params:
        autotune_jobs.add_job(
            kernel=("kernel_library/rmsnorm_linear.py", "online_rmsnorm_linear_MKN"),
            input_tensor_shapes=[lhs.shape, rhs.shape],
            kernel_kwargs={
                "eps": 1e-6,
                "NUM_BLOCK_M": NUM_BLOCK_M,
                "NUM_BLOCK_N": NUM_BLOCK_N,
                "NUM_BLOCK_K": NUM_BLOCK_K,
            },
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            postprocessing=postprocessing,
            data_type=data_type,
        )
    # Limit to 100 jobs if there are too many
    if autotune_jobs.num_jobs > 100:
        import random

        sampled_indices = random.sample(range(autotune_jobs.num_jobs), 100)
        sampled_jobs = ProfileJobs()
        for idx in sampled_indices:
            job = autotune_jobs[idx]
            sampled_jobs.jobs.append(job)
        jobs.extend(sampled_jobs)
    else:
        jobs.extend(autotune_jobs)


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    mn_shapes = [1024]
    k_shapes = [1024]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        create_jobs(jobs, M, N, K, "bf16")
    tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    tuner()
    kernels = ["rmsnorm_matmul_golden", "online_rmsnorm_linear_MKN"]
    plot_metric(cache_root_dir, "min_ms", kernels)
    plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
