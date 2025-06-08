# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.utils import GEMMCompatibility
from autotune.tune.benchmark import Benchmark
from autotune.tune.job import ProfileJobs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_library.rmsnorm_linear_golden import rmsnorm_correctness_postprocessing


def create_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    batch = 1
    data_type = bfloat16
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    if data_type == np.dtype("float32"):
        postprocessing = rmsnorm_correctness_postprocessing
    else:
        postprocessing = None
    for trial in range(10):
        jobs.add_job(
            kernel=("kernel_library/rmsnorm_linear_golden.py", "rmsnorm_matmul_golden"),
            input_tensors=(lhs, rhs),
            kernel_kwargs={"eps": 1e-6},
            compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
            preprocessing=GEMMCompatibility(transposed_lhs=False),
            postprocessing=postprocessing,
        )

    sizes = [1, 2, 4, 8, 16]
    NUM_BLOCK_M_options = sizes
    NUM_BLOCK_N_options = sizes
    NUM_BLOCK_K_options = sizes
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, NUM_BLOCK_K_options))
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K in params[:100]:
        jobs.add_job(
            kernel=("kernel_library/rmsnorm_linear.py", "online_rmsnorm_linear_MKN"),
            input_tensors=(lhs, rhs),
            kernel_kwargs={
                "eps": 1e-6,
                "NUM_BLOCK_M": NUM_BLOCK_M,
                "NUM_BLOCK_N": NUM_BLOCK_N,
                "NUM_BLOCK_K": NUM_BLOCK_K,
            },
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=GEMMCompatibility(transposed_lhs=False),
            postprocessing=postprocessing,
        )
    # TODO: implement jobs appending
    # jobs.sample(100)


if __name__ == "__main__":
    mn_shapes = [2048]
    k_shapes = [2048]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        create_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs)
    tuner()
    # plot_metric("min_ms", ["rmsnorm_matmul_golden", "online_rmsnorm_linear_MKN"])
    # plot_metric("mfu_estimated_percent", ["rmsnorm_matmul_golden", "online_rmsnorm_linear_MKN"])
