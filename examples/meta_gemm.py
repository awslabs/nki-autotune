# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations, product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.tune import generate_configs
from autotune.generation.meta_gemm import MetaGEMM
from autotune.modules.matmul import GEMMCorrectness


def get_configs():
    size_options = [2, 4, 8]
    loop_orders = list(permutations("MNK"))
    loop_orders = ["".join(loop_order) for loop_order in loop_orders]
    loop_orders = ["MNK"]
    lhs_positions = [1]
    rhs_positions = [0, 1, 2]
    params = {
        "NUM_BLOCK_M": size_options,
        "NUM_BLOCK_N": size_options,
        "NUM_BLOCK_K": size_options,
        "loop_order": loop_orders,
        "lhs_position": lhs_positions,
        "rhs_position": rhs_positions,
    }
    configs = generate_configs(**params)
    return configs


def add_jobs(jobs: ProfileJobs, M: int, N: int, K: int):
    data_type = "float32"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=False)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    lhs = np.random.normal(size=(M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    configs = get_configs()
    for config in configs:
        kernel = MetaGEMM(**config)
        # jobs.add_job(
        #     kernel=("autotune/modules/lhs_rhs.py", "lhs_rhs_gemm"),
        #     input_tensors=(lhs, rhs),
        #     kernel_kwargs=config,
        #     compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
        #     preprocessing=GEMMCompatibility(transposed_lhs=False),
        #     postprocessing=postprocessing,
        # )
        break
    return jobs


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    mn_shapes = [1024]
    k_shapes = [2048]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    jobs = ProfileJobs()
    for M, N, K in MNK:
        add_jobs(jobs, M, N, K)
    tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    # tuner()
    # kernels = ["lhs_rhs_gemm"]
    # plot_metric(cache_root_dir, "min_ms", kernels)
    # plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
