# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations, product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.visualize import plot_metric
from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.tune import generate_configs
from autotune.generation.meta_gemm import MetaGEMM
from autotune.modules.matmul import GEMMCompatibility, GEMMCorrectness


def get_template_configs():
    loop_orders = list(permutations("MNK"))
    loop_orders = ["".join(loop_order) for loop_order in loop_orders]
    lhs_positions = [0, 1, 2]
    rhs_positions = [0, 1, 2]
    template_params = {"loop_order": loop_orders, "lhs_position": lhs_positions, "rhs_position": rhs_positions}
    template_configs = generate_configs(**template_params)
    return template_configs


def get_configs():
    kernel_params = {"NUM_BLOCK_M": [1, 2, 4, 8, 16], "NUM_BLOCK_N": [1, 2, 4, 8, 16], "NUM_BLOCK_K": [1, 2, 4, 8, 16]}
    kernel_configs = generate_configs(**kernel_params)
    return kernel_configs


def add_jobs(all_jobs: ProfileJobs, M: int, N: int, K: int):
    data_type = "bf16"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=True)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    lhsT = np.random.normal(0, 0.001, size=(K, M)).astype(data_type)
    rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
    kernel_configs = get_configs()
    jobs = ProfileJobs()
    for template_id in enumerate(template_configs):
        code_file_path = f"/mnt/efs/generated_kernels/generated_gemm_kernel_{template_id}.py"
        for kernel_config in kernel_configs:
            jobs.add_job(
                kernel=(code_file_path, "lhs_rhs_gemm"),
                input_tensors=(lhsT, rhs),
                kernel_kwargs=kernel_config,
                compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
                preprocessing=GEMMCompatibility(transposed_lhs=True),
                postprocessing=postprocessing,
            )
    jobs.sample(100)
    all_jobs.extend(jobs)
    return all_jobs


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    template_configs = get_template_configs()
    for template_id, template_config in enumerate(template_configs):
        kernel = MetaGEMM(
            code_file_path=f"/mnt/efs/generated_kernels/generated_gemm_kernel_{template_id}.py",
            transposed_lhs=True,
            **template_config,
        )
    mn_shapes = [1024, 2048]
    k_shapes = [1024, 2048, 4096, 8192, 16384]
    MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    all_jobs = ProfileJobs()
    for M, N, K in MNK:
        add_jobs(all_jobs, M, N, K)
    tuner = Benchmark(jobs=all_jobs, cache_root_dir=cache_root_dir)
    tuner()
    kernels = ["lhs_rhs_gemm"]
    plot_metric(cache_root_dir, "min_ms", kernels)
    plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
