# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations
from typing import List

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.visualize import plot_metric
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


def add_jobs(all_jobs: ProfileJobs, kernels: List[MetaGEMM], M: int, N: int, K: int):
    data_type = "bf16"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=False)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    lhs = np.random.normal(0, 0.001, size=(M, K)).astype(data_type)
    rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
    kernel_configs = get_configs()
    jobs = ProfileJobs()
    for kernel in kernels:
        for kernel_config in kernel_configs:
            jobs.add_job(
                kernel=(kernel.code_file_path, "lhs_rhs_gemm"),
                input_tensors=(lhs, rhs),
                kernel_kwargs=kernel_config,
                compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
                preprocessing=GEMMCompatibility(transposed_lhs=False),
                postprocessing=postprocessing,
            )
    jobs.sample(100)
    all_jobs.extend(jobs)
    all_jobs.add_job(
        kernel=("autotune/modules/matmul.py", "lhs_rhs_gemm_np"),
        input_tensors=(lhs, rhs),
        kernel_kwargs={},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer",
        preprocessing=None,
        postprocessing=postprocessing,
    )
    return all_jobs


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-dev-cache"
    # template_configs = get_template_configs()
    # kernels = []
    # for template_id, template_config in enumerate(template_configs):
    #     kernel = MetaGEMM(
    #         code_file_path=f"/mnt/efs/generated_kernels/lhs_rhs_gemm/generated_gemm_kernel_{template_id}.py",
    #         transposed_lhs=False,
    #         **template_config,
    #     )
    #     kernels.append(kernel)
    # mn_shapes = [1024, 2048]
    # k_shapes = [1024, 2048, 4096, 8192, 16384]
    # MNK = list(product(mn_shapes, mn_shapes, k_shapes))
    # all_jobs = ProfileJobs()
    # for M, N, K in MNK:
    #     add_jobs(all_jobs, kernels, M, N, K)
    # tuner = Benchmark(jobs=all_jobs, cache_root_dir=cache_root_dir)
    # tuner()
    kernel_names = ["lhs_rhs_gemm", "lhs_rhs_gemm_np"]
    plot_metric(cache_root_dir, "min_ms", kernel_names)
    plot_metric(cache_root_dir, "mfu_estimated_percent", kernel_names)
