# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.job import ProfileJobs
from autotune.modules.lhsT_rhs import lhsT_rhs_gemm_general
from autotune.modules.matmul import GEMMCorrectness
from autotune.modules.pre_compile import pre_compile_kernel, save_kernel_to_file


def get_configs():
    configs = [
        {
            "NUM_BLOCK_M": 2,
            "NUM_BLOCK_N": 1,
            "NUM_BLOCK_K": 4,
            "loop_order": "MKN",
            "tensor_positions": {-1: ["result_block"], 0: [], 1: ["rhs_block"], 2: ["lhsT_block"]},
        }
    ]
    return configs


def get_jobs(M: int, N: int, K: int, configs: List[Dict[str, Any]]):
    data_type = "float32"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=True)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    lhsT = np.random.normal(size=(K, M)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    jobs = ProfileJobs()
    for config_id, config in enumerate(configs):
        kernel_code = pre_compile_kernel(lhsT_rhs_gemm_general, lhsT, rhs, **config)
        save_kernel_to_file(kernel_code, f"generated_lhsT_rhs_gemm_{config_id}.py", output_dir="generated_kernels")
        # jobs.add_job(
        #     kernel=("autotune/modules/lhsT_rhs.py", "lhsT_rhs_gemm_general"),
        #     input_tensors=(lhsT, rhs),
        #     kernel_kwargs=config,
        #     compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
        #     preprocessing=preprocessing,
        #     postprocessing=postprocessing,
        # )
    return jobs


if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    configs = get_configs()
    jobs = get_jobs(M=512, N=512, K=10240, configs=configs)
    # tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    # tuner()
    # kernels = ["lhsT_rhs_gemm_general"]
    # plot_metric(cache_root_dir, "min_ms", kernels)
    # plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
