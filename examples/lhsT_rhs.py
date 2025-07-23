# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.generation.lhsT_rhs import check_template, lhsT_rhs_gemm_general
from autotune.generation.specialize import save_code_to_file, specialize_kernel
from autotune.modules.matmul import GEMMCorrectness


def get_configs():
    configs = [
        {
            "NUM_BLOCK_M": 2,
            "NUM_BLOCK_N": 1,
            "NUM_BLOCK_K": 4,
            "loop_order": {"M": 0, "N": 1, "K": 2},
            "tensor_positions": {"rhs_block": 2, "lhsT_block": 2, "result_block": 1, "matmul": 2},
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
        check_template(config["loop_order"], config["tensor_positions"])
        kernel_code = specialize_kernel(lhsT_rhs_gemm_general, ["maybe_init", "maybe_compute", "maybe_save"], **config)
        generated_file = f"generated_kernels/generated_lhsT_rhs_{config_id}.py"
        save_code_to_file(generated_file, kernel_code, lhsT_rhs_gemm_general)
        jobs.add_job(
            kernel=(generated_file, "lhsT_rhs_gemm_general"),
            input_tensors=(lhsT, rhs),
            kernel_kwargs=config,
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=None,
            postprocessing=postprocessing,
        )
    jobs.add_job(
        kernel=("autotune/modules/lhsT_rhs.py", "lhsT_rhs_gemm"),
        input_tensors=(lhsT, rhs),
        kernel_kwargs={"NUM_BLOCK_M": 2, "NUM_BLOCK_N": 1, "NUM_BLOCK_K": 4, "template": "MNK"},
        compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
        preprocessing=None,
        postprocessing=postprocessing,
    )
    return jobs


if __name__ == "__main__":
    cache_root_dir = "/home/ubuntu/autotune-cache"
    configs = get_configs()
    jobs = get_jobs(M=512, N=512, K=10240, configs=configs)
    tuner = Benchmark(jobs=jobs, cache_root_dir=cache_root_dir)
    tuner()
    # kernels = ["lhsT_rhs_gemm_general"]
    # plot_metric(cache_root_dir, "min_ms", kernels)
    # plot_metric(cache_root_dir, "mfu_estimated_percent", kernels)
