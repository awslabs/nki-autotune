# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import random

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.gemm import GEMMCorrectness, generate_gemm_configs


def add_jobs(all_jobs: ProfileJobs, transposed_lhs: bool = False):
    # Dynamically find the project root directory
    current_file = os.path.abspath(__file__)  # /path/to/nki-autotune/examples/gemm.py
    examples_dir = os.path.dirname(current_file)  # /path/to/nki-autotune/examples/
    project_root = os.path.dirname(examples_dir)  # /path/to/nki-autotune/

    data_type = "float32"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=transposed_lhs)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")

    if transposed_lhs:
        baseline_kernel = (f"{project_root}/autotune/gemm/validation.py", "lhsT_rhs_gemm_np")
        meta_kernel = (f"{project_root}/autotune/gemm/kernels.py", "lhsT_rhs_meta_gemm")
    else:
        baseline_kernel = (f"{project_root}/autotune/gemm/validation.py", "lhs_rhs_gemm_np")
        meta_kernel = (f"{project_root}/autotune/gemm/kernels.py", "lhs_rhs_meta_gemm")

    shapes = [
        (1236, 2847, 1539),
        (1024, 2048, 2048),
        (3757, 1647, 2539),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (24576, 24576, 24576),
    ]
    for M, N, K in [(2048, 2048, 2048)]:
        if transposed_lhs:
            lhs_shape = (K, M)
        else:
            lhs_shape = (M, K)
        rhs_shape = (K, N)
        configs = generate_gemm_configs(M=M, N=N, K=K)
        configs = random.sample(configs, 10)
        for config in configs:
            all_jobs.add_job(
                kernel=meta_kernel,
                input_tensor_shapes=[lhs_shape, rhs_shape],
                data_type=data_type,
                kernel_kwargs={"config": config},
                compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
                postprocessing=postprocessing,
            )
        all_jobs.add_job(
            kernel=baseline_kernel,
            input_tensor_shapes=[lhs_shape, rhs_shape],
            data_type=data_type,
            kernel_kwargs={},
            compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer --tensorizer-options='--print-nki'",
            postprocessing=postprocessing,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmarks with different matrix configurations")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lhsT_rhs", "lhs_rhs", "both"],
        help="Matrix multiplication mode: lhsT_rhs (transposed LHS), lhs_rhs, or both",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/mnt/efs/autotune-dev-cache", help="Root directory for the benchmark cache"
    )
    args = parser.parse_args()
    all_jobs = ProfileJobs(cache_root_dir=args.cache_dir)
    if args.mode == "lhsT_rhs" or args.mode == "both":
        add_jobs(all_jobs, transposed_lhs=True)
    if args.mode == "lhs_rhs" or args.mode == "both":
        add_jobs(all_jobs, transposed_lhs=False)
    tuner = Benchmark(jobs=all_jobs)
    tuner()

    # if args.mode == "lhsT_rhs" or args.mode == "both":
    #     kernel_names = ["lhsT_rhs_gemm_np", "lhsT_rhs_meta_gemm", "nki_matmul_nmk_order"]
    #     plot_metric(args.cache_dir, "min_ms", kernel_names)
    #     plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
    # if args.mode == "lhs_rhs" or args.mode == "both":
    #     kernel_names = ["lhs_rhs_gemm_np", "lhs_rhs_meta_gemm"]
    #     plot_metric(args.cache_dir, "min_ms", kernel_names)
    #     plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
