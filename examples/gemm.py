# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from itertools import permutations

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.visualize import plot_metric
from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.generation.generate import generate_configs
from autotune.modules.matmul import GEMMConfig, GEMMCorrectness


def add_jobs(all_jobs: ProfileJobs, transposed_lhs: bool = False):
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
        baseline_kernel = ("/home/ec2-user/workplace/nki-autotune/autotune/modules/matmul.py", "lhsT_rhs_gemm_np")
        meta_kernel = (
            "/home/ec2-user/workplace/nki-autotune/autotune/generation/gemm_generate.py",
            "lhsT_rhs_meta_gemm",
        )
    else:
        baseline_kernel = ("/home/ec2-user/workplace/nki-autotune/autotune/modules/matmul.py", "lhs_rhs_gemm_np")
        meta_kernel = (
            "/home/ec2-user/workplace/nki-autotune/autotune/generation/gemm_generate.py",
            "lhs_rhs_meta_gemm",
        )

    kernel_params = {
        "NUM_BLOCK_M": [1, 2, 4, 8, 16, 32, 64, 128],
        "NUM_BLOCK_N": [1, 2, 4, 8, 16, 32, 64, 128],
        "NUM_BLOCK_K": [1, 2, 4, 8, 16, 32, 64, 128],
        "loop_order": ["".join(perm) for perm in permutations("MKN")],
        "lhs_position": [0, 1, 2],
        "rhs_position": [0, 1, 2],
    }
    kernel_params = {
        "NUM_BLOCK_M": [1],
        "NUM_BLOCK_N": [4],
        "NUM_BLOCK_K": [8],
        "loop_order": ["NKM"],
        "lhs_position": [1],
        "rhs_position": [1],
    }
    kernel_configs = generate_configs(**kernel_params)

    # for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384), (24576, 24576, 24576)]:
    for M, N, K in [(1024, 4096, 4659)]:
        if transposed_lhs:
            lhs_shape = (K, M)
        else:
            lhs_shape = (M, K)
        rhs_shape = (K, N)
        valid_kernel_configs = []
        for kernel_config in kernel_configs:
            try:
                gemm_config = GEMMConfig()
                gemm_config(lhs_shape=lhs_shape, rhs_shape=rhs_shape, transposed_lhs=transposed_lhs, **kernel_config)
                valid_kernel_configs.append(kernel_config)
                print(gemm_config)
            except Exception as e:
                pass
        # valid_kernel_configs = random.sample(valid_kernel_configs, 500)
        for kernel_config in valid_kernel_configs:
            all_jobs.add_job(
                kernel=meta_kernel,
                input_tensor_shapes=[lhs_shape, rhs_shape],
                data_type=data_type,
                kernel_kwargs=kernel_config,
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
    all_jobs = ProfileJobs()
    if args.mode == "lhsT_rhs" or args.mode == "both":
        add_jobs(all_jobs, transposed_lhs=True)
    if args.mode == "lhs_rhs" or args.mode == "both":
        add_jobs(all_jobs, transposed_lhs=False)
    tuner = Benchmark(jobs=all_jobs, cache_root_dir=args.cache_dir)
    tuner()

    if args.mode == "lhsT_rhs" or args.mode == "both":
        kernel_names = ["lhsT_rhs_gemm_np", "lhsT_rhs_meta_gemm"]
        plot_metric(args.cache_dir, "min_ms", kernel_names)
        plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
    if args.mode == "lhs_rhs" or args.mode == "both":
        kernel_names = ["lhs_rhs_gemm_np", "lhs_rhs_meta_gemm"]
        plot_metric(args.cache_dir, "min_ms", kernel_names)
        plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
