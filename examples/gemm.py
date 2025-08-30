# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.generation.generate import generate_configs
from autotune.modules.matmul import GEMMCorrectness


def get_configs():
    kernel_params = {
        "NUM_BLOCK_M": [1],
        "NUM_BLOCK_N": [2],
        "NUM_BLOCK_K": [2],
        "loop_order": ["MKN"],
        "lhs_position": [1],
        "rhs_position": [2],
    }
    kernel_configs = generate_configs(**kernel_params)
    return kernel_configs


def make_gemm_jobs(all_jobs: ProfileJobs, M: int, N: int, K: int, transposed_lhs: bool = False):
    data_type = "float32"
    if data_type == "float32":
        data_type = np.float32
        postprocessing = GEMMCorrectness(transposed_lhs=transposed_lhs)
    elif data_type == "bf16":
        data_type = bfloat16
        postprocessing = None
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")

    kernel_configs = get_configs()
    jobs = ProfileJobs()

    for kernel_config in kernel_configs:
        kernel_config["transposed_lhs"] = transposed_lhs
        print(kernel_config)
        if transposed_lhs:
            lhsT = np.random.normal(0, 0.001, size=(K, M)).astype(data_type)
            rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
            input_tensors = (lhsT, rhs)
        else:
            lhs = np.random.normal(0, 0.001, size=(M, K)).astype(data_type)
            rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
            input_tensors = (lhs, rhs)

        jobs.add_job(
            kernel=("/home/ec2-user/workplace/nki-autotune/autotune/generation/gemm_generate.py", "meta_gemm_wrapper"),
            input_tensors=input_tensors,
            kernel_kwargs=kernel_config,
            compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
            preprocessing=None,
            postprocessing=postprocessing,
        )

    all_jobs.extend(jobs)


def add_jobs(all_jobs: ProfileJobs, transposed_lhs: bool = False):
    print(f"Adding jobs with transposed={transposed_lhs} LHS matrix...")
    for M, N, K in [(129, 512, 128)]:
        make_gemm_jobs(all_jobs, M, N, K, transposed_lhs=transposed_lhs)


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
    # kernel_names = ["lhs_rhs_gemm", "lhsT_rhs_gemm"]
    # plot_metric(args.cache_dir, "min_ms", kernel_names)
    # plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
