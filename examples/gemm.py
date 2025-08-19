# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from glob import glob
from itertools import product

import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.visualize import plot_metric
from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.generation.generate import generate_configs
from autotune.modules.matmul import GEMMCompatibility, GEMMCorrectness


def get_configs():
    num_blocks = [1, 2, 4]
    kernel_params = {"NUM_BLOCK_M": num_blocks, "NUM_BLOCK_N": num_blocks, "NUM_BLOCK_K": num_blocks}
    # kernel_params = {"NUM_BLOCK_M": [1], "NUM_BLOCK_N": [1], "NUM_BLOCK_K": [1]}
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

    # Choose function names based on transposed_lhs
    kernel_func = "lhsT_rhs_gemm" if transposed_lhs else "lhs_rhs_gemm"
    kernel_func_np = "lhsT_rhs_gemm_np" if transposed_lhs else "lhs_rhs_gemm_np"
    num_repeats = 1

    kernel_files = glob(f"generated_kernels/{kernel_func}/*.py")
    for kernel_file in kernel_files:
        for kernel_config in kernel_configs:
            for _ in range(num_repeats):
                if transposed_lhs:
                    lhsT = np.random.normal(0, 0.001, size=(K, M)).astype(data_type)
                    rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
                    input_tensors = (lhsT, rhs)
                else:
                    lhs = np.random.normal(0, 0.001, size=(M, K)).astype(data_type)
                    rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
                    input_tensors = (lhs, rhs)

                jobs.add_job(
                    kernel=(kernel_file, kernel_func),
                    input_tensors=input_tensors,
                    kernel_kwargs=kernel_config,
                    compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
                    preprocessing=GEMMCompatibility(transposed_lhs=transposed_lhs),
                    postprocessing=postprocessing,
                )

    all_jobs.extend(jobs)

    # Add the numpy reference implementation
    if transposed_lhs:
        lhsT = np.random.normal(0, 0.001, size=(K, M)).astype(data_type)
        rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
        input_tensors = (lhsT, rhs)
    else:
        lhs = np.random.normal(0, 0.001, size=(M, K)).astype(data_type)
        rhs = np.random.normal(0, 0.001, size=(K, N)).astype(data_type)
        input_tensors = (lhs, rhs)

    all_jobs.add_job(
        kernel=("autotune/modules/matmul.py", kernel_func_np),
        input_tensors=input_tensors,
        kernel_kwargs={},
        compiler_flags="--target=trn1 --auto-cast=none --model-type=transformer --tensorizer-options='--print-nki'",
        preprocessing=None,
        postprocessing=postprocessing,
    )


def add_jobs(all_jobs: ProfileJobs, transposed_lhs: bool = False):
    print(f"Adding jobs with transposed={transposed_lhs} LHS matrix...")
    MNK = list(product([1025], [2014], [1111]))
    for M, N, K in [(128, 512, 128), (129, 512, 128)]:
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
    kernel_names = ["lhs_rhs_gemm", "lhsT_rhs_gemm"]
    plot_metric(args.cache_dir, "min_ms", kernel_names)
    plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
