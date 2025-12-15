# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Any

import numpy as np
from nkipy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.visualize import plot_metric
from autotune.gemm import GEMMCorrectness, sample_gemm_configs


def generate_shapes() -> list[tuple[int, int, int]]:
    """Generate (M, N, K) shape tuples for GEMM benchmarking."""
    shapes = []
    for size in range(512, 1024 + 1, 512):
        shapes.append((size, size, size))
    return shapes


def collect_job_configs(shapes: list[tuple[int, int, int]], transposed_lhs: bool) -> list[dict[str, Any]]:
    """Collect all job configurations for GEMM benchmarking.

    Args:
        shapes: List of (M, N, K) shape tuples
        transposed_lhs: Whether to transpose the left-hand side matrix

    Returns:
        List of kwargs dictionaries for ProfileJobs.add_job()
    """
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

    job_list = []

    for M, N, K in shapes:
        if transposed_lhs:
            lhs_shape = (K, M)
        else:
            lhs_shape = (M, K)
        rhs_shape = (K, N)
        configs = sample_gemm_configs(M=M, N=N, K=K, max_configs=10)
        for config in configs:
            job_list.append(
                {
                    "kernel": meta_kernel,
                    "input_tensor_shapes": {"lhs": lhs_shape, "rhs": rhs_shape},
                    "data_type": data_type,
                    "kernel_kwargs": {"config": config},
                    "compiler_flags": "--auto-cast=none --internal-tensorizer-opt-level=nki",
                    "postprocessing": postprocessing,
                    "mac_count": M * N * K,
                }
            )
        job_list.append(
            {
                "kernel": baseline_kernel,
                "input_tensor_shapes": {"lhs": lhs_shape, "rhs": rhs_shape},
                "data_type": data_type,
                "kernel_kwargs": {},
                "compiler_flags": "--auto-cast=none --model-type=transformer --tensorizer-options='--print-nki'",
                "postprocessing": postprocessing,
                "mac_count": M * N * K,
            }
        )

    return job_list


def run_jobs_in_batches(job_list: list[dict[str, Any]], cache_dir: str, batch_size) -> None:
    """Execute jobs in batches to prevent machine crashes from too many concurrent jobs.

    Args:
        job_list: List of job configuration dictionaries
        cache_dir: Root directory for the benchmark cache
        batch_size: Maximum jobs per batch (default: 10000)
    """
    total_jobs = len(job_list)
    num_batches = (total_jobs + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_jobs)
        batch = job_list[start_idx:end_idx]
        print(
            f"Executing batch {batch_num + 1}/{num_batches} with {len(batch)} jobs "
            f"(jobs {start_idx + 1}-{end_idx} of {total_jobs})..."
        )
        all_jobs = ProfileJobs(cache_root_dir=cache_dir, target_instance_family="trn2")
        for job_kwargs in batch:
            all_jobs.add_job(**job_kwargs)
        tuner = Benchmark(jobs=all_jobs)
        tuner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmarks with different matrix configurations")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lhsT_rhs", "lhs_rhs", "both"],
        help="Matrix multiplication mode: lhsT_rhs (transposed LHS), lhs_rhs, or both",
    )
    parser.add_argument("--cache-dir", type=str, help="Root directory for the benchmark cache")
    args = parser.parse_args()

    shapes = generate_shapes()
    all_job_configs = []
    if args.mode == "lhsT_rhs" or args.mode == "both":
        all_job_configs.extend(collect_job_configs(shapes, transposed_lhs=True))
    if args.mode == "lhs_rhs" or args.mode == "both":
        all_job_configs.extend(collect_job_configs(shapes, transposed_lhs=False))
    if all_job_configs:
        run_jobs_in_batches(all_job_configs, args.cache_dir, batch_size=20000)

    if args.mode == "lhsT_rhs" or args.mode == "both":
        kernel_names = ["lhsT_rhs_gemm_np", "lhsT_rhs_meta_gemm"]
        plot_metric(args.cache_dir, "min_ms", kernel_names)
        plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
    if args.mode == "lhs_rhs" or args.mode == "both":
        kernel_names = ["lhs_rhs_gemm_np", "lhs_rhs_meta_gemm"]
        plot_metric(args.cache_dir, "min_ms", kernel_names)
        plot_metric(args.cache_dir, "mfu_estimated_percent", kernel_names)
