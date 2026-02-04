# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tensor add benchmark using the new NEFF compilation backend.

This example demonstrates the full autotune pipeline:
1. Define multiple kernel configurations with different tensor sizes
2. Compile kernels in parallel using compile_nki_ir_kernel_to_neff API
3. Run compiled kernels on Trainium hardware
4. Verify numerical correctness against numpy golden reference

Run with:
    python examples/tensor_add_e2e.py --cache-dir /tmp/tensor_add_benchmark

Requirements:
    - Trainium hardware (trn1/trn2 instance)
    - NeuronX SDK installed
"""

import argparse
import os

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.metrics import check_correctness
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors.

    Args:
        a_input: Input tensor of shape [P, F] where P <= 128 (pmax).
        b_input: Input tensor of shape [P, F], must match a_input shape.

    Returns:
        c_output: Output tensor of shape [P, F] containing a_input + b_input.
    """
    assert a_input.shape == b_input.shape
    assert a_input.shape[0] <= nl.tile_size.pmax

    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    c_output = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    return c_output


class TensorAddCorrectness:
    """Postprocessing to verify tensor add against numpy golden.

    Compares kernel output against numpy reference implementation to verify
    numerical correctness of the compiled tensor add kernel on Trainium hardware.
    """

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ) -> None:
        """Compare kernel output with numpy reference implementation.

        Args:
            input_tensors: Dictionary with 'a_input' and 'b_input' tensors.
            kernel_kwargs: Dictionary of kernel keyword arguments (unused).
            nki_out_tensors: Tuple of kernel output arrays from Trainium hardware.

        Raises:
            AssertionError: If kernel output does not match expected result.
        """
        a_input = input_tensors["a_input"]
        b_input = input_tensors["b_input"]
        golden = a_input + b_input
        nki_out = nki_out_tensors[0]
        check_correctness(golden, nki_out, atol=1e-5, rtol=1e-5)


def generate_shapes() -> list[tuple[int, int]]:
    """Generate (P, F) shape tuples for tensor add benchmarking.

    Kernel constraints:
        - P must be <= 128 (pmax)

    Returns:
        List of valid (P, F) tuples.
    """
    shapes = [(64, 256), (64, 512), (128, 256), (128, 512), (128, 1024), (128, 2048)]
    return shapes


def collect_job_configs(shapes: list[tuple[int, int]]) -> list[dict]:
    """Collect all job configurations for tensor add benchmarking.

    Args:
        shapes: List of (P, F) shape tuples.

    Returns:
        List of kwargs dictionaries for ProfileJobs.add_job().
    """
    current_file = os.path.abspath(__file__)
    kernel_name = (current_file, "nki_tensor_add_kernel")
    postprocessing = TensorAddCorrectness()

    job_list = []
    for P, F in shapes:
        tensor_shape = (P, F)

        job_list.append(
            {
                "kernel": kernel_name,
                "input_tensor_shapes": {"a_input": tensor_shape, "b_input": tensor_shape},
                "output_tensor_shapes": {"c_output": tensor_shape},
                "data_type": np.float32,
                "kernel_kwargs": {},
                "compiler_flags": "--auto-cast=none --internal-tensorizer-opt-level=nki",
                "postprocessing": postprocessing,
            }
        )

    return job_list


def run_benchmark(job_list: list[dict], cache_dir: str) -> None:
    """Execute the full benchmark pipeline.

    Args:
        job_list: List of job configuration dictionaries.
        cache_dir: Root directory for the benchmark cache.
    """
    print(f"Running benchmark with {len(job_list)} kernel configurations...")

    jobs = ProfileJobs(cache_root_dir=cache_dir, target_instance_family="trn2")
    for job_kwargs in job_list:
        jobs.add_job(**job_kwargs)

    tuner = Benchmark(jobs=jobs, warmup=5, iters=50)
    tuner()

    print("\nBenchmark Results:")
    print("-" * 80)
    for job_index, job in jobs.jobs.items():
        if job.has_error:
            print(f"Job {job_index}: ERROR - {getattr(job, 'error', 'unknown')[:60]}")
        else:
            min_ms = getattr(job, "min_ms", "N/A")
            correct = job.is_correct
            shapes = job.input_tensor_shapes
            print(f"Job {job_index}: {shapes} -> min_ms={min_ms}, correct={correct}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tensor add e2e benchmark with new NEFF compilation backend")
    parser.add_argument("--cache-dir", type=str, required=True, help="Root directory for the benchmark cache")
    args = parser.parse_args()

    shapes = generate_shapes()
    job_configs = collect_job_configs(shapes)
    run_benchmark(job_configs, args.cache_dir)
