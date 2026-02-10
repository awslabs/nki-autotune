"""Run compiled NKI kernels on Neuron hardware for benchmarking."""

import os

from nkipy.runtime import BaremetalExecutor

from autotune.analysis.metrics import check_correctness, extract_metrics
from autotune.compiler.compile import TensorStub, create_spike_kernel, run_spike_kernel
from autotune.job import ProfileJobs
from autotune.utils import capture_error_message


def run_neuron_benchmarks(jobs: ProfileJobs, warmup: int, iters: int) -> None:
    """Run benchmarks on Neuron cores using BaremetalExecutor.

    Args:
        jobs: ProfileJobs containing all jobs to run.
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.
    """
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

    with BaremetalExecutor(verbose=0) as spike:
        for job_index in jobs.jobs:
            job = jobs.jobs[job_index]
            if job.has_error:
                continue

            try:
                output_tensors = [
                    TensorStub(shape=shape, dtype=job.data_type, name=name) for name, shape in job.output_shapes.items()
                ]
                spike_kernel = create_spike_kernel(
                    job.neff, job.kernel, job.input_tensors, output_tensors, job.scalar_kwargs
                )
                stats = spike.benchmark(
                    spike_kernel,
                    *job.input_tensors.values(),
                    **job.scalar_kwargs,
                    warmup_iterations=warmup,
                    benchmark_iterations=iters,
                )
                stats_dict = {
                    "mean_ms": stats.mean_ms,
                    "min_ms": stats.min_ms,
                    "max_ms": stats.max_ms,
                    "std_dev_ms": stats.std_dev_ms,
                    "iterations": stats.iterations,
                    "warmup_iterations": stats.warmup_iterations,
                }
                ntff_file, kernel_outputs = run_spike_kernel(
                    spike, spike_kernel, job.input_tensors, job.neff, job.scalar_kwargs
                )
                job.add_attributes(ntff=ntff_file, **stats_dict)
                if job.correctness_check is not None:
                    golden_fn, atol, rtol = job.correctness_check
                    golden = golden_fn(**job.input_tensors, **job.scalar_kwargs)
                    actual = kernel_outputs[0]
                    if golden.dtype != actual.dtype:
                        actual = actual.astype(golden.dtype)
                    check_correctness(golden, actual, atol=atol, rtol=rtol)
                    job.add_attributes(correctness_result=True)
                if os.path.exists(ntff_file):
                    metrics = extract_metrics(job.neff, ntff_file, latency=job.min_ms, matmul_mac_count=job.mac_count)
                    job.add_attributes(**metrics)

            except Exception as e:
                error_msg = capture_error_message(e)
                job.add_error(error_msg)


def run_on_neuron_core(warmup: int, iters: int, jobs: ProfileJobs) -> ProfileJobs:
    """Run benchmarks on a single Neuron core.

    Args:
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.
        jobs: ProfileJobs containing all jobs to run.

    Returns:
        Updated ProfileJobs with benchmark results.
    """
    run_neuron_benchmarks(jobs, warmup, iters)
    return jobs
