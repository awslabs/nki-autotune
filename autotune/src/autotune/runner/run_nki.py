"""Run compiled NKI kernels on Neuron hardware for benchmarking."""

import os
from typing import Any

import numpy as np
from nkipy.runtime import BaremetalExecutor

from autotune.analysis.metrics import calculate_mfu, check_correctness, extract_metrics
from autotune.compiler.compile import TensorStub, create_spike_kernel, run_spike_kernel
from autotune.job import ProfileJob, ProfileJobs
from autotune.utils import capture_error_message


def _run_correctness_check(job: ProfileJob, kernel_outputs: tuple[np.ndarray, ...]) -> None:
    """Run correctness verification against golden reference if configured.

    Args:
        job: The profiling job with correctness_check config.
        kernel_outputs: List of output arrays from kernel execution.
    """
    if job.correctness_check is None:
        return
    try:
        golden_fn, atol, rtol = job.correctness_check
        golden = golden_fn(**job.input_tensors, **getattr(job, "scalar_kwargs"))
        actual = kernel_outputs[0]
        if golden.dtype != actual.dtype:
            actual = actual.astype(golden.dtype)
        check_correctness(golden, actual, atol=atol, rtol=rtol)
        job.add_attributes(correctness_result=True)
    except Exception as e:
        error_msg = capture_error_message(e)
        job.add_attributes(correctness_result=False, correctness_error=error_msg)


def _run_metrics_extraction(job: ProfileJob) -> None:
    """Extract hardware metrics from NEFF/NTFF trace files.

    Args:
        job: The profiling job with ntff and neff paths.
    """
    ntff_file: str = getattr(job, "ntff")
    if not os.path.exists(ntff_file):
        return
    try:
        neff: str = getattr(job, "neff")
        min_ms: float = getattr(job, "min_ms")
        mac_count: int = getattr(job, "mac_count")
        metrics = extract_metrics(neff, ntff_file, latency=min_ms, matmul_mac_count=mac_count)
        job.add_attributes(**metrics)
    except Exception as e:
        error_msg = capture_error_message(e)
        job.add_attributes(metrics_error=error_msg)


def _benchmark_single_job(spike: BaremetalExecutor, job: ProfileJob, warmup: int, iters: int) -> tuple[np.ndarray, ...]:
    """Benchmark a single job and record timing + MFU metrics.

    Args:
        spike: The BaremetalExecutor instance.
        job: The profiling job to benchmark.
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.

    Returns:
        Kernel output arrays, or None if benchmarking failed.
    """
    output_tensors = [
        TensorStub(shape=shape, dtype=getattr(job, "data_type"), name=name)
        for name, shape in getattr(job, "output_shapes").items()
    ]
    neff: str = getattr(job, "neff")
    kernel: tuple[str, str] = getattr(job, "kernel")
    scalar_kwargs: dict[str, Any] = getattr(job, "scalar_kwargs")
    spike_kernel = create_spike_kernel(neff, kernel, job.input_tensors, output_tensors, scalar_kwargs)
    stats = spike.benchmark(
        spike_kernel, *job.input_tensors.values(), **scalar_kwargs, warmup_iterations=warmup, benchmark_iterations=iters
    )
    ntff_file, kernel_outputs = run_spike_kernel(spike, spike_kernel, job.input_tensors, neff, scalar_kwargs)
    mac_count: int = getattr(job, "mac_count")
    mfu = calculate_mfu(mac_count, stats.min_ms)
    durations = sorted(stats.durations_ms)
    n = len(durations)
    p50_ms = durations[n // 2] if n > 0 else 0.0
    p99_ms = durations[min(int(n * 0.99), n - 1)] if n > 0 else 0.0
    job.add_attributes(
        ntff=ntff_file,
        mean_ms=stats.mean_ms,
        min_ms=stats.min_ms,
        max_ms=stats.max_ms,
        std_dev_ms=stats.std_dev_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        iterations=stats.iterations,
        warmup_iterations=stats.warmup_iterations,
        mfu=mfu * 100,
    )
    return kernel_outputs


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
                outputs = _benchmark_single_job(spike, job, warmup, iters)
            except Exception as e:
                job.add_error(capture_error_message(e))
                continue

            if outputs is not None:
                _run_correctness_check(job, outputs)
                _run_metrics_extraction(job)


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
