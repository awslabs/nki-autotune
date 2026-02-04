import os

from nkipy.runtime import BaremetalExecutor

from autotune.core.compile import TensorStub, create_spike_kernel, run_spike_kernel
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics
from autotune.core.utils import capture_error_message


def run_neuron_benchmarks(jobs: ProfileJobs, warmup: int, iters: int) -> None:
    """Run benchmarks on Neuron cores using SpikeExecutor.

    Args:
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.
        jobs: ProfileJobs containing all jobs to run.
    """
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = jobs.target_instance_family

    with BaremetalExecutor(verbose=0) as spike:
        for job_index in jobs.jobs:
            job = jobs.jobs[job_index]
            if job.has_error:
                continue

            try:
                output_tensors = [
                    TensorStub(shape=shape, dtype=job.data_type, name=name)
                    for name, shape in job.output_tensor_shapes.items()
                ]
                spike_kernel = create_spike_kernel(
                    job.neff, job.kernel, job.input_tensors, output_tensors, job.kernel_kwargs
                )
                stats = spike.benchmark(
                    spike_kernel,
                    *job.input_tensors.values(),
                    **job.kernel_kwargs,
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
                    spike, spike_kernel, job.input_tensors, job.neff, job.kernel_kwargs
                )
                job.add_attributes(ntff=ntff_file, **stats_dict)
                job.postprocessing(job.input_tensors, job.kernel_kwargs, kernel_outputs)
                job.add_attributes(postprocessing_result=True)
                if os.path.exists(ntff_file):
                    metrics = extract_metrics(
                        job.neff,
                        ntff_file,
                        latency=job.min_ms,
                        matmul_mac_count=job.mac_count,
                        target_instance_family=jobs.target_instance_family,
                    )
                    job.add_attributes(**metrics)

            except Exception as e:
                error_msg = capture_error_message(e)
                job.add_error(error_msg)


def run_on_neuron_core(warmup: int, iters: int, jobs: ProfileJobs) -> ProfileJobs:
    """Run kernels with separated CPU compilation and Neuron execution phases.

    This function initializes ProfileResult objects for each job, then
    compiles all kernels on CPU (without SpikeExecutor), and finally
    runs benchmarks on Neuron cores (with SpikeExecutor).

    Args:
        warmup: Number of warmup iterations.
        iters: Number of benchmark iterations.
        jobs: ProfileJobs containing all jobs to run.

    Returns:
        Updated ProfileJobs with benchmark results.
    """
    jobs.initialize_input_tensors()

    run_neuron_benchmarks(jobs, warmup, iters)

    return jobs
