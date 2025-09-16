from neuronpy.runtime.spike import SpikeExecutor

from autotune.core.compile import create_spike_kernel, run_spike_kernel
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics
from autotune.core.utils import capture_error_message


def run_neuron_benchmarks(jobs: ProfileJobs, warmup: int, iters: int) -> None:
    """
    Run benchmarks on Neuron cores using SpikeExecutor.

    Args:
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        jobs: ProfileJobs containing all jobs to run
        results: List of ProfileResult objects to update with benchmark results
    """
    with SpikeExecutor(verbose=0) as spike:
        for job_index in jobs.jobs:
            job = jobs.jobs[job_index]
            # Skip if job already failed
            if job.has_error:
                continue

            try:
                spike_kernel = create_spike_kernel(job.neff, job.kernel, job.input_tensors, job.kernel_kwargs)
                stats = spike.benchmark(
                    spike_kernel,
                    *job.input_tensors,
                    **job.kernel_kwargs,
                    warmup_iterations=warmup,
                    benchmark_iterations=iters,
                    device_id=0,
                )
                ntff_file, kernel_outputs = run_spike_kernel(
                    spike, spike_kernel, job.input_tensors, job.neff, job.kernel_kwargs
                )
                job.add_attributes(ntff=ntff_file, **stats)
                job.postprocessing(job.input_tensors, job.kernel_kwargs, kernel_outputs)
                job.add_attributes(postprocessing_result=True)
                metrics = extract_metrics(
                    job.neff,
                    ntff_file,
                    latency=job.min_ms,
                    matmul_mac_count=job.matmul_mac_count,
                    target_instance_family=jobs.target_instance_family,
                )
                job.add_attributes(**metrics)

            except Exception as e:
                error_msg = capture_error_message(e)
                job.add_error(error_msg)


def run_on_neuron_core(warmup: int, iters: int, jobs: ProfileJobs) -> ProfileJobs:
    """
    Run kernels with separated CPU compilation and Neuron execution phases.

    This function initializes ProfileResult objects for each job, then
    compiles all kernels on CPU (without SpikeExecutor), and finally
    runs benchmarks on Neuron cores (with SpikeExecutor).

    Args:
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        jobs: ProfileJobs containing all jobs to run
    """

    # Pre-initialize all input tensors once for all jobs with the same shapes
    jobs.initialize_input_tensors()

    # Run benchmarks on Neuron (requires SpikeExecutor)
    run_neuron_benchmarks(jobs, warmup, iters)

    return jobs
