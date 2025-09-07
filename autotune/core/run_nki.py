import os
from typing import List

from neuronpy.runtime.spike import SpikeExecutor

from autotune.cache.results import ProfileResult, capture_error_message
from autotune.core.compile import compile_kernel, create_spike_kernel, run_spike_kernel
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics


def run_on_neuron_core(warmup: int, iters: int, jobs: ProfileJobs, results: List[ProfileResult]) -> List[ProfileResult]:
    """Run a Python script with a specific NEURON_RT_VISIBLE_CORES setting"""
    neuron_core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "NOT SET")

    # Pre-initialize all input tensors once for all jobs with the same shapes
    jobs.initialize_input_tensors()

    spike = SpikeExecutor(verbose=0)
    spike_instance = spike.__enter__()
    for job, result in zip(jobs.jobs, results):
        try:
            assert job.index == result.index, f"job index {job.index} mismatch result index {result.index}"
            neff = compile_kernel(
                kernel_name=job.kernel,
                input_tensors=job.input_tensors,
                kernel_kwargs=job.kernel_kwargs,
                target_instance_family=job.target_instance_family,
                compiler_flags=job.compiler_flags,
                output_dir=job.cache_dir,
            )
            result.add_fields(neff=neff)

            spike_kernel = create_spike_kernel(neff, job.kernel, job.input_tensors, job.kernel_kwargs)
            stats = spike.benchmark(
                spike_kernel,
                *job.input_tensors,
                **job.kernel_kwargs,
                warmup_iterations=warmup,
                benchmark_iterations=iters,
                device_id=0,
            )
            ntff_file, kernel_outputs = run_spike_kernel(
                spike, spike_kernel, job.input_tensors, neff, job.kernel_kwargs
            )
            result.add_fields(ntff=ntff_file, **stats)

            job.postprocessing(job.input_tensors, job.kernel_kwargs, kernel_outputs)
            result.add_fields(postprocessing_result=True)

            metrics = extract_metrics(
                neff,
                ntff_file,
                latency=result.min_ms,
                matmul_mac_count=result.matmul_mac_count,
                target_instance_family=job.target_instance_family,
            )
            result.add_fields(**metrics)
        except Exception as e:
            error_msg = capture_error_message(e)
            result.add_error(error_msg)
    spike.__exit__(None, None, None)
    return results
