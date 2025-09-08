from typing import List

from neuronpy.runtime.spike import SpikeExecutor

from autotune.cache.results import ProfileResult, capture_error_message
from autotune.core.compile import compile_kernel, create_spike_kernel, run_spike_kernel
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics


def compile_all_kernels(jobs: ProfileJobs, results: List[ProfileResult]) -> None:
    """
    Compile all kernels on CPU without requiring SpikeExecutor.

    Args:
        jobs: ProfileJobs containing all jobs to compile
        results: List of ProfileResult objects to update with compilation results
    """
    for job, result in zip(jobs.jobs, results):
        try:
            assert job.index == result.index, f"job index {job.index} mismatch result index {result.index}"

            # Compile kernel (CPU-only operation)
            neff = compile_kernel(
                kernel_name=job.kernel,
                input_tensors=job.input_tensors,
                kernel_kwargs=job.kernel_kwargs,
                target_instance_family=job.target_instance_family,
                compiler_flags=job.compiler_flags,
                output_dir=job.cache_dir,
            )
            result.add_fields(neff=neff)

        except Exception as e:
            error_msg = capture_error_message(e)
            result.add_error(error_msg)


def run_neuron_benchmarks(warmup: int, iters: int, jobs: ProfileJobs, results: List[ProfileResult]) -> None:
    """
    Run benchmarks on Neuron cores using SpikeExecutor.

    Args:
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        jobs: ProfileJobs containing all jobs to run
        results: List of ProfileResult objects to update with benchmark results
    """
    with SpikeExecutor(verbose=0) as spike:
        for job, result in zip(jobs.jobs, results):
            # Skip if compilation failed (no neff in result)
            if not hasattr(result, "neff") or result.neff is None:
                continue

            neff = result.neff

            try:

                # Create spike kernel (requires SpikeExecutor context)
                spike_kernel = create_spike_kernel(neff, job.kernel, job.input_tensors, job.kernel_kwargs)

                # Benchmark kernel (requires SpikeExecutor)
                stats = spike.benchmark(
                    spike_kernel,
                    *job.input_tensors,
                    **job.kernel_kwargs,
                    warmup_iterations=warmup,
                    benchmark_iterations=iters,
                    device_id=0,
                )

                # Run kernel and capture trace (requires SpikeExecutor)
                ntff_file, kernel_outputs = run_spike_kernel(
                    spike, spike_kernel, job.input_tensors, neff, job.kernel_kwargs
                )
                result.add_fields(ntff=ntff_file, **stats)

                # Postprocessing (CPU operation, but done here for logical flow)
                job.postprocessing(job.input_tensors, job.kernel_kwargs, kernel_outputs)
                result.add_fields(postprocessing_result=True)

                # Extract metrics (CPU operation, but done here for logical flow)
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


def run_on_neuron_core(warmup: int, iters: int, jobs: ProfileJobs, results: List[ProfileResult]) -> List[ProfileResult]:
    """
    Run kernels with separated CPU compilation and Neuron execution phases.

    This function first compiles all kernels on CPU (without SpikeExecutor),
    then runs benchmarks on Neuron cores (with SpikeExecutor).
    """

    # Pre-initialize all input tensors once for all jobs with the same shapes
    jobs.initialize_input_tensors()

    # Phase 1: Compile all kernels (CPU-only, no SpikeExecutor needed)
    compile_all_kernels(jobs, results)

    # Phase 2: Run benchmarks on Neuron (requires SpikeExecutor)
    run_neuron_benchmarks(warmup, iters, jobs, results)

    return results
