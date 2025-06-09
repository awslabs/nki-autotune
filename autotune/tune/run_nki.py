import argparse
import os
import subprocess
from typing import List

from neuronpy.runtime.spike import SpikeExecutor

from autotune.cache.results import ProfileResult, ProfileResults
from autotune.tune.job import ProfileJob, ProfileJobs
from autotune.tune.metrics import get_matmul_mac_count
from autotune.tune.utils import capture_error_message, create_spike_kernel, run_spike_kernel


def run_on_neuron_core(
    neuron_core_id: int, warmup: int, iters: int, job_ids: List[int], jobs: ProfileJobs, results: ProfileResults
) -> None:
    """Run a Python script with a specific NEURON_RT_VISIBLE_CORES setting"""
    cache_dirs = []
    for job_id in job_ids:
        job = jobs[job_id]
        result = results[job_id]
        job.save()
        result.save()
        cache_dirs.append(job.cache_dir)
    env = os.environ.copy()
    env["NEURON_RT_VISIBLE_CORES"] = str(neuron_core_id)
    cmd = ["python", "autotune/tune/run_nki.py"]
    cmd += ["--cache_dirs"] + cache_dirs
    cmd += ["--warmup", str(warmup)]
    cmd += ["--iters", str(iters)]
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.stderr:
        raise Exception(process.stderr)
    for job_id in job_ids:
        job = jobs[job_id]
        result = ProfileResult.load(job.cache_dir)
        results[job_id] = result


def main():
    """
    FIXME: make this data parallel
    1. Save results, jobs
    2. Pass results, jobs paths and job IDs
    3. Subprocess load results, jobs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dirs", type=str, nargs="+", help="A list of cahce dirs to process.")
    parser.add_argument("--warmup", type=int, help="Number of kernel warmup runs.")
    parser.add_argument("--iters", type=int, help="Number of kernel profile runs.")
    args = parser.parse_args()
    kernel_outputs = {}
    with SpikeExecutor(verbose=0) as spike:
        for cache_dir in args.cache_dirs:
            try:
                job = ProfileJob.load(cache_dir)
                result = ProfileResult.load(cache_dir)
                spike_kernel = create_spike_kernel(result.neff, job.kernel, job.input_tensors, job.kernel_kwargs)
                stats = spike.benchmark(
                    spike_kernel,
                    *job.input_tensors,
                    **job.kernel_kwargs,
                    warmup_iterations=args.warmup,
                    benchmark_iterations=args.iters,
                    device_id=0,
                )
                ntff_file, kernel_output = run_spike_kernel(
                    spike, spike_kernel, job.input_tensors, result.neff, job.kernel_kwargs
                )
                # FIXME: move matmul_mac_count to metrics extraction
                matmul_mac_count = get_matmul_mac_count(spike_kernel.traced_kernel)
                result.add_fields(ntff=ntff_file, **stats, matmul_mac_count=matmul_mac_count)
                kernel_outputs[job.index] = kernel_output
            except Exception as e:
                error_string = capture_error_message(e)
                result.add_error(error_string)
            result.save()


if __name__ == "__main__":
    main()
