# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from neuronpy.runtime.spike import SpikeExecutor
from tqdm import tqdm

from autotune.cache.directories import split_file_info
from autotune.cache.results import PerformanceMetrics
from autotune.tune.job import ProfileJobs
from autotune.tune.metrics import dump_profile_json, parse_hfu
from autotune.tune.utils import capture_error_message, compile_kernel, create_spike_kernel


class Benchmark:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        jobs: ProfileJobs,
        cache_dir: str,
        main_metric: str = "min_ms",
        lower_is_better: bool = True,
        warmup: int = 10,
        iters: int = 100,
    ):
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters
        self.results = self._init_results(main_metric, lower_is_better)
        self.num_workers = min(len(jobs.jobs), os.cpu_count() - 1)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def __call__(self):
        """
        Parallel compilation
        TODO:
        1. Parallel compile to NEFF
        2. Run postprocessing
        """
        self._parallel_compile_to_neff()
        self._profile()
        # self._extract_hfu()
        self._parallel_extract_hfu()
        self.results.save(cache_dir=self.cache_dir)
        """
        TODO: add postprocessing function. postprocessing_func = xxx.
        Indicate error source.
        postprocessing_func(kernel, kernel_args, **config):
            assert allclose
        """
        return None

    def _init_results(self, main_metric: str, lower_is_better: bool) -> PerformanceMetrics:
        results = PerformanceMetrics(sort_key=main_metric, lower_is_better=lower_is_better)
        for job in self.jobs:
            results.add_result(config=job.kwargs)
        return results

    def _parallel_compile_to_neff(self):
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                job = self.jobs[job_id]
                future = executor.submit(
                    compile_kernel, job.kernel_name, job.name, job.kernel_args, job.kwargs, self.cache_dir
                )
                futures.append((job_id, future))
        for job_id, future in tqdm(futures, total=len(futures), desc="Compiling kernels"):
            result = self.results[job_id]
            try:
                neff = future.result()
                result.add_fields(neff=neff)
            except Exception as e:
                error_string = capture_error_message(e)
                result.add_error(error_string)

    def _profile(self):
        with SpikeExecutor(verbose=0) as spike:
            for job_id in tqdm(range(self.jobs.num_jobs), total=self.jobs.num_jobs, desc="Profiling kernels"):
                job = self.jobs[job_id]
                result = self.results[job_id]
                try:
                    spike_kernel = create_spike_kernel(result.neff, job.kernel_name, job.kernel_args, job.kwargs)
                    stats = spike.benchmark(
                        spike_kernel,
                        *job.kernel_args,
                        **job.kwargs,
                        warmup_iterations=self.warmup,
                        benchmark_iterations=self.iters,
                        device_id=0,
                    )
                    self._capture_neff(spike, spike_kernel, job, result)
                    result.add_fields(**stats)
                except Exception as e:
                    error_string = capture_error_message(e)
                    result.add_error(error_string)

    def _capture_neff(self, spike, spike_kernel, job, result):
        """
        Generate trace profiles for compiled kernel files.

        This method processes each NEFF (Neuron Executable File Format) file by:
        1. Capturing a trace profile using neuron-profile
        2. Moving the resulting trace file (NTFF) to the appropriate location
        3. Creating an upload command for the profile data and logging it

        Args:
            neff_files (str): NEFF file to be traced.

        Raises:
            AssertionError: If any of the provided files is not a .neff file.

        Note:
            This method is used when the 'trace' flag is set to True, allowing
            for detailed performance analysis of the compiled kernels.
        """
        directory, neff_name, file_type = split_file_info(result.neff)
        spike.run(spike_kernel, *job.kernel_args, save_trace=True, artifacts_dir=directory, **job.kwargs)
        ntff_file = f"{directory}/{neff_name}.ntff"
        shutil.move(f"{directory}/profile.ntff", ntff_file)
        result.add_fields(ntff=ntff_file)

    def _extract_hfu(self):
        for job_id in tqdm(range(self.jobs.num_jobs), total=self.jobs.num_jobs, desc="Extracting HFU"):
            result = self.results[job_id]
            output_json_file = f"{self.cache_dir}/file.json"

            try:
                # Run subprocess with error checking and output capture
                dump_json_cmd = f"neuron-profile view -n {result.neff} -s {result.ntff} --output-format json --output-file {output_json_file}"
                process = subprocess.run(dump_json_cmd, shell=True, capture_output=True, text=True, check=False)

                # Check if the command was successful
                if process.returncode != 0:
                    raise Exception(f"Command failed with exit code {process.returncode}. Error: {process.stderr}")

                # Check if the file was created
                if not os.path.exists(output_json_file):
                    raise Exception(f"JSON output file was not created: {output_json_file}")

                with open(output_json_file, "r") as f:
                    data = json.load(f)
                    hfu = data["summary"][0]["hfu_estimated_percent"]
                result.add_fields(hfu=hfu)

            except Exception as e:
                error_string = capture_error_message(e)
                result.add_error(error_string)
                result.add_fields(hfu=0)
            finally:
                # Safe cleanup - check if files exist before removing
                if os.path.exists(output_json_file):
                    try:
                        os.remove(output_json_file)
                    except Exception as e:
                        print(f"Warning: Failed to remove {output_json_file}: {e}")

                if hasattr(result, "ntff") and result.ntff and os.path.exists(result.ntff):
                    try:
                        os.remove(result.ntff)
                        result.remove_fields("ntff")
                    except Exception as e:
                        print(f"Warning: Failed to remove {result.ntff}: {e}")

    def _parallel_extract_hfu(self):
        """Extract HFU data for all jobs in parallel."""
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                result = self.results[job_id]
                future = executor.submit(dump_profile_json, result.neff, result.ntff)
                futures.append((job_id, future))

        for job_id, future in tqdm(futures, desc="Extracting HFU"):
            result = self.results[job_id]
            try:
                profile_json = future.result()
                hfu = parse_hfu(profile_json)
                result.add_fields(hfu=hfu)
                if result.ntff and os.path.exists(result.ntff):
                    os.remove(result.ntff)
                    result.remove_fields("ntff")
            except Exception as e:
                error_msg = capture_error_message(e)
                result.add_error(error_msg)
                result.add_fields(hfu=0)
