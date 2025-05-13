# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict

import numpy as np
from neuronpy.runtime.spike import SpikeExecutor
from tqdm import tqdm

from autotune.cache.directories import split_file_info
from autotune.cache.results import PerformanceMetrics, PerformanceResult
from autotune.tune.job import ProfileJob, ProfileJobs
from autotune.tune.metrics import extract_metrics
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
        self.num_workers = max(self.num_workers, 1)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def __call__(self):
        self._parallel_preprocessing()
        self._parallel_compile_to_neff()
        kernel_outputs = self._execute()
        self._parallel_extract_metrics()
        self._parallel_postprocessing(kernel_outputs)
        self.results.save(cache_dir=self.cache_dir)

    def _init_results(self, main_metric: str, lower_is_better: bool) -> PerformanceMetrics:
        results = PerformanceMetrics(sort_key=main_metric, lower_is_better=lower_is_better)
        for job in self.jobs:
            # NOTE: hardcoded saving of job's data fields
            results.add_result(kernel_kwargs=job.kernel_kwargs, compiler_flags=job.compiler_flags)
        return results

    def _parallel_preprocessing(self):
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                job = self.jobs[job_id]
                future = executor.submit(job.preprocessing, job.input_tensors, job.kernel_kwargs)
                futures.append((job_id, future))
        for job_id, future in tqdm(futures, total=len(futures), desc="Preprocessing"):
            result = self.results[job_id]
            try:
                preprocessing_is_ok = future.result()
                result.add_fields(preprocessing_result=preprocessing_is_ok)
            except Exception as e:
                error_string = capture_error_message(e)
                result.add_error(error_string)

    def _parallel_compile_to_neff(self):
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                job = self.jobs[job_id]
                future = executor.submit(
                    compile_kernel,
                    job.kernel_name,
                    job.name,
                    job.input_tensors,
                    job.kernel_kwargs,
                    job.compiler_flags,
                    self.cache_dir,
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

    def _execute(self) -> Dict[int, np.ndarray]:
        kernel_outputs: Dict[int, np.ndarray] = {}
        with SpikeExecutor(verbose=0) as spike:
            for job_id in tqdm(range(self.jobs.num_jobs), total=self.jobs.num_jobs, desc="Executing kernels"):
                job = self.jobs[job_id]
                result = self.results[job_id]
                try:
                    spike_kernel = create_spike_kernel(
                        result.neff, job.kernel_name, job.input_tensors, job.kernel_kwargs
                    )
                    stats = spike.benchmark(
                        spike_kernel,
                        *job.input_tensors,
                        **job.kernel_kwargs,
                        warmup_iterations=self.warmup,
                        benchmark_iterations=self.iters,
                        device_id=0,
                    )
                    kernel_output = self._capture_neff(spike, spike_kernel, job, result)
                    result.add_fields(**stats)
                except Exception as e:
                    error_string = capture_error_message(e)
                    result.add_error(error_string)
                    kernel_output = None
                kernel_outputs[job_id] = kernel_output
        return kernel_outputs

    def _capture_neff(self, spike, spike_kernel, job: ProfileJob, result: PerformanceResult) -> np.ndarray:
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
        # TODO: re-use capture_neff from utils.py
        directory, neff_name, file_type = split_file_info(result.neff)
        # FIXME: an extra fp32 run for correctness
        kernel_output = spike.run(
            spike_kernel, *job.input_tensors, save_trace=True, artifacts_dir=directory, **job.kernel_kwargs
        )
        ntff_file = f"{directory}/{neff_name}.ntff"
        shutil.move(f"{directory}/profile.ntff", ntff_file)
        result.add_fields(ntff=ntff_file)
        return kernel_output

    def _parallel_extract_metrics(self):
        """Extract profile metrics for all jobs in parallel."""
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                result = self.results[job_id]
                future = executor.submit(extract_metrics, result.neff, result.ntff)
                futures.append((job_id, future))

        for job_id, future in tqdm(futures, desc="Extracting metrics"):
            result = self.results[job_id]
            try:
                metrics = future.result()
                result.add_fields(metrics=metrics)
                if result.ntff and os.path.exists(result.ntff):
                    os.remove(result.ntff)
                    result.remove_fields("ntff")
            except Exception as e:
                error_msg = capture_error_message(e)
                result.add_error(error_msg)
                result.add_fields(hfu=0)

    def _parallel_postprocessing(self, kernel_outputs: Dict[int, np.ndarray]):
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job_id in range(self.jobs.num_jobs):
                job = self.jobs[job_id]
                kernel_output = kernel_outputs[job_id]
                result = self.results[job_id]
                future = executor.submit(
                    job.postprocessing, job.input_tensors, job.kernel_kwargs, kernel_output, result.metrics
                )
                futures.append((job_id, future))
        for job_id, future in tqdm(futures, total=len(futures), desc="Postprocessing"):
            result = self.results[job_id]
            try:
                postprocessing_is_ok = future.result()
                result.add_fields(postprocessing_result=postprocessing_is_ok)
            except Exception as e:
                error_string = capture_error_message(e)
                result.add_error(error_string)
