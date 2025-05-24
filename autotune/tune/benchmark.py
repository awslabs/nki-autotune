# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict

import numpy as np
from neuronpy.runtime.spike import SpikeExecutor
from tqdm import tqdm

from autotune.cache.results import PerformanceMetrics
from autotune.tune.job import ProfileJobs
from autotune.tune.metrics import extract_metrics
from autotune.tune.utils import capture_error_message, compile_kernel, create_spike_kernel, run_spike_kernel


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
        self.valid_job_ids = list(range(self.jobs.num_jobs))
        self.kernel_outputs: Dict[int, np.ndarray] = {}
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def __call__(self):
        self._parallel_preprocessing()
        self._parallel_compile_to_neff()
        self._execute()
        self._parallel_postprocessing()
        self._parallel_extract_metrics()
        self.results.save(cache_dir=self.cache_dir)

    def _get_num_workers(self) -> int:
        num_workers = min(len(self.valid_job_ids), os.cpu_count() - 1)
        num_workers = max(num_workers, 1)
        return num_workers

    def _init_results(self, main_metric: str, lower_is_better: bool) -> PerformanceMetrics:
        results = PerformanceMetrics(sort_key=main_metric, lower_is_better=lower_is_better)
        for job in self.jobs:
            # NOTE: hardcoded saving of job's data fields
            results.add_result(
                name=job.name, kernel=job.kernel, kernel_kwargs=job.kernel_kwargs, compiler_flags=job.compiler_flags
            )
        return results

    def _parallel_execute(self, work_desc: str, submit_func, process_result_func):
        """
        General function for parallel execution of jobs.

        Args:
            work_desc: Description for the progress bar
            submit_func: Function that takes a job object and returns (func, args) tuple
                        for executor.submit
            process_result_func: Function that takes (result_obj, future_result) to process results
            num_workers: Number of workers for the ProcessPoolExecutor
        """
        futures = []
        num_workers = self._get_num_workers()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for job_id in list(self.valid_job_ids):
                func, args = submit_func(job_id)
                future = executor.submit(func, *args)
                futures.append((job_id, future))

        for job_id, future in tqdm(futures, total=len(futures), desc=work_desc):
            try:
                future_result = future.result()
                process_result_func(job_id, future_result)
            except Exception as e:
                error_string = capture_error_message(e)
                self.results[job_id].add_error(error_string)
                self.valid_job_ids.remove(job_id)

    def _parallel_preprocessing(self):
        def submit_func(job_id):
            job = self.jobs[job_id]
            return (job.preprocessing, (job.input_tensors, job.kernel_kwargs))

        def process_result_func(job_id, preprocessing_is_ok):
            result = self.results[job_id]
            result.add_fields(preprocessing_result=preprocessing_is_ok)

        self._parallel_execute("Preprocessing", submit_func, process_result_func)

    def _parallel_compile_to_neff(self):
        def submit_func(job_id):
            job = self.jobs[job_id]
            return (
                compile_kernel,
                (job.kernel, job.name, job.input_tensors, job.kernel_kwargs, job.compiler_flags, self.cache_dir),
            )

        def process_result_func(job_id, neff):
            result = self.results[job_id]
            result.add_fields(neff=neff)

        self._parallel_execute("Compiling kernels", submit_func, process_result_func)

    def _execute(self):
        with SpikeExecutor(verbose=0) as spike:
            for job_id in tqdm(self.valid_job_ids, total=len(self.valid_job_ids), desc="Executing kernels"):
                job = self.jobs[job_id]
                result = self.results[job_id]
                try:
                    spike_kernel = create_spike_kernel(result.neff, job.kernel, job.input_tensors, job.kernel_kwargs)
                    stats = spike.benchmark(
                        spike_kernel,
                        *job.input_tensors,
                        **job.kernel_kwargs,
                        warmup_iterations=self.warmup,
                        benchmark_iterations=self.iters,
                        device_id=0,
                    )
                    ntff_file, kernel_output = run_spike_kernel(
                        spike, spike_kernel, job.input_tensors, result.neff, job.kernel_kwargs
                    )
                    job.add_fields(spike_kernel=spike_kernel)
                    result.add_fields(ntff=ntff_file, **stats)
                    self.kernel_outputs[job_id] = kernel_output
                except Exception as e:
                    error_string = capture_error_message(e)
                    result.add_error(error_string)
                    self.valid_job_ids.remove(job_id)

    def _parallel_postprocessing(self):
        def submit_func(job_id):
            job = self.jobs[job_id]
            kernel_output = self.kernel_outputs[job_id]
            result = self.results[job_id]
            return (job.postprocessing, (job.input_tensors, job.kernel_kwargs, kernel_output, result.metrics))

        def process_result_func(job_id, postprocessing_is_ok):
            result = self.results[job_id]
            result.add_fields(postprocessing_result=postprocessing_is_ok)

        self._parallel_execute("Postprocessing", submit_func, process_result_func)

    def _parallel_extract_metrics(self):
        """Extract profile metrics for all jobs in parallel."""
        futures = []
        num_workers = self._get_num_workers()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for job_id in self.valid_job_ids:
                job = self.jobs[job_id]
                result = self.results[job_id]
                future = executor.submit(
                    extract_metrics, result.neff, result.ntff, result.min_ms, job.spike_kernel.traced_kernel
                )
                futures.append((job_id, future))

        for job_id, future in tqdm(futures, desc="Extracting metrics"):
            job = self.jobs[job_id]
            result = self.results[job_id]
            try:
                metrics = future.result()
                result.add_fields(metrics=metrics)
            except Exception as e:
                error_msg = capture_error_message(e)
                result.add_error(error_msg)
                result.add_fields(hfu=0)
                self.valid_job_ids.remove(job_id)
