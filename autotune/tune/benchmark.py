# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from autotune.cache.results import ProfileResults
from autotune.tune.job import ProfileJobs
from autotune.tune.metrics import extract_metrics
from autotune.tune.parallel import parallel_execute, parallel_execute_groups
from autotune.tune.run_nki import run_on_neuron_core
from autotune.tune.utils import capture_error_message, compile_kernel


class Benchmark:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(self, jobs: ProfileJobs, warmup: int = 10, iters: int = 100):
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters

    def __call__(self):
        self.valid_job_ids = list(range(self.jobs.num_jobs))
        self.kernel_outputs: Dict[int, np.ndarray] = {}
        self.results = self._init_results()
        self._parallel_init_jobs()
        self._parallel_preprocessing()
        self._parallel_compile_to_neff()
        self._parallel_run_kernels()
        # self._parallel_extract_metrics()
        # self._parallel_postprocessing()
        self.results.dump_summary()

    def _get_num_workers(self) -> int:
        num_workers = min(len(self.valid_job_ids), os.cpu_count() - 1)
        num_workers = max(num_workers, 1)
        return num_workers

    def _init_results(self) -> ProfileResults:
        results = ProfileResults(sort_key="min_ms", lower_is_better=True)
        for job in self.jobs:
            results.add_result(
                kernel=job.kernel,
                kernel_kwargs=job.kernel_kwargs,
                compiler_flags=job.compiler_flags,
                cache_dir=job.cache_dir,
            )
        return results

    def _parallel_init_jobs(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            funcs = []
            kwargs = []
            for job_id in job_group:
                funcs.append(self.jobs[job_id].init_job_dir)
                kwargs.append({})
            return funcs, kwargs

        def process_results(error: bool, job_id: int, output: None | str):
            if error and output:
                self._process_error(job_id, output)

        parallel_execute(
            executor_type="thread",
            num_workers=self._get_num_workers(),
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Init Job Directories",
            process_results_func=process_results,
        )

    def _parallel_preprocessing(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            funcs = []
            kwargs = []
            for job_id in job_group:
                job = self.jobs[job_id]
                funcs.append(job.preprocessing)
                kwargs.append({"input_tensors": job.input_tensors, "kernel_kwargs": job.kernel_kwargs})
            return funcs, kwargs

        def process_results(error: bool, job_id: int, preprocessing_is_ok: None | str):
            if error and preprocessing_is_ok:
                self._process_error(job_id, preprocessing_is_ok)
            else:
                self.results[job_id].add_fields(preprocessing_result=True)

        parallel_execute(
            executor_type="process",
            num_workers=self._get_num_workers(),
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Preprocessing",
            process_results_func=process_results,
        )

    def _parallel_compile_to_neff(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            funcs = []
            kwargs = []
            for job_id in job_group:
                job = self.jobs[job_id]
                funcs.append(compile_kernel)
                kwargs.append(
                    {
                        "kernel_name": job.kernel,
                        "input_tensors": job.input_tensors,
                        "kernel_kwargs": job.kernel_kwargs,
                        "compiler_flags": job.compiler_flags,
                        "output_dir": job.cache_dir,
                    }
                )
            return funcs, kwargs

        def process_results(error: bool, job_id: int, neff: str):
            if error:
                self._process_error(job_id, neff)
            else:
                self.results[job_id].add_fields(neff=neff)

        parallel_execute(
            executor_type="process",
            num_workers=self._get_num_workers(),
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Compiling Kernels",
            process_results_func=process_results,
        )

    def _parallel_run_kernels(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            kwargs = {
                "neuron_core_id": job_group_id,
                "warmup": self.warmup,
                "iters": self.iters,
                "job_ids": [],
                "jobs": self.jobs,
                "results": self.results,
            }
            for job_id in job_group:
                kwargs["job_ids"].append(job_id)
            return run_on_neuron_core, kwargs

        def process_results(error: bool, job_group: List[int], error_msg: None | str):
            if error and error_msg:
                for job_id in job_group:
                    self._process_error(job_id, error_msg)

        parallel_execute_groups(
            executor_type="thread",
            num_workers=32,
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Run Kernels",
            process_results_func=process_results,
        )

    def _parallel_postprocessing(self):
        def submit_func(job_id):
            job = self.jobs[job_id]
            kernel_output = self.kernel_outputs[job_id]
            return (job.postprocessing, (job.input_tensors, job.kernel_kwargs, kernel_output))

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
                    extract_metrics, result.neff, result.ntff, result.min_ms, result.matmul_mac_count
                )
                futures.append((job_id, future))

        for job_id, future in tqdm(futures, desc="Extracting metrics"):
            job = self.jobs[job_id]
            result = self.results[job_id]
            try:
                metrics = future.result()
                result.add_fields(**metrics)
            except Exception as e:
                error_msg = capture_error_message(e)
                result.add_error(error_msg)
                self.valid_job_ids.remove(job_id)

    def _process_error(self, job_id: int, error_msg: str):
        result = self.results[job_id]
        result.add_error(error_msg)
        if job_id in self.valid_job_ids:
            self.valid_job_ids.remove(job_id)
