# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from autotune.cache.results import ProfileResults
from autotune.core.compile import compile_kernel
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics, tensor_to_matmul_mac_count
from autotune.core.parallel import get_function_name, parallel_execute, parallel_execute_groups
from autotune.core.processing import postprocessing_fun_wrapper
from autotune.core.run_nki import run_on_neuron_core


class Benchmark:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(self, jobs: ProfileJobs, cache_root_dir: str, warmup: int = 10, iters: int = 100):
        self.jobs = jobs
        self.cache_root_dir = cache_root_dir
        self.warmup = warmup
        self.iters = iters

    def __call__(self):
        # FIXME: overlap compilation and execution
        self.results = self._init_results()
        self.results.dump_summary()
        # self._parallel_compile_to_neff()
        # self.results.dump_summary()
        # self._parallel_run_kernels()
        # self._parallel_extract_metrics()
        # self._parallel_postprocessing()
        # self.results.dump_summary()

    def _init_results(self) -> ProfileResults:
        results = ProfileResults(sort_key="min_ms", lower_is_better=True)
        for job in self.jobs:
            job.cache_root_dir = self.cache_root_dir
            job.kernel = get_function_name(job.kernel)
            matmul_mac_count = tensor_to_matmul_mac_count(job.input_tensor_shapes)
            results.add_result(
                kernel=job.kernel,
                kernel_kwargs=job.kernel_kwargs,
                compiler_flags=job.compiler_flags,
                cache_dir=job.cache_dir,
                matmul_mac_count=matmul_mac_count,
            )
            job.init_job_dir()
        return results

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
                        "target_instance_family": job.target_instance_family,
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

        # FIXME: check if result has error field
        for job_id in self.valid_job_ids:
            result = self.results[job_id]
            if "error" in result.attributes:
                error_msg = result.error
                self._process_error(job_id, error_msg)

    def _parallel_postprocessing(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            funcs = []
            kwargs = []
            for job_id in job_group:
                job = self.jobs[job_id]
                result = self.results[job_id]
                funcs.append(postprocessing_fun_wrapper)
                kwargs.append(
                    {
                        "processing_fun": job.postprocessing,
                        "input_tensors": job.input_tensors,
                        "kernel_kwargs": job.kernel_kwargs,
                        "cache_dir": job.cache_dir,
                    }
                )
            return funcs, kwargs

        def process_results(error: bool, job_id: int, metrics: str | None):
            if isinstance(metrics, str):
                assert (
                    error
                ), f"Expecting error=True when metrics is an error string. Received error {error}, metrics {metrics}."
                self._process_error(job_id, metrics)
            else:
                self.results[job_id].add_fields(postprocessing_result=True)

        parallel_execute(
            executor_type="process",
            num_workers=self._get_num_workers(),
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Postprocessing",
            process_results_func=process_results,
        )

    def _parallel_extract_metrics(self):
        def submit_jobs(job_group_id: int, job_group: List[int]):
            funcs = []
            kwargs = []
            for job_id in job_group:
                job = self.jobs[job_id]
                result = self.results[job_id]
                funcs.append(extract_metrics)
                kwargs.append(
                    {
                        "neff": result.neff,
                        "ntff": result.ntff,
                        "latency": result.min_ms,
                        "matmul_mac_count": result.matmul_mac_count,
                        "target_instance_family": job.target_instance_family,
                    }
                )
            return funcs, kwargs

        def process_results(error: bool, job_id: int, metrics: str | Dict[str, float]):
            if isinstance(metrics, str):
                assert (
                    error
                ), f"Expecting error=True when metrics is an error string. Received error {error}, metrics {metrics}."
                self._process_error(job_id, metrics)
            else:
                self.results[job_id].add_fields(**metrics)

        parallel_execute(
            executor_type="thread",
            num_workers=self._get_num_workers(),
            job_ids=self.valid_job_ids,
            submit_jobs_func=submit_jobs,
            work_desc="Extracting Metrics",
            process_results_func=process_results,
        )

    def _process_error(self, job_id: int, error_msg: str):
        result = self.results[job_id]
        result.add_error(error_msg)
        if job_id in self.valid_job_ids:
            self.valid_job_ids.remove(job_id)
