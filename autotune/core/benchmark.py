# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from tqdm import tqdm

from autotune.cache.results import ProfileResults
from autotune.core.job import ProfileJobs
from autotune.core.metrics import extract_metrics, tensor_to_matmul_mac_count
from autotune.core.parallel import parallel_execute, set_neuron_core, split_jobs_into_groups
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
        self._run_on_neuron_cores()
        # self._parallel_extract_metrics()
        # self._parallel_postprocessing()
        self.results.dump_summary()

    def _init_results(self) -> ProfileResults:
        results = ProfileResults(sort_key="min_ms", lower_is_better=True)
        for job in self.jobs:
            job.cache_root_dir = self.cache_root_dir
            matmul_mac_count = tensor_to_matmul_mac_count(job.input_tensor_shapes)
            results.add_result(
                index=job.index,
                kernel=job.kernel,
                kernel_kwargs=job.kernel_kwargs,
                compiler_flags=job.compiler_flags,
                cache_dir=job.cache_dir,
                matmul_mac_count=matmul_mac_count,
            )
            job.init_job_dir()
        return results

    def _run_on_neuron_cores(self):
        """Main function to launch 32 worker subprocesses."""
        num_neuron_cores = 16
        job_id_groups = split_jobs_into_groups(job_ids=list(range(self.jobs.num_jobs)), num_groups=num_neuron_cores)
        executors = []
        futures = {}
        for neuron_core_id in range(num_neuron_cores):
            rank_job_ids = job_id_groups[neuron_core_id]
            executor = ProcessPoolExecutor(max_workers=1, initializer=set_neuron_core, initargs=(neuron_core_id,))
            executors.append(executor)
            kwargs = {
                "warmup": self.warmup,
                "iters": self.iters,
                "jobs": [self.jobs[job_id] for job_id in rank_job_ids],
                "results": [self.results[job_id] for job_id in rank_job_ids],
            }
            future = executor.submit(run_on_neuron_core, **kwargs)
            futures[future] = neuron_core_id

        with tqdm(
            total=self.jobs.num_jobs,
            desc=f"Running {self.jobs.num_jobs} kernels on {num_neuron_cores} Neuron cores",
            unit="kernels",
        ) as pbar:
            for future in as_completed(futures):
                neuron_core_id = futures[future]
                updated_results = future.result()
                for updated_result in updated_results:
                    job_id = updated_result.index
                    self.results[job_id] = updated_result
                pbar.update(len(updated_results))
        for executor in executors:
            executor.shutdown(wait=True)

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
