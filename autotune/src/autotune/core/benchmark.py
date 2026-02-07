# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

from tqdm import tqdm

from autotune.core.job import ProfileJobs, compile_jobs
from autotune.core.parallel import set_neuron_core, split_jobs_into_groups
from autotune.core.run_nki import run_on_neuron_core

if TYPE_CHECKING:
    from autotune.core.results import BenchmarkResults


class Benchmark:
    """Benchmarks NKI kernels by compiling and running them on Neuron devices.

    Handles parallel compilation and execution of kernel jobs with performance profiling.
    """

    def __init__(self, jobs: ProfileJobs, warmup: int, iters: int) -> None:
        """Initialize benchmark configuration.

        Args:
            jobs: Collection of kernel jobs to benchmark.
            warmup: Number of warmup iterations before timing.
            iters: Number of iterations for performance measurement.
        """
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters

    def run(self) -> BenchmarkResults:
        """Execute the full benchmarking pipeline and return results.

        Compiles all kernels in parallel on CPU, then runs them on Neuron
        cores for profiling. Results are saved to JSON and returned as a
        queryable BenchmarkResults object.

        Returns:
            BenchmarkResults loaded from the cache directory.
        """
        from autotune.core.results import BenchmarkResults

        self._compile_all_kernels()
        self.jobs.dump_json()
        self._run_on_neuron_cores()
        self.jobs.dump_json()
        return BenchmarkResults.load(self.jobs.cache_root_dir)

    def _compile_all_kernels(self) -> None:
        """Compile all kernel jobs in parallel using multiple CPU workers."""
        cpu_count = os.cpu_count() or 1
        num_workers = min(max(cpu_count - 1, 1), len(self.jobs.jobs))
        num_jobs = len(self.jobs.jobs)
        job_id_groups = split_jobs_into_groups(job_ids=list(range(num_jobs)), num_groups=num_workers)

        pbar = tqdm(total=num_jobs, desc=f"Compiling {num_jobs} kernels on {num_workers} CPUs", unit="kernels")
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = {}
        for rank, rank_job_ids in enumerate(job_id_groups):
            rank_jobs = self.jobs.subset(rank_job_ids)
            future = executor.submit(compile_jobs, rank_jobs)
            futures[future] = (rank, rank_job_ids)
        for future in as_completed(futures):
            rank, rank_job_ids = futures[future]
            updated_jobs: ProfileJobs = future.result()
            for job_index in updated_jobs.jobs:
                updated_job = updated_jobs.jobs[job_index]
                self.jobs.jobs[job_index] = updated_job
            pbar.update(len(updated_jobs.jobs))

        pbar.close()
        executor.shutdown(wait=True)

    def _run_on_neuron_cores(self) -> None:
        """Execute compiled kernels on available Neuron cores for performance profiling."""
        valid_job_indices = [job_id for job_id in self.jobs.jobs if not self.jobs.jobs[job_id].has_error]
        num_neuron_cores = 128
        num_workers = min(num_neuron_cores, len(valid_job_indices))
        job_id_groups = split_jobs_into_groups(job_ids=valid_job_indices, num_groups=num_workers)

        pbar = tqdm(
            total=len(valid_job_indices),
            desc=f"Running {len(valid_job_indices)} kernels on {num_workers} Neuron cores",
            unit="kernels",
        )
        executors = []
        futures = {}
        for rank in range(num_workers):
            rank_job_ids = job_id_groups[rank]
            executor = ProcessPoolExecutor(max_workers=1, initializer=set_neuron_core, initargs=(rank,))
            executors.append(executor)
            kwargs = {"warmup": self.warmup, "iters": self.iters, "jobs": self.jobs.subset(rank_job_ids)}
            future = executor.submit(run_on_neuron_core, **kwargs)
            futures[future] = (rank, rank_job_ids)

        for future in as_completed(futures):
            neuron_core_id, rank_job_ids = futures[future]
            updated_jobs: ProfileJobs = future.result()
            for job_index in updated_jobs.jobs:
                updated_job = updated_jobs.jobs[job_index]
                self.jobs.jobs[job_index] = updated_job
            pbar.update(len(updated_jobs.jobs))

        pbar.close()
        for executor in executors:
            executor.shutdown(wait=True)
