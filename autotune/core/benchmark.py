# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from autotune.core.job import ProfileJobs, compile_jobs
from autotune.core.parallel import set_neuron_core, set_neuron_core_dynamic, split_jobs_into_groups
from autotune.core.run_nki import run_on_neuron_core, run_on_neuron_core_dynamic


class Benchmark:
    """Benchmarks NKI kernels by compiling and running them on Neuron devices.

    Handles parallel compilation and execution of kernel jobs with performance profiling.
    """

    def __init__(self, jobs: ProfileJobs, warmup: int = 10, iters: int = 100):
        """Initialize benchmark configuration.

        Args:
            jobs: Collection of kernel jobs to benchmark.
            warmup: Number of warmup iterations before timing.
            iters: Number of iterations for performance measurement.
        """
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters

    def __call__(self):
        """Execute the full benchmarking pipeline: compile, then run on device."""
        self._compile_and_run_dynamic_load_balancing()
        self.jobs.dump_json()

    def _compile_and_run_dynamic_load_balancing(self):
        # --- Compile Setup --- #
        cpu_count = os.cpu_count() or 1
        num_cpu_workers = min(max(cpu_count - 1, 1), len(self.jobs.jobs))
        num_jobs = len(self.jobs.jobs)
        job_id_groups = split_jobs_into_groups(job_ids=list(range(num_jobs)), num_groups=num_cpu_workers)
        compiler_executor = ProcessPoolExecutor(max_workers=num_cpu_workers)
        c_pbar = tqdm(total=num_jobs, desc=f"Compiling {num_jobs} kernels on {num_cpu_workers} CPUs", unit="kernels")

        # --- Neuron Execute Setup --- #
        num_neuron_workers = 32
        counter = mp.Value("i", 0)
        lock = mp.Lock()
        runner_executor = ProcessPoolExecutor(
            max_workers=num_neuron_workers, initializer=set_neuron_core_dynamic, initargs=(counter, lock)
        )
        e_pbar = tqdm(
            total=num_jobs,
            desc=f"Processing/Running {num_jobs} kernels on {num_neuron_workers} Neuron cores",
            unit="kernels",
        )

        # --- Compile + Execute --- #
        pending_futures = set()
        compiled_futures = set()
        for rank_job_ids in job_id_groups:
            rank_jobs = self.jobs.subset(rank_job_ids)
            compiling_future = compiler_executor.submit(compile_jobs, rank_jobs)
            compiled_futures.add(compiling_future)
            pending_futures.add(compiling_future)

        while pending_futures:
            for future in as_completed(pending_futures):
                pending_futures.remove(future)
                updated_jobs = future.result()
                for job_index in updated_jobs.jobs:
                    updated_job = updated_jobs.jobs[job_index]
                    self.jobs.jobs[job_index] = updated_job
                    if future in compiled_futures:
                        if not updated_job.has_error:
                            kwargs = {"warmup": self.warmup, "iters": self.iters, "jobs": self.jobs.subset([job_index])}
                            executing_future = runner_executor.submit(run_on_neuron_core_dynamic, **kwargs)
                            pending_futures.add(executing_future)
                        else:
                            e_pbar.update(1)
                        c_pbar.update(len(updated_jobs.jobs))
                    else:
                        e_pbar.update(1)

        # --- Cleanup --- #
        c_pbar.close()
        e_pbar.close()
        compiler_executor.shutdown(wait=True)
        runner_executor.shutdown(wait=True)

    def _compile_all_kernels(self):
        """Compile all kernel jobs in parallel using multiple CPU workers."""
        cpu_count = os.cpu_count() or 1  # Handle None case
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

    def _run_on_neuron_cores(self):
        """Execute compiled kernels on available Neuron cores for performance profiling."""
        valid_job_indices = []
        for job_id in self.jobs.jobs:
            job = self.jobs.jobs[job_id]
            if not job.has_error:
                valid_job_indices.append(job_id)
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
