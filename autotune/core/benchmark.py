# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from autotune.core.job import ProfileJobs, compile_jobs
from autotune.core.parallel import set_neuron_core, split_jobs_into_groups
from autotune.core.run_nki import run_on_neuron_core


class Benchmark:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(self, jobs: ProfileJobs, warmup: int = 10, iters: int = 100):
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters

    def __call__(self):
        self._compile_all_kernels()
        self.jobs.dump_json()
        self._run_on_neuron_cores()
        self.jobs.dump_json()

    def _compile_all_kernels(self):
        """Main function to launch Neuron core worker subprocesses."""
        num_workers = min(os.cpu_count() - 1, len(self.jobs.jobs))
        num_jobs = len(self.jobs.jobs)
        job_id_groups = split_jobs_into_groups(job_ids=list(range(num_jobs)), num_groups=num_workers)
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for rank, rank_job_ids in enumerate(job_id_groups):
                rank_jobs = self.jobs.subset(rank_job_ids)
                future = executor.submit(compile_jobs, rank_jobs)
                futures[future] = (rank, rank_job_ids)

        with tqdm(total=num_jobs, desc=f"Compiling {num_jobs} kernels on {num_workers} CPUs", unit="kernels") as pbar:
            for future in as_completed(futures):
                rank, rank_job_ids = futures[future]
                updated_jobs: ProfileJobs = future.result()
                for job_index in updated_jobs.jobs:
                    updated_job = updated_jobs.jobs[job_index]
                    self.jobs.jobs[job_index] = updated_job
                pbar.update(len(updated_jobs.jobs))

    def _run_on_neuron_cores(self):
        """Main function to launch Neuron core worker subprocesses."""
        valid_job_indices = []
        for job_id in self.jobs.jobs:
            job = self.jobs.jobs[job_id]
            if not job.has_error:
                valid_job_indices.append(job_id)
        num_neuron_cores = 32
        num_workers = min(num_neuron_cores, len(valid_job_indices))
        job_id_groups = split_jobs_into_groups(job_ids=valid_job_indices, num_groups=num_workers)
        executors = []
        futures = {}
        for rank in range(num_workers):
            rank_job_ids = job_id_groups[rank]
            executor = ProcessPoolExecutor(max_workers=1, initializer=set_neuron_core, initargs=(rank,))
            executors.append(executor)
            kwargs = {"warmup": self.warmup, "iters": self.iters, "jobs": self.jobs.subset(rank_job_ids)}
            future = executor.submit(run_on_neuron_core, **kwargs)
            futures[future] = (rank, rank_job_ids)

        with tqdm(
            total=len(valid_job_indices),
            desc=f"Running {len(valid_job_indices)} kernels on {num_workers} Neuron cores",
            unit="kernels",
        ) as pbar:
            for future in as_completed(futures):
                neuron_core_id, rank_job_ids = futures[future]
                updated_jobs: ProfileJobs = future.result()
                for job_index in updated_jobs.jobs:
                    updated_job = updated_jobs.jobs[job_index]
                    self.jobs.jobs[job_index] = updated_job
                pbar.update(len(updated_jobs.jobs))
        for executor in executors:
            executor.shutdown(wait=True)
