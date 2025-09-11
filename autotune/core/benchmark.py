# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from autotune.cache.results import ProfileResults
from autotune.core.job import ProfileJobs
from autotune.core.parallel import set_neuron_core, split_jobs_into_groups
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
        self.results = ProfileResults(sort_key="min_ms", lower_is_better=True)
        self._run_on_neuron_cores()
        self.results.dump_summary()

    def _run_on_neuron_cores(self):
        """Main function to launch Neuron core worker subprocesses."""
        num_neuron_cores = min(32, self.jobs.num_jobs)
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
                "jobs": self.jobs.subset(rank_job_ids),
                "cache_root_dir": self.cache_root_dir,
                "sort_key": self.results.sort_key,
                "lower_is_better": self.results.lower_is_better,
            }
            future = executor.submit(run_on_neuron_core, **kwargs)
            futures[future] = (neuron_core_id, rank_job_ids)

        with tqdm(
            total=self.jobs.num_jobs,
            desc=f"Running {self.jobs.num_jobs} kernels on {num_neuron_cores} Neuron cores",
            unit="kernels",
        ) as pbar:
            for future in as_completed(futures):
                neuron_core_id, rank_job_ids = futures[future]
                updated_results = future.result()
                for updated_result in updated_results:
                    self.results.results.append(updated_result)
                pbar.update(len(updated_results))
        for executor in executors:
            executor.shutdown(wait=True)
