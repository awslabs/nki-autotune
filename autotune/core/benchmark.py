# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from autotune.cache.results import ProfileResults
from autotune.core.job import ProfileJobs
from autotune.core.metrics import tensor_to_matmul_mac_count
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
        self.results = self._init_results()
        self.results.dump_summary()
        self._run_on_neuron_cores()
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
