import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Tuple

import numpy as np
from tqdm import tqdm

from autotune.cache.directories import get_hash_name
from autotune.tune.utils import capture_error_message


def dummy_pruning(*args, **kwargs):
    return True


def run_with_args_and_kwargs(func, args, kwargs):
    return func(*args, **kwargs)


class ProfileJob:
    # TODO: preprocessing, postprocessing components
    def __init__(
        self, kernel_name: str, kernel_args: Tuple[np.ndarray, ...], preprocessing: Callable, **kwargs
    ) -> None:
        self.kernel_name = kernel_name
        self.kernel_args: Tuple[np.ndarray, ...] = kernel_args
        self.preprocessing = preprocessing
        self.kwargs = kwargs
        self.name = get_hash_name(kernel_name, kernel_args, kwargs)

    def get_arg_shapes(self):
        arg_shapes = [arg.shape for arg in self.kernel_args]
        return arg_shapes

    def __repr__(self) -> str:
        arg_shapes = [str(arg.shape) for arg in self.kernel_args]
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())

        return (
            f"ProfileJob(kernel={self.kernel_name}, shapes={arg_shapes}, " f"kwargs={{{kwargs_str}}}, name={self.name})"
        )

    def __getattr__(self, name):
        """
        Called when an attribute lookup fails.
        Returns None for non-existent attributes instead of raising AttributeError.
        """
        return None


class ProfileJobs:
    def __init__(self) -> None:
        self.jobs: List[ProfileJob] = []

    def add_job(
        self,
        kernel_name: str,
        kernel_args: Tuple[np.ndarray, ...],
        *,
        preprocessing: Callable = dummy_pruning,
        **kwargs,
    ):
        job = ProfileJob(kernel_name, kernel_args, preprocessing, **kwargs)
        self.jobs.append(job)

    def _parallel_filter(self) -> List[ProfileJob]:
        futures = []
        num_workers = min(len(self.jobs), os.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for job_id in range(self.num_jobs):
                job = self.jobs[job_id]
                future = executor.submit(run_with_args_and_kwargs, job.preprocessing, job.get_arg_shapes(), job.kwargs)
                futures.append((job_id, future))

        valid_jobs: List[ProfileJob] = []
        for job_id, future in tqdm(futures, total=len(futures), desc="Filtering jobs"):
            job = self.jobs[job_id]
            try:
                success = future.result()
                valid_jobs.append(job)
            except Exception as e:
                error_msg = capture_error_message(e)
        return valid_jobs

    def sample(self, num_samples: int) -> None:
        """Sample only from valid jobs."""
        valid_jobs = self._parallel_filter()
        num_samples = min(num_samples, len(valid_jobs))
        if num_samples > 0:
            sampled_jobs = random.sample(valid_jobs, num_samples)
        self.jobs = sampled_jobs

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    def __repr__(self) -> str:
        """Return a string representation of ProfileJobs."""
        if len(self.jobs) == 0:
            return "ProfileJobs(jobs: None)"

        if len(self.jobs) <= 3:
            # For small collections, show all jobs
            jobs_str = ",\n  ".join(str(job) for job in self.jobs)
            result = f"ProfileJobs({len(self.jobs)} jobs):\n  {jobs_str}"
        else:
            # For larger collections, show first and last jobs with count
            result = (
                f"ProfileJobs({len(self.jobs)} jobs):\n"
                f"  {self.jobs[0]},\n"
                f"  ...({len(self.jobs) - 2} more jobs)...,\n"
                f"  {self.jobs[-1]}"
            )

        return result

    def __getitem__(self, index):
        """Allow indexing to access jobs in the original order they were added."""
        return self.jobs[index]
