import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from autotune.cache.directories import get_hash_name
from autotune.tune.utils import capture_error_message


def dummy_pruning(*args, **kwargs):
    return True


def run_with_args_and_kwargs(func, args, kwargs):
    return func(*args, **kwargs)


def get_batch_size(num_samples: int, total_num_samples: int):
    batch_size = max(num_samples, 1000)
    batch_size = min(batch_size, total_num_samples)
    return batch_size


class ProfileJob:
    # TODO: filter, postprocessing components
    def __init__(self, kernel_name: str, kernel_args: Tuple[np.ndarray, ...], filter: Callable, **kwargs) -> None:
        self.kernel_name = kernel_name
        self.kernel_args: Tuple[np.ndarray, ...] = kernel_args
        self.filter = filter
        self.kwargs = kwargs
        self.name = get_hash_name(kernel_name, kernel_args, kwargs)

    def get_arg_shapes(self):
        arg_shapes = [arg.shape for arg in self.kernel_args]
        return arg_shapes

    def add_fields(self, **kwargs):
        """
        Add additional fields to this ProfileJob instance.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        self, kernel_name: str, kernel_args: Tuple[np.ndarray, ...], *, filter: Callable = dummy_pruning, **kwargs
    ):
        job = ProfileJob(kernel_name, kernel_args, filter, **kwargs)
        self.jobs.append(job)

    def sample(self, num_samples: int) -> None:
        """Sample only from valid jobs."""
        valid_jobs: List[ProfileJob] = []
        remaining_num_samples = num_samples

        # Keep track of jobs we've already processed
        processed_job_ids: Set[int] = set()
        available_job_ids = list(range(self.num_jobs))

        while remaining_num_samples > 0 and available_job_ids:
            # 1. Randomly sample remaining jobs
            batch_size = get_batch_size(remaining_num_samples, len(available_job_ids))
            sampled_job_ids = random.sample(available_job_ids, batch_size)

            # Remove sampled jobs from available pool
            for job_id in sampled_job_ids:
                available_job_ids.remove(job_id)

            # 2. Process the sampled batch in parallel
            futures = []
            num_workers = min(len(sampled_job_ids), os.cpu_count() - 1)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for job_id in sampled_job_ids:
                    job = self.jobs[job_id]
                    future = executor.submit(run_with_args_and_kwargs, job.filter, job.get_arg_shapes(), job.kwargs)
                    futures.append((job_id, future))

            # Process results of this batch
            batch_valid_jobs: List[ProfileJob] = []
            for job_id, future in tqdm(
                futures, total=len(futures), desc=f"Sampling valid jobs (need {remaining_num_samples} more)"
            ):
                job = self.jobs[job_id]
                try:
                    success = future.result()
                    batch_valid_jobs.append(job)
                except Exception as e:
                    error_msg = capture_error_message(e)

                processed_job_ids.add(job_id)

            # 3. Update remaining count
            valid_jobs.extend(batch_valid_jobs[:remaining_num_samples])
            remaining_num_samples = num_samples - len(valid_jobs)
        self.jobs = valid_jobs

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
