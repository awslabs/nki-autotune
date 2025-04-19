import random
from typing import Callable, List, Tuple

import numpy as np

from autotune.cache.directories import get_hash_name


def dummy_pruning(*args, **kwargs):
    return True


class ProfileJobs:
    def __init__(self) -> None:
        self.valid_jobs: List[ProfileJob] = []  # Valid jobs
        self.invalid_jobs: List[ProfileJob] = []  # Jobs with errors
        self.all_jobs: List[ProfileJob] = []  # All jobs in original insertion order

    def add_job(self, kernel, kernel_args: Tuple[np.ndarray, ...], *, pruning_func: Callable = dummy_pruning, **kwargs):
        job = ProfileJob(kernel, kernel_args, pruning_func, **kwargs)
        try:
            job.prune()
            self.valid_jobs.append(job)  # Add to valid jobs if pruning succeeds
        except Exception as e:
            job.add_fields(error=e)
            self.invalid_jobs.append(job)  # Add to invalid jobs if there's an error

        self.all_jobs.append(job)  # Always add to all_jobs to maintain original order

    def sample(self, num_samples: int):
        """Sample only from valid jobs."""
        sampled_jobs = ProfileJobs()
        num_samples = min(num_samples, len(self.valid_jobs))
        if num_samples > 0:
            sampled_jobs.valid_jobs = random.sample(self.valid_jobs, num_samples)
        return sampled_jobs

    @property
    def has_valid_jobs(self) -> bool:
        """Check if there are any valid jobs."""
        return len(self.valid_jobs) > 0

    def __repr__(self) -> str:
        """Return a string representation of ProfileJobs, showing both valid and invalid jobs."""
        result = []

        # Handle valid jobs
        if not self.valid_jobs:
            result.append("Valid jobs: None")
        else:
            if len(self.valid_jobs) <= 2:
                # For small collections, show all jobs
                jobs_str = ",\n  ".join(str(job) for job in self.valid_jobs)
                result.append(f"Valid jobs ({len(self.valid_jobs)}):\n  {jobs_str}")
            else:
                # For larger collections, show first and last job with count
                result.append(
                    f"Valid jobs ({len(self.valid_jobs)}):\n  {self.valid_jobs[0]},\n  ...({len(self.valid_jobs) - 2} more jobs)...,\n  {self.valid_jobs[-1]}"
                )

        # Handle invalid jobs
        if not self.invalid_jobs:
            result.append("Invalid jobs: None")
        else:
            if len(self.invalid_jobs) <= 2:
                # For small collections, show all invalid jobs
                jobs_str = ",\n  ".join(str(job) for job in self.invalid_jobs)
                result.append(f"Invalid jobs ({len(self.invalid_jobs)}):\n  {jobs_str}")
            else:
                # For larger collections, show first and last job with count
                result.append(
                    f"Invalid jobs ({len(self.invalid_jobs)}):\n  {self.invalid_jobs[0]},\n  ...({len(self.invalid_jobs) - 2} more jobs)...,\n  {self.invalid_jobs[-1]}"
                )

        return "ProfileJobs(\n" + "\n\n".join(result) + "\n)"

    def __getitem__(self, index):
        """Allow indexing to access jobs in the original order they were added."""
        return self.all_jobs[index]


class ProfileJob:
    def __init__(self, kernel, kernel_args: Tuple[np.ndarray, ...], pruning_func: Callable, **kwargs) -> None:
        self.kernel = kernel
        self.kernel_args: Tuple[np.ndarray, ...] = kernel_args
        self.pruning_func = pruning_func
        self.kwargs = kwargs
        self.name = get_hash_name(kernel, kernel_args, kwargs)

    def prune(self):
        arg_shapes = [arg.shape for arg in self.kernel_args]
        self.pruning_func(*arg_shapes, **self.kwargs)

    def __repr__(self) -> str:
        arg_shapes = [str(arg.shape) for arg in self.kernel_args]
        kernel_name = getattr(self.kernel, "func_name", str(self.kernel))
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())

        repr_str = f"ProfileJob({type(self.kernel)} kernel={kernel_name}, shapes={arg_shapes}, kwargs={{{kwargs_str}}}, name={self.name}"

        # Add error information if available
        if hasattr(self, "error") and self.error is not None:
            repr_str += f", error={type(self.error).__name__}: {str(self.error)}"

        repr_str += ")"
        return repr_str

    def __getattr__(self, name):
        """
        Called when an attribute lookup fails.
        Returns None for non-existent attributes instead of raising AttributeError.
        """
        return None

    def add_fields(self, **kwargs):
        """
        Add additional fields to this ProfileJob instance.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
