import random
from typing import Callable, List, Tuple

import numpy as np

from autotune.cache.directories import get_hash_name


def dummy_pruning(*args, **kwargs):
    return True


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

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

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
        return self.jobs[index]


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

        repr_str = (
            f"ProfileJob(kernel={self.kernel_name}, shapes={arg_shapes}, kwargs={{{kwargs_str}}}, name={self.name}"
        )

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
