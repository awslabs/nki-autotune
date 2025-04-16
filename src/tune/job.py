import random
from typing import Callable, List, Tuple

import numpy as np
from neuronxcc.nki.compile import GenericKernel

from src.cache.directories import get_hash_name


def dummy_pruning(*args, **kwargs):
    return True


class ProfileJobs:
    def __init__(self) -> None:
        self.jobs: List[ProfileJob] = []

    def add_job(
        self,
        kernel: GenericKernel,
        kernel_args: Tuple[np.ndarray, ...],
        pruning_func: Callable = dummy_pruning,
        **kwargs,
    ):
        try:
            job = ProfileJob(kernel, kernel_args, pruning_func, **kwargs)
            job.prune()
            self.jobs.append(job)
        except Exception as e:
            job.add_fields(error=e)

    def sample(self, num_samples: int):
        sampled_jobs = ProfileJobs()
        num_samples = min(num_samples, len(self.jobs))
        selected_jobs = random.sample(self.jobs, num_samples)
        sampled_jobs.jobs = selected_jobs
        return sampled_jobs

    def __len__(self) -> int:
        """Return the number of jobs."""
        return len(self.jobs)

    def __repr__(self) -> str:
        """Return a string representation of ProfileJobs."""
        if not self.jobs:
            return "ProfileJobs(empty)"

        if len(self.jobs) <= 3:
            # For small collections, show all jobs
            jobs_str = ",\n  ".join(str(job) for job in self.jobs)
            return f"ProfileJobs(\n  {jobs_str}\n)"
        else:
            # For larger collections, show first 2 and last job with count
            jobs_str = ",\n  ".join(str(job) for job in self.jobs[:2])
            return f"ProfileJobs(\n  {jobs_str},\n  ...({len(self.jobs) - 3} more jobs)...,\n  {self.jobs[-1]}\n)"

    def __iter__(self):
        """Make ProfileJobs iterable."""
        return iter(self.jobs)


class ProfileJob:
    def __init__(
        self, kernel: GenericKernel, kernel_args: Tuple[np.ndarray, ...], pruning_func: Callable, **kwargs
    ) -> None:
        self.kernel: GenericKernel = kernel
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
        pruning_name = self.pruning_func.__name__
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"ProfileJob(kernel={kernel_name}, shapes={arg_shapes}, pruning={pruning_name}, kwargs={{{kwargs_str}}}, name={self.name})"

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
