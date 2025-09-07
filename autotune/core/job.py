import copy
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np

from autotune.core.compile import process_compiler_flags
from autotune.typing import (
    INPUT_TENSORS_DTYPE,
    KERNEL_DTYPE,
    KERNEL_KWARGS_DTYPE,
    OUTPUT_TENSORS_DTYPE,
    POSTPROCESSING_DTYPE,
)


def dummy_postprocessing(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_outputs: OUTPUT_TENSORS_DTYPE
) -> None:
    pass


def run_with_args_and_kwargs(func, args, kwargs):
    return func(args, kwargs)


def get_batch_size(num_samples: int, total_num_samples: int):
    batch_size = max(num_samples, 1000)
    batch_size = min(batch_size, total_num_samples)
    return batch_size


class ProfileJob:
    def __init__(
        self,
        index: int,
        kernel: KERNEL_DTYPE,
        input_tensor_shapes: List[Tuple[int, ...]],
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        postprocessing: POSTPROCESSING_DTYPE,
        data_type: np.dtype,
    ) -> None:
        self.index = index
        self.kernel = kernel
        self.input_tensor_shapes = input_tensor_shapes
        self.kernel_kwargs = kernel_kwargs
        self.postprocessing = postprocessing
        self.data_type = data_type
        self.target_instance_family, self.compiler_flags = process_compiler_flags(compiler_flags)
        self._input_tensors = None  # Cache for generated tensors

    @property
    def input_tensors(self) -> Tuple[np.ndarray, ...]:
        """Return the cached input tensors."""
        if self._input_tensors is None:
            raise ValueError(f"Input tensors not initialized for job {self.index}")
        return self._input_tensors

    @input_tensors.setter
    def input_tensors(self, value: Tuple[np.ndarray, ...]):
        """Set the input tensors for this job."""
        self._input_tensors = value

    @property
    def cache_dir(self) -> str:
        input_tensor_shapes_str = "_".join("x".join(str(dim) for dim in shape) for shape in self.input_tensor_shapes)
        _, kernel_name = self.kernel
        cache_dir = f"{self.cache_root_dir}/{kernel_name}/{input_tensor_shapes_str}/id{self.index}"
        return cache_dir

    @property
    def cache_root_dir(self) -> str:
        """
        Get the root directory for caching.

        Returns:
            str: The root directory path
        """
        return self._cache_root_dir

    @cache_root_dir.setter
    def cache_root_dir(self, value: str) -> None:
        """
        Set the root directory for caching.

        Args:
            value: Path to the root directory
        """
        self._cache_root_dir = value

    def get_arg_shapes(self):
        arg_shapes = [arg.shape for arg in self.input_tensors]
        return arg_shapes

    def init_job_dir(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def save(self):
        """
        Save the ProfileJob instance to its cache directory.
        """
        filepath = os.path.join(self.cache_dir, "profile_job.pkl")

        state = {}
        for key in ["index", "kernel", "input_tensors", "kernel_kwargs", "compiler_flags", "cache_dir"]:
            value = getattr(self, key)
            state[key] = value

        # Save using pickle
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> Dict:
        """
        Load a ProfileJob instance from disk.

        Args:
            filepath: Path to the saved file

        Returns:
            ProfileJob: The loaded ProfileJob instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        filepath = os.path.join(filepath, "profile_job.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ProfileJob file not found: {filepath}")

        # Load the state
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        return state

    def __repr__(self) -> str:
        arg_shapes = [str(arg.shape) for arg in self.input_tensors]
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kernel_kwargs.items())

        return f"ProfileJob(kernel={self.kernel}, input_tensor_shapes={arg_shapes}, " f"kwargs={{{kwargs_str}}})"


class ProfileJobs:
    def __init__(self) -> None:
        self.jobs: List[ProfileJob] = []
        self._tensor_cache: Dict[Tuple, Tuple[np.ndarray, ...]] = {}

    def add_job(
        self,
        kernel: KERNEL_DTYPE,
        input_tensor_shapes: List[Tuple[int, ...]],
        data_type: np.dtype,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        postprocessing: Optional[POSTPROCESSING_DTYPE] = None,
    ):
        if kernel_kwargs is None:
            kernel_kwargs = {}
        if compiler_flags is None:
            compiler_flags = ""
        if postprocessing is None:
            postprocessing = dummy_postprocessing
        job = ProfileJob(
            self.num_jobs, kernel, input_tensor_shapes, kernel_kwargs, compiler_flags, postprocessing, data_type
        )
        self.jobs.append(job)

    def extend(self, other_jobs: "ProfileJobs") -> "ProfileJobs":
        """
        Concatenate another ProfileJobs object to this one.
        This modifies the current object and also returns it for chaining.

        Args:
            other_jobs: Another ProfileJobs instance to append

        Returns:
            self: The modified ProfileJobs instance
        """

        # Get the current number of jobs as the starting index
        start_idx = self.num_jobs

        # Deep copy each job and update only the index
        for i, original_job in enumerate(other_jobs.jobs):
            # Create a deep copy of the original job
            job = copy.deepcopy(original_job)

            # Update only the index
            job.index = start_idx + i

            # Add to our jobs list
            self.jobs.append(job)

        return self

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

    def __iter__(self):
        """Allow iteration over ProfileJobs."""
        return iter(self.jobs)

    def subset(self, key):
        """
        Create a subset of ProfileJobs using a slice or list of indices.

        To access a single job, use ProfileJobs.jobs[index] directly.

        Args:
            key: A slice or list of indices to select jobs

        Returns:
            ProfileJobs: A new ProfileJobs instance containing the subset

        Raises:
            TypeError: If key is not a slice or list
        """
        if isinstance(key, (slice, list)):
            subset_jobs = ProfileJobs()
            if isinstance(key, slice):
                subset_jobs.jobs = self.jobs[key]
            else:  # list of indices
                subset_jobs.jobs = [self.jobs[i] for i in key]
            # Share the tensor cache with the subset
            subset_jobs._tensor_cache = self._tensor_cache
            return subset_jobs
        else:
            raise TypeError(
                f"subset() only accepts slice or list, got {type(key).__name__}. "
                f"To access a single job, use ProfileJobs.jobs[index] directly."
            )

    def initialize_input_tensors(self) -> None:
        for job in self.jobs:
            cache_key = (tuple(job.input_tensor_shapes), job.data_type)
            if cache_key not in self._tensor_cache:
                self._tensor_cache[cache_key] = tuple(
                    np.random.normal(0, 0.001, size=shape).astype(job.data_type) for shape in job.input_tensor_shapes
                )
            job.input_tensors = self._tensor_cache[cache_key]
