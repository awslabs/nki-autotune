# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np

from autotune.analysis.analyze import compute_mac_count
from autotune.compiler.compile import compile_kernel, resolve_kernel_ref
from autotune.types import (
    CORRECTNESS_CHECK_DTYPE,
    INPUT_TENSOR_SHAPES_DTYPE,
    INPUT_TENSORS_DTYPE,
    KERNEL_DTYPE,
    KERNEL_KWARGS_DTYPE,
)
from autotune.utils import capture_error_message


def workload_name(kernel_name: str, input_tensor_shapes: dict[str, tuple[int, ...]]) -> str:
    """Generate a canonical workload name from kernel and input shapes.

    Used for cache directory paths, JSON output, and visualization labels.

    Args:
        kernel_name: The kernel function name.
        input_tensor_shapes: Mapping of tensor names to shapes.

    Returns:
        Canonical string like 'nki_matmul_kernel/128x512_512x1024'.
    """
    shapes_str = "_".join("x".join(str(d) for d in shape) for shape in input_tensor_shapes.values())
    return f"{kernel_name}/{shapes_str}"


class ProfileJob:
    """Represents a single kernel profiling job with its configuration and performance metrics."""

    def __init__(
        self,
        index: int,
        main_metric: str,
        kernel: KERNEL_DTYPE,
        input_tensors: INPUT_TENSORS_DTYPE,
        input_tensor_shapes: INPUT_TENSOR_SHAPES_DTYPE,
        output_shapes: dict[str, tuple[int, ...]],
        mac_count: int,
        data_type: np.dtype,
        scalar_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        correctness_check: CORRECTNESS_CHECK_DTYPE,
        cache_root_dir: str,
    ) -> None:
        """Initialize a profiling job with kernel configuration and cache settings.

        Args:
            index: Unique identifier for this job.
            main_metric: Primary performance metric to optimize.
            kernel: Tuple of (filepath, function_name) identifying the kernel.
            input_tensors: Dictionary mapping tensor names to numpy arrays.
            input_tensor_shapes: Shapes of input tensors.
            output_shapes: Shapes of output tensors, mapping tensor name to shape tuple.
            mac_count: Number of MAC operations for performance calculation.
            data_type: Data type for input tensors.
            scalar_kwargs: Non-tensor keyword arguments for the kernel.
            compiler_flags: Compilation flags and target configuration.
            correctness_check: Tuple of (golden_fn, atol, rtol) for correctness
                verification, or None to skip.
            cache_root_dir: Root directory for caching results.
        """
        self.attributes: list[str] = []
        _, kernel_func_name = kernel
        wl_name = workload_name(kernel_func_name, input_tensor_shapes)
        wl_dir = f"{cache_root_dir}/{wl_name}"
        cache_dir = f"{wl_dir}/id{index}"
        self.add_attributes(
            index=index,
            kernel=kernel,
            input_tensor_shapes=input_tensor_shapes,
            output_shapes=output_shapes,
            data_type=data_type,
            scalar_kwargs=scalar_kwargs,
            compiler_flags=compiler_flags,
            cache_dir=cache_dir,
            mac_count=mac_count,
        )
        self._input_tensors = input_tensors
        self.correctness_check = correctness_check
        self.main_metric = main_metric
        self.workload_dir = wl_dir

    @property
    def input_tensors(self) -> INPUT_TENSORS_DTYPE:
        """Return the input tensors."""
        return self._input_tensors

    @input_tensors.setter
    def input_tensors(self, value: INPUT_TENSORS_DTYPE) -> None:
        """Set the input tensors for this job."""
        self._input_tensors = value

    def init_job_dir(self) -> None:
        """Initialize the cache directory for this job, removing any existing content."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation including only tracked attributes."""
        result = {}
        for attr_name in self.attributes:
            val = getattr(self, attr_name)
            try:
                json.dumps(val)
                result[attr_name] = val
            except (TypeError, ValueError):
                result[attr_name] = str(val)
        return result

    def __repr__(self) -> str:
        """Return a string representation of ProfileJob."""
        attribute_strs = []
        for attribute in self.attributes:
            if attribute == "error":
                attribute_str = getattr(self, attribute).split("\n")[0]
            else:
                attribute_str = getattr(self, attribute)
            attribute_strs.append(f"{attribute}={attribute_str}")
        attributes_str = ", ".join(attribute_strs)
        return f"ProfileJob({attributes_str})"

    def add_attributes(self, **kwargs: Any) -> None:
        """Add new attributes to the job and track them in the attributes list.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes.

        Raises:
            AssertionError: If an attribute already exists.
        """
        for key, value in kwargs.items():
            assert not hasattr(self, key), f"Attribute {key} already exists in ProfileJob."
            setattr(self, key, value)
            self.attributes.append(key)

    @property
    def has_error(self) -> bool:
        """Whether this job encountered an error."""
        return hasattr(self, "error")

    @property
    def is_correct(self) -> bool:
        """Whether correctness verification passed."""
        return hasattr(self, "correctness_result") and self.correctness_result is True

    @property
    def sort_val(self) -> tuple[int, float]:
        """Sorting key: (priority, metric_value). Lower is better."""
        if self.has_error:
            return (2, float("inf"))
        elif not hasattr(self, self.main_metric):
            return (1, float("inf"))
        else:
            return (0, getattr(self, self.main_metric))

    def add_error(self, error_msg: str) -> None:
        """Record an error message, keeping only the first error encountered.

        Args:
            error_msg: The error message to record.
        """
        if not self.has_error:
            self.add_attributes(error=error_msg)


class ProfileJobs:
    """Collection of ProfileJob instances with batch management."""

    def __init__(self, cache_root_dir: str) -> None:
        """Initialize job collection with cache directory and empty job list.

        Args:
            cache_root_dir: Root directory for caching results.
        """
        self.cache_root_dir = cache_root_dir
        self.jobs: dict[int, ProfileJob] = {}
        self.main_metric = "min_ms"

    def add_job(
        self,
        kernel: Callable,
        kernel_kwargs: dict[str, Any],
        output_shapes: dict[str, tuple[int, ...]],
        compiler_flags: str,
        correctness_check: CORRECTNESS_CHECK_DTYPE = None,
    ) -> None:
        """Create and add a new ProfileJob to the collection.

        Numpy arrays in kernel_kwargs are treated as tensor inputs (shapes and
        dtype inferred from them). Non-array values are passed as scalar keyword
        arguments. MAC count is automatically computed by tracing nc_matmul calls.

        Args:
            kernel: The NKI kernel function (or @nki.jit-wrapped).
            kernel_kwargs: All kernel arguments. Numpy arrays are tensor inputs;
                non-array values are scalar keyword arguments.
            output_shapes: Shapes of output tensors, mapping name to shape tuple.
            compiler_flags: Compilation flags string.
            correctness_check: Tuple of (golden_fn, atol, rtol) for correctness
                verification, or None to skip.

        Raises:
            ValueError: If kernel_kwargs contains no numpy arrays.
        """
        kernel_ref = resolve_kernel_ref(kernel)
        tensor_inputs = {k: v for k, v in kernel_kwargs.items() if isinstance(v, np.ndarray)}
        scalar_kwargs = {k: v for k, v in kernel_kwargs.items() if not isinstance(v, np.ndarray)}

        if not tensor_inputs:
            raise ValueError("kernel_kwargs must contain at least one numpy array as a tensor input.")

        input_tensor_shapes = {name: arr.shape for name, arr in tensor_inputs.items()}
        data_type = next(iter(tensor_inputs.values())).dtype
        mac_count = compute_mac_count(kernel, kernel_kwargs)

        job_index = len(self.jobs)
        job = ProfileJob(
            index=job_index,
            main_metric=self.main_metric,
            kernel=kernel_ref,
            input_tensors=tensor_inputs,
            input_tensor_shapes=input_tensor_shapes,
            output_shapes=output_shapes,
            mac_count=mac_count,
            data_type=data_type,
            scalar_kwargs=scalar_kwargs,
            compiler_flags=compiler_flags,
            correctness_check=correctness_check,
            cache_root_dir=self.cache_root_dir,
        )
        self.jobs[job_index] = job

    def extend(self, other_jobs: "ProfileJobs") -> None:
        """Append all jobs from another ProfileJobs collection.

        Args:
            other_jobs: ProfileJobs instance to append.
        """
        job_index_offset = len(self.jobs)
        for job_index in other_jobs.jobs:
            original_job = other_jobs.jobs[job_index]
            job = copy.deepcopy(original_job)
            job.index = job_index_offset + job_index
            self.jobs[job.index] = job

    def __repr__(self) -> str:
        """Return a string representation of ProfileJobs."""
        if len(self.jobs) == 0:
            return "ProfileJobs(jobs: None)"

        if len(self.jobs) <= 4:
            jobs_str = ",\n  ".join(str(job) for job in self.jobs.values())
            result = f"ProfileJobs({len(self.jobs)} jobs):\n  {jobs_str}"
        else:
            job_indices = sorted(self.jobs.keys())
            result = (
                f"ProfileJobs({len(self.jobs)} jobs):\n"
                f"  {self.jobs[job_indices[0]]},\n"
                f"  {self.jobs[job_indices[1]]},\n"
                f"  ...({len(self.jobs) - 4} more jobs)...,\n"
                f"  {self.jobs[job_indices[-2]]},\n"
                f"  {self.jobs[job_indices[-1]]}"
            )

        return result

    def subset(self, indices: list[int]) -> "ProfileJobs":
        """Create a new ProfileJobs containing only specified job indices.

        Args:
            indices: List of job indices to include.

        Returns:
            New ProfileJobs with selected jobs.
        """
        subset_jobs = ProfileJobs(self.cache_root_dir)
        subset_jobs.jobs = {index: self.jobs[index] for index in indices}
        return subset_jobs

    def dump_json(self) -> None:
        """Save performance metrics to JSON files, organized by workload directory.

        Raises:
            OSError: If the directory cannot be created or the file cannot be written.
        """
        filename = "perf_metrics.json"
        jobs_by_workload_dir: dict[str, list[int]] = {}

        for job_index in self.jobs:
            job = self.jobs[job_index]
            if job.workload_dir not in jobs_by_workload_dir:
                jobs_by_workload_dir[job.workload_dir] = []
            jobs_by_workload_dir[job.workload_dir].append(job_index)

        for wl_dir, job_indices in jobs_by_workload_dir.items():
            sorted_job_indices = sorted(job_indices, key=lambda ji: self.jobs[ji].sort_val)

            correct_count = 0
            error_count = 0
            error_types: dict[str, int] = {}
            for ji in sorted_job_indices:
                job = self.jobs[ji]
                if job.has_error:
                    error_count += 1
                    error_type = getattr(job, "error").split("\n")[0]
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1
                elif job.is_correct:
                    correct_count += 1

            json_data = {
                "metadata": {
                    "num_results": len(sorted_job_indices),
                    "num_correct_results": correct_count,
                    "num_error_results": error_count,
                    "error_types": error_types,
                    "main_metric": self.main_metric,
                },
                "results": [self.jobs[ji].to_dict() for ji in sorted_job_indices],
            }

            try:
                os.makedirs(wl_dir, exist_ok=True)
                filepath = os.path.join(wl_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=2, sort_keys=True)
            except Exception as e:
                raise OSError(f"Failed to save metrics to {filepath}: {str(e)}")


def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler to raise TimeoutError for compilation timeout."""
    raise TimeoutError("Compilation timed out after 10 minutes")


def compile_jobs(jobs: ProfileJobs) -> ProfileJobs:
    """Compile all kernel jobs with timeout protection and error handling.

    Args:
        jobs: Collection of jobs to compile.

    Returns:
        Updated ProfileJobs with compilation results.
    """
    for job_index in jobs.jobs:
        job = jobs.jobs[job_index]
        tmp_dir = tempfile.mkdtemp(prefix="nki_compile_")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)
            output_tensors = [(name, shape, job.data_type) for name, shape in job.output_shapes.items()]
            neff = compile_kernel(
                kernel_name=job.kernel,
                input_tensors=job.input_tensors,
                output_tensors=output_tensors,
                kernel_kwargs=job.scalar_kwargs,
                compiler_flags=job.compiler_flags,
                output_dir=tmp_dir,
            )
            neff_filename = os.path.basename(neff)
            neff = os.path.join(job.cache_dir, neff_filename)
            job.add_attributes(neff=neff)
        except Exception as e:
            error_msg = capture_error_message(e)
            job.add_error(error_msg)
        finally:
            signal.alarm(0)
            if os.listdir(tmp_dir):
                job.init_job_dir()
                for item_name in os.listdir(tmp_dir):
                    src_path = os.path.join(tmp_dir, item_name)
                    dst_path = os.path.join(job.cache_dir, item_name)
                    shutil.move(src_path, dst_path)
            shutil.rmtree(tmp_dir)
    return jobs
