import copy
import json
import os
import shutil
import signal
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from autotune.cache.results import capture_error_message
from autotune.core.compile import compile_kernel, process_compiler_flags
from autotune.core.metrics import tensor_to_matmul_mac_count
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
        data_type: np.dtype,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        postprocessing: Optional[POSTPROCESSING_DTYPE],
        cache_root_dir: str,
    ) -> None:
        self.attributes = []
        target_instance_family, compiler_flags = process_compiler_flags(compiler_flags)
        input_tensor_shapes_str = "_".join("x".join(str(dim) for dim in shape) for shape in input_tensor_shapes)
        _, kernel_name = kernel
        workload_dir = f"{cache_root_dir}/{kernel_name}/{input_tensor_shapes_str}"
        cache_dir = f"{workload_dir}/id{index}"
        self.add_attributes(
            index=index,
            kernel=kernel,
            input_tensor_shapes=input_tensor_shapes,
            data_type=data_type,
            kernel_kwargs=kernel_kwargs,
            target_instance_family=target_instance_family,
            compiler_flags=compiler_flags,
            cache_dir=cache_dir,
        )
        if postprocessing:
            self.postprocessing = postprocessing
        else:
            self.postprocessing = dummy_postprocessing
        self.metric_name = "min_ms"
        self.workload_dir = workload_dir

    @property
    def input_tensors(self) -> Tuple[np.ndarray, ...]:
        """Return the cached input tensors."""
        if hasattr(self, "_input_tensors"):
            return self._input_tensors
        else:
            raise ValueError(f"ProfileJob input tensors are not initialized")

    @input_tensors.setter
    def input_tensors(self, value: Tuple[np.ndarray, ...]):
        """Set the input tensors for this job."""
        self._input_tensors = value

    def init_job_dir(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation including only attributes in self.attributes."""
        result = {}
        for attr in self.attributes:
            result[attr] = str(getattr(self, attr))
        return result

    def __repr__(self) -> str:
        attribute_strs = []
        for attribute in self.attributes:
            if attribute == "error":
                attribute_str = getattr(self, attribute).split("\n")[0]
            else:
                attribute_str = getattr(self, attribute)
            attribute_strs.append(f"{attribute}={attribute_str}")
        attributes_str = ", ".join(attribute_strs)
        repr_str = f"ProfileJob({attributes_str})"
        return repr_str

    def add_attributes(self, **kwargs):
        """
        Add additional attributes to this instance.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes.
        """
        for key, value in kwargs.items():
            assert not hasattr(self, key), f"Attribute {key} already exists in ProfileJob."
            setattr(self, key, value)
            self.attributes.append(key)

    @property
    def has_error(self) -> bool:
        return hasattr(self, "error")

    @property
    def is_correct(self) -> bool:
        return hasattr(self, "postprocessing_result") and self.postprocessing_result == True

    @property
    def sort_val(self) -> Tuple[int, float]:
        if self.has_error:
            val = (2, float("inf"))
        elif not hasattr(self, self.metric_name):
            val = (1, float("inf"))
        else:
            val = (1, getattr(self, self.metric_name))
        return val

    def add_error(self, error_msg: str):
        """
        Add error information, but only if no error has been recorded yet.
        This ensures we keep the earliest error encountered.

        Args:
            error_msg: The error message to record
        """
        if not self.has_error:
            self.add_attributes(error=error_msg)


class ProfileJobs:
    def __init__(self, cache_root_dir: str) -> None:
        self.cache_root_dir = cache_root_dir
        self.jobs: Dict[int, ProfileJob] = {}
        self._tensor_cache: Dict[Tuple, Tuple[np.ndarray, ...]] = {}

    def add_job(self, **kwargs):
        job_index = len(self.jobs)
        job = ProfileJob(index=job_index, cache_root_dir=self.cache_root_dir, **kwargs)
        self.jobs[job_index] = job

    def extend(self, other_jobs: "ProfileJobs") -> None:
        """
        Concatenate another ProfileJobs object to this one.
        This modifies the current object and also returns it for chaining.

        Args:
            other_jobs: Another ProfileJobs instance to append
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
            # For small collections, show all jobs
            jobs_str = ",\n  ".join(str(job) for job in self.jobs.values())
            result = f"ProfileJobs({len(self.jobs)} jobs):\n  {jobs_str}"
        else:
            # For larger collections, show first and last jobs with count
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

    def subset(self, indices: List[int]) -> "ProfileJobs":
        subset_jobs = ProfileJobs(self.cache_root_dir)
        subset_jobs.jobs = {index: self.jobs[index] for index in indices}
        subset_jobs._tensor_cache = self._tensor_cache
        return subset_jobs

    def initialize_input_tensors(self) -> None:
        for job_index in self.jobs:
            job = self.jobs[job_index]
            cache_key = (tuple(job.input_tensor_shapes), job.data_type)
            if cache_key not in self._tensor_cache:
                self._tensor_cache[cache_key] = tuple(
                    np.random.normal(0, 0.001, size=shape).astype(job.data_type) for shape in job.input_tensor_shapes
                )
            job.input_tensors = self._tensor_cache[cache_key]

    def dump_json(self):
        """
        Dump the summary to JSON files.
        Results within each cache directory are sorted by the sort_key.

        Raises:
            OSError: If the directory cannot be created or the file cannot be written
        """
        filename = "perf_metrics.json"
        jobs_by_workload_dir: Dict[str, List[int]] = {}

        for job_index in self.jobs:
            job = self.jobs[job_index]
            if job.workload_dir not in jobs_by_workload_dir:
                jobs_by_workload_dir[job.workload_dir] = []
            jobs_by_workload_dir[job.workload_dir].append(job_index)

        for workload_dir in jobs_by_workload_dir:
            job_indices = jobs_by_workload_dir[workload_dir]
            sorted_job_indices = sorted(job_indices, key=lambda job_index: self.jobs[job_index].sort_val)

            correct_count = 0
            error_count = 0
            error_types: Dict[str, int] = {}
            for job_index in sorted_job_indices:
                job = self.jobs[job_index]
                if job.is_correct:
                    correct_count += 1
                if job.has_error:
                    error_count += 1
                    error_type = job.error.split("\n")[0]
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1

            json_data = {
                "metadata": {
                    "num_results": len(sorted_job_indices),
                    "num_correct_results": correct_count,
                    "num_error_results": error_count,
                    "error_types": error_types,
                },
                "results": [self.jobs[job_index].to_dict() for job_index in sorted_job_indices],
            }

            try:
                os.makedirs(workload_dir, exist_ok=True)
                filepath = os.path.join(workload_dir, "perf_metrics.json")
                with open(filepath, "w") as f:
                    json.dump(json_data, f, indent=2, sort_keys=True)
            except Exception as e:
                raise OSError(f"Failed to save metrics to {filepath}: {str(e)}")


def timeout_handler(signum, frame):
    raise TimeoutError("Compilation timed out after 10 minutes")


def compile_jobs(jobs: ProfileJobs) -> ProfileJobs:
    jobs.initialize_input_tensors()
    for job_index in jobs.jobs:
        job = jobs.jobs[job_index]
        tmp_dir = tempfile.mkdtemp(prefix="nki_compile_")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)
            matmul_mac_count = tensor_to_matmul_mac_count(job.input_tensor_shapes)
            job.add_attributes(matmul_mac_count=matmul_mac_count)
            neff = compile_kernel(
                kernel_name=job.kernel,
                input_tensors=job.input_tensors,
                kernel_kwargs=job.kernel_kwargs,
                target_instance_family=job.target_instance_family,
                compiler_flags=job.compiler_flags,
                output_dir=tmp_dir,
            )
            job.init_job_dir()
            for item_name in os.listdir(tmp_dir):
                src_path = os.path.join(tmp_dir, item_name)
                dst_path = os.path.join(job.cache_dir, item_name)
                shutil.move(src_path, dst_path)
            neff_filename = os.path.basename(neff)
            neff = os.path.join(job.cache_dir, neff_filename)
            job.add_attributes(neff=neff)
        except Exception as e:
            error_msg = capture_error_message(e)
            job.add_error(error_msg)
        finally:
            signal.alarm(0)
            shutil.rmtree(tmp_dir)
    return jobs
