import os
import pickle
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Set

from tqdm import tqdm

from autotune.tune.utils import capture_error_message
from autotune.typing import (
    INPUT_TENSORS_DTYPE,
    KERNEL_DTYPE,
    KERNEL_KWARGS_DTYPE,
    OUTPUT_TENSOR_DTYPE,
    POSTPROCESSING_DTYPE,
    PREPROCESSING_DTYPE,
)


def dummy_preprocessing(input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE) -> None:
    pass


def dummy_postprocessing(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_output: OUTPUT_TENSOR_DTYPE
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
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        preprocessing: PREPROCESSING_DTYPE,
        postprocessing: POSTPROCESSING_DTYPE,
    ) -> None:
        self.index = index
        self.kernel = kernel
        self.input_tensors = input_tensors
        self.kernel_kwargs = kernel_kwargs
        self.compiler_flags = compiler_flags
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    @property
    def cache_dir(self) -> str:
        input_tensor_shapes = "_".join("x".join(str(dim) for dim in tensor.shape) for tensor in self.input_tensors)
        cache_dir = f"{self.cache_root_dir}/{self.kernel[1]}/{input_tensor_shapes}/id{self.index}"
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

    def add_job(
        self,
        kernel: KERNEL_DTYPE,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE | None = None,
        compiler_flags: str | None = None,
        preprocessing: PREPROCESSING_DTYPE | None = None,
        postprocessing: POSTPROCESSING_DTYPE | None = None,
    ):
        if kernel_kwargs is None:
            kernel_kwargs = {}
        if compiler_flags is None:
            compiler_flags = ""
        if preprocessing is None:
            preprocessing = dummy_preprocessing
        if postprocessing is None:
            postprocessing = dummy_postprocessing
        job = ProfileJob(
            self.num_jobs, kernel, input_tensors, kernel_kwargs, compiler_flags, preprocessing, postprocessing
        )
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
                    future = executor.submit(
                        run_with_args_and_kwargs, job.preprocessing, job.input_tensors, job.kernel_kwargs
                    )
                    futures.append((job_id, future))

            # Process results of this batch
            batch_valid_jobs: List[ProfileJob] = []
            for job_id, future in tqdm(
                futures, total=len(futures), desc=f"Sampling valid jobs (need {remaining_num_samples} more)"
            ):
                job = self.jobs[job_id]
                try:
                    success = future.result()
                    if success:
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
