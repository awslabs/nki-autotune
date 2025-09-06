import inspect
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from tqdm import tqdm

from autotune.cache.results import capture_error_message
from autotune.typing.infra_types import KERNEL_DTYPE


def split_jobs_into_groups(job_ids: List[int], num_groups: int):
    """
    Split job_ids into evenly distributed groups with randomized order.
    """
    groups: List[List[int]] = [[] for _ in range(num_groups)]

    # Create a copy and shuffle it to randomize job distribution
    shuffled_job_ids = job_ids.copy()
    random.shuffle(shuffled_job_ids)

    # Distribute shuffled jobs using round-robin
    for i, job_id in enumerate(shuffled_job_ids):
        group_idx = i % num_groups
        groups[group_idx].append(job_id)

    return groups


def get_executor(executor_type: str):
    if executor_type == "process":
        pool_executor = ProcessPoolExecutor
    elif executor_type == "thread":
        pool_executor = ThreadPoolExecutor
    else:
        raise Exception(f"Invalid executor type {executor_type}. Expecting process | thread.")
    return pool_executor


def parallel_execute(
    executor_type: str,
    num_workers: int,
    job_ids: List[int],
    submit_jobs_func: Callable[[int, List[int]], Tuple[List[Callable], List[Dict[str, Any]]]],
    work_desc: str,
    process_results_func: Callable[[bool, int, Any], None],
) -> None:
    """
    Execute jobs in parallel using either process or thread pool executors.

    This function distributes jobs across multiple workers, executes them in parallel,
    and processes their results as they complete.

    Args:
        executor_type: Type of executor to use ("process" or "thread")
        num_workers: Number of parallel workers to use
        job_ids: List of job IDs to process
        submit_jobs_func: Function that prepares callables and arguments for submission
            Args:
                job_group_id (int): Index of the job group
                job_group (List[int]): A group of job IDs to process
            Returns:
                Tuple[List[Callable], List[Dict[str, Any]]]: A tuple containing:
                    - List of callable functions to execute
                    - List of keyword argument dictionaries for each function
        work_desc: Description for the progress bar display (without worker count)
        process_results_func: Function that processes results from completed futures
            Args:
                error (bool): Whether an error occurred during execution
                job_id (int): ID of the completed job
                future_result (Any): The result returned by the completed future or error message

    Raises:
        ValueError: If executor_type is not "process" or "thread"
        AssertionError: If the number of functions doesn't match the number of argument sets

    Returns:
        None
    """
    pool_executor = get_executor(executor_type)

    # Add worker count to the description
    full_desc = f"{work_desc} [Workers: {num_workers}]"

    # Create a dictionary to map futures to job IDs
    future_to_job_id = {}

    job_groups = split_jobs_into_groups(job_ids, num_workers)
    with pool_executor(max_workers=num_workers) as executor:
        # Submit all jobs
        for job_group_id, job_group in enumerate(job_groups):
            if not job_group:
                continue

            funcs, kwargs = submit_jobs_func(job_group_id, job_group)
            assert len(funcs) == len(
                kwargs
            ), f"Functions must match with kwargs. Received {len(funcs)} funcs and {len(kwargs)} kwargs."

            for job_id, func, kwarg in zip(job_group, funcs, kwargs):
                future = executor.submit(func, **kwarg)
                future_to_job_id[future] = job_id

        # Process results as they complete
        with tqdm(total=len(future_to_job_id), desc=full_desc) as progress_bar:
            for future in as_completed(future_to_job_id.keys()):
                job_id = future_to_job_id[future]
                try:
                    future_result = future.result()
                    error = False
                except Exception as e:
                    future_result = capture_error_message(e)
                    error = True

                process_results_func(error, job_id, future_result)
                progress_bar.update(1)


def parallel_execute_groups(
    executor_type: str,
    num_workers: int,
    job_ids: List[int],
    submit_jobs_func: Callable[[int, List[int]], Tuple[Callable, Dict[str, Any]]],
    work_desc: str,
    process_results_func: Callable[[bool, List[int], Any], None],
) -> None:
    """
    Execute groups of jobs in parallel using either process or thread pool executors.

    This function distributes groups of jobs across multiple workers, executes them in parallel,
    and processes their results as they complete. Unlike parallel_execute, this function
    processes entire groups of jobs with a single function call.

    Args:
        executor_type: Type of executor to use ("process" or "thread")
        num_workers: Number of parallel workers to use
        job_ids: List of job IDs to process
        submit_jobs_func: Function that prepares a callable and arguments for a job group
            Args:
                job_group_id (int): Index of the job group
                job_group (List[int]): A group of job IDs to process
            Returns:
                Tuple[Callable, Dict[str, Any]]: A tuple containing:
                    - A callable function to execute for this group
                    - A dictionary of keyword arguments for the function
        work_desc: Description for the progress bar display (without worker count)
        process_results_func: Function that processes results from completed futures
            Args:
                error (bool): Whether an error occurred during execution
                job_group (List[int]): List of job IDs in the completed group
                future_result (Any): The result returned by the completed future or error message

    Raises:
        ValueError: If executor_type is not "process" or "thread"

    Returns:
        None
    """
    pool_executor = get_executor(executor_type)

    # Split jobs into groups
    job_groups = split_jobs_into_groups(job_ids, num_workers)

    # Calculate the total number of jobs
    total_jobs = len(job_ids)

    # Add worker count and total jobs to the description
    full_desc = f"{work_desc} [Workers: {num_workers}]"

    # Create a dictionary to map futures to job groups
    future_to_job_group = {}

    with pool_executor(max_workers=num_workers) as executor:
        # Submit all job groups
        for job_group_id, job_group in enumerate(job_groups):
            if not job_group:
                continue

            func, kwargs = submit_jobs_func(job_group_id, job_group)
            future = executor.submit(func, **kwargs)
            future_to_job_group[future] = job_group

        # Process results as they complete
        with tqdm(total=total_jobs, desc=full_desc) as progress_bar:
            for future in as_completed(future_to_job_group.keys()):
                job_group = future_to_job_group[future]
                jobs_in_group = len(job_group)

                try:
                    future_result = future.result()
                    error = False
                except Exception as e:
                    future_result = capture_error_message(e)
                    error = True

                process_results_func(error, job_group, future_result)

                # Update the progress bar based on number of jobs in this group
                progress_bar.update(jobs_in_group)


def get_function_name(func) -> KERNEL_DTYPE:
    """
    Extract the absolute path and name of an imported function.

    Args:
        func: The function object to analyze

    Returns:
        Tuple[str, str]: A tuple containing (absolute_file_path, function_name)
        Returns (None, None) if the function's module cannot be determined
    """
    # Get the function name
    func_name = func.__name__ if hasattr(func, "__name__") else str(func)

    # Get the module where the function is defined
    module = inspect.getmodule(func)

    # Get the file path of the module
    if module and hasattr(module, "__file__") and module.__file__:
        module_path = module.__file__
        # Convert to absolute path
        absolute_path = os.path.abspath(module_path)
    else:
        raise Exception(f"Cannot parse absolute path and function name for {func}")
    return absolute_path, func_name


def set_neuron_core(core_id: int):
    """
    Initializer function that runs once when each worker process starts.
    Sets the NEURON_RT_VISIBLE_CORES environment variable.
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
