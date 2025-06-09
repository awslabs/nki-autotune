from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from tqdm import tqdm

from autotune.tune.utils import capture_error_message


def split_jobs_into_groups(job_ids: List[int], num_groups: int):
    """
    Split job_ids into evenly distributed groups.

    Args:
        job_ids: List of job IDs to split
        num_groups: Number of groups to split into

    Returns:
        List of lists, where each inner list contains job IDs for that group
    """
    groups: List[List[int]] = [[] for _ in range(num_groups)]

    for i, job_id in enumerate(job_ids):
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
    and processes their results.

    The function supports two job submission patterns:
    1. One-to-one mapping: Each job has its own dedicated function and arguments.
       This is the case when len(funcs) == len(args).
    2. Batch processing: A single function processes multiple jobs with a single set
       of arguments. This is the case when len(funcs) == 1.

    Args:
        executor_type: Type of executor to use ("process" or "thread")
        num_workers: Number of parallel workers to use
        job_ids: List of job IDs to process
        submit_jobs_func: Function that prepares callables and arguments for submission
            Args:
                job_group_id (int): Index of the job group
                job_group (List[int]): A group of job IDs to process
            Returns:
                Tuple[List[Callable], List[Tuple[Any, ...]]]: A tuple containing:
                    - List of callable functions to execute
                    - List of argument tuples for each function
        work_desc: Description for the progress bar display
        process_results_func: Function that processes results from completed futures
            Args:
                job_ids (Union[int, List[int]]): ID or list of IDs for the job(s)
                future_result (Any): The result returned by the completed future

    Raises:
        Exception: If executor_type is not "process" or "thread"
        Exception: If the number of functions doesn't match the number of argument sets
            or isn't exactly 1

    Returns:
        None
    """
    pool_executor = get_executor(executor_type)

    futures: List[Tuple[int, Any]] = []
    job_groups = split_jobs_into_groups(job_ids, num_workers)
    with pool_executor(max_workers=num_workers) as executor:
        for job_group_id, job_group in enumerate(job_groups):
            if not job_group:
                continue
            funcs, kwargs = submit_jobs_func(job_group_id, job_group)
            assert len(funcs) == len(
                kwargs
            ), f"Functions must match with kwargs. Received {len(funcs)} funcs and {len(kwargs)} kwargs."
            for job_id, func, kwarg in zip(job_group, funcs, kwargs):
                future = executor.submit(func, **kwarg)
                futures.append((job_id, future))

    for job_id, future in tqdm(futures, total=len(futures), desc=work_desc):
        try:
            future_result = future.result()
            error = False
        except Exception as e:
            future_result = capture_error_message(e)
            error = True
        process_results_func(error, job_id, future_result)


def parallel_execute_groups(
    executor_type: str,
    num_workers: int,
    job_ids: List[int],
    submit_jobs_func: Callable[[int, List[int]], Tuple[Callable, Dict[str, Any]]],
    work_desc: str,
    process_results_func: Callable[[bool, List[int], Any], None],
) -> None:
    """
    Execute jobs in parallel using either process or thread pool executors.

    This function distributes jobs across multiple workers, executes them in parallel,
    and processes their results.

    The function supports two job submission patterns:
    1. One-to-one mapping: Each job has its own dedicated function and arguments.
       This is the case when len(funcs) == len(args).
    2. Batch processing: A single function processes multiple jobs with a single set
       of arguments. This is the case when len(funcs) == 1.

    Args:
        executor_type: Type of executor to use ("process" or "thread")
        num_workers: Number of parallel workers to use
        job_ids: List of job IDs to process
        submit_jobs_func: Function that prepares callables and arguments for submission
            Args:
                job_group_id (int): Index of the job group
                job_group (List[int]): A group of job IDs to process
            Returns:
                Tuple[List[Callable], List[Tuple[Any, ...]]]: A tuple containing:
                    - List of callable functions to execute
                    - List of argument tuples for each function
        work_desc: Description for the progress bar display
        process_results_func: Function that processes results from completed futures
            Args:
                job_ids (Union[int, List[int]]): ID or list of IDs for the job(s)
                future_result (Any): The result returned by the completed future

    Raises:
        Exception: If executor_type is not "process" or "thread"
        Exception: If the number of functions doesn't match the number of argument sets
            or isn't exactly 1

    Returns:
        None
    """
    pool_executor = get_executor(executor_type)

    futures = []
    job_groups = split_jobs_into_groups(job_ids, num_workers)
    with pool_executor(max_workers=num_workers) as executor:
        for job_group_id, job_group in enumerate(job_groups):
            if not job_group:
                continue
            func, kwargs = submit_jobs_func(job_group_id, job_group)
            future = executor.submit(func, **kwargs)
            futures.append((job_group, future))

    for job_group, future in tqdm(futures, total=len(futures), desc=work_desc):
        try:
            future_result = future.result()
            error = False
        except Exception as e:
            future_result = capture_error_message(e)
            error = True
        process_results_func(error, job_group, future_result)
