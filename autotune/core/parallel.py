import inspect
import os
import random
from typing import List

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
