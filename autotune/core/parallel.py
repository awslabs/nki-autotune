import os
import random
from typing import List


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


def set_neuron_core(core_id: int):
    """
    Initializer function that runs once when each worker process starts.
    Sets the NEURON_RT_VISIBLE_CORES environment variable.
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
