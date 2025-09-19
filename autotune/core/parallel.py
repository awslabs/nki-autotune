import os
import random
from multiprocessing import Lock, Value
from typing import List

from neuronpy.runtime.spike import SpikeExecutor

worker_spike_executor = None


def split_jobs_into_groups(job_ids: List[int], num_groups: int) -> List[List[int]]:
    """
    Split job_ids into evenly distributed groups with randomized order.
    Ensures no empty groups are created and number of groups <= num_groups.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")

    if not job_ids:
        return []

    # Calculate effective number of groups (no more than jobs available)
    effective_num_groups = min(len(job_ids), num_groups)
    groups: List[List[int]] = [[] for _ in range(effective_num_groups)]

    # Create a copy and shuffle it to randomize job distribution
    shuffled_job_ids = job_ids.copy()
    random.shuffle(shuffled_job_ids)

    # Distribute shuffled jobs using round-robin
    for i, job_id in enumerate(shuffled_job_ids):
        group_idx = i % effective_num_groups
        groups[group_idx].append(job_id)

    return groups


def set_neuron_core(core_id: int):
    """
    Initializer function that runs once when each worker process starts.
    Sets the NEURON_RT_VISIBLE_CORES environment variable.
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"


def set_neuron_core_dynamic(core_counter: Value, lock: Lock):
    with lock:
        set_neuron_core(core_counter.value)
        core_counter.value += 1
    global worker_spike_executor
    worker_spike_executor = SpikeExecutor(verbose=0)
