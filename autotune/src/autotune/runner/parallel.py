"""Utilities for distributing profiling jobs across workers."""

import os
import random


def split_jobs_into_groups(job_ids: list[int], num_groups: int) -> list[list[int]]:
    """Split job_ids into evenly distributed groups with randomized order.

    Ensures no empty groups are created and number of groups <= num_groups.

    Args:
        job_ids: List of job identifiers to distribute.
        num_groups: Maximum number of groups to create.

    Returns:
        List of job ID groups, each a list of integers.

    Raises:
        ValueError: If num_groups is not positive.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")

    if not job_ids:
        return []

    effective_num_groups = min(len(job_ids), num_groups)
    groups: list[list[int]] = [[] for _ in range(effective_num_groups)]

    shuffled_job_ids = job_ids.copy()
    random.shuffle(shuffled_job_ids)

    for i, job_id in enumerate(shuffled_job_ids):
        group_idx = i % effective_num_groups
        groups[group_idx].append(job_id)

    return groups


def set_neuron_core(core_id: int) -> None:
    """Initializer function for worker processes to set NEURON_RT_VISIBLE_CORES.

    Args:
        core_id: The Neuron core ID to make visible to this worker.
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
