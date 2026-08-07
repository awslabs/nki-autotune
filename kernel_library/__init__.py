"""Retained nkigym workloads and their best-known schedules."""

from kernel_library.registry import WorkloadKey, load_workload, workload_keys, workload_names, workload_shapes
from kernel_library.workload import InputSpecs, Workload

__all__ = [
    "InputSpecs",
    "Workload",
    "WorkloadKey",
    "load_workload",
    "workload_keys",
    "workload_names",
    "workload_shapes",
]
