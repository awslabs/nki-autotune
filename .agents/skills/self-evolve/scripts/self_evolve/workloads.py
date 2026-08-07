"""Load retained workloads from kernel_library."""

from __future__ import annotations

from kernel_library import load_workload, workload_names, workload_shapes

_MFU_REGRESSION_WORKLOADS = (
    ("attention", "q16384_kv16384_d128"),
    ("matmul-lhs", "m2048_k2048_n2048"),
    ("matmul-lhs-t", "m2048_k2048_n2048"),
    ("rmsnorm-matmul", "m2048_k2048_n2048"),
)


def mfu_regression_workloads() -> tuple[tuple[str, str], ...]:
    """Return exact workload shapes covered by fixed MFU thresholds."""
    return _MFU_REGRESSION_WORKLOADS


__all__ = ["load_workload", "mfu_regression_workloads", "workload_names", "workload_shapes"]
