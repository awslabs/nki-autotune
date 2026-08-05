"""Load retained workloads from kernel_library."""

from __future__ import annotations

from importlib import import_module

from kernel_library import Workload

_WORKLOAD_MODULES = {
    "attention": "kernel_library.attention.ladder",
    "matmul-lhs": "kernel_library.matmul.lhs_rhs.ladder",
    "matmul-lhs-t": "kernel_library.matmul.lhsT_rhs.ladder",
    "rmsnorm-matmul": "kernel_library.rmsnorm_matmul.ladder",
}
_MFU_REGRESSION_WORKLOADS = ("attention", "matmul-lhs", "matmul-lhs-t", "rmsnorm-matmul")


def workload_names() -> tuple[str, ...]:
    """Return the selectable kernel-library workload names."""
    return tuple(_WORKLOAD_MODULES)


def mfu_regression_workload_names() -> tuple[str, ...]:
    """Return workloads covered by fixed test-owned MFU thresholds."""
    return _MFU_REGRESSION_WORKLOADS


def load_workload(name: str) -> Workload:
    """Load and validate one retained kernel-library workload."""
    module_name = _WORKLOAD_MODULES.get(name)
    if module_name is None:
        choices = ", ".join(workload_names())
        raise ValueError(f"unknown workload {name!r}; choose one of: {choices}")
    module = import_module(module_name)
    workload = getattr(module, "WORKLOAD", None)
    if not isinstance(workload, Workload):
        raise ValueError(f"{module_name} does not expose a Workload named WORKLOAD")
    return workload


__all__ = ["load_workload", "mfu_regression_workload_names", "workload_names"]
