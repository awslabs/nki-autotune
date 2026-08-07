"""Discover retained workloads by exact workload and shape."""

from __future__ import annotations

from importlib import import_module

from kernel_library.workload import Workload

WorkloadKey = tuple[str, str]

_WORKLOAD_MODULES: dict[WorkloadKey, str] = {
    ("attention", "q16384_kv16384_d128"): "kernel_library.attention_q16384_kv16384_d128",
    ("matmul-lhs", "m2048_k2048_n2048"): "kernel_library.matmul_lhs_rhs_m2048_k2048_n2048",
    ("matmul-lhs-t", "m2048_k2048_n2048"): "kernel_library.matmul_lhs_t_rhs_m2048_k2048_n2048",
    ("rmsnorm-matmul", "m2048_k2048_n2048"): "kernel_library.rmsnorm_matmul_m2048_k2048_n2048",
}


def workload_keys() -> tuple[WorkloadKey, ...]:
    """Return every retained ``(workload, shape)`` key."""
    return tuple(_WORKLOAD_MODULES)


def workload_names() -> tuple[str, ...]:
    """Return the retained workload names in registry order."""
    return tuple(dict.fromkeys(name for name, _shape in _WORKLOAD_MODULES))


def workload_shapes(name: str) -> tuple[str, ...]:
    """Return retained shape keys for one workload."""
    if name not in workload_names():
        choices = ", ".join(workload_names())
        raise ValueError(f"unknown workload {name!r}; choose one of: {choices}")
    return tuple(shape for workload_name, shape in _WORKLOAD_MODULES if workload_name == name)


def load_workload(name: str, shape: str) -> Workload:
    """Load and validate one exact retained workload."""
    module_name = _WORKLOAD_MODULES.get((name, shape))
    if module_name is None:
        choices = ", ".join(workload_shapes(name))
        raise ValueError(f"unknown shape {shape!r} for workload {name!r}; choose one of: {choices}")
    module = import_module(module_name)
    if getattr(module, "WORKLOAD_NAME", None) != name:
        raise ValueError(f"{module_name} does not declare WORKLOAD_NAME={name!r}")
    if getattr(module, "SHAPE", None) != shape:
        raise ValueError(f"{module_name} does not declare SHAPE={shape!r}")
    workload = getattr(module, "WORKLOAD", None)
    if not isinstance(workload, Workload):
        raise ValueError(f"{module_name} does not expose a Workload named WORKLOAD")
    return workload


__all__ = ["WorkloadKey", "load_workload", "workload_keys", "workload_names", "workload_shapes"]
