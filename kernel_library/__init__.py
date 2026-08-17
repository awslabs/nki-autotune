"""Automatically discovered NumPy workload dictionaries."""

from __future__ import annotations

import inspect
import math
import pkgutil
from collections.abc import Callable
from importlib import import_module
from typing import TypedDict, cast

import numpy as np

from nkigym.search.types import InputSpecs

ArrayResult = np.ndarray | tuple[np.ndarray, ...]
InputGenerator = Callable[[InputSpecs, int], dict[str, np.ndarray]]


class Workload(TypedDict):
    """One NumPy reference, input contract, and historical performance."""

    numpy_ref: Callable[..., ArrayResult]
    input_specs: InputSpecs
    input_generator: InputGenerator
    atol: float
    rtol: float
    best_historical_mfu: float


_FIELDS = frozenset(Workload.__required_keys__)


def _validate_workload(module_name: str, raw_workload: object) -> Workload:
    """Validate and return one imported workload dictionary."""
    if not isinstance(raw_workload, dict):
        raise TypeError(f"{module_name}.WORKLOAD must be a dictionary")
    values = cast(dict[str, object], raw_workload)
    if set(values) != _FIELDS:
        raise ValueError(f"{module_name}.WORKLOAD fields must be exactly {sorted(_FIELDS)}")
    numpy_ref = values["numpy_ref"]
    input_specs = values["input_specs"]
    input_generator = values["input_generator"]
    if not callable(numpy_ref) or not isinstance(input_specs, dict) or not callable(input_generator):
        raise TypeError(f"{module_name}.WORKLOAD has an invalid reference, input_specs, or input_generator")
    if list(inspect.signature(numpy_ref).parameters) != list(input_specs):
        raise ValueError(f"{module_name}.numpy_ref parameters must match input_specs")
    if list(inspect.signature(input_generator).parameters) != ["input_specs", "seed"]:
        raise ValueError(f"{module_name}.input_generator parameters must be input_specs and seed")
    for field in ("atol", "rtol", "best_historical_mfu"):
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{module_name}.{field} must be finite")
    return cast(Workload, values)


def _discover_workloads() -> dict[str, Workload]:
    """Import every workload module in this package."""
    discovered: dict[str, Workload] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.name.startswith("_"):
            module = import_module(f"{__name__}.{module_info.name}")
            discovered[module_info.name] = _validate_workload(module.__name__, getattr(module, "WORKLOAD", None))
    if not discovered:
        raise RuntimeError("kernel_library contains no workload modules")
    return discovered


WORKLOADS = _discover_workloads()

__all__ = ["InputGenerator", "InputSpecs", "WORKLOADS", "Workload"]
