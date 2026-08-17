"""Automatically discovered NAKB workloads with copied PyTorch goldens."""

from __future__ import annotations

import inspect
import math
import pkgutil
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import TypedDict, cast

import numpy as np
import torch

from nkigym.profile import InputSpecs

TorchResult = torch.Tensor | tuple[torch.Tensor | None, ...] | dict[str, torch.Tensor | np.ndarray] | np.ndarray
InputGenerator = Callable[[InputSpecs, int], dict[str, np.ndarray]]
ArgumentAdapter = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class TorchReference:
    """A copied NAKB PyTorch golden with workload-specific arguments bound."""

    function: Callable[..., object]
    parameters: tuple[str, ...]
    bound_kwargs: Mapping[str, object] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    subscript: int | None = None
    argument_adapter: ArgumentAdapter | None = None

    @property
    def __signature__(self) -> inspect.Signature:
        """Expose only the tensor parameters retained by the workload."""
        parameters = tuple(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in self.parameters)
        return inspect.Signature(parameters)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call the copied reference with its exact case arguments."""
        public_arguments = self.__signature__.bind(*args, **kwargs).arguments
        call_arguments = dict(self.bound_kwargs)
        for name, value in public_arguments.items():
            call_arguments[self.aliases.get(name, name)] = value
        target = self.function
        if self.subscript is not None:
            target = cast(Callable[..., object], target[self.subscript])  # type: ignore[index]
        call_arguments = _coerce_enum_args(target, call_arguments)
        call_arguments = {name: _to_torch_argument(value) for name, value in call_arguments.items()}
        if self.argument_adapter is not None:
            call_arguments = self.argument_adapter(call_arguments)
        return target(**call_arguments)


def _to_torch_argument(value: object) -> object:
    """Apply NAKB's default NumPy-to-Torch reference argument conversion."""
    if not isinstance(value, np.ndarray):
        return value
    dtype_name = str(value.dtype)
    if "x4" in dtype_name:
        return value
    if value.dtype == np.uint32:
        value = value.astype(np.int32)
    elif "bfloat16" in dtype_name or "float8" in dtype_name:
        value = value.astype(np.float32)
    tensor = torch.from_numpy(value)
    if tensor.dtype == torch.float16:
        tensor = tensor.float()
    return tensor


def _coerce_enum_args(function: Callable[..., object], inputs: dict[str, object]) -> dict[str, object]:
    """Apply NAKB's conversion for integer and string enum arguments."""
    annotations = inspect.get_annotations(function, eval_str=True)
    coerced = dict(inputs)
    for parameter_name, annotation in annotations.items():
        if parameter_name not in coerced:
            continue
        enum_type = annotation
        if typing.get_origin(annotation) is not None:
            enum_type = next(
                (
                    argument
                    for argument in typing.get_args(annotation)
                    if isinstance(argument, type) and issubclass(argument, Enum)
                ),
                None,
            )
        if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
            continue
        value = coerced[parameter_name]
        if isinstance(value, Enum):
            continue
        if isinstance(value, int):
            coerced[parameter_name] = enum_type(value)
        elif isinstance(value, str):
            members = enum_type.__members__
            if value in members:
                coerced[parameter_name] = members[value]
            else:
                coerced[parameter_name] = {name.lower(): member for name, member in members.items()}[value.lower()]
    return coerced


class _WorkloadFields(TypedDict):
    """Fields shared by every strict workload dictionary."""

    input_specs: InputSpecs
    input_generator: InputGenerator
    atol: float
    rtol: float
    nakb_latency_ms: float
    best_historical_latency_ms: float


class Workload(_WorkloadFields):
    """One target workload backed by an exact copied NAKB PyTorch reference."""

    torch_ref: TorchReference


class SynthesisWorkload(_WorkloadFields):
    """One workload exposed through the compatibility alias registry."""

    torch_ref: TorchReference


_WORKLOAD_FIELDS = frozenset(Workload.__required_keys__)
_GROUPED_SYNTHESIS_WORKLOADS = {"dynamic_elementwise_add_m512_h256": ("dynamic_elementwise_add", 0)}


def _validate_input_specs(module_name: str, raw_input_specs: object) -> InputSpecs:
    """Validate strict tensor-only input specifications."""
    if not isinstance(raw_input_specs, dict) or not raw_input_specs:
        raise TypeError(f"{module_name}.input_specs must be a non-empty dictionary")
    input_specs = cast(dict[str, object], raw_input_specs)
    for parameter_name, raw_spec in input_specs.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError(f"{module_name}.input_specs contains an invalid parameter name")
        if not isinstance(raw_spec, tuple) or len(raw_spec) != 2:
            raise TypeError(f"{module_name}.input_specs[{parameter_name!r}] must be (shape, dtype)")
        shape, dtype = raw_spec
        if (
            not isinstance(shape, tuple)
            or not shape
            or not all(isinstance(extent, int) and not isinstance(extent, bool) and extent > 0 for extent in shape)
        ):
            raise ValueError(f"{module_name}.input_specs[{parameter_name!r}] has an invalid shape")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"{module_name}.input_specs[{parameter_name!r}] has an invalid dtype")
    return cast(InputSpecs, input_specs)


def _validate_workload(module_name: str, raw_workload: object) -> Workload:
    """Validate one exact seven-field workload dictionary."""
    if not isinstance(raw_workload, dict):
        raise TypeError(f"{module_name} workload must be a dictionary")
    values = cast(dict[str, object], raw_workload)
    if set(values) != _WORKLOAD_FIELDS:
        raise ValueError(f"{module_name} workload fields must be exactly {sorted(_WORKLOAD_FIELDS)}")
    torch_ref = values["torch_ref"]
    input_specs = _validate_input_specs(module_name, values["input_specs"])
    input_generator = values["input_generator"]
    if not isinstance(torch_ref, TorchReference) or not callable(input_generator):
        raise TypeError(f"{module_name} has a non-callable reference or input generator")
    owner_module = module_name.partition(".WORKLOADS")[0]
    if torch_ref.function.__module__ != owner_module:
        raise ValueError(f"{module_name}.torch_ref must be defined directly in {owner_module}")
    if list(inspect.signature(torch_ref).parameters) != list(input_specs):
        raise ValueError(f"{module_name}.torch_ref parameters must match input_specs")
    if list(inspect.signature(input_generator).parameters) != ["input_specs", "seed"]:
        raise ValueError(f"{module_name}.input_generator parameters must be input_specs and seed")
    for field in ("atol", "rtol", "nakb_latency_ms", "best_historical_latency_ms"):
        value = values[field]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{module_name}.{field} must be a finite float")
    atol = cast(float, values["atol"])
    rtol = cast(float, values["rtol"])
    if atol < 0.0 or rtol < 0.0:
        raise ValueError(f"{module_name} tolerances must be non-negative")
    for field in ("nakb_latency_ms", "best_historical_latency_ms"):
        latency = cast(float, values[field])
        if latency <= 0.0:
            raise ValueError(f"{module_name}.{field} must be positive")
    return cast(Workload, values)


def _validate_grouped_workloads(module_name: str, raw_workloads: object) -> tuple[Workload, ...]:
    """Validate the strict workloads grouped in one NAKB module."""
    if not isinstance(raw_workloads, tuple) or not raw_workloads:
        raise TypeError(f"{module_name}.WORKLOADS must be a non-empty tuple")
    workloads = tuple(
        _validate_workload(f"{module_name}.WORKLOADS[{index}]", raw_workload)
        for index, raw_workload in enumerate(raw_workloads)
    )
    return workloads


def _discover_workloads() -> dict[str, tuple[Workload, ...]]:
    """Discover grouped NAKB workload targets."""
    nakb_workloads: dict[str, tuple[Workload, ...]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.ispkg or module_info.name.startswith("_"):
            continue
        if not module_info.name.startswith("nakb_"):
            raise RuntimeError(f"kernel_library contains non-NAKB module {module_info.name!r}")
        module_name = f"{__name__}.{module_info.name}"
        module = import_module(module_name)
        workload_name = module_info.name.removeprefix("nakb_")
        nakb_workloads[workload_name] = _validate_grouped_workloads(module_name, getattr(module, "WORKLOADS", None))
    if not nakb_workloads:
        raise RuntimeError("kernel_library contains no NAKB workloads")
    return dict(sorted(nakb_workloads.items()))


def _workload_aliases(nakb_workloads: dict[str, tuple[Workload, ...]]) -> dict[str, SynthesisWorkload]:
    """Return configured aliases for selected NAKB workloads."""
    workloads: dict[str, SynthesisWorkload] = {}
    for workload_name, (workload_type, workload_index) in _GROUPED_SYNTHESIS_WORKLOADS.items():
        grouped_workloads = nakb_workloads.get(workload_type)
        if grouped_workloads is None or workload_index >= len(grouped_workloads):
            raise RuntimeError(f"missing grouped synthesis workload {workload_name!r}")
        workloads[workload_name] = cast(SynthesisWorkload, grouped_workloads[workload_index])
    return dict(sorted(workloads.items()))


NAKB_WORKLOADS = _discover_workloads()
WORKLOADS = _workload_aliases(NAKB_WORKLOADS)

__all__ = [
    "ArgumentAdapter",
    "InputGenerator",
    "InputSpecs",
    "NAKB_WORKLOADS",
    "SynthesisWorkload",
    "TorchReference",
    "TorchResult",
    "WORKLOADS",
    "Workload",
]
