"""Frozen NAKB targets, reference computations, and accuracy criteria."""

from __future__ import annotations

import inspect
import math
import pkgutil
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import Literal, TypedDict, cast

import numpy as np
import torch

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
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
    selection_ties: Literal["reference", "any"] = "reference"
    activation_storage_dtype: Literal["bfloat16"] | None = None
    activation_residency: Literal["auto", "sbuf", "shared_hbm"] = "auto"
    arithmetic: Literal["reference", "native"] = "reference"
    matmul_input_dtype: Literal["bfloat16"] | None = None

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


@dataclass(frozen=True)
class AccuracySpec:
    """Immutable tolerances, output selection, and comparison rules for one target."""

    atol: float
    rtol: float
    mode: Literal["global_max", "topk"] = "global_max"
    output_indices: tuple[int, ...] | None = None
    output_tolerances: tuple[tuple[float, float], ...] = ()
    output_views: tuple[tuple[int, tuple[int, ...], tuple[slice, ...]], ...] = ()
    output_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """Require finite, non-negative floating-point tolerances."""
        for tolerances in ((self.atol, self.rtol), *self.output_tolerances):
            if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in tolerances):
                raise ValueError("accuracy tolerances must be finite, non-negative floats")


def validate_nakb_outputs(
    actual: np.ndarray | tuple[np.ndarray, ...],
    expected: np.ndarray | tuple[np.ndarray, ...],
    inputs: dict[str, np.ndarray],
    criteria: dict[str, object],
) -> None:
    """Apply the copied NAKB maxAllClose and selected-value top-k criteria."""
    actuals = actual if isinstance(actual, tuple) else (actual,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    views, groups = criteria.get("output_views", ()), criteria.get("output_groups", ())
    if not isinstance(views, tuple) or not isinstance(groups, tuple):
        raise ValueError("NAKB output views and groups must be tuples")
    if views:
        actuals = tuple(actuals[i].reshape(shape)[selection] for i, shape, selection in views)
        expecteds = tuple(expecteds[i].reshape(shape)[selection] for i, shape, selection in views)
    if groups:
        actuals = tuple(np.concatenate([actuals[i].reshape(-1) for i in group]) for group in groups)
        expecteds = tuple(np.concatenate([expecteds[i].reshape(-1) for i in group]) for group in groups)
    selected = criteria["output_indices"]
    selected = tuple(range(len(expecteds))) if selected is None else selected
    tolerances = criteria["output_tolerances"]
    if not isinstance(selected, tuple) or any(type(index) is not int for index in selected):
        raise ValueError("NAKB output selection must contain integer indices")
    if not isinstance(tolerances, tuple):
        raise ValueError("NAKB per-output tolerances must be a tuple")
    if not tolerances:
        atol, rtol = criteria["atol"], criteria["rtol"]
        if not isinstance(atol, float) or not isinstance(rtol, float):
            raise ValueError("NAKB tolerances must be floats")
        tolerances = ((atol, rtol),) * len(selected)
    if len(actuals) != len(expecteds) or len(selected) != len(tolerances):
        raise AssertionError("NAKB output count or per-output tolerance count differs")
    pairs = [(actuals[index], expecteds[index]) for index in selected]
    if criteria["mode"] == "topk":
        if len(actuals) != 2 or len(selected) != 2 or actuals[1].dtype.kind not in "iu":
            raise AssertionError("top-k requires values and integer indices")
        values, indices = actuals
        source = inputs["inp"]
        rotation = next((key for key in inputs if key.startswith("rotation_topk_")), None)
        if rotation is not None:
            stages = int(rotation.rsplit("_", 1)[1])
            width = source.shape[1] // stages - values.shape[1]
            source = source.reshape(source.shape[0], stages, -1)[:, :, :width].reshape(source.shape[0], -1)
        if values.shape != indices.shape or values.shape != expecteds[0].shape:
            raise AssertionError("top-k output shapes differ from the reference")
        if indices.ndim != 2 or source.ndim != 2 or source.shape[0] != indices.shape[0]:
            raise AssertionError("top-k input and index row counts differ")
        if np.any(indices < 0) or np.any(indices >= source.shape[1]):
            raise AssertionError("top-k index is outside the input row")
        gathered = np.take_along_axis(source, indices.astype(np.int64), axis=-1)
        golden_values = np.sort(expecteds[0], axis=-1)
        pairs = [(np.sort(values, axis=-1), golden_values), (np.sort(gathered, axis=-1), golden_values)]
    for (left, right), (atol, rtol) in zip(pairs, tolerances, strict=True):
        if left.shape != right.shape:
            raise AssertionError(f"NAKB output shape {left.shape} differs from {right.shape}")
        try:
            left_float, right_float = left.astype(float), right.astype(float)
        except (TypeError, ValueError):
            if left.dtype != right.dtype or not np.array_equal(left.view(np.uint8), right.view(np.uint8)):
                raise AssertionError("packed output bytes differ")
            continue
        if np.any(np.isfinite(left_float) != np.isfinite(right_float)):
            raise AssertionError("NAKB output finiteness differs")
        finite = np.abs(right_float[np.isfinite(right_float)])
        bound = atol + rtol * float(finite.max(initial=0.0))
        with np.errstate(invalid="ignore"):
            difference = np.abs(left_float - right_float)
        matching_inf = np.isinf(left_float) & np.isinf(right_float) & (np.sign(left_float) == np.sign(right_float))
        difference = np.where(matching_inf, 0.0, difference)
        if not np.all(difference <= bound):
            raise AssertionError(f"NAKB maximum absolute error {difference.max()} exceeds {bound}")


def accuracy_validation(workload: Workload) -> tuple[str, dict[str, object]]:
    """Serialize the same NumPy-only validator for local and remote execution."""
    accuracy = workload["accuracy"]
    criteria: dict[str, object] = {
        "atol": accuracy.atol,
        "rtol": accuracy.rtol,
        "mode": accuracy.mode,
        "output_indices": accuracy.output_indices,
        "output_tolerances": accuracy.output_tolerances,
        "output_views": accuracy.output_views,
        "output_groups": accuracy.output_groups,
    }
    return inspect.getsource(validate_nakb_outputs), criteria


class Workload(TypedDict):
    """One fixed NAKB target with its numerical and performance criteria."""

    torch_ref: TorchReference
    input_specs: InputSpecs
    input_generator: InputGenerator
    nakb_latency_ms: float
    accuracy: AccuracySpec


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
    """Validate one literal workload."""
    if not isinstance(raw_workload, dict):
        raise TypeError(f"{module_name} workload must be a dictionary")
    values = cast(dict[str, object], raw_workload)
    if set(values) != _WORKLOAD_FIELDS:
        raise ValueError(f"{module_name} must define its workload fields and accuracy specification")
    accuracy = values["accuracy"]
    if not isinstance(accuracy, AccuracySpec):
        raise TypeError(f"{module_name}.accuracy must be an AccuracySpec")
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
    latency = values["nakb_latency_ms"]
    if type(latency) is not float or not math.isfinite(latency) or latency <= 0.0:
        raise ValueError(f"{module_name}.nakb_latency_ms must be a finite, positive float")
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
            raise RuntimeError(f"benchmark contains non-NAKB module {module_info.name!r}")
        module_name = f"{__name__}.{module_info.name}"
        module = import_module(module_name)
        workload_name = module_info.name.removeprefix("nakb_")
        nakb_workloads[workload_name] = _validate_grouped_workloads(module_name, getattr(module, "WORKLOADS", None))
    if not nakb_workloads:
        raise RuntimeError("benchmark contains no NAKB workloads")
    return dict(sorted(nakb_workloads.items()))


def _workload_aliases(nakb_workloads: dict[str, tuple[Workload, ...]]) -> dict[str, Workload]:
    """Return configured aliases for selected NAKB workloads."""
    workloads: dict[str, Workload] = {}
    for workload_name, (workload_type, workload_index) in _GROUPED_SYNTHESIS_WORKLOADS.items():
        grouped_workloads = nakb_workloads.get(workload_type)
        if grouped_workloads is None or workload_index >= len(grouped_workloads):
            raise RuntimeError(f"missing grouped synthesis workload {workload_name!r}")
        workloads[workload_name] = grouped_workloads[workload_index]
    return dict(sorted(workloads.items()))


NAKB_WORKLOADS = _discover_workloads()
WORKLOADS = _workload_aliases(NAKB_WORKLOADS)

__all__ = [
    "AccuracySpec",
    "ArgumentAdapter",
    "InputGenerator",
    "InputSpecs",
    "NAKB_WORKLOADS",
    "TorchReference",
    "TorchResult",
    "WORKLOADS",
    "Workload",
    "accuracy_validation",
    "validate_nakb_outputs",
]
