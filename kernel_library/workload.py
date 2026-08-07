"""Shared metadata for one retained kernel workload."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.environment import Action

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
InputGenerator = Callable[[InputSpecs, int], dict[str, np.ndarray]]


def _validate_input_specs(input_specs: InputSpecs) -> None:
    """Reject malformed workload input specifications."""
    if not input_specs:
        raise ValueError("input_specs must not be empty")
    for name, (shape, dtype) in input_specs.items():
        if not name.isidentifier():
            raise ValueError(f"input name must be a Python identifier: {name!r}")
        if not shape or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape
        ):
            raise ValueError(f"input {name!r} must have a non-empty positive shape")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"input {name!r} must have a dtype")


def _validate_parameters(name: str, function: Callable[..., np.ndarray], input_specs: InputSpecs) -> None:
    """Require one callable parameter per input in declaration order."""
    parameters = list(inspect.signature(function).parameters)
    if parameters != list(input_specs):
        raise ValueError(f"{name} parameters {parameters} do not match input_specs keys {list(input_specs)}")


def _validate_input_generator(input_generator: InputGenerator) -> None:
    """Require the shared input-generator calling convention."""
    parameters = list(inspect.signature(input_generator).parameters)
    if parameters != ["input_specs", "seed"]:
        raise ValueError(f"input_generator parameters {parameters} must be ['input_specs', 'seed']")


def _validate_tolerance(name: str, value: float) -> None:
    """Require a finite non-negative comparison tolerance."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_optional_percentage(name: str, value: float | None) -> None:
    """Require an optional finite percentage."""
    if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0.0 or value > 100.0):
        raise ValueError(f"{name} must be a finite percentage in [0, 100] or None")


@dataclass(frozen=True)
class Workload:
    """Define one workload and its best-known optimization result.

    Attributes:
        input_specs: Input names mapped to shape and dtype.
        f_numpy: NumPy reference implementation.
        f_nkigym: Canonical nkigym operator graph.
        input_generator: Seeded FP32 input generation for CPU validation.
        atol: Absolute tolerance for CPU validation.
        rtol: Relative tolerance for CPU validation.
        best_action_ladder: Ordered actions for the best retained schedule.
        historical_best_mfu: Highest measured MFU percentage, or ``None`` before profiling.
        reference_mfu: Best manually achieved Kaena MFU percentage for the same workload and shape.
    """

    input_specs: InputSpecs
    f_numpy: Callable[..., np.ndarray]
    f_nkigym: Callable[..., np.ndarray]
    input_generator: InputGenerator
    atol: float
    rtol: float
    best_action_ladder: tuple[Action, ...] = ()
    historical_best_mfu: float | None = None
    reference_mfu: float | None = None

    def __post_init__(self) -> None:
        """Validate the workload contract at its declaration site."""
        _validate_input_specs(self.input_specs)
        _validate_parameters("f_numpy", self.f_numpy, self.input_specs)
        _validate_parameters("f_nkigym", self.f_nkigym, self.input_specs)
        _validate_input_generator(self.input_generator)
        _validate_tolerance("atol", self.atol)
        _validate_tolerance("rtol", self.rtol)
        if not getattr(self.f_nkigym, "__nkigym_kernel__", False):
            raise ValueError("f_nkigym must be decorated with @nkigym_kernel")
        if not isinstance(self.best_action_ladder, tuple):
            raise ValueError("best_action_ladder must be a tuple")
        _validate_optional_percentage("historical_best_mfu", self.historical_best_mfu)
        _validate_optional_percentage("reference_mfu", self.reference_mfu)

    def generate_inputs(self, seed: int, input_specs: InputSpecs | None = None) -> dict[str, np.ndarray]:
        """Generate and validate replayable FP32 inputs."""
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        selected_specs = self.input_specs if input_specs is None else input_specs
        _validate_input_specs(selected_specs)
        inputs = self.input_generator(selected_specs, seed)
        if list(inputs) != list(selected_specs):
            raise ValueError(
                f"generated input keys {list(inputs)} do not match input_specs keys {list(selected_specs)}"
            )
        for name, (shape, _dtype) in selected_specs.items():
            value = inputs[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"generated input {name!r} must be a NumPy array")
            if value.shape != shape:
                raise ValueError(f"generated input {name!r} has shape {value.shape}, expected {shape}")
            if value.dtype != np.float32:
                raise ValueError(f"generated input {name!r} has dtype {value.dtype}, expected float32")
        return inputs


__all__ = ["InputGenerator", "InputSpecs", "Workload"]
