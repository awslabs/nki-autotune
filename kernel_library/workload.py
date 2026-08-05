"""Shared metadata for one retained kernel workload."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.environment import Action

InputSpecs = dict[str, tuple[tuple[int, ...], str]]


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


@dataclass(frozen=True)
class Workload:
    """Define one workload and its best-known optimization result.

    Attributes:
        input_specs: Input names mapped to shape and dtype.
        f_numpy: NumPy reference implementation.
        f_nkigym: Canonical nkigym operator graph.
        best_action_ladder: Ordered actions for the best retained schedule.
        historical_best_mfu: Highest measured MFU percentage, or ``None`` before profiling.
    """

    input_specs: InputSpecs
    f_numpy: Callable[..., np.ndarray]
    f_nkigym: Callable[..., np.ndarray]
    best_action_ladder: tuple[Action, ...] = ()
    historical_best_mfu: float | None = None

    def __post_init__(self) -> None:
        """Validate the workload contract at its declaration site."""
        _validate_input_specs(self.input_specs)
        _validate_parameters("f_numpy", self.f_numpy, self.input_specs)
        _validate_parameters("f_nkigym", self.f_nkigym, self.input_specs)
        if not getattr(self.f_nkigym, "__nkigym_kernel__", False):
            raise ValueError("f_nkigym must be decorated with @nkigym_kernel")
        if not isinstance(self.best_action_ladder, tuple):
            raise ValueError("best_action_ladder must be a tuple")
        if self.historical_best_mfu is not None and (
            isinstance(self.historical_best_mfu, bool)
            or not math.isfinite(self.historical_best_mfu)
            or self.historical_best_mfu < 0.0
            or self.historical_best_mfu > 100.0
        ):
            raise ValueError("historical_best_mfu must be a finite percentage in [0, 100] or None")


__all__ = ["InputSpecs", "Workload"]
