"""Types shared by the SSH profile client and the Trn2 host worker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

InputSpecs = dict[str, tuple[tuple[int, ...], str]]


@dataclass(frozen=True)
class ProfileConfig:
    """Static workload configuration for one kernel profile."""

    input_specs: InputSpecs
    neuronx_cc_args: tuple[str, ...]
    lnc: int
    confirmation: bool = False

    def __post_init__(self) -> None:
        """Validate dimensions and hardware controls at the API boundary."""
        if not self.input_specs:
            raise ValueError("input_specs must not be empty")
        for name, (shape, dtype) in self.input_specs.items():
            if not name.isidentifier() or not shape or any(dimension <= 0 for dimension in shape) or not dtype:
                raise ValueError(f"invalid input specification {name!r}: shape={shape!r}, dtype={dtype!r}")
        if any(not isinstance(argument, str) for argument in self.neuronx_cc_args):
            raise ValueError("neuronx_cc_args must contain only strings")
        if self.lnc not in {1, 2} or not isinstance(self.confirmation, bool):
            raise ValueError("invalid profile execution controls")


@dataclass(frozen=True)
class ProfileRequest:
    """Validated request consumed by the installed Trn2 worker."""

    func_name: str
    config: ProfileConfig

    def __post_init__(self) -> None:
        """Validate the function identifier received from the client."""
        if not self.func_name.isidentifier():
            raise ValueError(f"invalid kernel function name {self.func_name!r}")


@dataclass(frozen=True)
class ProfileResult:
    """Single-kernel result returned by the installed Trn2 worker."""

    profiler_summary: dict[str, object] | None
    error: str | None
    elapsed_s: float
    compile_s: float
    profile_s: float

    def __post_init__(self) -> None:
        """Validate result state and worker timings."""
        if (self.profiler_summary is None) == (self.error is None) or min(
            self.elapsed_s, self.compile_s, self.profile_s
        ) < 0:
            raise ValueError("profile result must contain one summary/error and non-negative timings")


@dataclass(frozen=True)
class ProfileMetrics:
    """Core metrics and the raw Neuron Explorer summary."""

    mfu_percent: float
    latency_ms: float
    profiler_summary: dict[str, object]
    outputs: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid core measurements and copy the raw summary."""
        for name, value in (("mfu_percent", self.mfu_percent), ("latency_ms", self.latency_ms)):
            numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"profile metric {name!r} must be finite and non-negative")
        object.__setattr__(self, "profiler_summary", dict(self.profiler_summary))
        object.__setattr__(self, "outputs", tuple(output.copy() for output in self.outputs))

    def as_dict(self) -> dict[str, float]:
        """Return the stable core measurements."""
        return {"mfu_percent": self.mfu_percent, "latency_ms": self.latency_ms}
