"""Types shared by the SSH profile client and the Trn2 host worker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

InputSpecs = dict[str, tuple[tuple[int, ...], str]]


def _validate_label(label: str) -> None:
    """Require one safe relative directory name."""
    if not label or label in {".", ".."} or Path(label).name != label:
        raise ValueError(f"invalid profile label {label!r}")


@dataclass(frozen=True)
class ProfileConfig:
    """Static workload configuration for one kernel profile."""

    input_specs: InputSpecs
    neuronx_cc_args: tuple[str, ...]
    lnc: int

    def __post_init__(self) -> None:
        """Validate dimensions and hardware controls at the API boundary."""
        if not self.input_specs:
            raise ValueError("input_specs must not be empty")
        for name, (shape, dtype) in self.input_specs.items():
            if not name.isidentifier():
                raise ValueError(f"input name must be a Python identifier: {name!r}")
            if not shape or any(dimension <= 0 for dimension in shape):
                raise ValueError(f"input {name!r} must have a non-empty positive shape")
            if not dtype:
                raise ValueError(f"input {name!r} must have a dtype")
        if any(not isinstance(argument, str) for argument in self.neuronx_cc_args):
            raise ValueError("neuronx_cc_args must contain only strings")
        if self.lnc not in {1, 2}:
            raise ValueError("lnc must be 1 or 2")


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
class BatchProfileJob:
    """One labeled request consumed by the batch worker."""

    label: str
    request: ProfileRequest

    def __post_init__(self) -> None:
        """Reject labels that could escape the batch artifact directory."""
        _validate_label(self.label)


@dataclass(frozen=True)
class BatchProfileRequest:
    """Validated collection of profile jobs executed on one Trn2 host."""

    jobs: tuple[BatchProfileJob, ...]
    max_workers: int

    def __post_init__(self) -> None:
        """Require unique jobs, positive parallelism, and one LNC mode."""
        if not self.jobs:
            raise ValueError("batch profile jobs must not be empty")
        labels = [job.label for job in self.jobs]
        if len(labels) != len(set(labels)):
            raise ValueError("batch profile labels must be unique")
        if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool) or self.max_workers <= 0:
            raise ValueError("batch profile workers must be a positive integer")
        if len({job.request.config.lnc for job in self.jobs}) != 1:
            raise ValueError("batch profile jobs must use one LNC configuration")


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
        if (self.profiler_summary is None) == (self.error is None):
            raise ValueError("profile result must contain exactly one summary or error")
        if min(self.elapsed_s, self.compile_s, self.profile_s) < 0:
            raise ValueError("profile result timings must not be negative")


@dataclass(frozen=True)
class ProfileMetrics:
    """Core metrics and the raw Neuron Explorer summary."""

    mfu_percent: float
    latency_ms: float
    profiler_summary: dict[str, object]

    def __post_init__(self) -> None:
        """Reject invalid core measurements and copy the raw summary."""
        required = {"mfu_percent": self.mfu_percent, "latency_ms": self.latency_ms}
        for name, value in required.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"profile metric {name!r} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"profile metric {name!r} must be finite and non-negative")
        object.__setattr__(self, "profiler_summary", dict(self.profiler_summary))

    def as_dict(self) -> dict[str, float]:
        """Return the stable core measurements."""
        return {"mfu_percent": self.mfu_percent, "latency_ms": self.latency_ms}


__all__ = [
    "BatchProfileJob",
    "BatchProfileRequest",
    "InputSpecs",
    "ProfileConfig",
    "ProfileMetrics",
    "ProfileRequest",
    "ProfileResult",
]
