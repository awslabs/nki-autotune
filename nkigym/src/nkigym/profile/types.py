"""Types shared by the SSH profile client and the Trn2 host worker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileConfig:
    """Static workload configuration for one kernel profile."""

    input_specs: dict[str, tuple[tuple[int, ...], str]]
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
class ProfileResult:
    """Single-kernel result returned by the installed Trn2 worker."""

    profiler_summary: dict[str, object] | None
    error: str | None
    elapsed_s: float

    def __post_init__(self) -> None:
        """Validate worker timing."""
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must not be negative")


__all__ = ["ProfileConfig", "ProfileRequest", "ProfileResult"]
