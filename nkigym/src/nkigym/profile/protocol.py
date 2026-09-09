"""JSON protocol for one local-to-Trn2 kernel profile request."""

from __future__ import annotations

from nkigym.profile.types import ProfileConfig, ProfileRequest, ProfileResult


def request_payload(func_name: str, config: ProfileConfig) -> dict[str, object]:
    """Serialize one kernel's workload metadata for the host worker."""
    input_specs = {name: {"shape": list(shape), "dtype": dtype} for name, (shape, dtype) in config.input_specs.items()}
    return {
        "func_name": func_name,
        "input_specs": input_specs,
        "neuronx_cc_args": list(config.neuronx_cc_args),
        "lnc": config.lnc,
        "confirmation": config.confirmation,
    }


def parse_request(payload: object) -> ProfileRequest:
    """Validate one request decoded from JSON."""
    if not isinstance(payload, dict):
        raise ValueError("profile request must be a JSON object")
    config = ProfileConfig(
        input_specs=_parse_input_specs(payload.get("input_specs")),
        neuronx_cc_args=_string_tuple(payload.get("neuronx_cc_args"), "neuronx_cc_args"),
        lnc=_required_integer(payload, "lnc"),
        confirmation=_required_boolean(payload, "confirmation"),
    )
    return ProfileRequest(func_name=_required_string(payload, "func_name"), config=config)


def parse_result(payload: object) -> ProfileResult:
    """Validate one worker result decoded from JSON."""
    if not isinstance(payload, dict):
        raise ValueError("profile result must be a JSON object")
    summary, error = payload.get("profiler_summary"), payload.get("error")
    if (summary is not None and not isinstance(summary, dict)) or (error is not None and not isinstance(error, str)):
        raise ValueError("profile summary and error must be an object/null and string/null")
    return ProfileResult(
        profiler_summary=dict(summary) if summary is not None else None,
        error=error,
        elapsed_s=_non_negative_float(payload.get("elapsed_s"), "elapsed_s"),
        compile_s=_non_negative_float(payload.get("compile_s", 0.0), "compile_s"),
        profile_s=_non_negative_float(payload.get("profile_s", 0.0), "profile_s"),
    )


def _required_string(payload: dict[object, object], field: str) -> str:
    """Read one required non-empty string."""
    if not isinstance(value := payload.get(field), str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_integer(payload: dict[object, object], field: str) -> int:
    """Read one required integer while rejecting JSON booleans."""
    if not isinstance(value := payload.get(field), int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_boolean(payload: dict[object, object], field: str) -> bool:
    """Read one required JSON boolean."""
    if not isinstance(value := payload.get(field), bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _non_negative_float(value: object, field: str) -> float:
    """Validate one non-negative numeric field."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative and numeric")
    return float(value)


def _positive_integer_tuple(value: object, field: str) -> tuple[int, ...]:
    """Validate a non-empty array of positive integers."""
    if not (items := value if isinstance(value, list) else []) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in items
    ):
        raise ValueError(f"{field} must be a non-empty array of positive integers")
    return tuple(items)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Validate an array of strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(value)


def _parse_input_specs(value: object) -> dict[str, tuple[tuple[int, ...], str]]:
    """Validate serialized input tensor specifications."""
    if not isinstance(value, dict) or not value:
        raise ValueError("input_specs must be a non-empty object")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw_spec in value.items():
        if not isinstance(name, str) or not isinstance(raw_spec, dict):
            raise ValueError("input_specs must map names to objects")
        specs[name] = (
            _positive_integer_tuple(raw_spec.get("shape"), f"input_specs.{name}.shape"),
            _required_string(raw_spec, "dtype"),
        )
    return specs
