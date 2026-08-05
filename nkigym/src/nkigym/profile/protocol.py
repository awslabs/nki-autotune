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
    }


def parse_request(payload: object) -> ProfileRequest:
    """Validate one request decoded from JSON."""
    if not isinstance(payload, dict):
        raise ValueError("profile request must be a JSON object")
    func_name = _required_string(payload, "func_name")
    input_specs = _parse_input_specs(payload.get("input_specs"))
    neuronx_cc_args = _string_tuple(payload.get("neuronx_cc_args"), "neuronx_cc_args")
    lnc = _required_integer(payload, "lnc")
    config = ProfileConfig(input_specs=input_specs, neuronx_cc_args=neuronx_cc_args, lnc=lnc)
    return ProfileRequest(func_name=func_name, config=config)


def result_payload(result: ProfileResult) -> dict[str, object]:
    """Serialize one worker result to JSON-compatible values."""
    return {"profiler_summary": result.profiler_summary, "error": result.error, "elapsed_s": result.elapsed_s}


def parse_result(payload: object) -> ProfileResult:
    """Validate one worker result decoded from JSON."""
    if not isinstance(payload, dict):
        raise ValueError("profile result must be a JSON object")
    summary = payload.get("profiler_summary")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError("profiler_summary must be an object or null")
    error = payload.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("error must be a string or null")
    elapsed_s = payload.get("elapsed_s")
    if not isinstance(elapsed_s, (int, float)) or isinstance(elapsed_s, bool):
        raise ValueError("elapsed_s must be numeric")
    typed_summary = dict(summary) if summary is not None else None
    return ProfileResult(profiler_summary=typed_summary, error=error, elapsed_s=float(elapsed_s))


def _required_string(payload: dict[object, object], field: str) -> str:
    """Read one required non-empty string."""
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_integer(payload: dict[object, object], field: str) -> int:
    """Read one required integer while rejecting JSON booleans."""
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer_tuple(value: object, field: str) -> tuple[int, ...]:
    """Validate a non-empty array of positive integers."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array of positive integers")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ValueError(f"{field} must be a non-empty array of positive integers")
    return tuple(value)


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
        shape = _positive_integer_tuple(raw_spec.get("shape"), f"input_specs.{name}.shape")
        dtype = _required_string(raw_spec, "dtype")
        specs[name] = (shape, dtype)
    return specs


__all__ = ["parse_request", "parse_result", "request_payload", "result_payload"]
