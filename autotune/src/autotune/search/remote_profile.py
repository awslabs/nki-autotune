"""Remote worker: read a kernel batch from stdin and profile it on-device."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from autotune.runner.api import profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob
from autotune.search.profile_evaluator import evaluations_from_profile_output

_RESULT_PREFIX = "AUTOTUNE_PROFILE_RESULT="


@dataclass(frozen=True)
class RemoteKernel:
    """One standalone kernel in a remote profile request."""

    source: str
    func_name: str


@dataclass(frozen=True)
class RemoteProfileRequest:
    """Validated batch profile request received over SSH stdin."""

    kernels: dict[str, RemoteKernel]
    output_shape: tuple[int, ...]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    neuronx_cc_args: tuple[str, ...]
    seed: int
    neuron_platform_target: str


def _parse_args() -> argparse.Namespace:
    """Parse the remote cache destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    return parser.parse_args()


def _parse_request(text: str) -> RemoteProfileRequest:
    """Validate JSON input and return the concrete runner request."""
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("profile request must be a JSON object")
    kernels = _kernels(payload.get("kernels"))
    neuron_platform_target = _required_string(payload, "neuron_platform_target")
    output_shape = _integer_tuple(payload.get("output_shape"), "output_shape")
    neuronx_cc_args = _string_tuple(payload.get("neuronx_cc_args"), "neuronx_cc_args")
    seed = payload.get("seed")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    input_specs = _input_specs(payload.get("input_specs"))
    return RemoteProfileRequest(
        kernels=kernels,
        output_shape=output_shape,
        input_specs=input_specs,
        neuronx_cc_args=neuronx_cc_args,
        seed=seed,
        neuron_platform_target=neuron_platform_target,
    )


def _kernels(value: object) -> dict[str, RemoteKernel]:
    """Validate the non-empty labeled kernel mapping."""
    if not isinstance(value, dict) or not value:
        raise ValueError("kernels must be a non-empty object")
    kernels: dict[str, RemoteKernel] = {}
    for label, raw_kernel in value.items():
        if not isinstance(label, str) or not isinstance(raw_kernel, dict):
            raise ValueError("kernels must map string labels to objects")
        source = _required_string(raw_kernel, "source")
        func_name = _required_string(raw_kernel, "func_name")
        kernels[label] = RemoteKernel(source=source, func_name=func_name)
    return kernels


def _required_string(payload: dict[object, object], key: str) -> str:
    """Read one required string field."""
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    """Validate a non-empty JSON integer array."""
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field} must be a non-empty integer array")
    return tuple(value)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Validate a JSON string array."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _input_specs(value: object) -> dict[str, tuple[tuple[int, ...], str]]:
    """Validate serialized ``KernelJob.input_specs``."""
    if not isinstance(value, dict):
        raise ValueError("input_specs must be an object")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not isinstance(spec, list) or len(spec) != 2:
            raise ValueError("each input_specs entry must be name: [shape, dtype]")
        shape = _integer_tuple(spec[0], f"input_specs.{name}.shape")
        dtype = spec[1]
        if not isinstance(dtype, str):
            raise ValueError(f"input_specs.{name}.dtype must be a string")
        specs[name] = (shape, dtype)
    return specs


def _main() -> None:
    """Profile one request batch and emit one machine-readable result line."""
    args = _parse_args()
    request = _parse_request(sys.stdin.read())
    jobs = {
        label: KernelJob(
            source=kernel.source,
            func_name=kernel.func_name,
            output_shape=request.output_shape,
            input_specs=request.input_specs,
            neuronx_cc_args=request.neuronx_cc_args,
        )
        for label, kernel in request.kernels.items()
    }
    output = profile(
        jobs,
        cache_dir=args.cache,
        seed=request.seed,
        neuron_platform_target=request.neuron_platform_target,
        collect_detailed_profile=False,
    )
    result = {"evaluations": _serialize_evaluations(output, tuple(jobs))}
    print(_RESULT_PREFIX + json.dumps(result), flush=True)


def _serialize_evaluations(output: ProfileOutput, labels: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Convert runner rows into labeled JSON-safe evaluation objects."""
    evaluations = evaluations_from_profile_output(output, labels)
    return {
        label: {"score": evaluation.score, "metrics": evaluation.metrics, "message": evaluation.message}
        for label, evaluation in evaluations.items()
    }


if __name__ == "__main__":
    _main()
