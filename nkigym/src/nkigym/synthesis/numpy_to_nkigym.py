"""Purely programmatic NumPy-to-nkigym synthesis."""

from __future__ import annotations

import hashlib
import inspect
import linecache
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from nkigym.ir import build_initial_ir
from nkigym.synthesis._lower import lower_to_source
from nkigym.synthesis._specialized import ArrayResult, lower_specialized_reference
from nkigym.synthesis._trace import trace_numpy

_VALIDATION_ATOL = 5e-3
_VALIDATION_RTOL = 5e-3
_FLOATING_DTYPES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float8_e4m3",
        "float8_e4m3fn",
        "float8_e4m3fn_x4",
        "float8_e5m2",
        "float8_e5m2_x4",
        "float4_e2m1fn_x4",
        "tfloat32",
    }
)

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
InputAdapter = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
OutputAdapter = Callable[[ArrayResult], ArrayResult]


@dataclass(frozen=True)
class SynthesizedKernel:
    """One synthesized callable plus its normalized validation contract."""

    source: str
    function: Callable[..., ArrayResult]
    input_specs: InputSpecs
    adapt_inputs: InputAdapter
    adapt_output: OutputAdapter


def compile_numpy_to_nkigym(f_numpy: Callable[..., ArrayResult], input_specs: InputSpecs, seed: int = 0) -> str:
    """Translate supported NumPy math into a deterministic ``f_nkigym``.

    The compiler executes ``f_numpy`` once with shape-only symbolic tensors,
    lowers the resulting expression DAG to existing ``NKIOp`` primitives,
    and checks the generated graph numerically at fp32. It performs no model
    calls, source-code prompting, search, or retry loop.

    Supported operations are floating-point casts, rank-two transpose and
    matmul, scalar and per-row broadcast arithmetic, common unary activations,
    and sum, maximum, or mean reduction over the free axis.

    Args:
        f_numpy: NumPy reference with parameters matching ``input_specs``.
        input_specs: Input names mapped to ``(shape, dtype)``.
        seed: Seed for deterministic fp32 validation inputs.

    Returns:
        Complete Python source defining a decorated ``f_nkigym``.

    Raises:
        ValueError: The function contract or NumPy program is unsupported.
        RuntimeError: The deterministic lowering fails numerical validation.
    """
    return synthesize_numpy_to_nkigym(f_numpy, input_specs, seed).source


def synthesize_numpy_to_nkigym(
    f_numpy: Callable[..., ArrayResult], input_specs: InputSpecs, seed: int = 0
) -> SynthesizedKernel:
    """Synthesize a callable and any ABI adapters required by its lowering."""
    specialized = lower_specialized_reference(f_numpy, input_specs)
    if specialized is None:
        _validate_inputs(f_numpy, input_specs)
        expression = trace_numpy(cast(Callable[..., np.ndarray], f_numpy), input_specs)
        source = lower_to_source(expression, input_specs)
        function = _exec_nkigym_source(source)
        artifact = SynthesizedKernel(
            source=source,
            function=function,
            input_specs=input_specs,
            adapt_inputs=lambda inputs: inputs,
            adapt_output=lambda result: result,
        )
        reference_inputs = _random_inputs(input_specs, seed)
    else:
        source = specialized.source
        function = _exec_nkigym_source(source)
        artifact = SynthesizedKernel(
            source=source,
            function=function,
            input_specs=specialized.input_specs,
            adapt_inputs=specialized.adapt_inputs,
            adapt_output=specialized.adapt_output,
        )
        reference_inputs = specialized.validation_inputs(seed)
    validation = _run_artifact_validation(artifact, f_numpy, reference_inputs)
    if not validation["passed"]:
        raise RuntimeError(f"programmatic synthesis validation failed: {validation}")
    return artifact


def _validate_inputs(f_numpy: Callable[..., ArrayResult], input_specs: InputSpecs) -> None:
    """Validate the callable signature and static tensor specifications."""
    if not callable(f_numpy):
        raise ValueError("f_numpy must be callable")
    if not input_specs:
        raise ValueError("input_specs must not be empty")
    parameters = list(inspect.signature(f_numpy).parameters)
    if parameters != list(input_specs):
        raise ValueError(f"f_numpy params {parameters} != input_specs keys {list(input_specs)}")
    for name, (shape, dtype) in input_specs.items():
        if not name.isidentifier():
            raise ValueError(f"input name must be a Python identifier: {name!r}")
        if len(shape) != 2:
            raise ValueError(f"input {name!r} must be rank two, got shape {shape}")
        if any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape):
            raise ValueError(f"input {name!r} must have positive integer dimensions")
        if dtype not in _FLOATING_DTYPES:
            raise ValueError(f"input {name!r} has unsupported NKI floating dtype {dtype!r}")


def _exec_nkigym_source(source: str) -> Callable[..., ArrayResult]:
    """Execute generated source and return its decorated function."""
    namespace: dict[str, Any] = {"__name__": "__nkigym_generated__"}
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    filename = f"<nkigym-synthesis-{digest}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    function = namespace.get("f_nkigym")
    if not callable(function) or not getattr(function, "__nkigym_kernel__", False):
        raise ValueError("generated source did not define a decorated f_nkigym")
    return cast(Callable[..., ArrayResult], function)


def _run_artifact_validation(
    artifact: SynthesizedKernel, f_numpy: Callable[..., ArrayResult], reference_inputs: dict[str, np.ndarray]
) -> dict[str, bool | float | str | None]:
    """Validate one synthesized artifact against its unchanged reference ABI."""
    try:
        _ = build_initial_ir(artifact.function, artifact.input_specs)
        reference_copy = {name: value.copy() for name, value in reference_inputs.items()}
        kernel_inputs = artifact.adapt_inputs(reference_inputs)
        expected = artifact.adapt_output(f_numpy(**reference_copy))
        actual = artifact.function(**{name: value.copy() for name, value in kernel_inputs.items()})
        result = _compare_results(actual, expected)
    except Exception as error:
        result = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    return result


def _compare_results(actual: ArrayResult, expected: ArrayResult) -> dict[str, bool | float | str | None]:
    """Compare one single- or multiple-output result."""
    actual_arrays = actual if isinstance(actual, tuple) else (actual,)
    expected_arrays = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_arrays) != len(expected_arrays):
        return {
            "passed": False,
            "error": f"output count mismatch: actual {len(actual_arrays)} vs expected {len(expected_arrays)}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    maximum_absolute = 0.0
    maximum_relative = 0.0
    passed = True
    error_message: str | None = None
    for actual_array, expected_array in zip(actual_arrays, expected_arrays, strict=True):
        actual_value = np.asarray(actual_array)
        expected_value = np.asarray(expected_array)
        if actual_value.shape != expected_value.shape:
            passed = False
            error_message = f"output shape mismatch: actual {actual_value.shape} vs expected {expected_value.shape}"
            break
        absolute = np.abs(actual_value - expected_value)
        maximum_absolute = max(maximum_absolute, float(absolute.max(initial=0.0)))
        relative = absolute / (np.abs(expected_value) + _VALIDATION_ATOL)
        maximum_relative = max(maximum_relative, float(relative.max(initial=0.0)))
        passed = passed and bool(
            np.allclose(actual_value, expected_value, atol=_VALIDATION_ATOL, rtol=_VALIDATION_RTOL)
        )
    if not passed and error_message is None:
        error_message = f"fp32 mismatch at atol={_VALIDATION_ATOL} rtol={_VALIDATION_RTOL}"
    return {
        "passed": passed,
        "error": error_message,
        "max_abs_diff": maximum_absolute,
        "max_rel_diff": maximum_relative,
    }


def _random_inputs(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic fp32 inputs for the expression lowering."""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


__all__ = ["SynthesizedKernel", "compile_numpy_to_nkigym", "synthesize_numpy_to_nkigym"]
