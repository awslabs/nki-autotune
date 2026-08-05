"""Purely programmatic NumPy-to-nkigym synthesis."""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from nkigym.ir import build_initial_ir
from nkigym.synthesis._lower import lower_to_source
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


def compile_numpy_to_nkigym(f_numpy: Callable[..., np.ndarray], input_specs: InputSpecs, seed: int = 0) -> str:
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
    _validate_inputs(f_numpy, input_specs)
    expression = trace_numpy(f_numpy, input_specs)
    source = lower_to_source(expression, input_specs)
    validation = _run_validation(source, f_numpy, input_specs, seed)
    if not validation["passed"]:
        raise RuntimeError(f"programmatic synthesis validation failed: {validation}")
    return source


def _validate_inputs(f_numpy: Callable[..., np.ndarray], input_specs: InputSpecs) -> None:
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


def _exec_nkigym_source(source: str) -> Callable[..., np.ndarray]:
    """Execute generated source and return its decorated function."""
    namespace: dict[str, Any] = {"__name__": "__nkigym_generated__"}
    filename = "<nkigym-synthesis>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    function = namespace.get("f_nkigym")
    if not callable(function) or not getattr(function, "__nkigym_kernel__", False):
        raise ValueError("generated source did not define a decorated f_nkigym")
    return cast(Callable[..., np.ndarray], function)


def _run_validation(
    source: str, f_numpy: Callable[..., np.ndarray], input_specs: InputSpecs, seed: int
) -> dict[str, bool | float | str | None]:
    """Run one deterministic fp32 numerical validation."""
    try:
        f_nkigym = _exec_nkigym_source(source)
        _ = build_initial_ir(f_nkigym, input_specs)
        rng = np.random.default_rng(seed)
        inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}
        expected = np.asarray(f_numpy(**{name: value.copy() for name, value in inputs.items()}))
        actual = np.asarray(f_nkigym(**{name: value.copy() for name, value in inputs.items()}))
        if actual.shape != expected.shape:
            result: dict[str, bool | float | str | None] = {
                "passed": False,
                "error": f"output shape mismatch: actual {actual.shape} vs expected {expected.shape}",
                "max_abs_diff": None,
                "max_rel_diff": None,
            }
        else:
            absolute = np.abs(actual - expected)
            max_abs = float(absolute.max())
            max_rel = float((absolute / (np.abs(expected) + _VALIDATION_ATOL)).max())
            passed = bool(np.allclose(actual, expected, atol=_VALIDATION_ATOL, rtol=_VALIDATION_RTOL))
            result = {
                "passed": passed,
                "error": None if passed else f"fp32 mismatch at atol={_VALIDATION_ATOL} rtol={_VALIDATION_RTOL}",
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
            }
    except Exception as error:
        result = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    return result


__all__ = ["compile_numpy_to_nkigym"]
