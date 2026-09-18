"""Synthesized kernel artifacts and validation helpers."""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from nkigym.profile import InputSpecs

ArrayResult = np.ndarray | tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SynthesizedKernel:
    """One synthesized callable plus its normalized validation contract."""

    source: str
    function: Callable[..., ArrayResult]
    input_specs: InputSpecs
    adapt_inputs: Callable[[dict[str, object]], dict[str, np.ndarray]]
    adapt_output: Callable[[object], ArrayResult]


def _exec_nkigym_source(source: str) -> Callable[..., ArrayResult]:
    """Execute generated source and return its decorated function."""
    namespace: dict[str, Any] = {"__name__": "__nkigym_generated__"}
    filename = f"<nkigym-synthesis-{hashlib.sha256(source.encode()).hexdigest()[:16]}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    if not callable(function := namespace.get("f_nkigym")) or not getattr(function, "__nkigym_kernel__", False):
        raise ValueError("generated source did not define a decorated f_nkigym")
    return cast(Callable[..., ArrayResult], function)


def _results_match(actual: ArrayResult, expected: ArrayResult, selection_input: np.ndarray | None = None) -> bool:
    """Compare one single- or multiple-output result."""
    actual_arrays = actual if isinstance(actual, tuple) else (actual,)
    expected_arrays = expected if isinstance(expected, tuple) else (expected,)
    if selection_input is not None:
        indices = actual_arrays[-1]
        valid = len(actual_arrays) == 2 and indices.dtype.kind in "iu"
        if not valid or np.any(indices < 0) or np.any(indices >= selection_input.shape[-1]):
            return False
        selected = np.take_along_axis(selection_input, indices.astype(np.int64), axis=-1)
        actual_arrays = (np.sort(actual_arrays[0], axis=-1), np.sort(selected, axis=-1))
        expected_arrays = (np.sort(expected_arrays[0], axis=-1),) * 2
    return len(actual_arrays) == len(expected_arrays) and all(
        left.shape == right.shape and np.allclose(left, right, atol=5e-3, rtol=5e-3)
        for left, right in zip(actual_arrays, expected_arrays, strict=True)
    )


__all__ = ["SynthesizedKernel"]
