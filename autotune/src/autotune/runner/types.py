"""Shared types, constants, and utilities for the autotune runner.

Safe to import on both coordinator and worker machines — no nki or
nkipy dependencies.
"""

import os
import sys
import traceback
from typing import NamedTuple

import ml_dtypes
import numpy as np

_DTYPE_CACHE: dict[str, np.dtype] = {}


def resolve_dtype(name: str) -> np.dtype:
    """Resolve a dtype string, handling bfloat16 via ml_dtypes.

    Standard numpy does not support bfloat16. Uses ml_dtypes when
    needed and caches results.
    """
    if name not in _DTYPE_CACHE:
        try:
            dt = np.dtype(name)
        except TypeError:
            dt = np.dtype(getattr(ml_dtypes, name))
        _DTYPE_CACHE[name] = dt
    return _DTYPE_CACHE[name]


class KernelJob(NamedTuple):
    """Per-kernel configuration for remote profiling.

    Attributes:
        source: NKI kernel source code string.
        func_name: Name of the ``@nki.jit`` function inside ``source``.
        output_shape: Shape of the kernel's HBM output tensor. Supplied
            by the caller (traced once on the coordinator for
            nkigym-generated kernels, hardcoded for reference kernels)
            to avoid unreliable AST parsing on the worker.
        input_specs: Map of param name to (shape, dtype_str).
        neuronx_cc_args: Extra flags forwarded to neuronx-cc via
            ``CompileOptions.set_pipeline_options(*args)``. Empty for
            nkigym-generated kernels; hand-allocated reference kernels
            (e.g. nkilib's ``attention_cte``) typically need
            ``("enable-linear-scan-allocation=false",
            "enable-instruction-scheduling=false")`` — equivalent to
            nkilib's ``disable_backend_optimizations()``.
        lnc: Logical NeuronCore count (1 or 2).
    """

    source: str
    func_name: str
    output_shape: tuple[int, ...]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    neuronx_cc_args: tuple[str, ...] = ()
    lnc: int = 1


class CompileResult(NamedTuple):
    """Result of compiling a single NKI kernel to NEFF."""

    kernel_name: str
    nki_path: str
    neff_path: str
    error: str


class ProfileResult(NamedTuple):
    """Benchmark result for a single kernel.

    ``profiler_summary`` is the raw ``neuron-profile view
    --output-format summary-json`` dict for the post-compiler NTFF
    trace. Every number we report — wall-clock time, MFU/MBU/ceiling,
    engine-active times, cycle counts, DMA bytes — lives there.

    ``profile_detailed``, ``neff_b64``, and ``ntff_b64`` are collected
    only when the caller passes ``collect_detailed_profile=True`` to
    :func:`remote_profile`:

    * ``profile_detailed`` is the full ``--output-format json`` dict
      (per-instruction trace, per-engine active intervals, per-layer
      summaries, DMA throughput history).
    * ``neff_b64`` is the base64-encoded compiled NEFF binary.
    * ``ntff_b64`` is the base64-encoded NTFF trace. Callers can decode
      and feed these back into ``neuron-profile view`` offline.

    All optional fields are ``None`` when compile or hardware execution
    failed, or when detailed collection is off.
    """

    kernel_name: str
    hardware_output: str
    profiler_summary: dict | None = None
    profile_detailed: dict | None = None
    neff_b64: str | None = None
    ntff_b64: str | None = None


class OutputSpec(NamedTuple):
    """Output tensor specification for a compiled kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


def capture_error(exc: Exception) -> str:
    """Capture the full traceback from an exception as a string."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def make_failure(kernel_name: str, hardware_output: str) -> ProfileResult:
    """Create a failed ProfileResult with a null ``profiler_summary``."""
    return ProfileResult(kernel_name=kernel_name, hardware_output=hardware_output)


def compile_failure_result(cr: CompileResult) -> ProfileResult:
    """Convert a failed CompileResult into a ProfileResult."""
    return make_failure(cr.kernel_name, cr.error)


def tensor_inputs(kwargs: dict) -> list[np.ndarray]:
    """Extract tensor (ndim > 0) values from kwargs, preserving order."""
    return [v for v in kwargs.values() if hasattr(v, "ndim") and v.ndim > 0]


def profiler_percent(summary: dict | None, field: str) -> float | None:
    """Read a profiler summary-json fractional field and scale to percent.

    ``neuron-profile`` reports utilization fields as fractions in [0, 1]
    (or a negative sentinel when unavailable). Returns ``None`` when the
    summary is missing the field or the value is a sentinel.
    """
    result: float | None = None
    if summary is not None:
        value = summary.get(field)
        if value is not None:
            scaled = float(value) * 100.0
            if scaled >= 0:
                result = scaled
    return result


_DEFAULT_VENV_PYTHON = "/home/ubuntu/venvs/kernel-env/bin/python"


def ensure_venv_on_path() -> None:
    """Add the current venv bin directory to PATH if not already present."""
    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + path
