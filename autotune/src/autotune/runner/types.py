"""Shared types, constants, and utilities for the autotune runner.

Safe to import on both coordinator and worker machines — no nki or
nkipy dependencies.
"""

import logging
import os
import sys
import traceback
from typing import NamedTuple

import ml_dtypes
import numpy as np

logger = logging.getLogger(__name__)

_PE_FREQ_HZ = 2.4e9
_TRN2_FLOPS_PER_CYCLE: dict[str, int] = {
    "float8_e4m3fn": 4 * 128 * 128,
    "float8_e5m2": 4 * 128 * 128,
    "float16": 2 * 128 * 128,
    "bfloat16": 2 * 128 * 128,
    "float32": 2 * 128 * 128,
}
_BF16_FLOPS_PER_CYCLE = 2 * 128 * 128

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
        input_specs: Map of param name to (shape, dtype_str).
        golden_source: Source code of the golden numpy reference function.
        golden_func_name: Name of the golden function.
        atol: Absolute tolerance for CPU sim vs golden comparison.
        rtol: Relative tolerance for CPU sim vs golden comparison.
    """

    source: str
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    golden_source: str
    golden_func_name: str
    atol: float
    rtol: float


class CompileResult(NamedTuple):
    """Result of compiling a single NKI kernel to NEFF."""

    kernel_name: str
    nki_path: str
    neff_path: str
    error: str


class ProfileResult(NamedTuple):
    """Benchmark result for a single kernel."""

    kernel_name: str
    min_ms: float
    mean_ms: float
    p50_ms: float
    p99_ms: float
    mac_count: int
    mfu: float
    cpu_sim: dict
    hardware_output: str


class BenchmarkConfig(NamedTuple):
    """Grouping of benchmark parameters passed through the pipeline."""

    warmup: int
    iters: int
    mac_count: int
    input_dtype_name: str


class OutputSpec(NamedTuple):
    """Output tensor specification for a compiled kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


def capture_error(exc: Exception) -> str:
    """Capture the full traceback from an exception as a string."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sim_not_run() -> dict:
    """Create a fresh 'not run' CPU simulation status."""
    return {"passed": False, "error": "not run"}


def make_failure(kernel_name: str, hardware_output: str, mac_count: int, cpu_sim: dict | None = None) -> ProfileResult:
    """Create a failed ProfileResult."""
    return ProfileResult(
        kernel_name=kernel_name,
        min_ms=0.0,
        mean_ms=0.0,
        p50_ms=0.0,
        p99_ms=0.0,
        mac_count=mac_count,
        mfu=0.0,
        cpu_sim=cpu_sim if cpu_sim is not None else _sim_not_run(),
        hardware_output=hardware_output,
    )


def compile_failure_result(cr: CompileResult, mac_count: int, cpu_sim: dict | None = None) -> ProfileResult:
    """Convert a failed CompileResult into a ProfileResult."""
    return make_failure(cr.kernel_name, cr.error, mac_count, cpu_sim=cpu_sim)


def tensor_inputs(kwargs: dict) -> list[np.ndarray]:
    """Extract tensor (ndim > 0) values from kwargs, preserving order."""
    return [v for v in kwargs.values() if hasattr(v, "ndim") and v.ndim > 0]


def calculate_mfu(mac_count: int, time_ms: float, dtype_name: str) -> float:
    """Calculate MFU percentage for trn2 NeuronCore-v3 TensorEngine.

    Returns:
        MFU as a percentage (e.g. 24.0 means 24% utilization).
    """
    if dtype_name not in _TRN2_FLOPS_PER_CYCLE:
        logger.warning("Unknown dtype %r for MFU; using BF16 peak", dtype_name)
    flops_per_cycle = _TRN2_FLOPS_PER_CYCLE.get(dtype_name, _BF16_FLOPS_PER_CYCLE)
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / flops_per_cycle
    return 100.0 * theoretical_pe_cycles / actual_pe_cycles


def percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list via nearest-rank."""
    idx = max(0, min(int(pct / 100 * len(sorted_vals)), len(sorted_vals) - 1))
    return sorted_vals[idx]


_DEFAULT_VENV_PYTHON = "/home/ubuntu/venvs/kernel-env/bin/python"


def ensure_venv_on_path() -> None:
    """Add the current venv bin directory to PATH if not already present."""
    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + path


class ProfileConfig(NamedTuple):
    """Infrastructure configuration for remote profiling.

    Per-kernel settings (golden, tolerances, input_specs) are in KernelJob.
    This bundles infra params that rarely change.
    """

    seed: int = 42
    neuron_platform_target: str = "trn2"
    venv_python: str = _DEFAULT_VENV_PYTHON
