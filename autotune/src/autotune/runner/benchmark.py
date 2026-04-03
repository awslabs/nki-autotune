"""Hardware benchmarking and CPU simulation for NKI kernels.

Worker-only module — imports nki and nkipy at top level. Not safe to
import on the coordinator machine.
"""

from typing import Any

import nki
import numpy as np
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel

from autotune.runner.compile import load_kernel
from autotune.runner.types import (
    BenchmarkConfig,
    CompileResult,
    OutputSpec,
    ProfileResult,
    calculate_mfu,
    capture_error,
    percentile,
    resolve_dtype,
    tensor_inputs,
)


class TracedKernel:
    """Minimal traced kernel wrapper required by CompiledKernel."""

    def __init__(self, func: Any, code: Any) -> None:
        """Initialize with kernel function and HLO module."""
        self.func = func
        self._code = code

    @property
    def __name__(self) -> str:
        """Return the kernel function name."""
        return self.func.__name__


def create_compiled_kernel(
    neff_path: str, nki_path: str, func_name: str, kernel_kwargs: dict[str, Any], out: OutputSpec
) -> CompiledKernel:
    """Create a CompiledKernel from a NEFF for BaremetalExecutor.

    Args:
        neff_path: Path to the compiled NEFF binary.
        nki_path: Path to the kernel source file.
        func_name: Name of the kernel function.
        kernel_kwargs: Input tensors and scalar params.
        out: Output tensor specification.
    """
    kernel = load_kernel(nki_path, func_name)
    hlo = HLOModule(name=func_name)
    for name, tensor in kernel_kwargs.items():
        if hasattr(tensor, "ndim") and tensor.ndim > 0:
            hlo.add_parameter(tensor.shape, tensor.dtype, name=name)
    hlo.set_results([HLOTensor(shape=out.shape, dtype=out.dtype, name=out.name)])
    return CompiledKernel(TracedKernel(kernel, hlo), neff_path)


def _run_timing(
    spike: BaremetalExecutor, compiled: CompiledKernel, kernel_kwargs: dict[str, Any], config: BenchmarkConfig
) -> tuple[float, float, float, float, float]:
    """Run benchmark timing and return (min_ms, mean_ms, p50, p99, mfu).

    Args:
        spike: Active BaremetalExecutor session.
        compiled: Compiled kernel to benchmark.
        kernel_kwargs: Input tensors.
        config: Benchmark configuration.
    """
    stats = spike.benchmark(
        compiled,
        *tensor_inputs(kernel_kwargs),
        warmup_iterations=config.warmup,
        benchmark_iterations=config.iters,
        mode="device",
    )
    min_ms = stats.min_ms
    mean_ms = stats.mean_ms
    sorted_durations = sorted(stats.durations_ms)
    p50_ms = percentile(sorted_durations, 50)
    p99_ms = percentile(sorted_durations, 99)
    mfu = 0.0
    if config.mac_count > 0 and min_ms > 0:
        mfu = calculate_mfu(config.mac_count, min_ms, config.input_dtype_name)
    return min_ms, mean_ms, p50_ms, p99_ms, mfu


def _check_correctness(
    spike: BaremetalExecutor,
    compiled: CompiledKernel,
    kernel_kwargs: dict[str, Any],
    golden: np.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """Compare hardware output against golden reference.

    Raises:
        AssertionError: If outputs don't match within tolerance.
    """
    run_out = spike.run(compiled, *tensor_inputs(kernel_kwargs))
    if isinstance(run_out, np.ndarray):
        run_out = (run_out,)
    actual = run_out[0]
    if golden.dtype != actual.dtype:
        actual = actual.astype(golden.dtype)
    np.testing.assert_allclose(actual, golden, atol=atol, rtol=rtol)


def benchmark_one(
    spike: BaremetalExecutor,
    cr: CompileResult,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    out: OutputSpec,
    config: BenchmarkConfig,
    golden: np.ndarray,
) -> ProfileResult:
    """Benchmark a single compiled variant on a Neuron core.

    Args:
        spike: Active BaremetalExecutor session.
        cr: Compilation result with NEFF path.
        func_name: Kernel function name.
        kernel_kwargs: Input tensors and scalar params.
        out: Output tensor specification.
        config: Benchmark configuration.
        golden: Reference output. Empty (size 0) to skip correctness check.
    """
    min_ms = mean_ms = p50_ms = p99_ms = mfu = 0.0
    correct = False
    error = ""
    try:
        compiled = create_compiled_kernel(cr.neff_path, cr.nki_path, func_name, kernel_kwargs, out)
        min_ms, mean_ms, p50_ms, p99_ms, mfu = _run_timing(spike, compiled, kernel_kwargs, config)
        if golden.size > 0:
            _check_correctness(spike, compiled, kernel_kwargs, golden, config.atol, config.rtol)
        correct = True
    except Exception as e:
        error = capture_error(e)
    return ProfileResult(
        kernel_name=cr.kernel_name,
        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mac_count=config.mac_count,
        mfu=mfu,
        correct=correct,
        error=error,
    )


def simulate_one(nki_path: str, func_name: str, kernel_kwargs: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Run a kernel through the NKI CPU simulator.

    Args:
        nki_path: Path to the kernel source file.
        func_name: Name of the @nki.jit decorated function.
        kernel_kwargs: Input tensors and scalar params.

    Returns:
        Tuple of (output array, error string). Output is empty on failure.
    """
    output = np.empty(0)
    error = ""
    try:
        kernel_func = load_kernel(nki_path, func_name)
        sim_kwargs: dict[str, Any] = {}
        for k, v in kernel_kwargs.items():
            if hasattr(v, "ndim") and v.ndim > 0:
                sim_kwargs[k] = v.copy()
            else:
                sim_kwargs[k] = v
        result = nki.simulate(kernel_func)(**sim_kwargs)
        output = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        error = capture_error(e)
    return output, error


def generate_tensors(tensor_specs: dict[str, dict], seed: int) -> dict[str, np.ndarray]:
    """Generate random input tensors from shapes, dtypes, and a seed.

    Args:
        tensor_specs: Map of param name to {"shape": [...], "dtype": "..."}.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for name, spec in tensor_specs.items():
        shape = tuple(spec["shape"])
        dtype = resolve_dtype(spec["dtype"])
        tensors[name] = rng.standard_normal(shape).astype(dtype)
    return tensors


def compute_golden(golden_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute a golden reference function to compute expected output.

    Args:
        golden_source: Source code containing the golden function.
        func_name: Name of the golden function.
        kernel_kwargs: Input arrays (will be cast to float64).

    Returns:
        Expected output at float64 precision.
    """
    g: dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    exec(golden_source, g)  # noqa: S102
    func = g[func_name]
    f64_kwargs = {k: v.astype(np.float64) for k, v in kernel_kwargs.items() if hasattr(v, "ndim") and v.ndim > 0}
    return func(**f64_kwargs)
