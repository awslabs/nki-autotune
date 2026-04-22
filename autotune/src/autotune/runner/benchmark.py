"""Hardware benchmarking and CPU simulation for NKI kernels.

Worker-only module — imports nki and nkipy at top level. Not safe to
import on the coordinator machine.
"""

import re
from pathlib import Path
from typing import Any, cast

import nki
import numpy as np
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel
from spike.spike_model import BenchmarkResult

from autotune.runner.compile import load_kernel
from autotune.runner.types import (
    BenchmarkConfig,
    CompileResult,
    OutputSpec,
    ProfileResult,
    _sim_not_run,
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
    stats = cast(
        BenchmarkResult,
        spike.benchmark(
            compiled,
            *tensor_inputs(kernel_kwargs),
            warmup_iterations=config.warmup,
            benchmark_iterations=config.iters,
            mode="device",
        ),
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


def benchmark_one(
    spike: BaremetalExecutor,
    cr: CompileResult,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    out: OutputSpec,
    config: BenchmarkConfig,
    cpu_sim: dict | None = None,
) -> ProfileResult:
    """Benchmark a single compiled variant on a Neuron core.

    Timing only — no numerical correctness checks. The cpu_sim field
    carries the CPU simulation result from earlier in the pipeline.

    Args:
        spike: Active BaremetalExecutor session.
        cr: Compilation result with NEFF path.
        func_name: Kernel function name.
        kernel_kwargs: Input tensors.
        out: Output tensor specification.
        config: Benchmark configuration.
        cpu_sim: CPU simulation status dict.
    """
    min_ms: float | None = None
    mean_ms: float | None = None
    p50_ms: float | None = None
    p99_ms: float | None = None
    mfu: float | None = None
    hardware_output = f"{list(out.shape)} {out.dtype}"
    try:
        compiled = create_compiled_kernel(cr.neff_path, cr.nki_path, func_name, kernel_kwargs, out)
        min_ms, mean_ms, p50_ms, p99_ms, mfu = _run_timing(spike, compiled, kernel_kwargs, config)
    except Exception as e:
        hardware_output = capture_error(e)
        min_ms = mean_ms = p50_ms = p99_ms = mfu = None
    return ProfileResult(
        kernel_name=cr.kernel_name,
        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mac_count=config.mac_count,
        mfu=mfu,
        cpu_sim=cpu_sim if cpu_sim is not None else _sim_not_run(),
        hardware_output=hardware_output,
    )


def simulate_one(nki_path: str, func_name: str, kernel_kwargs: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Run a kernel through the NKI CPU simulator in float32 end-to-end.

    Everything runs at float32: inputs are cast, and the kernel source
    is rewritten to a sibling ``.sim.py`` so every ``dtype=nl.<name>``
    token declares ``nl.float32``. This keeps the numerical check
    apples-to-apples with ``compute_golden()`` and bypasses
    ``nisa.dma_transpose``'s strict dtype assert — the hardware path
    still reads ``nki_path`` with the user's declared dtypes.

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
        sim_path = _rewrite_to_fp32(nki_path)
        kernel_func = load_kernel(sim_path, func_name)
        sim_kwargs: dict[str, Any] = {}
        for k, v in kernel_kwargs.items():
            if hasattr(v, "ndim") and v.ndim > 0:
                sim_kwargs[k] = v.astype(np.float32)
            else:
                sim_kwargs[k] = v
        result = nki.simulate(kernel_func)(**sim_kwargs)
        output = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        error = capture_error(e)
    return output, error


def _rewrite_to_fp32(nki_path: str) -> str:
    """Return a sibling ``.sim.py`` with every ``dtype=nl.<name>`` forced to ``nl.float32``.

    Targets the codegen's dtype-bearing tokens without touching
    unrelated ``nl.*`` names (``nl.ndarray``, ``nl.sbuf``,
    ``nl.psum``, ``nl.shared_hbm``, etc.).
    """
    source = Path(nki_path).read_text()
    rewritten = re.sub(r"dtype=nl\.\w+", "dtype=nl.float32", source)
    sim_path = Path(nki_path).with_suffix(".sim.py")
    sim_path.write_text(rewritten)
    return str(sim_path)


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


def compute_golden(nkigym_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute the nkigym math function as the golden reference, in float32.

    The source carries its own imports (``from nkigym.ops.matmul
    import NKIMatmul`` etc.). Each ``NKIOp`` has a pure-numpy
    ``__call__`` — invoking the math function therefore runs a
    numpy simulation of the kernel at float32. Inputs are cast to
    float32 to match ``simulate_one``'s fp32-end-to-end path.

    Args:
        nkigym_source: Source code of the nkigym math function.
        func_name: Name of the nkigym function within the source.
        kernel_kwargs: Input arrays (will be cast to float32).

    Returns:
        Expected output at float32 precision.
    """
    g: dict[str, Any] = {"__builtins__": __builtins__, "__name__": "__nkigym_golden__"}
    exec(nkigym_source, g)  # noqa: S102
    func = g[func_name]
    f32_kwargs = {k: v.astype(np.float32) for k, v in kernel_kwargs.items() if hasattr(v, "ndim") and v.ndim > 0}
    return func(**f32_kwargs)
