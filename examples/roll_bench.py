"""Benchmark unrolled vs compact NKI matmul kernels.

Self-contained script with no nkigym dependency. Imports pre-generated
kernels from compact.py and unrolled.py in the same directory.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/roll_bench.py --cache-dir /fsx/weittang/roll_cache_1
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from compact import nkigym_matmul as compact_kernel
from neuronxcc.nki_standalone import NKI_IR_VERSION, compile_nki_ir_kernel_to_neff
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel
from unrolled import nkigym_matmul as unrolled_kernel

logger = logging.getLogger(__name__)

_PE_FREQ_HZ = 2.4e9
_BF16_FLOPS_PER_CYCLE = 2 * 128 * 128
_TRN2_FLOPS_PER_CYCLE: dict[str, int] = {
    "float8_e4m3fn": 4 * 128 * 128,
    "float8_e5m2": 4 * 128 * 128,
    "float16": 2 * 128 * 128,
    "bfloat16": 2 * 128 * 128,
}

_WARMUP = 10
_ITERS = 100
_K = 2048
_M = 2048
_N = 2048
_MAC_COUNT = _K * _M * _N
_FUNC_NAME = "nkigym_matmul"
_DTYPE = "float16"


class VariantResult(NamedTuple):
    """Benchmark result for a single kernel variant."""

    min_ms: float
    mean_ms: float
    p50_ms: float
    p99_ms: float
    mfu: float
    correct: bool
    error: str


@dataclass
class TensorStub:
    """Tensor descriptor for the NeuronX compilation API."""

    shape: tuple[int, ...]
    dtype: np.dtype
    name: str


class _TracedKernel:
    """Minimal traced kernel wrapper required by CompiledKernel."""

    def __init__(self, func: Any, code: Any) -> None:
        """Initialize with kernel function and HLO module."""
        self.func = func
        self._code = code

    @property
    def __name__(self) -> str:
        """Return the kernel function name."""
        return self.func.__name__


def _capture_error(exc: Exception) -> str:
    """Capture the full traceback from an exception as a string."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _suppress_stdout() -> tuple[int, int]:
    """Redirect stdout/stderr to devnull for quiet compilation.

    Returns:
        Saved file descriptors to restore later.
    """
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved


def _restore_stdout(saved: tuple[int, int]) -> None:
    """Restore stdout/stderr from saved file descriptors.

    Args:
        saved: Tuple of (fd1, fd2) from _suppress_stdout.
    """
    os.dup2(saved[0], 1)
    os.dup2(saved[1], 2)
    os.close(saved[0])
    os.close(saved[1])


def _compile_kernel(kernel_func: Any, input_tensors: dict[str, np.ndarray], output_dir: str) -> str:
    """Compile an NKI kernel function to NEFF using neuronxcc.

    Args:
        kernel_func: The @nki.jit decorated kernel function.
        input_tensors: Map of input names to numpy arrays.
        output_dir: Directory for compiled NEFF artifacts.

    Returns:
        Path to the compiled NEFF file.

    Raises:
        RuntimeError: If NEFF file is not produced.
    """
    tempfile.tempdir = "/tmp/nki_artifacts"
    os.makedirs(tempfile.tempdir, exist_ok=True)
    stubs = [TensorStub(shape=(_M, _N), dtype=np.dtype(_DTYPE), name="output")]
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    compile_nki_ir_kernel_to_neff(
        kernel_func=kernel_func,
        kernel_inputs_dict=input_tensors,
        kernel_outputs=stubs,
        platform_target="trn2",
        logical_nc_config=1,
        output_directory=output_dir,
        version=NKI_IR_VERSION.beta2,
        additional_compiler_args="--internal-compiler-debug-mode=penguin",
    )
    neff_path = os.path.join(output_dir, "file.neff")
    if not os.path.exists(neff_path):
        raise RuntimeError(f"NEFF file not found at: {neff_path}")
    return neff_path


def _compile_variant(
    cache_dir: Path, name: str, kernel_func: Any, input_tensors: dict[str, np.ndarray]
) -> tuple[str, float]:
    """Compile one kernel variant with suppressed compiler output.

    Args:
        cache_dir: Root cache directory.
        name: Variant name (e.g. 'unrolled', 'compact').
        kernel_func: The @nki.jit decorated kernel function.
        input_tensors: Dummy input tensors for compilation.

    Returns:
        Tuple of (neff_path, compile_seconds).
    """
    compile_dir = str(cache_dir / "neff" / name)
    if os.path.exists(compile_dir):
        shutil.rmtree(compile_dir)
    os.makedirs(compile_dir)
    saved = _suppress_stdout()
    t0 = time.monotonic()
    try:
        neff_path = _compile_kernel(kernel_func, input_tensors, compile_dir)
    finally:
        _restore_stdout(saved)
    elapsed = time.monotonic() - t0
    logger.info("Compiled %s in %.1fs", name, elapsed)
    return neff_path, elapsed


def _calculate_mfu(time_ms: float) -> float:
    """Calculate MFU for trn2 NeuronCore-v3 TensorEngine.

    Uses hardcoded MAC count and float16 peak throughput.

    Args:
        time_ms: Measured latency in milliseconds.

    Returns:
        Model FLOPS utilization ratio.
    """
    flops_per_cycle = _TRN2_FLOPS_PER_CYCLE.get(_DTYPE, _BF16_FLOPS_PER_CYCLE)
    flops = 2 * _MAC_COUNT
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / flops_per_cycle
    return theoretical_pe_cycles / actual_pe_cycles


def _create_compiled_kernel(neff_path: str, kernel_func: Any, kernel_kwargs: dict[str, np.ndarray]) -> CompiledKernel:
    """Create a CompiledKernel from a NEFF for BaremetalExecutor.

    Args:
        neff_path: Path to the compiled NEFF file.
        kernel_func: The @nki.jit decorated kernel function.
        kernel_kwargs: Input tensors for shape/dtype introspection.

    Returns:
        A CompiledKernel ready for execution.
    """
    hlo = HLOModule(name=_FUNC_NAME)
    for name, tensor in kernel_kwargs.items():
        hlo.add_parameter(tensor.shape, tensor.dtype, name=name)
    hlo.set_results([HLOTensor(shape=(_M, _N), dtype=np.dtype(_DTYPE), name="output")])
    return CompiledKernel(_TracedKernel(kernel_func, hlo), neff_path)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list via nearest-rank."""
    idx = max(0, min(int(pct / 100 * len(sorted_vals)), len(sorted_vals) - 1))
    return sorted_vals[idx]


def _benchmark_one(
    spike: BaremetalExecutor,
    neff_path: str,
    kernel_func: Any,
    kernel_kwargs: dict[str, np.ndarray],
    expected: np.ndarray,
) -> VariantResult:
    """Benchmark a single compiled variant on a Neuron core.

    Args:
        spike: Active baremetal executor.
        neff_path: Path to compiled NEFF.
        kernel_func: The @nki.jit decorated kernel function.
        kernel_kwargs: Input tensors.
        expected: Reference output for correctness check.

    Returns:
        VariantResult with timing, MFU, and correctness.
    """
    min_ms = mean_ms = p50_ms = p99_ms = mfu = 0.0
    correct = False
    error = ""
    try:
        compiled = _create_compiled_kernel(neff_path, kernel_func, kernel_kwargs)
        stats = spike.benchmark(
            compiled, *kernel_kwargs.values(), warmup_iterations=_WARMUP, benchmark_iterations=_ITERS, mode="device"
        )
        min_ms, mean_ms = stats.min_ms, stats.mean_ms
        sorted_durations = sorted(stats.durations_ms)
        p50_ms = _percentile(sorted_durations, 50)
        p99_ms = _percentile(sorted_durations, 99)
        if _MAC_COUNT > 0 and min_ms > 0:
            mfu = _calculate_mfu(min_ms)
        outputs = spike.run(compiled, *kernel_kwargs.values())
        actual = outputs if isinstance(outputs, np.ndarray) else outputs[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        correct = True
    except Exception as e:
        error = _capture_error(e)
    return VariantResult(
        min_ms=min_ms, mean_ms=mean_ms, p50_ms=p50_ms, p99_ms=p99_ms, mfu=mfu, correct=correct, error=error
    )


def _make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random float16 inputs and float64 reference output.

    Returns:
        Tuple of (lhs, rhs, expected).
    """
    rng = np.random.default_rng(42)
    lhs = rng.standard_normal((_K, _M)).astype(np.float16)
    rhs = rng.standard_normal((_K, _N)).astype(np.float16)
    expected = lhs.astype(np.float64).T @ rhs.astype(np.float64)
    return lhs, rhs, expected


def _count_lines(filename: str) -> int:
    """Count lines in a source file relative to the script directory.

    Args:
        filename: Name of the source file.

    Returns:
        Number of lines in the file.
    """
    path = Path(__file__).resolve().parent / filename
    return path.read_text().count("\n") + 1


def _print_results(results: list[tuple[str, int, float, VariantResult]]) -> None:
    """Print comparison table of benchmark results.

    Args:
        results: List of (name, lines, compile_s, VariantResult).
    """
    logger.info("")
    logger.info("%-12s %8s %10s %10s %10s %8s", "Variant", "Lines", "Compile", "min_ms", "MFU", "Correct")
    logger.info("-" * 62)
    for name, lines, compile_s, vr in results:
        if vr.error:
            logger.info("%-12s  ERROR: %s", name, vr.error.strip().split("\n")[-1])
        else:
            logger.info("%-12s %8d %9.1fs %10.3f %10.5f %8s", name, lines, compile_s, vr.min_ms, vr.mfu, vr.correct)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with cache_dir.
    """
    parser = argparse.ArgumentParser(description="Benchmark unrolled vs compact NKI matmul kernels")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for compilation artifacts")
    return parser.parse_args()


def main() -> None:
    """Compile and benchmark unrolled vs compact NKI kernel variants."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("nki.compiler.backends.neuron.TraceKernel").setLevel(logging.WARNING)
    args = parse_args()
    cache_dir: Path = args.cache_dir
    lhs, rhs, expected = _make_inputs()
    kernel_kwargs = {"lhs": lhs, "rhs": rhs}
    dummy = {"lhs": np.zeros((_K, _M), np.float16), "rhs": np.zeros((_K, _N), np.float16)}
    variants = [
        ("unrolled", unrolled_kernel, _count_lines("unrolled.py")),
        ("compact", compact_kernel, _count_lines("compact.py")),
    ]
    compiled: list[tuple[str, Any, int, str, float]] = []
    for name, kernel_func, lines in variants:
        logger.info("Compiling %s (%d lines)...", name, lines)
        neff_path, compile_s = _compile_variant(cache_dir, name, kernel_func, dummy)
        compiled.append((name, kernel_func, lines, neff_path, compile_s))
    logger.info("MACs: %d", _MAC_COUNT)
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    results: list[tuple[str, int, float, VariantResult]] = []
    with BaremetalExecutor(verbose=0) as spike:
        for name, kernel_func, lines, neff_path, compile_s in compiled:
            logger.info("Benchmarking %s...", name)
            vr = _benchmark_one(spike, neff_path, kernel_func, kernel_kwargs, expected)
            results.append((name, lines, compile_s, vr))
    _print_results(results)


if __name__ == "__main__":
    main()
