"""Compile and benchmark NKI kernel variants on Neuron hardware."""

import importlib.util
import logging
import os
import shutil
import signal
import sys
import tempfile
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from neuronxcc.nki_standalone import NKI_IR_VERSION, compile_nki_ir_kernel_to_neff
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel

logger = logging.getLogger(__name__)

_PE_FREQ_HZ = 2.4e9
_BF16_FLOPS_PER_CYCLE = 2 * 128 * 128

_TRN2_FLOPS_PER_CYCLE: dict[str, int] = {
    "float8_e4m3fn": 4 * 128 * 128,
    "float8_e5m2": 4 * 128 * 128,
    "float16": 2 * 128 * 128,
    "bfloat16": 2 * 128 * 128,
}


def _init_compile_worker() -> None:
    """Silence compiler diagnostic noise in worker processes.

    Redirects stdout/stderr to /dev/null at the OS file-descriptor level
    so bare print() calls in neuronxcc are suppressed. Also sets the
    NKI TraceKernel logger to WARNING.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    logging.getLogger("nki.compiler.backends.neuron.TraceKernel").setLevel(logging.WARNING)


def _capture_error(exc: Exception) -> str:
    """Capture the full traceback from an exception as a string."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class CompileResult(NamedTuple):
    """Result of compiling a single NKI kernel to NEFF.

    Empty ``neff_path`` indicates compilation failure.
    Non-empty ``error`` contains the error message on failure.
    """

    nki_path: str
    neff_path: str
    error: str


class VariantResult(NamedTuple):
    """Benchmark result for a single kernel variant.

    Non-empty ``error`` indicates execution failure.
    """

    nki_path: str
    min_ms: float
    mean_ms: float
    p50_ms: float
    p99_ms: float
    mac_count: int
    mfu: float
    correct: bool
    error: str
    arithmetic_intensity: float = 0.0
    roofline_bound: str = ""
    roofline_peak_tflops: float = 0.0
    roofline_efficiency: float = 0.0


@dataclass
class TensorStub:
    """Tensor descriptor for the NeuronX compilation API."""

    shape: tuple[int, ...]
    dtype: np.dtype
    name: str


@dataclass
class _BenchmarkConfig:
    """Shared configuration for hardware benchmarking."""

    func_name: str
    kernel_kwargs: dict[str, np.ndarray]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: np.dtype
    expected: np.ndarray
    warmup: int
    iters: int
    mac_count: int
    input_dtype_name: str
    roofline_map: dict[str, tuple[float, str, float]] = field(default_factory=dict)


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


@dataclass
class SearchResults:
    """Combined search and benchmark results."""

    variants: list
    variant_results: list[VariantResult] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of variants."""
        return len(self.variants)

    def summary(self) -> None:
        """Log a one-line benchmark summary."""
        valid = [r for r in self.variant_results if not r.error]
        errors = [r for r in self.variant_results if r.error]
        logger.info("Benchmark results: %d succeeded, %d failed", len(valid), len(errors))


def _load_kernel(nki_path: str, func_name: str) -> Any:
    """Load a kernel function from an NKI source file."""
    module_name = f"nki_kernel_{Path(nki_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, nki_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {nki_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler that raises TimeoutError for compilation."""
    raise TimeoutError("Compilation timed out after 10 minutes")


def _compile_nki_kernel(
    nki_path: str,
    func_name: str,
    input_tensors: dict[str, np.ndarray],
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    output_dir: str,
) -> str:
    """Compile an NKI kernel file to NEFF using neuronxcc.

    Returns:
        Path to the compiled NEFF file.

    Raises:
        RuntimeError: If NEFF file is not produced.
    """
    tempfile.tempdir = "/tmp/nki_artifacts"
    os.makedirs(tempfile.tempdir, exist_ok=True)
    kernel = _load_kernel(nki_path, func_name)
    stubs = [TensorStub(shape=output_shape, dtype=output_dtype, name=output_name)]
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    compile_nki_ir_kernel_to_neff(
        kernel_func=kernel,
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


def _compile_worker(
    nki_path: str,
    func_name: str,
    input_shapes: dict[str, tuple[int, ...]],
    input_dtype_name: str,
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype_name: str,
    compile_dir: str,
) -> CompileResult:
    """Top-level picklable worker for parallel NKI compilation."""
    neff_path = ""
    error = ""
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(600)
        in_dtype = np.dtype(input_dtype_name)
        out_dtype = np.dtype(output_dtype_name)
        input_tensors = {name: np.zeros(shape, dtype=in_dtype) for name, shape in input_shapes.items()}
        neff_path = _compile_nki_kernel(
            nki_path, func_name, input_tensors, output_name, output_shape, out_dtype, compile_dir
        )
    except Exception as e:
        error = _capture_error(e)
    finally:
        signal.alarm(0)
    return CompileResult(nki_path=nki_path, neff_path=neff_path, error=error)


@dataclass
class CompilationPool:
    """Manages parallel background compilation of NKI kernels.

    Attributes:
        func_name: Kernel function name in each NKI file.
        input_shapes: Map of input parameter names to shapes.
        input_dtype_name: String dtype name for inputs.
        output_name: Output tensor name.
        output_shape: Output tensor shape.
        output_dtype_name: String dtype name for output.
        cache_dir: Root cache directory for NEFF artifacts.
    """

    func_name: str
    input_shapes: dict[str, tuple[int, ...]]
    input_dtype_name: str
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype_name: str
    cache_dir: Path
    _executor: Any = field(default=None, init=False, repr=False)
    _futures: list[Future] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Start the process pool for background compilation."""
        cpu_count = os.cpu_count() or 1
        workers = max(cpu_count - 1, 1)
        self._executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_compile_worker)
        logger.info("Compilation pool: %d workers", workers)

    def submit(self, nki_path: str) -> None:
        """Submit an NKI kernel for background compilation."""
        compile_dir = self.cache_dir / "neff" / Path(nki_path).stem
        if compile_dir.exists():
            shutil.rmtree(compile_dir)
        compile_dir.mkdir(parents=True)
        future = self._executor.submit(
            _compile_worker,
            nki_path,
            self.func_name,
            self.input_shapes,
            self.input_dtype_name,
            self.output_name,
            self.output_shape,
            self.output_dtype_name,
            str(compile_dir),
        )
        self._futures.append(future)

    def wait_all(self) -> list[CompileResult]:
        """Wait for all compilation jobs to finish."""
        results: list[CompileResult] = []
        errors = 0
        for future in as_completed(self._futures):
            cr = future.result()
            if cr.error:
                errors += 1
            results.append(cr)
        logger.info("Compilation: %d succeeded, %d failed", len(results) - errors, errors)
        return results

    def shutdown(self) -> None:
        """Shut down the process pool without waiting for workers to exit.

        All futures are already resolved after ``wait_all()``, so worker
        processes can terminate asynchronously.  ``wait=False`` lets us
        skip joining the idle worker processes.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False)


def _calculate_mfu(mac_count: int, time_ms: float, dtype_name: str) -> float:
    """Calculate MFU for trn2 NeuronCore-v3 TensorEngine (2.4 GHz).

    FLOPS/cycle by dtype: FP8(E4/E5) 4*128*128, BF16/FP16 2*128*128.
    Falls back to BF16 for unrecognized dtypes.
    """
    if dtype_name not in _TRN2_FLOPS_PER_CYCLE:
        logger.warning("Unknown dtype %r for MFU; using BF16 peak", dtype_name)
    flops_per_cycle = _TRN2_FLOPS_PER_CYCLE.get(dtype_name, _BF16_FLOPS_PER_CYCLE)
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / flops_per_cycle
    return theoretical_pe_cycles / actual_pe_cycles


def _create_compiled_kernel(neff_path: str, nki_path: str, cfg: _BenchmarkConfig) -> CompiledKernel:
    """Create a CompiledKernel from a NEFF for BaremetalExecutor."""
    kernel = _load_kernel(nki_path, cfg.func_name)
    hlo = HLOModule(name=cfg.func_name)
    for name, tensor in cfg.kernel_kwargs.items():
        hlo.add_parameter(tensor.shape, tensor.dtype, name=name)
    hlo.set_results([HLOTensor(shape=cfg.output_shape, dtype=cfg.output_dtype, name=cfg.output_name)])
    return CompiledKernel(_TracedKernel(kernel, hlo), neff_path)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list via nearest-rank."""
    idx = max(0, min(int(pct / 100 * len(sorted_vals)), len(sorted_vals) - 1))
    return sorted_vals[idx]


def _enrich_roofline(cfg: _BenchmarkConfig, nki_path: str, min_ms: float) -> tuple[float, str, float, float]:
    """Look up roofline data and compute efficiency from measured latency."""
    entry = cfg.roofline_map.get(nki_path)
    ai, bound, peak, eff = 0.0, "", 0.0, 0.0
    if entry is not None:
        ai, bound, peak = entry
        if peak > 0 and min_ms > 0:
            achieved = (cfg.mac_count * 2) / (min_ms / 1000) / 1e12
            eff = achieved / peak
    return ai, bound, peak, eff


def _benchmark_one(spike: BaremetalExecutor, cr: CompileResult, cfg: _BenchmarkConfig) -> VariantResult:
    """Benchmark a single compiled variant on a Neuron core."""
    min_ms = mean_ms = p50_ms = p99_ms = mfu = 0.0
    correct = False
    error = ""
    try:
        compiled = _create_compiled_kernel(cr.neff_path, cr.nki_path, cfg)
        stats = spike.benchmark(
            compiled,
            *cfg.kernel_kwargs.values(),
            warmup_iterations=cfg.warmup,
            benchmark_iterations=cfg.iters,
            mode="device",
        )
        min_ms, mean_ms = stats.min_ms, stats.mean_ms
        sorted_durations = sorted(stats.durations_ms)
        p50_ms = _percentile(sorted_durations, 50)
        p99_ms = _percentile(sorted_durations, 99)
        if cfg.mac_count > 0 and min_ms > 0:
            mfu = _calculate_mfu(cfg.mac_count, min_ms, cfg.input_dtype_name)
        outputs = spike.run(compiled, *cfg.kernel_kwargs.values())
        actual = outputs if isinstance(outputs, np.ndarray) else outputs[0]
        np.testing.assert_allclose(actual, cfg.expected, rtol=1e-3, atol=1e-3)
        correct = True
    except Exception as e:
        error = _capture_error(e)

    ai, r_bound, r_peak, r_eff = _enrich_roofline(cfg, cr.nki_path, min_ms)
    return VariantResult(
        nki_path=cr.nki_path,
        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mac_count=cfg.mac_count,
        mfu=mfu,
        correct=correct,
        error=error,
        arithmetic_intensity=ai,
        roofline_bound=r_bound,
        roofline_peak_tflops=r_peak,
        roofline_efficiency=r_eff,
    )


def _run_core_worker(core_id: int, nki_neff_pairs: list[tuple[str, str]], cfg: _BenchmarkConfig) -> list[VariantResult]:
    """Run benchmarks for a batch of variants on one pinned Neuron core."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    results: list[VariantResult] = []
    with BaremetalExecutor(verbose=0) as spike:
        for nki_path, neff_path in nki_neff_pairs:
            cr = CompileResult(nki_path=nki_path, neff_path=neff_path, error="")
            results.append(_benchmark_one(spike, cr, cfg))
    return results


def _compile_failure_result(cr: CompileResult, mac_count: int) -> VariantResult:
    """Convert a failed CompileResult into a VariantResult preserving the error."""
    return VariantResult(
        nki_path=cr.nki_path,
        min_ms=0.0,
        mean_ms=0.0,
        p50_ms=0.0,
        p99_ms=0.0,
        mac_count=mac_count,
        mfu=0.0,
        correct=False,
        error=cr.error,
    )


def _make_benchmark_cfg(
    func_name: str,
    kernel_kwargs: dict[str, np.ndarray],
    expected: np.ndarray,
    warmup: int,
    iters: int,
    mac_count: int,
    input_dtype_name: str,
    roofline_map: dict[str, tuple[float, str, float]],
) -> _BenchmarkConfig:
    """Build a _BenchmarkConfig from the common benchmark parameters."""
    return _BenchmarkConfig(
        func_name=func_name,
        kernel_kwargs=kernel_kwargs,
        output_name="output",
        output_shape=expected.shape,
        output_dtype=np.dtype(input_dtype_name),
        expected=expected,
        warmup=warmup,
        iters=iters,
        mac_count=mac_count,
        input_dtype_name=input_dtype_name,
        roofline_map=roofline_map,
    )


def _collect_hw_results(hw_futures: list[Future]) -> list[VariantResult]:
    """Wait for all hardware benchmark futures and collect results."""
    results: list[VariantResult] = []
    for hw_future in as_completed(hw_futures):
        results.extend(hw_future.result())
    return results


def stream_compile_and_run(pool: CompilationPool, cfg: _BenchmarkConfig) -> tuple[int, int, list[VariantResult]]:
    """Stream compilation results into hardware benchmarks as they complete."""
    compile_errors: list[VariantResult] = []
    hw_futures: list[Future] = []
    hw_executor = ProcessPoolExecutor(max_workers=128)
    core_id = 0
    for future in as_completed(pool._futures):
        cr = future.result()
        if cr.error:
            compile_errors.append(_compile_failure_result(cr, cfg.mac_count))
        else:
            pair = [(cr.nki_path, cr.neff_path)]
            hw_futures.append(hw_executor.submit(_run_core_worker, core_id, pair, cfg))
            core_id += 1
    pool.shutdown()
    failed = len(pool._futures) - core_id
    logger.info("Compilation: %d succeeded, %d failed", core_id, failed)
    logger.info("Running %d variants on %d Neuron cores", core_id, core_id)
    hw_results = _collect_hw_results(hw_futures)
    hw_executor.shutdown(wait=False)
    logger.info("Hardware run complete: %d results", len(compile_errors) + len(hw_results))
    return core_id, failed, compile_errors + hw_results
