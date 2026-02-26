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

from nkigym.ir import GymProgram

logger = logging.getLogger(__name__)


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

    variants: list[GymProgram]
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
        """Shut down the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)


def _set_neuron_core(core_id: int) -> None:
    """Worker initializer to pin a Neuron core."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"


def _split_into_groups(items: list[Any], num_groups: int) -> list[list[Any]]:
    """Distribute items across groups via round-robin."""
    groups: list[list[Any]] = []
    if items:
        effective = min(len(items), num_groups)
        groups = [[] for _ in range(effective)]
        for i, item in enumerate(items):
            groups[i % effective].append(item)
    return groups


def _calculate_mfu(mac_count: int, time_ms: float) -> float:
    """Calculate estimated MFU for trn2 (2.4 GHz, 128x128 PE)."""
    pe_freq = 2.4e9
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * pe_freq
    theoretical_pe_cycles = flops / (2 * 128 * 128)
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
            mfu = _calculate_mfu(cfg.mac_count, min_ms)
        outputs = spike.run(compiled, *cfg.kernel_kwargs.values())
        actual = outputs if isinstance(outputs, np.ndarray) else outputs[0]
        np.testing.assert_allclose(actual, cfg.expected, rtol=1e-4, atol=1e-4)
        correct = True
    except Exception as e:
        error = _capture_error(e)
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
    )


def _run_core_worker(nki_neff_pairs: list[tuple[str, str]], cfg: _BenchmarkConfig) -> list[VariantResult]:
    """Run benchmarks for a batch of variants on one Neuron core."""
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    results: list[VariantResult] = []
    with BaremetalExecutor(verbose=0) as spike:
        for nki_path, neff_path in nki_neff_pairs:
            cr = CompileResult(nki_path=nki_path, neff_path=neff_path, error="")
            results.append(_benchmark_one(spike, cr, cfg))
    return results


def _dispatch_to_cores(groups: list[list[tuple[str, str]]], cfg: _BenchmarkConfig) -> list[VariantResult]:
    """Spawn one ProcessPoolExecutor per Neuron core and collect results."""
    all_results: list[VariantResult] = []
    executors: list[ProcessPoolExecutor] = []
    futures: dict[Future, int] = {}
    for rank, group in enumerate(groups):
        executor = ProcessPoolExecutor(max_workers=1, initializer=_set_neuron_core, initargs=(rank,))
        executors.append(executor)
        future = executor.submit(_run_core_worker, group, cfg)
        futures[future] = rank
    for future in as_completed(futures):
        all_results.extend(future.result())
    for executor in executors:
        executor.shutdown(wait=True)
    return all_results


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


def run_on_hardware(
    compile_results: list[CompileResult],
    func_name: str,
    kernel_kwargs: dict[str, np.ndarray],
    expected: np.ndarray,
    warmup: int,
    iters: int,
    mac_count: int,
) -> list[VariantResult]:
    """Run compiled variants on Neuron hardware in parallel across up to 128 cores.

    Args:
        compile_results: Results from compilation (may include failures).
        func_name: Kernel function name.
        kernel_kwargs: Input tensors dict.
        expected: Expected output for correctness check.
        warmup: Warmup iterations.
        iters: Benchmark iterations.
        mac_count: Total MAC count (same for all variants).

    Returns:
        List of VariantResult for all variants including compile failures.
    """
    valid = [(cr.nki_path, cr.neff_path) for cr in compile_results if not cr.error]
    all_results = [_compile_failure_result(cr, mac_count) for cr in compile_results if cr.error]
    if valid:
        cfg = _BenchmarkConfig(
            func_name=func_name,
            kernel_kwargs=kernel_kwargs,
            output_name="output",
            output_shape=expected.shape,
            output_dtype=expected.dtype,
            expected=expected,
            warmup=warmup,
            iters=iters,
            mac_count=mac_count,
        )
        groups = _split_into_groups(valid, 128)
        logger.info("Running %d variants on %d Neuron cores", len(valid), len(groups))
        all_results.extend(_dispatch_to_cores(groups, cfg))
        logger.info("Hardware run complete: %d results", len(all_results))
    else:
        logger.warning("No successfully compiled variants to benchmark")
    return all_results
