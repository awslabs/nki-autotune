"""Remote worker for NKI kernel compilation and benchmarking.

Runs on remote Trn nodes. Receives a JSON payload on stdin containing
kernel source code, tensor specs, and benchmark config. Compiles each
kernel to NEFF, benchmarks on Neuron hardware, and writes JSON results
to stdout. All logging goes to stderr.

No dependency on nkigym — this is a standalone profiling backend.
The autotune package is bundled and sent over SSH by the coordinator.
"""

import ast
import importlib.util
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

_DTYPE_CACHE: dict[str, np.dtype] = {}


def _resolve_dtype(name: str) -> np.dtype:
    """Resolve a dtype string, handling bfloat16 via ml_dtypes.

    Standard numpy does not support bfloat16. This uses ml_dtypes
    when needed and caches results.
    """
    if name in _DTYPE_CACHE:
        return _DTYPE_CACHE[name]
    try:
        dt = np.dtype(name)
    except TypeError:
        import ml_dtypes  # type: ignore[import-untyped]

        dt = np.dtype(getattr(ml_dtypes, name))
    _DTYPE_CACHE[name] = dt
    return dt


if TYPE_CHECKING:
    from nkipy.runtime import BaremetalExecutor, CompiledKernel

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
    correct: bool
    error: str


def _capture_error(exc: Exception) -> str:
    """Capture the full traceback from an exception as a string."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _tensor_inputs(kwargs: dict[str, Any]) -> list[np.ndarray]:
    """Extract tensor (ndim > 0) values from kernel kwargs, preserving order."""
    return [v for v in kwargs.values() if hasattr(v, "ndim") and v.ndim > 0]


def _make_failure(kernel_name: str, error: str, mac_count: int = 0) -> ProfileResult:
    """Create a failed ProfileResult."""
    return ProfileResult(
        kernel_name=kernel_name,
        min_ms=0.0,
        mean_ms=0.0,
        p50_ms=0.0,
        p99_ms=0.0,
        mac_count=mac_count,
        mfu=0.0,
        correct=False,
        error=error,
    )


def _compile_failure_result(cr: CompileResult, mac_count: int) -> ProfileResult:
    """Convert a failed CompileResult into a ProfileResult."""
    return _make_failure(cr.kernel_name, cr.error, mac_count)


def detect_func_name(source: str) -> str:
    """Detect the @nki.jit decorated function name from kernel source.

    Parses the source with AST and finds the first function decorated
    with ``@nki.jit``.

    Raises:
        ValueError: If no @nki.jit decorated function is found.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_str = ast.unparse(dec)
            if dec_str in ("nki.jit", "nki.jit()"):
                return node.name
    raise ValueError("No @nki.jit decorated function found in source")


def detect_output_spec(source: str) -> tuple[int, ...]:
    """Detect the output tensor shape from kernel source.

    Finds the first ``nl.ndarray(..., buffer=nl.shared_hbm)`` call and
    extracts the shape tuple.

    Returns:
        Output tensor shape.

    Raises:
        ValueError: If no HBM output tensor is found.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_str = ast.unparse(node.func)
        if func_str != "nl.ndarray":
            continue
        is_hbm = False
        for kw in node.keywords:
            if kw.arg == "buffer" and ast.unparse(kw.value) == "nl.shared_hbm":
                is_hbm = True
                break
        if not is_hbm:
            continue
        if not node.args:
            continue
        shape_node = node.args[0]
        if isinstance(shape_node, ast.Tuple):
            dims = []
            for elt in shape_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    raise ValueError(f"Non-constant dimension in output shape: {ast.unparse(elt)}")
            return tuple(dims)
    raise ValueError("No nl.ndarray(..., buffer=nl.shared_hbm) found in source")


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


def _inline_scalars(nki_path: str, scalar_inputs: dict[str, Any]) -> str:
    """Rewrite NKI kernel source to inline scalar params as local constants.

    neuronxcc cannot handle 0-d tensor inputs. This rewrites the
    kernel source to remove scalar params from the function signature
    and define them as local variables in the function body.

    Returns:
        Path to the rewritten kernel file.
    """
    source = Path(nki_path).read_text()
    for name in scalar_inputs:
        source = re.sub(rf"(def \w+\([^)]*),\s*{name}\s*([,)])", rf"\1\2", source)
        source = re.sub(rf"(def \w+\(\s*){name}\s*,\s*", rf"\1", source)

    def_match = re.search(r"(def \w+\([^)]*\):)", source)
    if def_match:
        insert_pos = def_match.end()
        assignments = ""
        for name, val in scalar_inputs.items():
            assignments += f"\n    {name} = {val!r}"
        source = source[:insert_pos] + assignments + source[insert_pos:]

    rewritten_path = str(Path(nki_path).with_suffix(".inlined.py"))
    Path(rewritten_path).write_text(source)
    return rewritten_path


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler that raises TimeoutError for compilation."""
    raise TimeoutError("Compilation timed out after 10 minutes")


def _init_compile_worker() -> None:
    """Silence compiler diagnostic noise in worker subprocesses.

    Also removes NEURON_RT_VISIBLE_CORES so compile workers don't
    try to allocate Neuron cores (set by the benchmark prewarm thread).
    """
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + path
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _compile_nki_kernel(
    nki_path: str,
    func_name: str,
    input_tensors: dict[str, np.ndarray],
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    output_dir: str,
) -> str:
    """Compile an NKI kernel file to NEFF via nki.compiler.

    Uses the TracerFrontend to lower the kernel to MLIR, then
    compiles to BIR and finally to a NEFF binary.

    Returns:
        Path to the compiled NEFF file.
    """
    from nki.compiler.driver import CompileOptions, compile_bir_to_neff, compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.framework.kernel import Kernel

    tempfile.tempdir = output_dir
    os.makedirs(tempfile.tempdir, exist_ok=True)

    tensor_inputs: dict[str, np.ndarray] = {}
    scalar_inputs: dict[str, Any] = {}
    for name, val in input_tensors.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
            tensor_inputs[name] = val
        else:
            scalar_inputs[name] = val
    if scalar_inputs:
        nki_path = _inline_scalars(nki_path, scalar_inputs)
    kernel_func = _load_kernel(nki_path, func_name)

    kernel = Kernel(kernel_func)
    neff_file = os.path.join(output_dir, "file.neff")
    opts = CompileOptions(target="trn2", lnc=1, output_path=neff_file, artifacts_dir=output_dir)
    frontend = TracerFrontend()
    bir, cr = compile_to_bir(
        kernel, frontend=frontend, inputs=tensor_inputs, compile_opts=opts, output_names=[output_name]
    )

    input_arrays = [np.zeros(s.shape, dtype=np.dtype(s.dtype)) for s in cr.input_specs]
    neff_ck = compile_bir_to_neff(
        opts, bir, input_arrays, cr.argument_names, cr.output_names, input_output_aliases=cr.input_output_aliases
    )
    if not os.path.isfile(neff_file):
        raise RuntimeError(f"NEFF file not found at: {neff_file}")
    return neff_file


def _compile_one(
    kernel_name: str,
    nki_path: str,
    func_name: str,
    input_shapes: dict[str, tuple[int, ...]],
    input_dtype_name: str,
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype_name: str,
    compile_dir: str,
    scalar_params: dict[str, float] | None = None,
) -> CompileResult:
    """Top-level picklable worker for parallel NKI compilation."""
    neff_path = ""
    error = ""
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(600)
        in_dtype = _resolve_dtype(input_dtype_name)
        out_dtype = _resolve_dtype(output_dtype_name)
        input_tensors: dict[str, Any] = {name: np.zeros(shape, dtype=in_dtype) for name, shape in input_shapes.items()}
        if scalar_params:
            for name, value in scalar_params.items():
                input_tensors[name] = value
        neff_path = _compile_nki_kernel(
            nki_path, func_name, input_tensors, output_name, output_shape, out_dtype, compile_dir
        )
    except Exception as e:
        error = _capture_error(e)
    finally:
        signal.alarm(0)
    return CompileResult(kernel_name=kernel_name, nki_path=nki_path, neff_path=neff_path, error=error)


def _calculate_mfu(mac_count: int, time_ms: float, dtype_name: str) -> float:
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


def _create_compiled_kernel(
    neff_path: str,
    nki_path: str,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
) -> "CompiledKernel":
    """Create a CompiledKernel from a NEFF for BaremetalExecutor."""
    from nkipy.core.backend.hlo import HLOModule, HLOTensor
    from nkipy.runtime import CompiledKernel

    kernel = _load_kernel(nki_path, func_name)
    hlo = HLOModule(name=func_name)
    for name, tensor in kernel_kwargs.items():
        if hasattr(tensor, "ndim") and tensor.ndim > 0:
            hlo.add_parameter(tensor.shape, tensor.dtype, name=name)
    hlo.set_results([HLOTensor(shape=output_shape, dtype=output_dtype, name=output_name)])
    return CompiledKernel(_TracedKernel(kernel, hlo), neff_path)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile from a pre-sorted list via nearest-rank."""
    idx = max(0, min(int(pct / 100 * len(sorted_vals)), len(sorted_vals) - 1))
    return sorted_vals[idx]


def _benchmark_one(
    spike: "BaremetalExecutor",
    cr: CompileResult,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    warmup: int,
    iters: int,
    mac_count: int,
    input_dtype_name: str,
    golden: np.ndarray | None,
    atol: float,
    rtol: float,
) -> ProfileResult:
    """Benchmark a single compiled variant on a Neuron core."""
    min_ms = mean_ms = p50_ms = p99_ms = mfu = 0.0
    correct = False
    error = ""
    try:
        compiled = _create_compiled_kernel(
            cr.neff_path, cr.nki_path, func_name, kernel_kwargs, output_name, output_shape, output_dtype
        )
        stats = spike.benchmark(
            compiled,
            *_tensor_inputs(kernel_kwargs),
            warmup_iterations=warmup,
            benchmark_iterations=iters,
            mode="device",
        )
        min_ms, mean_ms = stats.min_ms, stats.mean_ms  # type: ignore[attr-defined]
        sorted_durations = sorted(stats.durations_ms)  # type: ignore[attr-defined]
        p50_ms = _percentile(sorted_durations, 50)
        p99_ms = _percentile(sorted_durations, 99)
        if mac_count > 0 and min_ms > 0:
            mfu = _calculate_mfu(mac_count, min_ms, input_dtype_name)

        if golden is not None:
            run_out: Any = spike.run(compiled, *_tensor_inputs(kernel_kwargs))
            if isinstance(run_out, np.ndarray):
                run_out = (run_out,)
            actual = run_out[0]
            if golden.dtype != actual.dtype:
                actual = actual.astype(golden.dtype)
            np.testing.assert_allclose(actual, golden, atol=atol, rtol=rtol)

        correct = True
    except Exception as e:
        error = _capture_error(e)

    return ProfileResult(
        kernel_name=cr.kernel_name,
        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mac_count=mac_count,
        mfu=mfu,
        correct=correct,
        error=error,
    )


def detect_neuron_cores() -> int:
    """Detect available Neuron cores via neuron-ls."""
    result = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"neuron-ls failed: {result.stderr[:500]}")
    devices = json.loads(result.stdout)
    total_cores = sum(d["nc_count"] for d in devices)
    if total_cores == 0:
        raise RuntimeError("neuron-ls reports 0 Neuron cores")
    return total_cores


def _generate_tensors(tensor_specs: dict[str, dict], seed: int) -> dict[str, np.ndarray]:
    """Generate random input tensors from shapes, dtypes, and a seed."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, spec in tensor_specs.items():
        shape = tuple(spec["shape"])
        dtype = _resolve_dtype(spec["dtype"])
        tensors[name] = rng.standard_normal(shape).astype(dtype)
    return tensors


def _compute_golden(golden_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
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


def worker_main() -> None:
    """Compile and benchmark NKI kernels on a remote host.

    Reads a JSON payload from stdin containing kernel sources and
    benchmark config. Writes results JSON to stdout.
    """
    data = sys.stdin.buffer.read()
    payload = json.loads(data)

    host = payload["host"]
    neuron_target = payload["neuron_platform_target"]
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = neuron_target

    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + path

    config = payload["config"]
    warmup = config["warmup"]
    iters = config["iters"]
    mac_count = config["mac_count"]
    input_dtype_name = config["input_dtype_name"]
    atol = config.get("atol", 1e-2)
    rtol = config.get("rtol", 1e-2)

    kernel_names = payload["kernel_names"]
    sources = payload["sources"]
    tensor_specs = payload["tensor_specs"]
    seed = payload["seed"]
    scalar_params = payload.get("scalar_params", {})
    golden_source = payload.get("golden_source")
    golden_func_name = payload.get("golden_func_name")

    kernel_kwargs: dict[str, Any] = _generate_tensors(tensor_specs, seed)
    for name, value in scalar_params.items():
        kernel_kwargs[name] = np.float64(value)

    golden: np.ndarray | None = None
    if golden_source and golden_func_name:
        golden = _compute_golden(golden_source, golden_func_name, kernel_kwargs)

    input_shapes = {k: v.shape for k, v in kernel_kwargs.items() if hasattr(v, "ndim") and v.ndim > 0}

    logging.basicConfig(level=logging.INFO, format=f"[{host}] %(message)s")

    t_start = time.monotonic()
    work_dir = Path(tempfile.mkdtemp(prefix=f"autotune-{host}-"))
    all_results: list[ProfileResult] = []
    try:
        nki_dir = work_dir / "nki"
        nki_dir.mkdir()
        neff_dir = work_dir / "neff"
        neff_dir.mkdir()

        nki_paths: dict[str, str] = {}
        for kname in kernel_names:
            p = nki_dir / kname
            p.write_text(sources[kname])
            nki_paths[kname] = str(p)

        neuron_cores = detect_neuron_cores()
        cpu_cores = os.cpu_count() or 1
        compile_workers = min(max(cpu_cores - 1, 1), len(kernel_names))

        logger.info(
            "Worker starting: %d kernels, %d CPU cores, %d Neuron cores (setup %.1fs)",
            len(kernel_names),
            cpu_cores,
            neuron_cores,
            time.monotonic() - t_start,
        )

        kernel_func_names: dict[str, str] = {}
        kernel_output_shapes: dict[str, tuple[int, ...]] = {}
        for kname in kernel_names:
            kernel_func_names[kname] = detect_func_name(sources[kname])
            kernel_output_shapes[kname] = detect_output_spec(sources[kname])

        output_name = "hbm_tensor_0"
        output_dtype = _resolve_dtype(input_dtype_name)

        scalar_dict = {k: float(v) for k, v in scalar_params.items()} if scalar_params else None

        compile_results: list[CompileResult] = []
        compile_errors: list[ProfileResult] = []
        compiled_count = 0

        compile_executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=_init_compile_worker)
        compile_futures: list[Future] = []
        for kname in kernel_names:
            compile_dir = neff_dir / Path(kname).stem
            compile_dir.mkdir(parents=True, exist_ok=True)
            compile_futures.append(
                compile_executor.submit(
                    _compile_one,
                    kname,
                    nki_paths[kname],
                    kernel_func_names[kname],
                    input_shapes,
                    input_dtype_name,
                    output_name,
                    kernel_output_shapes[kname],
                    output_dtype.str,
                    str(compile_dir),
                    scalar_dict,
                )
            )

        """
        Pipeline: start BaremetalExecutor and benchmark each kernel
        as it finishes compiling, overlapping compilation with
        benchmarking. The first benchmark() call pays ~3 s Neuron
        runtime init, which overlaps with remaining compilations.
        """
        from nkipy.runtime import BaremetalExecutor

        os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

        hw_results: list[ProfileResult] = []
        with BaremetalExecutor(verbose=0) as spike:
            for f in as_completed(compile_futures):
                cr = f.result()
                compile_results.append(cr)
                if cr.error:
                    compile_errors.append(_compile_failure_result(cr, mac_count))
                    continue
                compiled_count += 1
                bench_kwargs = {
                    "func_name": kernel_func_names[cr.kernel_name],
                    "kernel_kwargs": kernel_kwargs,
                    "output_name": output_name,
                    "output_shape": kernel_output_shapes[cr.kernel_name],
                    "output_dtype": output_dtype,
                    "warmup": warmup,
                    "iters": iters,
                    "mac_count": mac_count,
                    "input_dtype_name": input_dtype_name,
                    "golden": golden,
                    "atol": atol,
                    "rtol": rtol,
                }
                hw_results.append(_benchmark_one(spike, cr, **bench_kwargs))
        compile_executor.shutdown(wait=True)

        t_compiled = time.monotonic()
        logger.info("Compile+benchmark took %.1fs (pipelined)", t_compiled - t_start)
        logger.info("Compilation: %d succeeded, %d failed", compiled_count, len(compile_errors))

        all_results = compile_errors + hw_results

        compiler_logs: dict[str, str] = {}
        if config.get("collect_compiler_logs", False):
            for cr in compile_results:
                stem = Path(cr.kernel_name).stem
                log_path = neff_dir / stem / "log-neuron-cc.txt"
                if log_path.exists():
                    compiler_logs[cr.kernel_name] = log_path.read_text()

        logger.info(
            "Done: %d compiled, %d benchmarked, %d errors (total %.1fs)",
            compiled_count,
            len(hw_results),
            len(compile_errors),
            time.monotonic() - t_start,
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    output = {"host": host, "results": [r._asdict() for r in all_results], "compiler_logs": compiler_logs}
    sys.stdout.buffer.write(json.dumps(output).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    worker_main()
