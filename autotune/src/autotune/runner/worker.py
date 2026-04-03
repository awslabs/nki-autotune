"""Remote worker entry point for NKI kernel compilation and benchmarking.

Runs on remote Trn nodes. Receives a JSON payload on stdin containing
kernel source code, tensor specs, and benchmark config. Compiles each
kernel to NEFF, benchmarks on Neuron hardware, and writes JSON results
to stdout. All logging goes to stderr.

The autotune package is bundled and sent over SSH by the coordinator.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from nkipy.runtime import BaremetalExecutor

from autotune.runner.benchmark import benchmark_one, compute_golden, generate_tensors, simulate_one
from autotune.runner.compile import compile_one, init_compile_worker
from autotune.runner.detect import detect_kernel_info, detect_neuron_cores
from autotune.runner.types import (
    BenchmarkConfig,
    CompileResult,
    OutputSpec,
    ProfileResult,
    compile_failure_result,
    ensure_venv_on_path,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

_OUTPUT_TENSOR_NAME = "hbm_tensor_0"


def _parse_payload() -> dict[str, Any]:
    """Read JSON payload from stdin and return parsed dict."""
    data = sys.stdin.buffer.read()
    return json.loads(data)


def _setup_env(payload: dict[str, Any]) -> None:
    """Configure environment variables from payload."""
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = payload["neuron_platform_target"]
    ensure_venv_on_path()


def _write_kernel_files(kernel_names: list[str], sources: dict[str, str], nki_dir: Path) -> dict[str, str]:
    """Write kernel source files to disk and return path mapping."""
    nki_paths: dict[str, str] = {}
    for kname in kernel_names:
        p = nki_dir / kname
        p.write_text(sources[kname])
        nki_paths[kname] = str(p)
    return nki_paths


def _run_simulations(
    kernel_names: list[str], nki_paths: dict[str, str], func_names: dict[str, str], kernel_kwargs: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Run CPU simulation for each kernel and return outputs and errors."""
    sim_outputs: dict[str, np.ndarray] = {}
    sim_errors: dict[str, str] = {}
    t0 = time.monotonic()
    for kname in kernel_names:
        sim_out, sim_err = simulate_one(nki_paths[kname], func_names[kname], kernel_kwargs)
        if sim_err:
            sim_errors[kname] = sim_err
            logger.info("CPU sim FAILED for %s: %s", kname, sim_err[:200])
        else:
            sim_outputs[kname] = sim_out
    logger.info(
        "CPU simulation: %d passed, %d failed (%.1fs)", len(sim_outputs), len(sim_errors), time.monotonic() - t0
    )
    return sim_outputs, sim_errors


def _submit_compilations(
    kernel_names: list[str],
    nki_paths: dict[str, str],
    func_names: dict[str, str],
    output_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
    input_dtype_name: str,
    output_name: str,
    output_dtype: np.dtype,
    neff_dir: Path,
    scalar_dict: dict[str, float],
) -> tuple[ProcessPoolExecutor, list[Future]]:
    """Submit all kernels for parallel compilation.

    Returns:
        Tuple of (executor, list of futures).
    """
    cpu_cores = os.cpu_count() or 1
    compile_workers = min(max(cpu_cores - 1, 1), len(kernel_names))
    executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=init_compile_worker)
    futures: list[Future] = []
    for kname in kernel_names:
        compile_dir = neff_dir / Path(kname).stem
        compile_dir.mkdir(parents=True, exist_ok=True)
        futures.append(
            executor.submit(
                compile_one,
                kname,
                nki_paths[kname],
                func_names[kname],
                input_shapes,
                input_dtype_name,
                output_name,
                output_shapes[kname],
                output_dtype.str,
                str(compile_dir),
                scalar_dict,
            )
        )
    return executor, futures


def _benchmark_compiled(
    compile_futures: list[Future],
    spike: BaremetalExecutor,
    func_names: dict[str, str],
    kernel_kwargs: dict[str, Any],
    output_shapes: dict[str, tuple[int, ...]],
    output_name: str,
    output_dtype: np.dtype,
    config: BenchmarkConfig,
    sim_outputs: dict[str, np.ndarray],
    golden: np.ndarray,
) -> tuple[list[CompileResult], list[ProfileResult], list[ProfileResult]]:
    """Benchmark each kernel as it finishes compiling.

    Returns:
        Tuple of (compile_results, compile_errors, hw_results).
    """
    compile_results: list[CompileResult] = []
    compile_errors: list[ProfileResult] = []
    hw_results: list[ProfileResult] = []
    for f in as_completed(compile_futures):
        cr = f.result()
        compile_results.append(cr)
        if cr.error:
            compile_errors.append(compile_failure_result(cr, config.mac_count))
            continue
        sim_golden = sim_outputs.get(cr.kernel_name)
        effective_golden = sim_golden if sim_golden is not None else golden
        effective_atol = 1e-3 if sim_golden is not None else config.atol
        effective_rtol = 1e-3 if sim_golden is not None else config.rtol
        out = OutputSpec(name=output_name, shape=output_shapes[cr.kernel_name], dtype=output_dtype)
        bench_config = BenchmarkConfig(
            warmup=config.warmup,
            iters=config.iters,
            mac_count=config.mac_count,
            input_dtype_name=config.input_dtype_name,
            atol=effective_atol,
            rtol=effective_rtol,
        )
        hw_results.append(
            benchmark_one(spike, cr, func_names[cr.kernel_name], kernel_kwargs, out, bench_config, effective_golden)
        )
    return compile_results, compile_errors, hw_results


def _collect_compiler_logs(compile_results: list[CompileResult], neff_dir: Path, collect: bool) -> dict[str, str]:
    """Gather compiler log files if collection is enabled."""
    logs: dict[str, str] = {}
    for cr in compile_results if collect else []:
        stem = Path(cr.kernel_name).stem
        log_path = neff_dir / stem / "log-neuron-cc.txt"
        if log_path.exists():
            logs[cr.kernel_name] = log_path.read_text()
    return logs


def _build_bench_config(config: dict[str, Any]) -> BenchmarkConfig:
    """Build BenchmarkConfig from raw payload config dict."""
    return BenchmarkConfig(
        warmup=config["warmup"],
        iters=config["iters"],
        mac_count=config["mac_count"],
        input_dtype_name=config["input_dtype_name"],
        atol=config.get("atol", 1e-2),
        rtol=config.get("rtol", 1e-2),
    )


def _build_kernel_kwargs(payload: dict[str, Any], scalar_params: dict[str, float]) -> dict[str, Any]:
    """Generate input tensors and merge scalar params."""
    kernel_kwargs: dict[str, Any] = generate_tensors(payload["tensor_specs"], payload["seed"])
    for name, value in scalar_params.items():
        kernel_kwargs[name] = np.float64(value)
    return kernel_kwargs


def _compute_golden_ref(payload: dict[str, Any], kernel_kwargs: dict[str, Any]) -> np.ndarray:
    """Compute golden reference output if source is provided."""
    golden_source = payload.get("golden_source", "")
    golden_func_name = payload.get("golden_func_name", "")
    golden = np.empty(0)
    if golden_source and golden_func_name:
        golden = compute_golden(golden_source, golden_func_name, kernel_kwargs)
    return golden


def _detect_kernels(
    kernel_names: list[str], sources: dict[str, str]
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    """Detect function names and output shapes for each kernel."""
    func_names: dict[str, str] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}
    for kname in kernel_names:
        fname, oshape = detect_kernel_info(sources[kname])
        func_names[kname] = fname
        output_shapes[kname] = oshape
    return func_names, output_shapes


def _compile_and_simulate(
    kernel_names: list[str],
    sources: dict[str, str],
    kernel_kwargs: dict[str, Any],
    bench_config: BenchmarkConfig,
    scalar_params: dict[str, float],
    nki_dir: Path,
    neff_dir: Path,
) -> tuple[
    dict[str, str], dict[str, tuple[int, ...]], dict[str, np.ndarray], dict[str, str], ProcessPoolExecutor, list[Future]
]:
    """Write kernels, run CPU sim, and submit compilations.

    Returns:
        Tuple of (func_names, output_shapes, sim_outputs, sim_errors, executor, futures).
    """
    nki_paths = _write_kernel_files(kernel_names, sources, nki_dir)
    func_names, output_shapes = _detect_kernels(kernel_names, sources)

    neuron_cores = detect_neuron_cores()
    logger.info("Worker ready: %d kernels, %d CPU, %d NC", len(kernel_names), os.cpu_count() or 1, neuron_cores)

    sim_outputs, sim_errors = _run_simulations(kernel_names, nki_paths, func_names, kernel_kwargs)

    input_shapes = {k: v.shape for k, v in kernel_kwargs.items() if hasattr(v, "ndim") and v.ndim > 0}
    scalar_dict = {k: float(v) for k, v in scalar_params.items()}
    executor, futures = _submit_compilations(
        kernel_names,
        nki_paths,
        func_names,
        output_shapes,
        input_shapes,
        bench_config.input_dtype_name,
        _OUTPUT_TENSOR_NAME,
        resolve_dtype(bench_config.input_dtype_name),
        neff_dir,
        scalar_dict,
    )
    return func_names, output_shapes, sim_outputs, sim_errors, executor, futures


def _run_hw_benchmarks(
    compile_futures: list[Future],
    executor: ProcessPoolExecutor,
    func_names: dict[str, str],
    kernel_kwargs: dict[str, Any],
    output_shapes: dict[str, tuple[int, ...]],
    bench_config: BenchmarkConfig,
    sim_outputs: dict[str, np.ndarray],
    golden: np.ndarray,
) -> tuple[list[CompileResult], list[ProfileResult], list[ProfileResult]]:
    """Start BaremetalExecutor and benchmark compiled kernels.

    The first benchmark() call pays ~3s Neuron runtime init,
    which overlaps with remaining compilations.
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    output_dtype = resolve_dtype(bench_config.input_dtype_name)

    with BaremetalExecutor(verbose=0) as spike:
        compile_results, compile_errors, hw_results = _benchmark_compiled(
            compile_futures,
            spike,
            func_names,
            kernel_kwargs,
            output_shapes,
            _OUTPUT_TENSOR_NAME,
            output_dtype,
            bench_config,
            sim_outputs,
            golden,
        )
    executor.shutdown(wait=True)
    return compile_results, compile_errors, hw_results


def _run_pipeline(
    payload: dict[str, Any], bench_config: BenchmarkConfig, kernel_kwargs: dict[str, Any], golden: np.ndarray
) -> tuple[list[ProfileResult], dict[str, str], dict[str, str]]:
    """Execute the compile-simulate-benchmark pipeline.

    Returns:
        Tuple of (all_results, compiler_logs, sim_errors).
    """
    host = payload["host"]
    kernel_names = payload["kernel_names"]
    scalar_params = payload.get("scalar_params", {})
    t_start = time.monotonic()

    work_dir = Path(tempfile.mkdtemp(prefix=f"autotune-{host}-"))
    try:
        nki_dir = work_dir / "nki"
        nki_dir.mkdir()
        neff_dir = work_dir / "neff"
        neff_dir.mkdir()

        func_names, output_shapes, sim_outputs, sim_errors, executor, futures = _compile_and_simulate(
            kernel_names, payload["sources"], kernel_kwargs, bench_config, scalar_params, nki_dir, neff_dir
        )
        compile_results, compile_errors, hw_results = _run_hw_benchmarks(
            futures, executor, func_names, kernel_kwargs, output_shapes, bench_config, sim_outputs, golden
        )

        logger.info(
            "Done: %d compiled, %d benchmarked, %d errors (%.1fs)",
            len(compile_results) - len(compile_errors),
            len(hw_results),
            len(compile_errors),
            time.monotonic() - t_start,
        )

        compiler_logs = _collect_compiler_logs(
            compile_results, neff_dir, payload["config"].get("collect_compiler_logs", False)
        )
        all_results = compile_errors + hw_results
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return all_results, compiler_logs, sim_errors


def worker_main() -> None:
    """Compile and benchmark NKI kernels on a remote host.

    Reads a JSON payload from stdin containing kernel sources and
    benchmark config. Writes results JSON to stdout.
    """
    payload = _parse_payload()
    _setup_env(payload)
    logging.basicConfig(level=logging.INFO, format=f"[{payload['host']}] %(message)s")

    bench_config = _build_bench_config(payload["config"])
    scalar_params = payload.get("scalar_params", {})
    kernel_kwargs = _build_kernel_kwargs(payload, scalar_params)
    golden = _compute_golden_ref(payload, kernel_kwargs)

    all_results, compiler_logs, sim_errors = _run_pipeline(payload, bench_config, kernel_kwargs, golden)

    output = {
        "host": payload["host"],
        "results": [r._asdict() for r in all_results],
        "compiler_logs": compiler_logs,
        "sim_errors": sim_errors,
    }
    sys.stdout.buffer.write(json.dumps(output).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    worker_main()
