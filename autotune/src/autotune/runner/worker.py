"""Remote worker entry point for NKI kernel compilation and benchmarking.

Runs on remote Trn nodes. Receives a JSON payload on stdin containing
per-kernel job configs and benchmark settings. Compiles each kernel
to NEFF, benchmarks on Neuron hardware, and writes JSON results
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
from autotune.runner.compare import assert_close
from autotune.runner.compile import compile_one, init_compile_worker
from autotune.runner.detect import detect_kernel_info, detect_mac_count, detect_neuron_cores
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


def _cpu_sim_status(sim_output: np.ndarray, sim_error: str, golden: np.ndarray, atol: float, rtol: float) -> dict:
    """Build a structured CPU simulation status dict.

    Compares NKI CPU sim output against golden numpy output.

    Returns:
        Dict with 'passed' (bool) and either margin details or 'error' string.
    """
    result: dict = {"passed": False, "error": ""}
    if sim_error:
        result["error"] = sim_error[:200]
    else:
        try:
            result = assert_close(sim_output, golden, atol=atol, rtol=rtol)
        except AssertionError as e:
            result["error"] = str(e)[:200]
    return result


def _process_kernel_job(kname: str, job: dict[str, Any], seed: int, nki_dir: Path) -> dict[str, Any]:
    """Process a single kernel job: generate tensors, golden, detect info, run CPU sim.

    Returns a dict with all per-kernel data needed for compile and benchmark.
    """
    source = job["source"]
    filename = kname if kname.endswith(".py") else f"{kname}.py"
    nki_path = nki_dir / filename
    nki_path.write_text(source)

    func_name, output_shape = detect_kernel_info(source)
    mac_count = detect_mac_count(source)

    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in job["tensor_specs"].items()}
    kwargs = generate_tensors(tensor_specs, seed)

    golden = compute_golden(job["golden_source"], job["golden_func_name"], kwargs)

    sim_output, sim_error = simulate_one(str(nki_path), func_name, kwargs)
    cpu_sim = _cpu_sim_status(sim_output, sim_error, golden, job["atol"], job["rtol"])

    input_dtype_name = next(iter(job["tensor_specs"].values()))[1]
    return {
        "nki_path": str(nki_path),
        "func_name": func_name,
        "output_shape": output_shape,
        "mac_count": mac_count,
        "kwargs": kwargs,
        "cpu_sim": cpu_sim,
        "input_dtype_name": input_dtype_name,
    }


def _submit_compilations(
    kernel_data: dict[str, dict[str, Any]], neff_dir: Path
) -> tuple[ProcessPoolExecutor, list[Future]]:
    """Submit all kernels for parallel compilation.

    Returns:
        Tuple of (executor, list of futures).
    """
    cpu_cores = os.cpu_count() or 1
    compile_workers = min(max(cpu_cores - 1, 1), len(kernel_data))
    executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=init_compile_worker)
    futures: list[Future] = []
    for kname, kd in kernel_data.items():
        compile_dir = neff_dir / Path(kname).stem
        compile_dir.mkdir(parents=True, exist_ok=True)
        input_shapes = {k: v.shape for k, v in kd["kwargs"].items() if hasattr(v, "ndim") and v.ndim > 0}
        futures.append(
            executor.submit(
                compile_one,
                kname,
                kd["nki_path"],
                kd["func_name"],
                input_shapes,
                kd["input_dtype_name"],
                _OUTPUT_TENSOR_NAME,
                kd["output_shape"],
                kd["input_dtype_name"],
                str(compile_dir),
                {},
            )
        )
    return executor, futures


def _benchmark_compiled(
    compile_futures: list[Future],
    spike: BaremetalExecutor,
    kernel_data: dict[str, dict[str, Any]],
    warmup: int,
    iters: int,
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
        kd = kernel_data[cr.kernel_name]
        if cr.error:
            compile_errors.append(compile_failure_result(cr, kd["mac_count"], cpu_sim=kd["cpu_sim"]))
            continue
        out = OutputSpec(
            name=_OUTPUT_TENSOR_NAME, shape=kd["output_shape"], dtype=resolve_dtype(kd["input_dtype_name"])
        )
        bench_config = BenchmarkConfig(
            warmup=warmup, iters=iters, mac_count=kd["mac_count"], input_dtype_name=kd["input_dtype_name"]
        )
        hw_results.append(benchmark_one(spike, cr, kd["func_name"], kd["kwargs"], out, bench_config, kd["cpu_sim"]))
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


def _run_hw_benchmarks(
    executor: ProcessPoolExecutor,
    futures: list[Future],
    kernel_data: dict[str, dict[str, Any]],
    config: dict[str, Any],
    neff_dir: Path,
) -> tuple[list[ProfileResult], dict[str, str]]:
    """Compile, benchmark on hardware, and collect logs.

    Returns:
        Tuple of (all_results, compiler_logs).
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

    with BaremetalExecutor(verbose=0) as spike:
        compile_results, compile_errors, hw_results = _benchmark_compiled(
            futures, spike, kernel_data, config["warmup"], config["iters"]
        )
    executor.shutdown(wait=True)

    compiler_logs = _collect_compiler_logs(compile_results, neff_dir, config.get("collect_compiler_logs", False))
    return compile_errors + hw_results, compiler_logs


def _run_pipeline(payload: dict[str, Any]) -> tuple[list[ProfileResult], dict[str, str]]:
    """Execute the per-kernel compile-simulate-benchmark pipeline.

    Returns:
        Tuple of (all_results, compiler_logs).
    """
    host = payload["host"]
    kernel_jobs = payload["kernel_jobs"]
    seed = payload["seed"]
    config = payload["config"]
    t_start = time.monotonic()

    work_dir = Path(tempfile.mkdtemp(prefix=f"autotune-{host}-"))
    try:
        nki_dir = work_dir / "nki"
        nki_dir.mkdir()
        neff_dir = work_dir / "neff"
        neff_dir.mkdir()

        neuron_cores = detect_neuron_cores()
        logger.info("Worker ready: %d kernels, %d CPU, %d NC", len(kernel_jobs), os.cpu_count() or 1, neuron_cores)

        kernel_data: dict[str, dict[str, Any]] = {}
        for kname, job in kernel_jobs.items():
            kernel_data[kname] = _process_kernel_job(kname, job, seed, nki_dir)
        logger.info("CPU sim done: %d kernels", len(kernel_data))

        executor, futures = _submit_compilations(kernel_data, neff_dir)
        all_results, compiler_logs = _run_hw_benchmarks(executor, futures, kernel_data, config, neff_dir)

        logger.info("Done: %d results (%.1fs)", len(all_results), time.monotonic() - t_start)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return all_results, compiler_logs


def worker_main() -> None:
    """Compile and benchmark NKI kernels on a remote host.

    Reads a JSON payload from stdin containing per-kernel job configs
    and benchmark settings. Writes results JSON to stdout.
    """
    payload = _parse_payload()
    _setup_env(payload)
    logging.basicConfig(level=logging.INFO, format=f"[{payload['host']}] %(message)s")

    all_results, compiler_logs = _run_pipeline(payload)

    output = {"host": payload["host"], "results": [r._asdict() for r in all_results], "compiler_logs": compiler_logs}
    sys.stdout.buffer.write(json.dumps(output).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    worker_main()
