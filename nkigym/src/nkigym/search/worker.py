"""Remote worker for NKI kernel compilation and benchmarking.

Sent to gym nodes over SSH as part of a file bundle containing the full
nkigym package.  Workers import directly from nkigym — no code duplication.

Protocol:
    stdin:  JSON payload
    stdout: JSON results
    stderr: logging
"""

import glob as _glob
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

from nkigym.search.compile import (
    CompileResult,
    VariantResult,
    _BenchmarkConfig,
    _capture_error,
    _compile_failure_result,
    _compile_worker,
    _detect_neuron_cores,
    _init_compile_worker,
    _make_failure,
    _run_core_worker,
)
from nkigym.simulate import simulate_kernel

logger = logging.getLogger(__name__)


def _generate_tensors(tensor_specs: dict[str, dict], seed: int) -> dict[str, np.ndarray]:
    """Generate random input tensors from shapes, dtypes, and a seed.

    Args:
        tensor_specs: Map of tensor name to {"shape": [...], "dtype": "..."}.
        seed: Random seed for reproducible generation.

    Returns:
        Dict of tensor name to numpy array.
    """
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, spec in tensor_specs.items():
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec["dtype"])
        tensors[name] = rng.standard_normal(shape).astype(dtype)
    return tensors


def _compute_expected(user_func_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute the user's reference function to compute expected output.

    The user function uses ``nkigym.<op>(...)`` calls.  Since the full
    nkigym package is bundled, we import it directly — no stubs needed.

    Args:
        user_func_source: Source code of the user's function.
        func_name: Name of the function to call.
        kernel_kwargs: Input arrays (will be cast to float64).

    Returns:
        Expected output at float64 precision.
    """
    import nkigym

    g: dict[str, Any] = {"nkigym": nkigym, "np": np, "__builtins__": __builtins__}
    exec(user_func_source, g)  # noqa: S102
    func = g[func_name]
    f64_kwargs = {k: v.astype(np.float64) for k, v in kernel_kwargs.items()}
    return func(**f64_kwargs)


def _decode_payload(data: bytes) -> tuple[str, _BenchmarkConfig, list[str], dict[str, str], np.ndarray, str]:
    """Decode a lightweight JSON payload from the coordinator.

    Workers generate their own random tensors from the seed and compute
    expected output locally using the user function source.
    """
    payload = json.loads(data)
    config = payload["config"]
    tensor_specs = payload["tensor_specs"]
    seed = payload["seed"]
    kernel_kwargs = _generate_tensors(tensor_specs, seed)
    expected = _compute_expected(payload["user_func_source"], config["func_name"], kernel_kwargs)
    cfg = _BenchmarkConfig(
        func_name=config["func_name"],
        kernel_kwargs=kernel_kwargs,
        output_name=config["output_name"],
        output_shape=tuple(config["output_shape"]),
        output_dtype=np.dtype(config["output_dtype"]),
        warmup=config["warmup"],
        iters=config["iters"],
        mac_count=config["mac_count"],
        input_dtype_name=config["input_dtype_name"],
    )
    return (payload["host"], cfg, payload["nki_names"], payload["sources"], expected, payload["neuron_platform_target"])


def _cpu_verify_one(nki_path: str, func_name: str, kernel_kwargs: dict[str, np.ndarray], expected: np.ndarray) -> str:
    """Verify one NKI kernel via CPU simulation.

    Returns:
        Empty string on success, error traceback on failure.
    """
    try:
        nki_source = Path(nki_path).read_text()
        actual = simulate_kernel(nki_source, func_name, kernel_kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    except Exception as e:
        return _capture_error(e)
    return ""


def worker_main() -> None:
    """CPU-verify, compile, and benchmark NKI kernels on a remote host.

    Reads a JSON payload from stdin containing config, NKI sources,
    and tensor specs.  Writes results JSON to stdout.
    All logging goes to stderr.
    """
    data = sys.stdin.buffer.read()
    host, cfg, nki_names, host_sources, expected, neuron_target = _decode_payload(data)

    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = neuron_target

    all_results: list[VariantResult] = []
    compile_elapsed = 0.0
    benchmark_elapsed = 0.0
    compiler_logs: dict[str, str] = {}

    work_dir = Path(tempfile.mkdtemp(prefix=f"nkigym-{host}-"))
    try:
        nki_dir = work_dir / "nki"
        nki_dir.mkdir()
        neff_dir = work_dir / "neff"
        neff_dir.mkdir()

        nki_paths: list[str] = []
        for name in nki_names:
            p = nki_dir / name
            p.write_text(host_sources[name])
            nki_paths.append(str(p))

        neuron_cores = _detect_neuron_cores()
        cpu_cores = os.cpu_count() or 1

        logging.basicConfig(level=logging.INFO, format=f"[{host}] %(message)s")
        logger.info(
            "Worker starting: %d variants, %d CPU cores, %d Neuron cores, neff_dir=%s",
            len(nki_paths),
            cpu_cores,
            neuron_cores,
            neff_dir,
        )

        sim_kwargs = {k: v.astype(np.float64) for k, v in cfg.kernel_kwargs.items()}
        input_shapes = {k: v.shape for k, v in cfg.kernel_kwargs.items()}

        t0 = time.monotonic()

        """
        Pipeline parallel CPU verification into parallel compilation:
        run verify across half the CPU cores, and as each variant passes,
        immediately submit it for compilation on the other half.  Both
        pools run concurrently so compilation starts while verification
        is still in flight.
        """
        verify_workers = max(cpu_cores // 2, 1)
        compile_workers = max(cpu_cores // 2, 1)
        verify_executor = ProcessPoolExecutor(max_workers=verify_workers)
        compile_executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=_init_compile_worker)

        verify_errors: list[VariantResult] = []
        compile_futures: list[Future] = []
        verified_count = 0
        verify_futures = {
            verify_executor.submit(_cpu_verify_one, nki_path, cfg.func_name, sim_kwargs, expected): nki_path
            for nki_path in nki_paths
        }
        for f in as_completed(verify_futures):
            nki_path = verify_futures[f]
            error = f.result()
            if error:
                verify_errors.append(_make_failure(nki_path, error, cfg.mac_count))
            else:
                verified_count += 1
                compile_dir = neff_dir / Path(nki_path).stem
                compile_dir.mkdir(parents=True, exist_ok=True)
                compile_futures.append(
                    compile_executor.submit(
                        _compile_worker,
                        nki_path,
                        cfg.func_name,
                        input_shapes,
                        cfg.input_dtype_name,
                        cfg.output_name,
                        cfg.output_shape,
                        cfg.output_dtype.str,
                        str(compile_dir),
                    )
                )
        verify_executor.shutdown(wait=True)

        verify_elapsed = time.monotonic() - t0
        logger.info(
            "CPU verify: %d passed, %d failed in %.1fs (compilation already in flight)",
            verified_count,
            len(verify_errors),
            verify_elapsed,
        )

        compile_results: list[CompileResult] = []
        for f in as_completed(compile_futures):
            compile_results.append(f.result())
        compile_executor.shutdown(wait=True)

        compile_elapsed = time.monotonic() - t0
        compile_errors: list[VariantResult] = []
        nki_neff_pairs: list[tuple[str, str]] = []
        for cr in compile_results:
            if cr.error:
                compile_errors.append(_compile_failure_result(cr, cfg.mac_count))
            else:
                nki_neff_pairs.append((cr.nki_path, cr.neff_path))

        logger.info(
            "Verify+compile: %d compiled, %d compile errors in %.1fs total",
            len(nki_neff_pairs),
            len(compile_errors),
            compile_elapsed,
        )

        t2 = time.monotonic()
        num_hw_workers = min(neuron_cores, len(nki_neff_pairs))
        hw_results: list[VariantResult] = []
        if nki_neff_pairs:
            core_batches: list[list[tuple[str, str]]] = [[] for _ in range(num_hw_workers)]
            for i, pair in enumerate(nki_neff_pairs):
                core_batches[i % num_hw_workers].append(pair)

            with ProcessPoolExecutor(max_workers=num_hw_workers) as executor:
                hw_futures: list[Future] = []
                for core_id, batch in enumerate(core_batches):
                    if batch:
                        hw_futures.append(executor.submit(_run_core_worker, core_id, batch, cfg))
                for hw_future in as_completed(hw_futures):
                    hw_results.extend(hw_future.result())

        benchmark_elapsed = time.monotonic() - t2
        all_results = verify_errors + compile_errors + hw_results
        logger.info("Benchmark: %d results in %.1fs", len(hw_results), benchmark_elapsed)

        """
        Collect compiler log files from each variant's compile directory
        before cleanup.
        """
        compiler_logs = {}
        for cr in compile_results:
            variant_stem = Path(cr.nki_path).stem
            compile_dir = neff_dir / variant_stem
            log_candidates = _glob.glob(str(compile_dir / "log-neuron*"))
            if log_candidates:
                try:
                    compiler_logs[variant_stem] = Path(log_candidates[0]).read_text()
                except OSError:
                    pass
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    output = {"host": host, "results": [r._asdict() for r in all_results], "compiler_logs": compiler_logs}
    result_bytes = json.dumps(output).encode("utf-8")
    sys.stdout.buffer.write(result_bytes)
    sys.stdout.buffer.flush()
    logger.info(
        "Worker done: %d results (verify+compile %.1fs + benchmark %.1fs = %.1fs total)",
        len(all_results),
        compile_elapsed,
        benchmark_elapsed,
        compile_elapsed + benchmark_elapsed,
    )


if __name__ == "__main__":
    worker_main()
