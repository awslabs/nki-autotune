"""Distribute compilation and benchmarking across remote Trainium hosts.

The coordinator renders NKI source files and writes them to the shared /fsx
filesystem.  This module launches SSH workers on gym hosts.  Each worker
compiles its assigned NKI sources to NEFFs using local CPU cores, then
benchmarks them on local Neuron cores, and writes results back to /fsx as JSON.
"""

import json
import logging
import os
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nkigym.search.compile import (
    CompileResult,
    VariantResult,
    _BenchmarkConfig,
    _compile_failure_result,
    _compile_worker,
    _init_compile_worker,
    _run_core_worker,
)

logger = logging.getLogger(__name__)

_VENV_PYTHON = "/fsx/venvs/kernel-env/bin/python"
_SSH_TIMEOUT_SEC = 3600
_MANIFEST_VERSION = 2
_NEURON_SYSFS = "/sys/devices/virtual/neuron_device"


def _detect_neuron_cores() -> int:
    """Detect the number of Neuron cores on the local host via sysfs.

    Counts ``neuron_core*`` directories under each ``neuronN`` device
    in ``/sys/devices/virtual/neuron_device/``.

    Returns:
        Total number of Neuron cores available.

    Raises:
        RuntimeError: If no Neuron devices are found.
    """
    import glob

    device_dirs = sorted(glob.glob(os.path.join(_NEURON_SYSFS, "neuron*")))
    if not device_dirs:
        raise RuntimeError(f"No Neuron devices found under {_NEURON_SYSFS}")
    total_cores = 0
    for dev_dir in device_dirs:
        cores = glob.glob(os.path.join(dev_dir, "neuron_core*"))
        total_cores += len(cores)
    return total_cores


def _serialize_config(cfg: _BenchmarkConfig, manifest_dir: Path) -> dict:
    """Serialize a _BenchmarkConfig to a JSON-safe dict.

    Numpy arrays in ``kernel_kwargs`` are saved as ``.npy`` files under
    *manifest_dir/tensors/*; the dict stores their paths instead.
    """
    tensor_dir = manifest_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_paths: dict[str, str] = {}
    for name, arr in cfg.kernel_kwargs.items():
        path = tensor_dir / f"{name}.npy"
        np.save(str(path), arr)
        tensor_paths[name] = str(path)
    return {
        "func_name": cfg.func_name,
        "kernel_kwargs_paths": tensor_paths,
        "output_name": cfg.output_name,
        "output_shape": list(cfg.output_shape),
        "output_dtype": cfg.output_dtype.str,
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "mac_count": cfg.mac_count,
        "input_dtype_name": cfg.input_dtype_name,
    }


def _save_expected(expected: np.ndarray, manifest_dir: Path) -> str:
    """Save the expected reference output for CPU verification.

    Args:
        expected: Reference output from running the user function at float64.
        manifest_dir: Directory for shared manifest files.

    Returns:
        Path to the saved .npy file.
    """
    path = manifest_dir / "expected.npy"
    np.save(str(path), expected)
    return str(path)


def _deserialize_config(data: dict) -> _BenchmarkConfig:
    """Reconstruct a _BenchmarkConfig from a serialized dict."""
    kwargs: dict[str, np.ndarray] = {}
    for name, path in data["kernel_kwargs_paths"].items():
        kwargs[name] = np.load(path)
    return _BenchmarkConfig(
        func_name=data["func_name"],
        kernel_kwargs=kwargs,
        output_name=data["output_name"],
        output_shape=tuple(data["output_shape"]),
        output_dtype=np.dtype(data["output_dtype"]),
        warmup=data["warmup"],
        iters=data["iters"],
        mac_count=data["mac_count"],
        input_dtype_name=data["input_dtype_name"],
    )


def _variant_to_dict(r: VariantResult) -> dict:
    """Convert a VariantResult to a JSON-safe dict."""
    return r._asdict()


def _dict_to_variant(d: dict) -> VariantResult:
    """Reconstruct a VariantResult from a dict."""
    return VariantResult(**d)


def _make_failure(nki_path: str, error: str, mac_count: int) -> VariantResult:
    """Create a failed VariantResult for a pair that could not be benchmarked."""
    return VariantResult(
        nki_path=nki_path,
        min_ms=0.0,
        mean_ms=0.0,
        p50_ms=0.0,
        p99_ms=0.0,
        mac_count=mac_count,
        mfu=0.0,
        correct=False,
        error=error,
    )


def distribute(
    nki_names: list[str],
    cfg: _BenchmarkConfig,
    hosts: list[str],
    cache_dir: Path,
    expected: np.ndarray,
    sources: dict[str, str],
) -> list[VariantResult]:
    """Distribute CPU verification, compilation, and benchmarking across remote hosts.

    Each host's assigned NKI sources are embedded directly in its manifest
    JSON on Lustre (single file per host, avoiding per-source metadata
    overhead).  Each remote host extracts sources to local ``/tmp``,
    CPU-verifies, compiles, and benchmarks.

    Args:
        nki_names: NKI source filenames.
        cfg: Benchmark configuration (kernel kwargs, warmup, iters, etc.).
        hosts: SSH hostnames (e.g. ``["gym-1", "gym-2", ...]``).
        cache_dir: Shared /fsx cache directory for manifests and results.
        expected: Reference output for CPU verification.
        sources: Map of NKI filename to source code.

    Returns:
        Merged list of VariantResult from all hosts.
    """
    if not nki_names:
        return []

    active_hosts = hosts[: len(nki_names)] if len(nki_names) < len(hosts) else hosts
    manifest_dir = cache_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_dir = cache_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config_data = _serialize_config(cfg, manifest_dir)
    expected_path = _save_expected(expected, manifest_dir)

    host_assignments: dict[str, list[str]] = {h: [] for h in active_hosts}
    for i, nki_name in enumerate(nki_names):
        host = active_hosts[i % len(active_hosts)]
        host_assignments[host].append(nki_name)

    manifest_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}
    for host, names in host_assignments.items():
        rpath = results_dir / f"{host}.json"
        host_sources = {n: sources[n] for n in names}
        manifest = {
            "version": _MANIFEST_VERSION,
            "host": host,
            "config": config_data,
            "nki_names": names,
            "sources": host_sources,
            "expected_path": expected_path,
            "results_path": str(rpath),
        }
        mpath = manifest_dir / f"{host}.json"
        mpath.write_text(json.dumps(manifest, indent=2))
        manifest_paths[host] = mpath
        result_paths[host] = rpath

    logger.info(
        "Distributing %d variants across %d hosts: %s",
        len(nki_names),
        len(active_hosts),
        ", ".join(f"{h}({len(host_assignments[h])})" for h in active_hosts),
    )

    procs: dict[str, subprocess.Popen] = {}
    for host, mpath in manifest_paths.items():
        cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            host,
            f"{_VENV_PYTHON} -m nkigym.search.remote --manifest {mpath}",
        ]
        procs[host] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    t0 = time.monotonic()
    all_results: list[VariantResult] = []
    for host, proc in procs.items():
        try:
            returncode = proc.wait(timeout=_SSH_TIMEOUT_SEC)
            if returncode != 0:
                stderr = (proc.stderr.read() or b"").decode(errors="replace")
                error_msg = f"SSH worker on {host} exited with code {returncode}: {stderr[:500]}"
                logger.error(error_msg)
                for nki_name in host_assignments[host]:
                    all_results.append(
                        _make_failure(nki_name, error_msg, cfg.mac_count)
                    )
                continue

            stderr_text = (proc.stderr.read() or b"").decode(errors="replace")
            for line in stderr_text.strip().splitlines():
                logger.info("  %s", line)

            rpath = result_paths[host]
            if not rpath.exists():
                error_msg = f"Results file missing from {host}: {rpath}"
                logger.error(error_msg)
                for nki_name in host_assignments[host]:
                    all_results.append(
                        _make_failure(nki_name, error_msg, cfg.mac_count)
                    )
                continue

            host_elapsed = time.monotonic() - t0
            data = json.loads(rpath.read_text())
            for r in data["results"]:
                all_results.append(_dict_to_variant(r))
            logger.info("Host %s: %d results collected (%.1fs)", host, len(data["results"]), host_elapsed)

        except subprocess.TimeoutExpired:
            proc.kill()
            error_msg = f"SSH worker on {host} timed out after {_SSH_TIMEOUT_SEC}s"
            logger.error(error_msg)
            for nki_name in host_assignments[host]:
                all_results.append(
                    _make_failure(nki_name, error_msg, cfg.mac_count)
                )

    elapsed = time.monotonic() - t0
    logger.info(
        "Distributed complete: %d results in %.1fs (%.1f variants/sec)",
        len(all_results),
        elapsed,
        len(nki_names) / elapsed if elapsed > 0 else 0,
    )
    return all_results


def _collect_hw_results(hw_futures: list[Future]) -> list[VariantResult]:
    """Wait for all hardware benchmark futures and collect results."""
    results: list[VariantResult] = []
    for hw_future in as_completed(hw_futures):
        results.extend(hw_future.result())
    return results


def _cpu_verify_one(nki_path: str, func_name: str, kernel_kwargs: dict[str, np.ndarray], expected: np.ndarray) -> str:
    """Verify one NKI kernel via CPU simulation.

    Returns:
        Empty string on success, error traceback on failure.
    """
    from nkigym.search.compile import _capture_error
    from nkigym.simulate import simulate_kernel

    try:
        nki_source = Path(nki_path).read_text()
        actual = simulate_kernel(nki_source, func_name, kernel_kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    except Exception as e:
        return _capture_error(e)
    return ""


def worker_main(manifest_path: str) -> None:
    """CPU-verify, compile, and benchmark NKI kernels on a remote host.

    This is the entry point when invoked via SSH as::

        python -m nkigym.search.remote --manifest /fsx/.../gym-1.json

    Each worker: (1) CPU-verifies NKI sources against reference output,
    (2) compiles verified sources to NEFFs using local CPU cores,
    (3) benchmarks NEFFs on local Neuron cores.  Results are written
    to the path specified in the manifest.
    """
    import tempfile

    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

    manifest = json.loads(Path(manifest_path).read_text())
    host = manifest["host"]
    cfg = _deserialize_config(manifest["config"])
    nki_names: list[str] = manifest["nki_names"]
    expected = np.load(manifest["expected_path"])
    results_path = Path(manifest["results_path"])

    work_dir = Path(tempfile.mkdtemp(prefix=f"nkigym-{host}-"))
    nki_dir = work_dir / "nki"
    nki_dir.mkdir()
    neff_dir = work_dir / "neff"
    neff_dir.mkdir()

    host_sources: dict[str, str] = manifest["sources"]
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
        len(nki_paths), cpu_cores, neuron_cores, neff_dir,
    )

    sim_kwargs = {k: v.astype(np.float64) for k, v in cfg.kernel_kwargs.items()}
    input_shapes = {k: v.shape for k, v in cfg.kernel_kwargs.items()}

    t0 = time.monotonic()

    """
    Pipeline CPU verification with compilation: as each variant passes
    verification, immediately submit it for compilation.  This overlaps
    the sequential ~435ms-per-variant CPU simulation with the parallel
    compilation, hiding most of the verification latency.
    """
    compile_workers = max(cpu_cores - 1, 1)
    compile_executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=_init_compile_worker)
    verify_errors: list[VariantResult] = []
    compile_futures: list[Future] = []
    verified_count = 0
    for nki_path in nki_paths:
        error = _cpu_verify_one(nki_path, cfg.func_name, sim_kwargs, expected)
        if error:
            verify_errors.append(_make_failure(nki_path, error, cfg.mac_count))
        else:
            verified_count += 1
            compile_dir = neff_dir / Path(nki_path).stem
            compile_dir.mkdir(parents=True, exist_ok=True)
            compile_futures.append(compile_executor.submit(
                _compile_worker,
                nki_path,
                cfg.func_name,
                input_shapes,
                cfg.input_dtype_name,
                cfg.output_name,
                cfg.output_shape,
                cfg.output_dtype.str,
                str(compile_dir),
            ))

    verify_elapsed = time.monotonic() - t0
    logger.info(
        "CPU verify: %d passed, %d failed in %.1fs (compilation already in flight)",
        verified_count, len(verify_errors), verify_elapsed,
    )

    compile_results: list[CompileResult] = []
    for f in as_completed(compile_futures):
        compile_results.append(f.result())
    compile_executor.shutdown(wait=False)

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
        len(nki_neff_pairs), len(compile_errors), compile_elapsed,
    )

    t2 = time.monotonic()
    """
    Cap hw workers to avoid excessive process-spawn overhead.  Each
    BaremetalExecutor init takes seconds; with 128 cores and ~135 variants,
    using all cores means 128 processes for 1-2 variants each.  Capping at
    min(neuron_cores, num_variants, 32) keeps init cost manageable while
    still providing enough parallelism.
    """
    num_hw_workers = min(neuron_cores, len(nki_neff_pairs), 16)
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
            hw_results = _collect_hw_results(hw_futures)

    benchmark_elapsed = time.monotonic() - t2
    all_results = verify_errors + compile_errors + hw_results
    logger.info(
        "Benchmark: %d results in %.1fs",
        len(hw_results), benchmark_elapsed,
    )

    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)

    output = {
        "host": host,
        "results": [_variant_to_dict(r) for r in all_results],
    }
    tmp_path = results_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(output, indent=2))
    tmp_path.rename(results_path)
    logger.info(
        "Worker done: %d results (verify+compile %.1fs + benchmark %.1fs = %.1fs total)",
        len(all_results), compile_elapsed, benchmark_elapsed,
        compile_elapsed + benchmark_elapsed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NKI Gym remote worker")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    args = parser.parse_args()
    worker_main(args.manifest)
