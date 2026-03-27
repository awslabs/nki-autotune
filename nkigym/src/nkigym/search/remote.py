"""Distribute compilation and benchmarking across remote Trainium hosts.

The coordinator renders NKI source files in memory, then sends them to
remote workers over SSH stdin as a lightweight JSON payload containing
only source code, shapes, dtypes, a random seed, and the user function.
Workers generate their own tensors locally.  Each worker CPU-verifies,
compiles, and benchmarks, then returns results as JSON on stdout.
No shared filesystem or nkigym installation required on workers —
the worker script is sent over SSH.
"""

import base64
import json
import logging
import subprocess
import threading
import time
from pathlib import Path

from nkigym.search.compile import VariantResult, _BenchmarkConfig, _make_failure

logger = logging.getLogger(__name__)

_REQUIRED_CONFIG_KEYS = {"venv_python", "hosts", "ssh_timeout_sec", "neuron_platform_target"}

"""
Cache the worker file bundle to avoid re-reading for each host.
The bundle is a JSON dict mapping relative path -> source code.
The entire nkigym package is included so workers always have the
same code as the coordinator — single source of truth, zero
maintenance when nkigym changes.
"""
_worker_bundle_cache: bytes | None = None

_NKIGYM_ROOT = Path(__file__).parent.parent


def _get_worker_bundle() -> bytes:
    """Build a JSON bundle of worker.py + the entire nkigym package.

    Returns base64-encoded bytes (single line, no newlines) of a JSON
    dict mapping relative path to source code.  The bundle is sent as
    the first line of stdin; the bootstrap reads it, unpacks files to
    a temp directory preserving directory structure, adds it to
    sys.path, and execs worker.py.  The JSON work payload follows on
    the remaining stdin.
    """
    global _worker_bundle_cache  # noqa: PLW0603
    if _worker_bundle_cache is None:
        bundle: dict[str, str] = {}
        bundle["worker.py"] = (_NKIGYM_ROOT / "search" / "worker.py").read_text()
        for py_file in sorted(_NKIGYM_ROOT.rglob("*.py")):
            rel = py_file.relative_to(_NKIGYM_ROOT.parent)
            bundle[str(rel)] = py_file.read_text()
        _worker_bundle_cache = base64.b64encode(json.dumps(bundle).encode("utf-8"))
    return _worker_bundle_cache


def load_remote_config(config_path: Path) -> dict:
    """Load remote worker configuration from a JSON file.

    Args:
        config_path: Explicit path to the JSON config file.

    Returns:
        Dict with keys: venv_python, hosts, ssh_timeout_sec, neuron_platform_target.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    missing = _REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        raise ValueError(f"remote config {config_path} missing keys: {missing}")
    return cfg


def _encode_payload(
    host: str,
    cfg: _BenchmarkConfig,
    nki_names: list[str],
    sources: dict[str, str],
    neuron_target: str,
    user_func_source: str,
    seed: int,
) -> bytes:
    """Build a lightweight JSON payload for a remote worker.

    No tensor data is included — workers generate their own random
    inputs from the shared seed and compute expected output locally
    using the user function source.

    Args:
        host: Hostname identifier for this worker.
        cfg: Benchmark configuration with kernel kwargs.
        nki_names: NKI source filenames assigned to this worker.
        sources: Map of NKI filename to source code.
        neuron_target: Neuron platform target (e.g. "trn2").
        user_func_source: Source code of the user's reference function.
        seed: Random seed for reproducible tensor generation.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    payload = {
        "host": host,
        "neuron_platform_target": neuron_target,
        "config": {
            "func_name": cfg.func_name,
            "output_name": cfg.output_name,
            "output_shape": list(cfg.output_shape),
            "output_dtype": cfg.output_dtype.str,
            "warmup": cfg.warmup,
            "iters": cfg.iters,
            "mac_count": cfg.mac_count,
            "input_dtype_name": cfg.input_dtype_name,
        },
        "nki_names": nki_names,
        "sources": {n: sources[n] for n in nki_names},
        "tensor_specs": {
            name: {"shape": list(arr.shape), "dtype": arr.dtype.str} for name, arr in cfg.kernel_kwargs.items()
        },
        "user_func_source": user_func_source,
        "seed": seed,
    }
    return json.dumps(payload).encode("utf-8")


def distribute(
    nki_names: list[str],
    cfg: _BenchmarkConfig,
    hosts: list[str],
    sources: dict[str, str],
    remote_cfg: dict,
    user_func_source: str,
    seed: int,
) -> tuple[list[VariantResult], dict[str, str]]:
    """Distribute CPU verification, compilation, and benchmarking across remote hosts.

    Sends each host the self-contained worker script and a lightweight
    JSON payload over SSH stdin.  No tensor data is transferred — workers
    generate their own random inputs from the shared seed and compute
    expected output locally using the user function source.

    Args:
        nki_names: NKI source filenames.
        cfg: Benchmark configuration (kernel kwargs, warmup, iters, etc.).
        hosts: SSH hostnames (e.g. ``["gym-1", "gym-2", ...]``).
        sources: Map of NKI filename to source code.
        remote_cfg: Remote config dict from ``load_remote_config()``.
        user_func_source: Source code of the user's reference function.
        seed: Random seed for reproducible tensor generation.

    Returns:
        Tuple of (results, compiler_logs) where compiler_logs maps variant
        stem to the contents of its ``log-neuron-cc.txt`` file.
    """
    if not nki_names:
        return [], {}

    venv_python = remote_cfg["venv_python"]
    ssh_timeout = remote_cfg["ssh_timeout_sec"]
    neuron_target = remote_cfg["neuron_platform_target"]
    b64_bundle = _get_worker_bundle()

    active_hosts = hosts[: len(nki_names)] if len(nki_names) < len(hosts) else hosts
    host_assignments: dict[str, list[str]] = {h: [] for h in active_hosts}
    for i, nki_name in enumerate(nki_names):
        host = active_hosts[i % len(active_hosts)]
        host_assignments[host].append(nki_name)

    logger.info(
        "Distributing %d variants across %d hosts: %s",
        len(nki_names),
        len(active_hosts),
        ", ".join(f"{h}({len(host_assignments[h])})" for h in active_hosts),
    )

    def _feed_stdin(proc: subprocess.Popen, bundle_line: bytes, payload: bytes) -> None:
        """Write bundle + payload to proc stdin in a background thread.

        First line: base64-encoded file bundle (read by bootstrap).
        Remaining bytes: JSON work payload (read by worker.py).
        """
        try:
            proc.stdin.write(bundle_line + b"\n")
            proc.stdin.write(payload)
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()

    procs: dict[str, subprocess.Popen] = {}
    writers: list[threading.Thread] = []
    try:
        for host, names in host_assignments.items():
            payload = _encode_payload(host, cfg, names, sources, neuron_target, user_func_source, seed)
            """
            Bootstrap reads the file bundle from the first line of stdin,
            unpacks all files preserving directory structure, adds the
            temp dir to sys.path, and execs worker.py.  The remaining
            stdin (JSON work payload) is read by worker.py.
            """
            bootstrap = (
                "import base64,json,sys,os,tempfile;"
                "b=json.loads(base64.b64decode(sys.stdin.buffer.readline()));"
                "d=tempfile.mkdtemp();"
                "[os.makedirs(os.path.join(d,os.path.dirname(n)),exist_ok=True) or "
                'open(os.path.join(d,n),"w").write(s) for n,s in b.items()];'
                "sys.path.insert(0,d);"
                'exec(open(os.path.join(d,"worker.py")).read())'
            )
            cmd = [
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "BatchMode=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                host,
                f"{venv_python} -c '{bootstrap}'",
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            t = threading.Thread(target=_feed_stdin, args=(proc, b64_bundle, payload), daemon=True)
            t.start()
            writers.append(t)
            procs[host] = proc

        for t in writers:
            t.join()

        """
        _feed_stdin closes proc.stdin; clear the reference so
        proc.communicate() does not attempt to flush/close it again.
        """
        for proc in procs.values():
            proc.stdin = None

        t0 = time.monotonic()
        all_results: list[VariantResult] = []
        all_compiler_logs: dict[str, str] = {}
        for host, proc in procs.items():
            try:
                stdout_data, stderr_data = proc.communicate(timeout=ssh_timeout)
                if proc.returncode != 0:
                    stderr = stderr_data.decode(errors="replace")
                    error_msg = f"SSH worker on {host} exited with code {proc.returncode}: {stderr[:2000]}"
                    logger.error(error_msg)
                    for nki_name in host_assignments[host]:
                        all_results.append(_make_failure(nki_name, error_msg, cfg.mac_count))
                    continue

                stderr_text = stderr_data.decode(errors="replace")
                for line in stderr_text.strip().splitlines():
                    logger.info("  %s", line)

                if not stdout_data:
                    error_msg = f"No results from {host} (empty stdout)"
                    logger.error(error_msg)
                    for nki_name in host_assignments[host]:
                        all_results.append(_make_failure(nki_name, error_msg, cfg.mac_count))
                    continue

                try:
                    data = json.loads(stdout_data)
                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Malformed JSON from {host}: {e}"
                    logger.error(error_msg)
                    for nki_name in host_assignments[host]:
                        all_results.append(_make_failure(nki_name, error_msg, cfg.mac_count))
                    continue
                for r in data["results"]:
                    all_results.append(VariantResult(**r))
                all_compiler_logs.update(data.get("compiler_logs", {}))
                host_elapsed = time.monotonic() - t0
                logger.info("Host %s: %d results collected (%.1fs)", host, len(data["results"]), host_elapsed)

            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                error_msg = f"SSH worker on {host} timed out after {ssh_timeout}s"
                logger.error(error_msg)
                for nki_name in host_assignments[host]:
                    all_results.append(_make_failure(nki_name, error_msg, cfg.mac_count))

    except BaseException:
        for proc in procs.values():
            proc.kill()
        for proc in procs.values():
            proc.communicate()
        raise

    elapsed = time.monotonic() - t0
    logger.info(
        "Distributed complete: %d results in %.1fs (%.1f variants/sec)",
        len(all_results),
        elapsed,
        len(nki_names) / elapsed if elapsed > 0 else 0,
    )
    return all_results, all_compiler_logs
