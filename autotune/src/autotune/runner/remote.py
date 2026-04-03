"""Distribute NKI kernel profiling across remote Trainium hosts via SSH.

The coordinator sends kernel source code and benchmark config to remote
workers as a JSON payload over SSH stdin. Workers compile, benchmark,
and return results as JSON on stdout.
"""

import base64
import json
import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autotune.runner.types import _DEFAULT_VENV_PYTHON, ProfileResult, make_failure

logger = logging.getLogger(__name__)

_REQUIRED_CONFIG_KEYS = {"hosts", "ssh_timeout_sec"}

_AUTOTUNE_ROOT = Path(__file__).parent.parent

_worker_bundle_cache: bytes = b""


def _get_worker_bundle() -> bytes:
    """Build a base64-encoded bundle of the autotune package.

    The bundle is a JSON dict mapping relative path to source code.
    Sent as the first line of stdin to the bootstrap script on each
    remote host.
    """
    global _worker_bundle_cache  # noqa: PLW0603
    if not _worker_bundle_cache:
        bundle: dict[str, str] = {}
        bundle["worker.py"] = (_AUTOTUNE_ROOT / "runner" / "worker.py").read_text()
        for py_file in sorted(_AUTOTUNE_ROOT.rglob("*.py")):
            rel = py_file.relative_to(_AUTOTUNE_ROOT.parent)
            bundle[str(rel)] = py_file.read_text()
        _worker_bundle_cache = base64.b64encode(json.dumps(bundle).encode("utf-8"))
    return _worker_bundle_cache


def _feed_stdin(proc: subprocess.Popen[bytes], bundle_line: bytes, payload: bytes) -> None:
    """Write bundle + payload to proc stdin in a background thread.

    First line: base64-encoded file bundle (read by bootstrap).
    Remaining bytes: JSON work payload (read by worker.py).
    """
    stdin = proc.stdin
    if stdin is None:
        raise RuntimeError("Process stdin is None — cannot feed payload")
    try:
        stdin.write(bundle_line + b"\n")
        stdin.write(payload)
    except BrokenPipeError:
        logger.warning("Broken pipe writing to SSH process")
    finally:
        stdin.close()


def _fail_host(results: list[ProfileResult], kernel_names: list[str], error_msg: str, mac_count: int) -> None:
    """Append a failure ProfileResult for every kernel assigned to a host."""
    for kname in kernel_names:
        results.append(make_failure(kname, error_msg, mac_count))


def load_remote_config(config_path: str) -> dict[str, Any]:
    """Load remote worker configuration from a JSON file.

    Expected keys: ``hosts``, ``ssh_timeout_sec``.
    Optional: ``venv_python``, ``neuron_platform_target``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    missing = _REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        raise ValueError(f"remote config {config_path} missing keys: {missing}")
    cfg.setdefault("neuron_platform_target", "trn2")
    return cfg


def _build_host_assignments(kernel_names: list[str], hosts: list[str]) -> dict[str, list[str]]:
    """Round-robin distribute shuffled kernel names across hosts."""
    shuffled = list(kernel_names)
    random.shuffle(shuffled)
    active_hosts = hosts[: len(kernel_names)] if len(kernel_names) < len(hosts) else hosts
    assignments: dict[str, list[str]] = {h: [] for h in active_hosts}
    for i, kname in enumerate(shuffled):
        host = active_hosts[i % len(active_hosts)]
        assignments[host].append(kname)
    return assignments


_BOOTSTRAP_SCRIPT = (
    "import base64,json,sys,os,tempfile,importlib.util;"
    "b=json.loads(base64.b64decode(sys.stdin.buffer.readline()));"
    "d=tempfile.mkdtemp();"
    "[os.makedirs(os.path.join(d,os.path.dirname(n)),exist_ok=True) or "
    'open(os.path.join(d,n),"w").write(s) for n,s in b.items()];'
    "sys.path.insert(0,d);"
    'spec=importlib.util.spec_from_file_location("worker",os.path.join(d,"worker.py"));'
    "mod=importlib.util.module_from_spec(spec);"
    'sys.modules["worker"]=mod;'
    "spec.loader.exec_module(mod);"
    "mod.worker_main()"
)


def _build_ssh_cmd(host: str, venv_python: str) -> list[str]:
    """Build the SSH command list for a remote worker."""
    return [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=/tmp/autotune-ssh-%r@%h:%p",
        "-o",
        "ControlPersist=300",
        host,
        f"{venv_python} -c '{_BOOTSTRAP_SCRIPT}'",
    ]


def _launch_ssh_workers(
    host_assignments: dict[str, list[str]], kernels: dict[str, str], payload_base: dict[str, Any], venv_python: str
) -> tuple[dict[str, subprocess.Popen[bytes]], list[threading.Thread]]:
    """Launch SSH workers for each host and feed payloads.

    Returns:
        Tuple of (host-to-process map, list of writer threads).
    """
    b64_bundle = _get_worker_bundle()
    procs: dict[str, subprocess.Popen[bytes]] = {}
    writers: list[threading.Thread] = []

    for host, names in host_assignments.items():
        payload = dict(payload_base)
        payload["host"] = host
        payload["kernel_names"] = names
        payload["sources"] = {n: kernels[n] for n in names}
        payload_bytes = json.dumps(payload).encode("utf-8")

        cmd = _build_ssh_cmd(host, venv_python)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t = threading.Thread(target=_feed_stdin, args=(proc, b64_bundle, payload_bytes), daemon=True)
        t.start()
        writers.append(t)
        procs[host] = proc

    return procs, writers


def _read_host_output(
    host: str, proc: subprocess.Popen[bytes], timeout: int, out: dict[str, tuple[bytes, bytes, int]]
) -> None:
    """Read stdout/stderr from a single host process into out dict."""
    try:
        stdout_data, stderr_data = proc.communicate(timeout=timeout)
        out[host] = (stdout_data, stderr_data, proc.returncode)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        out[host] = (b"", b"", -1)


def _collect_host_outputs(
    procs: dict[str, subprocess.Popen[bytes]], timeout: int
) -> dict[str, tuple[bytes, bytes, int]]:
    """Read all host outputs concurrently via threads.

    Workers produce ~3MB of output (results + compiler logs) and the OS
    pipe buffer is only 64KB. Sequential communicate() causes pipeline
    stalls. Threading the reads eliminates this.
    """
    host_outputs: dict[str, tuple[bytes, bytes, int]] = {}
    readers: list[threading.Thread] = []
    for host, proc in procs.items():
        rt = threading.Thread(target=_read_host_output, args=(host, proc, timeout, host_outputs), daemon=True)
        rt.start()
        readers.append(rt)
    for rt in readers:
        rt.join(timeout=timeout)
    return host_outputs


def _host_error_message(host: str, returncode: int, stderr_data: bytes, stdout_data: bytes) -> str:
    """Determine error message for a host, or empty string if success."""
    _ERROR_TEMPLATES = {-1: "SSH worker on {host} timed out"}
    msg = _ERROR_TEMPLATES.get(returncode, "")
    if not msg and returncode != 0:
        stderr = stderr_data.decode(errors="replace")
        msg = f"SSH worker on {host} exited {returncode}: {stderr[:2000]}"
    if not msg and not stdout_data:
        msg = f"No results from {host} (empty stdout)"
    return msg


def _parse_host_result(
    host: str,
    stdout_data: bytes,
    stderr_data: bytes,
    assigned_kernels: list[str],
    mac_count: int,
    all_results: list[ProfileResult],
    all_compiler_logs: dict[str, str],
    all_sim_errors: dict[str, str],
) -> None:
    """Parse one host's JSON output and append to aggregate lists."""
    stderr_text = stderr_data.decode(errors="replace")
    for line in stderr_text.strip().splitlines():
        logger.info("  [%s] %s", host, line)

    try:
        data = json.loads(stdout_data)
    except (json.JSONDecodeError, ValueError) as e:
        _fail_host(all_results, assigned_kernels, f"Malformed JSON from {host}: {e}", mac_count)
        return

    for r in data["results"]:
        all_results.append(ProfileResult(**r))
    for kname, log_text in data.get("compiler_logs", {}).items():
        all_compiler_logs[kname] = log_text
    for kname, err_text in data.get("sim_errors", {}).items():
        all_sim_errors[kname] = err_text
    logger.info("Host %s: %d results", host, len(data["results"]))


def _process_host_outputs(
    procs: dict[str, subprocess.Popen[bytes]],
    host_outputs: dict[str, tuple[bytes, bytes, int]],
    host_assignments: dict[str, list[str]],
    mac_count: int,
) -> tuple[list[ProfileResult], dict[str, str], dict[str, str]]:
    """Process outputs from all hosts into aggregate results."""
    all_results: list[ProfileResult] = []
    all_compiler_logs: dict[str, str] = {}
    all_sim_errors: dict[str, str] = {}

    for host in procs:
        stdout_data, stderr_data, returncode = host_outputs.get(host, (b"", b"", -1))
        error = _host_error_message(host, returncode, stderr_data, stdout_data)
        if error:
            logger.error(error)
            _fail_host(all_results, host_assignments[host], error, mac_count)
            continue
        _parse_host_result(
            host,
            stdout_data,
            stderr_data,
            host_assignments[host],
            mac_count,
            all_results,
            all_compiler_logs,
            all_sim_errors,
        )

    return all_results, all_compiler_logs, all_sim_errors


def _write_cache_files(cache_dir: str, kernels: dict[str, str], compiler_logs: dict[str, str]) -> None:
    """Write kernel sources and compiler logs to cache directory."""
    nki_dir = os.path.join(cache_dir, "nki")
    neff_dir = os.path.join(cache_dir, "neff")
    os.makedirs(nki_dir, exist_ok=True)
    os.makedirs(neff_dir, exist_ok=True)

    for kname, source in kernels.items():
        with open(os.path.join(nki_dir, kname), "w") as f:
            f.write(source)

    for kname, log_text in compiler_logs.items():
        stem = Path(kname).stem
        log_dir = os.path.join(neff_dir, stem)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "log-neuron-cc.txt"), "w") as f:
            f.write(log_text)


def _write_results_json(
    cache_dir: str, kernels: dict[str, str], results: list[ProfileResult], profiler: "RemoteProfiler"
) -> None:
    """Write results.json with search metrics and per-variant data."""
    successes = [r for r in results if not r.error]
    times = [r.min_ms for r in successes]

    variants = []
    for r in results:
        rd = r._asdict()
        rd["nki_path"] = f"nki/{r.kernel_name}"
        variants.append(rd)

    results_data = {
        "metadata": {
            "num_kernels": len(kernels),
            "wallclock_s": profiler._last_elapsed,
            "hosts": profiler.hosts,
            "num_hosts": len(profiler.hosts),
        },
        "metrics": {
            "best_min_ms": min(times) if times else None,
            "worst_min_ms": max(times) if times else None,
            "mean_min_ms": sum(times) / len(times) if times else None,
            "best_kernel": min(successes, key=lambda r: r.min_ms).kernel_name if successes else None,
        },
        "variants": variants,
        "sim_errors": profiler.sim_errors,
    }
    with open(os.path.join(cache_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)


@dataclass
class RemoteProfiler:
    """Distributes NKI kernel profiling across remote Trainium hosts.

    Attributes:
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        venv_python: Path to the Python executable on remote hosts.
        ssh_timeout_sec: Timeout in seconds for SSH communication.
        neuron_platform_target: Neuron platform target (default "trn2").
        warmup: Number of warmup iterations before timing.
        iters: Number of benchmark iterations.
    """

    hosts: list[str]
    venv_python: str = _DEFAULT_VENV_PYTHON
    ssh_timeout_sec: int = 600
    neuron_platform_target: str = "trn2"
    warmup: int = 5
    iters: int = 20
    compiler_logs: dict[str, str] = field(default_factory=dict, repr=False)
    sim_errors: dict[str, str] = field(default_factory=dict, repr=False)
    _last_elapsed: float = field(default=0.0, repr=False)
    _collect_compiler_logs: bool = field(default=False, repr=False)

    @classmethod
    def from_config(cls, config_path: str, **overrides: Any) -> "RemoteProfiler":
        """Create a RemoteProfiler from a JSON config file.

        Args:
            config_path: Path to the remote config JSON.
            **overrides: Override any config values.
        """
        cfg = load_remote_config(config_path)
        kwargs = {
            "hosts": cfg["hosts"],
            "venv_python": cfg.get("venv_python", _DEFAULT_VENV_PYTHON),
            "ssh_timeout_sec": cfg["ssh_timeout_sec"],
            "neuron_platform_target": cfg.get("neuron_platform_target", "trn2"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _build_payload_base(
        self,
        input_specs: dict[str, tuple[tuple[int, ...], str]],
        scalar_params: dict[str, float],
        mac_count: int,
        seed: int,
        golden_source: str,
        golden_func_name: str,
        atol: float,
        rtol: float,
    ) -> dict[str, Any]:
        """Build the common payload dict shared by all hosts."""
        input_dtype_name = next(iter(input_specs.values()))[1]
        tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in input_specs.items()}
        return {
            "neuron_platform_target": self.neuron_platform_target,
            "config": {
                "warmup": self.warmup,
                "iters": self.iters,
                "mac_count": mac_count,
                "input_dtype_name": input_dtype_name,
                "atol": atol,
                "rtol": rtol,
                "collect_compiler_logs": self._collect_compiler_logs,
            },
            "tensor_specs": tensor_specs,
            "scalar_params": scalar_params,
            "seed": seed,
            "golden_source": golden_source,
            "golden_func_name": golden_func_name,
        }

    def profile(
        self,
        kernels: dict[str, str],
        input_specs: dict[str, tuple[tuple[int, ...], str]],
        scalar_params: dict[str, float],
        mac_count: int,
        seed: int,
        golden_source: str,
        golden_func_name: str,
        atol: float,
        rtol: float,
    ) -> list[ProfileResult]:
        """Profile NKI kernels across remote hosts.

        Args:
            kernels: Map of kernel filename to source code string.
            input_specs: Map of param name to (shape, dtype_str).
            scalar_params: Map of scalar param names to float values.
            mac_count: MAC operations for MFU calculation (0 to skip).
            seed: Random seed for reproducible tensor generation.
            golden_source: Source code of golden reference (empty to skip).
            golden_func_name: Name of the golden function (empty to skip).
            atol: Absolute tolerance for correctness check.
            rtol: Relative tolerance for correctness check.

        Returns:
            List of ProfileResult, one per kernel.
        """
        host_assignments = _build_host_assignments(list(kernels.keys()), self.hosts)
        logger.info(
            "Distributing %d kernels across %d hosts: %s",
            len(kernels),
            len(host_assignments),
            ", ".join(f"{h}({len(v)})" for h, v in host_assignments.items()),
        )

        payload_base = self._build_payload_base(
            input_specs, scalar_params, mac_count, seed, golden_source, golden_func_name, atol, rtol
        )
        return self._execute_workers(host_assignments, kernels, payload_base, mac_count)

    def _execute_workers(
        self,
        host_assignments: dict[str, list[str]],
        kernels: dict[str, str],
        payload_base: dict[str, Any],
        mac_count: int,
    ) -> list[ProfileResult]:
        """Launch workers, collect outputs, and update profiler state."""
        procs, writers = _launch_ssh_workers(host_assignments, kernels, payload_base, self.venv_python)
        try:
            for t in writers:
                t.join()
            for proc in procs.values():
                proc.stdin = None

            t0 = time.monotonic()
            host_outputs = _collect_host_outputs(procs, self.ssh_timeout_sec)
            all_results, all_compiler_logs, all_sim_errors = _process_host_outputs(
                procs, host_outputs, host_assignments, mac_count
            )
        except BaseException:
            for proc in procs.values():
                proc.kill()
            for proc in procs.values():
                proc.communicate()
            raise

        self.compiler_logs = all_compiler_logs
        self.sim_errors = all_sim_errors
        self._last_elapsed = time.monotonic() - t0
        logger.info("Profile complete: %d results in %.1fs", len(all_results), self._last_elapsed)
        return all_results

    def save_cache(self, cache_dir: str, kernels: dict[str, str], results: list[ProfileResult]) -> None:
        """Save profile results to disk following the standard cache layout.

        Args:
            cache_dir: Directory to write cache files to.
            kernels: Map of kernel filename to source code string.
            results: List of ProfileResult from the most recent profile() call.
        """
        _write_cache_files(cache_dir, kernels, self.compiler_logs)
        _write_results_json(cache_dir, kernels, results, self)
        logger.info("Cache saved to %s", cache_dir)
