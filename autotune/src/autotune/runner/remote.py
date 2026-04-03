"""Distribute NKI kernel profiling across remote Trainium hosts via SSH.

The coordinator sends kernel source code and benchmark config to remote
workers as a lightweight JSON payload over SSH stdin. Workers compile,
benchmark, and return results as JSON on stdout. No shared filesystem
required — the autotune package is bundled and bootstrapped on each host.

Usage::

    from autotune.runner.remote import RemoteProfiler

    profiler = RemoteProfiler.from_config("remote_config.json")
    results = profiler.profile(
        kernels={"add_v0.py": add_source, "add_v1.py": add_source_v1},
        input_specs={"a": ((128, 512), "bfloat16"), "b": ((128, 512), "bfloat16")},
    )
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

import numpy as np

from autotune.runner.worker import ProfileResult, _make_failure

logger = logging.getLogger(__name__)


def _fail_host(results: list[ProfileResult], kernel_names: list[str], error_msg: str, mac_count: int) -> None:
    """Append a failure ProfileResult for every kernel assigned to a host."""
    for kname in kernel_names:
        results.append(_make_failure(kname, error_msg, mac_count))


_DEFAULT_VENV_PYTHON = "/home/ubuntu/venvs/kernel-env/bin/python"

_REQUIRED_CONFIG_KEYS = {"hosts", "ssh_timeout_sec"}

_AUTOTUNE_ROOT = Path(__file__).parent.parent

_worker_bundle_cache: bytes | None = None


def _get_worker_bundle() -> bytes:
    """Build a base64-encoded bundle of the autotune package.

    The bundle is a JSON dict mapping relative path to source code.
    Sent as the first line of stdin to the bootstrap script on each
    remote host.
    """
    global _worker_bundle_cache  # noqa: PLW0603
    if _worker_bundle_cache is None:
        bundle: dict[str, str] = {}
        bundle["worker.py"] = (_AUTOTUNE_ROOT / "runner" / "worker.py").read_text()
        for py_file in sorted(_AUTOTUNE_ROOT.rglob("*.py")):
            rel = py_file.relative_to(_AUTOTUNE_ROOT.parent)
            bundle[str(rel)] = py_file.read_text()
        _worker_bundle_cache = base64.b64encode(json.dumps(bundle).encode("utf-8"))
    return _worker_bundle_cache


def load_remote_config(config_path: str | Path) -> dict:
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
    _last_elapsed: float = field(default=0.0, repr=False)
    _collect_compiler_logs: bool = field(default=False, repr=False)

    @classmethod
    def from_config(cls, config_path: str | Path, **overrides: Any) -> "RemoteProfiler":
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

    def profile(
        self,
        kernels: dict[str, str],
        input_specs: dict[str, tuple[tuple[int, ...], np.dtype | str]],
        scalar_params: dict[str, float] | None = None,
        mac_count: int = 0,
        seed: int = 42,
        golden_source: str | None = None,
        golden_func_name: str | None = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> list[ProfileResult]:
        """Profile NKI kernels across remote hosts.

        Function names and output shapes are auto-detected from each
        kernel's ``@nki.jit`` decorator and ``nl.ndarray(..., buffer=nl.shared_hbm)``.

        Args:
            kernels: Map of kernel filename to source code string.
            input_specs: Map of param name to (shape, dtype). Dtype can be
                a numpy dtype (e.g. ``np.float16``) or a string (e.g. ``"bfloat16"``).
            scalar_params: Map of scalar param names to float values.
            mac_count: MAC operations for MFU calculation (0 to skip).
            seed: Random seed for reproducible tensor generation.
            golden_source: Source code of a golden reference function.
            golden_func_name: Name of the golden function.
            atol: Absolute tolerance for correctness check.
            rtol: Relative tolerance for correctness check.

        Returns:
            List of ProfileResult, one per kernel.
        """
        kernel_names = list(kernels.keys())
        if not kernel_names:
            return []

        def _dtype_str(dt: np.dtype | str) -> str:
            if isinstance(dt, str):
                return dt
            return dt.name

        input_dtype_name = _dtype_str(next(iter(input_specs.values()))[1])

        tensor_specs = {
            name: {"shape": list(shape), "dtype": _dtype_str(dt)} for name, (shape, dt) in input_specs.items()
        }

        shuffled_names = list(kernel_names)
        random.shuffle(shuffled_names)

        active_hosts = self.hosts[: len(kernel_names)] if len(kernel_names) < len(self.hosts) else self.hosts
        host_assignments: dict[str, list[str]] = {h: [] for h in active_hosts}
        for i, kname in enumerate(shuffled_names):
            host = active_hosts[i % len(active_hosts)]
            host_assignments[host].append(kname)

        logger.info(
            "Distributing %d kernels across %d hosts: %s",
            len(kernel_names),
            len(active_hosts),
            ", ".join(f"{h}({len(host_assignments[h])})" for h in active_hosts),
        )

        b64_bundle = _get_worker_bundle()

        config_payload = {
            "warmup": self.warmup,
            "iters": self.iters,
            "mac_count": mac_count,
            "input_dtype_name": input_dtype_name,
            "atol": atol,
            "rtol": rtol,
            "collect_compiler_logs": self._collect_compiler_logs,
        }

        procs: dict[str, subprocess.Popen] = {}
        writers: list[threading.Thread] = []

        try:
            for host, names in host_assignments.items():
                payload = {
                    "host": host,
                    "neuron_platform_target": self.neuron_platform_target,
                    "config": config_payload,
                    "kernel_names": names,
                    "sources": {n: kernels[n] for n in names},
                    "tensor_specs": tensor_specs,
                    "scalar_params": scalar_params or {},
                    "seed": seed,
                    "golden_source": golden_source,
                    "golden_func_name": golden_func_name,
                }
                payload_bytes = json.dumps(payload).encode("utf-8")

                bootstrap = (
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
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    "ControlPath=/tmp/autotune-ssh-%r@%h:%p",
                    "-o",
                    "ControlPersist=300",
                    host,
                    f"{self.venv_python} -c '{bootstrap}'",
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                t = threading.Thread(target=_feed_stdin, args=(proc, b64_bundle, payload_bytes), daemon=True)
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
            all_results: list[ProfileResult] = []
            all_compiler_logs: dict[str, str] = {}

            """
            Read all hosts concurrently. Workers produce ~3MB of output
            (results + compiler logs) and the OS pipe buffer is only
            64KB. Sequential communicate() causes each subsequent host
            to stall on its stdout write while we drain the previous
            host, adding ~4s per host. Threading the reads eliminates
            this pipeline stall.
            """
            host_outputs: dict[str, tuple[bytes, bytes, int]] = {}

            def _read_host(host: str, proc: subprocess.Popen[bytes]) -> None:
                try:
                    stdout_data, stderr_data = proc.communicate(timeout=self.ssh_timeout_sec)
                    host_outputs[host] = (stdout_data, stderr_data, proc.returncode)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    host_outputs[host] = (b"", b"", -1)

            readers = []
            for host, proc in procs.items():
                rt = threading.Thread(target=_read_host, args=(host, proc), daemon=True)
                rt.start()
                readers.append(rt)
            for rt in readers:
                rt.join(timeout=self.ssh_timeout_sec)

            for host in procs:
                stdout_data, stderr_data, returncode = host_outputs.get(host, (b"", b"", -1))
                elapsed = time.monotonic() - t0

                if returncode == -1:
                    error_msg = f"SSH worker on {host} timed out after {self.ssh_timeout_sec}s"
                    logger.error(error_msg)
                    _fail_host(all_results, host_assignments[host], error_msg, mac_count)
                    continue

                if returncode != 0:
                    stderr = stderr_data.decode(errors="replace")
                    error_msg = f"SSH worker on {host} exited with code {returncode}: {stderr[:2000]}"
                    logger.error(error_msg)
                    _fail_host(all_results, host_assignments[host], error_msg, mac_count)
                    continue

                stderr_text = stderr_data.decode(errors="replace")
                for line in stderr_text.strip().splitlines():
                    logger.info("  [%s] %s", host, line)

                if not stdout_data:
                    error_msg = f"No results from {host} (empty stdout)"
                    logger.error(error_msg)
                    _fail_host(all_results, host_assignments[host], error_msg, mac_count)
                    continue

                try:
                    data = json.loads(stdout_data)
                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Malformed JSON from {host}: {e}"
                    logger.error(error_msg)
                    _fail_host(all_results, host_assignments[host], error_msg, mac_count)
                    continue

                for r in data["results"]:
                    all_results.append(ProfileResult(**r))
                for kname, log_text in data.get("compiler_logs", {}).items():
                    all_compiler_logs[kname] = log_text
                logger.info("Host %s: %d results (%.1fs)", host, len(data["results"]), elapsed)

        except BaseException:
            for proc in procs.values():
                proc.kill()
            for proc in procs.values():
                proc.communicate()
            raise

        self.compiler_logs = all_compiler_logs
        self._last_elapsed = time.monotonic() - t0
        logger.info("Profile complete: %d results in %.1fs", len(all_results), self._last_elapsed)
        return all_results

    def save_cache(self, cache_dir: str, kernels: dict[str, str], results: list[ProfileResult]) -> None:
        """Save profile results to disk following the standard cache layout.

        Structure::

            cache_dir/
                results.json
                nki/
                    kernel_v0.py
                    kernel_v1.py
                    ...
                neff/
                    kernel_v0/
                        log-neuron-cc.txt
                    kernel_v1/
                        log-neuron-cc.txt
                    ...

        Args:
            cache_dir: Directory to write cache files to.
            kernels: Map of kernel filename to source code string.
            results: List of ProfileResult from the most recent profile() call.
        """
        nki_dir = os.path.join(cache_dir, "nki")
        neff_dir = os.path.join(cache_dir, "neff")
        os.makedirs(nki_dir, exist_ok=True)
        os.makedirs(neff_dir, exist_ok=True)

        for kname, source in kernels.items():
            with open(os.path.join(nki_dir, kname), "w") as f:
                f.write(source)

        for kname, log_text in self.compiler_logs.items():
            stem = os.path.splitext(kname)[0]
            log_dir = os.path.join(neff_dir, stem)
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "log-neuron-cc.txt"), "w") as f:
                f.write(log_text)

        num_kernels = len(kernels)
        successes = [r for r in results if not r.error]
        times = [r.min_ms for r in successes]

        variants = []
        for r in results:
            rd = r._asdict()
            rd["nki_path"] = f"nki/{r.kernel_name}"
            variants.append(rd)

        results_data = {
            "search": {
                "unique_schedules": num_kernels,
                "qualifying_schedules": len(successes),
                "total_visited": num_kernels,
            },
            "compilation": {"wallclock_s": self._last_elapsed, "hosts": self.hosts, "num_hosts": len(self.hosts)},
            "metrics": {
                "best_min_ms": min(times) if times else None,
                "worst_min_ms": max(times) if times else None,
                "mean_min_ms": sum(times) / len(times) if times else None,
                "best_kernel": min(successes, key=lambda r: r.min_ms).kernel_name if successes else None,
            },
            "variants": variants,
        }
        with open(os.path.join(cache_dir, "results.json"), "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Cache saved to %s", cache_dir)


@dataclass
class ProfileOutput:
    """Results from a remote profiling run.

    Attributes:
        results: Per-kernel profiling results.
        compiler_logs: Map of kernel name to compiler log text.
        elapsed_s: Total wallclock time in seconds.
        hosts: SSH hostnames used for profiling.
        cache_dir: Path to cache directory, if saved.
    """

    results: list[ProfileResult]
    compiler_logs: dict[str, str]
    elapsed_s: float
    hosts: list[str]
    cache_dir: str | None = None

    @property
    def successes(self) -> list[ProfileResult]:
        """Results that completed without error."""
        return [r for r in self.results if not r.error]

    @property
    def failures(self) -> list[ProfileResult]:
        """Results that errored."""
        return [r for r in self.results if r.error]

    def __str__(self) -> str:
        """Human-readable summary with per-kernel timing table."""
        lines: list[str] = []
        successes = self.successes
        failures = self.failures

        if successes:
            header = f"{'Kernel':<30} {'min_ms':>10} {'mean_ms':>10} {'p50_ms':>10} {'p99_ms':>10}"
            if any(r.mfu > 0 for r in successes):
                header += f" {'mfu':>8}"
            lines.append(header)
            lines.append("-" * len(header))
            for r in sorted(successes, key=lambda r: r.min_ms):
                row = (
                    f"{r.kernel_name:<30} {r.min_ms:>10.4f} {r.mean_ms:>10.4f}" f" {r.p50_ms:>10.4f} {r.p99_ms:>10.4f}"
                )
                if any(s.mfu > 0 for s in successes):
                    row += f" {r.mfu:>7.2f}%"
                lines.append(row)

        if failures:
            lines.append(f"\n{len(failures)} failures:")
            for r in failures:
                lines.append(f"  {r.kernel_name}: {r.error[:200]}")

        times = [r.min_ms for r in successes]
        lines.append(f"\nSummary:")
        lines.append(f"  Succeeded:  {len(successes)}/{len(self.results)}")
        lines.append(f"  Hosts:      {len(self.hosts)} ({', '.join(self.hosts)})")
        lines.append(f"  Wallclock:  {self.elapsed_s:.1f}s")
        if times:
            lines.append(f"  Best:       {min(times):.4f} ms ({successes[0].kernel_name})")
            lines.append(f"  Worst:      {max(times):.4f} ms")
            lines.append(f"  Mean:       {sum(times) / len(times):.4f} ms")
        if self.cache_dir:
            lines.append(f"  Cache:      {self.cache_dir}")

        return "\n".join(lines)


def remote_profile(
    kernels: dict[str, str],
    input_specs: dict[str, tuple[tuple[int, ...], np.dtype | str]],
    hosts: list[str],
    cache_dir: str | None = None,
    scalar_params: dict[str, float] | None = None,
    mac_count: int = 0,
    seed: int = 42,
    golden_source: str | None = None,
    golden_func_name: str | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    warmup: int = 5,
    iters: int = 20,
    ssh_timeout_sec: int = 600,
    neuron_platform_target: str = "trn2",
    venv_python: str = _DEFAULT_VENV_PYTHON,
) -> ProfileOutput:
    """Profile NKI kernels across remote Trainium hosts.

    Single-call API that distributes kernel compilation and benchmarking
    to remote workers, optionally saving results to a cache directory.
    Function name, output shape, and output dtype are auto-detected
    from the kernel source (``@nki.jit`` decorator and
    ``nl.ndarray(..., buffer=nl.shared_hbm)``).

    Args:
        kernels: Map of kernel filename to source code string.
        input_specs: Map of param name to (shape, dtype). Dtype can be
            a numpy dtype (e.g. ``np.float16``) or a string (e.g. ``"bfloat16"``).
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        cache_dir: If set, save results/sources/logs to this directory.
        scalar_params: Map of scalar param names to float values.
        mac_count: MAC operations for MFU calculation (0 to skip).
        seed: Random seed for reproducible tensor generation.
        golden_source: Source code of a golden reference function.
        golden_func_name: Name of the golden function.
        atol: Absolute tolerance for correctness check.
        rtol: Relative tolerance for correctness check.
        warmup: Number of warmup iterations before timing.
        iters: Number of benchmark iterations.
        ssh_timeout_sec: Timeout in seconds for SSH communication.
        neuron_platform_target: Neuron platform target.
        venv_python: Path to the Python executable on remote hosts.

    Returns:
        ProfileOutput with results, compiler logs, and elapsed time.
    """
    profiler = RemoteProfiler(
        hosts=hosts,
        venv_python=venv_python,
        ssh_timeout_sec=ssh_timeout_sec,
        neuron_platform_target=neuron_platform_target,
        warmup=warmup,
        iters=iters,
        _collect_compiler_logs=cache_dir is not None,
    )
    results = profiler.profile(
        kernels=kernels,
        input_specs=input_specs,
        scalar_params=scalar_params,
        mac_count=mac_count,
        seed=seed,
        golden_source=golden_source,
        golden_func_name=golden_func_name,
        atol=atol,
        rtol=rtol,
    )
    if cache_dir:
        profiler.save_cache(cache_dir, kernels, results)

    return ProfileOutput(
        results=results,
        compiler_logs=profiler.compiler_logs,
        elapsed_s=profiler._last_elapsed,
        hosts=hosts,
        cache_dir=cache_dir,
    )


def _feed_stdin(proc: subprocess.Popen[bytes], bundle_line: bytes, payload: bytes) -> None:
    """Write bundle + payload to proc stdin in a background thread.

    First line: base64-encoded file bundle (read by bootstrap).
    Remaining bytes: JSON work payload (read by worker.py).
    """
    stdin = proc.stdin
    if stdin is None:
        return
    try:
        stdin.write(bundle_line + b"\n")
        stdin.write(payload)
    except BrokenPipeError:
        pass
    finally:
        stdin.close()
