"""Distribute NKI kernel profiling across remote Trainium hosts via SSH.

The coordinator sends per-kernel job configs and benchmark settings to
remote workers as a JSON payload over SSH stdin. Workers compile,
benchmark, and return results as JSON on stdout.
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

from autotune.runner.types import _DEFAULT_VENV_PYTHON, KernelJob, ProfileResult, make_failure

logger = logging.getLogger(__name__)

_AUTOTUNE_ROOT = Path(__file__).parent.parent
_NKIGYM_ROOT = Path(__file__).resolve().parents[4] / "nkigym" / "src" / "nkigym"

_worker_bundle_cache: bytes = b""


def _get_worker_bundle() -> bytes:
    """Build a base64-encoded bundle of the autotune and nkigym packages.

    The bundle is a JSON dict mapping relative path to source code.
    Sent as the first line of stdin to the bootstrap script on each
    remote host.
    """
    global _worker_bundle_cache  # noqa: PLW0603
    if not _worker_bundle_cache:
        bundle: dict[str, str] = {}
        bundle["worker.py"] = (_AUTOTUNE_ROOT / "runner" / "worker.py").read_text()
        for pkg_root in [_AUTOTUNE_ROOT, _NKIGYM_ROOT]:
            for py_file in sorted(pkg_root.rglob("*.py")):
                rel = py_file.relative_to(pkg_root.parent)
                bundle[str(rel)] = py_file.read_text()
        _worker_bundle_cache = base64.b64encode(json.dumps(bundle).encode("utf-8"))
    return _worker_bundle_cache


def _feed_stdin(proc: subprocess.Popen[bytes], bundle_line: bytes, payload: bytes) -> None:
    """Write bundle + payload to proc stdin in a background thread."""
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


def _fail_host(results: list[ProfileResult], kernel_names: list[str], error_msg: str) -> None:
    """Append a failure ProfileResult for every kernel assigned to a host."""
    for kname in kernel_names:
        results.append(make_failure(kname, error_msg, 0))


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


def _launch_ssh_workers(
    host_assignments: dict[str, list[str]],
    kernels: dict[str, KernelJob],
    payload_base: dict[str, Any],
    venv_python: str,
) -> tuple[dict[str, subprocess.Popen[bytes]], list[threading.Thread]]:
    """Launch SSH workers for each host and feed payloads."""
    b64_bundle = _get_worker_bundle()
    procs: dict[str, subprocess.Popen[bytes]] = {}
    writers: list[threading.Thread] = []

    for host, names in host_assignments.items():
        payload = dict(payload_base)
        payload["host"] = host
        payload["kernel_jobs"] = {}
        for n in names:
            job = kernels[n]
            payload["kernel_jobs"][n] = {
                "source": job.source,
                "func_name": job.func_name,
                "output_shape": list(job.output_shape),
                "tensor_specs": {name: (list(shape), dt) for name, (shape, dt) in job.input_specs.items()},
                "nkigym_source": job.nkigym_source,
                "nkigym_func_name": job.nkigym_func_name,
                "mac_count": job.mac_count,
                "atol": job.atol,
                "rtol": job.rtol,
                "neuronx_cc_args": list(job.neuronx_cc_args),
            }
        payload_bytes = json.dumps(payload).encode("utf-8")

        cmd = _build_ssh_cmd(host, venv_python)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t = threading.Thread(target=_feed_stdin, args=(proc, b64_bundle, payload_bytes), daemon=True)
        t.start()
        writers.append(t)
        procs[host] = proc

    return procs, writers


def _read_host_output(host: str, proc: subprocess.Popen[bytes], out: dict[str, tuple[bytes, bytes, int]]) -> None:
    """Read stdout/stderr from a single host process into out dict."""
    stdout_data, stderr_data = proc.communicate()
    out[host] = (stdout_data, stderr_data, proc.returncode)


def _collect_host_outputs(procs: dict[str, subprocess.Popen[bytes]]) -> dict[str, tuple[bytes, bytes, int]]:
    """Read all host outputs concurrently via threads."""
    host_outputs: dict[str, tuple[bytes, bytes, int]] = {}
    readers: list[threading.Thread] = []
    for host, proc in procs.items():
        rt = threading.Thread(target=_read_host_output, args=(host, proc, host_outputs), daemon=True)
        rt.start()
        readers.append(rt)
    for rt in readers:
        rt.join()
    return host_outputs


def _host_error_message(host: str, returncode: int, stderr_data: bytes, stdout_data: bytes) -> str:
    """Determine error message for a host, or empty string if success."""
    msg = ""
    if returncode != 0:
        stderr = stderr_data.decode(errors="replace")
        msg = f"SSH worker on {host} exited {returncode}: {stderr}"
    if not msg and not stdout_data:
        msg = f"No results from {host} (empty stdout)"
    return msg


def _parse_host_result(
    host: str,
    stdout_data: bytes,
    stderr_data: bytes,
    assigned_kernels: list[str],
    all_results: list[ProfileResult],
    all_compiler_logs: dict[str, str],
) -> None:
    """Parse one host's JSON output and append to aggregate lists."""
    stderr_text = stderr_data.decode(errors="replace")
    for line in stderr_text.strip().splitlines():
        logger.info("  [%s] %s", host, line)

    try:
        data = json.loads(stdout_data)
    except (json.JSONDecodeError, ValueError) as e:
        _fail_host(all_results, assigned_kernels, f"Malformed JSON from {host}: {e}")
        return

    for r in data["results"]:
        all_results.append(ProfileResult(**r))
    for kname, log_text in data.get("compiler_logs", {}).items():
        all_compiler_logs[kname] = log_text
    logger.info("Host %s: %d results", host, len(data["results"]))


def _process_host_outputs(
    procs: dict[str, subprocess.Popen[bytes]],
    host_outputs: dict[str, tuple[bytes, bytes, int]],
    host_assignments: dict[str, list[str]],
) -> tuple[list[ProfileResult], dict[str, str]]:
    """Process outputs from all hosts into aggregate results."""
    all_results: list[ProfileResult] = []
    all_compiler_logs: dict[str, str] = {}

    for host in procs:
        stdout_data, stderr_data, returncode = host_outputs.get(host, (b"", b"", -1))
        error = _host_error_message(host, returncode, stderr_data, stdout_data)
        if error:
            logger.error(error)
            _fail_host(all_results, host_assignments[host], error)
            continue
        _parse_host_result(host, stdout_data, stderr_data, host_assignments[host], all_results, all_compiler_logs)

    return all_results, all_compiler_logs


def write_kernel_sources(cache_dir: str, kernels: dict[str, KernelJob]) -> None:
    """Write kernel source files to ``<cache>/<stem>/<stem>.py``."""
    for kname, job in kernels.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, f"{stem}.py"), "w") as f:
            f.write(job.source)


def _write_compiler_logs(cache_dir: str, compiler_logs: dict[str, str]) -> None:
    """Write compiler logs to ``<cache>/<stem>/log-neuron-cc.txt``."""
    for kname, log_text in compiler_logs.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, "log-neuron-cc.txt"), "w") as f:
            f.write(log_text)


def _kernel_sort_key(kernel_name: str) -> tuple[int, int, str]:
    """Natural sort key for kernel names.

    Names shaped like ``<prefix>_<N>.py`` sort numerically by
    ``N`` (so ``kernel_2.py`` precedes ``kernel_10.py``); names
    without a numeric suffix sort alphabetically after all
    numerically-sorted names.
    """
    stem = Path(kernel_name).stem
    tail = stem.rsplit("_", 1)[-1]
    key = (0, int(tail), stem) if tail.isdigit() else (1, 0, stem)
    return key


def _write_results_json(
    cache_dir: str, kernels: dict[str, KernelJob], results: list[ProfileResult], profiler: "RemoteProfiler"
) -> None:
    """Write results.json with metrics and per-kernel data."""
    successes: list[tuple[ProfileResult, float, float]] = []
    passed_cpu_sim = success = sbuf_oom = 0
    for r in results:
        sim_ok = bool(r.cpu_sim.get("passed"))
        hw_ok = r.min_ms is not None and r.mfu is not None
        if r.min_ms is not None and r.mfu is not None:
            successes.append((r, r.min_ms, r.mfu))
        if sim_ok:
            passed_cpu_sim += 1
        if sim_ok and hw_ok:
            success += 1
        if not hw_ok and "Out of memory in sbuf" in r.hardware_output:
            sbuf_oom += 1
    times = [t for _, t, _ in successes]
    mfus = [m for _, _, m in successes]

    kernel_entries = []
    for r in sorted(results, key=lambda r: _kernel_sort_key(r.kernel_name)):
        rd = r._asdict()
        stem = Path(r.kernel_name).stem
        rd["kernel_path"] = f"{stem}/{stem}.py"
        kernel_entries.append(rd)

    best_kernel = min(successes, key=lambda s: s[1])[0].kernel_name if successes else None
    worst_kernel = max(successes, key=lambda s: s[1])[0].kernel_name if successes else None
    results_data = {
        "metadata": {"num_kernels": len(kernels), "wallclock_s": profiler._last_elapsed, "hosts": profiler.hosts},
        "metrics": {
            "best_min_ms": min(times) if times else None,
            "worst_min_ms": max(times) if times else None,
            "mean_min_ms": sum(times) / len(times) if times else None,
            "best_kernel": best_kernel,
            "worst_kernel": worst_kernel,
            "best_mfu": max(mfus) if mfus else None,
            "worst_mfu": min(mfus) if mfus else None,
            "passed_cpu_sim": passed_cpu_sim,
            "success": success,
            "sbuf_oom": sbuf_oom,
        },
        "kernels": kernel_entries,
    }
    with open(os.path.join(cache_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)


@dataclass
class RemoteProfiler:
    """Distributes NKI kernel profiling across remote Trainium hosts.

    Attributes:
        hosts: SSH hostnames (e.g. ["gym-1", "gym-2"]).
        venv_python: Path to the Python executable on remote hosts.
        neuron_platform_target: Neuron platform target (default "trn2").
        warmup: Number of warmup iterations before timing.
        iters: Number of benchmark iterations.
    """

    hosts: list[str]
    venv_python: str = _DEFAULT_VENV_PYTHON
    neuron_platform_target: str = "trn2"
    warmup: int = 5
    iters: int = 20
    seed: int = 42
    compiler_logs: dict[str, str] = field(default_factory=dict, repr=False)
    _last_elapsed: float = field(default=0.0, repr=False)
    _collect_compiler_logs: bool = field(default=False, repr=False)

    def _build_payload_base(self) -> dict[str, Any]:
        """Build the common payload dict shared by all hosts."""
        return {
            "neuron_platform_target": self.neuron_platform_target,
            "seed": self.seed,
            "config": {
                "warmup": self.warmup,
                "iters": self.iters,
                "collect_compiler_logs": self._collect_compiler_logs,
            },
        }

    def profile(self, kernels: dict[str, KernelJob]) -> list[ProfileResult]:
        """Profile NKI kernels across remote hosts.

        Args:
            kernels: Map of kernel filename to KernelJob.

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

        payload_base = self._build_payload_base()
        return self._execute_workers(host_assignments, kernels, payload_base)

    def _execute_workers(
        self, host_assignments: dict[str, list[str]], kernels: dict[str, KernelJob], payload_base: dict[str, Any]
    ) -> list[ProfileResult]:
        """Launch workers, collect outputs, and update profiler state."""
        procs, writers = _launch_ssh_workers(host_assignments, kernels, payload_base, self.venv_python)
        try:
            for t in writers:
                t.join()
            for proc in procs.values():
                proc.stdin = None

            t0 = time.monotonic()
            host_outputs = _collect_host_outputs(procs)
            all_results, all_compiler_logs = _process_host_outputs(procs, host_outputs, host_assignments)
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

    def save_cache(self, cache_dir: str, kernels: dict[str, KernelJob], results: list[ProfileResult]) -> None:
        """Save profile results to disk following the standard cache layout."""
        write_kernel_sources(cache_dir, kernels)
        _write_compiler_logs(cache_dir, self.compiler_logs)
        _write_results_json(cache_dir, kernels, results, self)
        logger.info("Cache saved to %s", cache_dir)
