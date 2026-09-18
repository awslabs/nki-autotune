"""Run NKI kernels end-to-end in fp32, locally or in batches over SSH."""

import inspect
import json
import os
import pickle
import secrets
import shlex
import subprocess
import tempfile
import textwrap
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import numpy as np

from nkigym.profile.ssh import _SSH_OPTIONS, SSHTransportError, _CommandRunner, _require_command, _validate_host

_P = ParamSpec("_P")
_R = TypeVar("_R")
_SSH_WORKER_OPTIONS = (*_SSH_OPTIONS, "-n")
_REMOTE_PYTHON = '"$HOME"/venvs/kernel-env/bin/python'
ArrayResult = np.ndarray | tuple[np.ndarray, ...]
_SerializedCase = tuple[int, str, str, str, dict[str, np.ndarray], ArrayResult, tuple[str, dict[str, object]] | None]


@dataclass(frozen=True)
class FP32SimulationCase:
    """One rendered kernel and reference result for remote fp32 validation."""

    label: str
    kernel: str
    func_name: str
    inputs: dict[str, np.ndarray]
    expected: ArrayResult
    validation: tuple[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class _SimulationFailure:
    """A failed case reported by one remote worker."""

    case_index: int
    label: str
    exception_type: str
    traceback: str


@dataclass(frozen=True)
class _HostResult:
    """Validated result metadata from one remote worker."""

    host: str
    completed: int
    failure: _SimulationFailure | None


def simulate_fp32(kernel: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap a NKI kernel so ``nki.simulate`` runs in ``nl.float32`` end-to-end.

    Rewrites every reduced-precision floating dtype (bfloat16, float16,
    float8_*, float4_*, tfloat32) referenced as ``nl.<dtype>`` in the
    kernel source to ``nl.float32``, then re-execs the rewritten source in
    a copy of the kernel's original module globals. The returned wrapper
    casts each numpy input tensor to ``np.float32`` before invoking
    ``nki.simulate`` so the simulator sees fp32 end-to-end.
    """
    from nkigym.profile.simulate_nki_worker import _fp32_source, _simulate_kernel_fp32

    func = getattr(kernel, "func", kernel)
    source = _fp32_source(textwrap.dedent(inspect.getsource(func)))
    namespace: dict = dict(func.__globals__)
    exec(source, namespace)  # noqa: S102

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Cast numpy inputs to fp32 and forward to the simulated kernel."""
        cast_args = tuple(_fp32_value(argument) for argument in args)
        cast_kwargs = {name: _fp32_value(argument) for name, argument in kwargs.items()}
        return cast(_R, _simulate_kernel_fp32(namespace[func.__name__], (cast_args, cast_kwargs)))

    return wrapper


def batch_simulate_fp32(
    hosts: list[str], cases: list[FP32SimulationCase], atol: float, rtol: float, timeout_s: int = 7200
) -> int:
    """Validate rendered kernels in fp32 across a list of SSH hosts.

    Exact duplicate kernels sharing the same inputs and reference output are
    simulated once. Cases are greedily balanced across every configured host.
    Each host receives one pickle bundle, simulates with a local process pool,
    and compares outputs remotely. Only compact result metadata returns.

    Args:
        hosts: SSH destinations provisioned with ``~/venvs/kernel-env``.
        cases: Rendered kernels, inputs, and expected fp32 outputs.
        atol: Absolute tolerance passed to ``numpy.testing.assert_allclose``.
        rtol: Relative tolerance passed to ``numpy.testing.assert_allclose``.
        timeout_s: Total timeout for each remote host batch.

    Returns:
        The number of successfully validated cases.
    """
    _validate_batch(hosts, cases, atol, rtol, timeout_s)
    if cases:
        for command in ("ssh", "rsync"):
            _require_command(command)
        unique_cases = sorted(
            {
                (c.kernel, c.func_name, id(c.inputs), id(c.expected), repr(c.validation)): (i, c)
                for i, c in reversed(list(enumerate(cases)))
            }.values(),
            key=lambda item: item[0],
        )
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            host_capacities = list(zip(hosts, executor.map(_detect_host_cpu_count, hosts), strict=True))
        partitions = _partition_cases(host_capacities, unique_cases)
        with tempfile.TemporaryDirectory(prefix="nkigym-simulate-") as raw_directory:
            requests = _write_requests(Path(raw_directory), partitions, atol, rtol)
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                results = list(executor.map(lambda request: _run_remote_batch(*request, timeout_s=timeout_s), requests))
        _raise_batch_failure(results)
        if (unique_completed := sum(result.completed for result in results)) != len(unique_cases):
            raise RuntimeError(f"remote simulation completed {unique_completed} of {len(unique_cases)} distinct cases")
    return len(cases)


def _fp32_value(value: object) -> object:
    """Cast floating NumPy inputs to fp32 while preserving integer arrays."""
    return value.astype(np.float32) if isinstance(value, np.ndarray) and value.dtype.kind == "f" else value


def _validate_batch(
    hosts: list[str], cases: list[FP32SimulationCase], atol: float, rtol: float, timeout_s: int
) -> None:
    """Reject malformed batch inputs before opening an SSH connection."""
    if not hosts:
        raise ValueError("at least one SSH host is required")
    for host in hosts:
        _validate_host(host)
    for name, tolerance in (("absolute", atol), ("relative", rtol)):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} tolerance must be finite and non-negative")
    if timeout_s <= 0:
        raise ValueError("batch simulation timeout must be positive")
    for case in cases:
        if not case.label:
            raise ValueError("simulation case label must not be empty")
        if not case.kernel.strip():
            raise ValueError(f"{case.label}: kernel source must not be empty")
        if not case.func_name.isidentifier():
            raise ValueError(f"{case.label}: invalid kernel function name {case.func_name!r}")


@cache
def _detect_host_cpu_count(host: str) -> int:
    """Return the logical CPUs available to the remote simulation process."""
    source = (
        'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)'
    )
    output = _CommandRunner(30).run(
        f"Detecting CPU count on {host}",
        ["ssh", *_SSH_WORKER_OPTIONS, host, f"{_REMOTE_PYTHON} -c {shlex.quote(source)}"],
        None,
    )
    try:
        cpu_count = int(output.strip())
    except ValueError as error:
        raise RuntimeError(f"detecting CPU count on {host} returned {output.strip()!r}") from error
    if cpu_count <= 0:
        raise RuntimeError(f"detecting CPU count on {host} returned {cpu_count}")
    return cpu_count


def _partition_cases(
    host_capacities: list[tuple[str, int]], cases: list[tuple[int, FP32SimulationCase]]
) -> list[tuple[str, int, list[_SerializedCase]]]:
    """Spread large case groups across CPU capacity while keeping short transfers local."""

    def group_weight(group: list[tuple[int, FP32SimulationCase]]) -> int:
        """Estimate loop-expanded simulation work for one shared-input group."""
        return sum(sum(16 ** (s.find("nisa") // 4) for s in c.kernel.splitlines() if "nisa." in s) for _, c in group)

    grouped: dict[tuple[int, int], list[tuple[int, FP32SimulationCase]]] = {}
    for indexed_case in cases:
        grouped.setdefault((id((case := indexed_case[1]).inputs), id(case.expected)), []).append(indexed_case)

    active_hosts = sorted(host_capacities, key=lambda item: -item[1])[: min(len(host_capacities), len(cases))]
    assigned: list[list[tuple[int, FP32SimulationCase]]] = [[] for _host in active_hosts]
    weights = [0 for _host in active_hosts]
    for group in grouped.values():
        small = sum(value.nbytes for value in group[0][1].inputs.values()) < 1 << 28
        count = min(len(group), len(active_hosts)) if small or len(group) >= sum(c for _, c in active_hosts) else 1
        selected: set[int] = set()
        for chunk in (group[index::count] for index in range(count)):
            host_index = min(
                range(len(active_hosts)),
                key=lambda index: (index in selected, weights[index] / active_hosts[index][1], weights[index], index),
            )
            selected.add(host_index)
            assigned[host_index].extend(chunk)
            weights[host_index] += group_weight(chunk)
    partitions = []
    for (host, cpu_count), host_cases in zip(active_hosts, assigned, strict=True):
        serialized = [
            (index, case.label, case.kernel, case.func_name, case.inputs, case.expected, case.validation)
            for index, case in sorted(host_cases, key=lambda item: item[0])
        ]
        if serialized:
            partitions.append((host, min(cpu_count, len(serialized)), serialized))
    return partitions


def _write_requests(
    directory: Path, partitions: list[tuple[str, int, list[_SerializedCase]]], atol: float, rtol: float
) -> list[tuple[str, Path, Path]]:
    """Write one request bundle and result path per host."""
    requests: list[tuple[str, Path, Path]] = []
    for index, (host, worker_count, cases) in enumerate(partitions):
        (host_directory := directory / f"host-{index}").mkdir()
        request_path, result_path = host_directory / "request.pkl", host_directory / "result.json"
        with request_path.open("wb") as request_file:
            pickle.dump((cases, atol, rtol, worker_count), request_file, protocol=pickle.HIGHEST_PROTOCOL)
        requests.append((host, request_path, result_path))
    return requests


def _run_remote_batch(host: str, request_path: Path, result_path: Path, timeout_s: int) -> _HostResult:
    """Upload and execute one host partition, then parse its result."""
    remote_run = f".cache/nkigym-simulate/runs/{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
    rsync_shell = shlex.join(("ssh", *_SSH_OPTIONS))
    runner = _CommandRunner(timeout_s)
    if request_path.stat().st_size >= 1 << 30:
        subprocess.run(("pigz", "-1", "-f", str(request_path)), check=True)
        request_path = Path(f"{request_path}.gz")
    try:
        runner.run(
            "Checking remote simulation environment",
            [
                "ssh",
                *_SSH_WORKER_OPTIONS,
                host,
                (
                    f"test -x {_REMOTE_PYTHON} && "
                    f"{_REMOTE_PYTHON} -c 'import nki, numpy' && "
                    f'mkdir -p "$HOME"/{remote_run}'
                ),
            ],
            None,
        )
        runner.run(
            "Uploading simulation batch",
            [
                "rsync",
                "-az" if request_path.stat().st_size >= 1 << 30 else "-a",
                "-e",
                rsync_shell,
                str(Path(__file__).with_name("simulate_nki_worker.py").resolve()),
                str(request_path),
                f"{host}:{remote_run}/",
            ],
            None,
        )
        _start_remote_worker(host, remote_run, runner)
        _wait_for_remote_result(host, remote_run, runner)
        runner.run(
            "Downloading simulation result",
            ["rsync", "-a", "-e", rsync_shell, f"{host}:{remote_run}/result.json", str(result_path)],
            None,
        )
    except SSHTransportError as error:
        raise RuntimeError(f"SSH batch simulation failed for {host}: {error}\n{runner.log[-5000:]}") from error
    finally:
        runner.cleanup(host, remote_run, True)
    return _read_host_result(host, result_path)


def _start_remote_worker(host: str, remote_run: str, runner: _CommandRunner) -> None:
    """Publish the process group before unpacking and running a detached worker."""
    worker = (
        f"OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=1 {_REMOTE_PYTHON} "
        f'"$HOME"/{remote_run}/simulate_nki_worker.py '
        f'--worker "$HOME"/{remote_run}/request.pkl '
        f'"$HOME"/{remote_run}/result.json'
    )
    script = (
        f'printf "%s\\n" "$$" > "$HOME"/{remote_run}/worker.pgid; '
        f'(test ! -f "$HOME"/{remote_run}/request.pkl.gz || gzip -df "$HOME"/{remote_run}/request.pkl.gz) && {worker}; '
        f'status=$?; printf "%s\\n" "$status" > "$HOME"/{remote_run}/worker.exit'
    )
    launch = f"setsid -f sh -c {shlex.quote(script)} " f'>"$HOME"/{remote_run}/worker.log 2>&1 < /dev/null'
    command = (
        f'{launch}; attempts=0; while test "$attempts" -lt 100; do '
        f'test -s "$HOME"/{remote_run}/worker.pgid && exit 0; '
        "attempts=$((attempts + 1)); sleep 0.1; done; exit 1"
    )
    runner.run("Starting remote simulation worker", ["ssh", *_SSH_WORKER_OPTIONS, host, command], None)


def _wait_for_remote_result(host: str, remote_run: str, runner: _CommandRunner) -> None:
    """Poll atomic worker files without depending on SSH session teardown."""
    result = f'"$HOME"/{remote_run}/result.json'
    exit_status = f'"$HOME"/{remote_run}/worker.exit'
    probe = (
        f"if test -f {result}; then printf ready; "
        f"elif test -f {exit_status}; then printf failed:; cat {exit_status}; "
        "else printf pending; fi"
    )
    runner.lines.append("==> Waiting for remote simulation result\n")
    while True:
        state = runner.run(
            "Checking remote simulation result", ["ssh", "-n", *_SSH_OPTIONS, host, probe], None, timeout_s=30.0
        ).strip()
        if state == "ready":
            return
        if state.startswith("failed:"):
            runner.run(
                "Reading remote simulation failure",
                ["ssh", "-n", *_SSH_OPTIONS, host, f'cat "$HOME"/{remote_run}/worker.log'],
                None,
            )
            raise SSHTransportError(
                f"remote simulation worker exited with status {state.removeprefix('failed:')}", runner.log
            )
        if state != "pending":
            raise SSHTransportError(f"remote simulation worker returned malformed state {state!r}", runner.log)
        time.sleep(max(0.0, min(0.25, runner.remaining_s)))


def _read_host_result(host: str, result_path: Path) -> _HostResult:
    """Parse and validate one downloaded worker result."""
    if not result_path.is_file():
        raise RuntimeError(f"SSH batch simulation returned no result for {host}")
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"SSH batch simulation returned malformed result for {host}")
    assigned, completed, raw_failure = (raw.get(name) for name in ("assigned", "completed", "failure"))
    if type(assigned) is not int or type(completed) is not int or not 0 <= completed <= assigned:
        raise RuntimeError(f"SSH batch simulation returned invalid case counts for {host}")
    if (failure := _parse_failure(host, raw_failure)) is None and completed != assigned:
        raise RuntimeError(f"SSH batch simulation on {host} stopped without reporting a failure")
    return _HostResult(host=host, completed=completed, failure=failure)


def _parse_failure(host: str, raw_failure: object) -> _SimulationFailure | None:
    """Parse optional failure metadata from one remote host."""
    if raw_failure is None:
        return None
    fields = {"case_index": int, "label": str, "exception_type": str, "traceback": str}
    if not isinstance(raw_failure, dict) or any(type(raw_failure.get(k)) is not t for k, t in fields.items()):
        raise RuntimeError(f"SSH batch simulation returned invalid failure fields for {host}")
    if raw_failure["case_index"] < 0:
        raise RuntimeError(f"SSH batch simulation returned a negative case index for {host}")
    return _SimulationFailure(**{key: raw_failure[key] for key in fields})


def _raise_batch_failure(results: list[_HostResult]) -> None:
    """Raise the earliest simulation failure in original case order."""
    failures = [(result.host, result.failure) for result in results if result.failure is not None]
    if failures:
        host, failure = min(failures, key=lambda item: item[1].case_index)
        error_type = AssertionError if failure.exception_type == "AssertionError" else RuntimeError
        raise error_type(f"{failure.label}\nremote host: {host}\n{failure.traceback}")
