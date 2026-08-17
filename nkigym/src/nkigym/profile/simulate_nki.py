"""Run NKI kernels end-to-end in fp32, locally or in batches over SSH."""

from __future__ import annotations

import inspect
import json
import os
import pickle
import secrets
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import nki
import numpy as np

from nkigym.profile.simulate_nki_worker import _fp32_source

_P = ParamSpec("_P")
_R = TypeVar("_R")
_SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=30",
    "-o",
    "ControlPath=~/.ssh/nkigym-%C",
)
_SSH_WORKER_OPTIONS = (*_SSH_OPTIONS, "-n", "-o", "ControlMaster=no", "-o", "ControlPath=none")
_REMOTE_PYTHON = '"$HOME"/venvs/kernel-env/bin/python'
_REMOTE_RUN_ROOT = ".cache/nkigym-simulate/runs"
_REMOTE_RESULT_POLL_SECONDS = 5.0
_WORKER_SOURCE = Path(__file__).with_name("simulate_nki_worker.py")
ArrayResult = np.ndarray | tuple[np.ndarray, ...]
_SerializedCase = tuple[int, str, str, str, dict[str, np.ndarray], ArrayResult]


@dataclass(frozen=True)
class FP32SimulationCase:
    """One rendered kernel and reference result for remote fp32 validation."""

    label: str
    kernel: str
    func_name: str
    inputs: dict[str, np.ndarray]
    expected: ArrayResult


_IndexedCase = tuple[int, FP32SimulationCase]


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
    assigned: int
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
    func = getattr(kernel, "func", kernel)
    source = _fp32_source(textwrap.dedent(inspect.getsource(func)))
    namespace: dict = dict(func.__globals__)
    exec(source, namespace)  # noqa: S102
    simulated = nki.simulate(namespace[func.__name__])

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Cast numpy inputs to fp32 and forward to the simulated kernel."""
        cast_args = tuple(_fp32_value(argument) for argument in args)
        cast_kwargs = {name: _fp32_value(argument) for name, argument in kwargs.items()}
        return cast(_R, simulated(*cast_args, **cast_kwargs))

    return wrapper


def batch_simulate_fp32(
    hosts: list[str],
    cases: list[FP32SimulationCase],
    atol: float,
    rtol: float,
    timeout_s: int = 7200,
    workers_per_host: int = 4,
) -> int:
    """Validate rendered kernels in fp32 across a list of SSH hosts.

    Exact duplicate kernels sharing the same inputs and reference output are
    simulated once. Workload groups are greedily balanced across hosts without
    copying their shared arrays to multiple hosts. Each host receives one pickle
    bundle, simulates with a local process pool, and compares outputs remotely.
    Only compact result metadata returns.

    Args:
        hosts: SSH destinations provisioned with ``~/venvs/kernel-env``.
        cases: Rendered kernels, inputs, and expected fp32 outputs.
        atol: Absolute tolerance passed to ``numpy.testing.assert_allclose``.
        rtol: Relative tolerance passed to ``numpy.testing.assert_allclose``.
        timeout_s: Total timeout for each remote host batch.
        workers_per_host: Simulator processes to run on each physical host.

    Returns:
        The number of successfully validated cases.
    """
    _validate_batch(hosts, cases, atol, rtol, timeout_s, workers_per_host)
    completed = 0
    if cases:
        _require_command("ssh")
        _require_command("rsync")
        unique_cases = _deduplicate_cases(cases)
        partitions = _partition_cases(hosts, unique_cases)
        with tempfile.TemporaryDirectory(prefix="nkigym-simulate-") as raw_directory:
            directory = Path(raw_directory)
            requests = _write_requests(directory, partitions, atol, rtol, workers_per_host)
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                futures = [
                    executor.submit(_run_remote_batch, host, request_path, result_path, timeout_s)
                    for host, request_path, result_path in requests
                ]
                results = [future.result() for future in futures]
        _raise_batch_failure(results)
        unique_completed = sum(result.completed for result in results)
        if unique_completed != len(unique_cases):
            raise RuntimeError(f"remote simulation completed {unique_completed} of {len(unique_cases)} distinct cases")
        completed = len(cases)
    return completed


def _fp32_value(value: object) -> object:
    """Cast floating NumPy inputs to fp32 while preserving integer arrays."""
    result = value.astype(np.float32) if isinstance(value, np.ndarray) and value.dtype.kind == "f" else value
    return result


def _validate_batch(
    hosts: list[str], cases: list[FP32SimulationCase], atol: float, rtol: float, timeout_s: int, workers_per_host: int
) -> None:
    """Reject malformed batch inputs before opening an SSH connection."""
    if not hosts:
        raise ValueError("at least one SSH host is required")
    for host in hosts:
        _validate_host(host)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("absolute tolerance must be finite and non-negative")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("relative tolerance must be finite and non-negative")
    if timeout_s <= 0:
        raise ValueError("batch simulation timeout must be positive")
    if not isinstance(workers_per_host, int) or isinstance(workers_per_host, bool) or workers_per_host <= 0:
        raise ValueError("workers per host must be a positive integer")
    for case in cases:
        if not case.label:
            raise ValueError("simulation case label must not be empty")
        if not case.kernel.strip():
            raise ValueError(f"{case.label}: kernel source must not be empty")
        if not case.func_name.isidentifier():
            raise ValueError(f"{case.label}: invalid kernel function name {case.func_name!r}")


def _validate_host(host: str) -> None:
    """Reject empty or option-shaped SSH destinations."""
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise ValueError(f"invalid SSH host {host!r}")


def _require_command(command: str) -> None:
    """Fail before transport when one local executable is unavailable."""
    if shutil.which(command) is None:
        raise FileNotFoundError(f"{command} is not on PATH")


def _deduplicate_cases(cases: list[FP32SimulationCase]) -> list[_IndexedCase]:
    """Keep the earliest state for each exact kernel and shared input set."""
    unique: list[_IndexedCase] = []
    seen: set[tuple[str, str, int, int]] = set()
    for index, case in enumerate(cases):
        key = (case.kernel, case.func_name, id(case.inputs), id(case.expected))
        if key not in seen:
            seen.add(key)
            unique.append((index, case))
    return unique


def _partition_cases(hosts: list[str], cases: list[_IndexedCase]) -> list[tuple[str, list[_SerializedCase]]]:
    """Balance shared-input case groups across hosts without duplicating arrays."""
    grouped: dict[tuple[int, int], list[_IndexedCase]] = {}
    for indexed_case in cases:
        case = indexed_case[1]
        grouped.setdefault((id(case.inputs), id(case.expected)), []).append(indexed_case)
    active_hosts = hosts[: min(len(hosts), len(grouped))]
    assigned: list[list[_IndexedCase]] = [[] for _host in active_hosts]
    weights = [0 for _host in active_hosts]
    weighted_groups = [(sum(len(case.kernel) for _index, case in group), group) for group in grouped.values()]
    for weight, group in sorted(weighted_groups, key=lambda item: (-item[0], item[1][0][0])):
        host_index = min(range(len(active_hosts)), key=lambda index: (weights[index], index))
        assigned[host_index].extend(group)
        weights[host_index] += weight
    partitions = []
    for host, host_cases in zip(active_hosts, assigned, strict=True):
        serialized = [
            (index, case.label, case.kernel, case.func_name, case.inputs, case.expected)
            for index, case in sorted(host_cases, key=lambda item: item[0])
        ]
        partitions.append((host, serialized))
    return partitions


def _write_requests(
    directory: Path,
    partitions: list[tuple[str, list[_SerializedCase]]],
    atol: float,
    rtol: float,
    workers_per_host: int,
) -> list[tuple[str, Path, Path]]:
    """Write one request bundle and result path per host."""
    requests: list[tuple[str, Path, Path]] = []
    for index, (host, cases) in enumerate(partitions):
        host_directory = directory / f"host-{index}"
        host_directory.mkdir()
        request_path = host_directory / "request.pkl"
        result_path = host_directory / "result.json"
        payload = (cases, atol, rtol, workers_per_host)
        with request_path.open("wb") as request_file:
            pickle.dump(payload, request_file, protocol=pickle.HIGHEST_PROTOCOL)
        requests.append((host, request_path, result_path))
    return requests


def _run_remote_batch(host: str, request_path: Path, result_path: Path, timeout_s: int) -> _HostResult:
    """Upload and execute one host partition, then parse its result."""
    run_id = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
    remote_run = f"{_REMOTE_RUN_ROOT}/{run_id}"
    rsync_shell = shlex.join(("ssh", *_SSH_OPTIONS))
    deadline = time.monotonic() + timeout_s
    log: list[str] = []
    failure: RuntimeError | None = None
    try:
        _run_transport_command(
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
            deadline,
            log,
        )
        _run_transport_command(
            "Uploading simulation batch",
            [
                "rsync",
                "-a",
                "-e",
                rsync_shell,
                str(_WORKER_SOURCE.resolve()),
                str(request_path),
                f"{host}:{remote_run}/",
            ],
            deadline,
            log,
        )
        _start_remote_worker(host, remote_run, deadline, log)
        _wait_for_remote_result(host, remote_run, deadline, log)
        _run_transport_command(
            "Downloading simulation result",
            ["rsync", "-a", "-e", rsync_shell, f"{host}:{remote_run}/result.json", str(result_path)],
            deadline,
            log,
        )
    except RuntimeError as error:
        failure = error
    finally:
        _cleanup_remote(host, remote_run, log)
    if failure is not None:
        detail = "".join(log)[-5000:]
        raise RuntimeError(f"SSH batch simulation failed for {host}: {failure}\n{detail}") from failure
    return _read_host_result(host, result_path)


def _start_remote_worker(host: str, remote_run: str, deadline: float, log: list[str]) -> None:
    """Start one detached worker whose result and exit status are atomic files."""
    worker = (
        f"{_REMOTE_PYTHON} "
        f'"$HOME"/{remote_run}/simulate_nki_worker.py '
        f'--worker "$HOME"/{remote_run}/request.pkl '
        f'"$HOME"/{remote_run}/result.json'
    )
    script = f'{worker}; status=$?; printf "%s\\n" "$status" > "$HOME"/{remote_run}/worker.exit'
    command = f"setsid -f sh -c {shlex.quote(script)} " f'>"$HOME"/{remote_run}/worker.log 2>&1 < /dev/null'
    _run_transport_command(
        "Starting remote simulation worker", ["ssh", *_SSH_WORKER_OPTIONS, host, command], deadline, log
    )


def _wait_for_remote_result(host: str, remote_run: str, deadline: float, log: list[str]) -> None:
    """Poll atomic worker files without depending on SSH session teardown."""
    result = f'"$HOME"/{remote_run}/result.json'
    exit_status = f'"$HOME"/{remote_run}/worker.exit'
    probe = (
        f"if test -f {result}; then printf ready; "
        f"elif test -f {exit_status}; then printf failed:; cat {exit_status}; "
        "else printf pending; fi"
    )
    log.append("==> Waiting for remote simulation result\n")
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise RuntimeError("Simulating kernel batch exceeded the batch simulation timeout")
        try:
            completed = subprocess.run(
                ["ssh", "-n", *_SSH_OPTIONS, host, probe],
                text=True,
                capture_output=True,
                timeout=min(30.0, remaining_s),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            _record_timeout_output(error, log)
            raise RuntimeError("Checking remote simulation result timed out") from error
        if completed.returncode != 0:
            log.extend((completed.stdout, completed.stderr))
            raise RuntimeError(f"Checking remote simulation result failed with exit {completed.returncode}")
        state = completed.stdout.strip()
        if state == "ready":
            return
        if state.startswith("failed:"):
            _run_transport_command(
                "Reading remote simulation failure",
                ["ssh", "-n", *_SSH_OPTIONS, host, f'cat "$HOME"/{remote_run}/worker.log'],
                deadline,
                log,
            )
            raise RuntimeError(f"remote simulation worker exited with status {state.removeprefix('failed:')}")
        if state != "pending":
            raise RuntimeError(f"remote simulation worker returned malformed state {state!r}")
        time.sleep(min(_REMOTE_RESULT_POLL_SECONDS, remaining_s))


def _run_transport_command(stage: str, command: list[str], deadline: float, log: list[str]) -> None:
    """Run one transport stage against a shared host deadline."""
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise RuntimeError(f"{stage} exceeded the batch simulation timeout")
    log.append(f"==> {stage}\n")
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=remaining_s, check=False)
    except subprocess.TimeoutExpired as error:
        _record_timeout_output(error, log)
        raise RuntimeError(f"{stage} exceeded the batch simulation timeout") from error
    log.extend((completed.stdout, completed.stderr))
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit {completed.returncode}")


def _record_timeout_output(error: subprocess.TimeoutExpired, log: list[str]) -> None:
    """Preserve partial subprocess output attached to a timeout."""
    stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
    stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
    if stdout:
        log.append(stdout)
    if stderr:
        log.append(stderr)


def _cleanup_remote(host: str, remote_run: str, log: list[str]) -> None:
    """Best-effort removal of one remote simulation directory."""
    command = ["ssh", *_SSH_OPTIONS, host, f'rm -rf "$HOME"/{remote_run}']
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
        if completed.returncode != 0:
            log.append("==> Remote cleanup failed\n")
            log.extend((completed.stdout, completed.stderr))
    except subprocess.TimeoutExpired:
        log.append("==> Remote cleanup timed out\n")


def _read_host_result(host: str, result_path: Path) -> _HostResult:
    """Parse and validate one downloaded worker result."""
    if not result_path.is_file():
        raise RuntimeError(f"SSH batch simulation returned no result for {host}")
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"SSH batch simulation returned malformed result for {host}")
    assigned = raw.get("assigned")
    completed = raw.get("completed")
    raw_failure = raw.get("failure")
    if not isinstance(assigned, int) or isinstance(assigned, bool) or assigned < 0:
        raise RuntimeError(f"SSH batch simulation returned invalid assigned count for {host}")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0 or completed > assigned:
        raise RuntimeError(f"SSH batch simulation returned invalid completed count for {host}")
    failure = _parse_failure(host, raw_failure)
    if failure is None and completed != assigned:
        raise RuntimeError(f"SSH batch simulation on {host} stopped without reporting a failure")
    return _HostResult(host=host, assigned=assigned, completed=completed, failure=failure)


def _parse_failure(host: str, raw_failure: object) -> _SimulationFailure | None:
    """Parse optional failure metadata from one remote host."""
    failure = None
    if raw_failure is not None:
        if not isinstance(raw_failure, dict):
            raise RuntimeError(f"SSH batch simulation returned malformed failure for {host}")
        case_index = raw_failure.get("case_index")
        label = raw_failure.get("label")
        exception_type = raw_failure.get("exception_type")
        remote_traceback = raw_failure.get("traceback")
        if (
            not isinstance(case_index, int)
            or isinstance(case_index, bool)
            or case_index < 0
            or not isinstance(label, str)
            or not isinstance(exception_type, str)
            or not isinstance(remote_traceback, str)
        ):
            raise RuntimeError(f"SSH batch simulation returned invalid failure fields for {host}")
        failure = _SimulationFailure(
            case_index=case_index, label=label, exception_type=exception_type, traceback=remote_traceback
        )
    return failure


def _raise_batch_failure(results: list[_HostResult]) -> None:
    """Raise the earliest simulation failure in original case order."""
    failures: list[tuple[str, _SimulationFailure]] = []
    for result in results:
        if result.failure is not None:
            failures.append((result.host, result.failure))
    if failures:
        host, failure = min(failures, key=lambda item: item[1].case_index)
        message = f"{failure.label}\nremote host: {host}\n{failure.traceback}"
        if failure.exception_type == "AssertionError":
            raise AssertionError(message)
        raise RuntimeError(message)
