"""Deterministic candidate gate execution and logging."""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from develop_nkigym.types import GateResult, GateSpec

_TIMEOUT_EXIT_CODE = 124
_START_FAILURE_EXIT_CODE = 127
_TERMINATION_GRACE_SECONDS = 5
_TERMINATION_POLL_SECONDS = 0.05
_LOGGER = logging.getLogger(__name__)
_GATE_ARTIFACT_DIRECTORY_ENV = "NKIGYM_GATE_ARTIFACT_DIRECTORY"


def candidate_environment(worktree: Path) -> dict[str, str]:
    """Build an environment that imports candidate code before controller code."""
    environment = dict(os.environ)
    nkigym_source = worktree / "nkigym/src"
    if not (nkigym_source / "nkigym").is_dir():
        raise ValueError(f"candidate nkigym source is missing: {nkigym_source}")
    source_roots = (nkigym_source, worktree)
    existing = environment.get("PYTHONPATH")
    entries = [str(path) for path in source_roots]
    if existing:
        skill_scripts = Path(__file__).resolve().parents[1]
        entries.extend(
            entry
            for entry in existing.split(os.pathsep)
            if entry and Path(entry).expanduser().resolve() != skill_scripts
        )
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _log_name(name: str) -> str:
    """Convert a gate name into a stable log filename."""
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-")
    if not normalized:
        raise ValueError(f"gate name does not contain a usable filename: {name!r}")
    return f"{normalized}.log"


def _linux_process_group_has_active_members(process_group_id: int) -> bool:
    """Return whether Linux procfs shows a non-zombie group member."""
    active = False
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text(encoding="utf-8").rpartition(") ")[2].split()
        except OSError:
            continue
        if len(fields) >= 3 and fields[0] != "Z" and int(fields[2]) == process_group_id:
            active = True
            break
    return active


def _process_group_has_active_members(process_group_id: int) -> bool:
    """Return whether the operating system still has an active group member."""
    exists = True
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        exists = False
    if exists and Path("/proc").is_dir():
        exists = _linux_process_group_has_active_members(process_group_id)
    return exists


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the gate process group and wait for shutdown."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
    else:
        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        while _process_group_has_active_members(process.pid) and time.monotonic() < deadline:
            time.sleep(_TERMINATION_POLL_SECONDS)
        if _process_group_has_active_members(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                _LOGGER.debug("gate process group %s exited before SIGKILL", process.pid)
        process.wait()


def _interrupt_process_group(process: subprocess.Popen[str]) -> None:
    """Interrupt a gate so nested subprocess cleanup handlers can run."""
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
    else:
        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        while _process_group_has_active_members(process.pid) and time.monotonic() < deadline:
            time.sleep(_TERMINATION_POLL_SECONDS)
        if _process_group_has_active_members(process.pid):
            _terminate_process_group(process)
        else:
            process.wait()


def run_gate(
    spec: GateSpec, worktree: Path, gate_directory: Path, additional_environment: tuple[tuple[str, str], ...] = ()
) -> GateResult:
    """Run one gate in the candidate and capture combined output."""
    candidate_root = worktree.resolve()
    working_directory = (candidate_root / spec.working_directory).resolve()
    if not working_directory.is_relative_to(candidate_root):
        raise ValueError(f"gate working directory escapes the candidate worktree: {spec.working_directory}")
    if not working_directory.is_dir():
        raise ValueError(f"gate working directory does not exist: {working_directory}")
    log_path = gate_directory / _log_name(spec.name)
    artifact_directory = gate_directory / f"{log_path.stem}-artifacts"
    environment = candidate_environment(candidate_root)
    environment.update(spec.environment)
    environment.update(additional_environment)
    environment[_GATE_ARTIFACT_DIRECTORY_ENV] = str(artifact_directory)
    _LOGGER.info("gate | started | %s | log=%s", spec.name, log_path)
    started = time.monotonic()
    timed_out = False
    exit_code = _START_FAILURE_EXIT_CODE
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(spec.argv)}\n\n")
        try:
            process = subprocess.Popen(
                spec.argv,
                cwd=working_directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            log.write(f"\nFailed to start gate: {error}\n")
        else:
            try:
                process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = _TIMEOUT_EXIT_CODE
                _terminate_process_group(process)
                log.write(f"\nGate exceeded timeout of {spec.timeout_seconds} seconds\n")
            except KeyboardInterrupt:
                _interrupt_process_group(process)
                raise
            else:
                if process.returncode is None:
                    raise RuntimeError("gate process completed without a return code")
                exit_code = process.returncode
        duration = time.monotonic() - started
        log.write(f"\nexit_code={exit_code} duration_seconds={duration:.3f}\n")
    result = GateResult(
        name=spec.name,
        argv=spec.argv,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration,
        log_path=log_path,
        artifact_directory=artifact_directory,
    )
    status = "passed" if result.passed else "failed"
    _LOGGER.info("gate | %s | %s | duration=%.1fs | log=%s", status, spec.name, duration, log_path)
    return result


def run_gates(
    specs: tuple[GateSpec, ...],
    worktree: Path,
    gate_directory: Path,
    additional_environments: Mapping[str, tuple[tuple[str, str], ...]] | None = None,
) -> tuple[GateResult, ...]:
    """Run gates sequentially through the first failure."""
    gate_directory.mkdir(parents=True, exist_ok=True)
    results: list[GateResult] = []
    for spec in specs:
        environment = () if additional_environments is None else additional_environments.get(spec.name, ())
        result = run_gate(spec, worktree, gate_directory, environment)
        results.append(result)
        if not result.passed:
            break
    return tuple(results)


__all__ = ["candidate_environment", "run_gate", "run_gates"]
