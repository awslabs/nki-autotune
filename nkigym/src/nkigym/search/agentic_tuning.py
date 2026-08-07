"""Execute and validate one isolated agentic tuning search."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

_TIMEOUT_EXIT_CODE = 124
_START_FAILURE_EXIT_CODE = 127
_TERMINATION_GRACE_SECONDS = 5
_TERMINATION_POLL_SECONDS = 0.05
_REFINEMENT_PREFIX = "[refinement] "
_LOG_POLL_SECONDS = 0.1
_LOGGER = logging.getLogger(__name__)

AGENTIC_TUNING_CONTEXT_VERSION = 1
AGENTIC_TUNING_MAX_REGRESSION_POINTS = 1.0


def agentic_target_score(historical_best_score: float | None) -> float | None:
    """Return the minimum score accepted relative to historical evidence."""
    target = (
        None
        if historical_best_score is None
        else max(0.0, historical_best_score - AGENTIC_TUNING_MAX_REGRESSION_POINTS)
    )
    return target


@dataclass(frozen=True)
class AgenticTuningSpec:
    """Command contract for producing one measured agentic tuning trace."""

    name: str
    argv: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    timeout_seconds: int
    target_score: float | None = None

    def __post_init__(self) -> None:
        """Reject an invalid optional MFU target."""
        target = self.target_score
        if target is not None and (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(target)
            or target < 0.0
            or target > 100.0
        ):
            raise ValueError("agentic tuning target_score must be a finite percentage in [0, 100] or null")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "argv": list(self.argv),
            "required_artifacts": list(self.required_artifacts),
            "timeout_seconds": self.timeout_seconds,
            "target_score": self.target_score,
        }

    @classmethod
    def from_dict(cls, value: object) -> AgenticTuningSpec:
        """Decode an agentic tuning command from a JSON-compatible object."""
        if not isinstance(value, dict):
            raise ValueError("agentic tuning spec must be an object")
        name = value.get("name")
        raw_argv = value.get("argv")
        raw_artifacts = value.get("required_artifacts")
        timeout_seconds = value.get("timeout_seconds")
        raw_target_score = value.get("target_score")
        if not isinstance(name, str) or not name:
            raise ValueError("agentic tuning spec name must be a non-empty string")
        if not isinstance(raw_argv, list) or not raw_argv or any(not isinstance(arg, str) for arg in raw_argv):
            raise ValueError("agentic tuning spec argv must be a non-empty list of strings")
        if (
            not isinstance(raw_artifacts, list)
            or not raw_artifacts
            or any(not isinstance(path, str) or not path for path in raw_artifacts)
        ):
            raise ValueError("agentic tuning spec required_artifacts must be a non-empty list of strings")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError("agentic tuning spec timeout_seconds must be a positive integer")
        if raw_target_score is not None and (
            isinstance(raw_target_score, bool)
            or not isinstance(raw_target_score, (int, float))
            or not math.isfinite(raw_target_score)
            or raw_target_score < 0.0
            or raw_target_score > 100.0
        ):
            raise ValueError("agentic tuning spec target_score must be a finite percentage in [0, 100] or null")
        return cls(
            name=name,
            argv=tuple(raw_argv),
            required_artifacts=tuple(raw_artifacts),
            timeout_seconds=timeout_seconds,
            target_score=None if raw_target_score is None else float(raw_target_score),
        )


@dataclass(frozen=True)
class AgenticTuningContext:
    """Inputs shared by controller-driven candidate tuning."""

    program_directory: Path
    baseline_tree: str
    tuning: AgenticTuningSpec
    historical_best_score: float | None

    def __post_init__(self) -> None:
        """Derive the policy's accepted MFU floor from historical evidence."""
        target = agentic_target_score(self.historical_best_score)
        if target is not None:
            object.__setattr__(self, "tuning", replace(self.tuning, target_score=target))

    def as_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible tuning context."""
        return {
            "version": AGENTIC_TUNING_CONTEXT_VERSION,
            "program_directory": str(self.program_directory),
            "baseline_tree": self.baseline_tree,
            "agentic_tuning": self.tuning.as_dict(),
            "historical_best_score": self.historical_best_score,
        }

    @classmethod
    def from_dict(cls, value: object) -> AgenticTuningContext:
        """Decode and validate a controller-provided tuning context."""
        if not isinstance(value, dict):
            raise ValueError("agentic tuning context must be a JSON object")
        version = value.get("version")
        raw_program_directory = value.get("program_directory")
        baseline_tree = value.get("baseline_tree")
        raw_historical_best = value.get("historical_best_score")
        if version != AGENTIC_TUNING_CONTEXT_VERSION:
            raise ValueError(f"unsupported agentic tuning context version: {version!r}")
        if not isinstance(raw_program_directory, str) or not raw_program_directory:
            raise ValueError("agentic tuning context program_directory must be a non-empty string")
        if not isinstance(baseline_tree, str) or not baseline_tree:
            raise ValueError("agentic tuning context baseline_tree must be a non-empty string")
        if raw_historical_best is not None and (
            isinstance(raw_historical_best, bool)
            or not isinstance(raw_historical_best, (int, float))
            or not math.isfinite(raw_historical_best)
        ):
            raise ValueError("agentic tuning context historical_best_score must be finite or null")
        historical_best = None if raw_historical_best is None else float(raw_historical_best)
        return cls(
            program_directory=Path(raw_program_directory).expanduser().resolve(),
            baseline_tree=baseline_tree,
            tuning=AgenticTuningSpec.from_dict(value.get("agentic_tuning")),
            historical_best_score=historical_best,
        )

    @classmethod
    def from_path(cls, path: Path) -> AgenticTuningContext:
        """Load a tuning context from one JSON artifact."""
        decoded = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class AgenticTuningResult:
    """Process and artifact result from one agentic tuning search."""

    command: tuple[str, ...]
    exit_code: int
    timed_out: bool
    duration_seconds: float
    artifact_directory: Path
    log_path: Path
    missing_artifacts: tuple[str, ...]
    worktree_modified: bool
    program_modified: bool
    best_score: float | None
    result_error: str | None

    @property
    def passed(self) -> bool:
        """Return whether tuning completed without corrupting evidence or inputs."""
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.missing_artifacts
            and not self.worktree_modified
            and not self.program_modified
            and self.result_error is None
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "artifact_directory": str(self.artifact_directory),
            "log_path": str(self.log_path),
            "missing_artifacts": list(self.missing_artifacts),
            "worktree_modified": self.worktree_modified,
            "program_modified": self.program_modified,
            "best_score": self.best_score,
            "result_error": self.result_error,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> AgenticTuningResult:
        """Decode an agentic tuning result from a JSON-compatible object."""
        if not isinstance(value, dict):
            raise ValueError("agentic tuning result must be an object")
        raw_command = value.get("command")
        exit_code = value.get("exit_code")
        timed_out = value.get("timed_out")
        raw_duration = value.get("duration_seconds")
        raw_artifact_directory = value.get("artifact_directory")
        raw_log_path = value.get("log_path")
        raw_missing = value.get("missing_artifacts")
        worktree_modified = value.get("worktree_modified")
        program_modified = value.get("program_modified")
        raw_best_score = value.get("best_score")
        result_error = value.get("result_error")
        if (
            not isinstance(raw_command, list)
            or not raw_command
            or any(not isinstance(argument, str) for argument in raw_command)
        ):
            raise ValueError("agentic tuning result command must be a non-empty list of strings")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("agentic tuning result exit_code must be an integer")
        if not isinstance(timed_out, bool):
            raise ValueError("agentic tuning result timed_out must be a boolean")
        if (
            isinstance(raw_duration, bool)
            or not isinstance(raw_duration, (int, float))
            or not math.isfinite(raw_duration)
            or raw_duration < 0
        ):
            raise ValueError("agentic tuning result duration_seconds must be finite and non-negative")
        if not isinstance(raw_artifact_directory, str) or not raw_artifact_directory:
            raise ValueError("agentic tuning result artifact_directory must be a non-empty string")
        if not isinstance(raw_log_path, str) or not raw_log_path:
            raise ValueError("agentic tuning result log_path must be a non-empty string")
        if not isinstance(raw_missing, list) or any(not isinstance(path, str) for path in raw_missing):
            raise ValueError("agentic tuning result missing_artifacts must be a list of strings")
        if not isinstance(worktree_modified, bool):
            raise ValueError("agentic tuning result worktree_modified must be a boolean")
        if not isinstance(program_modified, bool):
            raise ValueError("agentic tuning result program_modified must be a boolean")
        if raw_best_score is not None and (
            isinstance(raw_best_score, bool)
            or not isinstance(raw_best_score, (int, float))
            or not math.isfinite(raw_best_score)
        ):
            raise ValueError("agentic tuning result best_score must be finite or null")
        if result_error is not None and not isinstance(result_error, str):
            raise ValueError("agentic tuning result result_error must be a string or null")
        best_score = None if raw_best_score is None else float(raw_best_score)
        return cls(
            command=tuple(raw_command),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=float(raw_duration),
            artifact_directory=Path(raw_artifact_directory),
            log_path=Path(raw_log_path),
            missing_artifacts=tuple(raw_missing),
            worktree_modified=worktree_modified,
            program_modified=program_modified,
            best_score=best_score,
            result_error=result_error,
        )


def _relay_refinement_line(line: str) -> None:
    """Relay one useful child-process update through search logging."""
    message = line.rstrip()
    if message.startswith(_REFINEMENT_PREFIX):
        _LOGGER.info("tuning | %s", message.removeprefix(_REFINEMENT_PREFIX))


def _relay_refinement_updates(log: TextIO, stopped: threading.Event) -> None:
    """Follow a tuning log until the owning subprocess finishes."""
    while True:
        line = log.readline()
        if line:
            _relay_refinement_line(line)
        elif stopped.wait(_LOG_POLL_SECONDS):
            for remaining in log:
                _relay_refinement_line(remaining)
            break


def _required_path(artifact_directory: Path, relative: str) -> Path:
    """Resolve one required tuning artifact without permitting traversal."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
        raise ValueError(f"invalid required agentic tuning artifact path: {relative!r}")
    return artifact_directory / path


def _read_best_score(artifact_directory: Path) -> tuple[float | None, str | None]:
    """Read the best finite score from an agentic tuning result."""
    result_path = artifact_directory / "result.json"
    best_score: float | None = None
    error: str | None = None
    if result_path.is_file():
        try:
            decoded = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("result must be a JSON object")
            history = decoded.get("history")
            if not isinstance(history, list):
                raise ValueError("result history must be a list")
            scores: list[float] = []
            for index, entry in enumerate(history):
                if not isinstance(entry, dict):
                    raise ValueError(f"result history entry {index} must be an object")
                score = entry.get("score")
                if score is None:
                    continue
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"result history entry {index} has invalid score {score!r}")
                scores.append(float(score))
            if scores:
                best_score = max(scores)
        except (OSError, json.JSONDecodeError, ValueError) as caught:
            error = str(caught)
    return best_score, error


def _program_snapshot(program_directory: Path) -> tuple[tuple[str, bytes], ...]:
    """Capture every file that can affect a persisted tuning program."""
    files = sorted(path for path in program_directory.rglob("*") if path.is_file())
    return tuple((str(path.relative_to(program_directory)), path.read_bytes()) for path in files)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the tuning process group and wait for shutdown."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
    else:
        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(_TERMINATION_POLL_SECONDS)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
        process.wait()


def _run_process(
    command: tuple[str, ...],
    worktree: Path,
    environment: Mapping[str, str],
    log_path: Path,
    header: str,
    timeout_seconds: int,
) -> tuple[int, bool, float]:
    """Run one isolated tuning command and capture its combined output."""
    started = time.monotonic()
    timed_out = False
    exit_code = _START_FAILURE_EXIT_CODE
    relay_stopped = threading.Event()
    relay_thread: threading.Thread | None = None
    relay_log: TextIO | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        if _LOGGER.isEnabledFor(logging.INFO):
            relay_log = log_path.open(encoding="utf-8", errors="replace")
            relay_thread = threading.Thread(
                target=_relay_refinement_updates,
                args=(relay_log, relay_stopped),
                name="nkigym-agentic-tuning-log",
                daemon=True,
            )
            relay_thread.start()
        try:
            process = subprocess.Popen(
                command,
                cwd=worktree,
                env=dict(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            log.write(f"\nFailed to start agentic tuning: {error}\n")
        else:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = _TIMEOUT_EXIT_CODE
                _terminate_process_group(process)
                log.write(f"\nAgentic tuning exceeded timeout of {timeout_seconds} seconds\n")
            except KeyboardInterrupt:
                _terminate_process_group(process)
                raise
            else:
                if process.returncode is None:
                    raise RuntimeError("agentic tuning process completed without a return code")
                exit_code = process.returncode
        finally:
            if relay_thread is not None:
                relay_stopped.set()
                relay_thread.join()
            if relay_log is not None:
                relay_log.close()
        duration = time.monotonic() - started
        log.write(f"\nexit_code={exit_code} duration_seconds={duration:.3f}\n")
    return exit_code, timed_out, duration


def _candidate_environment(worktree: Path) -> dict[str, str]:
    """Build the subprocess environment for one candidate checkout."""
    environment = dict(os.environ)
    source_roots = (worktree / "nkigym/src", worktree)
    existing = environment.get("PYTHONPATH")
    entries = [str(path) for path in source_roots]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def run_agentic_tuning(
    spec: AgenticTuningSpec,
    program_directory: Path,
    worktree: Path,
    output_directory: Path,
    *,
    source_fingerprint: Callable[[], str],
) -> AgenticTuningResult:
    """Run one evidence-producing search without permitting input changes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = output_directory / "trace"
    artifact_directory.mkdir()
    isolated_program_directory = output_directory / "program"
    shutil.copytree(program_directory, isolated_program_directory)
    program_before = _program_snapshot(isolated_program_directory)
    source_before = source_fingerprint()
    log_path = output_directory / "agentic-tuning.log"
    target_arguments = () if spec.target_score is None else ("--target-score", str(spec.target_score))
    command = (
        *spec.argv,
        *target_arguments,
        "--trace-dir",
        str(artifact_directory),
        "--program-dir",
        str(isolated_program_directory),
    )
    process_environment = _candidate_environment(worktree)
    process_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    header = (
        f"source_program_dir={program_directory}\n"
        f"program_dir={isolated_program_directory}\n"
        f"trace_dir={artifact_directory}\n\n"
    )
    _LOGGER.info("tuning | started | artifacts=%s | log=%s", artifact_directory, log_path)
    exit_code, timed_out, duration = _run_process(
        command, worktree, process_environment, log_path, header, spec.timeout_seconds
    )
    missing = tuple(
        relative for relative in spec.required_artifacts if not _required_path(artifact_directory, relative).is_file()
    )
    worktree_modified = source_before != source_fingerprint()
    program_modified = program_before != _program_snapshot(isolated_program_directory)
    if worktree_modified:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Agentic tuning modified candidate source files\n")
    if program_modified:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Agentic tuning modified its isolated program files\n")
    best_score, result_error = _read_best_score(artifact_directory)
    result = AgenticTuningResult(
        command=command,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration,
        artifact_directory=artifact_directory,
        log_path=log_path,
        missing_artifacts=missing,
        worktree_modified=worktree_modified,
        program_modified=program_modified,
        best_score=best_score,
        result_error=result_error,
    )
    status = "passed" if result.passed else "failed"
    score = "unavailable" if result.best_score is None else f"{result.best_score:.2f}%"
    _LOGGER.info(
        "tuning | %s | best_mfu=%s | duration=%.1fs | log=%s", status, score, result.duration_seconds, result.log_path
    )
    return result


__all__ = [
    "AGENTIC_TUNING_CONTEXT_VERSION",
    "AGENTIC_TUNING_MAX_REGRESSION_POINTS",
    "AgenticTuningContext",
    "AgenticTuningResult",
    "AgenticTuningSpec",
    "agentic_target_score",
    "run_agentic_tuning",
]
