"""Typed contracts for deterministic develop-nkigym skill support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nkigym.search.agentic_tuning import AgenticTuningResult, AgenticTuningSpec
from nkigym.search.program import ProgramSpec

NextAction = Literal["validate", "tune", "edit", "check", "accept", "complete"]
WorkflowMode = Literal["repair", "improve"]
RUN_RECORD_SCHEMA_VERSION = 18


def _required_dict(value: object, name: str) -> dict[str, object]:
    """Return one required JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(value: object, name: str) -> str:
    """Return one required non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_integer(value: object, name: str) -> int:
    """Return one required non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_percentage(value: object, name: str) -> float | None:
    """Return one optional finite percentage."""
    if value is None:
        percentage = None
    elif (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        or value > 100.0
    ):
        raise ValueError(f"{name} must be a finite percentage in [0, 100] or null")
    else:
        percentage = float(value)
    return percentage


@dataclass(frozen=True)
class GateSpec:
    """One deterministic command used to evaluate a candidate."""

    name: str
    argv: tuple[str, ...]
    working_directory: str
    timeout_seconds: int
    environment: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "argv": list(self.argv),
            "working_directory": self.working_directory,
            "timeout_seconds": self.timeout_seconds,
            "environment": dict(self.environment),
        }

    @classmethod
    def from_dict(cls, value: object) -> GateSpec:
        """Decode one gate specification."""
        decoded = _required_dict(value, "gate spec")
        name = _required_string(decoded.get("name"), "gate spec name")
        raw_argv = decoded.get("argv")
        working_directory = _required_string(decoded.get("working_directory"), "gate working_directory")
        timeout_seconds = _required_integer(decoded.get("timeout_seconds"), "gate timeout_seconds")
        raw_environment = decoded.get("environment")
        if not isinstance(raw_argv, list) or not raw_argv or any(not isinstance(item, str) for item in raw_argv):
            raise ValueError("gate argv must be a non-empty list of strings")
        if not isinstance(raw_environment, dict) or any(
            not isinstance(key, str) or not isinstance(item, str) for key, item in raw_environment.items()
        ):
            raise ValueError("gate environment must be an object of strings")
        return cls(
            name=name,
            argv=tuple(raw_argv),
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
            environment=tuple((key, item) for key, item in raw_environment.items()),
        )


@dataclass(frozen=True)
class GateResult:
    """Result and artifact locations for one candidate gate."""

    name: str
    argv: tuple[str, ...]
    exit_code: int
    timed_out: bool
    duration_seconds: float
    log_path: Path
    artifact_directory: Path

    @property
    def passed(self) -> bool:
        """Return whether the gate completed successfully."""
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "log_path": str(self.log_path),
            "artifact_directory": str(self.artifact_directory),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> GateResult:
        """Decode one gate result."""
        decoded = _required_dict(value, "gate result")
        raw_argv = decoded.get("argv")
        exit_code = decoded.get("exit_code")
        timed_out = decoded.get("timed_out")
        duration = decoded.get("duration_seconds")
        if not isinstance(raw_argv, list) or any(not isinstance(item, str) for item in raw_argv):
            raise ValueError("gate result argv must be a list of strings")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("gate result exit_code must be an integer")
        if not isinstance(timed_out, bool):
            raise ValueError("gate result timed_out must be a boolean")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("gate result duration_seconds must be finite and non-negative")
        return cls(
            name=_required_string(decoded.get("name"), "gate result name"),
            argv=tuple(raw_argv),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=float(duration),
            log_path=Path(_required_string(decoded.get("log_path"), "gate result log_path")),
            artifact_directory=Path(
                _required_string(decoded.get("artifact_directory"), "gate result artifact_directory")
            ),
        )


@dataclass(frozen=True)
class BaselineCheckAttempt:
    """Immutable canonical-gate evidence for one unchanged baseline tree."""

    index: int
    artifact_directory: Path
    baseline_tree: str
    gates: tuple[GateResult, ...]
    worktree_modified: bool
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "index": self.index,
            "artifact_directory": str(self.artifact_directory),
            "baseline_tree": self.baseline_tree,
            "gates": [gate.as_dict() for gate in self.gates],
            "worktree_modified": self.worktree_modified,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> BaselineCheckAttempt:
        """Decode one baseline check attempt."""
        decoded = _required_dict(value, "baseline check attempt")
        raw_gates = decoded.get("gates")
        worktree_modified = decoded.get("worktree_modified")
        passed = decoded.get("passed")
        if not isinstance(raw_gates, list):
            raise ValueError("baseline check attempt gates must be a list")
        if not isinstance(worktree_modified, bool):
            raise ValueError("baseline check attempt worktree_modified must be a boolean")
        if not isinstance(passed, bool):
            raise ValueError("baseline check attempt passed must be a boolean")
        return cls(
            index=_required_integer(decoded.get("index"), "baseline check attempt index"),
            artifact_directory=Path(
                _required_string(decoded.get("artifact_directory"), "baseline check attempt artifact_directory")
            ),
            baseline_tree=_required_string(decoded.get("baseline_tree"), "baseline check attempt baseline_tree"),
            gates=tuple(GateResult.from_dict(item) for item in raw_gates),
            worktree_modified=worktree_modified,
            passed=passed,
        )


@dataclass(frozen=True)
class CheckAttempt:
    """Immutable evidence for one checked candidate tree."""

    index: int
    artifact_directory: Path
    candidate_tree: str
    candidate_fingerprint: str
    changed_files: tuple[str, ...]
    patch_path: Path
    gates: tuple[GateResult, ...]
    worktree_modified: bool
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "index": self.index,
            "artifact_directory": str(self.artifact_directory),
            "candidate_tree": self.candidate_tree,
            "candidate_fingerprint": self.candidate_fingerprint,
            "changed_files": list(self.changed_files),
            "patch_path": str(self.patch_path),
            "gates": [gate.as_dict() for gate in self.gates],
            "worktree_modified": self.worktree_modified,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> CheckAttempt:
        """Decode one checked candidate."""
        decoded = _required_dict(value, "check attempt")
        raw_changed = decoded.get("changed_files")
        raw_gates = decoded.get("gates")
        worktree_modified = decoded.get("worktree_modified")
        passed = decoded.get("passed")
        if not isinstance(raw_changed, list) or any(not isinstance(item, str) for item in raw_changed):
            raise ValueError("check attempt changed_files must be a list of strings")
        if not isinstance(raw_gates, list):
            raise ValueError("check attempt gates must be a list")
        if not isinstance(worktree_modified, bool):
            raise ValueError("check attempt worktree_modified must be a boolean")
        if not isinstance(passed, bool):
            raise ValueError("check attempt passed must be a boolean")
        return cls(
            index=_required_integer(decoded.get("index"), "check attempt index"),
            artifact_directory=Path(
                _required_string(decoded.get("artifact_directory"), "check attempt artifact_directory")
            ),
            candidate_tree=_required_string(decoded.get("candidate_tree"), "check attempt candidate_tree"),
            candidate_fingerprint=_required_string(
                decoded.get("candidate_fingerprint"), "check attempt candidate_fingerprint"
            ),
            changed_files=tuple(raw_changed),
            patch_path=Path(_required_string(decoded.get("patch_path"), "check attempt patch_path")),
            gates=tuple(GateResult.from_dict(item) for item in raw_gates),
            worktree_modified=worktree_modified,
            passed=passed,
        )


@dataclass(frozen=True)
class CycleState:
    """Durable state for one evidence-driven implementation cycle."""

    index: int
    baseline_tree: str
    baseline_checks: tuple[BaselineCheckAttempt, ...]
    baseline_tuning: AgenticTuningResult | None
    tuning_attempts: tuple[AgenticTuningResult, ...]
    checks: tuple[CheckAttempt, ...]
    accepted_tree: str | None

    @property
    def passed(self) -> bool:
        """Return whether this cycle promoted a candidate tree."""
        return self.accepted_tree is not None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "index": self.index,
            "baseline_tree": self.baseline_tree,
            "baseline_checks": [attempt.as_dict() for attempt in self.baseline_checks],
            "baseline_tuning": self.baseline_tuning.as_dict() if self.baseline_tuning is not None else None,
            "tuning_attempts": [attempt.as_dict() for attempt in self.tuning_attempts],
            "checks": [attempt.as_dict() for attempt in self.checks],
            "accepted_tree": self.accepted_tree,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> CycleState:
        """Decode one cycle state."""
        decoded = _required_dict(value, "cycle")
        raw_baseline_checks = decoded.get("baseline_checks", [])
        raw_baseline_tuning = decoded.get("baseline_tuning")
        raw_tuning_attempts = decoded.get("tuning_attempts")
        raw_checks = decoded.get("checks")
        raw_accepted_tree = decoded.get("accepted_tree")
        if not isinstance(raw_baseline_checks, list):
            raise ValueError("cycle baseline_checks must be a list")
        if not isinstance(raw_tuning_attempts, list):
            raise ValueError("cycle tuning_attempts must be a list")
        if not isinstance(raw_checks, list):
            raise ValueError("cycle checks must be a list")
        if raw_accepted_tree is not None and (not isinstance(raw_accepted_tree, str) or not raw_accepted_tree):
            raise ValueError("cycle accepted_tree must be a non-empty string or null")
        return cls(
            index=_required_integer(decoded.get("index"), "cycle index"),
            baseline_tree=_required_string(decoded.get("baseline_tree"), "cycle baseline_tree"),
            baseline_checks=tuple(BaselineCheckAttempt.from_dict(item) for item in raw_baseline_checks),
            baseline_tuning=(
                None if raw_baseline_tuning is None else AgenticTuningResult.from_dict(raw_baseline_tuning)
            ),
            tuning_attempts=tuple(AgenticTuningResult.from_dict(item) for item in raw_tuning_attempts),
            checks=tuple(CheckAttempt.from_dict(item) for item in raw_checks),
            accepted_tree=raw_accepted_tree,
        )


@dataclass(frozen=True)
class RunConfig:
    """Inputs for creating one isolated development run."""

    repository: Path
    artifact_root: Path
    program: ProgramSpec
    agentic_tuning: AgenticTuningSpec
    gates: tuple[GateSpec, ...]
    improvement_round_limit: int
    initial_historical_best_score: float | None
    base_revision: str


@dataclass(frozen=True)
class RunRecord:
    """Durable mechanical state for one resumable development run."""

    run_id: str
    source_repository: Path
    run_directory: Path
    worktree: Path
    base_sha: str
    initial_tree: str
    program_directory: Path
    agentic_tuning: AgenticTuningSpec
    gates: tuple[GateSpec, ...]
    improvement_round_limit: int
    initial_historical_best_score: float | None
    cycles: tuple[CycleState, ...]

    @property
    def current_cycle(self) -> CycleState:
        """Return the active implementation cycle."""
        if not self.cycles:
            raise RuntimeError("run record contains no cycles")
        return self.cycles[-1]

    def as_dict(self) -> dict[str, object]:
        """Return the versioned JSON-compatible run record."""
        return {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_repository": str(self.source_repository),
            "run_directory": str(self.run_directory),
            "worktree": str(self.worktree),
            "base_sha": self.base_sha,
            "initial_tree": self.initial_tree,
            "program_directory": str(self.program_directory),
            "agentic_tuning": self.agentic_tuning.as_dict(),
            "gates": [gate.as_dict() for gate in self.gates],
            "improvement_round_limit": self.improvement_round_limit,
            "initial_historical_best_score": self.initial_historical_best_score,
            "cycles": [cycle.as_dict() for cycle in self.cycles],
        }

    @classmethod
    def from_dict(cls, value: object) -> RunRecord:
        """Decode one versioned run record."""
        decoded = _required_dict(value, "run record")
        version = decoded.get("schema_version")
        raw_gates = decoded.get("gates")
        raw_cycles = decoded.get("cycles")
        if version not in {16, 17, RUN_RECORD_SCHEMA_VERSION}:
            raise ValueError(f"unsupported run record schema version: {version!r}")
        if not isinstance(raw_gates, list):
            raise ValueError("run record gates must be a list")
        if not isinstance(raw_cycles, list) or not raw_cycles:
            raise ValueError("run record cycles must be a non-empty list")
        return cls(
            run_id=_required_string(decoded.get("run_id"), "run record run_id"),
            source_repository=Path(_required_string(decoded.get("source_repository"), "run record source_repository")),
            run_directory=Path(_required_string(decoded.get("run_directory"), "run record run_directory")),
            worktree=Path(_required_string(decoded.get("worktree"), "run record worktree")),
            base_sha=_required_string(decoded.get("base_sha"), "run record base_sha"),
            initial_tree=_required_string(decoded.get("initial_tree"), "run record initial_tree"),
            program_directory=Path(_required_string(decoded.get("program_directory"), "run record program_directory")),
            agentic_tuning=AgenticTuningSpec.from_dict(decoded.get("agentic_tuning")),
            gates=tuple(GateSpec.from_dict(item) for item in raw_gates),
            improvement_round_limit=_required_integer(
                decoded.get("improvement_round_limit", 1), "run record improvement_round_limit"
            ),
            initial_historical_best_score=_optional_percentage(
                decoded.get("initial_historical_best_score"), "run record initial_historical_best_score"
            ),
            cycles=tuple(CycleState.from_dict(item) for item in raw_cycles),
        )


@dataclass(frozen=True)
class RunStatus:
    """Current run state exposed to the controlling Codex skill."""

    run_id: str
    run_directory: Path
    worktree: Path
    program_directory: Path
    cycle_index: int
    baseline_tree: str
    candidate_fingerprint: str
    mode: WorkflowMode
    improvement_round_limit: int
    completed_improvement_rounds: int
    historical_best_score: float | None
    latest_baseline_check_directory: Path | None
    baseline_tuning_artifact_directory: Path | None
    latest_tuning_log: Path | None
    latest_check_directory: Path | None
    failed_gate_log: Path | None
    latest_check_worktree_modified: bool
    changed_files: tuple[str, ...]
    next_action: NextAction

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible status record."""
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "worktree": str(self.worktree),
            "program_directory": str(self.program_directory),
            "cycle_index": self.cycle_index,
            "baseline_tree": self.baseline_tree,
            "candidate_fingerprint": self.candidate_fingerprint,
            "mode": self.mode,
            "improvement_round_limit": self.improvement_round_limit,
            "completed_improvement_rounds": self.completed_improvement_rounds,
            "historical_best_score": self.historical_best_score,
            "latest_baseline_check_directory": (
                None if self.latest_baseline_check_directory is None else str(self.latest_baseline_check_directory)
            ),
            "baseline_tuning_artifact_directory": (
                None
                if self.baseline_tuning_artifact_directory is None
                else str(self.baseline_tuning_artifact_directory)
            ),
            "latest_tuning_log": None if self.latest_tuning_log is None else str(self.latest_tuning_log),
            "latest_check_directory": (
                None if self.latest_check_directory is None else str(self.latest_check_directory)
            ),
            "failed_gate_log": None if self.failed_gate_log is None else str(self.failed_gate_log),
            "latest_check_worktree_modified": self.latest_check_worktree_modified,
            "changed_files": list(self.changed_files),
            "next_action": self.next_action,
        }


__all__ = [
    "RUN_RECORD_SCHEMA_VERSION",
    "BaselineCheckAttempt",
    "CheckAttempt",
    "CycleState",
    "GateResult",
    "GateSpec",
    "NextAction",
    "RunConfig",
    "RunRecord",
    "RunStatus",
    "WorkflowMode",
]
