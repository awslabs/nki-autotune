"""Shared typed contracts for the workflow orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nkigym.search.agentic_tuning import AgenticTuningResult, AgenticTuningSpec
from nkigym.search.program import ProgramSpec

Verdict = Literal["passed", "failed", "stopped"]


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


@dataclass(frozen=True)
class RunConfig:
    """Controls for one isolated continuous IR-development workflow."""

    repository: Path
    artifact_root: Path
    program: ProgramSpec
    agentic_tuning: AgenticTuningSpec
    gates: tuple[GateSpec, ...]
    goal: str
    codex_executable: str
    base_revision: str
    max_cycles: int | None
    max_thread_start_attempts: int
    agent_timeout_seconds: int


@dataclass(frozen=True)
class CodexInvocationResult:
    """Operational result of one Codex CLI invocation."""

    command: tuple[str, ...]
    exit_code: int
    timed_out: bool
    duration_seconds: float
    thread_id: str | None
    event_log: Path
    stderr_log: Path
    final_message: Path

    @property
    def passed(self) -> bool:
        """Return whether Codex completed without a timeout or process error."""
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "command": list(self.command),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "thread_id": self.thread_id,
            "event_log": str(self.event_log),
            "stderr_log": str(self.stderr_log),
            "final_message": str(self.final_message),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GateResult:
    """Result and log location for one candidate gate."""

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


@dataclass(frozen=True)
class AttemptResult:
    """All controller-owned evidence for one candidate attempt."""

    index: int
    codex: CodexInvocationResult
    changed_files: tuple[str, ...]
    patch_path: Path
    gates: tuple[GateResult, ...]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "index": self.index,
            "codex": self.codex.as_dict(),
            "changed_files": list(self.changed_files),
            "patch_path": str(self.patch_path),
            "has_diff": bool(self.changed_files),
            "gates": [gate.as_dict() for gate in self.gates],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CycleResult:
    """One accepted, failed, or in-progress IR development cycle."""

    index: int
    baseline_tree: str
    accepted_tree: str | None
    thread_id: str | None
    baseline_refinement: AgenticTuningResult
    historical_best_score: float | None
    attempts: tuple[AttemptResult, ...]
    failure_reason: str | None

    @property
    def passed(self) -> bool:
        """Return whether the cycle produced a fully accepted candidate."""
        passed = self.accepted_tree is not None and bool(self.attempts) and self.attempts[-1].passed
        return passed

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "index": self.index,
            "baseline_tree": self.baseline_tree,
            "accepted_tree": self.accepted_tree,
            "thread_id": self.thread_id,
            "baseline_refinement": self.baseline_refinement.as_dict(),
            "historical_best_score": self.historical_best_score,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "failure_reason": self.failure_reason,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class WorkflowResult:
    """Final location and state for one bounded or explicitly stopped run."""

    run_id: str
    verdict: Verdict
    run_directory: Path
    worktree: Path
    base_sha: str
    initial_tree: str
    initial_refinement: AgenticTuningResult | None
    cycles: tuple[CycleResult, ...]

    @property
    def thread_id(self) -> str | None:
        """Return the most recent editing thread identifier."""
        thread_id = self.cycles[-1].thread_id if self.cycles else None
        return thread_id

    @property
    def baseline_refinement(self) -> AgenticTuningResult | None:
        """Return the initial tuning result for compatibility with run consumers."""
        return self.initial_refinement

    @property
    def attempts(self) -> tuple[AttemptResult, ...]:
        """Return every attempt in execution order across all cycles."""
        attempts = tuple(attempt for cycle in self.cycles for attempt in cycle.attempts)
        return attempts

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "run_directory": str(self.run_directory),
            "worktree": str(self.worktree),
            "base_sha": self.base_sha,
            "initial_tree": self.initial_tree,
            "thread_id": self.thread_id,
            "initial_refinement": self.initial_refinement.as_dict() if self.initial_refinement is not None else None,
            "accepted_cycles": sum(cycle.passed for cycle in self.cycles),
            "cycles": [cycle.as_dict() for cycle in self.cycles],
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }
