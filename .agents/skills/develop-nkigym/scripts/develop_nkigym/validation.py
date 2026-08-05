"""Structural validation for durable develop-nkigym workflow state."""

from __future__ import annotations

import math
from pathlib import Path

from develop_nkigym.types import CycleState, GateSpec, RunConfig, RunRecord

AGENTIC_TUNING_GATE_NAME = "agentic-tuning"


def baseline_gates(gates: tuple[GateSpec, ...]) -> tuple[GateSpec, ...]:
    """Return canonical gates that validate implementation health before tuning."""
    baseline = tuple(gate for gate in gates if gate.name != AGENTIC_TUNING_GATE_NAME)
    return baseline


def validate_config(config: RunConfig, repository: Path) -> None:
    """Reject unsafe or incomplete run configuration."""
    artifact_root = config.artifact_root.expanduser().resolve()
    gate_names = tuple(gate.name for gate in config.gates)
    if artifact_root == repository or artifact_root.is_relative_to(repository):
        raise ValueError("artifact_root must be outside the source repository")
    if not config.program.name.isidentifier():
        raise ValueError(f"program name must be a Python identifier: {config.program.name!r}")
    if not config.program.nkigym_source.strip():
        raise ValueError("program source must not be empty")
    if not config.program.input_specs:
        raise ValueError("program input_specs must not be empty")
    if not config.agentic_tuning.argv:
        raise ValueError("agentic tuning command must not be empty")
    if config.agentic_tuning.timeout_seconds < 1:
        raise ValueError("agentic tuning timeout must be positive")
    if not config.agentic_tuning.required_artifacts:
        raise ValueError("agentic tuning must require artifacts")
    if not config.gates:
        raise ValueError("at least one candidate gate is required")
    if (
        isinstance(config.improvement_round_limit, bool)
        or not isinstance(config.improvement_round_limit, int)
        or config.improvement_round_limit < 1
    ):
        raise ValueError("improvement_round_limit must be a positive integer")
    if config.initial_historical_best_score is not None and (
        isinstance(config.initial_historical_best_score, bool)
        or not math.isfinite(config.initial_historical_best_score)
        or config.initial_historical_best_score < 0.0
        or config.initial_historical_best_score > 100.0
    ):
        raise ValueError("initial_historical_best_score must be a finite percentage in [0, 100] or None")
    if len(set(gate_names)) != len(gate_names):
        raise ValueError("candidate gate names must be unique")
    if gate_names.count(AGENTIC_TUNING_GATE_NAME) != 1:
        raise ValueError("candidate gates must contain exactly one agentic-tuning gate")
    if not baseline_gates(config.gates):
        raise ValueError("candidate gates must contain at least one baseline validation gate")
    for gate in config.gates:
        if not gate.argv or gate.timeout_seconds < 1:
            raise ValueError(f"gate {gate.name!r} must have a command and positive timeout")


def _validate_baseline_checks(cycle: CycleState, gate_names: tuple[str, ...]) -> None:
    """Validate persisted baseline-check ordering and verdicts."""
    indices = tuple(attempt.index for attempt in cycle.baseline_checks)
    if tuple(sorted(indices)) != indices:
        raise ValueError(f"cycle {cycle.index} baseline check indices are not ordered")
    if len(set(indices)) != len(indices):
        raise ValueError(f"cycle {cycle.index} baseline check indices are not unique")
    for attempt in cycle.baseline_checks:
        observed_gate_names = tuple(gate.name for gate in attempt.gates)
        if attempt.baseline_tree != cycle.baseline_tree:
            raise ValueError(f"cycle {cycle.index} baseline check targets a different tree")
        if observed_gate_names != gate_names[: len(observed_gate_names)]:
            raise ValueError(f"cycle {cycle.index} baseline check {attempt.index} gates are out of order")
        expected_passed = (
            len(attempt.gates) == len(gate_names)
            and all(gate.passed for gate in attempt.gates)
            and not attempt.worktree_modified
        )
        if attempt.passed != expected_passed:
            raise ValueError(f"cycle {cycle.index} baseline check {attempt.index} has an inconsistent verdict")


def _validate_cycle(record: RunRecord, cycle: CycleState, expected_index: int, gate_names: tuple[str, ...]) -> None:
    """Validate one cycle and its immutable evidence."""
    baseline_gate_names = tuple(gate.name for gate in baseline_gates(record.gates))
    if cycle.index != expected_index:
        raise ValueError("cycle indices must be contiguous")
    _validate_baseline_checks(cycle, baseline_gate_names)
    if cycle.baseline_tuning is not None and (
        not cycle.baseline_tuning.passed or cycle.baseline_tuning.best_score is None
    ):
        raise ValueError(f"cycle {cycle.index} baseline tuning is not valid measured evidence")
    if tuple(sorted(check.index for check in cycle.checks)) != tuple(check.index for check in cycle.checks):
        raise ValueError(f"cycle {cycle.index} check indices are not ordered")
    if len({check.index for check in cycle.checks}) != len(cycle.checks):
        raise ValueError(f"cycle {cycle.index} check indices are not unique")
    for check in cycle.checks:
        observed_gate_names = tuple(gate.name for gate in check.gates)
        if observed_gate_names != gate_names[: len(observed_gate_names)]:
            raise ValueError(f"cycle {cycle.index} check {check.index} gates are out of order")
        expected_passed = (
            len(check.gates) == len(record.gates)
            and all(gate.passed for gate in check.gates)
            and not check.worktree_modified
        )
        if check.passed != expected_passed:
            raise ValueError(f"cycle {cycle.index} check {check.index} has an inconsistent verdict")


def _validate_cycle_chain(record: RunRecord) -> None:
    """Validate promotion links between consecutive cycles."""
    for expected_index, cycle in enumerate(record.cycles):
        if cycle.accepted_tree is None and expected_index != len(record.cycles) - 1:
            raise ValueError("only the current cycle may be unaccepted")
        if cycle.accepted_tree is not None:
            if expected_index == len(record.cycles) - 1:
                raise ValueError("an accepted cycle must be followed by its promoted baseline")
            next_cycle = record.cycles[expected_index + 1]
            if next_cycle.baseline_tree != cycle.accepted_tree:
                raise ValueError("promoted cycle tree does not match the next baseline")
            if not any(check.passed and check.candidate_tree == cycle.accepted_tree for check in cycle.checks):
                raise ValueError(f"cycle {cycle.index} accepted tree has no matching passing check")


def validate_record(record: RunRecord, run_directory: Path) -> None:
    """Check structural invariants needed for deterministic resume."""
    if record.run_directory.expanduser().resolve() != run_directory:
        raise ValueError(f"run record belongs to a different directory: {record.run_directory}")
    if record.worktree.expanduser().resolve().parent != run_directory:
        raise ValueError("run worktree must be directly below the run directory")
    if record.program_directory.expanduser().resolve().parent != run_directory:
        raise ValueError("run program directory must be directly below the run directory")
    if not record.worktree.is_dir():
        raise ValueError(f"run worktree does not exist: {record.worktree}")
    if not record.program_directory.is_dir():
        raise ValueError(f"run program directory does not exist: {record.program_directory}")
    gate_names = tuple(gate.name for gate in record.gates)
    if len(set(gate_names)) != len(gate_names) or gate_names.count(AGENTIC_TUNING_GATE_NAME) != 1:
        raise ValueError("run record contains invalid candidate gates")
    if not baseline_gates(record.gates):
        raise ValueError("run record contains no baseline validation gates")
    if not record.agentic_tuning.argv or record.agentic_tuning.timeout_seconds < 1:
        raise ValueError("run record contains invalid agentic tuning configuration")
    if any(not gate.argv or gate.timeout_seconds < 1 for gate in record.gates):
        raise ValueError("run record contains an invalid candidate gate")
    if (
        isinstance(record.improvement_round_limit, bool)
        or not isinstance(record.improvement_round_limit, int)
        or record.improvement_round_limit < 1
    ):
        raise ValueError("run record contains an invalid improvement round limit")
    if record.initial_historical_best_score is not None and (
        isinstance(record.initial_historical_best_score, bool)
        or not math.isfinite(record.initial_historical_best_score)
        or record.initial_historical_best_score < 0.0
        or record.initial_historical_best_score > 100.0
    ):
        raise ValueError("run record contains an invalid initial historical best score")
    if record.cycles[0].baseline_tree != record.initial_tree:
        raise ValueError("first cycle does not start at the initial tree")
    for expected_index, cycle in enumerate(record.cycles):
        _validate_cycle(record, cycle, expected_index, gate_names)
    _validate_cycle_chain(record)


__all__ = ["AGENTIC_TUNING_GATE_NAME", "baseline_gates", "validate_config", "validate_record"]
