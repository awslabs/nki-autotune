"""Durable deterministic support for one Codex-directed nkigym development run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nkigym.search.agentic_tuning import (
    AgenticTuningContext,
    AgenticTuningResult,
    agentic_target_score,
    run_agentic_tuning,
)
from nkigym.search.program import write_program
from self_evolve.gates import run_gates
from self_evolve.git import (
    CandidateSnapshot,
    create_candidate_tree,
    resolve_repository,
    resolve_revision,
    snapshot_candidate,
)
from self_evolve.types import (
    BaselineCheckAttempt,
    CheckAttempt,
    CycleState,
    NextAction,
    RunConfig,
    RunRecord,
    RunStatus,
    WorkflowMode,
)
from self_evolve.validation import AGENTIC_TUNING_GATE_NAME, baseline_gates, validate_config, validate_record

_RUN_RECORD_FILENAME = "run.json"
_RUN_LOCK_FILENAME = "run.lock"


def _new_run_id() -> str:
    """Return a sortable run identifier with collision resistance."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically and durably replace one formatted JSON object."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_record(run_directory: Path) -> RunRecord:
    """Read and validate one canonical run record."""
    resolved_directory = run_directory.expanduser().resolve()
    record_path = resolved_directory / _RUN_RECORD_FILENAME
    decoded = json.loads(record_path.read_text(encoding="utf-8"))
    record = RunRecord.from_dict(decoded)
    validate_record(record, resolved_directory)
    return record


def _write_record(record: RunRecord) -> None:
    """Atomically persist one run record."""
    _write_json(record.run_directory / _RUN_RECORD_FILENAME, record.as_dict())


@contextmanager
def _exclusive_run_lock(run_directory: Path) -> Iterator[None]:
    """Reject overlapping mutating commands for one run."""
    lock_path = run_directory.expanduser().resolve() / _RUN_LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another workflow command is already running for {run_directory}") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _candidate_fingerprint(snapshot: CandidateSnapshot) -> str:
    """Hash one exact baseline-to-candidate binary patch."""
    digest = hashlib.sha256(snapshot.patch.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _historical_best_score(record: RunRecord) -> float | None:
    """Return the best score from accepted baseline evidence."""
    scores = [
        cycle.baseline_tuning.best_score
        for cycle in record.cycles
        if cycle.baseline_tuning is not None and cycle.baseline_tuning.best_score is not None
    ]
    if record.initial_historical_best_score is not None:
        scores.append(record.initial_historical_best_score)
    best = max(scores) if scores else None
    return best


def _next_attempt_directory(parent: Path) -> tuple[int, Path]:
    """Reserve a new numbered immutable attempt directory."""
    parent.mkdir(parents=True, exist_ok=True)
    indices = [int(path.name) for path in parent.iterdir() if path.is_dir() and path.name.isdecimal()]
    index = max(indices, default=-1) + 1
    directory = parent / f"{index:03d}"
    directory.mkdir()
    return index, directory


def _latest_tuning_result(cycle: CycleState) -> AgenticTuningResult | None:
    """Return the most recently executed or inherited tuning evidence."""
    result = cycle.tuning_attempts[-1] if cycle.tuning_attempts else cycle.baseline_tuning
    return result


def _cycle_has_started(cycle: CycleState, snapshot: CandidateSnapshot) -> bool:
    """Return whether a legacy cycle already produced evidence or edits."""
    started = bool(cycle.baseline_tuning or cycle.tuning_attempts or cycle.checks or snapshot.changed_files)
    return started


def _candidate_next_action(cycle: CycleState, snapshot: CandidateSnapshot, fingerprint: str) -> NextAction:
    """Derive the next action once a cycle is ready for implementation edits."""
    if not snapshot.changed_files:
        action: NextAction = "edit"
    elif cycle.checks and cycle.checks[-1].candidate_fingerprint == fingerprint:
        action = "accept" if cycle.checks[-1].passed else "edit"
    else:
        action = "check"
    return action


def _next_action(cycle: CycleState, snapshot: CandidateSnapshot, fingerprint: str) -> NextAction:
    """Derive the only valid next command from artifacts and candidate content."""
    latest_baseline_check = cycle.baseline_checks[-1] if cycle.baseline_checks else None
    if latest_baseline_check is None and not _cycle_has_started(cycle, snapshot):
        action: NextAction = "validate"
    elif latest_baseline_check is not None and not latest_baseline_check.passed:
        action = _candidate_next_action(cycle, snapshot, fingerprint)
    elif cycle.baseline_tuning is None and not cycle.tuning_attempts and not snapshot.changed_files:
        action = "tune"
    else:
        action = _candidate_next_action(cycle, snapshot, fingerprint)
    return action


def _workflow_mode(cycle: CycleState) -> WorkflowMode:
    """Return whether the active cycle repairs bugs or makes one improvement."""
    latest_baseline_check = cycle.baseline_checks[-1] if cycle.baseline_checks else None
    legacy_candidate_failure = latest_baseline_check is None and any(not check.passed for check in cycle.checks)
    failed_tuning = cycle.baseline_tuning is None and bool(cycle.tuning_attempts)
    if (latest_baseline_check is not None and not latest_baseline_check.passed) or legacy_candidate_failure:
        mode: WorkflowMode = "repair"
    elif failed_tuning:
        mode = "repair"
    else:
        mode = "improve"
    return mode


def _completed_improvement_rounds(record: RunRecord) -> int:
    """Count accepted ordinary improvement cycles, excluding repair cycles."""
    completed = sum(cycle.accepted_tree is not None and _workflow_mode(cycle) == "improve" for cycle in record.cycles)
    return completed


def _status(record: RunRecord) -> RunStatus:
    """Derive a public status without persisting conversational state."""
    cycle = record.current_cycle
    snapshot = snapshot_candidate(record.worktree, cycle.baseline_tree)
    fingerprint = _candidate_fingerprint(snapshot)
    latest_tuning = _latest_tuning_result(cycle)
    latest_baseline_check = cycle.baseline_checks[-1] if cycle.baseline_checks else None
    latest_check = cycle.checks[-1] if cycle.checks else None
    failed_gate = None
    matching_check = latest_check is not None and latest_check.candidate_fingerprint == fingerprint
    if matching_check and latest_check is not None:
        failed_gate = next((gate for gate in latest_check.gates if not gate.passed), None)
    elif latest_baseline_check is not None and not latest_baseline_check.passed:
        failed_gate = next((gate for gate in latest_baseline_check.gates if not gate.passed), None)
    completed_improvement_rounds = _completed_improvement_rounds(record)
    if completed_improvement_rounds >= record.improvement_round_limit:
        next_action: NextAction = "complete"
    else:
        next_action = _next_action(cycle, snapshot, fingerprint)
    status = RunStatus(
        run_id=record.run_id,
        run_directory=record.run_directory,
        worktree=record.worktree,
        program_directory=record.program_directory,
        cycle_index=cycle.index,
        baseline_tree=cycle.baseline_tree,
        candidate_fingerprint=fingerprint,
        mode=_workflow_mode(cycle),
        improvement_round_limit=record.improvement_round_limit,
        completed_improvement_rounds=completed_improvement_rounds,
        historical_best_score=_historical_best_score(record),
        latest_baseline_check_directory=(
            None if latest_baseline_check is None else latest_baseline_check.artifact_directory
        ),
        baseline_tuning_artifact_directory=(
            None if cycle.baseline_tuning is None else cycle.baseline_tuning.artifact_directory
        ),
        latest_tuning_log=None if latest_tuning is None else latest_tuning.log_path,
        latest_check_directory=None if latest_check is None else latest_check.artifact_directory,
        failed_gate_log=None if failed_gate is None else failed_gate.log_path,
        latest_check_worktree_modified=False if latest_check is None else latest_check.worktree_modified,
        changed_files=snapshot.changed_files,
        next_action=next_action,
    )
    return status


def _replace_current_cycle(record: RunRecord, cycle: CycleState) -> RunRecord:
    """Return a record with its active cycle replaced."""
    cycles = (*record.cycles[:-1], cycle)
    return replace(record, cycles=cycles)


def _passing_tuning_result(check: CheckAttempt) -> AgenticTuningResult:
    """Load the tuning result certified by a passing candidate check."""
    matching = tuple(gate for gate in check.gates if gate.name == AGENTIC_TUNING_GATE_NAME)
    if len(matching) != 1 or not matching[0].passed:
        raise RuntimeError("passing check does not contain one passing agentic-tuning gate")
    result_path = matching[0].artifact_directory / "result.json"
    try:
        decoded = json.loads(result_path.read_text(encoding="utf-8"))
        result = AgenticTuningResult.from_dict(decoded)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"passing agentic-tuning gate has invalid evidence: {result_path}") from error
    if not result.passed or result.best_score is None:
        raise RuntimeError(f"passing agentic-tuning gate has unusable measured evidence: {result_path}")
    return result


def create_run(config: RunConfig) -> RunStatus:
    """Snapshot the source workspace and initialize one resumable run."""
    repository = resolve_repository(config.repository)
    validate_config(config, repository)
    base_sha = resolve_revision(repository, config.base_revision)
    initial_tree = create_candidate_tree(repository, base_sha)
    run_id = _new_run_id()
    run_directory = config.artifact_root.expanduser().resolve() / run_id
    run_directory.mkdir(parents=True)
    program_directory = run_directory / "program"
    write_program(config.program, program_directory)
    cycle = CycleState(
        index=0,
        baseline_tree=initial_tree,
        baseline_checks=(),
        baseline_tuning=None,
        tuning_attempts=(),
        checks=(),
        accepted_tree=None,
    )
    record = RunRecord(
        run_id=run_id,
        source_repository=repository,
        run_directory=run_directory,
        worktree=repository,
        base_sha=base_sha,
        initial_tree=initial_tree,
        program_directory=program_directory,
        agentic_tuning=config.agentic_tuning,
        gates=config.gates,
        improvement_round_limit=config.improvement_round_limit,
        initial_historical_best_score=config.initial_historical_best_score,
        cycles=(cycle,),
    )
    _write_record(record)
    return _status(record)


def status_run(run_directory: Path) -> RunStatus:
    """Return the action implied by one durable run and its source checkout."""
    record = _read_record(run_directory)
    return _status(record)


def validate_run(run_directory: Path) -> RunStatus:
    """Run canonical health gates against the unchanged active baseline."""
    resolved_directory = run_directory.expanduser().resolve()
    with _exclusive_run_lock(resolved_directory):
        record = _read_record(resolved_directory)
        status = _status(record)
        cycle = record.current_cycle
        retrying = (
            status.next_action == "edit"
            and not status.changed_files
            and bool(cycle.baseline_checks)
            and not cycle.baseline_checks[-1].passed
        )
        if status.next_action != "validate" and not retrying:
            raise RuntimeError(f"run requires {status.next_action}, not validate")
        if status.changed_files:
            raise RuntimeError("restore the cycle baseline before running baseline validation")
        attempt_index, attempt_directory = _next_attempt_directory(
            record.run_directory / "cycles" / f"{cycle.index:03d}" / "baseline-checks"
        )
        gates = run_gates(baseline_gates(record.gates), record.worktree, attempt_directory / "gates")
        final_snapshot = snapshot_candidate(record.worktree, cycle.baseline_tree)
        final_tree = create_candidate_tree(record.worktree, cycle.baseline_tree)
        worktree_modified = bool(final_snapshot.changed_files) or final_tree != cycle.baseline_tree
        expected_gate_count = len(baseline_gates(record.gates))
        passed = len(gates) == expected_gate_count and all(gate.passed for gate in gates) and not worktree_modified
        attempt = BaselineCheckAttempt(
            index=attempt_index,
            artifact_directory=attempt_directory,
            baseline_tree=cycle.baseline_tree,
            gates=gates,
            worktree_modified=worktree_modified,
            passed=passed,
        )
        _write_json(attempt_directory / "result.json", attempt.as_dict())
        updated_cycle = replace(cycle, baseline_checks=(*cycle.baseline_checks, attempt))
        updated_record = _replace_current_cycle(record, updated_cycle)
        _write_record(updated_record)
        status = _status(updated_record)
    return status


def tune_run(run_directory: Path) -> RunStatus:
    """Produce baseline tuning evidence for the active cycle."""
    resolved_directory = run_directory.expanduser().resolve()
    with _exclusive_run_lock(resolved_directory):
        record = _read_record(resolved_directory)
        status = _status(record)
        cycle = record.current_cycle
        retrying = (
            status.next_action == "edit"
            and not status.changed_files
            and cycle.baseline_tuning is None
            and bool(cycle.tuning_attempts)
        )
        if status.next_action != "tune" and not retrying:
            raise RuntimeError(f"run requires {status.next_action}, not tune")
        if status.changed_files:
            raise RuntimeError("restore the cycle baseline before running baseline tuning")
        _, output_directory = _next_attempt_directory(record.run_directory / "cycles" / f"{cycle.index:03d}" / "tuning")
        tuning = replace(record.agentic_tuning, target_score=agentic_target_score(_historical_best_score(record)))
        result = run_agentic_tuning(
            spec=tuning,
            program_directory=record.program_directory,
            worktree=record.worktree,
            output_directory=output_directory,
            source_fingerprint=lambda: snapshot_candidate(record.worktree, cycle.baseline_tree).patch,
        )
        _write_json(output_directory / "result.json", result.as_dict())
        baseline_tuning = result if result.passed and result.best_score is not None else None
        updated_cycle = replace(
            cycle, baseline_tuning=baseline_tuning, tuning_attempts=(*cycle.tuning_attempts, result)
        )
        updated_record = _replace_current_cycle(record, updated_cycle)
        _write_record(updated_record)
        status = _status(updated_record)
    return status


def check_run(run_directory: Path) -> RunStatus:
    """Run canonical gates against one exact edited candidate."""
    resolved_directory = run_directory.expanduser().resolve()
    with _exclusive_run_lock(resolved_directory):
        record = _read_record(resolved_directory)
        status = _status(record)
        if status.next_action != "check":
            raise RuntimeError(f"run requires {status.next_action}, not check")
        cycle = record.current_cycle
        snapshot = snapshot_candidate(record.worktree, cycle.baseline_tree)
        if not snapshot.changed_files:
            raise RuntimeError("candidate must differ from the cycle baseline")
        fingerprint = _candidate_fingerprint(snapshot)
        candidate_tree = create_candidate_tree(record.worktree, cycle.baseline_tree)
        attempt_index, attempt_directory = _next_attempt_directory(
            record.run_directory / "cycles" / f"{cycle.index:03d}" / "checks"
        )
        patch_path = attempt_directory / "diff.patch"
        patch_path.write_text(snapshot.patch, encoding="utf-8")
        _write_json(
            attempt_directory / "candidate.json",
            {
                "candidate_tree": candidate_tree,
                "candidate_fingerprint": fingerprint,
                "changed_files": list(snapshot.changed_files),
            },
        )
        context = AgenticTuningContext(
            program_directory=record.program_directory.resolve(),
            baseline_tree=cycle.baseline_tree,
            tuning=record.agentic_tuning,
            historical_best_score=_historical_best_score(record),
        )
        gate_directory = attempt_directory / "gates"
        context_path = gate_directory / "agentic-tuning-artifacts" / "agentic-tuning-context.json"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(context_path, context.as_dict())
        gates = run_gates(record.gates, record.worktree, gate_directory)
        final_snapshot = snapshot_candidate(record.worktree, cycle.baseline_tree)
        final_tree = create_candidate_tree(record.worktree, cycle.baseline_tree)
        worktree_modified = _candidate_fingerprint(final_snapshot) != fingerprint or final_tree != candidate_tree
        passed = len(gates) == len(record.gates) and all(gate.passed for gate in gates) and not worktree_modified
        check = CheckAttempt(
            index=attempt_index,
            artifact_directory=attempt_directory,
            candidate_tree=candidate_tree,
            candidate_fingerprint=fingerprint,
            changed_files=snapshot.changed_files,
            patch_path=patch_path,
            gates=gates,
            worktree_modified=worktree_modified,
            passed=passed,
        )
        _write_json(attempt_directory / "result.json", check.as_dict())
        updated_cycle = replace(cycle, checks=(*cycle.checks, check))
        updated_record = _replace_current_cycle(record, updated_cycle)
        _write_record(updated_record)
        status = _status(updated_record)
    return status


def accept_run(run_directory: Path) -> RunStatus:
    """Promote the exactly checked candidate and open the next cycle."""
    resolved_directory = run_directory.expanduser().resolve()
    with _exclusive_run_lock(resolved_directory):
        record = _read_record(resolved_directory)
        status = _status(record)
        if status.next_action != "accept":
            raise RuntimeError(f"run requires {status.next_action}, not accept")
        cycle = record.current_cycle
        check = cycle.checks[-1]
        snapshot = snapshot_candidate(record.worktree, cycle.baseline_tree)
        candidate_tree = create_candidate_tree(record.worktree, cycle.baseline_tree)
        if _candidate_fingerprint(snapshot) != check.candidate_fingerprint:
            raise RuntimeError("candidate content changed after its passing check")
        if candidate_tree != check.candidate_tree:
            raise RuntimeError("candidate Git tree changed after its passing check")
        tuning = _passing_tuning_result(check)
        baseline_check = BaselineCheckAttempt(
            index=0,
            artifact_directory=check.artifact_directory,
            baseline_tree=candidate_tree,
            gates=tuple(gate for gate in check.gates if gate.name != AGENTIC_TUNING_GATE_NAME),
            worktree_modified=False,
            passed=True,
        )
        accepted_cycle = replace(cycle, accepted_tree=candidate_tree)
        next_cycle = CycleState(
            index=cycle.index + 1,
            baseline_tree=candidate_tree,
            baseline_checks=(baseline_check,),
            baseline_tuning=tuning,
            tuning_attempts=(),
            checks=(),
            accepted_tree=None,
        )
        updated_record = replace(record, cycles=(*record.cycles[:-1], accepted_cycle, next_cycle))
        _write_record(updated_record)
        status = _status(updated_record)
    return status


__all__ = ["accept_run", "check_run", "create_run", "status_run", "tune_run", "validate_run"]
