"""Artifact-driven controller for continuous IR development cycles."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from developer.codex import CodexRunner
from developer.gates import candidate_environment, run_gates
from developer.git import (
    create_candidate_tree,
    create_detached_worktree,
    resolve_repository,
    resolve_revision,
    set_worktree_baseline,
    snapshot_candidate,
)
from developer.prompts import initial_prompt, retry_prompt
from developer.types import AttemptResult, CycleResult, GateResult, RunConfig, Verdict, WorkflowResult
from nkigym.search.agentic_tuning import (
    AGENTIC_TUNING_CONTEXT_ENV,
    AgenticTuningContext,
    AgenticTuningResult,
    run_agentic_tuning,
)
from nkigym.search.program import write_program

_SCHEMA_VERSION = 14
_LOGGER = logging.getLogger(__name__)
_RELEVANT_IMPLEMENTATION_PREFIXES = (
    "nkigym/src/nkigym/codegen/",
    "nkigym/src/nkigym/environment/",
    "nkigym/src/nkigym/ir/",
    "nkigym/src/nkigym/ops/",
    "nkigym/src/nkigym/transforms/",
)


def _score_text(score: float | None) -> str:
    """Format one optional MFU score for terminal progress."""
    text = "unavailable" if score is None else f"{score:.2f}%"
    return text


def _utc_timestamp() -> str:
    """Return a stable UTC timestamp for artifacts."""
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    """Return a sortable run identifier with collision resistance."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically write a formatted JSON artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_passing_agentic_tuning_result(attempt: AttemptResult) -> AgenticTuningResult:
    """Load the tuning evidence accepted by the final pytest gate."""
    matching_gates = tuple(gate for gate in attempt.gates if gate.name == "agentic-tuning")
    if len(matching_gates) != 1 or not matching_gates[0].passed:
        raise RuntimeError("accepted attempt does not contain one passing agentic-tuning gate")
    result_path = matching_gates[0].artifact_directory / "result.json"
    try:
        decoded = json.loads(result_path.read_text(encoding="utf-8"))
        result = AgenticTuningResult.from_dict(decoded)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"passing agentic-tuning gate has invalid result artifact: {result_path}") from error
    return result


def _validate_config(config: RunConfig, repository: Path) -> None:
    """Reject unsafe or incomplete workflow configuration."""
    goal = config.goal.strip()
    artifact_root = config.artifact_root.expanduser().resolve()
    if not goal:
        raise ValueError("goal must not be empty")
    if config.max_cycles is not None and config.max_cycles < 1:
        raise ValueError("max_cycles must be at least 1 when configured")
    if config.max_thread_start_attempts < 1:
        raise ValueError("max_thread_start_attempts must be at least 1")
    if config.agent_timeout_seconds < 1:
        raise ValueError("agent_timeout_seconds must be at least 1")
    if not config.program.name.isidentifier():
        raise ValueError(f"program name must be a Python identifier: {config.program.name!r}")
    if not config.program.nkigym_source.strip():
        raise ValueError("program source must not be empty")
    if not config.program.input_specs:
        raise ValueError("program input_specs must not be empty")
    if not config.agentic_tuning.argv:
        raise ValueError("agentic tuning command is not configured")
    if config.agentic_tuning.timeout_seconds < 1:
        raise ValueError("agentic tuning timeout_seconds must be at least 1")
    if not config.agentic_tuning.required_artifacts:
        raise ValueError("agentic tuning must declare at least one required artifact")
    if not config.gates:
        raise ValueError(f"program {config.program.name!r} has no evaluation gates")
    if artifact_root == repository or artifact_root.is_relative_to(repository):
        raise ValueError("artifact_root must be outside the source repository")


def _relevant_implementation_files(changed_files: tuple[str, ...]) -> tuple[str, ...]:
    """Return candidate Python changes in the IR execution stack."""
    relevant = tuple(
        path
        for path in changed_files
        if path.endswith(".py") and any(path.startswith(prefix) for prefix in _RELEVANT_IMPLEMENTATION_PREFIXES)
    )
    return relevant


def _all_gates_passed(gates: tuple[GateResult, ...], expected_count: int) -> bool:
    """Require every configured deterministic gate to run and pass."""
    passed = len(gates) == expected_count and all(gate.passed for gate in gates)
    return passed


def _attempt_passed(gates: tuple[GateResult, ...], expected_gate_count: int) -> bool:
    """Accept a candidate exactly when every configured pytest gate passes."""
    passed = _all_gates_passed(gates, expected_gate_count)
    return passed


def _latest_thread_id(cycles: tuple[CycleResult, ...]) -> str | None:
    """Return the most recent cycle's editing thread identifier."""
    thread_id = cycles[-1].thread_id if cycles else None
    return thread_id


def _run_record(
    config: RunConfig,
    run_id: str,
    state: str,
    repository: Path,
    base_sha: str,
    initial_tree: str,
    worktree: Path,
    cycles: tuple[CycleResult, ...],
    initial_refinement: AgenticTuningResult | None,
) -> dict[str, object]:
    """Build the durable top-level run record."""
    record = {
        "schema_version": _SCHEMA_VERSION,
        "updated_at": _utc_timestamp(),
        "run_id": run_id,
        "state": state,
        "source_repository": str(repository),
        "base_sha": base_sha,
        "initial_tree": initial_tree,
        "worktree": str(worktree),
        "goal": config.goal,
        "program": config.program.as_dict(),
        "agentic_tuning": config.agentic_tuning.as_dict(),
        "gates": [gate.as_dict() for gate in config.gates],
        "controller": {
            "codex_executable": config.codex_executable,
            "base_revision": config.base_revision,
            "max_cycles": config.max_cycles,
            "continuous": config.max_cycles is None,
            "max_thread_start_attempts": config.max_thread_start_attempts,
            "agent_timeout_seconds": config.agent_timeout_seconds,
        },
        "thread_id": _latest_thread_id(cycles),
        "initial_refinement": initial_refinement.as_dict() if initial_refinement is not None else None,
        "accepted_cycles": sum(cycle.passed for cycle in cycles),
        "cycles": [cycle.as_dict() for cycle in cycles],
    }
    return record


def _reconcile_thread_id(current: str | None, observed: str | None) -> str | None:
    """Keep one stable Codex thread identifier within an improvement cycle."""
    thread_id = current
    if current is None:
        thread_id = observed
    elif observed is not None and observed != current:
        raise RuntimeError(f"resumed Codex session changed thread ID from {current} to {observed}")
    return thread_id


def _run_attempt(
    config: RunConfig,
    index: int,
    prompt: str,
    runner: CodexRunner,
    thread_id: str | None,
    worktree: Path,
    baseline_tree: str,
    program_directory: Path,
    cycle_directory: Path,
    historical_best_score: float | None,
) -> tuple[AttemptResult, str | None]:
    """Run one editing turn and every configured pytest gate."""
    attempt_directory = cycle_directory / "attempts" / f"{index:03d}"
    attempt_directory.mkdir(parents=True)
    (attempt_directory / "prompt.md").write_text(prompt, encoding="utf-8")
    codex = runner.run(worktree, prompt, attempt_directory, thread_id)
    next_thread_id = _reconcile_thread_id(thread_id, codex.thread_id)
    snapshot = snapshot_candidate(worktree, baseline_tree)
    relevant_files = _relevant_implementation_files(snapshot.changed_files)
    patch_path = attempt_directory / "diff.patch"
    patch_path.write_text(snapshot.patch, encoding="utf-8")
    _write_json(
        attempt_directory / "changed-files.json",
        {"files": list(snapshot.changed_files), "relevant_implementation_files": list(relevant_files)},
    )
    context_path = attempt_directory / "agentic-tuning-context.json"
    context = AgenticTuningContext(
        program_directory=program_directory.resolve(),
        baseline_tree=baseline_tree,
        tuning=config.agentic_tuning,
        historical_best_score=historical_best_score,
    )
    _write_json(context_path, context.as_dict())

    gate_results: tuple[GateResult, ...] = ()
    expected_gate_count = len(config.gates)
    if codex.passed and relevant_files:
        gate_directory = attempt_directory / "gates"
        gate_results = run_gates(
            config.gates,
            worktree,
            gate_directory,
            {"agentic-tuning": ((AGENTIC_TUNING_CONTEXT_ENV, str(context_path.resolve())),)},
        )
    passed = _attempt_passed(gate_results, expected_gate_count)
    attempt = AttemptResult(
        index=index,
        codex=codex,
        changed_files=snapshot.changed_files,
        patch_path=patch_path,
        gates=gate_results,
        passed=passed,
    )
    _write_json(attempt_directory / "result.json", attempt.as_dict())
    return attempt, next_thread_id


def _checkpoint_candidate(worktree: Path, baseline_tree: str) -> str:
    """Create and verify an internal tree for the next improvement cycle."""
    accepted_tree = create_candidate_tree(worktree, baseline_tree)
    set_worktree_baseline(worktree, accepted_tree)
    return accepted_tree


def _cycle_result(
    index: int,
    baseline_tree: str,
    baseline_refinement: AgenticTuningResult,
    historical_best_score: float | None,
    thread_id: str | None,
    attempts: tuple[AttemptResult, ...],
    accepted_tree: str | None,
    failure_reason: str | None,
) -> CycleResult:
    """Build one durable cycle result."""
    cycle = CycleResult(
        index=index,
        baseline_tree=baseline_tree,
        accepted_tree=accepted_tree,
        thread_id=thread_id,
        baseline_refinement=baseline_refinement,
        historical_best_score=historical_best_score,
        attempts=attempts,
        failure_reason=failure_reason,
    )
    return cycle


def _write_running_record(
    config: RunConfig,
    run_id: str,
    repository: Path,
    base_sha: str,
    initial_tree: str,
    worktree: Path,
    run_directory: Path,
    cycles: tuple[CycleResult, ...],
    initial_refinement: AgenticTuningResult,
) -> None:
    """Persist the current continuous workflow state."""
    if cycles and cycles[-1].failure_reason is not None:
        state = "failed"
    else:
        state = "running" if cycles and cycles[-1].passed else "improving"
    record = _run_record(
        config, run_id, state, repository, base_sha, initial_tree, worktree, cycles, initial_refinement
    )
    _write_json(run_directory / "run.json", record)


def _develop_cycles(
    config: RunConfig,
    run_id: str,
    repository: Path,
    base_sha: str,
    initial_tree: str,
    worktree: Path,
    program_directory: Path,
    run_directory: Path,
    initial_refinement: AgenticTuningResult,
) -> tuple[tuple[CycleResult, ...], Verdict]:
    """Retry each candidate until accepted and continuously start new cycles."""
    runner = CodexRunner(config.codex_executable, config.agent_timeout_seconds)
    baseline = initial_refinement
    historical_best_score = initial_refinement.best_score
    baseline_tree = initial_tree
    cycles: list[CycleResult] = []
    verdict: Verdict = "passed"
    try:
        while verdict == "passed" and (config.max_cycles is None or len(cycles) < config.max_cycles):
            cycle_index = len(cycles)
            cycle_directory = run_directory / "cycles" / f"{cycle_index:03d}"
            cycle_directory.mkdir(parents=True)
            _LOGGER.info(
                "cycle %03d | started | latest_mfu=%s | historical_best_mfu=%s | tree=%s",
                cycle_index,
                _score_text(baseline.best_score),
                _score_text(historical_best_score),
                baseline_tree[:12],
            )
            cycle_prompt = initial_prompt(config, baseline, historical_best_score, program_directory)
            prompt = cycle_prompt
            (cycle_directory / "prompt.md").write_text(cycle_prompt, encoding="utf-8")
            attempts: list[AttemptResult] = []
            thread_id: str | None = None
            thread_start_attempts = 0
            accepted_refinement: AgenticTuningResult | None = None
            cycle = _cycle_result(
                cycle_index, baseline_tree, baseline, historical_best_score, thread_id, (), None, None
            )
            cycles.append(cycle)
            _write_running_record(
                config,
                run_id,
                repository,
                base_sha,
                initial_tree,
                worktree,
                run_directory,
                tuple(cycles),
                initial_refinement,
            )

            while not cycle.passed and cycle.failure_reason is None:
                attempt_index = len(attempts)
                _LOGGER.info("cycle %03d attempt %03d | started", cycle_index, attempt_index)
                attempt, thread_id = _run_attempt(
                    config,
                    attempt_index,
                    prompt,
                    runner,
                    thread_id,
                    worktree,
                    baseline_tree,
                    program_directory,
                    cycle_directory,
                    historical_best_score,
                )
                attempts.append(attempt)
                accepted_tree: str | None = None
                if attempt.passed:
                    accepted_refinement = _load_passing_agentic_tuning_result(attempt)
                    accepted_tree = _checkpoint_candidate(worktree, baseline_tree)
                failure_reason: str | None = None
                if not attempt.passed:
                    if thread_id is None:
                        thread_start_attempts += 1
                        if thread_start_attempts >= config.max_thread_start_attempts:
                            failure_reason = (
                                "Codex did not emit thread.started after "
                                f"{thread_start_attempts} fresh-start attempts"
                            )
                        else:
                            prompt = cycle_prompt
                    else:
                        prompt = retry_prompt(attempt)
                outcome = "accepted" if attempt.passed else ("failed" if failure_reason is not None else "retry")
                passed_gates = sum(gate.passed for gate in attempt.gates)
                target_score = (
                    accepted_refinement.best_score if attempt.passed and accepted_refinement is not None else None
                )
                _LOGGER.info(
                    "cycle %03d attempt %03d | %s | changed_files=%d | gates=%d/%d | candidate_mfu=%s",
                    cycle_index,
                    attempt_index,
                    outcome,
                    len(attempt.changed_files),
                    passed_gates,
                    len(config.gates),
                    _score_text(target_score),
                )
                cycle = _cycle_result(
                    cycle_index,
                    baseline_tree,
                    baseline,
                    historical_best_score,
                    thread_id,
                    tuple(attempts),
                    accepted_tree,
                    failure_reason,
                )
                cycles[-1] = cycle
                _write_running_record(
                    config,
                    run_id,
                    repository,
                    base_sha,
                    initial_tree,
                    worktree,
                    run_directory,
                    tuple(cycles),
                    initial_refinement,
                )
                if failure_reason is not None:
                    verdict = "failed"

            if cycle.failure_reason is None:
                if accepted_refinement is None or cycle.accepted_tree is None:
                    raise RuntimeError("accepted cycle has no passing agentic tuning artifact or candidate tree")
                accepted_score = accepted_refinement.best_score
                if accepted_score is None:
                    raise RuntimeError("accepted cycle has no valid MFU score")
                if historical_best_score is None or accepted_score > historical_best_score:
                    historical_best_score = accepted_score
                _LOGGER.info(
                    "cycle %03d | accepted | candidate_mfu=%s | historical_best_mfu=%s | tree=%s",
                    cycle_index,
                    _score_text(accepted_score),
                    _score_text(historical_best_score),
                    cycle.accepted_tree[:12],
                )
                baseline = accepted_refinement
                baseline_tree = cycle.accepted_tree
    except KeyboardInterrupt:
        verdict = "stopped"
        _LOGGER.info("workflow | stop requested")
    return tuple(cycles), verdict


def _workflow_result(
    run_id: str,
    verdict: Verdict,
    run_directory: Path,
    worktree: Path,
    base_sha: str,
    initial_tree: str,
    initial_refinement: AgenticTuningResult | None,
    cycles: tuple[CycleResult, ...],
) -> WorkflowResult:
    """Build the final public workflow result."""
    result = WorkflowResult(
        run_id=run_id,
        verdict=verdict,
        run_directory=run_directory,
        worktree=worktree,
        base_sha=base_sha,
        initial_tree=initial_tree,
        initial_refinement=initial_refinement,
        cycles=cycles,
    )
    return result


def run_workflow(config: RunConfig) -> WorkflowResult:
    """Continuously tune, improve, verify, and promote IR candidates."""
    repository = resolve_repository(config.repository)
    _validate_config(config, repository)
    base_sha = resolve_revision(repository, config.base_revision)
    initial_tree = create_candidate_tree(repository, base_sha)
    run_id = _new_run_id()
    run_directory = config.artifact_root.expanduser().resolve() / run_id
    run_directory.mkdir(parents=True)
    _LOGGER.info("run %s | started | program=%s | artifacts=%s", run_id, config.program.name, run_directory)
    worktree = run_directory / "worktree"
    create_detached_worktree(repository, worktree, base_sha, initial_tree)
    _LOGGER.info(
        "run %s | worktree ready | base=%s | initial_tree=%s | path=%s",
        run_id,
        base_sha[:12],
        initial_tree[:12],
        worktree,
    )
    program_directory = run_directory / "program"
    write_program(config.program, program_directory)
    _write_json(
        run_directory / "run.json",
        _run_record(config, run_id, "tuning", repository, base_sha, initial_tree, worktree, (), None),
    )

    initial_directory = run_directory / "agentic-tuning" / "initial"
    initial_refinement: AgenticTuningResult | None = None
    cycles: tuple[CycleResult, ...] = ()
    stopped = False
    development_verdict: Verdict = "failed"
    try:
        _LOGGER.info("run %s | initial tuning", run_id)
        initial_refinement = run_agentic_tuning(
            spec=config.agentic_tuning,
            program_directory=program_directory,
            worktree=worktree,
            output_directory=initial_directory,
            environment=candidate_environment(worktree),
            source_fingerprint=lambda: snapshot_candidate(worktree, initial_tree).patch,
        )
    except KeyboardInterrupt:
        stopped = True
    if initial_refinement is not None:
        _write_json(initial_directory / "result.json", initial_refinement.as_dict())
    if initial_refinement is not None and initial_refinement.passed and not stopped:
        _LOGGER.info("run %s | baseline ready | best_mfu=%s", run_id, _score_text(initial_refinement.best_score))
        cycles, development_verdict = _develop_cycles(
            config,
            run_id,
            repository,
            base_sha,
            initial_tree,
            worktree,
            program_directory,
            run_directory,
            initial_refinement,
        )

    verdict: Verdict = "failed"
    if stopped:
        verdict = "stopped"
    elif initial_refinement is not None and initial_refinement.passed:
        verdict = development_verdict
    result = _workflow_result(
        run_id, verdict, run_directory, worktree, base_sha, initial_tree, initial_refinement, cycles
    )
    final_snapshot = snapshot_candidate(worktree, base_sha)
    (run_directory / "final.patch").write_text(final_snapshot.patch, encoding="utf-8")
    final_record = {
        **_run_record(
            config, run_id, verdict, repository, base_sha, initial_tree, worktree, cycles, initial_refinement
        ),
        "result": result.as_dict(),
    }
    _write_json(run_directory / "run.json", final_record)
    _LOGGER.info(
        "run %s | %s | accepted_cycles=%d | artifacts=%s",
        run_id,
        verdict,
        sum(cycle.passed for cycle in cycles),
        run_directory,
    )
    return result
