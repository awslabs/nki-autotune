"""Tests for durable refinement in the source checkout."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from nkigym.search.agentic_tuning import AgenticTuningResult, AgenticTuningSpec
from nkigym.search.program import ProgramSpec
from self_evolve.git import create_candidate_tree, snapshot_candidate
from self_evolve.types import CheckAttempt, GateResult, GateSpec, RunConfig
from self_evolve.workflow import _candidate_fingerprint, _read_record, _write_record, accept_run, create_run, status_run


def _git(repository: Path, *arguments: str) -> str:
    """Run Git in one temporary repository."""
    completed = subprocess.run(("git", "-C", str(repository), *arguments), check=True, text=True, capture_output=True)
    return completed.stdout


def _repository(path: Path) -> Path:
    """Create a branch checkout with staged, unstaged, and untracked content."""
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    implementation = path / "nkigym/src/nkigym/example.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text('VALUE = "committed"\n', encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "initial")
    implementation.write_text('VALUE = "working"\n', encoding="utf-8")
    staged = path / "staged.txt"
    staged.write_text("staged\n", encoding="utf-8")
    _git(path, "add", "staged.txt")
    (path / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    return path


def _config(repository: Path, artifact_root: Path) -> RunConfig:
    """Return a minimal valid durable-run configuration."""
    config = RunConfig(
        repository=repository,
        artifact_root=artifact_root,
        program=ProgramSpec(
            name="f_nkigym",
            nkigym_source="def f_nkigym(x):\n    return x\n",
            input_specs={"x": ((1,), "float32")},
            workload_guidance="Test direct-checkout mechanics.",
            neuronx_cc_args=(),
        ),
        agentic_tuning=AgenticTuningSpec(
            name="agentic-tuning", argv=("tune",), required_artifacts=("result.json",), timeout_seconds=30
        ),
        gates=(
            GateSpec(name="baseline", argv=("baseline",), working_directory=".", timeout_seconds=30),
            GateSpec(name="agentic-tuning", argv=("tune",), working_directory=".", timeout_seconds=30),
        ),
        improvement_round_limit=1,
        initial_historical_best_score=49.0,
        base_revision="HEAD",
    )
    return config


def _gate_result(name: str, artifact_directory: Path) -> GateResult:
    """Return one passing gate result."""
    result = GateResult(
        name=name,
        argv=(name,),
        exit_code=0,
        timed_out=False,
        duration_seconds=1.0,
        log_path=artifact_directory / f"{name}.log",
        artifact_directory=artifact_directory,
    )
    return result


def test_create_run_uses_source_checkout_without_changing_user_state(tmp_path: Path) -> None:
    """Run creation snapshots the current branch without creating a worktree."""
    repository = _repository(tmp_path / "source")
    status_before = _git(repository, "status", "--short")

    created = create_run(_config(repository, tmp_path / "artifacts"))
    resumed = status_run(created.run_directory)
    record = json.loads((created.run_directory / "run.json").read_text(encoding="utf-8"))
    worktree_count = _git(repository, "worktree", "list", "--porcelain").count("worktree ")

    assert created.worktree == repository.resolve()
    assert resumed.worktree == repository.resolve()
    assert created.next_action == "validate"
    assert created.changed_files == ()
    assert record["schema_version"] == 19
    assert record["worktree"] == str(repository.resolve())
    assert not (created.run_directory / "worktree").exists()
    assert worktree_count == 1
    assert _git(repository, "status", "--short") == status_before


def test_accept_run_preserves_branch_head_and_index(tmp_path: Path) -> None:
    """Acceptance records a baseline without rewriting the source Git state."""
    repository = _repository(tmp_path / "source")
    created = create_run(_config(repository, tmp_path / "artifacts"))
    record = _read_record(created.run_directory)
    cycle = record.current_cycle
    implementation = repository / "nkigym/src/nkigym/example.py"
    implementation.write_text('VALUE = "candidate"\n', encoding="utf-8")
    snapshot = snapshot_candidate(repository, cycle.baseline_tree)
    candidate_tree = create_candidate_tree(repository, cycle.baseline_tree)
    fingerprint = _candidate_fingerprint(snapshot)
    check_directory = created.run_directory / "cycles/000/checks/000"
    agentic_directory = check_directory / "gates/agentic-tuning-artifacts"
    agentic_directory.mkdir(parents=True)
    tuning = AgenticTuningResult(
        command=("tune",),
        exit_code=0,
        timed_out=False,
        duration_seconds=1.0,
        artifact_directory=agentic_directory,
        log_path=agentic_directory / "tuning.log",
        missing_artifacts=(),
        worktree_modified=False,
        program_modified=False,
        best_score=50.0,
        result_error=None,
    )
    (agentic_directory / "result.json").write_text(json.dumps(tuning.as_dict(), indent=2) + "\n", encoding="utf-8")
    patch_path = check_directory / "diff.patch"
    patch_path.write_text(snapshot.patch, encoding="utf-8")
    check = CheckAttempt(
        index=0,
        artifact_directory=check_directory,
        candidate_tree=candidate_tree,
        candidate_fingerprint=fingerprint,
        changed_files=snapshot.changed_files,
        patch_path=patch_path,
        gates=(
            _gate_result("baseline", check_directory / "gates/baseline-artifacts"),
            _gate_result("agentic-tuning", agentic_directory),
        ),
        worktree_modified=False,
        passed=True,
    )
    updated_cycle = replace(cycle, checks=(check,))
    _write_record(replace(record, cycles=(updated_cycle,)))
    branch_before = _git(repository, "branch", "--show-current")
    head_before = _git(repository, "rev-parse", "HEAD")
    index_before = _git(repository, "diff", "--cached", "--binary")

    accepted = accept_run(created.run_directory)

    assert accepted.next_action == "complete"
    assert accepted.worktree == repository.resolve()
    assert _git(repository, "branch", "--show-current") == branch_before
    assert _git(repository, "rev-parse", "HEAD") == head_before
    assert _git(repository, "diff", "--cached", "--binary") == index_before
    assert implementation.read_text(encoding="utf-8") == 'VALUE = "candidate"\n'
