"""Agentic tuning acceptance for the configured target workload."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from nkigym.search.agentic_tuning import (
    AGENTIC_TUNING_CONTEXT_ENV,
    AgenticTuningContext,
    AgenticTuningResult,
    run_agentic_tuning,
)

MAX_MFU_REGRESSION_POINTS = 1.0
MAX_FAILURE_LOG_CHARACTERS = 12000
GATE_ARTIFACT_DIRECTORY_ENV = "NKIGYM_GATE_ARTIFACT_DIRECTORY"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_context() -> AgenticTuningContext:
    """Load the controller-owned target workload and historical MFU context."""
    configured_path = os.environ.get(AGENTIC_TUNING_CONTEXT_ENV)
    if configured_path is None:
        raise AssertionError(f"{AGENTIC_TUNING_CONTEXT_ENV} must identify the agentic tuning context")
    return AgenticTuningContext.from_path(Path(configured_path))


def _artifact_directory(tmp_path: Path) -> Path:
    """Return the controller artifact directory or a pytest temporary directory."""
    configured_directory = os.environ.get(GATE_ARTIFACT_DIRECTORY_ENV)
    directory = Path(configured_directory) if configured_directory is not None else tmp_path
    return directory.expanduser().resolve()


def _candidate_environment() -> dict[str, str]:
    """Import candidate nkigym source in nested tuning processes."""
    environment = dict(os.environ)
    source = str(REPOSITORY_ROOT / "nkigym/src")
    existing = environment.get("PYTHONPATH")
    entries = [source]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _run_git(arguments: tuple[str, ...], environment: dict[str, str]) -> str:
    """Run one Git command required for an exact source tree fingerprint."""
    command = ("git", "-C", str(REPOSITORY_ROOT), *arguments)
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AssertionError(f"{' '.join(command)} failed with exit {completed.returncode}: {detail}")
    return completed.stdout.strip()


def _source_tree(baseline_tree: str) -> str:
    """Return the current non-ignored workspace as a Git tree hash."""
    with tempfile.TemporaryDirectory(prefix="agentic-tuning-source-index-") as temporary:
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        _run_git(("read-tree", baseline_tree), environment)
        _run_git(("add", "--all", "--", "."), environment)
        tree = _run_git(("write-tree",), environment)
    return tree


def _failure_report(result: AgenticTuningResult, record: dict[str, object]) -> str:
    """Render bounded tuning evidence for one failed assertion."""
    log = ""
    if result.log_path.is_file():
        log = result.log_path.read_text(encoding="utf-8", errors="replace")[-MAX_FAILURE_LOG_CHARACTERS:]
    report = json.dumps(record, indent=2, sort_keys=True)
    return f"{report}\n\n{log}"


def test_agentic_tuning_produces_valid_non_regressing_evidence(tmp_path: Path) -> None:
    """Target tuning produces measured evidence within the allowed MFU fluctuation."""
    context = _load_context()
    artifact_directory = _artifact_directory(tmp_path)
    result = run_agentic_tuning(
        spec=context.tuning,
        program_directory=context.program_directory,
        worktree=REPOSITORY_ROOT,
        output_directory=artifact_directory,
        environment=_candidate_environment(),
        source_fingerprint=lambda: _source_tree(context.baseline_tree),
    )
    candidate_score = result.best_score
    historical_best = context.historical_best_score
    minimum_score = None if historical_best is None else historical_best - MAX_MFU_REGRESSION_POINTS
    within_tolerance = candidate_score is not None and (minimum_score is None or candidate_score >= minimum_score)
    record = {
        **result.as_dict(),
        "historical_best_score": historical_best,
        "max_mfu_regression_points": MAX_MFU_REGRESSION_POINTS,
        "minimum_accepted_score": minimum_score,
        "accepted": result.passed and within_tolerance,
    }
    result_path = artifact_directory / "result.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)

    assert result.passed, _failure_report(result, record)
    assert candidate_score is not None, "agentic tuning produced no valid measured MFU score"
    if historical_best is not None:
        assert candidate_score >= historical_best - MAX_MFU_REGRESSION_POINTS, (
            f"candidate best MFU {candidate_score:.2f}% trails historical best {historical_best:.2f}% by "
            f"{historical_best - candidate_score:.2f} points; limit is {MAX_MFU_REGRESSION_POINTS:.2f}"
        )
