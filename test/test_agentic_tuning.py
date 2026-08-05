"""Agentic tuning acceptance for the current developer target workload."""

from __future__ import annotations

import json
import os
from pathlib import Path

from developer.gates import candidate_environment
from developer.git import snapshot_candidate
from nkigym.search.agentic_tuning import (
    AGENTIC_TUNING_CONTEXT_ENV,
    AgenticTuningContext,
    AgenticTuningResult,
    run_agentic_tuning,
)

MAX_MFU_REGRESSION_POINTS = 1.0
MAX_FAILURE_LOG_CHARACTERS = 12000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_context() -> AgenticTuningContext:
    """Load the controller-owned target workload and historical MFU context."""
    configured_path = os.environ.get(AGENTIC_TUNING_CONTEXT_ENV)
    if configured_path is None:
        raise AssertionError(f"{AGENTIC_TUNING_CONTEXT_ENV} must identify the agentic tuning context")
    return AgenticTuningContext.from_path(Path(configured_path))


def _artifact_directory(tmp_path: Path) -> Path:
    """Return the controller artifact directory or a pytest temporary directory."""
    configured_directory = os.environ.get("DEVELOPER_GATE_ARTIFACT_DIRECTORY")
    directory = Path(configured_directory) if configured_directory is not None else tmp_path
    return directory.expanduser().resolve()


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
        environment=candidate_environment(REPOSITORY_ROOT),
        source_fingerprint=lambda: snapshot_candidate(REPOSITORY_ROOT, context.baseline_tree).patch,
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
