"""Agentic tuning acceptance for the configured target workload."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nkigym.search.agentic_tuning import AgenticTuningContext, AgenticTuningResult, run_agentic_tuning

MAX_MFU_REGRESSION_POINTS = 1.0
MAX_FAILURE_LOG_CHARACTERS = 12000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_context(path: Path) -> AgenticTuningContext:
    """Load the controller-owned target workload and historical MFU context."""
    return AgenticTuningContext.from_path(path)


def _run_git(arguments: tuple[str, ...]) -> bytes:
    """Run one Git command required for an exact source tree fingerprint."""
    command = ("git", "-C", str(REPOSITORY_ROOT), *arguments)
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).decode(errors="replace")
        raise AssertionError(f"{' '.join(command)} failed with exit {completed.returncode}: {detail}")
    return completed.stdout


def _source_fingerprint(baseline_tree: str) -> str:
    """Return a fingerprint of tracked changes and untracked file contents."""
    digest = hashlib.sha256()
    digest.update(_run_git(("diff", "--binary", baseline_tree, "--", ".")))
    untracked = _run_git(("ls-files", "--others", "--exclude-standard", "-z"))
    digest.update(untracked)
    for relative_path in filter(None, untracked.split(b"\0")):
        content = (REPOSITORY_ROOT / relative_path.decode()).read_bytes()
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _failure_report(result: AgenticTuningResult, record: dict[str, object]) -> str:
    """Render bounded tuning evidence for one failed assertion."""
    log = ""
    if result.log_path.is_file():
        log = result.log_path.read_text(encoding="utf-8", errors="replace")[-MAX_FAILURE_LOG_CHARACTERS:]
    report = json.dumps(record, indent=2, sort_keys=True)
    return f"{report}\n\n{log}"


def test_agentic_tuning_produces_valid_non_regressing_evidence(tmp_path: Path) -> None:
    """Target tuning produces measured evidence within the allowed MFU fluctuation."""
    artifact_directory = tmp_path.parents[1]
    context = _load_context(artifact_directory / "agentic-tuning-context.json")
    result = run_agentic_tuning(
        spec=context.tuning,
        program_directory=context.program_directory,
        worktree=REPOSITORY_ROOT,
        output_directory=artifact_directory,
        source_fingerprint=lambda: _source_fingerprint(context.baseline_tree),
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
