"""Evidence-based prompts for each Codex implementation cycle."""

from __future__ import annotations

from pathlib import Path

from developer.types import AttemptResult, RunConfig
from nkigym.search.agentic_tuning import AgenticTuningResult

_MAX_GATE_LOG_CHARACTERS = 12000
_MAX_EVIDENCE_FILES = 200


def _input_specs_text(config: RunConfig) -> str:
    """Render the user input specifications for the agent prompt."""
    lines = [f"- {name}: shape={shape}, dtype={dtype}" for name, (shape, dtype) in config.program.input_specs.items()]
    return "\n".join(lines)


def _evidence_inventory(refinement: AgenticTuningResult) -> str:
    """Render a bounded inventory without interpreting tuning artifacts."""
    paths = sorted(path.relative_to(refinement.artifact_directory) for path in refinement.artifact_directory.rglob("*"))
    files = [path for path in paths if (refinement.artifact_directory / path).is_file()]
    visible = files[:_MAX_EVIDENCE_FILES]
    lines = [f"- {path}" for path in visible]
    if len(visible) < len(files):
        lines.append(f"- {len(files) - len(visible)} additional files omitted from this inventory")
    return "\n".join(lines)


def initial_prompt(
    config: RunConfig, refinement: AgenticTuningResult, historical_best_score: float | None, program_directory: Path
) -> str:
    """Build one evidence-based IR development prompt."""
    prompt = f"""# IR and transform improvement task

Goal:
{config.goal.strip()}

Target nkigym function:
{config.program.name}

Input specifications:
{_input_specs_text(config)}

Read-only program artifacts:
- nkigym program: {program_directory / "f_nkigym.py"}
- program metadata: {program_directory / "program.json"}

The candidate repository was initialized from an exact non-ignored snapshot of the caller's workspace. Its Git status
may therefore show baseline changes relative to the configured base commit. Treat all files present at the start of
this cycle as the baseline: preserve them, do not clean or revert them, and evaluate your work relative to the
controller's internal baseline tree.

Profiler-guided tuning evidence:
- artifact directory: {refinement.artifact_directory}
- tuning log: {refinement.log_path}
- historical best MFU: {_score_text(historical_best_score)}

Artifact inventory:
{_evidence_inventory(refinement)}

Study the measured search tree, parent links, branch decisions, generated kernels, successful profiles, and failed
profiles in that artifact directory. Compare sibling outcomes instead of treating the latest branch as final. Then
inspect the current IR, operation contracts, transforms, code generator, and acceptance tests in the candidate
repository.
Use concrete evidence from the trace to choose one focused, high-confidence implementation improvement. Valid work
includes:
- fixing or extending the IR representation, invariants, dependency analysis, or scheduling model;
- fixing or extending operation contracts or code generation required by the IR design;
- fixing, removing, consolidating, adding, or generalizing a transform;
- making a transform semantically atomic.

Implement the chosen improvement and run the repository acceptance tests. Do not add transform-specific regression
tests: the repository intentionally uses only deterministic source-size and transform API limits, random rollout
correctness, agentic evaluation of every public transform for atomicity and genericity, best-known-kernel MFU
regression, and target-workload agentic tuning. Their implementations are the canonical acceptance criteria.

Do not modify the read-only program or tuning artifacts. The controller will run all five pytest files and accept the
cycle exactly when all five pass. The final test owns target tuning, evidence validation, and its permitted MFU
fluctuation. Do not commit changes.

The controller evaluates repository changes and commands directly. Your final prose is retained only for human review
and is not parsed as workflow state.
"""
    return prompt


def _tail(path: Path) -> str:
    """Read a bounded suffix of a text artifact for retry context."""
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = text[-_MAX_GATE_LOG_CHARACTERS:]
    return suffix


def _score_text(score: float | None) -> str:
    """Format an optional tuning score for a prompt."""
    text = "no successful profile" if score is None else f"{score:.6f}"
    return text


def retry_prompt(attempt: AttemptResult) -> str:
    """Build a continuation prompt entirely from controller-owned artifacts."""
    gate_sections = []
    for gate in attempt.gates:
        gate_sections.append(
            f"## Gate: {gate.name}\n"
            f"exit_code={gate.exit_code} timed_out={str(gate.timed_out).lower()}\n\n"
            f"```text\n{_tail(gate.log_path)}\n```"
        )
    gate_report = "\n\n".join(gate_sections) or "No gates ran because the process failed or no relevant change exists."
    changed = "\n".join(f"- {path}" for path in attempt.changed_files) or "- none"
    prompt = f"""Continue this implementation cycle and fix the failed attempt.

Codex process:
- exit_code={attempt.codex.exit_code}
- timed_out={str(attempt.codex.timed_out).lower()}

Codex stderr:
```text
{_tail(attempt.codex.stderr_log)}
```

Files changed since this cycle's accepted baseline:
{changed}

Gate results:
{gate_report}

Inspect the current files and make the smallest coherent correction. The cycle cannot finish until all repository
acceptance tests pass. The final pytest gate owns agentic tuning evidence and MFU acceptance. Do not add
transform-specific regression tests, modify program or tuning artifacts, or commit.
"""
    return prompt


__all__ = ["initial_prompt", "retry_prompt"]
