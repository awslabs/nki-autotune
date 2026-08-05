"""Create durable runs for the develop-nkigym skill."""

from __future__ import annotations

import os
from pathlib import Path

from develop_nkigym.defaults import agentic_tuning_spec, gates
from develop_nkigym.git import resolve_repository
from develop_nkigym.types import RunConfig, RunStatus
from develop_nkigym.workflow import create_run as create_configured_run
from kernel_library import Workload
from nkigym.search.profiled_refinement import ReasoningEffort
from nkigym.search.program import program_from_callable

_SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
_REASONING_EFFORT: ReasoningEffort = "high"
_PROFILE_TIMEOUT_SECONDS = 1800
_POLICY_TIMEOUT_SECONDS = 600
_TUNING_TIMEOUT_SECONDS = 21_300
_PROFILE_LNC = 1


def default_artifact_root(repository: Path) -> Path:
    """Return the persistent skill artifact location for a repository."""
    configured = os.environ.get("XDG_STATE_HOME")
    state_root = Path(configured).expanduser() if configured is not None else Path.home() / ".local/state"
    return state_root / "develop-nkigym" / repository.name


def create_run(
    workload: Workload,
    profile_host: str,
    improvement_round_limit: int,
    /,
    *,
    repository: Path | None = None,
    artifact_root: Path | None = None,
    base_revision: str = "HEAD",
) -> RunStatus:
    """Create a durable run and return before tuning or editing begins."""
    if not isinstance(profile_host, str) or not profile_host.strip():
        raise ValueError("profile_host must not be empty")
    if (
        isinstance(improvement_round_limit, bool)
        or not isinstance(improvement_round_limit, int)
        or improvement_round_limit < 1
    ):
        raise ValueError("improvement_round_limit must be a positive integer")
    source_repository = resolve_repository(Path.cwd() if repository is None else repository)
    output_root = default_artifact_root(source_repository) if artifact_root is None else artifact_root
    codex_executable = os.environ.get("CODEX_EXECUTABLE", "codex")
    tuning = agentic_tuning_spec(
        host=profile_host,
        reasoning_effort=_REASONING_EFFORT,
        profile_timeout_seconds=_PROFILE_TIMEOUT_SECONDS,
        policy_timeout_seconds=_POLICY_TIMEOUT_SECONDS,
        tuning_timeout_seconds=_TUNING_TIMEOUT_SECONDS,
        lnc=_PROFILE_LNC,
        codex_executable=codex_executable,
    )
    config = RunConfig(
        repository=source_repository,
        artifact_root=output_root,
        program=program_from_callable(workload.f_nkigym, workload.input_specs, _SCHEDULER_OFF_ARGS, _PROFILE_LNC),
        agentic_tuning=tuning,
        gates=gates(profile_host),
        improvement_round_limit=improvement_round_limit,
        initial_historical_best_score=workload.historical_best_mfu,
        base_revision=base_revision,
    )
    status = create_configured_run(config)
    return status


__all__ = ["create_run", "default_artifact_root"]
