"""Public nkigym-first entry point for transform development."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from developer.defaults import agentic_tuning_spec, gates
from developer.git import resolve_repository
from developer.orchestrator import run_workflow
from developer.types import RunConfig, WorkflowResult
from nkigym.search.profiled_refinement import ReasoningEffort
from nkigym.search.program import program_from_callable
from nkigym.search.types import InputSpecs

_SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
_REASONING_EFFORT: ReasoningEffort = "high"
_PROFILE_TIMEOUT_SECONDS = 1800
_POLICY_TIMEOUT_SECONDS = 600
_REFINEMENT_TIMEOUT_SECONDS = 21_300
_PROFILE_LNC = 1
_MAX_CYCLES: int | None = None
_MAX_THREAD_START_ATTEMPTS = 3
_AGENT_TIMEOUT_SECONDS = 3600
_GOAL = (
    "Continuously improve the IR, operation contracts, code generation, and transforms using trace and profile evidence"
)


def _default_artifact_root(repository: Path) -> Path:
    """Return the persistent artifact location for a repository."""
    configured = os.environ.get("XDG_STATE_HOME")
    state_root = Path(configured).expanduser() if configured is not None else Path.home() / ".local/state"
    return state_root / "developer" / repository.name


def developer(f_nkigym: Callable[..., Any], input_specs: InputSpecs, profile_host: str, /) -> WorkflowResult:
    """Continuously develop nkigym IR and transforms until explicitly stopped."""
    if not isinstance(profile_host, str) or not profile_host.strip():
        raise ValueError("profile_host must not be empty")
    repository = resolve_repository(Path.cwd())
    artifact_root = _default_artifact_root(repository)
    codex_executable = os.environ.get("CODEX_EXECUTABLE", "codex")
    tuning = agentic_tuning_spec(
        host=profile_host,
        reasoning_effort=_REASONING_EFFORT,
        profile_timeout_seconds=_PROFILE_TIMEOUT_SECONDS,
        policy_timeout_seconds=_POLICY_TIMEOUT_SECONDS,
        tuning_timeout_seconds=_REFINEMENT_TIMEOUT_SECONDS,
        lnc=_PROFILE_LNC,
        codex_executable=codex_executable,
    )
    run_config = RunConfig(
        repository=repository,
        artifact_root=artifact_root,
        program=program_from_callable(f_nkigym, input_specs, _SCHEDULER_OFF_ARGS, _PROFILE_LNC),
        agentic_tuning=tuning,
        gates=gates(profile_host),
        goal=_GOAL,
        codex_executable=codex_executable,
        base_revision="HEAD",
        max_cycles=_MAX_CYCLES,
        max_thread_start_attempts=_MAX_THREAD_START_ATTEMPTS,
        agent_timeout_seconds=_AGENT_TIMEOUT_SECONDS,
    )
    result = run_workflow(run_config)
    return result


__all__ = ["developer"]
