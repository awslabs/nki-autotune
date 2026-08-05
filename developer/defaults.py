"""Default compiler and evaluation configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from developer.types import GateSpec
from nkigym.search.agentic_tuning import AgenticTuningSpec
from nkigym.search.profiled_refinement import ReasoningEffort


def developer_python() -> str:
    """Return the interpreter used by compiler and gate processes."""
    configured = os.environ.get("DEVELOPER_PYTHON")
    kernel_python = Path.home() / "venvs/kernel-env/bin/python"
    if configured is not None:
        executable = configured
    elif kernel_python.is_file():
        executable = str(kernel_python)
    else:
        executable = sys.executable
    return executable


def agentic_tuning_spec(
    host: str,
    reasoning_effort: ReasoningEffort,
    profile_timeout_seconds: int,
    policy_timeout_seconds: int,
    tuning_timeout_seconds: int,
    lnc: int,
    codex_executable: str,
) -> AgenticTuningSpec:
    """Build the nkigym agentic tuning command."""
    argv = [
        developer_python(),
        "-m",
        "nkigym.search.agentic_tuning_cli",
        "--host",
        host,
        "--reasoning-effort",
        reasoning_effort,
        "--profile-timeout-seconds",
        str(profile_timeout_seconds),
        "--policy-timeout-seconds",
        str(policy_timeout_seconds),
        "--lnc",
        str(lnc),
        "--codex-executable",
        codex_executable,
    ]
    spec = AgenticTuningSpec(
        name="agentic-tuning",
        argv=tuple(argv),
        required_artifacts=("result.json", "nodes/node_000/evaluation.json"),
        timeout_seconds=tuning_timeout_seconds,
    )
    return spec


def gates(profile_host: str) -> tuple[GateSpec, ...]:
    """Return the five candidate acceptance evaluations."""
    test_python = developer_python()
    configured = (
        GateSpec(
            name="code-bloat",
            argv=(test_python, "-m", "pytest", "-q", "test/test_code_bloat.py", "-s"),
            working_directory=".",
            timeout_seconds=120,
        ),
        GateSpec(
            name="random-rollout-correctness",
            argv=(test_python, "-m", "pytest", "-q", "test/test_random_rollout.py", "-s"),
            working_directory=".",
            timeout_seconds=7200,
        ),
        GateSpec(
            name="transform-evaluation",
            argv=(test_python, "-m", "pytest", "-q", "test/test_transform_evaluation.py", "-s"),
            working_directory=".",
            timeout_seconds=3900,
        ),
        GateSpec(
            name="mfu-regression",
            argv=(test_python, "-m", "pytest", "-q", "test/test_mfu_regression.py", "-s"),
            working_directory=".",
            timeout_seconds=7500,
            environment=(("NKI_PROFILE_HOST", profile_host),),
        ),
        GateSpec(
            name="agentic-tuning",
            argv=(test_python, "-m", "pytest", "-q", "test/test_agentic_tuning.py", "-s"),
            working_directory=".",
            timeout_seconds=21600,
        ),
    )
    return configured


__all__ = ["agentic_tuning_spec", "developer_python", "gates"]
