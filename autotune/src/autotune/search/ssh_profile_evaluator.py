"""Profile local search states on an always-on remote Trn2 host."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from autotune.search.profile_evaluator import ProfileEvaluatorConfig
from autotune.search.types import Evaluation, EvaluationMetric
from nkigym.codegen import render
from nkigym.ir import KernelIR

_RESULT_PREFIX = "AUTOTUNE_PROFILE_RESULT="
_SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no")


@dataclass(frozen=True)
class SSHProfileConfig:
    """Repository and transport settings for remote state evaluation."""

    host: str
    local_repo: Path
    remote_repo_subdir: str
    remote_cache_subdir: str
    remote_activate: str
    timeout_s: int


@dataclass(frozen=True)
class ProfileKernel:
    """Standalone NKI source and its entry-point name."""

    source: str
    func_name: str


class SSHNKIProfileEvaluator:
    """Synchronize source once, then profile each state over SSH."""

    def __init__(self, profile_config: ProfileEvaluatorConfig, ssh_config: SSHProfileConfig) -> None:
        """Store profile and SSH controls."""
        _validate_remote_subdir(ssh_config.remote_repo_subdir, "remote_repo_subdir")
        _validate_remote_subdir(ssh_config.remote_cache_subdir, "remote_cache_subdir")
        self.profile_config = profile_config
        self.ssh_config = ssh_config
        self._synced = False

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Send one rendered state to the remote worker and retrieve artifacts."""
        label = f"node_{node_id:03d}"
        evaluations = self.profile_sources(
            kernels={label: ProfileKernel(source=render(state), func_name=f"nki_{state.func_name}")},
            cache_dir=cache_dir / "profile",
            run_id=label,
        )
        return evaluations[label]

    def profile_sources(self, kernels: dict[str, ProfileKernel], cache_dir: Path, run_id: str) -> dict[str, Evaluation]:
        """Profile one or more standalone kernels in a single remote runner call."""
        if not kernels:
            raise ValueError("at least one profile kernel is required")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
            raise ValueError(f"invalid profile run_id {run_id!r}")
        self._ensure_synced()
        request = self._request(kernels)
        remote_cache = f"{self.ssh_config.remote_cache_subdir}/{run_id}"
        command = self._remote_command(remote_cache)
        completed = subprocess.run(
            ["ssh", *_SSH_OPTIONS, self.ssh_config.host, command],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=self.ssh_config.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"remote profile failed with exit {completed.returncode}: "
                f"{(completed.stdout + completed.stderr)[-3000:]}"
            )
        evaluations = _parse_evaluations(completed.stdout)
        if set(evaluations) != set(kernels):
            raise RuntimeError(f"remote profile returned labels {sorted(evaluations)}, expected {sorted(kernels)}")
        self._sync_artifacts(remote_cache, cache_dir)
        return evaluations

    def _ensure_synced(self) -> None:
        """Synchronize the repository to the remote host once."""
        if not self._synced:
            remote_repo = shlex.quote(self.ssh_config.remote_repo_subdir)
            subprocess.run(["ssh", *_SSH_OPTIONS, self.ssh_config.host, f'mkdir -p "$HOME"/{remote_repo}'], check=True)
            command = [
                "rsync",
                "-az",
                "--delete",
                "--exclude",
                ".git",
                "--exclude",
                "__pycache__",
                "--exclude",
                "*.pyc",
                "--exclude",
                ".pytest_cache",
                "--exclude",
                ".mypy_cache",
                "--exclude",
                "build",
                "--exclude",
                ".venv",
                "-e",
                "ssh " + " ".join(_SSH_OPTIONS),
                f"{self.ssh_config.local_repo}/",
                f"{self.ssh_config.host}:{self.ssh_config.remote_repo_subdir}/",
            ]
            subprocess.run(command, check=True)
            self._synced = True

    def _request(self, kernels: dict[str, ProfileKernel]) -> dict[str, object]:
        """Serialize standalone kernels and their static runner configuration."""
        input_specs = {name: [list(shape), dtype] for name, (shape, dtype) in self.profile_config.input_specs.items()}
        return {
            "kernels": {
                name: {"source": kernel.source, "func_name": kernel.func_name} for name, kernel in kernels.items()
            },
            "output_shape": list(self.profile_config.output_shape),
            "input_specs": input_specs,
            "neuronx_cc_args": list(self.profile_config.neuronx_cc_args),
            "seed": self.profile_config.seed,
            "neuron_platform_target": self.profile_config.neuron_platform_target,
        }

    def _remote_command(self, remote_cache: str) -> str:
        """Build the fixed remote worker command without embedding request data."""
        repo = shlex.quote(self.ssh_config.remote_repo_subdir)
        cache = shlex.quote(remote_cache)
        return (
            f"{self.ssh_config.remote_activate} && "
            f'cd "$HOME"/{repo} && '
            f'rm -rf "$HOME"/{cache} && mkdir -p "$HOME"/{cache} && '
            "PYTHONPATH=.:nkigym/src:autotune/src "
            f'python -m autotune.search.remote_profile --cache "$HOME"/{cache}'
        )

    def _sync_artifacts(self, remote_cache: str, local_cache: Path) -> None:
        """Reverse-sync one completed remote profile directory."""
        local_cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                "ssh " + " ".join(_SSH_OPTIONS),
                f"{self.ssh_config.host}:{remote_cache}/",
                f"{local_cache}/",
            ],
            check=True,
        )


def _parse_evaluations(stdout: str) -> dict[str, Evaluation]:
    """Extract labeled evaluations from the worker's mixed process output."""
    result_line = next((line for line in reversed(stdout.splitlines()) if line.startswith(_RESULT_PREFIX)), None)
    if result_line is None:
        raise RuntimeError(f"remote profile emitted no result marker: {stdout[-3000:]}")
    payload = json.loads(result_line[len(_RESULT_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("remote profile result must be an object")
    raw_evaluations = payload.get("evaluations")
    if not isinstance(raw_evaluations, dict):
        raise ValueError("remote profile result lacks evaluations")
    evaluations: dict[str, Evaluation] = {}
    for label, raw_evaluation in raw_evaluations.items():
        if not isinstance(label, str) or not isinstance(raw_evaluation, dict):
            raise ValueError("remote profile evaluations must map labels to objects")
        evaluations[label] = _parse_evaluation(raw_evaluation)
    return evaluations


def _parse_evaluation(payload: dict[object, object]) -> Evaluation:
    """Validate one labeled worker evaluation."""
    score = payload.get("score")
    if score is not None and not isinstance(score, (int, float)):
        raise ValueError("remote profile score must be numeric or null")
    metrics = payload.get("metrics")
    message = payload.get("message")
    if not isinstance(metrics, dict) or not isinstance(message, str):
        raise ValueError("remote profile result lacks metrics or message")
    typed_metrics: dict[str, EvaluationMetric] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not isinstance(value, (float, int, str, bool, type(None))):
            raise ValueError("remote profile metric has unsupported type")
        typed_metrics[key] = value
    return Evaluation(score=float(score) if score is not None else None, metrics=typed_metrics, message=message)


def _validate_remote_subdir(value: str, field: str) -> None:
    """Reject absolute or parent-traversing remote paths."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{field} must be a non-empty relative path without '..'")


__all__ = ["ProfileKernel", "SSHProfileConfig", "SSHNKIProfileEvaluator"]
