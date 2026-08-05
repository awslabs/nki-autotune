"""Prepare retained kernel endpoints for the nkigym-only MFU test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from develop_nkigym.workloads import load_workload, mfu_regression_workload_names
from kernel_library import Workload
from nkigym.codegen import render
from nkigym.environment import KernelMDP

_GATE_ARTIFACT_DIRECTORY_ENV = "NKIGYM_GATE_ARTIFACT_DIRECTORY"
_MFU_ENDPOINT_MANIFEST_ENV = "NKIGYM_MFU_ENDPOINT_MANIFEST"
_MANIFEST_FILENAME = "endpoints.json"


def _encoded_input_specs(workload: Workload) -> dict[str, object]:
    """Encode workload inputs for the test-owned manifest decoder."""
    return {name: {"shape": list(shape), "dtype": dtype} for name, (shape, dtype) in workload.input_specs.items()}


def _endpoint_record(name: str, workload: Workload) -> dict[str, object]:
    """Build and encode one retained best-known endpoint."""
    environment = KernelMDP(
        workload.f_nkigym,
        workload.input_specs,
        transforms=[transform for transform, _option in workload.best_action_ladder],
    )
    endpoint = environment.reset()
    for action in workload.best_action_ladder:
        endpoint = environment.step(endpoint, action)
    return {
        "name": name,
        "kernel": render(endpoint),
        "func_name": f"nki_{endpoint.func_name}",
        "input_specs": _encoded_input_specs(workload),
    }


def _write_manifest(path: Path) -> None:
    """Write all retained endpoints consumed by the MFU regression test."""
    endpoints = [_endpoint_record(name, load_workload(name)) for name in mfu_regression_workload_names()]
    path.write_text(json.dumps({"schema_version": 1, "endpoints": endpoints}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Prepare endpoint artifacts, then execute the canonical pytest gate."""
    configured_directory = os.environ.get(_GATE_ARTIFACT_DIRECTORY_ENV)
    if configured_directory is None:
        raise RuntimeError(f"{_GATE_ARTIFACT_DIRECTORY_ENV} must identify the gate artifact directory")
    artifact_directory = Path(configured_directory).expanduser().resolve()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_directory / _MANIFEST_FILENAME
    _write_manifest(manifest_path)

    environment = dict(os.environ)
    environment[_MFU_ENDPOINT_MANIFEST_ENV] = str(manifest_path)
    command = (sys.executable, "-m", "pytest", "-q", "test/test_mfu_regression.py", "-s")
    completed = subprocess.run(command, env=environment, check=False)
    return completed.returncode


__all__ = ["main"]
