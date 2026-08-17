"""Neuron Explorer execution used only by the installed Trn2 worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(stage: str, command: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one Neuron Explorer command and preserve its diagnostics."""
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"{stage} failed with exit {completed.returncode}{suffix}")
    return completed


def _environment(lnc: int, visible_core: int) -> dict[str, str]:
    """Return an isolated Neuron runtime environment for one logical core."""
    if lnc not in {1, 2}:
        raise ValueError("lnc must be 1 or 2")
    if not isinstance(visible_core, int) or isinstance(visible_core, bool) or visible_core < 0:
        raise ValueError("visible core must be a non-negative integer")
    environment = dict(os.environ)
    path_entries = (
        str(Path(sys.executable).parent),
        "/opt/aws/neuron/bin",
        *environment.get("PATH", "").split(os.pathsep),
    )
    environment["PATH"] = os.pathsep.join(dict.fromkeys(path_entries))
    environment["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    environment["NEURON_RT_VISIBLE_CORES"] = str(visible_core)
    return environment


def benchmark_kernel(neff_path: Path, artifacts_dir: Path, lnc: int, visible_core: int) -> dict[str, object]:
    """Capture one NEFF execution and return its Neuron Explorer summary."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ntff_path = artifacts_dir / "profile.ntff"
    environment = _environment(lnc, visible_core)
    _run(
        "Neuron Explorer capture",
        ["neuron-explorer", "capture", "--neff", str(neff_path), "--session-file", str(ntff_path)],
        environment,
    )
    if not ntff_path.is_file():
        raise RuntimeError(f"Neuron Explorer returned without creating {ntff_path}")
    completed = _run(
        "Neuron Explorer summary",
        [
            "neuron-explorer",
            "view",
            "--neff-path",
            str(neff_path),
            "--session-file",
            str(ntff_path),
            "--output-format",
            "summary-json",
        ],
        environment,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Neuron Explorer returned invalid summary JSON: {error}") from error
    if not isinstance(payload, dict) or len(payload) != 1:
        raise RuntimeError("Neuron Explorer summary must contain exactly one model")
    raw_summary = next(iter(payload.values()))
    if not isinstance(raw_summary, dict):
        raise RuntimeError("Neuron Explorer returned a non-object model summary")
    return dict(raw_summary)
