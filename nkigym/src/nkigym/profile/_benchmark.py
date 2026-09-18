"""Neuron Explorer execution used only by the installed Trn2 worker."""

import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast


def _run(stage: str, command: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one Neuron Explorer command and preserve its diagnostics."""
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"{stage} failed with exit {completed.returncode}{suffix}")
    return completed


def _environment(lnc: int, visible_core: int | None) -> dict[str, str]:
    """Return an isolated Neuron runtime environment for one logical core."""
    invalid_core = visible_core is not None and (
        not isinstance(visible_core, int) or isinstance(visible_core, bool) or visible_core < 0
    )
    if lnc not in {1, 2} or invalid_core:
        raise ValueError("invalid Neuron runtime configuration")
    environment = dict(os.environ)
    entries = (str(Path(sys.executable).parent), "/opt/aws/neuron/bin", *environment.get("PATH", "").split(os.pathsep))
    environment["PATH"] = os.pathsep.join(dict.fromkeys(entries))
    environment["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    if visible_core is not None:
        environment["NEURON_RT_VISIBLE_CORES"] = str(visible_core)
    return environment


def available_logical_cores(lnc: int) -> tuple[int, ...]:
    """Return logical core IDs on devices with no active Neuron process."""
    completed = _run("neuron-ls", ["neuron-ls", "--json-output"], _environment(lnc, None))
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("neuron-ls output must be a JSON array")
    cores = tuple(
        core
        for device in payload
        if isinstance(device, dict) and not device.get("neuron_processes")
        for core in device.get("neuroncore_ids", ())
        if isinstance(core, int) and not isinstance(core, bool)
    )
    if not cores:
        raise RuntimeError("neuron-ls reported no unoccupied logical NeuronCores")
    return cores


def _summary(neff_path: Path, ntff_path: Path, environment: dict[str, str]) -> dict[str, object]:
    """Return one execution summary from a multi-execution capture."""
    completed = _run(
        "Neuron Explorer summary",
        ["neuron-explorer", "view", "-n", str(neff_path), "-s", str(ntff_path), "--output-format", "summary-json"],
        environment,
    )
    payload = json.loads(completed.stdout)
    raw_summary = next(iter(payload.values()), None) if isinstance(payload, dict) and len(payload) == 1 else None
    if not isinstance(raw_summary, dict):
        raise RuntimeError("Neuron Explorer returned a non-object model summary")
    return dict(raw_summary)


def _aggregate(summaries: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Return a fixed trimmed-mean summary across captured executions."""
    result = dict(summaries[0])
    for name in result:
        values = [
            float(value)
            for summary in summaries
            if isinstance((value := summary.get(name)), (int, float)) and not isinstance(value, bool)
        ]
        if len(values) == len(summaries):
            result[name] = statistics.fmean(sorted(values)[1:-1])
    result["nkigym_execution_total_times"] = [summary["total_time"] for summary in summaries]
    return result


def _execution_paths(path: Path, count: int) -> tuple[Path, ...]:
    """Return every execution trace path in one capture."""
    return (path,) + tuple(path.with_name(f"{path.stem}_exec_{index}{path.suffix}") for index in range(2, count + 1))


def _confirmation_capture(summaries: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Estimate one repeatable best latency from a fixed execution capture."""
    times = tuple(cast(float, summary["total_time"]) for summary in summaries)
    startup_mode = times[0] <= 0.85 * statistics.median(times[1:])
    estimate = times[0] if startup_mode else statistics.quantiles(times, n=20, method="inclusive")[0]
    result = dict(min(summaries, key=lambda summary: abs(cast(float, summary["total_time"]) - estimate)))
    result["total_time"] = estimate
    return result


def _capture(
    neff_path: Path, ntff_path: Path, executions: int, environment: dict[str, str], input_args: list[str]
) -> None:
    """Capture repeated executions using the same serialized input set."""
    args = [f"--neff={neff_path}", f"--num-exec={executions}", f"--session-file={ntff_path}", *input_args]
    _run(f"Neuron Explorer {ntff_path.stem}", ["neuron-explorer", "capture", *args], environment)


def benchmark_kernel(
    neff_path: Path, artifacts_dir: Path, lnc: int, visible_core: int, confirmation: bool, input_args: list[str]
) -> dict[str, object]:
    """Capture repeated NEFF executions and return their robust summary."""
    environment = _environment(lnc, visible_core)
    _capture(neff_path, artifacts_dir / "warmup.ntff", 20, environment, input_args)
    captures, execution_count = (3, 20) if confirmation else (5, 1)
    capture_paths = tuple(artifacts_dir / f"profile_capture_{index}.ntff" for index in range(captures))
    for path in capture_paths:
        _capture(neff_path, path, execution_count, environment, input_args)
    ntff_paths = tuple(path for capture in capture_paths for path in _execution_paths(capture, execution_count))
    with ThreadPoolExecutor(max_workers=8) as executor:
        summaries = tuple(executor.map(lambda path: _summary(neff_path, path, environment), ntff_paths))
    if confirmation:
        summaries = tuple(
            _confirmation_capture(summaries[index : index + execution_count])
            for index in range(0, len(summaries), execution_count)
        )
    return _aggregate(summaries)
