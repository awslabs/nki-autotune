"""One-call SSH profiling for a standalone NKI kernel."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from nkigym.profile.protocol import batch_request_payload, parse_batch_result, parse_result, request_payload
from nkigym.profile.ssh import SSHTransportError, profile_batch_over_ssh, profile_over_ssh
from nkigym.profile.types import (
    BatchProfileJob,
    BatchProfileRequest,
    InputSpecs,
    ProfileConfig,
    ProfileMetrics,
    ProfileRequest,
    ProfileResult,
)

_RESULT_FILE = "result.json"
_BATCH_RESULT_FILE = "batch_result.json"
_COMPILER_LOG_FILE = "log-neuron-cc.txt"
_TRANSPORT_LOG_FILE = "transport.log"


def profile_metrics(
    host: str,
    kernel: str,
    func_name: str,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    neuronx_cc_args: tuple[str, ...] = (),
    lnc: int = 1,
    timeout_s: int = 1800,
) -> ProfileMetrics:
    """Profile one NKI kernel and return core metrics plus the raw summary."""
    if not kernel.strip():
        raise ValueError("kernel source must not be empty")
    if not func_name.isidentifier():
        raise ValueError(f"invalid kernel function name {func_name!r}")
    config = ProfileConfig(input_specs=input_specs, neuronx_cc_args=neuronx_cc_args, lnc=lnc)
    output_dir = Path(cache_dir).expanduser().resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    kernel_path = output_dir / "kernel.py"
    request_path = output_dir / "request.json"
    kernel_path.write_text(kernel, encoding="utf-8")
    request_path.write_text(json.dumps(request_payload(func_name, config), indent=2) + "\n", encoding="utf-8")
    try:
        transport_log = profile_over_ssh(host, kernel_path, request_path, output_dir, timeout_s)
    except SSHTransportError as error:
        (output_dir / _TRANSPORT_LOG_FILE).write_text(error.log, encoding="utf-8")
        raise RuntimeError(f"SSH profile failed for {host}: {error}\n{error.log[-3000:]}") from error
    (output_dir / _TRANSPORT_LOG_FILE).write_text(transport_log, encoding="utf-8")
    result_path = output_dir / _RESULT_FILE
    if not result_path.is_file():
        raise RuntimeError(f"SSH profile returned no {_RESULT_FILE}: {transport_log[-3000:]}")
    result = parse_result(json.loads(result_path.read_text(encoding="utf-8")))
    compiler_log_path = output_dir / _COMPILER_LOG_FILE
    compiler_log = compiler_log_path.read_text(encoding="utf-8") if compiler_log_path.is_file() else ""
    return _metrics(result, compiler_log)


def profile_many(
    host: str,
    kernels: dict[str, str],
    func_name: str,
    input_specs: InputSpecs,
    cache_dir: str | Path,
    neuronx_cc_args: tuple[str, ...] = (),
    lnc: int = 1,
    max_workers: int = 8,
    timeout_s: int = 1800,
    required_successes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Profile labeled kernels concurrently and return one aggregate report."""
    empty_kernels = [label for label, kernel in kernels.items() if not kernel.strip()]
    if empty_kernels:
        raise ValueError(f"{empty_kernels[0]}: kernel source must not be empty")
    config = ProfileConfig(input_specs=input_specs, neuronx_cc_args=neuronx_cc_args, lnc=lnc)
    requests = tuple(
        BatchProfileJob(label=label, request=ProfileRequest(func_name=func_name, config=config)) for label in kernels
    )
    batch_request = BatchProfileRequest(jobs=requests, max_workers=max_workers)
    missing_required = sorted(set(required_successes) - kernels.keys())
    if missing_required:
        raise ValueError(f"required profile labels are not present: {missing_required}")
    output_dir = Path(cache_dir).expanduser().resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    for (label, kernel), request in zip(kernels.items(), requests, strict=True):
        job_dir = output_dir / label
        job_dir.mkdir()
        (job_dir / "kernel.py").write_text(kernel, encoding="utf-8")
        (job_dir / "request.json").write_text(
            json.dumps(request_payload(request.request.func_name, request.request.config), indent=2) + "\n",
            encoding="utf-8",
        )
    request_path = output_dir / "request.json"
    request_path.write_text(json.dumps(batch_request_payload(batch_request), indent=2) + "\n", encoding="utf-8")
    try:
        transport_log = profile_batch_over_ssh(host, output_dir, request_path, output_dir, timeout_s)
    except SSHTransportError as error:
        (output_dir / _TRANSPORT_LOG_FILE).write_text(error.log, encoding="utf-8")
        raise RuntimeError(f"SSH batch profile failed for {host}: {error}\n{error.log[-3000:]}") from error
    (output_dir / _TRANSPORT_LOG_FILE).write_text(transport_log, encoding="utf-8")
    batch_result_path = output_dir / _BATCH_RESULT_FILE
    if not batch_result_path.is_file():
        raise RuntimeError(f"SSH batch profile returned no {_BATCH_RESULT_FILE}: {transport_log[-3000:]}")
    elapsed_s, workers, labels = parse_batch_result(json.loads(batch_result_path.read_text(encoding="utf-8")))
    expected_labels = tuple(kernels)
    if labels != expected_labels:
        raise RuntimeError(f"SSH batch profile returned labels {labels}, expected {expected_labels}")
    measurements: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}
    for label in labels:
        job_dir = output_dir / label
        result_path = job_dir / _RESULT_FILE
        if not result_path.is_file():
            raise RuntimeError(f"SSH batch profile returned no {label}/{_RESULT_FILE}")
        result = parse_result(json.loads(result_path.read_text(encoding="utf-8")))
        compiler_log_path = job_dir / _COMPILER_LOG_FILE
        compiler_log = compiler_log_path.read_text(encoding="utf-8") if compiler_log_path.is_file() else ""
        if result.error is not None:
            failures[label] = _profile_failure_message(result.error, compiler_log)
        else:
            measurements[label] = {
                **_metrics(result, compiler_log).as_dict(),
                "elapsed_s": result.elapsed_s,
                "compile_s": result.compile_s,
                "profile_s": result.profile_s,
            }
    results = {"wall_time_s": elapsed_s, "workers": workers, "successes": measurements, "failures": failures}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failed_required = [label for label in required_successes if label not in measurements]
    if failed_required:
        label = failed_required[0]
        raise RuntimeError(f"required profile {label!r} failed: {failures[label]}")
    return {
        "successes": measurements,
        "failure_count": len(failures),
        "wall_time_s": elapsed_s,
        "workers": workers,
        "results": f"{output_dir.name}/results.json",
    }


def _metrics(result: ProfileResult, compiler_log: str) -> ProfileMetrics:
    """Extract core metrics and preserve the raw Neuron Explorer summary."""
    if result.error is not None:
        raise RuntimeError(_profile_failure_message(result.error, compiler_log))
    summary = result.profiler_summary
    if summary is None:
        raise RuntimeError("profiler returned no summary")
    mfu = _summary_number(summary, "mfu_estimated_percent")
    latency = _summary_number(summary, "total_time")
    if mfu is None:
        raise RuntimeError("profiler summary has no valid mfu_estimated_percent")
    if latency is None:
        raise RuntimeError("profiler summary has no valid total_time")
    return ProfileMetrics(mfu_percent=mfu * 100.0, latency_ms=latency * 1000.0, profiler_summary=summary)


def _summary_number(summary: dict[str, object], name: str) -> float | None:
    """Read one finite non-negative summary value, allowing negative sentinels."""
    raw_value = summary.get(name)
    value: float | None = None
    if raw_value is not None:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise RuntimeError(f"profiler summary metric {name} must be numeric")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise RuntimeError(f"profiler summary metric {name} must be finite")
        if numeric_value >= 0:
            value = numeric_value
    return value


def _profile_failure_message(error: str, compiler_log: str) -> str:
    """Prefer a specific compiler diagnostic over a wrapper traceback."""
    lines = compiler_log.splitlines()
    diagnostic = next(
        (
            line
            for line in reversed(lines)
            if "[NCC_" in line or "Out of memory" in line or "Allocated memory out of bound" in line
        ),
        "",
    )
    if not diagnostic:
        diagnostic = next((line for line in reversed(lines) if " ERROR " in line), "")
    if not diagnostic:
        diagnostic = error
    return diagnostic[-1000:]


__all__ = ["profile_many", "profile_metrics"]
