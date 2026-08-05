"""One-call SSH profiling for a standalone NKI kernel."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nkigym.profile.protocol import parse_result, request_payload
from nkigym.profile.ssh import SSHTransportError, profile_over_ssh
from nkigym.profile.types import ProfileConfig, ProfileResult

_RESULT_FILE = "result.json"
_COMPILER_LOG_FILE = "log-neuron-cc.txt"
_TRANSPORT_LOG_FILE = "transport.log"


def profile(
    host: str,
    kernel: str,
    func_name: str,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    neuronx_cc_args: tuple[str, ...] = (),
    lnc: int = 1,
    timeout_s: int = 1800,
) -> tuple[float, float]:
    """Profile one NKI kernel over SSH.

    Args:
        host: SSH destination for an installed profile host.
        kernel: Complete standalone ``kernel.py`` source.
        func_name: NKI function to load from the source.
        input_specs: Input names mapped to ``(shape, dtype)``.
        cache_dir: Local destination for the kernel and profiler artifacts.
        neuronx_cc_args: Optional Neuron compiler pipeline arguments.
        lnc: Logical NeuronCore count, either 1 or 2.
        timeout_s: Total SSH operation timeout in seconds.

    Returns:
        A ``(mfu_percent, latency_ms)`` tuple.
    """
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


def _metrics(result: ProfileResult, compiler_log: str) -> tuple[float, float]:
    """Extract MFU percent and latency milliseconds or raise the kernel failure."""
    if result.error is not None:
        raise RuntimeError(_profile_failure_message(result.error, compiler_log))
    summary = result.profiler_summary
    if summary is None:
        raise RuntimeError("profiler returned no summary")
    mfu = summary.get("mfu_estimated_percent")
    latency = summary.get("total_time")
    if not isinstance(mfu, (int, float)) or isinstance(mfu, bool) or float(mfu) < 0:
        raise RuntimeError("profiler summary has no valid mfu_estimated_percent")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or float(latency) < 0:
        raise RuntimeError("profiler summary has no valid total_time")
    return float(mfu) * 100.0, float(latency) * 1000.0


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


__all__ = ["profile"]
