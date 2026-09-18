"""One-call SSH profiling for a standalone NKI kernel."""

import json
import math
import shutil
from pathlib import Path
from typing import cast

import numpy as np

from nkigym.profile.protocol import parse_result, request_payload
from nkigym.profile.ssh import SSHTransportError, profile_over_ssh
from nkigym.profile.types import InputSpecs, ProfileConfig, ProfileMetrics, ProfileResult, _physical_numpy_dtype


def _adapt_exact_inputs(input_specs: InputSpecs, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate logical exact inputs and cast them to the generated physical ABI."""
    if set(inputs) != set(input_specs):
        raise ValueError("exact inputs must match input_specs")
    adapted: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        value = inputs[name]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"input {name!r} must be an ndarray with shape {shape}")
        adapted[name] = value.astype(_physical_numpy_dtype(dtype_name), copy=False)
    return adapted


def _read_execution_outputs(directory: Path) -> tuple[np.ndarray, ...]:
    """Load typed output arrays returned by the unified Trn2 worker."""
    metadata_path = directory / "outputs.json"
    if not metadata_path.is_file():
        raise RuntimeError("SSH execution returned no outputs.json")
    metadata_items = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_items, list):
        raise ValueError("SSH execution outputs must be an array")
    outputs: list[np.ndarray] = []
    for metadata in metadata_items:
        if not isinstance(metadata, dict):
            raise ValueError("SSH execution output metadata must be objects")
        shape = tuple(cast(list[int], metadata["shape"]))
        dtype = _physical_numpy_dtype(dtype_name := cast(str, metadata["dtype"]))
        path = directory / cast(str, metadata["file"])
        if not path.is_file():
            raise RuntimeError(f"SSH execution returned no {path}")
        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"output {path.name!r} has {path.stat().st_size} bytes, expected {expected_bytes}")
        output = np.fromfile(path, dtype=dtype).reshape(shape)
        outputs.append(output.astype(np.float32, copy=False) if "float" in dtype_name else output)
    if not outputs:
        raise RuntimeError("SSH execution returned no output tensors")
    return tuple(outputs)


def profile_metrics(
    host: str,
    kernel: str,
    func_name: str,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    neuronx_cc_args: tuple[str, ...] = (),
    lnc: int = 1,
    timeout_s: int = 1800,
    confirmation: bool = False,
    inputs: dict[str, np.ndarray] | None = None,
) -> ProfileMetrics:
    """Profile one NKI kernel, optionally executing exact inputs and returning outputs."""
    if not kernel.strip() or not func_name.isidentifier():
        raise ValueError("kernel source and function name must be valid")
    if inputs is not None:
        inputs = _adapt_exact_inputs(input_specs, inputs)
    config = ProfileConfig(input_specs=input_specs, neuronx_cc_args=neuronx_cc_args, lnc=lnc, confirmation=confirmation)
    output_dir = Path(cache_dir).expanduser().resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    input_path = output_dir / ("request" if inputs is not None else "kernel.py")
    input_path.mkdir(parents=True) if inputs is not None else output_dir.mkdir(parents=True)
    kernel_path = input_path / "kernel.py" if inputs is not None else input_path
    kernel_path.write_text(kernel, encoding="utf-8")
    if inputs is not None:
        (inputs_dir := input_path / "inputs").mkdir()
        for index, name in enumerate(input_specs):
            inputs[name].tofile(inputs_dir / f"input_{index:03d}.bin")
    request_path = input_path / "request.json" if inputs is not None else output_dir / "request.json"
    request_path.write_text(json.dumps(request_payload(func_name, config), indent=2) + "\n", encoding="utf-8")
    try:
        profile_over_ssh(host, input_path, request_path, output_dir, timeout_s)
    except SSHTransportError as error:
        raise RuntimeError(f"SSH profile failed for {host}: {error}\n{error.log[-3000:]}") from error
    result = parse_result(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    compiler_log_path = output_dir / "log-neuron-cc.txt"
    compiler_log = compiler_log_path.read_text(encoding="utf-8") if compiler_log_path.is_file() else ""
    outputs = () if inputs is None or result.error is not None else _read_execution_outputs(output_dir)
    return _metrics(result, compiler_log, outputs)


def _metrics(result: ProfileResult, compiler_log: str, outputs: tuple[np.ndarray, ...] = ()) -> ProfileMetrics:
    """Extract core metrics and preserve the raw Neuron Explorer summary."""
    if result.error is not None:
        raise RuntimeError(_profile_failure_message(result.error, compiler_log))
    if (summary := result.profiler_summary) is None:
        raise RuntimeError("profiler returned no summary")
    mfu, latency = (_summary_number(summary, name) for name in ("mfu_estimated_percent", "total_time"))
    if mfu is None or latency is None:
        missing = "mfu_estimated_percent" if mfu is None else "total_time"
        raise RuntimeError(f"profiler summary has no valid {missing}")
    return ProfileMetrics(
        mfu_percent=mfu * 100.0, latency_ms=latency * 1000.0, profiler_summary=summary, outputs=outputs
    )


def _summary_number(summary: dict[str, object], name: str) -> float | None:
    """Read one finite non-negative summary value, allowing negative sentinels."""
    raw_value = summary.get(name)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise RuntimeError(f"profiler summary metric {name} must be numeric")
    numeric_value = float(raw_value)
    if not math.isfinite(numeric_value):
        raise RuntimeError(f"profiler summary metric {name} must be finite")
    return numeric_value if numeric_value >= 0 else None


def _profile_failure_message(error: str, compiler_log: str) -> str:
    """Prefer a specific compiler diagnostic over a wrapper traceback."""
    lines = compiler_log.splitlines()
    markers = ("[NCC_", "Out of memory", "Allocated memory out of bound")
    diagnostic = next((line for line in reversed(lines) if any(marker in line for marker in markers)), "")
    diagnostic = diagnostic or next((line for line in reversed(lines) if " ERROR " in line), "") or error
    return diagnostic[-1000:]
