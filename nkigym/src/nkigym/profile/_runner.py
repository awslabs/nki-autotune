"""Single-kernel compile and profile pipeline for an installed Trn2 host."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Any

import ml_dtypes
import numpy as np

from nkigym.profile._benchmark import benchmark_kernel
from nkigym.profile._compile import compile_kernel
from nkigym.profile.types import ProfileConfig, ProfileResult

_COMPILE_TIMEOUT_S = 600
_DTYPE_CACHE: dict[str, np.dtype] = {}
_OUTPUT_PATTERN = re.compile(r'saved output "([^"]+)" as "([^"]+)"')


def _timeout_handler(signum: int, frame: FrameType | None) -> None:
    """Abort a compiler invocation that exceeds the fixed host timeout."""
    raise TimeoutError(f"NKI compilation exceeded {_COMPILE_TIMEOUT_S} seconds")


def _resolve_dtype(name: str) -> np.dtype:
    """Resolve a NumPy dtype name, including ``bfloat16``."""
    if name not in _DTYPE_CACHE:
        try:
            dtype = np.dtype(name)
        except TypeError:
            dtype = np.dtype(getattr(ml_dtypes, name))
        _DTYPE_CACHE[name] = dtype
    return _DTYPE_CACHE[name]


def _compile_with_timeout(
    kernel_path: Path,
    func_name: str,
    inputs: dict[str, np.ndarray],
    compile_dir: Path,
    config: ProfileConfig,
    compiler_jobs: int | None,
) -> tuple[Path, tuple[Any, ...]]:
    """Compile one kernel while bounding compiler hangs."""
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_COMPILE_TIMEOUT_S)
    try:
        return compile_kernel(
            kernel_path, func_name, inputs, compile_dir, config.neuronx_cc_args, config.lnc, compiler_jobs
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _capture_outputs(
    neff_path: Path,
    inputs: dict[str, np.ndarray],
    output_specs: tuple[Any, ...],
    work_dir: Path,
    output_dir: Path,
    lnc: int,
    visible_core: int,
) -> None:
    """Execute one exact-input inference and retain typed output files."""
    input_args: list[str] = []
    input_dir = work_dir / "inputs"
    input_dir.mkdir()
    for index, (name, value) in enumerate(inputs.items()):
        path = input_dir / f"input_{index:03d}.bin"
        value.tofile(path)
        input_args.extend((name, str(path)))
    environment = dict(os.environ)
    environment["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    environment["NEURON_RT_VISIBLE_CORES"] = str(visible_core)
    command = [
        "neuron-explorer",
        "capture",
        "--disable-profile",
        "--save-output",
        "--num-exec=1",
        "--profile-nth-exec=0",
        "--neff",
        str(neff_path),
        *input_args,
    ]
    completed = subprocess.run(command, cwd=work_dir, env=environment, text=True, capture_output=True, check=False)
    log = completed.stdout + completed.stderr
    (output_dir / "capture.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"neuron-explorer capture failed with exit {completed.returncode}\n{log}")
    saved = _OUTPUT_PATTERN.findall(log)
    saved_by_name = {name: filename for name, filename in saved}
    outputs: list[dict[str, object]] = []
    for index, spec in enumerate(output_specs):
        name, shape, dtype = str(spec.name), tuple(spec.shape), _resolve_dtype(str(spec.dtype))
        raw_name = saved_by_name.get(name, saved[index][1] if index < len(saved) else "")
        raw_path = Path(raw_name) if Path(raw_name).is_absolute() else work_dir / raw_name
        if not raw_name or not raw_path.is_file():
            raise RuntimeError(f"neuron-explorer returned no output file for {name!r}")
        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if raw_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"output {name!r} has {raw_path.stat().st_size} bytes, expected {expected_bytes}")
        file_name = f"output_{index:03d}.bin"
        shutil.copy2(raw_path, output_dir / file_name)
        outputs.append({"name": name, "shape": list(shape), "dtype": str(dtype), "file": file_name})
    (output_dir / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")


def run_profile(
    kernel_path: Path,
    func_name: str,
    config: ProfileConfig,
    output_dir: Path,
    visible_core: int,
    compiler_jobs: int | None,
    inputs: dict[str, np.ndarray] | None = None,
    capture_outputs: bool = False,
) -> ProfileResult:
    """Compile one NKI kernel, optionally capture exact outputs, and profile the same NEFF."""
    entries = (str(Path(sys.executable).parent), "/opt/aws/neuron/bin", *os.environ.get("PATH", "").split(os.pathsep))
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(entries))
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(config.lnc)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    profile_s = 0.0
    summary: dict[str, object] | None = None
    error_text: str | None = None
    with tempfile.TemporaryDirectory(prefix="nkigym-profile-") as raw_work_dir:
        work_dir = Path(raw_work_dir)
        compile_dir = Path(raw_work_dir) / "compiler"
        profile_neff_path = output_dir / "file.neff"
        compile_started = time.monotonic()
        try:
            resolved_inputs = (
                {
                    name: np.zeros(shape, dtype=_resolve_dtype(dtype))
                    for name, (shape, dtype) in config.input_specs.items()
                }
                if inputs is None
                else inputs
            )
            neff_path, output_specs = _compile_with_timeout(
                kernel_path, func_name, resolved_inputs, compile_dir, config, compiler_jobs
            )
            shutil.copy2(neff_path, profile_neff_path)
            if capture_outputs:
                _capture_outputs(
                    profile_neff_path, resolved_inputs, output_specs, work_dir, output_dir, config.lnc, visible_core
                )
        except Exception:
            error_text = traceback.format_exc()
        compile_s = time.monotonic() - compile_started
        if (compiler_log := compile_dir / "log-neuron-cc.txt").is_file():
            shutil.copy2(compiler_log, output_dir / "log-neuron-cc.txt")
        if error_text is None:
            profile_started = time.monotonic()
            try:
                summary = benchmark_kernel(profile_neff_path, output_dir, config.lnc, visible_core, config.confirmation)
            except Exception:
                error_text = traceback.format_exc()
            profile_s = time.monotonic() - profile_started
    return ProfileResult(
        profiler_summary=summary,
        error=error_text,
        elapsed_s=time.monotonic() - started,
        compile_s=compile_s,
        profile_s=profile_s,
    )
