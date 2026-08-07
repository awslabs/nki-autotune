"""Single-kernel compile and profile pipeline for an installed Trn2 host."""

from __future__ import annotations

import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import FrameType

import ml_dtypes
import numpy as np

from nkigym.profile._benchmark import benchmark_kernel
from nkigym.profile._compile import compile_kernel
from nkigym.profile.types import ProfileConfig, ProfileResult

_COMPILE_TIMEOUT_S = 600
_COMPILER_LOG_FILE = "log-neuron-cc.txt"
_DTYPE_CACHE: dict[str, np.dtype] = {}


def _ensure_tools_on_path() -> None:
    """Expose the virtualenv and standard Neuron tool directories."""
    venv_bin = str(Path(sys.executable).parent)
    neuron_bin = "/opt/aws/neuron/bin"
    current = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys((venv_bin, neuron_bin, *current)))


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


def _capture_error(error: Exception) -> str:
    """Render one exception with its traceback for remote diagnostics."""
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _allocate_inputs(config: ProfileConfig) -> dict[str, np.ndarray]:
    """Allocate compile-time tensors from the declared input signatures."""
    return {name: np.zeros(shape, dtype=_resolve_dtype(dtype)) for name, (shape, dtype) in config.input_specs.items()}


def _compile_with_timeout(
    kernel_path: Path,
    func_name: str,
    inputs: dict[str, np.ndarray],
    compile_dir: Path,
    config: ProfileConfig,
    compiler_jobs: int | None,
) -> Path:
    """Compile one kernel while bounding compiler hangs."""
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_COMPILE_TIMEOUT_S)
    try:
        neff_path = compile_kernel(
            kernel_path, func_name, inputs, compile_dir, config.neuronx_cc_args, config.lnc, compiler_jobs
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    return neff_path


def _copy_compiler_log(compile_dir: Path, output_dir: Path) -> None:
    """Preserve the compiler diagnostic log when the compiler emitted one."""
    source = compile_dir / _COMPILER_LOG_FILE
    if source.is_file():
        shutil.copy2(source, output_dir / _COMPILER_LOG_FILE)


def run_profile(
    kernel_path: Path,
    func_name: str,
    config: ProfileConfig,
    output_dir: Path,
    visible_core: int,
    compiler_jobs: int | None,
) -> ProfileResult:
    """Compile and profile exactly one NKI kernel on the local Trn2 host."""
    _ensure_tools_on_path()
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(config.lnc)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    compile_s = 0.0
    profile_s = 0.0
    summary: dict[str, object] | None = None
    error_text: str | None = None
    with tempfile.TemporaryDirectory(prefix="nkigym-profile-") as raw_work_dir:
        compile_dir = Path(raw_work_dir) / "compiler"
        profile_neff_path = output_dir / "file.neff"
        compile_started = time.monotonic()
        try:
            inputs = _allocate_inputs(config)
            neff_path = _compile_with_timeout(kernel_path, func_name, inputs, compile_dir, config, compiler_jobs)
            shutil.copy2(neff_path, profile_neff_path)
        except Exception as error:
            error_text = _capture_error(error)
        compile_s = time.monotonic() - compile_started
        _copy_compiler_log(compile_dir, output_dir)
        if error_text is None:
            profile_started = time.monotonic()
            try:
                summary = benchmark_kernel(profile_neff_path, output_dir, config.lnc, visible_core)
            except Exception as error:
                error_text = _capture_error(error)
            profile_s = time.monotonic() - profile_started
    return ProfileResult(
        profiler_summary=summary,
        error=error_text,
        elapsed_s=time.monotonic() - started,
        compile_s=compile_s,
        profile_s=profile_s,
    )


__all__ = ["run_profile"]
