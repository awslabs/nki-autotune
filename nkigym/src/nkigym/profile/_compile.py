"""NKI compilation used only by the installed Trn2 profile worker."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from nki.compiler.driver import CompileOptions, compile_bir_to_neff, compile_to_bir
from nki.compiler.frontend import TracerFrontend
from nki.framework.kernel import Kernel


def load_kernel(kernel_path: Path, func_name: str) -> Any:
    """Load one NKI function from a standalone source file."""
    module_name = f"nkigym_profile_{kernel_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


@contextlib.contextmanager
def _capture_stderr() -> Generator[Path, None, None]:
    """Capture Python and native compiler stderr in one temporary file."""
    descriptor, raw_path = tempfile.mkstemp(suffix=".stderr")
    path = Path(raw_path)
    saved_descriptor = os.dup(2)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    try:
        yield path
    finally:
        os.dup2(saved_descriptor, 2)
        os.close(saved_descriptor)
        path.unlink(missing_ok=True)


def _run_compiler(kernel: Kernel, inputs: dict[str, np.ndarray], options: CompileOptions) -> None:
    """Trace the NKI function and lower it to a NEFF artifact."""
    with _capture_stderr() as stderr_path:
        try:
            bir = compile_to_bir(kernel, frontend=TracerFrontend(), inputs=inputs, compile_opts=options)
            input_specs = bir.descriptor.input_specs
            output_specs = bir.descriptor.output_specs
            input_arrays = [np.zeros(spec.shape, dtype=np.dtype(spec.dtype)) for spec in input_specs]
            compile_bir_to_neff(
                options, bir, input_arrays, [spec.name for spec in input_specs], [spec.name for spec in output_specs]
            )
        except Exception as error:
            stderr = stderr_path.read_text(encoding="utf-8").strip()
            if stderr:
                raise RuntimeError(f"{error}\n{stderr}") from error
            raise


def compile_kernel(
    kernel_path: Path,
    func_name: str,
    inputs: dict[str, np.ndarray],
    output_dir: Path,
    neuronx_cc_args: tuple[str, ...],
    lnc: int,
    compiler_jobs: int | None,
) -> Path:
    """Compile one NKI source file for Trn2 and return its NEFF path."""
    if compiler_jobs is not None and compiler_jobs <= 0:
        raise ValueError("compiler jobs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    neff_path = output_dir / "file.neff"
    backend_args = () if compiler_jobs is None else (f"--jobs={compiler_jobs}",)
    options = CompileOptions(
        target="trn2", lnc=lnc, output_path=str(neff_path), artifacts_dir=str(output_dir), neuronx_cc_args=backend_args
    )
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(output_dir)
    try:
        if neuronx_cc_args:
            options = options.set_pipeline_options(*neuronx_cc_args)
        kernel = Kernel(load_kernel(kernel_path, func_name))
        _run_compiler(kernel, inputs, options)
    finally:
        tempfile.tempdir = previous_tempdir
    if not neff_path.is_file():
        raise RuntimeError(f"compiler returned without creating {neff_path}")
    return neff_path
