"""NKI compilation used only by the Trn2 profile worker."""

import contextlib
import importlib.util
import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import nki.compiler._internal.ir as mlir_ir
import numpy as np

_compiler = importlib.import_module("nki.compiler.driver")
_frontend = importlib.import_module("nki.compiler.frontend")
_kernel = importlib.import_module("nki.framework.kernel")


class _StructuredLoopFrontend(_frontend.TracerFrontend):
    """Preserve explicit multi-iteration loops through NKI lowering."""

    def _compile(self, *args: Any, **kwargs: Any) -> Any:
        """Mark constant-bound loops before the standard frontend verifier runs."""
        result = super()._compile(*args, **kwargs)

        def preserve_loop(operation: mlir_ir.Operation) -> Any:
            """Prevent expansion of explicit loops with a non-unit upper bound."""
            if operation.name == "scf.for":
                upper = operation.operands[1].owner
                if isinstance(upper, mlir_ir.Operation) and upper.name == "arith.constant":
                    if mlir_ir.IntegerAttr(upper.attributes["value"]).value > 1:
                        operation.attributes["nki.no_unroll"] = mlir_ir.UnitAttr.get(context=operation.context)
            return getattr(mlir_ir, "WalkResult").ADVANCE

        result.module.operation.walk(preserve_loop)
        return result


def load_kernel(kernel_path: Path, func_name: str) -> object:
    """Load one NKI kernel without wrapping an existing JIT kernel twice."""
    spec = importlib.util.spec_from_file_location(f"nkigym_profile_{kernel_path.stem}", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, func_name)
    return kernel if isinstance(kernel, _kernel.Kernel) else _kernel.Kernel(kernel)


@contextlib.contextmanager
def _capture_stderr() -> Generator[Path, None, None]:
    """Capture Python and native compiler stderr in one temporary file."""
    with tempfile.NamedTemporaryFile() as stream, os.fdopen(os.dup(2), "wb") as saved:
        os.dup2(stream.fileno(), 2)
        try:
            yield Path(stream.name)
        finally:
            os.dup2(saved.fileno(), 2)


def _run_compiler(kernel: object, inputs: dict[str, np.ndarray], options: object) -> tuple[Any, ...]:
    """Trace the NKI function, lower it to NEFF, and return output specifications."""
    with _capture_stderr() as stderr_path:
        try:
            frontend = _StructuredLoopFrontend()
            bir = _compiler.compile_to_bir(kernel, frontend=frontend, inputs=inputs, compile_opts=options)
            input_specs, output_specs = bir.descriptor.input_specs, bir.descriptor.output_specs
            input_arrays = [inputs[spec.name].astype(np.dtype(spec.dtype), copy=False) for spec in input_specs]
            _compiler.compile_bir_to_neff(
                options, bir, input_arrays, [spec.name for spec in input_specs], [spec.name for spec in output_specs]
            )
            return tuple(output_specs)
        except Exception as error:
            if stderr := stderr_path.read_text(encoding="utf-8").strip():
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
) -> tuple[Path, tuple[Any, ...]]:
    """Compile one NKI source file for Trn2 and return its NEFF and output specifications."""
    if compiler_jobs is not None and compiler_jobs <= 0:
        raise ValueError("compiler jobs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    neff_path = output_dir / "file.neff"
    backend_args = () if compiler_jobs is None else (f"--jobs={compiler_jobs}",)
    options = _compiler.CompileOptions(
        target="trn2", lnc=lnc, output_path=str(neff_path), artifacts_dir=str(output_dir), neuronx_cc_args=backend_args
    ).set_pipeline_options(*neuronx_cc_args)
    previous_tempdir, tempfile.tempdir = tempfile.tempdir, str(output_dir)
    try:
        output_specs = _run_compiler(load_kernel(kernel_path, func_name), inputs, options)
    finally:
        tempfile.tempdir = previous_tempdir
    if not neff_path.is_file():
        raise RuntimeError(f"compiler returned without creating {neff_path}")
    return neff_path, output_specs
