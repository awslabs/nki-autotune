"""NKI kernel compilation pipeline.

Worker-only module — imports nki.compiler at top level. Not safe to
import on the coordinator machine.
"""

import contextlib
import importlib.util
import os
import re
import signal
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from nki.compiler.driver import CompileOptions, compile_bir_to_neff, compile_to_bir
from nki.compiler.frontend import TracerFrontend
from nki.framework.kernel import Kernel

from autotune.runner.types import CompileResult, capture_error, ensure_venv_on_path, resolve_dtype


def load_kernel(nki_path: str, func_name: str) -> Any:
    """Load a kernel function from an NKI source file.

    Args:
        nki_path: Path to the kernel .py file.
        func_name: Name of the function to extract.

    Returns:
        The kernel function object.
    """
    module_name = f"nki_kernel_{Path(nki_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, nki_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {nki_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def inline_scalars(nki_path: str, scalar_inputs: dict[str, Any]) -> str:
    """Rewrite NKI kernel source to inline scalar params as local constants.

    neuronxcc cannot handle 0-d tensor inputs. This rewrites the kernel
    source to remove scalar params from the function signature and define
    them as local variables in the function body.

    Returns:
        Path to the rewritten kernel file.
    """
    source = Path(nki_path).read_text()
    for name in scalar_inputs:
        source = re.sub(rf"(def \w+\([^)]*),\s*{name}\s*([,)])", rf"\1\2", source)
        source = re.sub(rf"(def \w+\(\s*){name}\s*,\s*", rf"\1", source)

    def_match = re.search(r"(def \w+\([^)]*\):)", source)
    if def_match:
        insert_pos = def_match.end()
        assignments = ""
        for name, val in scalar_inputs.items():
            assignments += f"\n    {name} = {val!r}"
        source = source[:insert_pos] + assignments + source[insert_pos:]

    rewritten_path = str(Path(nki_path).with_suffix(".inlined.py"))
    Path(rewritten_path).write_text(source)
    return rewritten_path


def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler that raises TimeoutError for compilation."""
    raise TimeoutError("Compilation timed out after 10 minutes")


def init_compile_worker() -> None:
    """Silence compiler diagnostic noise in worker subprocesses.

    Also removes NEURON_RT_VISIBLE_CORES so compile workers don't
    try to allocate Neuron cores (set by the benchmark prewarm thread).
    """
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    ensure_venv_on_path()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _separate_inputs(input_tensors: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split input_tensors into tensor and scalar dicts."""
    tensors: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    for name, val in input_tensors.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
            tensors[name] = val
        else:
            scalars[name] = val
    return tensors, scalars


@contextlib.contextmanager
def _capture_fd_stderr() -> Generator[str, None, None]:
    """Redirect fd 2 to a temp file for C-level stderr capture.

    Yields the path. Caller reads before exiting the with block.
    """
    fd, path = tempfile.mkstemp(suffix=".stderr")
    saved = os.dup(2)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield path
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        Path(path).unlink(missing_ok=True)


def _run_compiler(kernel: Kernel, tensor_inputs: dict[str, np.ndarray], output_name: str, opts: CompileOptions) -> None:
    """Run compile_to_bir and compile_bir_to_neff with stderr capture.

    Captures C-level stderr so compiler diagnostics (e.g. ``Out of
    memory in sbuf``) appear in the Python exception on failure.
    ``compile_bir_to_neff`` swallows neuronx-cc failures and returns
    a ``CompiledKernel`` with ``neuronx_cc_error`` set; surface that
    as an exception so the caller sees the real diagnostic instead
    of a generic "NEFF file not found".
    """
    frontend = TracerFrontend()
    with _capture_fd_stderr() as stderr_path:
        try:
            bir, cr = compile_to_bir(
                kernel, frontend=frontend, inputs=tensor_inputs, compile_opts=opts, output_names=[output_name]
            )
            input_arrays = [np.zeros(s.shape, dtype=np.dtype(s.dtype)) for s in cr.input_specs]
            compiled = compile_bir_to_neff(
                opts,
                bir,
                input_arrays,
                cr.argument_names,
                cr.output_names,
                input_output_aliases=cr.input_output_aliases,
            )
            if compiled.neuronx_cc_error:
                stderr_content = Path(stderr_path).read_text().strip()
                detail = (
                    f"{compiled.neuronx_cc_error}\n{stderr_content}" if stderr_content else compiled.neuronx_cc_error
                )
                raise RuntimeError(detail)
        except Exception as exc:
            stderr_content = Path(stderr_path).read_text().strip()
            if stderr_content:
                raise RuntimeError(f"{exc}\n{stderr_content}") from exc
            raise


def compile_nki_kernel(
    nki_path: str,
    func_name: str,
    input_tensors: dict[str, Any],
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    output_dir: str,
    neuronx_cc_args: tuple[str, ...],
) -> str:
    """Compile an NKI kernel file to NEFF via nki.compiler.

    ``neuronx_cc_args`` is forwarded to the MLIR pass pipeline via
    ``CompileOptions.set_pipeline_options(*args)``. Hand-allocated
    kernels (e.g. nkilib's ``attention_cte``) need
    ``("enable-linear-scan-allocation=false",
    "enable-instruction-scheduling=false")`` so neuronx-cc's scheduler
    doesn't reshuffle their explicit SBUF/PSUM address lifetimes.

    Returns:
        Path to the compiled NEFF file.
    """
    tempfile.tempdir = output_dir
    os.makedirs(tempfile.tempdir, exist_ok=True)

    tensor_inputs, scalar_inputs = _separate_inputs(input_tensors)
    if scalar_inputs:
        nki_path = inline_scalars(nki_path, scalar_inputs)
    kernel_func = load_kernel(nki_path, func_name)

    kernel = Kernel(kernel_func)
    neff_file = os.path.join(output_dir, "file.neff")
    opts = CompileOptions(target="trn2", lnc=1, output_path=neff_file, artifacts_dir=output_dir)
    if neuronx_cc_args:
        opts = opts.set_pipeline_options(*neuronx_cc_args)
    _run_compiler(kernel, tensor_inputs, output_name, opts)

    if not os.path.isfile(neff_file):
        raise RuntimeError(f"NEFF file not found at: {neff_file}")
    return neff_file


def compile_one(
    kernel_name: str,
    nki_path: str,
    func_name: str,
    input_shapes: dict[str, tuple[int, ...]],
    input_dtype_name: str,
    output_name: str,
    output_shape: tuple[int, ...],
    output_dtype_name: str,
    compile_dir: str,
    scalar_params: dict[str, float],
    neuronx_cc_args: tuple[str, ...],
) -> CompileResult:
    """Top-level picklable worker for parallel NKI compilation.

    Args:
        kernel_name: Identifier for this kernel variant.
        nki_path: Path to the kernel source file.
        func_name: Name of the @nki.jit function.
        input_shapes: Map of param name to shape.
        input_dtype_name: Dtype name for input tensors.
        output_name: Name of the output tensor.
        output_shape: Shape of the output tensor.
        output_dtype_name: Dtype name for the output tensor.
        compile_dir: Directory for compilation artifacts.
        scalar_params: Map of scalar param names to values.
        neuronx_cc_args: Extra MLIR pass-pipeline flags for
            ``CompileOptions.set_pipeline_options``.
    """
    neff_path = ""
    error = ""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        in_dtype = resolve_dtype(input_dtype_name)
        out_dtype = resolve_dtype(output_dtype_name)
        input_tensors: dict[str, Any] = {name: np.zeros(shape, dtype=in_dtype) for name, shape in input_shapes.items()}
        for name, value in scalar_params.items():
            input_tensors[name] = value
        neff_path = compile_nki_kernel(
            nki_path, func_name, input_tensors, output_name, output_shape, out_dtype, compile_dir, neuronx_cc_args
        )
    except Exception as e:
        error = capture_error(e)
    finally:
        signal.alarm(0)
    return CompileResult(kernel_name=kernel_name, nki_path=nki_path, neff_path=neff_path, error=error)
