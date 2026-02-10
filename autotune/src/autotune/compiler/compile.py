# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import types
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from neuronxcc.nki_standalone import NKI_IR_VERSION, compile_nki_ir_kernel_to_neff
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import CompiledKernel

from autotune.types import INPUT_TENSORS_DTYPE, KERNEL_DTYPE, KERNEL_KWARGS_DTYPE
from autotune.utils import split_file_info


@dataclass
class TensorStub:
    """Describes an output tensor for the compilation API.

    Attributes:
        shape: The shape of the tensor as a tuple of integers.
        dtype: The numpy data type of the tensor.
        name: The name identifier for the tensor.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    name: str


def resolve_kernel_ref(kernel: Callable) -> KERNEL_DTYPE:
    """Convert a kernel function to a picklable (filepath, name) tuple.

    Handles plain functions and @nki.jit-wrapped functions by unwrapping
    to find the underlying function's source file.

    Args:
        kernel: The kernel function or decorated wrapper.

    Returns:
        Tuple of (absolute_filepath, function_name).

    Raises:
        TypeError: If the source file cannot be determined.
    """
    func = kernel
    if hasattr(func, "func"):
        func = func.func
    elif hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    try:
        source_file = os.path.abspath(inspect.getfile(func))
    except TypeError as e:
        raise TypeError(
            f"Cannot determine source file for kernel {kernel!r}. "
            f"Ensure the kernel is defined in a .py file, not interactively."
        ) from e

    func_name = func.__name__
    return (source_file, func_name)


def get_kernel_by_name(kernel_name: KERNEL_DTYPE) -> types.FunctionType:
    """Load a kernel function by its module path and function name.

    Args:
        kernel_name: Tuple of (module_path, function_name) identifying the kernel.

    Returns:
        The kernel function loaded from the specified module.
    """
    module_path, func_name = kernel_name
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[func_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, func_name)
    return kernel


def compile_kernel(
    kernel_name: KERNEL_DTYPE,
    input_tensors: INPUT_TENSORS_DTYPE,
    output_tensors: list[tuple[str, tuple[int, ...], np.dtype]],
    kernel_kwargs: KERNEL_KWARGS_DTYPE,
    compiler_flags: str,
    output_dir: str,
) -> str:
    """Compile a NKI kernel to NEFF using the compile_nki_ir_kernel_to_neff API.

    Args:
        kernel_name: Tuple of (module_path, function_name) identifying the kernel.
        input_tensors: Dictionary mapping tensor names to numpy arrays.
        output_tensors: List of (name, shape, dtype) tuples describing kernel outputs.
        kernel_kwargs: Additional keyword arguments for the kernel.
        compiler_flags: Additional compiler arguments.
        output_dir: Directory to store compilation artifacts.

    Returns:
        Path to the generated NEFF file.
    """
    tempfile.tempdir = "/tmp/nki_artifacts"
    os.makedirs(tempfile.tempdir, exist_ok=True)

    kernel = get_kernel_by_name(kernel_name)

    kernel_inputs_dict = {**input_tensors, **kernel_kwargs}

    kernel_outputs = [TensorStub(shape=shape, dtype=dtype, name=name) for name, shape, dtype in output_tensors]

    full_compiler_flags = f"--internal-compiler-debug-mode=penguin {compiler_flags}".strip()

    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    compile_nki_ir_kernel_to_neff(
        kernel_func=kernel,
        kernel_inputs_dict=kernel_inputs_dict,
        kernel_outputs=kernel_outputs,
        platform_target="trn2",
        logical_nc_config=1,
        output_directory=output_dir,
        version=NKI_IR_VERSION.beta2,
        additional_compiler_args=full_compiler_flags,
    )

    neff_path = os.path.join(output_dir, "file.neff")
    if not os.path.exists(neff_path):
        raise RuntimeError(f"NEFF file not found at expected path: {neff_path}")

    return neff_path


class MinimalTracedKernel:
    """Minimal traced kernel wrapper for runtime execution.

    This class provides the minimal interface required by CompiledKernel
    without actually tracing the kernel function. It creates a minimal
    HLOModule with input/output tensor information.
    """

    def __init__(self, func: Callable, input_tensors: INPUT_TENSORS_DTYPE, output_tensors: list[TensorStub]):
        """Initialize the minimal traced kernel.

        Args:
            func: The original kernel function.
            input_tensors: Dictionary mapping tensor names to numpy arrays.
            output_tensors: List of TensorStub describing kernel outputs.
        """
        self.func = func
        self._code = HLOModule(name=func.__name__)
        for name, tensor in input_tensors.items():
            self._code.add_parameter(tensor.shape, tensor.dtype, name=name)
        output_hlo_tensors = [HLOTensor(shape=stub.shape, dtype=stub.dtype, name=stub.name) for stub in output_tensors]
        self._code.set_results(output_hlo_tensors)

    @property
    def __name__(self) -> str:
        """Return the name of the underlying kernel function."""
        return self.func.__name__


def create_spike_kernel(
    neff_path: str,
    kernel_name: KERNEL_DTYPE,
    input_tensors: INPUT_TENSORS_DTYPE,
    output_tensors: list[TensorStub],
    kernel_kwargs: KERNEL_KWARGS_DTYPE,
) -> CompiledKernel:
    """Create a CompiledKernel from a NEFF file for runtime execution.

    Args:
        neff_path: Path to the compiled NEFF file.
        kernel_name: Tuple of (module_path, function_name) identifying the kernel.
        input_tensors: Dictionary mapping tensor names to numpy arrays.
        output_tensors: List of TensorStub describing kernel outputs.
        kernel_kwargs: Additional keyword arguments for the kernel.

    Returns:
        A CompiledKernel ready for execution on Neuron hardware.
    """
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        func = kernel
    elif hasattr(kernel, "func"):
        func = kernel.func
    else:
        raise TypeError(f"Unsupported kernel type: {type(kernel)}")
    traced_kernel = MinimalTracedKernel(func, input_tensors, output_tensors)
    spike_kernel = CompiledKernel(traced_kernel, neff_path)
    return spike_kernel


def run_spike_kernel(
    spike, spike_kernel, input_tensors, neff: str, kernel_kwargs: KERNEL_KWARGS_DTYPE
) -> tuple[str, tuple[np.ndarray, ...]]:
    """Run a compiled kernel and collect profiling data.

    Args:
        spike: BaremetalExecutor instance.
        spike_kernel: CompiledKernel to execute.
        input_tensors: Dictionary mapping tensor names to numpy arrays.
        neff: Path to the NEFF file.
        kernel_kwargs: Additional keyword arguments for the kernel.

    Returns:
        Tuple of (ntff_file_path, kernel_outputs).
    """
    directory, neff_name, file_type = split_file_info(neff)
    if file_type != "neff":
        raise ValueError(f"{neff} is not a neff file.")
    kernel_outputs = spike.run(
        spike_kernel, *input_tensors.values(), save_trace=True, artifacts_dir=directory, **kernel_kwargs
    )
    ntff_file = os.path.join(directory, "profile.ntff")
    if isinstance(kernel_outputs, np.ndarray):
        kernel_outputs = tuple([kernel_outputs])
    return ntff_file, kernel_outputs
