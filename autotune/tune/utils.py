import shutil
import types
from typing import Dict, Tuple

import numpy as np
from neuronpy.core.compile import compile_to_neff, trace
from neuronpy.runtime.spike import CompiledKernel
from neuronxcc.nki.compile import GenericKernel

global_kernel = None


def set_kernel(kernel):
    global global_kernel
    global_kernel = kernel


def compile_kernel(neff_name: str, kernel_args: Tuple[np.ndarray, ...], configs: Dict, output_dir: str) -> str:
    """Standalone function to create and compile a NKI kernel"""
    compile_dir = f"{output_dir}/{neff_name}"
    if isinstance(global_kernel, types.FunctionType):
        traced_kernel = trace(global_kernel)
        compiler_args = "--model-type=transformer"
    elif isinstance(global_kernel, GenericKernel):
        traced_kernel = global_kernel
        compiler_args = "--internal-tensorizer-opt-level=nki"
    else:
        raise TypeError(f"{type(global_kernel)} {global_kernel} is not supported.")
    traced_kernel.specialize(*kernel_args, **configs)
    neff = compile_to_neff(
        trace_kernel=traced_kernel,
        output_dir=compile_dir,
        additional_compiler_args=f"{compiler_args} --target=trn1 --auto-cast=none",
    )
    output_neff = f"{output_dir}/{neff_name}.neff"
    shutil.move(neff, output_neff)
    shutil.rmtree(compile_dir)
    return output_neff


def create_spike_kernel(neff_path: str, kernel, kernel_args: Tuple[np.ndarray, ...], configs: Dict) -> CompiledKernel:
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    traced_kernel.specialize(*kernel_args, **configs)
    spike_kernel = CompiledKernel(traced_kernel.copy_kernel(), neff_path)
    return spike_kernel
