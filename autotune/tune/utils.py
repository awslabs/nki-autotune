import shutil
import types
from typing import Dict, Tuple

import numpy as np
from neuronpy.core.compile import compile_to_neff, trace
from neuronpy.runtime.spike import CompiledKernel
from neuronxcc.nki.compile import GenericKernel

from autotune.baseline.np_baselines import matmul_op, matmul_xt_op, rmsnorm_linear_op
from autotune.kernels.matmul import matmul_main
from autotune.kernels.rmsnorm_linear import (
    allocated_fused_rms_norm_qkv,
    blocked_fused_rms_norm_linear,
    stack_allocated_fused_rms_norm_qkv,
)


def get_kernel_by_name(kernel_name):
    kernels = {
        "matmul_op": matmul_op,
        "matmul_xt_op": matmul_xt_op,
        "rmsnorm_linear_op": rmsnorm_linear_op,
        "stack_allocated_fused_rms_norm_qkv": stack_allocated_fused_rms_norm_qkv,
        "matmul_main": matmul_main,
        "allocated_fused_rms_norm_qkv": allocated_fused_rms_norm_qkv,
        "blocked_fused_rms_norm_linear": blocked_fused_rms_norm_linear,
    }
    if kernel_name not in kernels:
        raise ValueError(f"Need to include {kernel_name} in kernel library.")
    return kernels[kernel_name]


def compile_kernel(
    kernel_name: str, neff_name: str, kernel_args: Tuple[np.ndarray, ...], configs: Dict, output_dir: str
) -> str:
    """Standalone function to create and compile a NKI or NeuronPy kernel"""
    compile_dir = f"{output_dir}/{neff_name}"
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
        compiler_args = "--model-type=transformer"
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
        compiler_args = "--internal-tensorizer-opt-level=nki"
        if kernel_name == "stack_allocated_fused_rms_norm_qkv":
            compiler_args = "--internal-nki-allocation"
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    # TODO: dump the specialized NKI kernels
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


def create_spike_kernel(
    neff_path: str, kernel_name: str, kernel_args: Tuple[np.ndarray, ...], configs: Dict
) -> CompiledKernel:
    # FIXME: args are used, kwargs are needed to run but not used
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    # FIXME: traced_kernel.matmul_mac_count is calculating model MAC. Need to calculate hardware MAC.
    traced_kernel.specialize(*kernel_args, **configs)
    spike_kernel = CompiledKernel(traced_kernel.copy_kernel(), neff_path)
    return spike_kernel
