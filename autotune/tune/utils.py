import importlib
import shutil
import sys
import traceback
import types
from typing import Dict, Tuple

import numpy as np
from neuronpy.core.compile import compile_to_neff, trace
from neuronpy.runtime.spike import CompiledKernel, SpikeExecutor
from neuronxcc.nki.compile import GenericKernel

from autotune.cache.directories import split_file_info
from autotune.core.lhs_rhs import gemm_main
from autotune.core.matmul import matmul_main
from autotune.golden.gemm import matmul_op, matmul_xt_op
from autotune.golden.rmsnorm_linear import rmsnorm_gemm_golden
from autotune.tune.metrics import extract_metrics


def capture_error_message(e) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_string = f"{exc_type.__name__}: {str(e)}\n"
    error_string += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    return error_string


def get_kernel_by_name(kernel_name: str):
    # TODO: implement a kernel library to add/load kernels
    kernels = {
        "matmul_op": matmul_op,
        "matmul_xt_op": matmul_xt_op,
        "rmsnorm_linear_op": rmsnorm_gemm_golden,
        "matmul_main": matmul_main,
        "non_transposed_matmul": gemm_main,
    }
    if kernel_name in kernels:
        kernel = kernels[kernel_name]
    else:
        kernel_module = importlib.import_module("kernel_library")
        kernel = getattr(kernel_module, kernel_name)
    return kernel


def compile_kernel(
    kernel_name: str, neff_name: str, kernel_args: Tuple[np.ndarray, ...], output_dir: str, **kwargs
) -> str:
    """Standalone function to create and compile a NKI or NeuronPy kernel"""
    kernel = get_kernel_by_name(kernel_name)
    allocated_kernels = ["stack_allocated_fused_rms_norm_qkv", "blocked_fused_rms_norm_linear"]
    compiler_args = ["--target=trn1", "--auto-cast=none"]
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
        compiler_args.append("--model-type=transformer")
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
        compiler_args.append("--internal-tensorizer-opt-level=nki")
        if kernel_name in allocated_kernels:
            compiler_args.append("--internal-nki-allocation")
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    # TODO: dump the specialized NKI kernels
    traced_kernel.specialize(*kernel_args, **kwargs)
    compile_dir = f"{output_dir}/{neff_name}"
    neff = compile_to_neff(
        trace_kernel=traced_kernel, output_dir=compile_dir, additional_compiler_args=" ".join(compiler_args)
    )
    output_neff = f"{output_dir}/{neff_name}.neff"
    shutil.move(neff, output_neff)
    shutil.rmtree(compile_dir)
    return output_neff


def create_spike_kernel(
    neff_path: str, kernel_name: str, kernel_args: Tuple[np.ndarray, ...], **kwargs
) -> CompiledKernel:
    # FIXME: args are used, kwargs are needed to run but not used
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    traced_kernel.specialize(*kernel_args, **kwargs)
    spike_kernel = CompiledKernel(traced_kernel.copy_kernel(), neff_path)
    return spike_kernel


def capture_neff(spike, spike_kernel, kernel_args, neff: str, **kwargs) -> Tuple[str, np.ndarray]:
    directory, neff_name, file_type = split_file_info(neff)
    if file_type != "neff":
        raise ValueError(f"{neff} is not a neff file.")
    kernel_output = spike.run(spike_kernel, *kernel_args, save_trace=True, artifacts_dir=directory, **kwargs)
    ntff_file = f"{directory}/{neff_name}.ntff"
    shutil.move(f"{directory}/profile.ntff", ntff_file)
    return ntff_file, kernel_output


def run_kernel(kernel_name: str, kernel_args: Tuple[np.ndarray, ...], **kwargs) -> Tuple[np.ndarray, Dict]:
    tmp_dir = "/tmp/autotune-workspace"
    neff = compile_kernel(kernel_name, kernel_name, kernel_args, tmp_dir, **kwargs)
    spike_kernel = create_spike_kernel(neff, kernel_name, kernel_args, **kwargs)
    with SpikeExecutor(verbose=0) as spike:
        stats = spike.benchmark(
            spike_kernel, *kernel_args, **kwargs, warmup_iterations=10, benchmark_iterations=100, device_id=0
        )
        ntff, kernel_output = capture_neff(spike, spike_kernel, kernel_args, neff, **kwargs)
    metrics = extract_metrics(neff, ntff)
    metrics.update(stats)
    upload_command = f'profile-upload -F "neff=@{neff}" -F "ntff=@{ntff}" -F name={kernel_name}'
    return kernel_output, metrics
