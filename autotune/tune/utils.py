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
from autotune.core.lhsT_rhs import matmul_main
from autotune.golden.gemm import matmul_op, matmul_xt_op
from autotune.golden.rmsnorm_linear import rmsnorm_gemm_golden
from autotune.tune.metrics import extract_metrics
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE


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
    kernel_name: str,
    neff_name: str,
    input_tensors: INPUT_TENSORS_DTYPE,
    kernel_kwargs: KERNEL_KWARGS_DTYPE,
    compiler_flags: str,
    output_dir: str,
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
    compiler_args = " ".join(compiler_args)
    # TODO: dump the specialized NKI kernels
    traced_kernel.specialize(*input_tensors, **kernel_kwargs)
    compile_dir = f"{output_dir}/{neff_name}"
    neff = compile_to_neff(trace_kernel=traced_kernel, output_dir=compile_dir, additional_compiler_args=compiler_flags)
    output_neff = f"{output_dir}/{neff_name}.neff"
    shutil.move(neff, output_neff)
    shutil.rmtree(compile_dir)
    return output_neff


def create_spike_kernel(
    neff_path: str, kernel_name: str, input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE
) -> CompiledKernel:
    # FIXME: args are used, kwargs are needed to run but not used
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    traced_kernel.specialize(*input_tensors, **kernel_kwargs)
    spike_kernel = CompiledKernel(traced_kernel.copy_kernel(), neff_path)
    return spike_kernel


def run_spike_kernel(
    spike, spike_kernel, input_tensors, neff: str, kernel_kwargs: KERNEL_KWARGS_DTYPE
) -> Tuple[str, np.ndarray]:
    directory, neff_name, file_type = split_file_info(neff)
    if file_type != "neff":
        raise ValueError(f"{neff} is not a neff file.")
    kernel_output = spike.run(spike_kernel, *input_tensors, save_trace=True, artifacts_dir=directory, **kernel_kwargs)
    ntff_file = f"{directory}/{neff_name}.ntff"
    shutil.move(f"{directory}/profile.ntff", ntff_file)
    return ntff_file, kernel_output


def run_kernel(kernel_name: str, input_tensors: Tuple[np.ndarray, ...], **kwargs) -> Tuple[np.ndarray, Dict]:
    tmp_dir = "/tmp/autotune-workspace"
    neff = compile_kernel(kernel_name, kernel_name, input_tensors, tmp_dir, **kwargs)
    spike_kernel = create_spike_kernel(neff, kernel_name, input_tensors, **kwargs)
    with SpikeExecutor(verbose=0) as spike:
        stats = spike.benchmark(
            spike_kernel, *input_tensors, **kwargs, warmup_iterations=10, benchmark_iterations=100, device_id=0
        )
        ntff, kernel_output = run_spike_kernel(spike, spike_kernel, input_tensors, neff, **kwargs)
    metrics = extract_metrics(neff, ntff)
    metrics.update(stats)
    upload_command = f'profile-upload -F "neff=@{neff}" -F "ntff=@{ntff}" -F name={kernel_name}'
    return kernel_output, metrics
