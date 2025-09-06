import importlib
import importlib.util
import shutil
import signal
import sys
import types
from typing import Dict, Tuple

import numpy as np
from neuronpy.core.compile import CompilationTarget, compile_to_neff, trace
from neuronpy.runtime.spike import CompiledKernel, SpikeExecutor
from neuronxcc.nki.compile import GenericKernel

from autotune.cache.directories import split_file_info
from autotune.core.metrics import extract_metrics
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_DTYPE, KERNEL_KWARGS_DTYPE


def process_compiler_flags(compiler_flags: str):
    """
    Process compiler flags string to extract target instance family and clean remaining flags.

    This function extracts the target instance family (trn1 or trn2) from compiler flags,
    removes that target flag from the string, and normalizes whitespace in the remaining flags.

    Args:
        compiler_flags: A string containing compiler flags, which must include either
                       '--target=trn1' or '--target=trn2'

    Returns:
        tuple: A tuple containing:
            - target_instance_family (str): The extracted target instance family ('trn1' or 'trn2')
            - compiler_flags (str): The remaining compiler flags with normalized whitespace

    Raises:
        NotImplementedError: If the compiler flags do not contain either '--target=trn1'
                             or '--target=trn2'
    """
    if "--target=trn1" in compiler_flags:
        target_instance_family = "trn1"
        compiler_flags = compiler_flags.replace("--target=trn1", "")
    elif "--target=trn2" in compiler_flags:
        target_instance_family = "trn2"
        compiler_flags = compiler_flags.replace("--target=trn2", "")
    else:
        raise NotImplementedError(
            f"Only support --target=trn1 or --target=trn2 in compiler flags. Received {compiler_flags}."
        )
    compiler_flags = " ".join(compiler_flags.split())
    return target_instance_family, compiler_flags


def split_data_by_core(data, core_id, total_cores):
    """
    Split a list of data evenly among cores.

    Args:
        data: List of items to split
        core_id: ID of current core (0-based)
        total_cores: Total number of cores

    Returns:
        Subset of data assigned to this core
    """
    # Calculate chunk size and starting position
    chunk_size = len(data) // total_cores
    remainder = len(data) % total_cores

    # Distribute the remainder evenly
    start_idx = core_id * chunk_size + min(core_id, remainder)
    # If core_id < remainder, this core gets one extra item
    end_idx = start_idx + chunk_size + (1 if core_id < remainder else 0)

    return data[start_idx:end_idx]


def parse_path_and_function(combined_str: str):
    """
    Parse a string containing a file path and function name in the format:
    path/to/file.py/function_name

    Args:
        combined_str (str): The combined path and function string

    Returns:
        tuple: (file_path, function_name)

    Raises:
        ValueError: If the string format is invalid
    """
    # Find the last slash which separates the file path from the function name
    last_slash_index = combined_str.rfind("/")

    if last_slash_index == -1:
        raise ValueError("Invalid format: expected 'path/to/file.py/function_name'")

    # Extract the file path and function name
    file_path = combined_str[:last_slash_index]
    function_name = combined_str[last_slash_index + 1 :]

    # Validate the extracted components
    if not file_path or not function_name:
        raise ValueError("Invalid format: both file path and function name must be non-empty")

    # Check if the file path likely ends with a Python file
    if not file_path.endswith(".py"):
        # This is just a warning check, not an error
        import warnings

        warnings.warn("File path doesn't end with '.py', which is unusual")

    return file_path, function_name


def get_kernel_by_name(kernel_name: KERNEL_DTYPE):
    # TODO: implement a kernel library to add/load kernels
    module_path, func_name = kernel_name
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[func_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, func_name)
    return kernel


def timeout_handler(signum, frame):
    raise TimeoutError("Compilation timed out after 3 minutes")


def compile_kernel(
    kernel_name: KERNEL_DTYPE,
    input_tensors: INPUT_TENSORS_DTYPE,
    kernel_kwargs: KERNEL_KWARGS_DTYPE,
    target_instance_family: str,
    compiler_flags: str,
    output_dir: str,
) -> str:
    """Standalone function to create and compile a NKI or NeuronPy kernel"""
    kernel = get_kernel_by_name(kernel_name)
    if isinstance(kernel, types.FunctionType):
        traced_kernel = trace(kernel)
    elif isinstance(kernel, GenericKernel):
        traced_kernel = kernel
    else:
        raise TypeError(f"{type(kernel)} {kernel} is not supported.")
    traced_kernel.specialize(*input_tensors, **kernel_kwargs)
    if target_instance_family == "trn1":
        target = CompilationTarget.TRN1
    elif target_instance_family == "trn2":
        target = CompilationTarget.TRN2
    else:
        raise Exception(f"target_instance_family {target_instance_family} must be trn1 or trn2")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)
    try:
        neff = compile_to_neff(
            trace_kernel=traced_kernel,
            output_dir=output_dir,
            target=target,
            additional_compiler_args=compiler_flags,
            save_artifacts=True,
        )
        return neff
    finally:
        signal.alarm(0)


def create_spike_kernel(
    neff_path: str, kernel_name: KERNEL_DTYPE, input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE
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
) -> Tuple[str, tuple[np.ndarray, ...]]:
    directory, neff_name, file_type = split_file_info(neff)
    if file_type != "neff":
        raise ValueError(f"{neff} is not a neff file.")
    kernel_outputs = spike.run(spike_kernel, *input_tensors, save_trace=True, artifacts_dir=directory, **kernel_kwargs)
    ntff_file = f"{directory}/{neff_name}.ntff"
    shutil.move(f"{directory}/profile.ntff", ntff_file)
    if type(kernel_outputs) is np.ndarray:
        kernel_outputs = tuple([kernel_outputs])
    return ntff_file, kernel_outputs


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
