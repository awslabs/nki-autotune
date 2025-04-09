import shutil
from typing import Dict, Tuple

import neuronxcc.nki as nki
from neuronxcc.nki.compile import GenericKernel

from src.cache.directories import dict_to_string


def profile_kernel(
    func: GenericKernel,
    args: Tuple,
    warmup: int = 10,
    iters: int = 100,
    cache_dir: str | None = None,
    configs: Dict = {},
    device_lock=None,
    benchmark_machine=None,
) -> Tuple[float, float, str]:
    """
    Profile the NKI kernel P99 latency
    Args:
        func (_type_): NKI kernel
        args (_type_): kernel inputs
        warmup (_type_): number of warmup runs
        iters (_type_): number of trials

    Returns:
        float: P99 latency in ms
    """
    if cache_dir:
        bench_func = nki.benchmark(
            warmup=warmup,
            iters=iters,
            device_lock=device_lock,
            benchmark_machine=benchmark_machine,
            save_neff_name="file.neff",
        )(func)
    else:
        bench_func = nki.benchmark(
            warmup=warmup, iters=iters, device_lock=device_lock, benchmark_machine=benchmark_machine
        )(func)
    bench_func(*args, **configs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    p99 /= 1000
    if cache_dir:
        configs_str = dict_to_string(configs)
        neff_filename = f"{cache_dir}/{func.func_name}-{configs_str}.neff"
        shutil.move("file.neff", neff_filename)
    else:
        neff_filename = ""
    pe_utilization = calculate_pe_utilization(args[0].tensor_shape, args[1].tensor_shape, p99, "trn1")
    return p99, pe_utilization, neff_filename


def calculate_pe_utilization(lhs_shape, rhs_shape, time_ms, target_instance_family):
    """
    Calculate hardware FLOPS utilization for a GEMM operation.

    Parameters:
    lhs_shape (tuple): Shape of matrix A (m, k)
    rhs_shape (tuple): Shape of matrix B (k, n)
    time_ms (float): Execution time in milliseconds

    Returns:
    dict: Dictionary containing FLOPS, TFLOPS, max TFLOPS, and utilization percentage
    """
    m, k = lhs_shape
    _k, n = rhs_shape
    assert k == _k, f"Incompatible matrix dimensions: {lhs_shape} and {rhs_shape}"

    if any(target_instance_family.startswith(family) for family in {"sunda", "trainium", "trn1", "inf2"}):
        pe_freq = 2.8 * 1e9  # Hz (2.8 GHz)
        num_lnc = 1
    elif any(target_instance_family.startswith(family) for family in {"trn2", "inf3", "gen3", "cayman"}):
        pe_freq = 2.4 * 1e9  # Hz (2.4 GHz)
        num_lnc = 2
    else:
        raise NotImplementedError("Unknown target instance: " + target_instance_family)

    # Calculate total FLOPS (2 operations per matrix element - multiply and add)
    flops = 2 * m * k * n
    actual_latency_s = time_ms / 1000
    actual_pe_cycles = actual_latency_s * pe_freq
    theoretical_pe_cycles = flops / (2 * 128 * 128 * num_lnc)
    pe_utilization = theoretical_pe_cycles / actual_pe_cycles
    return pe_utilization
