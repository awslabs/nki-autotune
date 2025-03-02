import neuronxcc.nki as nki
from neuronxcc.nki.compile import GenericKernel

import shutil, subprocess
from typing import Callable, Dict, Tuple


def test_design(
    func: GenericKernel,
    args: Tuple,
    kwargs: Dict,
    configs: Dict,
    device_lock,
    warmup: int,
    iters: int,
    cache_dir: str,
    benchmark_machine=None,
) -> float:
    """
    Returns:
        float: P99 latency in ms
    """
    bench_func = nki.benchmark(
        warmup=warmup,
        iters=iters,
        device_lock=device_lock,
        benchmark_machine=benchmark_machine,
        save_neff_name="file.neff",
    )(func)
    bench_func(*args, **configs, **kwargs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    profile_name = func.func_name + "-" + "-".join(f"{v}" for k, v in configs.items())
    # cmd = f"neuron-profile capture -n file.neff --profile-nth-exec={iters}"
    # subprocess.run(cmd, shell=True)
    shutil.move("file.neff", f"{cache_dir}/{profile_name}.neff")
    # shutil.move(f"profile_exec_{iters}.ntff", f"{cache_dir}/{profile_name}.ntff")
    p99 /= 1000
    return p99


def test_kernel(func, args, warmup: int, iters: int) -> float:
    """Profile the NKI kernel P99 latency

    Args:
        func (_type_): NKI kernel
        args (_type_): kernel inputs
        warmup (_type_): number of warmup runs
        iters (_type_): number of trials

    Returns:
        float: P99 latency in ms
    """
    bench_func = nki.benchmark(warmup=warmup, iters=iters)(func)
    bench_func(*args)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    p99 /= 1000
    return p99
