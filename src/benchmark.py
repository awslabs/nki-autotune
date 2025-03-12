import neuronxcc.nki as nki
from neuronxcc.nki.compile import GenericKernel

import shutil, subprocess
from typing import Dict, Tuple


def profile_kernel(
    func: GenericKernel,
    args: Tuple,
    configs: Dict = {},
    warmup: int = 10,
    iters: int = 100,
    device_lock=None,
    benchmark_machine=None,
    cache_dir: str | None = None,
    trace: bool = False,
) -> float:
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
    bench_func = nki.benchmark(
        warmup=warmup,
        iters=iters,
        device_lock=device_lock,
        benchmark_machine=benchmark_machine,
        save_neff_name="file.neff",
    )(func)
    bench_func(*args, **configs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    if trace:
        assert cache_dir is not None, "Must provide a cache dir when generating trace files"
        trace_cmd = f"neuron-profile capture -n file.neff --profile-nth-exec={iters}"
        subprocess.run(trace_cmd, shell=True)
        shutil.move(f"profile_exec_{iters}.ntff", f"{cache_dir}/{func.func_name}.ntff")
    if cache_dir:
        shutil.move("file.neff", f"{cache_dir}/{func.func_name}.neff")
    p99 /= 1000
    return p99
