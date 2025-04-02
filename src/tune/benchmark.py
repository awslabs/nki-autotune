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
) -> Tuple[float, str]:
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
    return p99, neff_filename
