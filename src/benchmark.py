import neuronxcc.nki as nki
from neuronxcc.nki.compile import GenericKernel

import shutil
from typing import Dict, Tuple


def dict_to_string(configs: Dict):
    result = []
    for key, val in configs.items():
        result.append(str(key))
        result.append(str(val))
    return "-".join(result)


def profile_kernel(
    func: GenericKernel,
    args: Tuple,
    cache_dir: str,
    configs: Dict = {},
    warmup: int = 10,
    iters: int = 100,
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
    configs_str = dict_to_string(configs)
    neff_filename = f"{cache_dir}/{func.func_name}-{configs_str}.neff"
    shutil.move("file.neff", neff_filename)
    p99 /= 1000
    return p99, neff_filename
