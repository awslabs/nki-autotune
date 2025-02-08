import neuronxcc.nki as nki
import shutil


def test_design(
    func,
    args,
    kwargs,
    configs,
    device_lock,
    warmup,
    iters,
    cache_dir,
    benchmark_machine=None,
):
    bench_func = nki.benchmark(
        warmup=warmup,
        iters=iters,
        device_lock=device_lock,
        benchmark_machine=benchmark_machine,
        save_neff_name="file.neff",
        save_trace_name="profile.ntff",
    )(func)
    bench_func(*args, **configs, **kwargs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    profile_name = "-".join(f"{v}" for k, v in configs.items())
    shutil.move("file.neff", f"{cache_dir}/{profile_name}.neff")
    shutil.move("profile.ntff", f"{cache_dir}/{profile_name}.ntff")
    return p99


def test_kernel(
    func,
    args,
    warmup,
    iters,
) -> float:
    """Profile the NKI kernel P99 latency

    Args:
        func (_type_): NKI kernel
        args (_type_): kernel inputs
        warmup (_type_): number of warmup runs
        iters (_type_): number of trials

    Returns:
        float: P99 latency
    """
    bench_func = nki.benchmark(
        warmup=warmup,
        iters=iters,
    )(func)
    bench_func(*args)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    return p99
