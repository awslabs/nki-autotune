import neuronxcc.nki as nki
import shutil, subprocess


def test_design(
    func,
    args,
    kwargs,
    configs,
    pruning_func,
    device_lock,
    warmup,
    iters,
    cache_dir,
    benchmark_machine=None,
) -> float:
    # Pruning func should auto fail if the inputs are illegal
    arg_shapes = [arg.tensor_shape for arg in args]
    pruning_func(*arg_shapes, **configs, **kwargs)
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
    cmd = f"neuron-profile capture -n file.neff --profile-nth-exec={iters}"
    subprocess.run(cmd, shell=True)
    shutil.move("file.neff", f"{cache_dir}/{profile_name}.neff")
    shutil.move(f"profile_exec_{iters}.ntff", f"{cache_dir}/{profile_name}.ntff")
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
