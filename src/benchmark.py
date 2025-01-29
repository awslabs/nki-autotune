import neuronxcc.nki as nki


def test_design(
    func,
    args,
    kwargs,
    configs,
    device_lock,
    warmup,
    iters,
    benchmark_machine=None,
):
    bench_func = nki.benchmark(
        warmup=warmup,
        iters=iters,
        device_lock=device_lock,
        benchmark_machine=benchmark_machine,
    )(func)
    bench_func(*args, **configs, **kwargs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    return p99
