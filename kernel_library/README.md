# Kernel Library

Each workload module exposes one `WORKLOAD` dictionary with exactly:

- `numpy_ref`
- `input_specs`
- `input_generator`
- `atol`
- `rtol`
- `best_historical_latency_ms`

`kernel_library.WORKLOADS` discovers every module automatically. Tests synthesize
the kernel from `numpy_ref`, generate inputs through `input_generator`, and use
the historical latency directly from the dictionary.

Rendered NKI kernels are generated artifacts and are not checked in.
