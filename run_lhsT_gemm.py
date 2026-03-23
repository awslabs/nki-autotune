"""Run template-based GEMM autotuning for 2048^3 fp16 matmul (lhsT @ rhs).

Renders standalone NKI kernel source files from GEMM configs,
then compiles and benchmarks them on Neuron hardware.
"""

import importlib.util
import os
import sys
from collections.abc import Callable

import numpy as np

from autotune.gemm import gemm_correctness_check, sample_gemm_configs
from autotune.gemm.render import render_gemm_nki_source
from autotune.job import ProfileJobs
from autotune.runner.benchmark import Benchmark

CACHE_DIR = "/fsx/weittang/template_cache_3"
M = 2048
N = 2048
K = 2048
WARMUP = 10
ITERS = 100
MAX_CONFIGS = 100


def _load_kernel_from_file(path: str, func_name: str) -> Callable:
    """Import and return a kernel function from a source file.

    Args:
        path: Absolute path to the .py source file.
        func_name: Name of the function to load.

    Returns:
        The loaded kernel function.

    Raises:
        ImportError: If the module cannot be loaded from the path.
    """
    spec = importlib.util.spec_from_file_location(func_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[func_name + "_" + path] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def _render_and_save_sources(configs: list[dict]) -> list[str]:
    """Render NKI source for each config and save to cache dir.

    Args:
        configs: List of GEMM config dicts.

    Returns:
        List of absolute paths to saved .py files.
    """
    sources_dir = os.path.join(CACHE_DIR, "sources")
    os.makedirs(sources_dir, exist_ok=True)
    paths = []
    for idx, cfg in enumerate(configs):
        source = render_gemm_nki_source(cfg, transposed_lhs=True)
        path = os.path.join(sources_dir, f"nki_v{idx}.py")
        with open(path, "w") as f:
            f.write(source)
        paths.append(path)
    return paths


def build_jobs() -> ProfileJobs:
    """Create benchmark jobs from rendered NKI kernel sources.

    Returns:
        ProfileJobs collection ready for benchmarking.
    """
    configs = sample_gemm_configs(M=M, N=N, K=K, max_configs=MAX_CONFIGS)
    print(f"Generated {len(configs)} GEMM configurations for {M}x{N}x{K}")

    source_paths = _render_and_save_sources(configs)
    print(f"Rendered {len(source_paths)} NKI kernel sources to {CACHE_DIR}/sources/")

    lhs = np.random.randn(K, M).astype(np.float16)
    rhs = np.random.randn(K, N).astype(np.float16)
    correctness = gemm_correctness_check(transposed_lhs=True)
    mac_count = M * N * K

    jobs = ProfileJobs(cache_root_dir=CACHE_DIR)
    for path in source_paths:
        kernel = _load_kernel_from_file(path, "matmul")
        jobs.add_job(
            kernel=kernel,
            kernel_kwargs={"a": lhs, "b": rhs},
            output_shapes={"result": (M, N)},
            compiler_flags="--auto-cast=none --internal-tensorizer-opt-level=nki",
            correctness_check=correctness,
        )
    for job in jobs.jobs.values():
        setattr(job, "mac_count", mac_count)
    return jobs


def main() -> None:
    """Run the full lhsT GEMM autotuning pipeline."""
    jobs = build_jobs()
    print(f"Running {len(jobs.jobs)} jobs (warmup={WARMUP}, iters={ITERS})")
    print(f"Cache dir: {CACHE_DIR}")
    benchmark = Benchmark(jobs=jobs, warmup=WARMUP, iters=ITERS)
    results = benchmark.run()
    results.summary(top_k=10)


if __name__ == "__main__":
    main()
