"""Run template-based GEMM autotuning for 2048x2048x2048 matmul."""

import argparse

import numpy as np

from autotune.gemm import gemm_correctness_check, lhs_rhs_meta_gemm, sample_gemm_configs
from autotune.job import ProfileJobs
from autotune.runner.benchmark import Benchmark

DEFAULT_CACHE_DIR = "/fsx/weittang/gemm_autotune_cache"
M = 2048
N = 2048
K = 2048
WARMUP = 10
ITERS = 100


def build_jobs(cache_dir: str) -> ProfileJobs:
    """Create benchmark jobs for all GEMM configurations.

    Args:
        cache_dir: Root directory for caching results.

    Returns:
        ProfileJobs collection with one job per config.
    """
    configs = sample_gemm_configs(M=M, N=N, K=K, max_configs=100)
    print(f"Generated {len(configs)} GEMM configurations for {M}x{N}x{K}")

    lhs = np.random.randn(M, K).astype(np.float32)
    rhs = np.random.randn(K, N).astype(np.float32)
    correctness = gemm_correctness_check(transposed_lhs=False)

    jobs = ProfileJobs(cache_root_dir=cache_dir)
    for config in configs:
        jobs.add_job(
            kernel=lhs_rhs_meta_gemm,
            kernel_kwargs={"lhs": lhs, "rhs": rhs, "config": config},
            output_shapes={"result": (M, N)},
            compiler_flags="--auto-cast=none --internal-tensorizer-opt-level=nki",
            correctness_check=correctness,
        )
    return jobs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="GEMM autotuning for 2048x2048x2048 matmul")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Root directory for caching results")
    return parser.parse_args()


def main() -> None:
    """Run the full GEMM autotuning pipeline."""
    args = parse_args()
    jobs = build_jobs(args.cache_dir)
    print(f"Running {len(jobs.jobs)} jobs (warmup={WARMUP}, iters={ITERS})")
    print(f"Cache dir: {args.cache_dir}")
    benchmark = Benchmark(jobs=jobs, warmup=WARMUP, iters=ITERS)
    results = benchmark.run()
    results.summary()


if __name__ == "__main__":
    main()
