"""Benchmark NKI kernel variants on Neuron hardware.

Compiles and runs NKI kernel files produced by search, collecting
timing data for each variant via the autotune backend.
"""

import logging
from pathlib import Path

import numpy as np

from autotune import Benchmark, BenchmarkResults, ProfileJobs
from autotune.compiler.compile import get_kernel_by_name

logger = logging.getLogger(__name__)


def _find_nki_variants(cache_dir: Path) -> list[Path]:
    """Find all NKI kernel variant files in the nki/ subdirectory.

    Args:
        cache_dir: Root cache directory from search.

    Returns:
        Sorted list of NKI kernel file paths.
    """
    nki_dir = cache_dir / "nki"
    return sorted(nki_dir.glob("nki_d*_v*.py"))


def benchmark_variants(
    cache_dir: Path,
    func_name: str,
    kernel_kwargs: dict[str, np.ndarray],
    output_name: str,
    output_shape: tuple[int, ...],
    warmup: int,
    iters: int,
) -> BenchmarkResults:
    """Compile and benchmark all NKI kernel variants on Neuron hardware.

    Finds all ``nki_d*_v*.py`` files produced by search, creates
    benchmark jobs for each, and runs them on Neuron cores.

    Args:
        cache_dir: Directory containing NKI kernel files from search.
        func_name: Function name inside each NKI kernel file.
        kernel_kwargs: Input tensors for the kernel (numpy arrays).
        output_name: Name of the output tensor (return variable).
        output_shape: Shape of the output tensor.
        warmup: Number of warmup iterations for benchmarking.
        iters: Number of benchmark iterations.

    Returns:
        BenchmarkResults with timing data for each variant.

    Raises:
        FileNotFoundError: If no NKI kernel files are found.
    """
    nki_files = _find_nki_variants(cache_dir)
    if not nki_files:
        raise FileNotFoundError(f"No NKI kernel files found in {cache_dir}")

    logger.info("Benchmarking %d NKI kernel variants", len(nki_files))

    jobs = ProfileJobs(cache_root_dir=str(cache_dir / "benchmark"))
    for nki_file in nki_files:
        kernel = get_kernel_by_name((str(nki_file), func_name))
        jobs.add_job(
            kernel=kernel, kernel_kwargs=kernel_kwargs, output_shapes={output_name: output_shape}, compiler_flags=""
        )

    results = Benchmark(jobs, warmup=warmup, iters=iters).run()
    logger.info("Benchmark complete: %d variants processed", len(nki_files))
    return results
