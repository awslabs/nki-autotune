"""NKI Gym tiling, search-based codegen, and hardware profiling demo.

Demonstrates the full pipeline on a tiled matmul:

1. Generate a tiled function from a high-level NKI op.
2. Use graph-based search to explore the transform space and
   generate up to 50 unique kernel variants via data reuse and
   operand merge transforms.
3. Lower every nkigym variant to an NKI kernel and write to cache.
4. Compile and benchmark all NKI kernels on Neuron hardware via
   the autotune ProfileJobs / Benchmark backend.
"""

from pathlib import Path

import numpy as np

import nkigym
from autotune import Benchmark, ProfileJobs
from autotune.compiler.compile import get_kernel_by_name
from nkigym.codegen import lower_ir_to_nki
from nkigym.search import search
from nkigym.transforms import DataReuseTransform, OperandMergeTransform


def matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Perform NKI matrix multiplication.

    Args:
        lhs: Left-hand side tensor of shape [K, M] (partition x free).
        rhs: Right-hand side tensor of shape [K, N] (partition x free).

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def main() -> None:
    """Run tiling, search-based transform exploration, codegen, and hardware profiling."""
    CACHE_ROOT = "/fsx/weittang/gym_cache/matmul"
    cache_path = Path(CACHE_ROOT)
    k, m, n = 1024, 1024, 1024
    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    kernel_kwargs = {"lhs": lhs, "rhs": rhs}

    variants = search(
        matmul,
        transforms=[DataReuseTransform(), OperandMergeTransform()],
        num_targets=50,
        seed=42,
        min_depth=1,
        save_cache=cache_path,
        kernel_kwargs=kernel_kwargs,
    )

    nki_func_name = "nki_matmul"
    jobs = ProfileJobs(cache_root_dir=CACHE_ROOT)

    for step, variant in enumerate(variants):
        nki_source = lower_ir_to_nki(variant)
        nki_path = str(cache_path / f"nki_matmul_{step}.py")
        Path(nki_path).write_text(nki_source)
        kernel = get_kernel_by_name((nki_path, nki_func_name))
        jobs.add_job(
            kernel=kernel,
            kernel_kwargs={"lhs": lhs.astype(np.float16), "rhs": rhs.astype(np.float16)},
            output_shapes={"result": (m, n)},
            compiler_flags="",
            correctness_check=(matmul, 1e-3, 1e-3),
        )

    results = Benchmark(jobs, warmup=10, iters=100).run()
    results.summary(top_k=len(variants))


if __name__ == "__main__":
    main()
