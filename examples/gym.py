"""NKI workload specification for dimension analysis.

This demonstrates how to analyze NKI functions for tiling using
the analyze_dimension function from nkigym.
"""

import os
from pathlib import Path

import numpy as np

import nkigym
from autotune import Benchmark, ProfileJobs
from autotune.core.compile import get_kernel_by_name
from nkigym.lower import lower_gym_to_nki
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import analyze_data_reuse, merge_reusable_tensors
from nkigym.utils import get_source

CACHE_ROOT = "/fsx/weittang/gym_cache"


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
    """Run the tiling and data reuse analysis demo."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    k, m, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    expected = matmul(lhs, rhs)

    (cache_path / "nkigym_matmul.py").write_text(get_source(matmul))
    np.testing.assert_allclose(matmul(lhs, rhs), expected)
    print("matmul matches golden")

    tiled_matmul = generate_tiled_function(matmul, input_shapes)
    (cache_path / "tiled_matmul.py").write_text(get_source(tiled_matmul))
    np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
    print("tiled_matmul matches golden")
    nki_source = lower_gym_to_nki(tiled_matmul)
    (cache_path / "nki_matmul.py").write_text(nki_source)

    groups = analyze_data_reuse(tiled_matmul)
    for i, group in enumerate(groups):
        tiled_matmul = merge_reusable_tensors(tiled_matmul, group[0], group[1])
        np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
        print(f"merged_matmul (pass {i + 1}) matches golden")
    (cache_path / "transformed_matmul.py").write_text(get_source(tiled_matmul))

    nki_source = lower_gym_to_nki(tiled_matmul)
    (cache_path / "nki_matmul_transformed.py").write_text(nki_source)

    nki_matmul = get_kernel_by_name((str(cache_path / "nki_matmul.py"), "nki_tiled_matmul"))
    nki_matmul_transformed = get_kernel_by_name((str(cache_path / "nki_matmul_transformed.py"), "nki_tiled_matmul"))

    lhs_f32 = lhs.astype(np.float32)
    rhs_f32 = rhs.astype(np.float32)

    jobs = ProfileJobs(cache_root_dir=str(cache_path / "benchmark"))
    for kernel in [nki_matmul, nki_matmul_transformed]:
        jobs.add_job(
            kernel=kernel,
            kernel_kwargs={"lhs": lhs_f32, "rhs": rhs_f32},
            output_shapes={"output": (m, n)},
            compiler_flags="--auto-cast=none",
            correctness_check=(matmul, 1e-3, 1e-3),
        )

    results = Benchmark(jobs=jobs, warmup=5, iters=10).run()
    results.summary()


if __name__ == "__main__":
    main()
