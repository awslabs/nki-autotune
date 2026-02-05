"""NumPy workload specification for dimension analysis.

This demonstrates how to analyze NumPy functions for tiling using
the analyze_dimension function from nkigym.
"""

import os
from pathlib import Path

import numpy as np

from nkigym.codegen import get_source
from nkigym.data_reuse import analyze_data_reuse, merge_reusable_tensors
from nkigym.tiling import generate_tiled_function

CACHE_ROOT = "/fsx/weittang/gym_cache"


def matmul(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Perform matrix multiplication.

    Args:
        mat_a: First input tensor of shape (M, K).
        mat_b: Second input tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return np.matmul(mat_a, mat_b)


def main() -> None:
    """Run the tiling and data reuse analysis demo."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    m, k, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"mat_a": (m, k), "mat_b": (k, n)}

    mat_a = np.random.randn(m, k)
    mat_b = np.random.randn(k, n)
    expected = matmul(mat_a, mat_b)

    tiled_matmul = generate_tiled_function(matmul, input_shapes)
    (cache_path / "tiled_matmul.py").write_text(get_source(tiled_matmul))
    np.testing.assert_allclose(tiled_matmul(mat_a, mat_b), expected)

    groups = analyze_data_reuse(tiled_matmul)
    for group in groups:
        tiled_matmul = merge_reusable_tensors(tiled_matmul, group[0], group[1])
        np.testing.assert_allclose(tiled_matmul(mat_a, mat_b), expected)
    (cache_path / "transformed_matmul.py").write_text(get_source(tiled_matmul))


if __name__ == "__main__":
    main()
