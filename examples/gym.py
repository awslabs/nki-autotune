"""NKI workload specification for dimension analysis.

This demonstrates how to analyze NKI functions for tiling using
the analyze_dimension function from nkigym.
"""

import os
from pathlib import Path

import numpy as np

import nkigym
from nkigym.codegen import get_source
from nkigym.data_reuse import analyze_data_reuse, merge_reusable_tensors
from nkigym.numpy_to_nki import lower_numpy_to_nki
from nkigym.tiling import generate_tiled_function

CACHE_ROOT = "/fsx/weittang/gym_cache"


def golden_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Reference matmul matching NKI nc_matmul semantics.

    Args:
        lhs: Left-hand side array of shape [K, M].
        rhs: Right-hand side array of shape [K, N].

    Returns:
        Result array of shape [M, N].
    """
    return np.matmul(lhs.T, rhs)


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
    expected = golden_matmul(lhs, rhs)

    (cache_path / "nkigym_matmul.py").write_text(get_source(matmul))
    np.testing.assert_allclose(matmul(lhs, rhs), expected)
    print("matmul matches golden")

    tiled_matmul = generate_tiled_function(matmul, input_shapes)
    (cache_path / "tiled_matmul.py").write_text(get_source(tiled_matmul))
    np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
    print("tiled_matmul matches golden")

    groups = analyze_data_reuse(tiled_matmul)
    for i, group in enumerate(groups):
        tiled_matmul = merge_reusable_tensors(tiled_matmul, group[0], group[1])
        np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
        print(f"merged_matmul (pass {i + 1}) matches golden")
    (cache_path / "transformed_matmul.py").write_text(get_source(tiled_matmul))

    nki_source = lower_numpy_to_nki(tiled_matmul)
    (cache_path / "nki_matmul.py").write_text(nki_source)


if __name__ == "__main__":
    main()
