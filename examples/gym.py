"""NumPy workload specification for dimension analysis.

This demonstrates how to analyze NumPy functions for tiling using
the analyze_dimension function from nkigym.
"""

import logging
import os

import numpy as np

from nkigym.data_reuse import analyze_data_reuse, merge_reusable_tensors
from nkigym.tiling import generate_tiled_function
from nkigym.visualize import get_source, setup_logging

CACHE_ROOT = "/fsx/weittang/kernelgen_cache"
os.makedirs(CACHE_ROOT, exist_ok=True)
setup_logging(f"{CACHE_ROOT}/debug.log")
logger = logging.getLogger(__name__)


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
    logger.debug("Starting tiling analysis demo")
    m, k, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"mat_a": (m, k), "mat_b": (k, n)}

    tiled_matmul = generate_tiled_function(matmul, input_shapes)
    logger.info(get_source(tiled_matmul))

    groups = analyze_data_reuse(tiled_matmul)
    logger.info(groups)

    if not groups:
        raise RuntimeError("No reusable tensor groups found")

    transformed = merge_reusable_tensors(tiled_matmul, groups[0][0], groups[0][1])
    logger.info(get_source(transformed))


if __name__ == "__main__":
    main()
