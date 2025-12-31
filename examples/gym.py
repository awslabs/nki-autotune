"""NumPy workload specification for RMSNorm + Matmul fusion.

This demonstrates how to define kernels using standard NumPy operations
and lower them to MLIR using NKIPy KernelGen.

See compute_graph/README.md for installation instructions.
"""

import logging
import os

import numpy as np
from nkipy_kernelgen import apply_passes, trace
from nkipy_kernelgen.transforms import remove_linalg_zero_fill

from compute_graph.visualize import setup_logging

cache_root = "/fsx/weittang/kernelgen_cache"
os.makedirs(cache_root, exist_ok=True)
setup_logging(f"{cache_root}/debug.log")
logger = logging.getLogger(__name__)

RMSNORM_EPSILON = 1e-6

# Matrix dimensions
M, K, N = 256, 128, 128


@trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
def rmsnorm_matmul(lhs, rhs):
    """Fused RMSNorm + Matmul: output = RMSNorm(lhs) @ rhs

    RMSNorm(x) = x / sqrt(mean(x^2) + epsilon)

    Args:
        lhs: Input tensor of shape (M, K)
        rhs: Weight tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    K_dim = lhs.shape[-1]
    lhs_square = np.square(lhs)
    lhs_sum_square = np.sum(lhs_square, axis=-1, keepdims=True)
    rmsnorm_factor = 1.0 / np.sqrt(lhs_sum_square / K_dim + RMSNORM_EPSILON)
    lhs_norm = lhs * rmsnorm_factor
    return np.matmul(lhs_norm, rhs)


if __name__ == "__main__":
    mlir_module = rmsnorm_matmul.to_mlir()
    logger.info(mlir_module)
    mlir_module = apply_passes(mlir_module, [remove_linalg_zero_fill])
    logger.info(mlir_module)
