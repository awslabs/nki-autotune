"""NKI Gym search: rmsnorm + matmul two-pass reduction kernel.

Demonstrates multi-pass schedule search: RMSNorm (activation+reduce
over K, then normalize) followed by matrix multiply, producing two
sequential reduction passes over the same dimension.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import nkigym
from nkigym.search import search


def rmsnorm_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSNorm(a) @ b: normalize rows of a then multiply by b.

    Args:
        a: Input tensor of shape [M, K].
        b: Weight tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    sum_sq = nkigym.activation(a, op="square", reduce_op=np.add)
    scaled = nkigym.tensor_scalar(sum_sq, op0=np.multiply, operand0=1 / 1024, op1=np.add, operand1=1e-6)
    rsqrt_val = nkigym.activation(scaled, op="rsqrt")
    a_normed = nkigym.tensor_scalar(a, rsqrt_val, op0=np.multiply)
    a_t = nkigym.transpose(a_normed)
    result = nkigym.nc_matmul(a_t, b)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NKI Gym rmsnorm+matmul search")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for storing output artifacts")
    return parser.parse_args()


def main() -> None:
    """Run schedule search on a 1024x1024 rmsnorm+matmul workload."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    cache_dir = args.cache_dir

    rng = np.random.default_rng(42)
    a = rng.standard_normal((1024, 1024)).astype(np.float16)
    b = rng.standard_normal((1024, 1024)).astype(np.float16)

    search(func=rmsnorm_matmul, num_targets=99999, seed=42, save_cache=cache_dir, kernel_kwargs={"a": a, "b": b})


if __name__ == "__main__":
    main()
