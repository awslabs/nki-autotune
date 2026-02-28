"""NKI Gym search: exhaustive exploration of transform variants.

Tiles a matmul workload and searches the space of atomic transforms
(data reuse, operand merge) for all unique kernel variants. Each variant
is written to the cache directory and verified for numerical correctness.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import nkigym
from nkigym.search import search
from nkigym.transforms import DataReuseTransform, OperandMergeTransform


def nkigym_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """NKI Gym matrix multiplication (NumPy simulation mode).

    Args:
        lhs: Left-hand side tensor of shape [K, M].
        rhs: Right-hand side tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NKI Gym search example")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"), help="Directory for storing output (default: cache)"
    )
    return parser.parse_args()


def main() -> None:
    """Run transform search on a tiled matmul workload."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    cache_dir = args.cache_dir

    k, m, n = 256, 256, 256
    rng = np.random.default_rng()
    lhs = rng.standard_normal((k, m)).astype(np.float16)
    rhs = rng.standard_normal((k, n)).astype(np.float16)

    search(
        func=nkigym_matmul,
        transforms=[DataReuseTransform(), OperandMergeTransform()],
        num_targets=1000,
        seed=-1,
        min_depth=10,
        save_cache=cache_dir,
        kernel_kwargs={"lhs": lhs, "rhs": rhs},
    )


if __name__ == "__main__":
    main()
