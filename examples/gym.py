"""NKI Gym search: combinatorial schedule exploration of kernel variants.

Demonstrates the schedule search pipeline: define a user function
with ``nkigym.<op>(...)`` calls, and the search automatically parses
dimensions, enumerates schedules, renders NKI kernels, compiles, and
benchmarks on hardware.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import nkigym
from nkigym.search import search


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply: stationary @ moving.

    Args:
        a: Stationary tensor of shape [K, M].
        b: Moving tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    c = nkigym.nc_matmul(a, b)
    return c


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NKI Gym search example")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/fsx/weittang/gym_cache"),
        help="Directory for storing output (default: /fsx/weittang/gym_cache)",
    )
    return parser.parse_args()


def main() -> None:
    """Run schedule search on a 2048x2048 matmul workload."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    cache_dir = args.cache_dir

    rng = np.random.default_rng(42)
    a = rng.standard_normal((2048, 2048)).astype(np.float16)
    b = rng.standard_normal((2048, 2048)).astype(np.float16)

    search(func=matmul, num_targets=99999, seed=42, save_cache=cache_dir, kernel_kwargs={"a": a, "b": b})


if __name__ == "__main__":
    main()
