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
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for storing output artifacts")
    parser.add_argument("--remote-config", type=Path, default=None, help="Path to remote.json config file")
    parser.add_argument("--local", action="store_true", help="Run benchmarks locally instead of via SSH")
    return parser.parse_args()


def main() -> None:
    """Run schedule search on a 2048x2048 matmul workload."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    remote_config = None
    if not args.local:
        if args.remote_config is None:
            raise SystemExit("--remote-config is required for distributed mode (or use --local)")
        remote_config = args.remote_config

    rng = np.random.default_rng(42)
    a = rng.standard_normal((2048, 2048)).astype(np.float16)
    b = rng.standard_normal((2048, 2048)).astype(np.float16)

    search(
        func=matmul,
        num_targets=99999,
        seed=42,
        save_cache=args.cache_dir,
        kernel_kwargs={"a": a, "b": b},
        remote_config=remote_config,
    )


if __name__ == "__main__":
    main()
