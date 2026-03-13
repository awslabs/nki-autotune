"""Shared test utilities and fixtures for pytest."""

import numpy as np


def normalize_source(source: str) -> str:
    """Normalize source code for comparison.

    Strips leading/trailing whitespace from each line, removes blank lines,
    and joins with single newlines.

    Args:
        source: Source code string.

    Returns:
        Normalized source string.
    """
    lines = [line.strip() for line in source.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def make_random_array(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate a deterministic random float32 array for testing.

    Args:
        shape: Shape of the array to generate.
        seed: Random seed for reproducibility.

    Returns:
        Random float32 array with values in [-1, 1] range.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
