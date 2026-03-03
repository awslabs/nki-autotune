"""Shared test utilities and fixtures for pytest."""

import numpy as np

from nkigym.ir import GymProgram


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


def assert_programs_numerically_equal(before: GymProgram, after: GymProgram, rtol: float, atol: float) -> None:
    """Assert two GymPrograms produce identical outputs for random inputs.

    Executes both programs via GymProgram.__call__(), generates
    deterministic random inputs from the before program's kwargs,
    and compares outputs with assert_allclose.

    Args:
        before: Reference program.
        after: Program to compare against.
        rtol: Relative tolerance for assert_allclose.
        atol: Absolute tolerance for assert_allclose.
    """
    input_shapes = {k: v.shape for k, v in before.kwargs.items() if isinstance(v, np.ndarray)}
    kwargs = {p: make_random_array(input_shapes[p], seed=42 + i) for i, p in enumerate(sorted(input_shapes))}
    expected = before(**kwargs)
    actual = after(**kwargs)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
