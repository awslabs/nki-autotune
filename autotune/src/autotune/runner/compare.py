"""Numeric comparison with worst-margin reporting."""

import numpy as np


def assert_close(actual: np.ndarray, desired: np.ndarray, atol: float, rtol: float) -> str:
    """Assert two arrays are element-wise close and return a margin summary.

    Uses the same tolerance formula as ``np.testing.assert_allclose``:
    ``|actual - desired| <= atol + rtol * |desired|``.

    Args:
        actual: Array to check.
        desired: Reference array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Human-readable string with worst-case margin info.

    Raises:
        AssertionError: If any element exceeds the tolerance.
    """
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol)
    abs_diff = np.abs(actual.astype(np.float64) - desired.astype(np.float64))
    threshold = atol + rtol * np.abs(desired.astype(np.float64))
    ratio = abs_diff / threshold
    worst_idx = int(np.argmax(ratio))
    worst_diff = float(abs_diff.flat[worst_idx])
    worst_thresh = float(threshold.flat[worst_idx])
    worst_margin = float(ratio.flat[worst_idx])
    return f"PASS: atol+rtol*|desired|={worst_thresh:.2e}," f" |diff|={worst_diff:.2e}, worst_margin={worst_margin:.4f}"
