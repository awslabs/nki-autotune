"""Tests for the NKI CPU simulator."""

from test.golden.render_data import RENDER_1, RENDER_2, RENDER_3, RENDER_4, RENDER_5, RENDER_6

import numpy as np

from nkigym.simulate import simulate_kernel


def _random_f64(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate a deterministic random float64 array."""
    return np.random.RandomState(seed).randn(*shape)


def test_simulate_matmul() -> None:
    """Simulate 256x256 matmul kernel (RENDER_1) against numpy a.T @ b."""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    expected = a.T @ b
    actual = simulate_kernel(RENDER_1, "matmul_kernel", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_matmul_tanh() -> None:
    """Simulate matmul+tanh kernel (RENDER_2) against numpy tanh(a.T @ b)."""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    expected = np.tanh(a.T @ b)
    actual = simulate_kernel(RENDER_2, "matmul_tanh_kernel", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_add() -> None:
    """Simulate element-wise add kernel (RENDER_3) against numpy x + y."""
    x = _random_f64((256, 256), 42)
    y = _random_f64((256, 256), 43)
    expected = x + y
    actual = simulate_kernel(RENDER_3, "add_kernel", {"x": x, "y": y})
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_simulate_matmul_swapped() -> None:
    """Simulate swapped-loop matmul (RENDER_4) — same result as a.T @ b."""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    expected = a.T @ b
    actual = simulate_kernel(RENDER_4, "matmul_swapped_kernel", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_matmul_add() -> None:
    """Simulate matmul+add kernel (RENDER_5) against numpy a.T @ b + bias."""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    bias = _random_f64((256, 256), 44)
    expected = a.T @ b + bias
    actual = simulate_kernel(RENDER_5, "matmul_add_kernel", {"a": a, "b": b, "bias": bias})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_matmul_red_middle() -> None:
    """Simulate reduction-in-middle matmul (RENDER_6) — same result as a.T @ b."""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    expected = a.T @ b
    actual = simulate_kernel(RENDER_6, "matmul_red_middle_kernel", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_zero_kernel() -> None:
    """Verify that a no-op kernel returns zeros."""
    bad_source = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def bad_kernel(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    return hbm_tensor_0
"""
    a = _random_f64((256, 256), 42)
    b = _random_f64((256, 256), 43)
    result = simulate_kernel(bad_source, "bad_kernel", {"a": a, "b": b})
    np.testing.assert_array_equal(result, np.zeros((256, 256)))
