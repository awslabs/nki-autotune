"""Tests for NKIKernel simulator."""

import numpy as np
from conftest import make_random_array
from golden.nki_codegen import MATMUL_TANH_KERNEL

import nkigym
from nkigym.codegen import codegen, simulate


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256x256 matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


_SHAPE = (256, 256)


class TestSimulate:
    """Simulate an NKIKernel and compare to numpy reference."""

    def test_golden_kernel(self) -> None:
        """Simulate the golden matmul+tanh kernel against numpy reference."""
        a = make_random_array(_SHAPE, seed=42).astype(np.float16)
        b = make_random_array(_SHAPE, seed=43).astype(np.float16)
        expected = np.tanh(a.astype(np.float64).T @ b.astype(np.float64))
        actual = simulate(MATMUL_TANH_KERNEL, {"a": a, "b": b})
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_codegen_kernel(self) -> None:
        """Simulate a codegen-produced kernel against numpy reference."""
        a = make_random_array(_SHAPE, seed=44)
        b = make_random_array(_SHAPE, seed=45)
        kernel = codegen(matmul_tanh, {"a": a, "b": b})
        expected = np.tanh(a.astype(np.float64).T @ b.astype(np.float64))
        actual = simulate(kernel, {"a": a, "b": b})
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_psum_accumulation(self) -> None:
        """Verify PSUM accumulates across two reduction steps."""
        a = make_random_array(_SHAPE, seed=46)
        b = make_random_array(_SHAPE, seed=47)
        kernel = codegen(matmul_tanh, {"a": a, "b": b})
        actual = simulate(kernel, {"a": a, "b": b})
        expected = np.tanh(a.astype(np.float64).T @ b.astype(np.float64))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_output_dtype_matches_input(self) -> None:
        """Output dtype should match the first input dtype."""
        a = np.zeros(_SHAPE, dtype=np.float16)
        b = np.zeros(_SHAPE, dtype=np.float16)
        kernel = codegen(matmul_tanh, {"a": a, "b": b})
        result = simulate(kernel, {"a": a, "b": b})
        assert result.dtype == np.float64

    def test_output_shape(self) -> None:
        """Output shape matches kernel.output_shape."""
        a = np.zeros(_SHAPE, dtype=np.float16)
        b = np.zeros(_SHAPE, dtype=np.float16)
        kernel = codegen(matmul_tanh, {"a": a, "b": b})
        result = simulate(kernel, {"a": a, "b": b})
        assert result.shape == _SHAPE
