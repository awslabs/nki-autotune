"""Numerical tests for nkigym user-facing op functions."""

import numpy as np
import pytest
from conftest import make_random_array

import nkigym

MATMUL_CASES = [
    ("128x64_128x32", (128, 64), (128, 32)),
    ("128x128_128x128", (128, 128), (128, 128)),
    ("64x32_64x64", (64, 32), (64, 64)),
]

ELEMENTWISE_SHAPES = [(128, 128), (64, 32), (32, 64)]

ACTIVATION_OPS = [("np.tanh", "tanh"), (None, "identity")]


class TestNKIMatmul:
    """Tests for nkigym.nc_matmul(): lhs.T @ rhs."""

    @pytest.mark.parametrize("name,lhs_shape,rhs_shape", MATMUL_CASES, ids=[c[0] for c in MATMUL_CASES])
    def test_simulate(self, name: str, lhs_shape: tuple[int, int], rhs_shape: tuple[int, int]) -> None:
        """Verify nc_matmul(lhs, rhs) == np.matmul(lhs.T, rhs)."""
        lhs = make_random_array(lhs_shape, seed=42)
        rhs = make_random_array(rhs_shape, seed=43)
        actual = nkigym.nc_matmul(lhs, rhs)
        expected = np.matmul(lhs.T, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestNKIActivation:
    """Tests for nkigym.activation(): element-wise activation."""

    @pytest.mark.parametrize("op_repr,op_id", ACTIVATION_OPS, ids=[t[1] for t in ACTIVATION_OPS])
    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ELEMENTWISE_SHAPES])
    def test_simulate(self, shape: tuple[int, int], op_repr: str, op_id: str) -> None:
        """Verify activation(data) matches the numpy activation."""
        data = make_random_array(shape, seed=42)
        op_fn = getattr(np, op_id) if op_repr is not None else None
        actual = nkigym.activation(data, op=op_fn)
        expected = op_fn(data) if op_fn is not None else data.copy()
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
