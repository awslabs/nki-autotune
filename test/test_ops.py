"""Numerical tests for NeuronOp simulate() methods."""

import numpy as np
import pytest
from conftest import make_random_array

from nkigym.ops import ActivationOp, MatmulOp, NcTransposeOp, TensorScalarOp, TensorTensorOp

MATMUL_CASES = [
    ("128x64_128x32", (128, 64), (128, 32)),
    ("128x128_128x128", (128, 128), (128, 128)),
    ("64x32_64x64", (64, 32), (64, 64)),
]

TRANSPOSE_SHAPES = [(128, 64), (64, 128), (32, 32)]

ELEMENTWISE_SHAPES = [(128, 128), (64, 32), (32, 64)]

TENSOR_TENSOR_OPS = [("np.add", "add"), ("np.multiply", "multiply"), ("np.subtract", "subtract")]

TENSOR_SCALAR_OPS = [("np.multiply", "multiply"), ("np.add", "add")]

ACTIVATION_OPS = [("np.tanh", "tanh"), (None, "identity")]


class TestMatmulOp:
    """Tests for MatmulOp.simulate(): lhs.T @ rhs."""

    @pytest.mark.parametrize("name,lhs_shape,rhs_shape", MATMUL_CASES, ids=[c[0] for c in MATMUL_CASES])
    def test_simulate(self, name: str, lhs_shape: tuple[int, int], rhs_shape: tuple[int, int]) -> None:
        """Verify simulate(lhs, rhs) == np.matmul(lhs.T, rhs)."""
        lhs = make_random_array(lhs_shape, seed=42)
        rhs = make_random_array(rhs_shape, seed=43)
        actual = MatmulOp().simulate(lhs, rhs)
        expected = np.matmul(lhs.T, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestNcTransposeOp:
    """Tests for NcTransposeOp.simulate(): transpose(data)."""

    @pytest.mark.parametrize("shape", TRANSPOSE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in TRANSPOSE_SHAPES])
    def test_simulate(self, shape: tuple[int, int]) -> None:
        """Verify simulate(data) == np.transpose(data)."""
        data = make_random_array(shape, seed=42)
        actual = NcTransposeOp().simulate(data)
        expected = np.transpose(data)
        np.testing.assert_array_equal(actual, expected)


class TestTensorTensorOp:
    """Tests for TensorTensorOp.simulate(): element-wise binary ops."""

    @pytest.mark.parametrize("op_repr,op_id", TENSOR_TENSOR_OPS, ids=[t[1] for t in TENSOR_TENSOR_OPS])
    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ELEMENTWISE_SHAPES])
    def test_simulate(self, shape: tuple[int, int], op_repr: str, op_id: str) -> None:
        """Verify simulate(a, b) matches the numpy operation."""
        a = make_random_array(shape, seed=42)
        b = make_random_array(shape, seed=43)
        op_fn = getattr(np, op_id)
        actual = TensorTensorOp().simulate(a, b, op=op_fn)
        expected = op_fn(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


class TestTensorScalarOp:
    """Tests for TensorScalarOp.simulate(): element-wise tensor-scalar ops."""

    @pytest.mark.parametrize("op_repr,op_id", TENSOR_SCALAR_OPS, ids=[t[1] for t in TENSOR_SCALAR_OPS])
    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ELEMENTWISE_SHAPES])
    def test_simulate(self, shape: tuple[int, int], op_repr: str, op_id: str) -> None:
        """Verify simulate(data, scalar) matches the numpy operation."""
        data = make_random_array(shape, seed=42)
        scalar = np.float32(2.5)
        op_fn = getattr(np, op_id)
        actual = TensorScalarOp().simulate(data, scalar, op=op_fn)
        expected = op_fn(data, scalar)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


class TestActivationOp:
    """Tests for ActivationOp.simulate(): element-wise activation."""

    @pytest.mark.parametrize("op_repr,op_id", ACTIVATION_OPS, ids=[t[1] for t in ACTIVATION_OPS])
    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ELEMENTWISE_SHAPES])
    def test_simulate(self, shape: tuple[int, int], op_repr: str | None, op_id: str) -> None:
        """Verify simulate(data) matches the numpy activation."""
        data = make_random_array(shape, seed=42)
        op_fn = getattr(np, op_id) if op_repr is not None else None
        actual = ActivationOp().simulate(data, op=op_fn)
        expected = op_fn(data) if op_fn is not None else data.copy()
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
