"""Unit tests for NKIOp.__call__ CPU simulation.

Each test calls a single op with hardcoded inputs from golden/ops_data.py
and asserts the output matches the hardcoded expected array.
"""

import numpy as np
from golden.ops_data import (
    ACTIVATION_DATA_1,
    ACTIVATION_EXP_1,
    ACTIVATION_RECIPROCAL_1,
    ACTIVATION_REDUCE_BIAS_1,
    ACTIVATION_REDUCE_DATA_1,
    ACTIVATION_REDUCE_EXP_1,
    ACTIVATION_REDUCE_SUM_1,
    AFFINE_SELECT_DATA_1,
    AFFINE_SELECT_EXPECTED_1,
    MATMUL_EXPECTED_1,
    MATMUL_MOVING_1,
    MATMUL_STATIONARY_1,
    TENSOR_REDUCE_DATA_1,
    TENSOR_REDUCE_MAX_1,
    TENSOR_REDUCE_NEG_MAX_1,
    TENSOR_REDUCE_SUM_1,
    TENSOR_SCALAR_COL_OPERAND_1,
    TENSOR_SCALAR_DATA_1,
    TENSOR_SCALAR_EXPECTED_ADD_COL_1,
    TENSOR_SCALAR_EXPECTED_MULTIPLY_1,
    TRANSPOSE_DATA_1,
    TRANSPOSE_EXPECTED_1,
)

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_matmul_stationary_t_times_moving() -> None:
    """NKIMatmul computes stationary.T @ moving = (2,3) result."""
    op = NKIMatmul()
    result = op(stationary=MATMUL_STATIONARY_1, moving=MATMUL_MOVING_1)
    np.testing.assert_array_equal(result, MATMUL_EXPECTED_1)


def test_matmul_output_shape() -> None:
    """NKIMatmul output shape is (M, N) from (K, M) and (K, N) inputs."""
    op = NKIMatmul()
    result = op(stationary=MATMUL_STATIONARY_1, moving=MATMUL_MOVING_1)
    assert result.shape == (2, 3)


def test_transpose_swap_axes() -> None:
    """NKITranspose swaps partition and free dims."""
    op = NKITranspose()
    result = op(data=TRANSPOSE_DATA_1)
    np.testing.assert_array_equal(result, TRANSPOSE_EXPECTED_1)


def test_transpose_output_shape() -> None:
    """NKITranspose output shape is (F, P) from (P, F) input."""
    op = NKITranspose()
    result = op(data=TRANSPOSE_DATA_1)
    assert result.shape == (3, 2)


def test_tensor_scalar_multiply_constant() -> None:
    """NKITensorScalar multiplies each element by scalar 3.0."""
    op = NKITensorScalar()
    result = op(data=TENSOR_SCALAR_DATA_1, operand0=3.0, op0="multiply")
    np.testing.assert_array_equal(result, TENSOR_SCALAR_EXPECTED_MULTIPLY_1)


def test_tensor_scalar_add_column_vector() -> None:
    """NKITensorScalar adds column vector with broadcast."""
    op = NKITensorScalar()
    result = op(data=TENSOR_SCALAR_DATA_1, operand0=TENSOR_SCALAR_COL_OPERAND_1, op0="add")
    np.testing.assert_array_equal(result, TENSOR_SCALAR_EXPECTED_ADD_COL_1)


def test_affine_select_causal_mask() -> None:
    """NKIAffineSelect applies causal mask with channel_multiplier=1, step=-1."""
    op = NKIAffineSelect()
    result = op(
        data=AFFINE_SELECT_DATA_1, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1
    )
    np.testing.assert_array_equal(result, AFFINE_SELECT_EXPECTED_1)


def test_affine_select_preserves_shape() -> None:
    """NKIAffineSelect output shape matches input shape."""
    op = NKIAffineSelect()
    result = op(
        data=AFFINE_SELECT_DATA_1, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1
    )
    assert result.shape == AFFINE_SELECT_DATA_1.shape


def test_tensor_reduce_max() -> None:
    """NKITensorReduce max along free axis."""
    op = NKITensorReduce()
    result = op(data=TENSOR_REDUCE_DATA_1, op="max", negate=False)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_MAX_1)


def test_tensor_reduce_max_negate() -> None:
    """NKITensorReduce max with negate=True."""
    op = NKITensorReduce()
    result = op(data=TENSOR_REDUCE_DATA_1, op="max", negate=True)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_NEG_MAX_1)


def test_tensor_reduce_sum() -> None:
    """NKITensorReduce sum along free axis."""
    op = NKITensorReduce()
    result = op(data=TENSOR_REDUCE_DATA_1, op="add", negate=False)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_SUM_1)


def test_tensor_reduce_output_1d() -> None:
    """NKITensorReduce output is 1D with shape (P,)."""
    op = NKITensorReduce()
    result = op(data=TENSOR_REDUCE_DATA_1, op="max", negate=False)
    assert result.shape == (2,)


def test_activation_exp() -> None:
    """NKIActivation exp function."""
    op = NKIActivation()
    result = op(data=ACTIVATION_DATA_1, op="exp")
    np.testing.assert_allclose(result, ACTIVATION_EXP_1, rtol=1e-14)


def test_activation_reciprocal() -> None:
    """NKIActivation reciprocal function."""
    op = NKIActivation()
    result = op(data=ACTIVATION_DATA_1, op="reciprocal")
    np.testing.assert_array_equal(result, ACTIVATION_RECIPROCAL_1)


def test_activation_reduce_exp_sum() -> None:
    """NKIActivationReduce exp activation with sum reduction."""
    op = NKIActivationReduce()
    elem_result, reduce_result = op(
        data=ACTIVATION_REDUCE_DATA_1, bias=ACTIVATION_REDUCE_BIAS_1, op="exp", reduce_op="add"
    )
    np.testing.assert_allclose(elem_result, ACTIVATION_REDUCE_EXP_1, rtol=1e-14)
    np.testing.assert_allclose(reduce_result, ACTIVATION_REDUCE_SUM_1, rtol=1e-14)


def test_activation_reduce_dual_output() -> None:
    """NKIActivationReduce returns both activation and reduction arrays."""
    op = NKIActivationReduce()
    elem_result, reduce_result = op(
        data=ACTIVATION_REDUCE_DATA_1, bias=ACTIVATION_REDUCE_BIAS_1, op="exp", reduce_op="add"
    )
    assert elem_result.shape == (2, 2)
    assert reduce_result.shape == (2,)
