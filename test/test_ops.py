"""Unit tests for NKIOp.__call__ CPU simulation.

Each test calls a single op with hardcoded inputs from golden/ops_data.py
and asserts the output matches the hardcoded expected array.
"""

import numpy as np
from golden.ops_data import (
    ACTIVATION_BIAS_SCALE_BIAS,
    ACTIVATION_BIAS_SCALE_DATA,
    ACTIVATION_BIAS_SCALE_EXPECTED,
    ACTIVATION_BIAS_SCALE_SCALE,
    ACTIVATION_DATA_1,
    ACTIVATION_EXP_1,
    ACTIVATION_RECIPROCAL_1,
    ACTIVATION_REDUCE_BIAS_1,
    ACTIVATION_REDUCE_DATA_1,
    ACTIVATION_REDUCE_EXP_1,
    ACTIVATION_REDUCE_SUM_1,
    AFFINE_SELECT_DATA_1,
    AFFINE_SELECT_EXPECTED_1,
    AR_SCALE,
    AR_SCALE_DATA,
    AR_SCALE_ELEM_EXPECTED,
    AR_SCALE_RED_EXPECTED,
    AS_OFFSET_DATA,
    AS_OFFSET_EXPECTED,
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
    TR_KEEPDIMS_DATA,
    TR_KEEPDIMS_MAX_EXPECTED,
    TRANSPOSE_DATA_1,
    TRANSPOSE_EXPECTED_1,
    TS_COMPOUND_COL,
    TS_COMPOUND_DATA,
    TS_COMPOUND_EXPECTED,
    TS_REVERSE0_EXPECTED,
    TS_REVERSE_DATA,
    TS_REVERSE_OPERAND0,
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
    F = AFFINE_SELECT_DATA_1.shape[1]
    result = op(
        pattern=[[-1, F]],
        channel_multiplier=1,
        on_true_tile=AFFINE_SELECT_DATA_1,
        on_false_value=-np.inf,
        cmp_op="greater_equal",
    )
    np.testing.assert_array_equal(result, AFFINE_SELECT_EXPECTED_1)


def test_affine_select_preserves_shape() -> None:
    """NKIAffineSelect output shape matches input shape."""
    op = NKIAffineSelect()
    F = AFFINE_SELECT_DATA_1.shape[1]
    result = op(
        pattern=[[-1, F]],
        channel_multiplier=1,
        on_true_tile=AFFINE_SELECT_DATA_1,
        on_false_value=-np.inf,
        cmp_op="greater_equal",
    )
    assert result.shape == AFFINE_SELECT_DATA_1.shape


def test_tensor_reduce_max() -> None:
    """NKITensorReduce max along free axis."""
    op = NKITensorReduce()
    result = op(op="max", data=TENSOR_REDUCE_DATA_1, axis=1, negate=False)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_MAX_1)


def test_tensor_reduce_max_negate() -> None:
    """NKITensorReduce max with negate=True."""
    op = NKITensorReduce()
    result = op(op="max", data=TENSOR_REDUCE_DATA_1, axis=1, negate=True)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_NEG_MAX_1)


def test_tensor_reduce_sum() -> None:
    """NKITensorReduce sum along free axis."""
    op = NKITensorReduce()
    result = op(op="add", data=TENSOR_REDUCE_DATA_1, axis=1, negate=False)
    np.testing.assert_array_equal(result, TENSOR_REDUCE_SUM_1)


def test_tensor_reduce_output_1d() -> None:
    """NKITensorReduce output is 1D with shape (P,)."""
    op = NKITensorReduce()
    result = op(op="max", data=TENSOR_REDUCE_DATA_1, axis=1, negate=False)
    assert result.shape == (2,)


def test_activation_exp() -> None:
    """NKIActivation exp function."""
    op = NKIActivation()
    result = op(op="exp", data=ACTIVATION_DATA_1)
    np.testing.assert_allclose(result, ACTIVATION_EXP_1, rtol=1e-14)


def test_activation_reciprocal() -> None:
    """NKIActivation reciprocal function."""
    op = NKIActivation()
    result = op(op="reciprocal", data=ACTIVATION_DATA_1)
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


def test_activation_with_bias_and_scale() -> None:
    """NKIActivation exp(data * scale + bias) matches numpy reference."""
    op = NKIActivation()
    result = op(
        op="exp", data=ACTIVATION_BIAS_SCALE_DATA, bias=ACTIVATION_BIAS_SCALE_BIAS, scale=ACTIVATION_BIAS_SCALE_SCALE
    )
    np.testing.assert_allclose(result, ACTIVATION_BIAS_SCALE_EXPECTED, rtol=1e-14)


def test_tensor_scalar_reverse0() -> None:
    """NKITensorScalar reverse0: operand0 - data instead of data - operand0."""
    op = NKITensorScalar()
    result = op(data=TS_REVERSE_DATA, op0="subtract", operand0=TS_REVERSE_OPERAND0, reverse0=True)
    np.testing.assert_allclose(result, TS_REVERSE0_EXPECTED, rtol=1e-14)


def test_tensor_scalar_compound_op() -> None:
    """NKITensorScalar compound: data * col_vector + 5.0."""
    op = NKITensorScalar()
    result = op(data=TS_COMPOUND_DATA, op0="multiply", operand0=TS_COMPOUND_COL, op1="add", operand1=5.0)
    np.testing.assert_allclose(result, TS_COMPOUND_EXPECTED, rtol=1e-14)


def test_tensor_reduce_keepdims() -> None:
    """NKITensorReduce max with keepdims=True preserves 2D shape."""
    op = NKITensorReduce()
    result = op(op="max", data=TR_KEEPDIMS_DATA, axis=1, keepdims=True)
    np.testing.assert_allclose(result, TR_KEEPDIMS_MAX_EXPECTED, rtol=1e-14)
    assert result.shape == (2, 1)


def test_activation_reduce_scale_no_bias() -> None:
    """NKIActivationReduce exp(data * scale) with no bias."""
    op = NKIActivationReduce()
    elem_result, reduce_result = op(op="exp", data=AR_SCALE_DATA, reduce_op="add", scale=AR_SCALE)
    np.testing.assert_allclose(elem_result, AR_SCALE_ELEM_EXPECTED, rtol=1e-14)
    np.testing.assert_allclose(reduce_result, AR_SCALE_RED_EXPECTED, rtol=1e-14)


def test_affine_select_with_offset() -> None:
    """NKIAffineSelect with offset=1 shifts the causal boundary."""
    op = NKIAffineSelect()
    result = op(
        pattern=[[-1, 3]],
        channel_multiplier=1,
        on_true_tile=AS_OFFSET_DATA,
        on_false_value=-99.0,
        cmp_op="greater_equal",
        offset=1,
    )
    np.testing.assert_allclose(result, AS_OFFSET_EXPECTED, rtol=1e-14)
