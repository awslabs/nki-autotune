"""Golden data for NKIOp.__call__ unit tests.

All values are hardcoded float64 literals. No codegen or runtime computation.
"""

import numpy as np

MATMUL_STATIONARY_1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
"""Shape (3, 2) = (K=3, M=2)."""

MATMUL_MOVING_1 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
"""Shape (3, 3) = (K=3, N=3)."""

MATMUL_EXPECTED_1 = np.array([[6.0, 8.0, 7.0], [8.0, 10.0, 10.0]], dtype=np.float64)
"""stationary.T @ moving = (2, 3) @ (3, 3) -> (2, 3)."""

TRANSPOSE_DATA_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
"""Shape (2, 3)."""

TRANSPOSE_EXPECTED_1 = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float64)
"""Shape (3, 2)."""

TENSOR_SCALAR_DATA_1 = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
TENSOR_SCALAR_EXPECTED_MULTIPLY_1 = np.array([[6.0, 12.0], [18.0, 24.0]], dtype=np.float64)
"""data * 3.0."""

TENSOR_SCALAR_COL_OPERAND_1 = np.array([10.0, 20.0], dtype=np.float64)
TENSOR_SCALAR_EXPECTED_ADD_COL_1 = np.array([[12.0, 14.0], [26.0, 28.0]], dtype=np.float64)
"""data + column_vector (broadcast)."""

AFFINE_SELECT_DATA_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
AFFINE_SELECT_EXPECTED_1 = np.array([[1.0, -np.inf, -np.inf], [4.0, 5.0, -np.inf], [7.0, 8.0, 9.0]], dtype=np.float64)
"""Causal mask: channel_multiplier=1, step=-1."""

TENSOR_REDUCE_DATA_1 = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64)
TENSOR_REDUCE_MAX_1 = np.array([5.0, 6.0], dtype=np.float64)
TENSOR_REDUCE_NEG_MAX_1 = np.array([-5.0, -6.0], dtype=np.float64)
TENSOR_REDUCE_SUM_1 = np.array([9.0, 12.0], dtype=np.float64)

ACTIVATION_DATA_1 = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
ACTIVATION_EXP_1 = np.array([[np.exp(0.0), np.exp(1.0)], [np.exp(-1.0), np.exp(2.0)]], dtype=np.float64)
ACTIVATION_RECIPROCAL_1 = np.array([[np.inf, 1.0], [-1.0, 0.5]], dtype=np.float64)
"""reciprocal(0) = inf, reciprocal(1) = 1, reciprocal(-1) = -1, reciprocal(2) = 0.5."""

ACTIVATION_REDUCE_DATA_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
ACTIVATION_REDUCE_BIAS_1 = np.array([-1.0, -3.0], dtype=np.float64)
ACTIVATION_REDUCE_EXP_1 = np.array([[np.exp(0.0), np.exp(1.0)], [np.exp(0.0), np.exp(1.0)]], dtype=np.float64)
"""exp(data + bias[:, None]) = exp([[0, 1], [0, 1]])."""

ACTIVATION_REDUCE_SUM_1 = np.array([1.0 + np.e, 1.0 + np.e], dtype=np.float64)
"""rowsum of exp result."""

"""
--- New feature test data below ---
"""

ACTIVATION_BIAS_SCALE_DATA = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
ACTIVATION_BIAS_SCALE_BIAS = np.array([0.5, -0.5], dtype=np.float64)
ACTIVATION_BIAS_SCALE_SCALE = 2.0
ACTIVATION_BIAS_SCALE_EXPECTED = np.array(
    [[np.exp(0.0 * 2.0 + 0.5), np.exp(1.0 * 2.0 + 0.5)], [np.exp(-1.0 * 2.0 - 0.5), np.exp(2.0 * 2.0 - 0.5)]],
    dtype=np.float64,
)
"""exp(data * 2.0 + bias[:, None])."""

TS_REVERSE_DATA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
TS_REVERSE_OPERAND0 = 10.0
TS_REVERSE0_EXPECTED = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float64)
"""reverse0: operand0 - data = 10 - data."""

TS_COMPOUND_DATA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
TS_COMPOUND_COL = np.array([100.0, 200.0], dtype=np.float64)
TS_COMPOUND_EXPECTED = np.array([[105.0, 205.0, 305.0], [805.0, 1005.0, 1205.0]], dtype=np.float64)
"""data * col_vec + 5.0 (compound op: multiply then add)."""

TR_KEEPDIMS_DATA = np.array([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], dtype=np.float64)
TR_KEEPDIMS_MAX_EXPECTED = np.array([[3.0], [6.0]], dtype=np.float64)
"""max along axis=1 with keepdims=True."""

AR_SCALE_DATA = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
AR_SCALE = 0.5
AR_SCALE_ELEM_EXPECTED = np.array(
    [[np.exp(1.0 * 0.5), np.exp(2.0 * 0.5)], [np.exp(3.0 * 0.5), np.exp(4.0 * 0.5)]], dtype=np.float64
)
"""exp(data * 0.5), no bias."""
AR_SCALE_RED_EXPECTED = np.array([np.exp(0.5) + np.exp(1.0), np.exp(1.5) + np.exp(2.0)], dtype=np.float64)
"""rowsum of exp(data * 0.5)."""

AS_OFFSET_DATA = np.ones((3, 3), dtype=np.float64)
AS_OFFSET_EXPECTED = np.array([[1.0, 1.0, -99.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
"""affine_select with offset=1, channel_multiplier=1, step=-1, cmp_op=greater_equal."""
