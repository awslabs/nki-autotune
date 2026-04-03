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
