# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np


def lhsT_rhs_gemm_golden(lhsT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Golden reference for GEMM with transposed LHS.

    Args:
        lhsT: Transposed left-hand side matrix.
        rhs: Right-hand side matrix.

    Returns:
        Expected matmul result as float32.
    """
    return lhsT_rhs_gemm_np(lhsT, rhs).astype(np.float32)


def lhs_rhs_gemm_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Golden reference for standard GEMM.

    Args:
        lhs: Left-hand side matrix.
        rhs: Right-hand side matrix.

    Returns:
        Expected matmul result as float32.
    """
    return lhs_rhs_gemm_np(lhs, rhs).astype(np.float32)


def gemm_correctness_check(transposed_lhs: bool) -> tuple[Callable, float, float]:
    """Factory that returns a correctness_check tuple for GEMM validation.

    Args:
        transposed_lhs: Whether the LHS is delivered transposed.

    Returns:
        A (golden_fn, atol, rtol) tuple suitable for correctness_check.
    """
    golden_fn = lhsT_rhs_gemm_golden if transposed_lhs else lhs_rhs_gemm_golden
    return (golden_fn, 1e-5, 1e-2)


def lhs_rhs_gemm_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate GEMM between lhs and rhs.

    Args:
        lhs: Left-hand side matrix or tensor. Can have an extra batch dimension.
        rhs: Right-hand side matrix.

    Returns:
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate GEMM between transposed lhsT and rhs.

    Args:
        lhsT: Transposed left-hand side matrix or tensor. Can have an extra batch dimension.
        rhs: Right-hand side matrix.

    Returns:
        Result of the matrix multiplication.
    """
    if len(lhsT.shape) == 2:
        lhs = np.transpose(lhsT, (1, 0))
    elif len(lhsT.shape) == 3:
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        raise NotImplementedError(f"lhsT shape {lhsT.shape} is not supported in GEMM.")
    return np.matmul(lhs, rhs)
