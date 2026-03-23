"""Golden reference implementations and correctness checking for GEMM."""

from collections.abc import Callable

import numpy as np


def lhsT_rhs_gemm_golden(**kwargs: np.ndarray) -> np.ndarray:
    """Golden reference for GEMM with transposed LHS.

    Accepts keyword arguments matching either rendered kernel (a, b)
    or MetaGEMM kernel (lhs, rhs, config) calling conventions.

    Args:
        **kwargs: Tensor inputs keyed by parameter name.

    Returns:
        Expected matmul result as float32.
    """
    lhs, rhs = _extract_gemm_inputs(kwargs)
    return lhsT_rhs_gemm_np(lhs, rhs).astype(np.float32)


def lhs_rhs_gemm_golden(**kwargs: np.ndarray) -> np.ndarray:
    """Golden reference for standard GEMM.

    Accepts keyword arguments matching either rendered kernel (a, b)
    or MetaGEMM kernel (lhs, rhs, config) calling conventions.

    Args:
        **kwargs: Tensor inputs keyed by parameter name.

    Returns:
        Expected matmul result as float32.
    """
    lhs, rhs = _extract_gemm_inputs(kwargs)
    return lhs_rhs_gemm_np(lhs, rhs).astype(np.float32)


def _extract_gemm_inputs(kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract LHS and RHS arrays from keyword arguments.

    Supports both (a, b) and (lhs, rhs) naming conventions.

    Args:
        kwargs: Keyword arguments containing the two matrix inputs.

    Returns:
        Tuple of (lhs, rhs) numpy arrays.

    Raises:
        KeyError: If neither naming convention is found.
    """
    name_map = {"a": ("a", "b"), "lhs": ("lhs", "rhs")}
    for key, (lhs_key, rhs_key) in name_map.items():
        if key in kwargs:
            return kwargs[lhs_key], kwargs[rhs_key]
    raise KeyError(f"Expected (a, b) or (lhs, rhs) keys, got: {list(kwargs.keys())}")


def gemm_correctness_check(transposed_lhs: bool) -> tuple[Callable, float, float]:
    """Factory that returns a correctness_check tuple for GEMM validation.

    Args:
        transposed_lhs: Whether the LHS is delivered transposed.

    Returns:
        A (golden_fn, atol, rtol) tuple suitable for correctness_check.
    """
    golden_fn = lhsT_rhs_gemm_golden if transposed_lhs else lhs_rhs_gemm_golden
    return (golden_fn, 1e-2, 1e-2)


def lhs_rhs_gemm_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate GEMM between lhs and rhs.

    Args:
        lhs: Left-hand side matrix.
        rhs: Right-hand side matrix.

    Returns:
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate GEMM between transposed lhsT and rhs.

    Args:
        lhsT: Transposed left-hand side matrix.
        rhs: Right-hand side matrix.

    Returns:
        Result of the matrix multiplication.

    Raises:
        NotImplementedError: If lhsT rank is not 2 or 3.
    """
    if len(lhsT.shape) == 2:
        lhs = np.transpose(lhsT, (1, 0))
    elif len(lhsT.shape) == 3:
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        raise NotImplementedError(f"lhsT shape {lhsT.shape} is not supported in GEMM.")
    return np.matmul(lhs, rhs)
