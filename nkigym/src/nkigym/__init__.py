"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Pipeline: user function -> parse -> analysis -> schedule -> render -> NKI source

Subpackages:
    ops: Operator definitions and registry (NKIOp, NKIMatmul, etc.)
    codegen: AST parsing and dimension analysis
    schedule: Schedule descriptor, enumeration, render
    search: Combinatorial search with hardware benchmarking
    utils: Code generation helpers
"""

from typing import Any

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_1d import NKIActivation1D
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.add import NKIAdd
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.multiply import NKIMultiply
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_const import NKITensorScalarConst
from nkigym.ops.transpose import NKITranspose


def nc_matmul(*args: Any, **kwargs: Any) -> Any:
    """Matrix multiply: stationary.T @ moving.

    Returns:
        Numpy array result of shape [M, N].
    """
    stationary, moving = args[0], args[1]
    return np.matmul(stationary.T, moving)


def _rsqrt(x: Any) -> Any:
    """Reciprocal square root: 1 / sqrt(x)."""
    return 1.0 / np.sqrt(x)


def _reciprocal(x: Any) -> Any:
    """Reciprocal: 1 / x."""
    return 1.0 / x


_STR_OPS: dict[str, Any] = {"square": np.square, "rsqrt": _rsqrt, "reciprocal": _reciprocal}


def activation(*args: Any, **kwargs: Any) -> Any:
    """Apply element-wise activation, optionally with reduction.

    Returns:
        Activated numpy array, or reduced 1D array if reduce_op given.
    """
    data = args[0]
    op_fn = kwargs.get("op")
    if isinstance(op_fn, str):
        op_fn = _STR_OPS[op_fn]
    activated = op_fn(data) if op_fn is not None else data
    reduce_op = kwargs.get("reduce_op")
    result = reduce_op.reduce(activated, axis=-1) if reduce_op is not None else activated
    return result


def add(*args: Any, **kwargs: Any) -> Any:
    """Element-wise addition of two tensors.

    Returns:
        Sum of the two input arrays.
    """
    return args[0] + args[1]


def multiply(*args: Any, **kwargs: Any) -> Any:
    """Element-wise multiplication of two tensors.

    Returns:
        Product of the two input arrays.
    """
    return args[0] * args[1]


def tensor_reduce(*args: Any, **kwargs: Any) -> Any:
    """Reduce across the free axis of a tensor.

    Returns:
        Reduced 1D numpy array.
    """
    data = args[0]
    op_fn = kwargs.get("op", np.add)
    return op_fn.reduce(data, axis=-1)


def _expand_operand(data: Any, operand0: Any) -> Any:
    """Expand operand0 for broadcasting against data if needed."""
    result = operand0
    if isinstance(operand0, np.ndarray) and data.ndim > operand0.ndim:
        pad = data.ndim - operand0.ndim
        result = operand0.reshape(operand0.shape + (1,) * pad)
    return result


def tensor_scalar(*args: Any, **kwargs: Any) -> Any:
    """Element-wise op between a tensor and a scalar/column vector.

    Supports two modes:
    - 2D broadcast: ``tensor_scalar(data, tensor_operand, op0=...)``
    - 1D compound: ``tensor_scalar(data, op0=..., operand0=literal, ...)``

    Returns:
        Result numpy array.
    """
    data = args[0]
    op0 = kwargs.get("op0", np.add)
    operand0 = args[1] if len(args) > 1 else kwargs["operand0"]
    expanded = _expand_operand(data, operand0)
    result = op0(data, expanded)
    op1 = kwargs.get("op1")
    if op1 is not None:
        result = op1(result, kwargs["operand1"])
    return result


def nc_transpose(x: Any) -> Any:
    """PE array transpose (alias for transpose in simulation).

    Returns:
        Transposed numpy array.
    """
    return x.T


def transpose(x: Any) -> Any:
    """Transpose a 2D tensor (dimension swap, no copy).

    Returns:
        Transposed numpy array.
    """
    return x.T


def affine_select(
    data: Any, cmp_op: str = "greater_equal", on_false_value: float = 0.0, channel_multiplier: int = 1, step: int = -1
) -> Any:
    """Position-predicated element select using affine index pattern.

    Generates affine_value = p * channel_multiplier + f * step per
    element, compares to 0 with cmp_op, selects data or on_false_value.

    Returns:
        Masked numpy array.
    """
    rows, cols = data.shape
    p_idx = np.arange(rows)[:, np.newaxis]
    f_idx = np.arange(cols)[np.newaxis, :]
    cmp_fns: dict[str, Any] = {"greater_equal": np.greater_equal}
    mask = cmp_fns[cmp_op](p_idx * channel_multiplier + f_idx * step, 0)
    return np.where(mask, data, on_false_value)


def activation_reduce(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Apply element-wise activation then reduce across the free axis.

    Returns:
        Tuple of (activated array, reduced 1D array).
    """
    data = args[0]
    bias = args[1] if len(args) > 1 else 0.0
    op_name = kwargs.get("op", "exp")
    reduce_op_name = kwargs.get("reduce_op", "add")
    op_fns: dict[str, Any] = {"exp": np.exp, "square": np.square}
    reduce_fns: dict[str, Any] = {"add": np.add}
    shifted = data + _expand_operand(data, bias) if len(args) > 1 else data
    activated = op_fns[op_name](shifted)
    reduced = reduce_fns[reduce_op_name].reduce(activated, axis=-1)
    return activated, reduced


def ndarray(shape: tuple[int, ...], **kwargs: Any) -> np.ndarray:
    """Allocate a numpy array (simulation stub).

    Returns:
        Zero-initialized numpy array.
    """
    dtype = kwargs.get("dtype", np.float32)
    return np.zeros(shape, dtype=dtype)


__all__ = [
    "NKIOp",
    "NKIAffineSelect",
    "NKIMatmul",
    "NKIActivation",
    "NKIActivation1D",
    "NKIActivationReduce",
    "NKIAdd",
    "NKIDmaCopy",
    "NKIMultiply",
    "NKITensorCopy",
    "NKITensorReduce",
    "NKITensorScalar",
    "NKITensorScalarConst",
    "NKITranspose",
    "nc_matmul",
    "nc_transpose",
    "activation",
    "activation_reduce",
    "add",
    "affine_select",
    "multiply",
    "tensor_reduce",
    "tensor_scalar",
    "transpose",
    "ndarray",
]
