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
from nkigym.ops.add import NKIAdd
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.multiply import NKIMultiply
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar


def nc_matmul(*args: Any, **kwargs: Any) -> Any:
    """Matrix multiply: stationary.T @ moving.

    Returns:
        Numpy array result of shape [M, N].
    """
    stationary, moving = args[0], args[1]
    return np.matmul(stationary.T, moving)


def activation(*args: Any, **kwargs: Any) -> Any:
    """Apply element-wise activation function.

    Returns:
        Activated numpy array.
    """
    data = args[0]
    op_fn = kwargs.get("op")
    result = op_fn(data) if op_fn is not None else data
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


def tensor_scalar(*args: Any, **kwargs: Any) -> Any:
    """Element-wise op between a tensor and a column vector.

    Returns:
        Result numpy array (same shape as data).
    """
    data, operand0 = args[0], args[1]
    op_fn = kwargs.get("op0", np.add)
    return op_fn(data, operand0[..., np.newaxis])


def transpose(x: Any) -> Any:
    """Transpose a 2D tensor (dimension swap, no copy).

    Returns:
        Transposed numpy array.
    """
    return x.T


def ndarray(shape: tuple[int, ...], **kwargs: Any) -> np.ndarray:
    """Allocate a numpy array (simulation stub).

    Returns:
        Zero-initialized numpy array.
    """
    dtype = kwargs.get("dtype", np.float32)
    return np.zeros(shape, dtype=dtype)


__all__ = [
    "NKIOp",
    "NKIMatmul",
    "NKIActivation",
    "NKIAdd",
    "NKIDmaCopy",
    "NKIMultiply",
    "NKITensorCopy",
    "NKITensorReduce",
    "NKITensorScalar",
    "nc_matmul",
    "activation",
    "add",
    "multiply",
    "tensor_reduce",
    "tensor_scalar",
    "transpose",
    "ndarray",
]
