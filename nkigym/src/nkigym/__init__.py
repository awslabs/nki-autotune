"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Pipeline: user function -> codegen -> NKIKernel -> transforms -> render -> NKI source

Subpackages:
    ops: Operator definitions and registry (NKIOp, NKIMatmul, etc.)
    codegen: AST-based codegen producing NKIKernel directly
    transforms: Optimization passes on NKIKernel (data reuse, operand merge)
    search: Transform graph search with hardware benchmarking
    utils: Code generation helpers and logging
    ir: Tensor reference types (TensorRef)
"""

from typing import Any

import numpy as np

from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul


def nc_matmul(*args: Any, **kwargs: Any) -> Any:
    """Matrix multiply: stationary.T @ moving, with optional accumulation.

    Returns:
        Numpy array result of shape [M, N].
    """
    stationary, moving = args[0], args[1]
    acc = kwargs.get("acc")
    result = np.matmul(stationary.T, moving)
    if acc is not None:
        result = np.asarray(acc) + result
    return result


def activation(*args: Any, **kwargs: Any) -> Any:
    """Apply element-wise activation function.

    Returns:
        Activated numpy array.
    """
    data = args[0]
    op_fn = kwargs.get("op")
    result = op_fn(data) if op_fn is not None else data
    return result


__all__ = ["NKIOp", "NKIMatmul", "nc_matmul", "activation"]
