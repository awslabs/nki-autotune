"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Math function API: each nkigym.* call dispatches to the corresponding
NKIOp.__call__() for CPU simulation with numpy at float64 precision.

When tracing is active (via EagerTracer), calls are also recorded
for kernel generation.
"""

from typing import Any

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

_ACTIVE_TRACER: Any = None
_NAME_COUNTER: dict[str, int] = {}
_ARRAY_NAMES: dict[int, str] = {}

_OP_MATMUL = NKIMatmul()
_OP_TRANSPOSE = NKITranspose()
_OP_TENSOR_SCALAR = NKITensorScalar()
_OP_AFFINE_SELECT = NKIAffineSelect()
_OP_TENSOR_REDUCE = NKITensorReduce()
_OP_ACTIVATION_REDUCE = NKIActivationReduce()
_OP_ACTIVATION = NKIActivation()


def _fresh_name(base: str) -> str:
    """Generate a unique name for a traced tensor.

    Args:
        base: Base name (e.g. ``"Q_t"``).

    Returns:
        Unique name string.
    """
    count = _NAME_COUNTER.get(base, 0)
    _NAME_COUNTER[base] = count + 1
    if count == 0:
        result = base
    else:
        result = f"{base}_{count}"
    return result


def set_tracer(tracer: Any) -> None:
    """Set the active tracer for math function tracing.

    Args:
        tracer: An EagerTracer instance, or None to disable.
    """
    global _ACTIVE_TRACER
    _ACTIVE_TRACER = tracer


def reset_names() -> None:
    """Reset the name counter for traced tensors."""
    _NAME_COUNTER.clear()


def reset_array_names() -> None:
    """Reset the array name registry."""
    _ARRAY_NAMES.clear()


def _register_array(arr: np.ndarray, name: str) -> None:
    """Register a numpy array with a traced name.

    Args:
        arr: Numpy array.
        name: Tensor name.
    """
    _ARRAY_NAMES[id(arr)] = name


def _find_tensor_name(arr: np.ndarray) -> str:
    """Find the traced name for a numpy array.

    Args:
        arr: Numpy array.

    Returns:
        Tensor name, or empty string if not found.
    """
    return _ARRAY_NAMES.get(id(arr), "")


def _trace_result(
    op: NKIOp,
    operand_map: dict[str, str],
    operand_arrays: dict[str, np.ndarray],
    config_kwargs: dict[str, Any],
    output_names: list[str],
    result_arrays: list[np.ndarray],
) -> None:
    """Record a traced op and register its output arrays.

    Only called when ``_ACTIVE_TRACER`` is not None.

    Args:
        op: The NKIOp instance.
        operand_map: Maps operand slot to traced tensor name.
        operand_arrays: Maps operand slot to numpy array.
        config_kwargs: Non-tensor keyword arguments.
        output_names: Names for each output.
        result_arrays: Numpy result arrays.
    """
    _ACTIVE_TRACER.trace_op(
        op=op,
        operand_map=operand_map,
        operand_arrays=operand_arrays,
        config_kwargs=config_kwargs,
        output_names=output_names,
        result_arrays=result_arrays,
    )
    for arr, name in zip(result_arrays, output_names):
        _register_array(arr, name)


def nc_matmul(stationary: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Matrix multiply: stationary.T @ moving.

    Args:
        stationary: Array of shape (K, M).
        moving: Array of shape (K, N).

    Returns:
        Result array of shape (M, N).
    """
    result = _OP_MATMUL(stationary=stationary, moving=moving)
    if _ACTIVE_TRACER is not None:
        _trace_result(
            op=_OP_MATMUL,
            operand_map={"stationary": _find_tensor_name(stationary), "moving": _find_tensor_name(moving)},
            operand_arrays={"stationary": stationary, "moving": moving},
            config_kwargs={},
            output_names=[_fresh_name("S")],
            result_arrays=[result],
        )
    return result


def nc_transpose(data: np.ndarray) -> np.ndarray:
    """Transpose: swap partition and free dims.

    Args:
        data: Array of shape (P, F).

    Returns:
        Transposed array of shape (F, P).
    """
    result = _OP_TRANSPOSE(data=data)
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        out_name = f"{source_name}_t" if source_name else "T"
        _trace_result(
            op=_OP_TRANSPOSE,
            operand_map={"data": source_name},
            operand_arrays={"data": data},
            config_kwargs={},
            output_names=[_fresh_name(out_name)],
            result_arrays=[result],
        )
    return result


def tensor_scalar(data: np.ndarray, operand0: Any, *, op0: str, **kwargs: Any) -> np.ndarray:
    """Element-wise op between tensor and scalar/column vector.

    Args:
        data: Array of shape (P, F).
        operand0: Scalar or column vector. Can be positional or keyword.
        op0: Operation name (multiply, subtract, add).
        **kwargs: Additional keyword arguments (operand0 if not positional).

    Returns:
        Result array.
    """
    if operand0 is None:
        operand0 = kwargs.get("operand0")
    if operand0 is None:
        raise ValueError("operand0 is required")
    result = _OP_TENSOR_SCALAR(data=data, operand0=operand0, op0=op0)
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        operand_map: dict[str, str] = {"data": source_name}
        operand_arrays: dict[str, np.ndarray] = {"data": data}
        config: dict[str, Any] = {"op0": op0}
        if isinstance(operand0, np.ndarray):
            operand_map["operand0"] = _find_tensor_name(operand0)
            operand_arrays["operand0"] = operand0
        else:
            config["operand0"] = operand0
        _trace_result(
            op=_OP_TENSOR_SCALAR,
            operand_map=operand_map,
            operand_arrays=operand_arrays,
            config_kwargs=config,
            output_names=[_fresh_name(f"{source_name}_scaled" if source_name else "ts")],
            result_arrays=[result],
        )
    return result


def affine_select(
    data: np.ndarray, *, cmp_op: str, on_false_value: float, channel_multiplier: int, step: int
) -> np.ndarray:
    """Position-predicated element select.

    Args:
        data: Array of shape (P, F).
        cmp_op: Comparison operation.
        on_false_value: Value when predicate is false.
        channel_multiplier: P-axis scale factor.
        step: F-axis step.

    Returns:
        Result array.
    """
    result = _OP_AFFINE_SELECT(
        data=data, cmp_op=cmp_op, on_false_value=on_false_value, channel_multiplier=channel_multiplier, step=step
    )
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        _trace_result(
            op=_OP_AFFINE_SELECT,
            operand_map={"data": source_name},
            operand_arrays={"data": data},
            config_kwargs={
                "cmp_op": cmp_op,
                "on_false_value": on_false_value,
                "channel_multiplier": channel_multiplier,
                "step": step,
            },
            output_names=[_fresh_name(f"masked_{source_name}" if source_name else "masked")],
            result_arrays=[result],
        )
    return result


def tensor_reduce(data: np.ndarray, *, reduce_op: str, negate: bool) -> np.ndarray:
    """Reduce along the free axis.

    Args:
        data: Array of shape (P, F).
        reduce_op: Reduction operation (max, add).
        negate: Whether to negate the result.

    Returns:
        1D array of shape (P,).
    """
    result = _OP_TENSOR_REDUCE(data=data, op=reduce_op, negate=negate)
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        prefix = "neg_" if negate else ""
        _trace_result(
            op=_OP_TENSOR_REDUCE,
            operand_map={"data": source_name},
            operand_arrays={"data": data},
            config_kwargs={"reduce_op": reduce_op, "negate": negate},
            output_names=[_fresh_name(f"{prefix}{reduce_op}_{source_name}" if source_name else f"{prefix}{reduce_op}")],
            result_arrays=[result],
        )
    return result


def activation_reduce(data: np.ndarray, bias: np.ndarray, *, op: str, reduce_op: str) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise activation with simultaneous reduction.

    Args:
        data: Array of shape (P, F).
        bias: Column vector of shape (P,).
        op: Activation function name.
        reduce_op: Reduction operation.

    Returns:
        Tuple of (activation output, reduction result).
    """
    elem_result, reduce_result = _OP_ACTIVATION_REDUCE(data=data, bias=bias, op=op, reduce_op=reduce_op)
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        act_name = _fresh_name(f"{op}_{source_name}" if source_name else f"{op}_out")
        red_name = _fresh_name(f"{reduce_op}_{op}" if source_name else f"{reduce_op}_out")
        _trace_result(
            op=_OP_ACTIVATION_REDUCE,
            operand_map={"data": source_name, "bias": _find_tensor_name(bias)},
            operand_arrays={"data": data, "bias": bias},
            config_kwargs={"op": op, "reduce_op": reduce_op},
            output_names=[act_name, red_name],
            result_arrays=[elem_result, reduce_result],
        )
    return elem_result, reduce_result


def activation(data: np.ndarray, *, op: str) -> np.ndarray:
    """Element-wise unary activation.

    Args:
        data: Array of shape (P, F) or (P,).
        op: Activation function name.

    Returns:
        Activated array.
    """
    result = _OP_ACTIVATION(data=data, op=op)
    if _ACTIVE_TRACER is not None:
        source_name = _find_tensor_name(data)
        _trace_result(
            op=_OP_ACTIVATION,
            operand_map={"data": source_name},
            operand_arrays={"data": data},
            config_kwargs={"op": op},
            output_names=[_fresh_name(f"{op}_{source_name}" if source_name else f"{op}_out")],
            result_arrays=[result],
        )
    return result


__all__ = [
    "NKIOp",
    "NKIActivation",
    "NKIActivationReduce",
    "NKIAffineSelect",
    "NKIMatmul",
    "NKITensorReduce",
    "NKITensorScalar",
    "NKITranspose",
    "nc_matmul",
    "nc_transpose",
    "tensor_scalar",
    "affine_select",
    "tensor_reduce",
    "activation_reduce",
    "activation",
    "set_tracer",
    "reset_names",
    "reset_array_names",
]
