"""Shape-only symbolic tracing for the supported NumPy synthesis subset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from nkigym.synthesis._trace_shape import (
    axes_argument,
    normalize_axes,
    normalize_reshape,
    normalize_transpose_axes,
    one_axis,
    optional_axes_argument,
    shape_argument,
)


@dataclass(frozen=True)
class Expr:
    """One immutable tensor expression in a traced NumPy program."""

    op: str
    shape: tuple[int, ...]
    args: tuple[Expr | float, ...] = ()
    name: str | None = None
    axis: int | None = None


class TraceTensor(NDArrayOperatorsMixin):
    """NumPy-compatible symbolic tensor that records supported operations."""

    __array_priority__ = 1000

    def __init__(self, expression: Expr) -> None:
        """Wrap one symbolic expression."""
        self.expression = expression

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the statically known tensor shape."""
        return self.expression.shape

    @property
    def ndim(self) -> int:
        """Return the statically known tensor rank."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the statically known element count."""
        return int(np.prod(self.shape))

    @property
    def dtype(self) -> np.dtype[np.float32]:
        """Expose the fp32 tracing dtype."""
        return np.dtype(np.float32)

    @property
    def T(self) -> TraceTensor:
        """Transpose a one- or two-dimensional tensor."""
        return self.transpose()

    def astype(
        self,
        dtype: np.dtype[Any] | type[Any] | str,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> TraceTensor:
        """Discard a floating-point cast while preserving symbolic math."""
        _ = (order, casting, subok, copy)
        target = np.dtype(dtype)
        if target.kind != "f":
            raise ValueError(f"programmatic synthesis only supports floating-point casts, got {target}")
        return self

    def transpose(self, *axes: int | tuple[int, ...]) -> TraceTensor:
        """Return a symbolic transpose for rank-two tensors."""
        if self.ndim == 1:
            result = self
        else:
            normalized = normalize_transpose_axes(self.ndim, axes)
            if self.ndim != 2 or normalized != (1, 0):
                raise ValueError(f"programmatic synthesis only supports 2D transpose, got axes={normalized}")
            result = TraceTensor(Expr(op="transpose", shape=(self.shape[1], self.shape[0]), args=(self.expression,)))
        return result

    def reshape(self, *shape: int | tuple[int, ...], order: str = "C") -> TraceTensor:
        """Return a singleton-only symbolic reshape."""
        if order != "C":
            raise ValueError("programmatic synthesis only supports C-order reshape")
        if len(shape) == 1 and isinstance(shape[0], tuple):
            requested = shape[0]
        elif all(isinstance(dimension, int) for dimension in shape):
            requested = tuple(dimension for dimension in shape if isinstance(dimension, int))
        else:
            raise ValueError(f"invalid reshape target {shape}")
        normalized = normalize_reshape(self.shape, requested)
        return _view(self, normalized)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> TraceTensor:
        """Remove singleton dimensions without emitting an NKI operation."""
        if axis is None:
            squeezed = tuple(dimension for dimension in self.shape if dimension != 1)
        else:
            axes = normalize_axes(axis, self.ndim)
            if any(self.shape[index] != 1 for index in axes):
                raise ValueError(f"cannot squeeze non-singleton shape {self.shape} at axes {axes}")
            squeezed = tuple(dimension for index, dimension in enumerate(self.shape) if index not in axes)
        return _view(self, squeezed)

    def __getitem__(self, index: object) -> TraceTensor:
        """Support full slices plus ``None`` singleton insertion."""
        items = index if isinstance(index, tuple) else (index,)
        expanded = _expand_ellipsis(items, self.ndim)
        source_axis = 0
        output_shape: list[int] = []
        for item in expanded:
            if item is None:
                output_shape.append(1)
            elif isinstance(item, slice) and item == slice(None):
                if source_axis >= self.ndim:
                    raise IndexError(f"too many indices for shape {self.shape}")
                output_shape.append(self.shape[source_axis])
                source_axis += 1
            else:
                raise ValueError("programmatic synthesis only supports full slices and None indexing")
        output_shape.extend(self.shape[source_axis:])
        return _view(self, tuple(output_shape))

    def __array__(self, dtype: np.dtype[Any] | None = None, copy: bool | None = None) -> np.ndarray:
        """Reject conversion because symbolic tensors have no allocated data."""
        _ = (dtype, copy)
        raise ValueError("programmatic synthesis cannot materialize a symbolic NumPy array")

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object) -> TraceTensor:
        """Trace supported NumPy ufunc calls."""
        if method != "__call__":
            raise ValueError(f"programmatic synthesis does not support ufunc method {ufunc.__name__}.{method}")
        options = dict(kwargs)
        output = options.pop("out", None)
        if output is not None:
            if not isinstance(output, tuple) or len(output) != 1 or output[0] is not self:
                raise ValueError("programmatic synthesis only supports in-place assignment to the left operand")
        if options:
            raise ValueError(f"unsupported {ufunc.__name__} options: {sorted(options)}")
        result = _trace_ufunc(ufunc, inputs)
        return result

    def __array_function__(
        self,
        function: Callable[..., object],
        types: tuple[type[Any], ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> TraceTensor:
        """Trace supported high-level NumPy functions."""
        _ = types
        result = _trace_array_function(function, args, kwargs)
        return result

    def __bool__(self) -> bool:
        """Reject data-dependent Python control flow."""
        raise ValueError("programmatic synthesis does not support data-dependent Python control flow")


def trace_numpy(function: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> Expr:
    """Execute a NumPy function once on symbolic inputs and return its expression."""
    symbolic_inputs = {
        name: TraceTensor(Expr(op="input", shape=tuple(shape), name=name))
        for name, (shape, _dtype) in input_specs.items()
    }
    try:
        result = function(**symbolic_inputs)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"failed to symbolically trace {function.__name__}: {error}") from error
    if not isinstance(result, TraceTensor):
        raise ValueError(f"{function.__name__} must return one NumPy tensor")
    if len(result.shape) != 2:
        raise ValueError(f"nkigym synthesis requires a rank-two output, got shape {result.shape}")
    return result.expression


def _trace_ufunc(ufunc: np.ufunc, inputs: tuple[object, ...]) -> TraceTensor:
    """Map one supported ufunc to a symbolic expression."""
    unary = {
        np.square: "square",
        np.sqrt: "sqrt",
        np.exp: "exp",
        np.tanh: "tanh",
        np.reciprocal: "reciprocal",
        np.negative: "negative",
        np.positive: "positive",
    }
    binary = {
        np.add: "add",
        np.subtract: "subtract",
        np.multiply: "multiply",
        np.true_divide: "divide",
        np.divide: "divide",
        np.maximum: "maximum",
        np.power: "power",
        np.matmul: "matmul",
    }
    if ufunc in unary and len(inputs) == 1:
        result = _unary_expression(unary[ufunc], _expression(inputs[0]))
    elif ufunc in binary and len(inputs) == 2:
        result = _binary_expression(binary[ufunc], _operand(inputs[0]), _operand(inputs[1]))
    else:
        raise ValueError(f"programmatic synthesis does not support NumPy ufunc {ufunc.__name__}")
    return TraceTensor(result)


def _trace_array_function(
    function: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]
) -> TraceTensor:
    """Map one supported NumPy function to a symbolic expression."""
    if function in {np.sum, np.max, np.amax, np.mean}:
        result = _trace_reduction(function, args, kwargs)
    elif function is np.transpose:
        tensor = _tensor_argument(args, 0, function)
        axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
        result = tensor.transpose() if axes is None else tensor.transpose(axes_argument(axes))
    elif function is np.reshape:
        tensor = _tensor_argument(args, 0, function)
        shape = kwargs.get("newshape", args[1] if len(args) > 1 else None)
        if shape is None:
            raise ValueError("np.reshape requires a target shape")
        result = tensor.reshape(shape_argument(shape), order=str(kwargs.get("order", "C")))
    elif function is np.squeeze:
        tensor = _tensor_argument(args, 0, function)
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        result = tensor.squeeze(optional_axes_argument(axis))
    elif function is np.expand_dims:
        tensor = _tensor_argument(args, 0, function)
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        if axis is None:
            raise ValueError("np.expand_dims requires axis")
        result = _expand_dims(tensor, axis)
    elif function is np.copy:
        result = _tensor_argument(args, 0, function)
    else:
        raise ValueError(f"programmatic synthesis does not support NumPy function {function.__name__}")
    return result


def _trace_reduction(
    function: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]
) -> TraceTensor:
    """Trace a one-axis sum, maximum, or mean."""
    tensor = _tensor_argument(args, 0, function)
    options = dict(kwargs)
    axis = options.pop("axis", args[1] if len(args) > 1 else None)
    dtype = options.pop("dtype", None)
    output = options.pop("out", None)
    keepdims = bool(options.pop("keepdims", False))
    initial = options.pop("initial", None)
    where = options.pop("where", True)
    if options or output is not None or initial is not None or where is not True:
        raise ValueError(f"unsupported {function.__name__} options")
    if dtype is not None:
        if not isinstance(dtype, (str, type, np.dtype)):
            raise ValueError(f"{function.__name__} received invalid dtype {dtype!r}")
        if np.dtype(dtype).kind != "f":
            raise ValueError(f"{function.__name__} only supports floating-point dtype")
    normalized_axis = one_axis(axis, tensor.ndim)
    reduced_shape = tuple(
        1 if keepdims and index == normalized_axis else dimension
        for index, dimension in enumerate(tensor.shape)
        if keepdims or index != normalized_axis
    )
    op = "reduce_max" if function in {np.max, np.amax} else "reduce_add"
    reduced = Expr(op=op, shape=reduced_shape, args=(tensor.expression,), axis=normalized_axis)
    if function is np.mean:
        reduced = _binary_expression("multiply", reduced, 1.0 / tensor.shape[normalized_axis])
    return TraceTensor(reduced)


def _binary_expression(op: str, left: Expr | float, right: Expr | float) -> Expr:
    """Build and simplify one binary expression."""
    if op == "matmul":
        result = _matmul_expression(left, right)
    elif op == "power":
        result = _power_expression(left, right)
    elif op == "divide":
        result = _division_expression(left, right)
    elif op == "add" and _is_scalar(right, 0.0):
        result = _require_expression(left, op)
    elif op == "add" and _is_scalar(left, 0.0):
        result = _require_expression(right, op)
    elif op == "multiply" and _is_scalar(right, 1.0):
        result = _require_expression(left, op)
    elif op == "multiply" and _is_scalar(left, 1.0):
        result = _require_expression(right, op)
    else:
        shape = _broadcast_shape(left, right, op)
        result = Expr(op=op, shape=shape, args=(left, right))
    return result


def _division_expression(left: Expr | float, right: Expr | float) -> Expr:
    """Rewrite division into multiply and reciprocal primitives."""
    if not isinstance(right, Expr):
        scalar = float(right)
        if scalar == 0.0:
            raise ValueError("programmatic synthesis does not support division by zero")
        result = _binary_expression("multiply", left, 1.0 / scalar)
    else:
        reciprocal_op = "rsqrt" if right.op == "sqrt" else "reciprocal"
        reciprocal_arg = _require_expression(right.args[0], "rsqrt") if reciprocal_op == "rsqrt" else right
        reciprocal = _unary_expression(reciprocal_op, reciprocal_arg)
        result = _binary_expression("multiply", left, reciprocal)
    return result


def _power_expression(left: Expr | float, right: Expr | float) -> Expr:
    """Map supported scalar powers to unary activation primitives."""
    expression = _require_expression(left, "power")
    if not isinstance(right, float):
        raise ValueError("programmatic synthesis requires a scalar exponent")
    operations = {2.0: "square", 0.5: "sqrt", -0.5: "rsqrt", -1.0: "reciprocal"}
    operation = operations.get(right)
    if operation is None:
        raise ValueError(f"programmatic synthesis does not support exponent {right}")
    return _unary_expression(operation, expression)


def _matmul_expression(left: Expr | float, right: Expr | float) -> Expr:
    """Build a rank-two matrix product with checked dimensions."""
    lhs = _require_expression(left, "matmul")
    rhs = _require_expression(right, "matmul")
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise ValueError(f"programmatic synthesis only supports 2D matmul, got {lhs.shape} @ {rhs.shape}")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"matmul dimension mismatch: {lhs.shape} @ {rhs.shape}")
    return Expr(op="matmul", shape=(lhs.shape[0], rhs.shape[1]), args=(lhs, rhs))


def _unary_expression(op: str, expression: Expr) -> Expr:
    """Build and simplify one unary expression."""
    if op == "positive":
        result = expression
    elif op == "negative":
        result = _binary_expression("multiply", expression, -1.0)
    else:
        result = Expr(op=op, shape=expression.shape, args=(expression,))
    return result


def _broadcast_shape(left: Expr | float, right: Expr | float, op: str) -> tuple[int, ...]:
    """Return NumPy's broadcast shape or raise a synthesis-specific error."""
    left_shape = left.shape if isinstance(left, Expr) else ()
    right_shape = right.shape if isinstance(right, Expr) else ()
    try:
        shape = tuple(np.broadcast_shapes(left_shape, right_shape))
    except ValueError as error:
        raise ValueError(f"{op} operands do not broadcast: {left_shape} and {right_shape}") from error
    return shape


def _operand(value: object) -> Expr | float:
    """Convert a symbolic tensor or real scalar into an expression operand."""
    if isinstance(value, TraceTensor):
        operand: Expr | float = value.expression
    elif isinstance(value, Real) and not isinstance(value, bool):
        operand = float(value)
    elif isinstance(value, np.generic) and np.issubdtype(value.dtype, np.floating):
        operand = float(value)
    else:
        raise ValueError(f"programmatic synthesis does not support operand type {type(value).__name__}")
    return operand


def _expression(value: object) -> Expr:
    """Require a symbolic tensor and return its expression."""
    operand = _operand(value)
    return _require_expression(operand, "unary operation")


def _require_expression(operand: Expr | float, operation: str) -> Expr:
    """Require a tensor operand for one operation."""
    if not isinstance(operand, Expr):
        raise ValueError(f"{operation} requires a tensor operand")
    return operand


def _is_scalar(operand: Expr | float, expected: float) -> bool:
    """Return whether an operand is one exact scalar."""
    return isinstance(operand, float) and operand == expected


def _tensor_argument(args: tuple[object, ...], index: int, function: Callable[..., object]) -> TraceTensor:
    """Return one required symbolic tensor argument."""
    if len(args) <= index:
        raise ValueError(f"{function.__name__} requires a symbolic tensor argument")
    value = args[index]
    if not isinstance(value, TraceTensor):
        raise ValueError(f"{function.__name__} requires a symbolic tensor argument")
    return value


def _view(tensor: TraceTensor, shape: tuple[int, ...]) -> TraceTensor:
    """Return a metadata-only shape view."""
    if shape == tensor.shape:
        result = tensor
    else:
        result = TraceTensor(Expr(op="view", shape=shape, args=(tensor.expression,)))
    return result


def _expand_dims(tensor: TraceTensor, axis: object) -> TraceTensor:
    """Insert one or more singleton dimensions."""
    axes = axes_argument(axis)
    rank = tensor.ndim + len(axes)
    normalized = tuple(item + rank if item < 0 else item for item in axes)
    if any(item < 0 or item >= rank for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"invalid expand_dims axes {axes} for rank {tensor.ndim}")
    shape = list(tensor.shape)
    for item in sorted(normalized):
        shape.insert(item, 1)
    return _view(tensor, tuple(shape))


def _expand_ellipsis(items: tuple[object, ...], rank: int) -> tuple[object, ...]:
    """Replace at most one ellipsis with the required full slices."""
    ellipses = sum(item is Ellipsis for item in items)
    if ellipses > 1:
        raise IndexError("an index can only have one ellipsis")
    if ellipses == 0:
        expanded = items
    else:
        consumed = sum(item is not None and item is not Ellipsis for item in items)
        fill = rank - consumed
        expanded_items: list[object] = []
        for item in items:
            if item is Ellipsis:
                expanded_items.extend(slice(None) for _ in range(fill))
            else:
                expanded_items.append(item)
        expanded = tuple(expanded_items)
    return expanded


__all__ = ["Expr", "trace_numpy"]
