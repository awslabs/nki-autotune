"""Deterministic lowering from symbolic NumPy expressions to nkigym source."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nkigym.synthesis._trace import Expr

_OP_MODULES = {
    "NKIActivation": "activation",
    "NKIActivationReduce": "activation_reduce",
    "NKIDMATranspose": "dma_transpose",
    "NKILoad": "load",
    "NKIMatmul": "matmul",
    "NKIStore": "store",
    "NKITensorCopy": "tensor_copy",
    "NKITensorReduce": "tensor_reduce",
    "NKITensorScalar": "tensor_scalar",
    "NKITensorTensor": "tensor_tensor",
}


@dataclass(frozen=True)
class _Value:
    """One emitted SBUF value and its logical NumPy shape."""

    name: str
    shape: tuple[int, ...]


class _Lowerer:
    """Lower one expression DAG into deterministic nkigym source lines."""

    def __init__(self, input_specs: dict[str, tuple[tuple[int, ...], str]]) -> None:
        """Initialize source state and emit one load per input."""
        self._input_specs = input_specs
        self._imports: set[str] = {"NKILoad", "NKIStore"}
        self._body: list[str] = []
        self._cache: dict[Expr, _Value] = {}
        self._inputs: dict[str, _Value] = {}
        self._counters: dict[str, int] = {}
        self._consumer_counts: dict[Expr, int] = {}
        for name, (shape, _dtype) in input_specs.items():
            value = _Value(name=f"sbuf_{name}", shape=tuple(shape))
            self._inputs[name] = value
            self._body.append(f"{value.name} = NKILoad()(src={name})")

    def build(self, expression: Expr) -> str:
        """Lower the output, store it, and render complete Python source."""
        self._consumer_counts = _count_consumers(expression)
        output = self.lower(expression)
        self._body.append(f"hbm_output = NKIStore()(src={output.name})")
        self._body.append("return hbm_output")
        imports = ["from nkigym.ops import nkigym_kernel"]
        imports.extend(
            f"from nkigym.ops.{_OP_MODULES[class_name]} import {class_name}" for class_name in sorted(self._imports)
        )
        parameters = ", ".join(self._input_specs)
        indented_body = "\n".join(f"    {line}" for line in self._body)
        return (
            "\n".join(imports)
            + "\n\n\n@nkigym_kernel\n"
            + f"def f_nkigym({parameters}):\n"
            + '    """Programmatically synthesized nkigym operator graph."""\n'
            + indented_body
            + "\n"
        )

    def lower(self, expression: Expr) -> _Value:
        """Lower one expression once and reuse common subexpressions."""
        cached = self._cache.get(expression)
        if cached is None:
            lowerers = {
                "input": self._lower_input,
                "view": self._lower_view,
                "transpose": self._lower_transpose,
                "matmul": self._lower_matmul,
                "square": self._lower_activation,
                "sqrt": self._lower_activation,
                "rsqrt": self._lower_activation,
                "exp": self._lower_activation,
                "tanh": self._lower_activation,
                "reciprocal": self._lower_activation,
                "reduce_add": self._lower_reduction,
                "reduce_max": self._lower_reduction,
                "add": self._lower_binary,
                "subtract": self._lower_binary,
                "multiply": self._lower_binary,
                "maximum": self._lower_binary,
            }
            lowerer = lowerers.get(expression.op)
            if lowerer is None:
                raise ValueError(f"no nkigym lowering for symbolic operation {expression.op!r}")
            cached = lowerer(expression)
            self._cache[expression] = cached
        return cached

    def _lower_input(self, expression: Expr) -> _Value:
        """Resolve one pre-emitted input load."""
        if expression.name is None or expression.name not in self._inputs:
            raise ValueError(f"unknown symbolic input {expression.name!r}")
        return self._inputs[expression.name]

    def _lower_view(self, expression: Expr) -> _Value:
        """Erase a singleton-only shape view."""
        source = self.lower(_only_expression_arg(expression))
        return _Value(name=source.name, shape=expression.shape)

    def _lower_transpose(self, expression: Expr) -> _Value:
        """Lower a standalone transpose to the DMA transpose primitive."""
        source = self.lower(_only_expression_arg(expression))
        return self._emit_transpose(source, expression.shape)

    def _lower_matmul(self, expression: Expr) -> _Value:
        """Lower logical ``lhs @ rhs`` to ``stationary.T @ moving``."""
        left, right = _two_expression_args(expression)
        unwrapped_left = _strip_views(left)
        if unwrapped_left.op == "transpose":
            stationary = self.lower(_only_expression_arg(unwrapped_left))
        else:
            stationary = self._emit_transpose(self.lower(left), (left.shape[1], left.shape[0]))
        moving = self.lower(right)
        psum_name = self._new_name("psum_matmul")
        self._emit(psum_name, "NKIMatmul", (), (f"stationary={stationary.name}", f"moving={moving.name}"))
        sbuf_name = self._new_name("sbuf_matmul")
        self._emit(sbuf_name, "NKITensorCopy", (), (f"src={psum_name}",))
        return _Value(name=sbuf_name, shape=expression.shape)

    def _lower_activation(self, expression: Expr) -> _Value:
        """Lower one unary activation with an extracted scalar affine input."""
        argument = _only_expression_arg(expression)
        base, scale, bias = _extract_affine(argument)
        source = self.lower(base)
        configuration = [f'op="{expression.op}"']
        if scale != 1.0:
            configuration.append(f"scale={_format_scalar(scale)}")
        if bias != 0.0:
            configuration.append(f"bias={_format_scalar(bias)}")
        name = self._new_name(f"sbuf_{expression.op}")
        self._emit(name, "NKIActivation", tuple(configuration), (f"data={source.name}",))
        return _Value(name=name, shape=expression.shape)

    def _lower_reduction(self, expression: Expr) -> _Value:
        """Lower a free-axis reduction, fusing a supported elementwise map."""
        data = _only_expression_arg(expression)
        if len(data.shape) != 2 or expression.axis != 1:
            raise ValueError(
                f"nkigym reductions require axis 1 of a rank-two tensor, got shape={data.shape} axis={expression.axis}"
            )
        map_ops = {"square", "exp", "tanh", "reciprocal", "sqrt", "rsqrt"}
        if data.op in map_ops and self._consumer_counts.get(data, 0) == 1:
            result = self._emit_activation_reduction(expression, data)
        else:
            source = self.lower(data)
            reduce_op = "add" if expression.op == "reduce_add" else "maximum"
            name = self._new_name(f"sbuf_reduce_{reduce_op}")
            self._emit(name, "NKITensorReduce", (f'op="{reduce_op}"', "axis=1"), (f"data={source.name}",))
            result = _Value(name=name, shape=expression.shape)
        return result

    def _emit_activation_reduction(self, reduction: Expr, mapped: Expr) -> _Value:
        """Emit one activation-reduce primitive for a mapped reduction."""
        mapped_argument = _only_expression_arg(mapped)
        base, scale, bias = _extract_affine(mapped_argument)
        source = self.lower(base)
        reduce_op = "add" if reduction.op == "reduce_add" else "max"
        configuration = [f'op="{mapped.op}"', f'reduce_op="{reduce_op}"']
        if scale != 1.0:
            configuration.append(f"scale={_format_scalar(scale)}")
        if bias != 0.0:
            configuration.append(f"bias={_format_scalar(bias)}")
        name = self._new_name(f"sbuf_{mapped.op}_{reduce_op}")
        self._emit(name, "NKIActivationReduce", tuple(configuration), (f"data={source.name}",))
        return _Value(name=name, shape=reduction.shape)

    def _lower_binary(self, expression: Expr) -> _Value:
        """Lower scalar, row-vector broadcast, or same-shape binary math."""
        left, right = expression.args
        if isinstance(left, float) or isinstance(right, float):
            result = self._lower_scalar_binary(expression, left, right)
        else:
            if not isinstance(left, Expr) or not isinstance(right, Expr):
                raise ValueError(f"{expression.op} requires tensor or scalar operands")
            result = self._lower_tensor_binary(expression, left, right)
        return result

    def _lower_scalar_binary(self, expression: Expr, left: Expr | float, right: Expr | float) -> _Value:
        """Lower one tensor-scalar operation."""
        tensor = right if isinstance(left, float) else left
        scalar = left if isinstance(left, float) else right
        if not isinstance(tensor, Expr) or not isinstance(scalar, float):
            raise ValueError(f"{expression.op} requires exactly one tensor and one scalar")
        source = self.lower(tensor)
        scalar_is_left = isinstance(left, float)
        if _is_vector_shape(expression.shape):
            scale, bias = _scalar_affine(expression.op, scalar, scalar_is_left)
            result = self._emit_affine_copy(source, expression.shape, scale, bias)
        else:
            if expression.op not in {"add", "subtract", "multiply"}:
                raise ValueError(f"NKITensorScalar does not support {expression.op}")
            configuration = [f'op0="{expression.op}"']
            if scalar_is_left and expression.op == "subtract":
                configuration.append("reverse0=True")
            name = self._new_name(f"sbuf_{expression.op}")
            self._emit(
                name,
                "NKITensorScalar",
                tuple(configuration),
                (f"data={source.name}", f"operand0={_format_scalar(scalar)}"),
            )
            result = _Value(name=name, shape=expression.shape)
        return result

    def _emit_affine_copy(self, source: _Value, output_shape: tuple[int, ...], scale: float, bias: float) -> _Value:
        """Lower scalar math on a reduction vector through activation-copy."""
        configuration = ['op="copy"']
        if scale != 1.0:
            configuration.append(f"scale={_format_scalar(scale)}")
        if bias != 0.0:
            configuration.append(f"bias={_format_scalar(bias)}")
        name = self._new_name("sbuf_affine")
        self._emit(name, "NKIActivation", tuple(configuration), (f"data={source.name}",))
        return _Value(name=name, shape=output_shape)

    def _lower_tensor_binary(self, expression: Expr, left: Expr, right: Expr) -> _Value:
        """Lower same-shape tensors or a matrix with a row-vector broadcast."""
        broadcast = _row_vector_broadcast(left.shape, right.shape, expression.shape)
        if broadcast is not None:
            matrix_expression, vector_expression, reverse = (
                (left, right, False) if broadcast == "right" else (right, left, True)
            )
            if expression.op not in {"add", "subtract", "multiply"}:
                raise ValueError(f"row-vector broadcast does not support {expression.op}")
            matrix = self.lower(matrix_expression)
            vector = self.lower(vector_expression)
            configuration = [f'op0="{expression.op}"']
            if reverse and expression.op == "subtract":
                configuration.append("reverse0=True")
            name = self._new_name(f"sbuf_{expression.op}")
            self._emit(
                name, "NKITensorScalar", tuple(configuration), (f"data={matrix.name}", f"operand0={vector.name}")
            )
        elif left.shape == right.shape == expression.shape:
            left_value = self.lower(left)
            right_value = self.lower(right)
            tensor_op = "maximum" if expression.op == "maximum" else expression.op
            name = self._new_name(f"sbuf_{tensor_op}")
            self._emit(
                name,
                "NKITensorTensor",
                (f'op="{tensor_op}"',),
                (f"data1={left_value.name}", f"data2={right_value.name}"),
            )
        else:
            raise ValueError(
                f"nkigym cannot lower {expression.op} broadcast {left.shape} with {right.shape} -> {expression.shape}"
            )
        return _Value(name=name, shape=expression.shape)

    def _emit_transpose(self, source: _Value, output_shape: tuple[int, ...]) -> _Value:
        """Emit an SBUF-to-SBUF DMA transpose."""
        name = self._new_name("sbuf_transpose")
        self._emit(name, "NKIDMATranspose", (), (f"src={source.name}",))
        return _Value(name=name, shape=output_shape)

    def _emit(self, target: str, class_name: str, configuration: tuple[str, ...], operands: tuple[str, ...]) -> None:
        """Append one NKIOp assignment and register its import."""
        self._imports.add(class_name)
        constructor = f"{class_name}({', '.join(configuration)})" if configuration else f"{class_name}()"
        self._body.append(f"{target} = {constructor}({', '.join(operands)})")

    def _new_name(self, stem: str) -> str:
        """Return a stable unique local name for one operation family."""
        index = self._counters.get(stem, 0)
        self._counters[stem] = index + 1
        return f"{stem}_{index}"


def lower_to_source(expression: Expr, input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Lower one symbolic expression to complete Python source."""
    return _Lowerer(input_specs).build(expression)


def _only_expression_arg(expression: Expr) -> Expr:
    """Return one required tensor argument."""
    if len(expression.args) != 1 or not isinstance(expression.args[0], Expr):
        raise ValueError(f"{expression.op} requires one tensor argument")
    return expression.args[0]


def _two_expression_args(expression: Expr) -> tuple[Expr, Expr]:
    """Return two required tensor arguments."""
    if len(expression.args) != 2 or not all(isinstance(argument, Expr) for argument in expression.args):
        raise ValueError(f"{expression.op} requires two tensor arguments")
    left, right = expression.args
    if not isinstance(left, Expr) or not isinstance(right, Expr):
        raise ValueError(f"{expression.op} requires two tensor arguments")
    return left, right


def _strip_views(expression: Expr) -> Expr:
    """Remove metadata-only views from an expression."""
    result = expression
    while result.op == "view":
        result = _only_expression_arg(result)
    return result


def _extract_affine(expression: Expr) -> tuple[Expr, float, float]:
    """Extract ``base * scale + bias`` from scalar binary expressions."""
    if expression.op == "multiply":
        extracted = _extract_scalar_binary(expression)
        if extracted is not None:
            base_expression, scalar, scalar_is_left = extracted
            _ = scalar_is_left
            base, scale, bias = _extract_affine(base_expression)
            result = (base, scale * scalar, bias * scalar)
        else:
            result = (expression, 1.0, 0.0)
    elif expression.op in {"add", "subtract"}:
        extracted = _extract_scalar_binary(expression)
        if extracted is not None:
            base_expression, scalar, scalar_is_left = extracted
            base, scale, bias = _extract_affine(base_expression)
            if expression.op == "add":
                result = (base, scale, bias + scalar)
            elif scalar_is_left:
                result = (base, -scale, scalar - bias)
            else:
                result = (base, scale, bias - scalar)
        else:
            result = (expression, 1.0, 0.0)
    else:
        result = (expression, 1.0, 0.0)
    return result


def _extract_scalar_binary(expression: Expr) -> tuple[Expr, float, bool] | None:
    """Return tensor, scalar, and scalar orientation for one binary node."""
    if len(expression.args) != 2:
        result = None
    else:
        left, right = expression.args
        if isinstance(left, float) and isinstance(right, Expr):
            result = (right, left, True)
        elif isinstance(left, Expr) and isinstance(right, float):
            result = (left, right, False)
        else:
            result = None
    return result


def _scalar_affine(operation: str, scalar: float, scalar_is_left: bool) -> tuple[float, float]:
    """Represent scalar vector math as activation-copy scale and bias."""
    if operation == "multiply":
        result = (scalar, 0.0)
    elif operation == "add":
        result = (1.0, scalar)
    elif operation == "subtract" and scalar_is_left:
        result = (-1.0, scalar)
    elif operation == "subtract":
        result = (1.0, -scalar)
    else:
        raise ValueError(f"scalar affine lowering does not support {operation}")
    return result


def _row_vector_broadcast(left: tuple[int, ...], right: tuple[int, ...], output: tuple[int, ...]) -> str | None:
    """Identify a matrix and its per-row vector broadcast operand."""
    if len(output) != 2:
        result = None
    elif left == output and _is_row_vector_for(right, output):
        result = "right"
    elif right == output and _is_row_vector_for(left, output):
        result = "left"
    else:
        result = None
    return result


def _is_row_vector_for(candidate: tuple[int, ...], matrix: tuple[int, ...]) -> bool:
    """Return whether a shape denotes one scalar per matrix row."""
    return candidate == (matrix[0],) or candidate == (matrix[0], 1)


def _is_vector_shape(shape: tuple[int, ...]) -> bool:
    """Return whether a logical shape is represented by a per-row vector."""
    return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)


def _format_scalar(value: float) -> str:
    """Format one finite scalar as stable Python source."""
    if not math.isfinite(value):
        raise ValueError(f"nkigym synthesis requires finite scalars, got {value}")
    return repr(float(value))


def _count_consumers(root: Expr) -> dict[Expr, int]:
    """Count incoming DAG edges for use-aware primitive fusion."""
    counts: dict[Expr, int] = {root: 0}
    visited: set[Expr] = set()

    def visit(expression: Expr) -> None:
        """Visit each unique expression and count its tensor arguments."""
        if expression not in visited:
            visited.add(expression)
            for argument in expression.args:
                if isinstance(argument, Expr):
                    counts[argument] = counts.get(argument, 0) + 1
                    visit(argument)

    visit(root)
    return counts


__all__ = ["lower_to_source"]
