"""Programmatic Torch-to-nkigym synthesis through Torch FX."""

import inspect
import operator
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from numbers import Real
from threading import Lock
from types import FunctionType
from typing import Any, cast

import numpy as np
import torch
from torch.fx import GraphModule, Node

from nkigym.codegen.torch_abi import (
    OUTPUT_LAYOUTS,
    attention_graph,
    cross_entropy_backward,
    direct_hbm_placeholder,
    nonzero_compact,
    packed_attention,
    routed_gather,
    routed_graph,
    special_graph,
    stable_softmax,
    token_attention_graph,
    trace_function,
)
from nkigym.codegen.torch_layout import Layouts as _Layouts
from nkigym.codegen.torch_layout import input_layouts as discover_input_layouts
from nkigym.codegen.torch_layout import layout_graph
from nkigym.codegen.torch_moe import emit_moe
from nkigym.codegen.torch_values import TorchSegments as _Segments
from nkigym.codegen.torch_values import TorchValue as _Value
from nkigym.codegen.torch_values import emit_activation, emit_cast, emit_cumsum, emit_reduce, emit_slice, emit_topk
from nkigym.codegen.torch_wide_topk import emit_wide_topk
from nkigym.ir import build_initial_ir
from nkigym.ir.dimension_analysis import _DIMENSION_TRACE_LOCK, analyze_dimensions
from nkigym.ops import _OP_MODULES
from nkigym.ops.dma_transpose import emit_oriented_value
from nkigym.ops.gather import emit_routed_gather
from nkigym.ops.grouped_load import emit_output_stores, emit_rotational_topk
from nkigym.ops.index_iota import emit_packed_topk_indices
from nkigym.ops.stream_shuffle_broadcast import _stream_shuffle_source, _supports_free_broadcast
from nkigym.ops.tensor_scalar import _tensor_scalar_operands, _vector_broadcast
from nkigym.ops.tiled_grouped_matmul import lower_grouped_attention
from nkigym.profile import InputSpecs
from nkigym.profile.abi import adapt_inputs, adapt_output, kernel_adapters, reference_graph
from nkigym.search.axis_groups import CanonicalTileError
from nkigym.synthesis.artifact import ArrayResult, SynthesizedKernel, _exec_nkigym_source, _results_match

_UNARY_OPERATIONS = {name: name for name in "exp log reciprocal rsqrt sqrt square tanh sigmoid silu".split()}
_BINARY_OPERATIONS = {
    item.split("=")[0]: item.split("=")[1]
    for item in "add=add iadd=add mul=multiply imul=multiply sub=subtract matmul=matmul "
    "maximum=maximum truediv=divide".split()
}
_DIRECT_LOWERINGS = frozenset(
    "argsort bincount cat cross_entropy cross_entropy_backward cumsum logsumexp moe_experts nonzero_compact "
    "grouped_attention packed_attention routed_gather sparse_topk_affinity "
    "stable_softmax topk".split()
)
_TORCH_TRACE_LOCK = Lock()


@dataclass(frozen=True)
class _Program:
    """One lowered program and its normalized ABI."""

    source: str
    function: Callable[..., ArrayResult]
    input_specs: InputSpecs
    edge_axes: dict[str, frozenset[int]]
    output_shapes: tuple[tuple[int, ...], ...]
    output_groups: tuple[int, ...]
    sort_topk_output: bool


class _Lowerer:
    """Lower one shape-propagated FX graph into deterministic nkigym source."""

    def __init__(self, graph_module: GraphModule, input_specs: InputSpecs) -> None:
        """Initialize graph, source, and value state."""
        self.graph_module, self.input_specs = graph_module, input_specs
        self.imports = {"NKILoad", "NKIStore"}
        self.body: list[str] = []
        self.values: dict[Node, _Value | _Segments | float | tuple[_Value | _Segments, ...]] = {}
        self.outputs: tuple[_Value, ...] = ()
        self.output_groups: tuple[int, ...] = ()
        self.sort_topk_output = False

    def build(self) -> str:
        """Lower every FX node and render one decorated function."""
        for node in self.graph_module.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem and not node.users:
                continue
            if node.op == "call_function" and _operation_name(node.target) == "_set_grad_enabled":
                continue
            if node.op == "placeholder":
                self._lower_placeholder(node)
            elif node.op == "call_function":
                self._lower_call_function(node)
            elif node.op == "call_method":
                self._lower_call_method(node)
            elif node.op == "output":
                self._lower_output(node)
            else:
                raise ValueError(f"Torch synthesis does not support FX node kind {node.op!r}")
        if not self.outputs:
            raise ValueError("Torch synthesis graph has no tensor output")
        emit_output_stores(self.outputs, self.body)
        imports = ["from nkigym.ops import nkigym_kernel"]
        imports.extend(
            f"from nkigym.ops.{_OP_MODULES[class_name]} import {class_name}" for class_name in sorted(self.imports)
        )
        header = "\n".join(imports)
        body = "\n".join(f"    {line}" for line in self.body)
        return (
            f"{header}\n\n\n@nkigym_kernel\n"
            f"def f_nkigym({', '.join(self.input_specs)}):\n"
            '    """Programmatically synthesized nkigym operator graph."""\n'
            f"{body}\n"
        )

    def _lower_placeholder(self, node: Node) -> None:
        """Emit one HBM-to-SBUF load for an input placeholder."""
        name, direct = str(node.target), direct_hbm_placeholder(node)
        self.values[node] = value = _Value(name if direct else f"sbuf_{name}", self.input_specs[name][0], is_hbm=direct)
        if not direct:
            self.body.append(f"{value.name} = NKILoad()(src={name})")

    def _lower_call_function(self, node: Node) -> None:
        """Lower one supported unary or binary call."""
        operation = _operation_name(node.target)
        if node.target is getattr:
            self._lower_getattr(node)
        elif operation == "getitem":
            self._lower_getitem(node)
        elif operation == "tensor":
            self.values[node] = float(cast(Real, node.args[0]))
        elif operation in _DIRECT_LOWERINGS:
            getattr(self, f"_lower_{operation}")(node)
        elif operation == "neg":
            self._lower_unary(node, "copy", -1.0)
        elif operation in {"conv1d", "conv2d", "conv3d"}:
            self._lower_convolution(node)
        elif operation in {"as_tensor", "reshape", "permute"}:
            self.values[node] = self._value(cast(Node, node.args[0]))
        elif operation == "astype":
            self._lower_cast(node)
        elif operation == "gelu":
            self._lower_unary(node, "gelu_apprx_tanh" if node.kwargs.get("approximate") == "tanh" else "gelu")
        elif operation == "absolute" and all(_operation_name(user.target) == "max" for user in node.users):
            self.values[node] = self._value(cast(Node, node.args[0]))
        elif operation == "max" and len(node.args) == 2:
            self._lower_binary(node, "maximum")
        elif operation == "max" and all(_operation_name(user.target) == "getitem" for user in node.users):
            self._lower_max_with_indices(node)
        elif operation in {"mean", "max"}:
            self._lower_reduce(node, "add" if operation == "mean" else "max")
        elif operation in {"clamp", "clip"}:
            self._lower_clip(node)
        else:
            if (unary := _UNARY_OPERATIONS.get(operation)) is not None:
                self._lower_unary(node, unary)
            elif (binary := _BINARY_OPERATIONS.get(operation)) == "matmul":
                self._lower_matmul(node)
            elif binary is not None:
                self._lower_binary(node, binary)
            else:
                raise ValueError(f"Torch synthesis does not support FX call {node.target!r}")

    def _lower_call_method(self, node: Node) -> None:
        """Lower supported tensor methods."""
        raw_source = self.values[cast(Node, node.args[0])]
        if isinstance(raw_source, _Segments) and node.target in {"float", "to"}:
            self.values[node] = raw_source
            return
        source = self._value(cast(Node, node.args[0]))
        if node.target in {"detach", "float", "long", "permute", "to", "unsqueeze", "view_as"}:
            self.values[node] = source
        elif node.target == "abs" and all(_operation_name(user.target) == "max" for user in node.users):
            self.values[node] = source
        elif node.target in {"reciprocal", "sqrt", "square"}:
            self._lower_unary(node, str(node.target))
        elif node.target in {"max", "mean"}:
            self._lower_reduce(node, "add" if node.target == "mean" else "max")
        elif node.target == "var":
            self._lower_var(node)
        elif node.target == "clamp":
            self._lower_clip(node)
        elif node.target == "transpose":
            if tuple(node.args[1:]) not in {(0, 1), (1, 0)}:
                raise ValueError(f"Torch transpose axes {node.args[1:]!r} are unsupported")
            self.values[node] = _transpose_view(source)
        elif node.target == "reshape":
            shape = _node_shape(node)
            self.values[node] = source if len(shape) > 2 else _reshape_view(source, shape)
        else:
            raise ValueError(f"Torch synthesis does not support method {node.target!r}")

    def _lower_getitem(self, node: Node) -> None:
        """Lower tuple projection or one contiguous free-axis slice."""
        source_node = cast(Node, node.args[0])
        source = self.values[source_node]
        if isinstance(source, tuple):
            self.values[node] = source[cast(int, node.args[1])]
            return
        if isinstance(source, _Segments):
            if source.axis == 0:
                interval = _free_slice(node.args[1], _node_shape(source_node)[-1], source.values[0].shape[-1])
                if interval is None:
                    raise ValueError("partition-segmented tensor indexing requires a free-axis slice")
                self.values[node] = _Segments(
                    tuple(
                        self._slice_value(value, *interval, node, f"_{index}")
                        for index, value in enumerate(source.values)
                    ),
                    axis=0,
                )
                return
            width = _static_prefix_width(node.args[1])
            if width != sum(value.shape[1] for value in source.values):
                raise ValueError("segmented tensor indexing requires its complete static prefix")
            self.values[node] = source
            return
        value = self._value(source_node)
        interval = _free_slice(node.args[1], _node_shape(source_node)[-1], value.shape[-1])
        self.values[node] = value if interval is None else self._slice_value(value, *interval, node)

    def _lower_cat(self, node: Node) -> None:
        """Represent a last-axis concatenation as ordered physical segments."""
        dimension = node.kwargs.get("dim", 0)
        values: list[_Value] = []
        for argument in cast(tuple[Node, ...], node.args[0]):
            value = self.values[argument]
            values.extend(value.values if isinstance(value, _Segments) else (self._value(argument),))
        if dimension not in {-1, len(_node_shape(node)) - 1} or not values:
            raise ValueError("Torch cat requires non-empty tensors on the last axis")
        if len({value.shape[0] for value in values}) != 1:
            raise ValueError("Torch cat requires matching partition extents")
        self.values[node] = _Segments(tuple(values))

    def _lower_getattr(self, node: Node) -> None:
        """Lower tensor transpose attributes and metadata."""
        source, attribute = self._value(cast(Node, node.args[0])), cast(str, node.args[1])
        if attribute == "T":
            self.values[node] = _transpose_view(source)
        elif attribute in {"dtype", "shape"}:
            self.values[node] = source
        else:
            raise ValueError(f"Torch tensor attribute {attribute!r} is unsupported")

    def _lower_unary(self, node: Node, operation: str, scale: float = 1.0) -> None:
        """Emit one unary activation."""
        self.values[node] = self._activate(self._value(cast(Node, node.args[0])), operation, f"sbuf_{node.name}", scale)

    def _activate(self, source: _Value, operation: str, name: str, scale: float = 1.0) -> _Value:
        """Emit one unary activation value."""
        return emit_activation(source, operation, name, scale, self.body, self.imports)

    def _cast(self, source: _Value, class_name: str, name: str) -> _Value:
        """Emit one activation-backed dtype cast."""
        return emit_cast(source, class_name, name, self.body, self.imports)

    def _reduce(self, source: _Value, name: str, op: str, reduce_op: str) -> _Value:
        """Emit one activation-backed free-axis reduction."""
        return emit_reduce(source, name, op, reduce_op, self.body, self.imports)

    def _lower_reduce(self, node: Node, reduction: str) -> None:
        """Lower a rank-two last-axis mean or maximum reduction."""
        source = self._value(source_node := cast(Node, node.args[0]))
        dimension = node.kwargs.get("dim", node.kwargs.get("axis"))
        keepdim = node.kwargs.get("keepdim", node.kwargs.get("keepdims"))
        if len(source.shape) != 2 or dimension not in {-1, 1} or not keepdim:
            raise ValueError(f"Torch reduction requires rank two, dim=-1, and keepdim=True, got {source.shape}")
        source = self._orient(source, False, node, "_data")
        target = _Value(f"sbuf_{node.name}", (source.shape[0],))
        absolute = reduction == "max" and _operation_name(source_node.target) in {"abs", "absolute"}
        class_name = "NKITensorScalarReduce" if absolute else "NKIActivationReduce"
        arguments = 'op0="abs", operand0=0.0, reduce_op="max"' if absolute else f'op="copy", reduce_op="{reduction}"'
        self.imports.add(class_name)
        reduced = target if reduction != "add" else _Value(f"{target.name}_sum", target.shape)
        self.body.append(f"{reduced.name} = {class_name}({arguments})(data={source.name})")
        if reduction == "add":
            target = self._activate(reduced, "copy", target.name, 1.0 / source.shape[1])
        self.values[node] = (target,) if node.target == "max" else target

    def _lower_max_with_indices(self, node: Node) -> None:
        """Lower a last-axis maximum and its index."""
        source = self._value(cast(Node, node.args[0]))
        dimension = node.kwargs.get("dim", node.kwargs.get("axis"))
        keepdim = node.kwargs.get("keepdim", node.kwargs.get("keepdims"))
        if len(source.shape) != 2 or dimension not in {-1, 1} or not keepdim:
            raise ValueError("Torch max with indices requires rank two, dim=-1, and keepdim=True")
        self.values[node] = self._emit_topk(source, 1, node)

    def _lower_cast(self, node: Node) -> None:
        """Lower wrapped NumPy casts used by Torch references."""
        source, dtype = self._value(cast(Node, node.args[0])), node.args[1]
        if dtype in {"bfloat16", "float16", "physical_bfloat16"}:
            class_name = {"bfloat16": "NKISemanticBF16Cast", "float16": "NKIFloat8Cast"}.get(dtype, "NKIBF16Cast")
            source = self._cast(
                source, class_name, f"sbuf_{'semantic_' if dtype != 'physical_bfloat16' else ''}{node.name}"
            )
        elif dtype != "float32":
            raise ValueError(f"Torch synthesis does not support cast dtype {dtype!r}")
        self.values[node] = source

    def _lower_clip(self, node: Node) -> None:
        """Lower scalar lower and upper clipping bounds."""
        source = self._value(cast(Node, node.args[0]))
        self.imports.add("NKITensorScalar")
        minimum = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
        maximum = node.kwargs.get("max", node.args[2] if len(node.args) > 2 else None)
        for operation, bound in (("maximum", minimum), ("minimum", maximum)):
            if bound is not None:
                target = _Value(f"sbuf_{node.name}_{operation}", source.shape, source.transposed)
                self.body.append(
                    f'{target.name} = NKITensorScalar(op0="{operation}")(data={source.name}, operand0={bound!r})'
                )
                source = target
        self.values[node] = source

    def _lower_var(self, node: Node) -> None:
        """Lower a rank-two population variance."""
        source = self._value(cast(Node, node.args[0]))
        if len(source.shape) != 2 or node.kwargs.get("dim") not in {-1, 1} or node.kwargs.get("correction") != 0:
            raise ValueError(f"Torch var requires rank two, dim=-1, and correction=0, got {source.shape}")
        source = self._orient(source, False, node, "_data")
        mean, centered = f"sbuf_{node.name}_mean", f"sbuf_{node.name}_centered"
        target = _Value(f"sbuf_{node.name}", (source.shape[0],))
        self.imports.update(("NKIActivationReduce", "NKITensorScalar"))
        self.body.extend(
            (
                f'{mean} = NKIActivationReduce(op="copy", reduce_op="add", '
                f"scale={1.0 / source.shape[1]!r})(data={source.name})",
                f'{centered} = NKITensorScalar(op0="subtract")(data={source.name}, operand0={mean})',
                f'{target.name} = NKIActivationReduce(op="square", reduce_op="add", '
                f"scale={source.shape[1] ** -0.5!r})(data={centered})",
            )
        )
        self.values[node] = target

    def _lower_binary(self, node: Node, operation: str) -> None:
        """Emit tensor-tensor or tensor-scalar pointwise math."""
        left = self._binary_operand(node.args[0])
        right = self._binary_operand(node.args[1])
        if isinstance(left, _Segments) or isinstance(right, _Segments):
            self.values[node] = self._emit_segment_binary(left, right, operation, node)
            return
        self.values[node] = self._emit_binary(left, right, operation, f"sbuf_{node.name}", node)

    def _emit_segment_binary(
        self, left: _Value | _Segments | float, right: _Value | _Segments | float, operation: str, node: Node
    ) -> _Segments:
        """Apply pointwise math independently to ordered free-axis segments."""
        template = left if isinstance(left, _Segments) else right
        if not isinstance(template, _Segments):
            raise TypeError("segmented binary lowering requires at least one segmented operand")
        left_values = self._segment_operand(left, template, node, "_left")
        right_values = self._segment_operand(right, template, node, "_right")
        values = tuple(
            self._emit_binary(lhs, rhs, operation, f"sbuf_{node.name}_{index}", node)
            for index, (lhs, rhs) in enumerate(zip(left_values, right_values, strict=True))
        )
        return _Segments(values)

    def _segment_operand(
        self, operand: _Value | _Segments | float, template: _Segments, node: Node, suffix: str
    ) -> tuple[_Value | float, ...]:
        """Split one full tensor to match segments or repeat a scalar."""
        if isinstance(operand, _Segments):
            if tuple(value.shape for value in operand.values) != tuple(value.shape for value in template.values):
                raise ValueError("Torch segmented operands require matching physical shapes")
            return operand.values
        if isinstance(operand, float):
            return (operand,) * len(template.values)
        operand = cast(_Value, operand)
        widths = tuple(value.shape[1] for value in template.values)
        if len(operand.shape) != 2 or sum(widths) != operand.shape[1]:
            raise ValueError("Torch segmented broadcast requires one full matching rank-two tensor")
        offset = 0
        values = []
        for index, width in enumerate(widths):
            values.append(self._slice_value(operand, offset, width, node, f"{suffix}_{index}"))
            offset += width
        return tuple(values)

    def _emit_binary(
        self, left: _Value | float, right: _Value | float, operation: str, target_name: str, node: Node
    ) -> _Value:
        """Emit one non-segmented tensor binary operation."""
        if operation == "divide":
            if isinstance(left, _Value) and isinstance(right, _Value):
                operation = "multiply"
                right = self._activate(right, "reciprocal", f"sbuf_{node.name}_reciprocal")
            elif isinstance(left, _Value) and isinstance(right, float):
                operation, right = "multiply", 1.0 / right
            elif not isinstance(left, _Value) or not isinstance(right, (_Value, float)):
                raise ValueError("Torch division requires a tensor numerator")
        if isinstance(left, _Value) and isinstance(right, _Value) and left.shape == right.shape:
            left = self._orient(left, False, node, "_left")
            right = self._orient(right, False, node, "_right")
            target = _Value(target_name, left.shape)
            self.imports.add("NKITensorTensor")
            self.body.append(
                f'{target.name} = NKITensorTensor(op="{operation}")(data1={left.name}, data2={right.name})'
            )
        else:
            if isinstance(left, _Value) and isinstance(right, _Value):
                broadcast = _vector_broadcast(left, right)
                if broadcast is None:
                    raise ValueError(f"Torch {operation} tensor broadcast is unsupported: {left.shape}, {right.shape}")
                matrix, vector, reverse, transposed = broadcast
                if transposed and _supports_free_broadcast(matrix.shape, vector.shape, operation):
                    matrix, vector = self._orient(matrix, False, node, "_data"), self._orient(
                        vector, False, node, "_operand"
                    )
                    broadcasted = _Value(f"{target_name}_broadcast", matrix.shape)
                    self.imports.add("NKIStreamShuffleBroadcast")
                    self.body.append(_stream_shuffle_source(broadcasted.name, vector.name, matrix.shape[0]))
                    operands = (broadcasted, matrix) if reverse else (matrix, broadcasted)
                    return self._emit_binary(*operands, operation, target_name, node)
                tensor = self._orient(matrix, transposed, node, "_data")
                operand = vector.name
            else:
                tensor, scalar, reverse = _tensor_scalar_operands(left, right)
                transposed, operand = tensor.transposed, repr(scalar)
            if operation not in {"add", "divide", "greater_equal", "less", "maximum", "subtract", "multiply"}:
                raise ValueError(f"NKITensorScalar does not support {operation}")
            target = _Value(target_name, tensor.shape, transposed=transposed)
            reverse_argument = ", reverse0=True" if reverse and operation == "subtract" else ""
            self.imports.add("NKITensorScalar")
            self.body.append(
                f'{target.name} = NKITensorScalar(op0="{operation}"{reverse_argument})'
                f"(data={tensor.name}, operand0={operand})"
            )
        return target

    def _lower_matmul(self, node: Node) -> None:
        """Lower one rank-two logical matrix product."""
        left, right = (self._value(cast(Node, argument)) for argument in node.args)
        self.values[node] = self._emit_matmul(left, right, node)

    def _emit_matmul(self, left: _Value, right: _Value, node: Node, suffix: str = "") -> _Value:
        """Emit one rank-two logical matrix product."""
        if len(left.shape) != 2 or len(right.shape) != 2:
            raise ValueError(f"Torch matmul requires rank-two tensors, got {left.shape} @ {right.shape}")
        target_shape = (left.shape[0], right.shape[1])
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"Torch matmul dimensions are inconsistent: {left.shape} @ {right.shape}")
        stationary = self._orient(left, True, node, f"{suffix}_stationary")
        right = self._orient(right, False, node, f"{suffix}_moving")
        psum_name = f"psum_{node.name}{suffix}"
        target = _Value(name=f"sbuf_{node.name}{suffix}", shape=target_shape)
        self.imports.update(("NKIMatmul", "NKITensorCopy"))
        self.body.extend(
            (
                f"{psum_name} = NKIMatmul()(stationary={stationary.name}, moving={right.name})",
                f"{target.name} = NKITensorCopy()(src={psum_name})",
            )
        )
        return target

    def _lower_convolution(self, node: Node) -> None:
        """Lower one adapter-normalized convolution as a matrix product."""
        data, weights = (self._value(cast(Node, argument)) for argument in node.args[:2])
        target = self._emit_matmul(data, weights, node)
        bias = node.args[2] if len(node.args) > 2 else None
        if isinstance(bias, Node):
            target = self._orient(target, True, node, "_bias_data")
            result = _Value(f"sbuf_{node.name}_bias", target.shape, transposed=True)
            self.imports.add("NKITensorScalar")
            self.body.append(
                f'{result.name} = NKITensorScalar(op0="add")(data={target.name}, operand0={self._value(bias).name})'
            )
            target = result
        self.values[node] = target

    def _lower_logsumexp(self, node: Node) -> None:
        """Lower one stable last-axis log-sum-exp."""
        source = self._value(cast(Node, node.args[0]))
        dimension = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        if len(source.shape) != 2 or dimension not in {-1, 1}:
            raise ValueError("Torch logsumexp requires rank two and dim=-1")
        self.values[node] = self._emit_logsumexp(source, node)

    def _softmax(self, source: _Value, node: Node, stem: str) -> _Value:
        """Emit stable last-axis softmax."""
        maximum = self._reduce(source, f"{stem}_maximum", "copy", "max")
        maximum = _Value(maximum.name, (source.shape[0], 1))
        centered = self._emit_binary(source, maximum, "subtract", f"{stem}_centered", node)
        exponentials = self._activate(centered, "exp", f"{stem}_exponentials")
        total = self._reduce(exponentials, f"{stem}_total", "copy", "add")
        reciprocal = self._activate(total, "reciprocal", f"{stem}_reciprocal")
        reciprocal = _Value(reciprocal.name, (source.shape[0], 1))
        return self._emit_binary(exponentials, reciprocal, "multiply", f"{stem}_weights", node)

    def _lower_packed_attention(self, node: Node) -> None:
        """Lower block-diagonal sequence-packed attention."""
        q, k, v, lower, upper = (self._value(cast(Node, argument)) for argument in node.args)
        scores = self._emit_matmul(q, k, node, "_scores")
        masked = _Value(f"sbuf_{node.name}_masked", scores.shape)
        self.imports.add("NKIRangeSelect")
        self.body.append(
            f'{masked.name} = NKIRangeSelect(width={scores.shape[1]}, comp_op0="greater_equal", comp_op1="less")'
            f"(on_true_tile={scores.name}, bound0={lower.name}, bound1={upper.name})"
        )
        weights = self._softmax(masked, node, f"sbuf_{node.name}")
        self.values[node] = self._emit_matmul(weights, v, node, "_output")

    def _lower_grouped_attention(self, node: Node) -> None:
        """Lower grouped token-generation attention."""
        self.values[node] = lower_grouped_attention(node, self.values, self.body, self.imports)

    def _lower_stable_softmax(self, node: Node) -> None:
        """Lower stable last-axis softmax."""
        self.values[node] = self._softmax(self._value(cast(Node, node.args[0])), node, f"sbuf_{node.name}")

    def _lower_routed_gather(self, node: Node) -> None:
        """Gather HBM rows for each physical sorted-index segment."""
        source, indices = cast(Node, node.args[0]), self.values[cast(Node, node.args[1])]
        while _operation_name(source.target) == "getitem":
            source = cast(Node, source.args[0])
        if source.op != "placeholder" or not isinstance(indices, (_Value, _Segments)):
            raise ValueError("routed gather requires one HBM placeholder and sorted indices")
        self.values[node] = emit_routed_gather(
            str(source.target), indices, _node_shape(node)[1], node.name, self.body, self.imports
        )

    def _emit_logsumexp(self, source: _Value, node: Node) -> _Value:
        """Emit max-shifted exponential reduction and logarithm."""
        source = self._orient(source, False, node, "_logsumexp_data")
        base = f"sbuf_{node.name}"
        maximum = self._reduce(source, f"{base}_maximum", "copy", "max")
        centered = _Value(f"{base}_centered", source.shape)
        self.imports.add("NKITensorScalar")
        self.body.append(
            f'{centered.name} = NKITensorScalar(op0="subtract")(data={source.name}, operand0={maximum.name})'
        )
        total = self._reduce(centered, f"{base}_total", "exp", "add")
        logged = self._activate(total, "log", f"{base}_logged")
        target = _Value(f"{base}_logsumexp", maximum.shape)
        self.body.append(f'{target.name} = NKITensorScalar(op0="add")(data={logged.name}, operand0={maximum.name})')
        return target

    def _lower_cross_entropy(self, node: Node) -> None:
        """Lower unreduced cross entropy against one-hot targets."""
        logits, targets = (self._value(cast(Node, argument)) for argument in node.args[:2])
        if logits.shape != targets.shape or node.kwargs.get("reduction") != "none":
            raise ValueError("Torch cross_entropy requires matching rank-two inputs and reduction='none'")
        logits = self._orient(logits, False, node, "_logits")
        targets = self._orient(targets, False, node, "_targets")
        product, selected = f"sbuf_{node.name}_product", f"sbuf_{node.name}_selected"
        loss = _Value(f"sbuf_{node.name}", (logits.shape[0],))
        lse = self._emit_logsumexp(logits, node)
        self.imports.update(("NKIActivationReduce", "NKITensorScalar", "NKITensorTensor"))
        self.body.extend(
            (
                f'{product} = NKITensorTensor(op="multiply")(data1={logits.name}, data2={targets.name})',
                f'{selected} = NKIActivationReduce(op="copy", reduce_op="add")(data={product})',
                f'{loss.name} = NKITensorScalar(op0="subtract")(data={lse.name}, operand0={selected})',
            )
        )
        self.values[node] = loss

    def _lower_cross_entropy_backward(self, node: Node) -> None:
        """Lower the analytical gradient of summed or mean cross entropy."""
        logits, targets = (
            self._orient(self._value(cast(Node, argument)), False, node, suffix)
            for argument, suffix in zip(node.args[:2], ("_logits", "_targets"), strict=True)
        )
        if logits.shape != targets.shape or len(logits.shape) != 2:
            raise ValueError("cross entropy backward requires matching rank-two logits and one-hot targets")
        reduction, positions = node.kwargs["reduction"], int(node.kwargs["positions"])
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"unsupported cross entropy backward reduction {reduction!r}")
        base = f"sbuf_{node.name}"
        maximum = self._reduce(logits, f"{base}_maximum", "copy", "max")
        centered = self._emit_binary(logits, maximum, "subtract", f"{base}_centered", node)
        exponentials = self._activate(centered, "exp", f"{base}_exponentials")
        total = self._reduce(exponentials, f"{base}_total", "copy", "add")
        reciprocal = self._activate(total, "reciprocal", f"{base}_reciprocal")
        probabilities = self._emit_binary(exponentials, reciprocal, "multiply", f"{base}_probabilities", node)
        gradient = self._emit_binary(probabilities, targets, "subtract", f"{base}_gradient", node)
        self.values[node] = self._activate(gradient, "copy", base, 1.0 / positions if reduction == "mean" else 1.0)

    def _lower_cumsum(self, node: Node) -> None:
        """Lower a rank-two last-axis cumulative sum."""
        source = self._orient(self._value(cast(Node, node.args[0])), False, node, "_data")
        if len(source.shape) != 2 or node.kwargs.get("dim") not in {-1, 1}:
            raise ValueError(f"Torch cumsum requires rank two, got {source.shape}")
        self.values[node] = emit_cumsum(source, f"sbuf_{node.name}", self.body, self.imports)

    def _lower_nonzero_compact(self, node: Node) -> None:
        """Lower stable nonzero indices and counts through the native instruction."""
        source = self._orient(self._value(cast(Node, node.args[0])), False, node, "_data")
        columns, tokens = int(node.args[1]), int(node.args[2])
        if source.shape != (1, columns * tokens):
            raise ValueError("nonzero compaction requires one flattened row per logical column")
        self.imports.add("NKINonzeroWithCount")
        values = []
        for column in range(columns):
            chunk = self._slice_value(source, column * tokens, tokens, node, f"_column_{column}")
            value = _Value(f"sbuf_{node.name}_output_{column}", (1, tokens + 1))
            self.body.append(
                f"{value.name} = NKINonzeroWithCount(input_width={tokens}, output_width={tokens + 1})(src={chunk.name})"
            )
            values.append(value)
        self.values[node] = _Segments(tuple(values))

    def _lower_moe_experts(self, node: Node) -> None:
        """Lower tiled selected-expert MLP evaluation."""
        inputs = cast(
            tuple[_Value, _Value, _Value, _Value, _Value], tuple(self._value(cast(Node, arg)) for arg in node.args[:5])
        )
        self.values[node] = emit_moe(inputs, int(node.args[5]), int(node.args[6]), node.name, self.body, self.imports)

    def _lower_bincount(self, node: Node) -> None:
        """Lower grouped histogram metadata with replicated threshold reductions."""
        source = self._cast(self._value(cast(Node, node.args[0])), "NKIFloat32Cast", f"sbuf_{node.name}_data")
        groups, experts = int(node.kwargs["groups"]), int(node.kwargs["minlength"])
        if source.shape[0] != 128 or groups < 1 or experts % groups:
            raise ValueError("grouped bincount requires 128 replicated rows and experts divisible by groups")
        group_size, outputs = experts // groups, ([], [], [])
        self.imports.update(("NKIActivationReduce", "NKIIota"))
        for start in range(0, groups, 128):
            thresholds = []
            for suffix, offset in (("lower", start * group_size), ("upper", (start + 1) * group_size)):
                threshold = _Value(f"sbuf_{node.name}_{suffix}_{start}", (128, 1))
                self.body.append(
                    f"{threshold.name} = NKIIota(partitions=128, width=1, pattern=[[0, 1]], "
                    f"offset={offset}, channel_multiplier={group_size})()"
                )
                thresholds.append(threshold)
            lower, upper = thresholds
            minimum = self._emit_binary(source, lower, "greater_equal", f"sbuf_{node.name}_minimum_{start}", node)
            maximum = self._emit_binary(source, upper, "less", f"sbuf_{node.name}_maximum_{start}", node)
            mask = self._emit_binary(minimum, maximum, "multiply", f"sbuf_{node.name}_mask_{start}", node)
            before = self._emit_binary(source, lower, "less", f"sbuf_{node.name}_before_{start}", node)
            counts = self._reduce(mask, f"sbuf_{node.name}_counts_{start}", "copy", "add")
            displacements = self._reduce(before, f"sbuf_{node.name}_displacements_{start}", "copy", "add")
            zeros = self._activate(counts, "copy", f"sbuf_{node.name}_zeros_{start}", 0.0)
            for values, value in zip(outputs, (counts, displacements, zeros), strict=True):
                values.append(self._cast(value, "NKIInt32Cast", f"{value.name}_int32"))
        self.values[node] = tuple(_Segments(tuple(values)) for values in outputs)

    def _lower_topk(self, node: Node) -> None:
        """Lower an exact descending top-k selection."""
        source = self._value(cast(Node, node.args[0]))
        k = node.kwargs.get("k", node.args[1] if len(node.args) > 1 else None)
        if isinstance(config := node.kwargs.get("rotational_config"), tuple) and isinstance(k, int):
            layout = cast(tuple[int, int, int, int, int], config)
            self.values[node] = emit_rotational_topk(
                source, k, layout, node.name, self.body, self.imports, self.input_specs
            )
            return
        if "wide_width" in node.kwargs:
            self.values[node] = emit_wide_topk(
                source, int(k), int(node.kwargs["wide_width"]), node.name, self.body, self.imports
            )
            return
        dimension, largest, sorted_output = (
            node.kwargs.get(name, default) for name, default in (("dim", -1), ("largest", True), ("sorted", True))
        )
        valid_k = isinstance(k, int) and 1 <= k <= source.shape[1]
        if len(source.shape) != 2 or not valid_k or dimension not in {-1, 1} or not largest:
            raise ValueError("Torch topk requires rank two, valid k, dim=-1, and largest=True")
        self.sort_topk_output |= not sorted_output
        self.values[node] = self._emit_topk(source, k, node)

    def _lower_sparse_topk_affinity(self, node: Node) -> None:
        """Lower exact sparse sigmoid or softmax router affinities."""
        logits = self._orient(self._value(cast(Node, node.args[0])), False, node, "_logits")
        raw_values, raw_indices = (self.values[cast(Node, argument)] for argument in node.args[1:3])
        if not all(isinstance(value, _Segments) and len(value.values) == 1 for value in (raw_values, raw_indices)):
            raise ValueError("sparse top-k affinity requires one native selection segment")
        assert isinstance(raw_values, _Segments) and isinstance(raw_indices, _Segments)
        values, indices = raw_values.values[0], raw_indices.values[0]
        k, activation = values.shape[1], node.args[3]
        if k not in {1, 8} or activation not in {"sigmoid", "softmax"}:
            raise ValueError("sparse top-k affinity supports sigmoid or softmax with k=1 or k=8")
        sources = [logits]
        if k == 8:
            replaced = _Value(f"sbuf_{node.name}_replaced", logits.shape)
            self.imports.add("NKIMatchReplace8")
            self.body.append(
                f"{replaced.name} = NKIMatchReplace8(imm=float('-inf'))(data={logits.name}, vals={values.name})"
            )
            sources.append(replaced)
        else:
            bounds = self._cast(indices, "NKIFloat32Cast", f"sbuf_{node.name}_bounds")
            selected = _Value(f"sbuf_{node.name}_selected", logits.shape)
            self.imports.add("NKIRangeSelect")
            self.body.append(
                f'{selected.name} = NKIRangeSelect(width={logits.shape[1]}, comp_op0="equal", comp_op1="equal")'
                f"(on_true_tile={logits.name}, bound0={bounds.name}, bound1={bounds.name})"
            )
            sources = [selected]
        operation = "sigmoid"
        if activation == "softmax":
            maximum = self._reduce(values, f"sbuf_{node.name}_maximum", "copy", "max")
            self.imports.add("NKITensorScalar")
            centered = [
                _Value(f"sbuf_{node.name}_centered_{index}", source.shape) for index, source in enumerate(sources)
            ]
            self.body.extend(
                f'{target.name} = NKITensorScalar(op0="subtract")(data={source.name}, operand0={maximum.name})'
                for target, source in zip(centered, sources, strict=True)
            )
            sources, operation = centered, "exp"
        transformed = [
            self._activate(source, operation, f"sbuf_{node.name}_transformed_{index}")
            for index, source in enumerate(sources)
        ]
        target = transformed[0]
        if k == 8:
            target = _Value(f"sbuf_{node.name}", logits.shape)
            self.imports.add("NKITensorTensor")
            self.body.append(
                f'{target.name} = NKITensorTensor(op="subtract")'
                f"(data1={transformed[0].name}, data2={transformed[1].name})"
            )
        if bool(node.args[4]) or activation == "softmax":
            total = self._reduce(target, f"sbuf_{node.name}_total", "copy", "add")
            reciprocal = self._activate(total, "reciprocal", f"sbuf_{node.name}_reciprocal")
            target = self._emit_binary(target, reciprocal, "multiply", f"{target.name}_normalized", node)
        self.values[node] = target

    def _lower_argsort(self, node: Node) -> None:
        """Lower a descending argsort prefix through exact top-k selection."""
        source_node = cast(Node, node.args[0])
        descending = bool(node.kwargs.get("descending", False))
        if _operation_name(source_node.target) == "neg" and not descending and len(source_node.users) == 1:
            source = self._value(cast(Node, source_node.args[0]))
        elif descending:
            source = self._value(source_node)
        else:
            raise ValueError("Torch argsort requires descending values or ascending negated values")
        widths = {
            _static_prefix_width(user.args[1]) for user in node.users if _operation_name(user.target) == "getitem"
        }
        if None in widths or len(widths) != 1 or len(widths) != len(node.users):
            raise ValueError("Torch argsort requires one static prefix width")
        if not isinstance(width := widths.pop(), int) or width < 1 or width > source.shape[1]:
            raise ValueError("Torch argsort prefix width is invalid")
        routed = any(c.target is routed_gather for u in node.users for c in u.users)
        self.values[node] = (
            emit_packed_topk_indices(source, width, node.name, self.body, self.imports)
            if routed and len(source_node.users) == 1 and source.shape[0] <= 128 and width % 8 == 0
            else self._emit_topk(source, width, node)[1]
        )

    def _emit_topk(self, source: _Value, k: int, node: Node) -> tuple[_Segments, _Segments]:
        """Emit repeated native top-eight selection rounds."""
        source = self._orient(source, False, node, "_data")
        self.values[cast(Node, node.args[0])] = source
        return emit_topk(source, k, node.name, self.body, self.imports)

    def _slice_value(self, value: _Value, start: int, width: int, node: Node | None, suffix: str = "") -> _Value:
        """Copy one contiguous free-axis interval."""
        base = value.name if node is None else f"sbuf_{node.name}"
        return emit_slice(value, start, width, base, suffix, self.body, self.imports)

    def _orient(self, source: _Value, transposed: bool, node: Node, suffix: str) -> _Value:
        """Materialize one requested physical orientation."""
        if source.transposed == transposed:
            return source
        target = _Value(f"sbuf_{node.name}{suffix}", source.shape, transposed, False, source.storage_dtype)
        emit_oriented_value(source, target, f"psum_{node.name}{suffix}", self.body, self.imports)
        return target

    def _lower_output(self, node: Node) -> None:
        """Record tensor leaves returned by the FX graph."""
        leaves = _flatten_output(node.args[0])
        outputs: list[_Value] = []
        groups: list[int] = []
        for item in leaves:
            value = self.values[item]
            segments = value.values if isinstance(value, _Segments) else (self._value(item),)
            outputs.extend(
                self._orient(segment, False, item, f"_output_{index}") for index, segment in enumerate(segments)
            )
            groups.append(-len(segments) if isinstance(value, _Segments) and value.axis == 0 else len(segments))
        self.outputs, self.output_groups = tuple(outputs), tuple(groups)

    def _binary_operand(self, argument: object) -> _Value | _Segments | float:
        """Resolve one FX tensor, segmented tensor, or finite scalar operand."""
        if isinstance(argument, Node):
            value = self.values[argument]
            if isinstance(value, tuple):
                raise ValueError(f"FX node {cast(Node, argument).name!r} has multiple tensor results")
            return value
        if isinstance(argument, Real) and not isinstance(argument, bool):
            return float(argument)
        raise ValueError(f"Torch synthesis operand {argument!r} is not a tensor or scalar")

    def _value(self, node: Node) -> _Value:
        """Return one previously lowered FX value."""
        value = self.values[node]
        if isinstance(value, _Segments) and len(value.values) == 1:
            value = value.values[0]
        if not isinstance(value, _Value):
            raise ValueError(f"FX node {node.name!r} is a scalar, not a tensor")
        return value


def synthesize_torch_to_nkigym(
    f_torch: Callable[..., object], input_specs: InputSpecs, seed: int = 0
) -> SynthesizedKernel:
    """Synthesize and validate one supported Torch tensor program."""
    if list(inspect.signature(f_torch).parameters) != list(input_specs):
        raise ValueError("f_torch parameters must match input_specs")
    graph_module = _trace_torch(f_torch, input_specs)
    operations = {_operation_name(node.target) for node in graph_module.graph.nodes}
    flatten_scan, has_convolution = "cumsum" in operations, bool(operations & {"conv1d", "conv2d", "conv3d"})
    has_topk = bool(operations & {"argsort", "nonzero_compact", "topk"}) and "moe_experts" not in operations
    input_layouts = discover_input_layouts(graph_module)
    small_free_widths = frozenset(
        shape[1]
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and len(shape := getattr(node.meta.get("tensor_meta", node.meta.get("example_value")), "shape", ())) == 2
        and shape[0] <= 32
    )
    preserve_vectors = (
        frozenset(input_specs)
        if flatten_scan or has_topk or any(t[0] == "one_hot" for t, _ in input_layouts.values())
        else frozenset(
            name
            for name, (shape, _dtype) in input_specs.items()
            if len(shape) == 2 and shape[0] == 1 and shape[1] in small_free_widths
        )
    )
    layout_names = (str(transform[0]) for transform, _shape in input_layouts.values())
    output_layout = next((name for name in layout_names if any(map(name.startswith, OUTPUT_LAYOUTS))), None)
    if output_layout is not None and output_layout.startswith("head_grouped"):
        output_layout = "head_grouped"
    normalized_specs = _normalize_specs(input_specs, preserve_vectors, input_layouts)
    program = _lower_program(layout_graph(graph_module, normalized_specs, input_layouts), normalized_specs)
    flatten_output = flatten_scan or any(len(shape) > 2 for shape, _dtype in input_specs.values())
    input_adapter, output_adapter = kernel_adapters(
        input_specs,
        program.input_specs,
        input_layouts,
        program.edge_axes,
        flatten_output,
        program.output_shapes,
        program.output_groups,
        program.sort_topk_output,
        has_convolution,
        output_layout,
    )
    artifact = SynthesizedKernel(program.source, program.function, program.input_specs, input_adapter, output_adapter)
    has_input_permutation = any(
        len(transform) == len(input_specs[name][0])
        and all(type(axis) is int for axis in transform)
        and tuple(sorted(cast(tuple[int, ...], transform))) == tuple(range(len(transform)))
        for name, (transform, _shape) in input_layouts.items()
    )
    has_input_permutation = has_topk or output_layout in {"grouped_context", "token_attention"} or has_input_permutation
    validation_specs = input_specs if has_input_permutation else _validation_specs(input_specs)
    validation_graph = _trace_torch(f_torch, validation_specs)
    validation_layouts = discover_input_layouts(validation_graph)
    normalized_validation_specs = _normalize_specs(validation_specs, preserve_vectors, validation_layouts)
    validation_program = _lower_program(
        layout_graph(validation_graph, normalized_validation_specs, validation_layouts), normalized_validation_specs
    )
    reference_inputs = _validation_inputs(validation_specs, seed)
    adapted = adapt_inputs(
        reference_inputs,
        validation_specs,
        validation_program.input_specs,
        validation_layouts,
        validation_program.edge_axes,
    )
    expected = adapt_output(
        f_torch(**reference_inputs),
        flatten_output,
        validation_program.output_shapes,
        validation_program.output_groups,
        validation_program.sort_topk_output,
        has_convolution,
        output_layout,
        next(iter(reference_inputs.values())).float().numpy() if has_topk else None,
    )
    with _DIMENSION_TRACE_LOCK:
        actual = validation_program.function(**adapted)
    if not _results_match(actual, expected):
        raise RuntimeError("programmatic synthesis validation failed: fp32 output mismatch")
    return artifact


def _lower_program(graph_module: GraphModule, input_specs: InputSpecs) -> _Program:
    """Lower one graph, padding dimensions that cannot be tiled canonically."""
    normalized = dict(input_specs)
    edge_axes: dict[str, set[int]] = {}
    while True:
        lowerer = _Lowerer(graph_module, normalized)
        source = lowerer.build()
        function = _exec_nkigym_source(source)
        analysis = analyze_dimensions(function, normalized)
        try:
            _ = build_initial_ir(function, normalized)
        except CanonicalTileError as error:
            output_dims = {dimension for name in analysis.return_names for dimension in analysis.tensors[name].dim_ids}
            changed = False
            for name, (shape, dtype) in normalized.items():
                padded = list(shape)
                for axis, dimension in enumerate(analysis.tensors[name].dim_ids):
                    if dimension == error.dimension:
                        padded[axis] = ((error.extent + error.minimum - 1) // error.minimum) * error.minimum
                        if dimension in output_dims:
                            edge_axes.setdefault(name, set()).add(axis)
                        changed = True
                normalized[name] = (tuple(padded), dtype)
            if not changed:
                raise
            continue
        return _Program(
            source,
            function,
            normalized,
            {name: frozenset(axes) for name, axes in edge_axes.items()},
            tuple(value.shape for value in lowerer.outputs),
            lowerer.output_groups,
            lowerer.sort_topk_output,
        )


def _trace_torch(f_torch: Callable[..., object], input_specs: InputSpecs) -> GraphModule:
    """Trace a callable and propagate static tensor shapes."""
    rewritten = (
        attention_graph(f_torch, input_specs)
        or token_attention_graph(f_torch, input_specs)
        or special_graph(f_torch, input_specs)
        or routed_graph(f_torch, input_specs)
        or reference_graph(f_torch, input_specs)
    )
    if rewritten is not None:
        return rewritten
    target = f_torch if inspect.isfunction(f_torch) else _named_wrapper(f_torch, tuple(input_specs))
    try:
        inputs = tuple(
            torch.empty(shape, dtype=_torch_dtype(dtype), device="meta") for shape, dtype in input_specs.values()
        )
        with _TORCH_TRACE_LOCK:
            torch._dynamo.reset()  # type: ignore[attr-defined]
            graph_module, _guards = torch._dynamo.export(  # type: ignore[attr-defined]
                target, aten_graph=False, assume_static_by_default=True
            )(*inputs)
        for node, name in zip(
            (node for node in graph_module.graph.nodes if node.op == "placeholder"), input_specs, strict=True
        ):
            node.target = name
    except Exception as error:
        raise ValueError(f"failed to trace Torch reference: {type(error).__name__}: {error}") from error
    return graph_module


def _named_wrapper(function: Callable[..., object], parameters: tuple[str, ...]) -> Callable[..., object]:
    """Build a traceable function with the callable object's public signature."""
    target = getattr(function, "function", function)
    subscript = getattr(function, "subscript", None)
    implementation = getattr(target, "_function", None)
    if subscript is not None and isinstance(implementation, FunctionType):
        original = target
        target = copy(target)
        cloned = trace_function(implementation)
        setattr(target, "_function", cloned)
        cloned.__globals__.update({name: target for name, value in cloned.__globals__.items() if value is original})
        target = cast(Any, target)[subscript]
    else:
        target = cast(Any, target)[subscript] if subscript is not None else target
        if isinstance(target, FunctionType):
            target = trace_function(target)
    aliases = getattr(function, "aliases", {})
    public = ", ".join(f"{aliases.get(name, name)!r}: {name}" for name in parameters)
    adapter = getattr(function, "argument_adapter", None) or (lambda arguments: arguments)
    namespace: dict[str, object] = {
        "function": target,
        "bound": dict(getattr(function, "bound_kwargs", {})),
        "adapter": adapter,
    }
    source = (
        f"def trace_target({', '.join(parameters)}):\n"
        "    arguments = dict(bound)\n"
        f"    arguments.update({{{public}}})\n"
        "    return function(**adapter(arguments))\n"
    )
    exec(compile(source, "<nkigym-torch-wrapper>", "exec"), namespace)
    return cast(Callable[..., object], namespace["trace_target"])


def _node_shape(node: Node) -> tuple[int, ...]:
    """Return one shape produced by FX shape propagation."""
    if (shape := getattr(node.meta.get("tensor_meta", node.meta.get("example_value")), "shape", None)) is None:
        raise ValueError(f"FX node {node.name!r} has no tensor shape")
    return tuple(int(extent) for extent in shape)


def _operation_name(target: object) -> str:
    """Return one normalized FX call name."""
    return str(getattr(target, "__name__", target)).removeprefix("wrapped_")


def _static_prefix_width(index: object) -> int | None:
    """Return the width of one ``[..., :k]`` index."""
    selector = index[-1] if isinstance(index, tuple) and index else None
    valid = isinstance(selector, slice) and selector.start is None and selector.step is None
    candidate = cast(slice, selector).stop if valid else None
    return candidate if isinstance(candidate, int) else None


def _free_slice(index: object, source_extent: int, normalized_extent: int) -> tuple[int, int] | None:
    """Map one last-axis slice into a normalized rank-two free-axis interval."""
    selector = index[-1] if isinstance(index, tuple) and index else index
    if not isinstance(selector, slice) or selector.step not in {None, 1}:
        return None
    start = 0 if selector.start is None else selector.start
    stop = source_extent if selector.stop is None else selector.stop
    if not all(isinstance(value, int) for value in (start, stop)) or not 0 <= start < stop <= source_extent:
        return None
    if normalized_extent % source_extent:
        raise ValueError(f"normalized free extent {normalized_extent} is not a multiple of {source_extent}")
    scale = normalized_extent // source_extent
    interval = (start * scale, (stop - start) * scale)
    return None if interval == (0, normalized_extent) else interval


def _flatten_output(value: object) -> tuple[Node, ...]:
    """Flatten tensor nodes from one nested FX output structure."""
    if isinstance(value, Node):
        return (value,)
    if isinstance(value, (dict, tuple, list)):
        items = value.values() if isinstance(value, dict) else value
        return tuple(node for item in items for node in _flatten_output(item))
    if value is None:
        return ()
    raise ValueError(f"Torch synthesis output leaf {value!r} is not a tensor")


def _transpose_view(value: _Value) -> _Value:
    """Return a metadata-only rank-two transpose."""
    if len(value.shape) != 2:
        raise ValueError(f"Torch transpose requires rank two, got {value.shape}")
    return _Value(value.name, tuple(reversed(value.shape)), not value.transposed, value.is_hbm, value.storage_dtype)


def _reshape_view(value: _Value, shape: tuple[int, ...]) -> _Value:
    """Return a no-copy rank-two singleton reshape."""
    if len(shape) != 2 or int(np.prod(value.shape)) != int(np.prod(shape)):
        raise ValueError(f"Torch reshape {value.shape} -> {shape} is unsupported")
    if len(value.shape) == 1 and shape == (1, value.shape[0]):
        return _Value(value.name, shape, transposed=True, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    physical_shape = tuple(reversed(value.shape)) if value.transposed else value.shape
    if physical_shape == shape:
        return _Value(value.name, shape, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    if physical_shape == tuple(reversed(shape)):
        return _Value(value.name, shape, transposed=True, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    raise ValueError(f"Torch reshape {value.shape} -> {shape} changes non-singleton layout")


def _torch_dtype(dtype: str) -> torch.dtype:
    """Map one input specification dtype to a shape-propagation dtype."""
    return {"bool": torch.bool, "uint32": torch.int32}.get(
        dtype,
        cast(torch.dtype, getattr(torch, dtype)) if dtype.startswith("int") or dtype == "uint8" else torch.float32,
    )


def _validation_inputs(input_specs: InputSpecs, seed: int) -> dict[str, torch.Tensor]:
    """Generate deterministic Torch inputs for fp32 validation."""
    generator = torch.Generator().manual_seed(seed)
    inputs: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in input_specs.items():
        value = torch.ones(shape, dtype=_torch_dtype(dtype))
        if value.dtype.is_floating_point and name not in {"bound_min", "bound_max"}:
            value = torch.randn(shape, generator=generator, dtype=value.dtype)
        elif value.dtype != torch.bool:
            value.fill_(shape[-2] if name == "bound_max" else int(name == "mask"))
        inputs[name] = value
    return inputs


def _validation_specs(input_specs: InputSpecs) -> InputSpecs:
    """Cap dimensions while preserving equalities and distinct large extents."""
    large = sorted({extent for shape, _dtype in input_specs.values() for extent in shape if extent > 128})
    caps = {extent: max(1, 128 - index) for index, extent in enumerate(large)}
    return {
        name: (tuple(caps.get(extent, extent) for extent in shape), dtype)
        for name, (shape, dtype) in input_specs.items()
    }


def _normalize_specs(input_specs: InputSpecs, preserve_matrix_vectors: frozenset[str], layouts: _Layouts) -> InputSpecs:
    """Flatten leading dimensions and represent broadcast vectors as rank one."""
    normalized: InputSpecs = {}
    for name, (shape, dtype) in input_specs.items():
        if name in layouts:
            transform, shape = layouts[name]
            float_layout = transform[0] == "one_hot" or transform[:2] == ("routed_tokens", "keys")
            dtype = "float32" if float_layout or transform[:2] == ("token_attention", "mask") else dtype
        elif len(shape) > 2:
            shape = (int(np.prod(shape[:-1])), shape[-1])
        elif len(shape) == 2 and 1 in shape and name not in preserve_matrix_vectors:
            shape = (max(shape),)
        normalized[name] = (shape, dtype)
    return normalized


__all__ = ["synthesize_torch_to_nkigym"]
