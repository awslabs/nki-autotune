"""FX graph layouts normalized by the generated Torch kernel ABI."""

from __future__ import annotations

import operator
from typing import cast

import numpy as np
import torch
from torch.fx import GraphModule, Node

from nkigym.codegen.torch_abi import InputSpecs, synthetic_graph

Layouts = dict[str, tuple[tuple[object, ...], tuple[int, ...]]]


def input_layouts(graph_module: GraphModule) -> Layouts:
    """Find input transforms normalized by the generated-kernel ABI."""
    layouts = {**head_grouped_layouts(graph_module), **standard_rope_layouts(graph_module)}
    for node in graph_module.graph.nodes:
        operation = _graph_operation(node)
        if node.op == "call_function" and operation in {"cross_entropy", "cross_entropy_backward"}:
            logits, targets = cast(tuple[Node, Node], node.args[:2])
            while targets.op == "call_method" and targets.target in {"long", "to"}:
                targets = cast(Node, targets.args[0])
            if logits.op == targets.op == "placeholder":
                layouts[str(targets.target)] = (("one_hot",), _graph_shape(logits))
            continue
        if node.op == "call_function" and operation in {"conv1d", "conv2d", "conv3d"}:
            layouts.update(_convolution_layouts(node, operation))
            continue
        source: Node | None = None
        target_shape, indexed, transform = None, False, ()
        if node.op == "call_method" and node.target == "reshape" and len(_graph_shape(node)) == 2:
            permute = node.args[0]
            if isinstance(permute, Node) and permute.target == "permute":
                source = cast(Node, permute.args[0])
                transform = permute.args[1] if isinstance(permute.args[1], (tuple, list)) else permute.args[1:]
                target_shape = _graph_shape(node)
        elif node.op == "call_function" and operation == "getitem":
            index = node.args[1]
            transform = index if isinstance(index, tuple) else (index,)
            marker = isinstance(transform[0], str) or transform == (1, 0)
            if marker or all(isinstance(item, (int, slice)) for item in transform):
                source, target_shape = cast(Node, node.args[0]), _graph_shape(node)
                indexed = not marker
        if source is None:
            continue
        current = cast(Node, source)
        while (current.op == "call_method" and current.target in {"float", "to", "view_as"}) or (
            current.op == "call_function" and _graph_operation(current) == "astype"
        ):
            current = cast(Node, current.args[0])
        if current.op == "placeholder" and target_shape is not None:
            if indexed:
                resolver = _scalar_broadcast_shape if int(np.prod(target_shape)) <= 1 else _slice_broadcast_shape
                target_shape = resolver(node, target_shape)
            layouts.setdefault(str(current.target), (tuple(transform), target_shape))
    return layouts


def head_grouped_layouts(graph_module: GraphModule) -> Layouts:
    """Find broadcasted rank-four half rotations that share one matrix ABI."""
    nodes = tuple(graph_module.graph.nodes)
    operations = {_graph_operation(node) for node in nodes}
    if "cat" not in operations or not any(node.op == "call_method" and node.target == "unsqueeze" for node in nodes):
        return {}
    data = tuple(node for node in nodes if node.op == "placeholder" and len(_graph_shape(node)) == 4)
    coefficients = tuple(node for node in nodes if node.op == "placeholder" and len(_graph_shape(node)) == 3)
    if not data or not coefficients:
        return {}
    batch, _heads, sequence, width = _graph_shape(data[0])
    if any((shape := _graph_shape(node))[0] != batch or shape[2:] != (sequence, width) for node in data) or any(
        _graph_shape(node) != (batch, sequence, width) for node in coefficients
    ):
        return {}
    shape = (batch * sequence, max(_graph_shape(node)[1] for node in data) * width)
    return {
        **{str(node.target): (("head_grouped_data",), shape) for node in data},
        **{str(node.target): (("head_grouped_coeff",), shape) for node in coefficients},
    }


def standard_rope_layouts(graph_module: GraphModule) -> Layouts:
    """Find a statically unrolled rotary complex multiply."""
    nodes = tuple(graph_module.graph.nodes)
    operations = {_graph_operation(node) for node in nodes}
    if not {"empty_like", "setitem", "stack"}.issubset(operations):
        return {}
    data = tuple(node for node in nodes if node.op == "placeholder" and len(_graph_shape(node)) == 4)
    coefficients = tuple(node for node in nodes if node.op == "placeholder" and len(_graph_shape(node)) == 3)
    if len(data) != 1 or len(coefficients) != 2:
        return {}
    width, batch, heads, sequence = _graph_shape(data[0])
    if any(_graph_shape(node) != (width // 2, batch, sequence) for node in coefficients):
        return {}
    contiguous = sum(operation == "empty_like" for operation in map(_graph_operation, nodes)) > 1
    data_transform = "rope_data_contiguous" if contiguous else "rope_data_interleaved"
    return {
        str(data[0].target): ((data_transform,), (batch * heads * sequence, width)),
        **{str(node.target): (("rope_coeff",), (batch * heads * sequence, width // 2)) for node in coefficients},
    }


def layout_graph(graph_module: GraphModule, input_specs: InputSpecs, layouts: Layouts) -> GraphModule:
    """Vectorize a statically unrolled standard rotary complex multiply."""
    data_names = [name for name, (transform, _shape) in layouts.items() if str(transform[0]).startswith("rope_data")]
    if not data_names:
        return graph_module
    coefficient_names = [name for name, (transform, _shape) in layouts.items() if transform[0] == "rope_coeff"]
    if len(data_names) != 1 or len(coefficient_names) != 2:
        raise ValueError("standard RoPE layout requires one data tensor and two coefficient tensors")
    graph, values, call = synthetic_graph(input_specs)
    data, (cosine, sine) = values[data_names[0]], (values[name] for name in coefficient_names)
    rows, width = input_specs[data_names[0]][0]
    half_shape = (rows, width // 2)
    real = call(operator.getitem, (data, (Ellipsis, slice(None, width // 2))), half_shape)
    imaginary = call(operator.getitem, (data, (Ellipsis, slice(width // 2, None))), half_shape)
    real_cosine = call(operator.mul, (real, cosine), half_shape)
    imaginary_sine = call(operator.mul, (imaginary, sine), half_shape)
    real_sine = call(operator.mul, (real, sine), half_shape)
    imaginary_cosine = call(operator.mul, (imaginary, cosine), half_shape)
    output_real = call(operator.sub, (real_cosine, imaginary_sine), half_shape)
    output_imaginary = call(operator.add, (real_sine, imaginary_cosine), half_shape)
    graph.output([call(torch.cat, ((output_real, output_imaginary),), (rows, width), dim=-1)])
    return GraphModule(torch.nn.Module(), graph)


def _graph_shape(node: Node) -> tuple[int, ...]:
    """Return the statically propagated shape of one FX node."""
    return tuple(int(extent) for extent in (node.meta.get("tensor_meta") or node.meta["example_value"]).shape)


def _graph_operation(node: Node) -> str:
    """Return one normalized FX operation name."""
    return str(getattr(node.target, "__name__", node.target)).removeprefix("wrapped_")


def _convolution_layouts(node: Node, operation: str) -> Layouts:
    """Return im2col and filter-matrix input layouts for one convolution."""
    data, weights = cast(tuple[Node, Node], node.args[:2])
    filter_source = (
        cast(Node, weights.args[0]) if weights.op == "call_method" and weights.target == "permute" else weights
    )
    if data.op != "placeholder" or filter_source.op != "placeholder":
        raise ValueError("Torch convolution inputs must resolve directly to placeholders")
    filter_shape, output_shape = _graph_shape(filter_source), _graph_shape(node)
    if operation == "conv2d":
        kernel, channels, outputs = filter_shape[2:], filter_shape[0], filter_shape[0]
        if node.kwargs.get("groups") != channels:
            raise ValueError("Torch conv2d adapter supports depthwise groups only")
        filter_transform = ("depthwise_filter",)
    else:
        kernel, channels, outputs = filter_shape[:-2], filter_shape[-2], filter_shape[-1]
        filter_transform = ("conv_filter", len(kernel))
    rank = len(kernel)
    raw_arguments = tuple(
        node.kwargs[name] if name in node.kwargs else node.args[index] if len(node.args) > index else default
        for name, index, default in (("stride", 3, 1), ("padding", 4, 0), ("dilation", 5, 1))
    )
    arguments = tuple((value,) * rank if isinstance(value, int) else tuple(value) for value in raw_arguments)
    rows = ((output_shape[0] * int(np.prod(output_shape[2:])) + 15) // 16) * 16
    columns = channels * int(np.prod(kernel))
    return {
        str(data.target): (("im2col", kernel, *arguments), (rows, columns)),
        str(filter_source.target): (filter_transform, (columns, outputs)),
    }


def _scalar_broadcast_shape(node: Node, _shape: tuple[int, ...]) -> tuple[int, ...]:
    """Infer one partition-vector shape for a dynamically indexed scalar."""
    pending, extents = list(node.users), set()
    while pending:
        user = pending.pop()
        if shape := _graph_shape(user):
            extents.add(int(np.prod(shape[:-1])) if len(shape) > 1 else int(np.prod(shape)))
        else:
            pending.extend(user.users)
    if len(extents) != 1:
        raise ValueError(f"dynamic scalar has ambiguous broadcast extents {sorted(extents)}")
    return (next(iter(extents)),)


def _slice_broadcast_shape(node: Node, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Expand a rank-two static slice to a flattened consumer broadcast."""
    candidates: set[tuple[int, int]] = set()
    if len(shape) == 2:
        for user in node.users:
            user_shape = _graph_shape(user)
            normalized = (int(np.prod(user_shape[:-1])), user_shape[-1]) if len(user_shape) > 2 else user_shape
            if len(normalized) == 2 and normalized[1] == shape[1] and normalized[0] % shape[0] == 0:
                candidates.add(normalized)
    return next(iter(candidates)) if len(candidates) == 1 else shape
