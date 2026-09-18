"""Torch-reference arrays normalized for generated kernel ABIs."""

from __future__ import annotations

import operator
from collections.abc import Callable
from types import FunctionType, SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import torch
from torch.fx import GraphModule, Node

from nkigym.ops.folded_load import grouped_context_input
from nkigym.ops.folded_store import grouped_context_attention_graph
from nkigym.ops.grouped_tensor_copy import grouped_attention

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
OUTPUT_LAYOUTS = "block_diagonal cross_entropy_rows grouped_context head_grouped metadata_groups nonzero_flat rope_data routed_tokens token_attention".split()


class _TraceParameter:
    """Mutable tensor holder used by trace-only module replacements."""

    value: torch.Tensor | None = None

    def copy_(self, value: torch.Tensor) -> _TraceParameter:
        """Record a copied tensor value."""
        self.value = value
        return self


class _TraceConvolution:
    """Trace-only functional replacement for a Torch convolution module."""

    def __init__(self, function: Callable[..., torch.Tensor], *_args: object, **kwargs: object) -> None:
        """Store functional convolution parameters."""
        self.function = function
        self.stride, self.padding, self.dilation = (kwargs[name] for name in ("stride", "padding", "dilation"))
        self.weight, self.bias = _TraceParameter(), _TraceParameter() if kwargs["bias"] else None

    def to(self, _dtype: torch.dtype) -> _TraceConvolution:
        """Preserve the lightweight trace-only module."""
        return self

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the configured functional convolution."""
        bias = None if self.bias is None else self.bias.value
        return self.function(data, self.weight.value, bias, self.stride, self.padding, self.dilation)


def _trace_conv1d(*args: object, **kwargs: object) -> _TraceConvolution:
    """Create one trace-only 1D convolution."""
    return _TraceConvolution(torch.conv1d, *args, **kwargs)


def _trace_conv3d(*args: object, **kwargs: object) -> _TraceConvolution:
    """Create one trace-only 3D convolution."""
    return _TraceConvolution(torch.conv3d, *args, **kwargs)


def trace_function(function: FunctionType) -> FunctionType:
    """Clone module-local helpers with trace-compatible dtypes and modules."""
    globals_ = dict(function.__globals__)
    replaced = False
    for name in ("ml_dtypes", "nl"):
        namespace = globals_.get(name)
        if namespace is not None and hasattr(namespace, "float8_e4m3"):
            globals_[name] = SimpleNamespace(**{**vars(namespace), "float8_e4m3": np.float16})
            replaced = True
    namespace = globals_.get("nn")
    if namespace is not None and (hasattr(namespace, "Conv1d") or hasattr(namespace, "Conv3d")):
        globals_["nn"] = SimpleNamespace(**{**vars(namespace), "Conv1d": _trace_conv1d, "Conv3d": _trace_conv3d})
        replaced = True
    if not replaced:
        return function
    result = function
    for name, value in tuple(globals_.items()):
        if isinstance(value, FunctionType) and value.__globals__ is function.__globals__:
            clone = FunctionType(value.__code__, globals_, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            globals_[name] = clone
            if value is function:
                result = clone
    return result


def _trace_marker(name: str) -> Callable[..., NoReturn]:
    """Create a named graph target that rejects direct execution."""

    def marker(*_args: object, **_kwargs: object) -> NoReturn:
        """Reject execution of a graph-only operation."""
        raise RuntimeError(f"{name} is a trace-only marker")

    marker.__name__ = marker.__qualname__ = name
    return marker


cross_entropy_backward = _trace_marker("cross_entropy_backward")
nonzero_compact = _trace_marker("nonzero_compact")
routed_gather = _trace_marker("routed_gather")
packed_attention = _trace_marker("packed_attention")
stable_softmax = _trace_marker("stable_softmax")
astype = _trace_marker("astype")
sparse_topk_affinity = _trace_marker("sparse_topk_affinity")
sorted_prefix = _trace_marker("sorted_prefix")
moe_experts = _trace_marker("moe_experts")


def block_diagonal(array: np.ndarray, shape: tuple[int, ...], transpose: bool) -> np.ndarray:
    """Pack one batched matrix into a block-diagonal rank-two ABI."""
    if array.ndim != 3 or len(shape) != 2:
        raise ValueError(f"block-diagonal layout requires rank three, got {array.shape}")
    if transpose or ((batch := array.shape[0]) * array.shape[1], batch * array.shape[2]) != shape:
        array = array.transpose(0, 2, 1)
    batch, rows, columns = array.shape
    if (batch * rows, batch * columns) != shape:
        raise ValueError(f"cannot normalize batched matrix shape {array.shape} to {shape}")
    output, indices = np.zeros(shape, dtype=array.dtype), np.arange(batch)
    output.reshape(batch, rows, batch, columns)[indices, :, indices, :] = array
    return output


def pad_array(array: np.ndarray, shape: tuple[int, ...], edge_axes: frozenset[int]) -> np.ndarray:
    """Pad one normalized array to its generated ABI shape."""
    if array.ndim != len(shape) or any(source > target for source, target in zip(array.shape, shape, strict=True)):
        raise ValueError(f"cannot pad shape {array.shape} to {shape}")
    for axis, target in enumerate(shape):
        if array.shape[axis] < target:
            widths = [(0, 0)] * array.ndim
            widths[axis] = (0, target - array.shape[axis])
            array = np.pad(array, widths, mode="edge" if axis in edge_axes else "constant")
    return array


def standard_rope_data(array: np.ndarray, shape: tuple[int, ...], interleaved: bool) -> np.ndarray:
    """Flatten a standard RoPE data tensor into grouped real and imaginary halves."""
    if array.ndim != 4 or shape != (int(np.prod(array.shape[1:])), array.shape[0]):
        raise ValueError(f"cannot normalize standard RoPE data shape {array.shape} to {shape}")
    matrix = array.transpose(1, 2, 3, 0).reshape(shape)
    return np.concatenate((matrix[:, ::2], matrix[:, 1::2]), axis=1) if interleaved else matrix


def standard_rope_coeff(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Repeat standard RoPE coefficients across heads and flatten rows."""
    if array.ndim != 3 or shape[1] != array.shape[0]:
        raise ValueError(f"cannot normalize standard RoPE coefficient shape {array.shape} to {shape}")
    width, batch, sequence = array.shape
    if shape != (batch * (heads := shape[0] // (batch * sequence)) * sequence, width):
        raise ValueError(f"cannot repeat standard RoPE coefficient shape {array.shape} to {shape}")
    return np.repeat(array.transpose(1, 2, 0)[:, None, :, :], heads, axis=1).reshape(shape)


def head_grouped(array: np.ndarray, shape: tuple[int, ...], repeat_heads: bool) -> np.ndarray:
    """Group hidden halves and normalize a broadcast head dimension."""
    if array.ndim == 3:
        array = array[:, None, :, :]
    if array.ndim != 4 or array.shape[-1] % 2 or len(shape) != 2:
        raise ValueError(f"head-grouped layout requires rank three or four with even width, got {array.shape}")
    batch, heads, sequence, width = array.shape
    target_heads = shape[1] // width
    if shape != (batch * sequence, target_heads * width) or heads > target_heads:
        raise ValueError(f"cannot normalize head-grouped shape {array.shape} to {shape}")
    if heads < target_heads:
        if repeat_heads and heads == 1:
            array = np.repeat(array, target_heads, axis=1)
        elif not repeat_heads:
            array = np.pad(array, ((0, 0), (0, target_heads - heads), (0, 0), (0, 0)))
        else:
            raise ValueError(f"cannot repeat {heads} coefficient heads to {target_heads}")
    return array.reshape(batch, target_heads, sequence, 2, width // 2).transpose(0, 2, 3, 1, 4).reshape(shape)


def convolution_columns(array: np.ndarray, transform: tuple[object, ...], shape: tuple[int, ...]) -> np.ndarray:
    """Convert channels-first convolution input into an im2col matrix."""
    kernel, stride, padding, dilation = cast(
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], transform[1:]
    )
    padded = np.pad(array, ((0, 0), (0, 0), *((extent, extent) for extent in padding)))
    output = tuple(
        (extent + 2 * pad - dilate * (window - 1) - 1) // step + 1
        for extent, window, step, pad, dilate in zip(array.shape[2:], kernel, stride, padding, dilation, strict=True)
    )
    patches = []
    for index in np.ndindex(*output):
        slices = tuple(
            slice(position * step, position * step + dilate * window, dilate)
            for position, step, dilate, window in zip(index, stride, dilation, kernel, strict=True)
        )
        patches.append(padded[(slice(None), slice(None), *slices)].reshape(array.shape[0], -1))
    columns = np.stack(patches, axis=1).reshape(array.shape[0] * int(np.prod(output)), -1)
    return pad_array(columns, shape, frozenset({0}))


def normalize_topk_output(values: np.ndarray, indices: np.ndarray, sort_output: bool) -> tuple[np.ndarray, np.ndarray]:
    """Order unsorted result pairs while retaining the reference's selected indices."""
    if sort_output:
        order = np.argsort(-values, axis=-1, kind="stable")
        values, indices = (np.take_along_axis(array, order, axis=-1) for array in (values, indices))
    return values, indices


def routed_input(
    kind: str, array: np.ndarray, expert_index: np.ndarray, top_k: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Build one stable-sort or gathered-data routed-token input."""
    routes = expert_index.reshape(-1).astype(np.int64)
    tokens = np.repeat(np.arange(expert_index.shape[0]), top_k)
    if kind == "routed_hidden":
        result = np.repeat(array, top_k, axis=0)
    elif kind == "routed_keys":
        sorted_routes = np.sort(routes)[::-1]
        positions = {int(expert): iter(np.flatnonzero(routes == expert)) for expert in np.unique(routes)}
        order = np.empty(routes.size, dtype=np.int64)
        for start in range(0, routes.size, 8):
            for offset in range(min(8, routes.size - start) - 1, -1, -1):
                order[start + offset] = next(positions[int(sorted_routes[start + offset])])
        keys = np.empty(routes.size, dtype=np.float32)
        keys[order] = np.arange(routes.size, 0, -1, dtype=np.float32)
        result = keys.reshape(1, -1)
    elif kind == "routed_data":
        selected = array[tokens, routes]
        result = np.stack((selected, tokens), axis=1)
    else:
        raise ValueError(f"unknown routed-token input layout {kind!r}")
    return result.reshape(shape)


def token_attention_input(
    kind: str,
    array: np.ndarray,
    active: np.ndarray | None,
    shape: tuple[int, ...],
    dimensions: tuple[int, int, int, int],
) -> np.ndarray:
    """Normalize token-generation attention inputs to rank-two matrices."""
    batch, queries, width, sequence = dimensions
    if kind in {"k", "v"} and active is None:
        raise ValueError(f"token-attention {kind} layout requires active cache values")
    if kind == "q":
        result = array.reshape(width, batch * queries)
    elif kind == "k":
        assert active is not None
        values = array.reshape(batch, width, sequence).copy()
        values[:, :, -queries:] = active.reshape(width, batch, queries).transpose(1, 0, 2)
        result = values.transpose(1, 0, 2).reshape(width, -1)
    elif kind == "v":
        assert active is not None
        values = array.reshape(batch, sequence, width).copy()
        values[:, -queries:, :] = active.reshape(batch, queries, width)
        result = values.reshape(batch, sequence // width, width, width).transpose(2, 0, 1, 3).reshape(width, -1)
    elif kind == "mask":
        valid = array.reshape(sequence // width, width, batch, 1, queries).transpose(1, 0, 2, 3, 4)
        result = np.where(valid.reshape(width, -1), 0.0, np.finfo(np.float32).min)
    else:
        raise ValueError(f"unknown token-attention input layout {kind!r}")
    return result.reshape(shape)


def direct_hbm_placeholder(node: Node) -> bool:
    """Return whether one routed placeholder is consumed directly from HBM."""
    if str(node.target) == "expert_down_weights" or (
        node.users and all(user.target in {torch.cumsum, torch.topk} for user in node.users)
    ):
        return True
    for user in node.users:
        operation = str(getattr(user.target, "__name__", user.target)).removeprefix("wrapped_")
        index = next(iter(user.args[1:]), None)
        routed = isinstance(index, tuple) and (
            index[:2] in {("routed_tokens", "data"), ("routed_tokens", "hidden")}
            or index[:1] in {("grouped_context",), ("metadata_groups",), ("moe_gate_up",), ("rotational_topk",)}
        )
        target = operation == "long" and any(getattr(c.target, "__name__", 0) == "cross_entropy" for c in user.users)
        if (operation == "cross_entropy" and user.args[0] is node) or (operation == "getitem" and routed) or target:
            return True
    return False


def synthetic_graph(input_specs: InputSpecs) -> tuple[torch.fx.Graph, dict[str, Node], Callable[..., Node]]:
    """Create one graph, its shaped placeholders, and a shaped-call helper."""
    graph = torch.fx.Graph()
    inputs = {name: graph.placeholder(name) for name in input_specs}
    for node, (shape, _dtype) in zip(inputs.values(), input_specs.values(), strict=True):
        node.meta["example_value"] = SimpleNamespace(shape=shape)

    def call(operation: object, args: tuple[object, ...], shape: tuple[int, ...], **kwargs: object) -> Node:
        """Add one statically shaped function call."""
        node = graph.call_function(operation, args, kwargs)  # type: ignore[arg-type]
        node.meta["example_value"] = SimpleNamespace(shape=shape)
        return node

    return graph, inputs, call


def routed_graph(f_torch: object, input_specs: InputSpecs) -> GraphModule | None:
    """Build a static stable-sort and gather graph for routed tokens."""
    target = getattr(f_torch, "function", f_torch)
    if getattr(target, "__name__", "") != "permute_routed_tokens_torch_ref":
        return None
    graph, inputs, call = synthetic_graph(input_specs)
    tokens, top_k = input_specs["expert_index"][0]
    routes, hidden = tokens * top_k, input_specs["hidden_input"][0][1]
    layouts = {
        "hidden_input": (("routed_tokens", "hidden", top_k), (routes, hidden)),
        "expert_index": (("routed_tokens", "keys", top_k), (1, routes)),
        "expert_affinities_masked": (("routed_tokens", "data", top_k), (routes, 2)),
    }
    values = {
        name: call(operator.getitem, (inputs[name], transform), shape) for name, (transform, shape) in layouts.items()
    }
    ordered = call(torch.argsort, (values["expert_index"],), (1, routes), dim=-1, descending=True)
    indices = call(operator.getitem, (ordered, (Ellipsis, slice(None, routes))), (1, routes))
    hidden_sorted = call(routed_gather, (values["hidden_input"], indices), (routes, hidden))
    data_sorted = call(routed_gather, (values["expert_affinities_masked"], indices), (routes, 2))
    graph.output([hidden_sorted, data_sorted])
    return GraphModule(torch.nn.Module(), graph)


def attention_graph(f_torch: object, input_specs: InputSpecs) -> GraphModule | None:
    """Build one static sequence-packed attention graph."""
    if grouped := grouped_context_attention_graph(f_torch, input_specs):
        return grouped
    target, bound = getattr(f_torch, "function", f_torch), getattr(f_torch, "bound_kwargs", {})
    if getattr(target, "__name__", "") != "attention_cte_torch_ref":
        return None
    graph, inputs, call = synthetic_graph(input_specs)
    batch, sequence, dimension = input_specs["q"][0][0], input_specs["bound_min"][0][1], input_specs["v"][0][2]
    matrix, scores = (batch * sequence, batch * dimension), (batch * dimension, batch * sequence)
    q_layout = "block_diagonal_t" if bound.get("tp_out") else "block_diagonal"
    q = call(operator.getitem, (inputs["q"], (q_layout,)), matrix)
    k = call(operator.getitem, (inputs["k"], ("block_diagonal",)), scores)
    v = call(operator.getitem, (inputs["v"], ("block_diagonal",)), matrix)
    bounds = [
        call(operator.getitem, (inputs[name], ("block_bounds", sequence)), (batch * sequence,))
        for name in ("bound_min", "bound_max")
    ]
    graph.output([call(packed_attention, (q, k, v, *bounds), matrix)])
    return GraphModule(torch.nn.Module(), graph)


def token_attention_graph(f_torch: object, input_specs: InputSpecs) -> GraphModule | None:
    """Build the static token-generation attention graph."""
    target = getattr(getattr(f_torch, "function", f_torch), "_function", None)
    if getattr(target, "__name__", "") != "_attention_tkg_torch_ref_impl":
        return None
    graph, inputs, call = synthetic_graph(input_specs)
    batch, _, width, sequence = input_specs["k_prior"][0]
    queries = input_specs["q"][0][1] // batch
    tiles = min(32, sequence // width)
    if width != 128 or tiles < 1 or sequence % (width * tiles):
        raise ValueError("token attention requires width 128 and a sequence divisible by its tile group")
    chunks = sequence // (width * tiles)
    dimensions = (batch, queries, width, sequence)
    layouts = {
        "q": (("token_attention", "q", *dimensions), (width, batch * queries)),
        "k_prior": (("token_attention", "k", *dimensions), (width, batch * sequence)),
        "v_prior": (("token_attention", "v", *dimensions), (width, batch * sequence)),
        "mask": (("token_attention", "mask", *dimensions), (width, sequence // width * batch * queries)),
    }
    values = {
        name: call(operator.getitem, (inputs[name], transform), shape) for name, (transform, shape) in layouts.items()
    }
    output = call(
        grouped_attention,
        tuple(values[name] for name in ("q", "k_prior", "v_prior", "mask")),
        (width, batch * queries),
        chunks=chunks,
        groups=batch,
        tiles=tiles,
        width=width,
        queries=queries,
    )
    graph.output([output])
    return GraphModule(torch.nn.Module(), graph)


def special_graph(f_torch: object, input_specs: InputSpecs) -> GraphModule | None:
    """Build a static graph for wide top-k or token-generation MoE."""
    target, bound = getattr(f_torch, "function", f_torch), getattr(f_torch, "bound_kwargs", {})
    name = getattr(target, "__name__", "")
    if name != "moe_block_tkg_torch_ref":
        return None
    expected = (
        bound.get("top_k") == 8
        and getattr(bound.get("router_act_fn"), "name", "") == "SOFTMAX"
        and bound.get("router_pre_norm") is False
        and bound.get("norm_topk_prob") is False
        and getattr(bound.get("expert_affinities_scaling_mode"), "name", "") == "POST_SCALE"
        and getattr(bound.get("hidden_act_fn"), "name", "") == "Swish"
        and bound.get("is_all_expert") is False
        and bound.get("skip_router_logits") is False
    )
    if not expected:
        raise ValueError("Torch MoE synthesis does not support the configured routing or activation mode")
    graph, inputs, call = synthetic_graph(input_specs)
    hidden_width = input_specs["inp"][0][-1]
    experts, weight_hidden, branches, intermediate = input_specs["expert_gate_up_weights"][0]
    if (
        weight_hidden != hidden_width
        or branches != 2
        or input_specs["expert_down_weights"][0] != (experts, intermediate, hidden_width)
        or intermediate % 128
    ):
        raise ValueError("Torch MoE synthesis requires aligned gate/up and down-projection weights")
    matrix, row = (1, hidden_width), (1, 1)
    hidden = call(torch.reshape, (call(astype, (inputs["inp"], "float32"), input_specs["inp"][0]), matrix), matrix)
    gamma = call(astype, (inputs["gamma"], "float32"), input_specs["gamma"][0])
    squared = call(torch.square, (hidden,), matrix)
    mean = call(torch.mean, (squared,), row, dim=-1, keepdim=True)
    rms = call(torch.sqrt, (call(operator.add, (mean, float(bound["eps"])), row),), row)
    normalized = call(operator.mul, (hidden, call(torch.reciprocal, (rms,), row)), matrix)
    normalized = call(operator.mul, (normalized, gamma), matrix)
    logits = call(operator.matmul, (normalized, inputs["router_weights"]), (1, experts))
    selected = graph.call_function(sorted_prefix, (logits,), {"k": 8, "dim": -1, "largest": True, "sorted": True})
    values = call(operator.getitem, (selected, 0), (1, 8))
    indices = call(operator.getitem, (selected, 1), (1, 8))
    affinities = call(stable_softmax, (values,), (1, 8))
    gate_shape, down_shape = (experts, hidden_width * 2 * intermediate), (experts, intermediate * hidden_width)
    gate_up = call(operator.getitem, (inputs["expert_gate_up_weights"], ("moe_gate_up",)), gate_shape)
    down = call(operator.getitem, (inputs["expert_down_weights"], ("moe_down",)), down_shape)
    output = call(moe_experts, (normalized, gate_up, down, affinities, indices, experts, intermediate), matrix)
    graph.output([output, logits])
    return GraphModule(torch.nn.Module(), graph)
