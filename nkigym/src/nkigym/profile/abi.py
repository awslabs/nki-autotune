"""Generated-kernel ABI adapters used by synthesis validation and profiling."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch.fx import GraphModule, Node

from nkigym.codegen.torch_abi import (
    block_diagonal,
    convolution_columns,
    cross_entropy_backward,
    head_grouped,
    moe_gate_up_input,
    nonzero_compact,
    normalize_topk_output,
    pad_array,
    routed_input,
    sparse_topk_affinity,
    standard_rope_coeff,
    standard_rope_data,
    synthetic_graph,
    token_attention_input,
)
from nkigym.codegen.torch_arrays import as_numpy as _as_numpy
from nkigym.codegen.torch_arrays import flatten_output_array as _flatten_output_array
from nkigym.codegen.torch_arrays import logical_output_shape
from nkigym.codegen.torch_layout import Layouts
from nkigym.profile.types import InputSpecs


def reference_graph(f_torch: object, input_specs: InputSpecs) -> GraphModule | None:
    """Replace data-dependent references with equivalent static FX graphs."""
    target, bound = getattr(f_torch, "function", f_torch), getattr(f_torch, "bound_kwargs", {})
    name, names = getattr(target, "__name__", ""), getattr(getattr(target, "__code__", None), "co_names", ())
    pre_norm = bound.get("router_pre_norm", True)
    router = name == "router_topk_torch_ref" and (not pre_norm or bound.get("norm_topk_prob", False))
    backward = {"cross_entropy", "requires_grad_", "backward"}.issubset(names)
    unstable_sort, metadata = (name == "argsort_unstable_torch_ref", name == "build_all_to_all_v_metadata_torch_ref")
    nonzero = name in {"find_nonzero_indices_torch_ref", "find_nonzero_indices_with_count_torch_ref"}
    static_qkv = name == "qkv_tkg_torch_ref" and getattr(bound.get("quantization_type"), "name", "") == "STATIC"
    if not (router or backward or unstable_sort or nonzero or static_qkv or metadata):
        return None
    graph, inputs, call = synthetic_graph(input_specs)
    if router:
        rows = ((input_specs["x"][0][0] + 15) // 16) * 16
        data = call(operator.getitem, (inputs["x"], ("edge_rows", rows)), (rows, input_specs["x"][0][1]))
        shape = (rows, input_specs["w"][0][1])
        logits = call(operator.matmul, (data, inputs["w"]), shape)
        if "w_bias" in inputs:
            logits = call(operator.add, (logits, inputs["w_bias"]), shape)
        k = int(bound["k"])
        selected = graph.call_function(torch.topk, (logits,), {"k": k, "dim": -1, "largest": True, "sorted": True})
        values = call(operator.getitem, (selected, 0), (shape[0], k))
        indices = call(operator.getitem, (selected, 1), (shape[0], k))
        activation = str(getattr(bound["act_fn"], "name", bound["act_fn"])).lower()
        affinity = call(
            sparse_topk_affinity, (logits, values, indices, activation, bool(bound.get("norm_topk_prob", False))), shape
        )
        output = [logits, indices, affinity]
    elif backward:
        kwargs = {"reduction": bound.get("reduction", "mean"), "positions": next(iter(input_specs.values()))[0][0]}
        output = [graph.call_function(cross_entropy_backward, tuple(inputs.values()), kwargs)]
    elif unstable_sort:
        data, shape = inputs["data"], input_specs["data"][0]
        if not bool(bound.get("descending", False)):
            data = call(operator.neg, (data,), shape)
        ordered = graph.call_function(torch.argsort, (data,), {"dim": -1, "descending": True})
        output = [call(operator.getitem, (ordered, (Ellipsis, slice(None, shape[-1]))), shape)]
    elif nonzero:
        shape = input_specs["input_tensor"][0]
        combined = name == "find_nonzero_indices_with_count_torch_ref"
        columns, tokens = (1, shape[-1]) if combined else (shape[1], shape[0])
        transform = ("nonzero_flat", columns, tokens)
        data = call(operator.getitem, (inputs["input_tensor"], transform), (1, columns * tokens))
        output = [call(nonzero_compact, (data, columns, tokens), (1, columns * (tokens + 1)))]
    elif metadata:
        groups, experts = int(bound["replica_group_size"]), int(bound["E"])
        elements, padded = int(np.prod(input_specs["expert_index"][0])), ((groups + 127) // 128) * 128
        transform = ("metadata_groups", elements, experts)
        data = call(operator.getitem, (inputs["expert_index"], transform), (128, max(128, elements)))
        result = call(torch.bincount, (data,), (3, padded), groups=groups, minlength=experts)
        output = [call(operator.getitem, (result, index), (1, padded)) for index in range(3)]
    else:
        rows = int(np.prod(input_specs["hidden"][0][:-1]))
        hidden_width, projected = input_specs["qkv_w"][0]
        matrix, row = (rows, hidden_width), (rows, 1)
        squared = call(torch.square, (inputs["hidden"],), matrix)
        mean = call(torch.mean, (squared,), row, dim=-1, keepdim=True)
        rms = call(torch.sqrt, (call(operator.add, (mean, float(bound["eps"])), row),), row)
        inverse = call(torch.reciprocal, (rms,), row)
        normalized = call(
            operator.mul, (call(operator.mul, (inputs["hidden"], inverse), matrix), inputs["norm_w"]), matrix
        )
        input_scale = call(operator.getitem, (inputs["qkv_in_scale"], (0, 0)), ())
        scaled = call(operator.truediv, (normalized, input_scale), matrix)
        quantized = call(torch.clamp, (scaled, -240.0, 240.0), matrix)
        projection = call(operator.matmul, (quantized, inputs["qkv_w"]), (rows, projected))
        scales = call(operator.getitem, (inputs["qkv_w_scale"], ("scale_rows", rows)), (rows, 3))
        scales = call(operator.mul, (scales, input_scale), (rows, 3))
        q_end = min(projected, int(bound["num_q_heads"]) * int(bound["d_head"]))
        k_end = min(projected, q_end + int(bound["num_kv_heads"]) * int(bound["d_head"]))
        ends = tuple(dict.fromkeys((0, q_end, k_end, projected)))
        parts = []
        for index, (start, stop) in enumerate(zip(ends, ends[1:])):
            value = call(operator.getitem, (projection, (Ellipsis, slice(start, stop))), (rows, stop - start))
            scale = call(operator.getitem, (scales, (Ellipsis, slice(index, index + 1))), (rows, 1))
            parts.append(call(operator.mul, (value, scale), (rows, stop - start)))
        output = [call(torch.cat, (tuple(parts),), (rows, projected), dim=-1)]
    graph.output(output)
    return GraphModule(torch.nn.Module(), graph)


def adapt_inputs(
    inputs: dict[str, object],
    input_specs: InputSpecs,
    kernel_specs: InputSpecs,
    layouts: Layouts,
    edge_axes: dict[str, frozenset[int]],
) -> dict[str, np.ndarray]:
    """Convert Torch or NumPy inputs to one generated kernel ABI."""
    adapted: dict[str, np.ndarray] = {}
    for name, value in inputs.items():
        array = _as_numpy(value)
        dtype = input_specs[name][1]
        array = array.astype(np.float32) if "float" in dtype or dtype == "bfloat16" else array
        if name in layouts:
            transform, shape = layouts[name]
            if transform[0] == "edge_rows":
                array = pad_array(array, shape, frozenset({0}))
            elif transform[0] == "scale_rows":
                array = np.tile(array[0, :], (shape[0], 1))
            elif transform[0] == "nonzero_flat":
                array = array.reshape(cast(int, transform[2]), cast(int, transform[1])).T.reshape(shape)
            elif transform[0] == "metadata_groups":
                source = array
                array = np.full(shape, transform[2], dtype=array.dtype)
                array[:, : cast(int, transform[1])] = np.tile(source.reshape(1, -1), (shape[0], 1))
            elif transform[0] in {"moe_gate_up", "moe_down"}:
                array = moe_gate_up_input(array, shape)
            elif transform[0] == "wide_topk":
                array = array.reshape(shape)
            elif str(transform[0]).startswith("block_diagonal"):
                array = block_diagonal(array, shape, False)
            elif transform[0] == "routed_tokens":
                array = routed_input(
                    f"routed_{transform[1]}", array, _as_numpy(inputs["expert_index"]), cast(int, transform[2]), shape
                )
            elif transform[0] == "token_attention":
                kind, dimensions = cast(str, transform[1]), cast(tuple[int, int, int, int], transform[2:])
                active_name = {"k": "k_active", "v": "v_active"}.get(kind)
                active = None if active_name is None else _as_numpy(inputs[active_name])
                array = token_attention_input(kind, array, active, shape, dimensions)
            elif transform[0] == "block_bounds":
                sequence = cast(int, transform[1])
                values = array.reshape(-1, sequence)
                array = (values + np.arange(values.shape[0])[:, None] * sequence).reshape(shape)
            elif transform[0] == "im2col":
                array = convolution_columns(array, transform, shape)
            elif transform[0] == "one_hot":
                indices = array.reshape(-1).astype(np.int64)
                if np.any(indices < 0) or np.any(indices >= shape[1]):
                    raise ValueError("one-hot indices exceed the generated free-axis extent")
                array = np.zeros(shape, dtype=np.float32)
                array[np.arange(indices.size), indices] = 1.0
            elif transform[0] == "conv_filter":
                rank = cast(int, transform[1])
                array = array.transpose((rank, *range(rank), rank + 1)).reshape(shape)
            elif transform[0] == "depthwise_filter":
                matrix = np.zeros(shape, dtype=array.dtype)
                view = matrix.reshape(array.shape[0], -1, array.shape[0])
                view[np.arange(array.shape[0]), :, np.arange(array.shape[0])] = array.reshape(array.shape[0], -1)
                array = matrix
            elif str(transform[0]).startswith("head_grouped"):
                array = head_grouped(array, shape, str(transform[0]).endswith("coeff"))
            elif str(transform[0]).startswith("rope_data"):
                array = standard_rope_data(array, shape, str(transform[0]).endswith("interleaved"))
            elif transform[0] == "rope_coeff":
                array = standard_rope_coeff(array, shape)
            else:
                permutation = tuple(item for item in transform if type(item) is int)
                array = (
                    array.transpose(permutation)
                    if len(permutation) == array.ndim and tuple(sorted(permutation)) == tuple(range(array.ndim))
                    else array[cast(Any, transform)]
                )
            if array.size == 1:
                array = np.full(shape, array.item(), dtype=array.dtype)
            elif (
                array.shape != shape
                and array.ndim == len(shape)
                and all(target % source == 0 for source, target in zip(array.shape, shape, strict=True))
            ):
                repeats = tuple(target // source for source, target in zip(array.shape, shape, strict=True))
                array = np.tile(array, repeats)
            else:
                array = array.reshape(shape)
        elif kernel_specs[name][0] != input_specs[name][0]:
            array = array.reshape(-1) if len(kernel_specs[name][0]) == 1 else array.reshape(-1, array.shape[-1])
        adapted[name] = pad_array(array, kernel_specs[name][0], edge_axes.get(name, frozenset()))
    return adapted


def adapt_output(
    result: object,
    flatten: bool,
    output_shapes: tuple[tuple[int, ...], ...],
    output_groups: tuple[int, ...],
    sort_topk_output: bool,
    channels_last_output: bool,
    output_layout: str | None,
    topk_source: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Flatten structured Torch outputs into generated ABI arrays."""
    leaves: list[np.ndarray] = []

    def append(value: object) -> None:
        """Append tensor leaves in container iteration order."""
        if isinstance(value, (dict, tuple, list)):
            for item in value.values() if isinstance(value, dict) else value:
                append(item)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            array = _as_numpy(value)
            if output_layout is not None and output_layout.startswith("block_diagonal") and array.ndim == 3:
                array = block_diagonal(
                    array, logical_output_shape(output_shapes, output_groups, len(leaves)), output_layout.endswith("_t")
                )
            elif output_layout == "token_attention" and array.ndim == 4:
                array = block_diagonal(
                    array[:, 0].transpose(0, 2, 1),
                    logical_output_shape(output_shapes, output_groups, len(leaves)),
                    False,
                )
            elif output_layout == "head_grouped" and array.ndim == 4:
                array = head_grouped(array, logical_output_shape(output_shapes, output_groups, len(leaves)), False)
            elif output_layout is not None and output_layout.startswith("rope_data") and array.ndim == 4:
                array = standard_rope_data(
                    array,
                    logical_output_shape(output_shapes, output_groups, len(leaves)),
                    output_layout.endswith("interleaved"),
                )
            elif channels_last_output and array.ndim > 2:
                array = np.moveaxis(array, 1, -1).reshape(-1, array.shape[1])
            elif flatten and array.ndim > 2:
                array = _flatten_output_array(array)
            leaves.append(array)
        elif value is not None:
            raise ValueError(f"Torch output leaf {value!r} is not a tensor")

    if output_layout == "nonzero_flat" and isinstance(result, dict) and "indices" in result:
        indices = cast(torch.Tensor, result["indices"]).detach().cpu().numpy()
        counts = cast(torch.Tensor, result["nonzero_counts"]).detach().cpu().numpy()
        result = np.concatenate((indices, counts[:, None]), axis=1).reshape(1, -1)
    elif output_layout == "routed_tokens":
        tensor = cast(torch.Tensor, result).flip(0)
        result = (tensor[:, :-3], tensor[:, -3:-2], tensor[:, -2:].to(torch.bfloat16).contiguous().view(torch.int32))
    elif output_layout == "metadata_groups":
        array = _as_numpy(result)
        start, width = array.shape[1], sum(shape[-1] for shape in output_shapes[: output_groups[0]])
        array = np.pad(array, ((0, 0), (0, width - start)))
        array[1, start:] = array[0, :start].sum()
        result = tuple(array[index : index + 1] for index in range(3))
    append(result)
    if len(leaves) == 2 and leaves[0].shape == leaves[1].shape and leaves[1].dtype.kind in "iu":
        leaves[:2] = normalize_topk_output(leaves[0], leaves[1], sort_topk_output, topk_source)
    if len(leaves) != len(output_groups):
        raise ValueError(f"Torch output has {len(leaves)} tensors, expected {len(output_groups)} logical outputs")
    expanded: list[np.ndarray] = []
    shape_index = 0
    for array, signed_size in zip(leaves, output_groups, strict=True):
        group_size, axis = abs(signed_size), 0 if signed_size < 0 else -1
        shapes = output_shapes[shape_index : shape_index + group_size]
        ends = np.cumsum([shape[axis] for shape in shapes])
        if group_size > 1 and ends[-1] != array.shape[axis]:
            raise ValueError(f"segmented output extent totals {ends[-1]}, expected {array.shape[axis]}")
        expanded.extend(np.split(array, ends[:-1], axis=axis))
        shape_index += group_size
    leaves = [
        pad_array(
            array.reshape(shape) if array.size == int(np.prod(shape)) else array, shape, frozenset(range(len(shape)))
        )
        for array, shape in zip(expanded, output_shapes, strict=True)
    ]
    return leaves[0] if len(leaves) == 1 else tuple(leaves)


def kernel_adapters(
    input_specs: InputSpecs,
    kernel_specs: InputSpecs,
    layouts: Layouts,
    edge_axes: dict[str, frozenset[int]],
    flatten: bool,
    output_shapes: tuple[tuple[int, ...], ...],
    output_groups: tuple[int, ...],
    sort_topk_output: bool,
    channels_last_output: bool,
    output_layout: str | None,
) -> tuple[
    Callable[[dict[str, object]], dict[str, np.ndarray]], Callable[[object], np.ndarray | tuple[np.ndarray, ...]]
]:
    """Create paired input and output adapters with source-aware top-k normalization."""
    state: dict[str, np.ndarray] = {}

    def inputs(values: dict[str, object]) -> dict[str, np.ndarray]:
        """Adapt inputs and retain the first logical source tensor."""
        adapted = adapt_inputs(values, input_specs, kernel_specs, layouts, edge_axes)
        state["source"] = next(iter(adapted.values()))
        return adapted

    def output(result: object) -> np.ndarray | tuple[np.ndarray, ...]:
        """Adapt outputs using the source retained by the paired input adapter."""
        return adapt_output(
            result,
            flatten,
            output_shapes,
            output_groups,
            sort_topk_output,
            channels_last_output,
            output_layout,
            state.get("source"),
        )

    return (inputs, output)
