"""Eager mode tracer.

Traces a math function to capture the op graph, assigns dimension
IDs by unifying shared axes across ops, and computes tile sizes.
"""

from typing import Any

import numpy as np

from nkigym.codegen.eager_types import DimInfo, TensorInfo, TracedOp
from nkigym.ops.base import NKIOp


def _resolve_dim(dim_id: str, aliases: dict[str, str]) -> str:
    """Resolve a dim ID through aliases to its canonical form.

    Args:
        dim_id: Possibly-aliased dimension ID.
        aliases: Alias mapping.

    Returns:
        Canonical dimension ID.
    """
    visited: set[str] = set()
    current = dim_id
    while current in aliases and current not in visited:
        visited.add(current)
        current = aliases[current]
    return current


def _unify_operand_dims(
    op: NKIOp, operand_map: dict[str, str], tensors: dict[str, TensorInfo], aliases: dict[str, str]
) -> dict[str, str]:
    """Build axis-label to dim-id mapping, merging shared dims.

    When a label appears in multiple operands, their dim IDs are merged.

    Args:
        op: The NKIOp instance.
        operand_map: Maps operand slot name to tensor name.
        tensors: All known tensors.
        aliases: Dim alias mapping (mutated in place).

    Returns:
        Label-to-canonical-dim mapping.
    """
    label_to_dim: dict[str, str] = {}
    for slot_name, axis_labels in op.OPERAND_AXES.items():
        tensor_name = operand_map.get(slot_name)
        if tensor_name is None:
            continue
        tinfo = tensors.get(tensor_name)
        if tinfo is None:
            continue
        for i, label in enumerate(axis_labels):
            if i >= len(tinfo.dims):
                continue
            tensor_dim = _resolve_dim(tinfo.dims[i], aliases)
            _unify_label(label, tensor_dim, label_to_dim, aliases)
    return label_to_dim


def _unify_label(label: str, tensor_dim: str, label_to_dim: dict[str, str], aliases: dict[str, str]) -> None:
    """Unify a single axis label with a tensor dim.

    Args:
        label: Axis label from the op.
        tensor_dim: Resolved dim ID from the tensor.
        label_to_dim: Label-to-dim mapping (mutated).
        aliases: Dim alias mapping (mutated).
    """
    if label not in label_to_dim:
        label_to_dim[label] = tensor_dim
        return
    existing = _resolve_dim(label_to_dim[label], aliases)
    if existing != tensor_dim:
        _merge_dims(existing, tensor_dim, aliases)
        label_to_dim[label] = _resolve_dim(existing, aliases)


def _merge_dims(keep: str, remove: str, aliases: dict[str, str]) -> None:
    """Merge two dimension IDs, keeping the lower-numbered one.

    Args:
        keep: First dimension ID.
        remove: Second dimension ID.
        aliases: Dim alias mapping (mutated).
    """
    if keep == remove:
        return
    keep_num = int(keep[1:])
    remove_num = int(remove[1:])
    if remove_num < keep_num:
        keep, remove = remove, keep
    aliases[remove] = keep


def _update_tensor_dims_after_merge(tensors: dict[str, TensorInfo], aliases: dict[str, str]) -> None:
    """Re-resolve all tensor dims through current aliases.

    Args:
        tensors: All known tensors (mutated in place).
        aliases: Dim alias mapping.
    """
    for tname, tinfo in list(tensors.items()):
        new_dims = tuple(_resolve_dim(d, aliases) for d in tinfo.dims)
        if new_dims != tinfo.dims:
            tensors[tname] = TensorInfo(
                name=tinfo.name,
                dims=new_dims,
                shape_2d=tinfo.shape_2d,
                is_input=tinfo.is_input,
                producer_op=tinfo.producer_op,
            )


def _register_outputs(
    op: NKIOp,
    output_names: list[str],
    result_arrays: list[np.ndarray],
    label_to_dim: dict[str, str],
    tensors: dict[str, TensorInfo],
    aliases: dict[str, str],
    dim_counter: int,
    op_idx: int,
) -> int:
    """Assign dim IDs to each output and register in tensors dict.

    Args:
        op: The NKIOp instance.
        output_names: Names for the outputs.
        result_arrays: Numpy result arrays.
        label_to_dim: Label-to-dim mapping (mutated).
        tensors: All known tensors (mutated).
        aliases: Dim alias mapping.
        dim_counter: Current dim counter value.
        op_idx: Index of this op.

    Returns:
        Updated dim counter.
    """
    output_keys = list(op.OUTPUT_AXES.keys())
    for out_idx, out_name in enumerate(output_names):
        out_key = output_keys[out_idx]
        out_labels = op.OUTPUT_AXES[out_key]
        arr = result_arrays[out_idx]
        out_dims = _collect_output_dims(out_labels, arr, label_to_dim, aliases, dim_counter)
        dim_counter = out_dims[1]
        resolved = out_dims[0]
        if len(resolved) > len(arr.shape):
            resolved = resolved[: len(arr.shape)]
        tensors[out_name] = TensorInfo(
            name=out_name, dims=tuple(resolved), shape_2d=arr.shape, is_input=False, producer_op=op_idx
        )
    return dim_counter


def _collect_output_dims(
    out_labels: tuple[str, ...],
    arr: np.ndarray,
    label_to_dim: dict[str, str],
    aliases: dict[str, str],
    dim_counter: int,
) -> tuple[list[str], int]:
    """Collect dim IDs for one output, allocating new dims as needed.

    Args:
        out_labels: Axis labels for this output.
        arr: Numpy result array.
        label_to_dim: Label-to-dim mapping (mutated).
        aliases: Dim alias mapping.
        dim_counter: Current dim counter.

    Returns:
        Tuple of (dim ID list, updated dim counter).
    """
    out_dims: list[str] = []
    for label_idx, label in enumerate(out_labels):
        if label in label_to_dim:
            out_dims.append(_resolve_dim(label_to_dim[label], aliases))
        elif label_idx < len(arr.shape):
            dim_id = f"d{dim_counter}"
            dim_counter += 1
            label_to_dim[label] = dim_id
            out_dims.append(dim_id)
    return out_dims, dim_counter


def _gather_dim_sizes(tensors: dict[str, TensorInfo], aliases: dict[str, str]) -> dict[str, int]:
    """Gather total sizes for each canonical dimension.

    Args:
        tensors: All known tensors.
        aliases: Dim alias mapping.

    Returns:
        Maps canonical dim ID to total size.
    """
    dim_sizes: dict[str, int] = {}
    for tinfo in tensors.values():
        for i, dim_id in enumerate(tinfo.dims):
            canonical = _resolve_dim(dim_id, aliases)
            if i < len(tinfo.shape_2d):
                size = tinfo.shape_2d[i]
                existing = dim_sizes.get(canonical, 0)
                dim_sizes[canonical] = max(existing, size)
    return dim_sizes


def _gather_tile_limits(ops: list[TracedOp], tensors: dict[str, TensorInfo], aliases: dict[str, str]) -> dict[str, int]:
    """Gather max tile size limits across all ops for each dimension.

    Args:
        ops: All traced ops.
        tensors: All known tensors.
        aliases: Dim alias mapping.

    Returns:
        Maps canonical dim ID to max tile limit.
    """
    dim_tile_limits: dict[str, int] = {}
    for traced_op in ops:
        label_to_dim = _build_label_map_for_limits(traced_op, tensors, aliases)
        for label, limit in traced_op.op.MAX_TILE_SIZES.items():
            dim_id = label_to_dim.get(label)
            if dim_id is None:
                continue
            canonical = _resolve_dim(dim_id, aliases)
            existing = dim_tile_limits.get(canonical, 0)
            dim_tile_limits[canonical] = max(existing, limit)
    return dim_tile_limits


def _build_label_map_for_limits(
    traced_op: TracedOp, tensors: dict[str, TensorInfo], aliases: dict[str, str]
) -> dict[str, str]:
    """Build label-to-dim mapping for tile limit gathering.

    Args:
        traced_op: A traced op.
        tensors: All known tensors.
        aliases: Dim alias mapping.

    Returns:
        Label-to-canonical-dim mapping.
    """
    label_to_dim: dict[str, str] = {}
    op = traced_op.op
    for slot_name, axis_labels in op.OPERAND_AXES.items():
        tensor_name = traced_op.operand_names.get(slot_name)
        if tensor_name is None:
            continue
        tinfo = tensors.get(tensor_name)
        if tinfo is None:
            continue
        for i, label in enumerate(axis_labels):
            if i < len(tinfo.dims):
                canonical = _resolve_dim(tinfo.dims[i], aliases)
                label_to_dim[label] = canonical
    _add_output_labels(traced_op, tensors, aliases, label_to_dim)
    return label_to_dim


def _add_output_labels(
    traced_op: TracedOp, tensors: dict[str, TensorInfo], aliases: dict[str, str], label_to_dim: dict[str, str]
) -> None:
    """Add output axis labels to label-to-dim mapping.

    Args:
        traced_op: A traced op.
        tensors: All known tensors.
        aliases: Dim alias mapping.
        label_to_dim: Label-to-dim mapping (mutated).
    """
    op = traced_op.op
    for out_idx, (out_key, out_labels) in enumerate(op.OUTPUT_AXES.items()):
        if out_idx >= len(traced_op.output_names):
            continue
        tname = traced_op.output_names[out_idx]
        tinfo = tensors.get(tname)
        if tinfo is None:
            continue
        for i, label in enumerate(out_labels):
            if i < len(tinfo.dims):
                canonical = _resolve_dim(tinfo.dims[i], aliases)
                label_to_dim[label] = canonical


def _dtype_for_op(op: NKIOp, out_key: str, traced_op: TracedOp, tensor_dtypes: dict[str, str]) -> str:
    """Determine the dtype expression for an op output.

    Args:
        op: The NKIOp instance.
        out_key: Output key name.
        traced_op: The traced op.
        tensor_dtypes: Known tensor dtype expressions.

    Returns:
        Dtype expression string.
    """
    result = "Q.dtype"
    if op.NAME in ("nc_matmul", "tensor_reduce"):
        result = "nl.float32"
    elif op.NAME == "activation_reduce" and out_key == "reduce_res":
        result = "nl.float32"
    else:
        result = _dtype_from_operands(traced_op, tensor_dtypes)
    return result


def _dtype_from_operands(traced_op: TracedOp, tensor_dtypes: dict[str, str]) -> str:
    """Infer dtype from operands, preferring the 'data' operand.

    Args:
        traced_op: The traced op.
        tensor_dtypes: Known tensor dtype expressions.

    Returns:
        Dtype expression string.
    """
    data_name = traced_op.operand_names.get("data")
    if data_name and data_name in tensor_dtypes:
        result = tensor_dtypes[data_name]
    else:
        first_op = next(iter(traced_op.operand_names.values()), None)
        if first_op and first_op in tensor_dtypes:
            result = tensor_dtypes[first_op]
        else:
            result = "Q.dtype"
    return result


class EagerTracer:
    """Traces a math function to capture the op graph.

    Records each nkigym.* call, assigns dimension IDs by unifying
    shared axes across ops, and computes tile sizes.

    Attributes:
        ops: List of traced ops.
        tensors: Maps tensor name to TensorInfo.
        dims: Maps dim_id to DimInfo.
        inputs: Input parameter names in order.
        input_shapes: Input parameter shapes.
        tensor_dtypes: Maps tensor name to dtype expression string.
    """

    def __init__(self) -> None:
        """Initialize empty tracer state."""
        self.ops: list[TracedOp] = []
        self.tensors: dict[str, TensorInfo] = {}
        self.dims: dict[str, DimInfo] = {}
        self.inputs: list[str] = []
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.tensor_dtypes: dict[str, str] = {}
        self._dim_counter = 0
        self._dim_aliases: dict[str, str] = {}

    def register_input(self, name: str, shape: tuple[int, ...]) -> None:
        """Register a kernel input parameter.

        Each axis gets a fresh dim ID. Unification happens later
        when ops reveal shared axes.

        Args:
            name: Parameter name.
            shape: Numpy array shape.
        """
        self.inputs.append(name)
        self.input_shapes[name] = shape
        dims: list[str] = []
        for _i in range(len(shape)):
            dim_id = f"d{self._dim_counter}"
            self._dim_counter += 1
            dims.append(dim_id)
        self.tensors[name] = TensorInfo(name=name, dims=tuple(dims), shape_2d=shape, is_input=True, producer_op=-1)

    def trace_op(
        self,
        op: NKIOp,
        operand_map: dict[str, str],
        operand_arrays: dict[str, np.ndarray],
        config_kwargs: dict[str, Any],
        output_names: list[str],
        result_arrays: list[np.ndarray],
    ) -> None:
        """Record a traced op call with dimension unification.

        Args:
            op: The NKIOp instance.
            operand_map: Maps operand slot name to tensor name.
            operand_arrays: Maps operand slot name to numpy array.
            config_kwargs: Non-tensor keyword arguments.
            output_names: Names for the outputs.
            result_arrays: Numpy result arrays.
        """
        op_idx = len(self.ops)
        label_to_dim = _unify_operand_dims(op, operand_map, self.tensors, self._dim_aliases)
        _update_tensor_dims_after_merge(self.tensors, self._dim_aliases)
        self._dim_counter = _register_outputs(
            op, output_names, result_arrays, label_to_dim, self.tensors, self._dim_aliases, self._dim_counter, op_idx
        )
        operand_shapes = _build_operand_shapes(traced_op_map=operand_map, tensors=self.tensors)
        self.ops.append(
            TracedOp(
                op_idx=op_idx,
                op=op,
                output_names=output_names,
                output_shapes=[arr.shape for arr in result_arrays],
                operand_names=operand_map,
                operand_shapes=operand_shapes,
                config_kwargs=config_kwargs,
            )
        )

    def compute_dim_info(self) -> None:
        """Compute tile sizes and num_blocks for all dimensions.

        Walks all ops to find the maximum tile size limit for each
        dimension, then computes num_blocks = total_size / tile_size.
        """
        dim_sizes = _gather_dim_sizes(self.tensors, self._dim_aliases)
        dim_tile_limits = _gather_tile_limits(self.ops, self.tensors, self._dim_aliases)
        for dim_id, total_size in dim_sizes.items():
            tile_limit = dim_tile_limits.get(dim_id, total_size)
            tile_size = min(tile_limit, total_size)
            num_blocks = total_size // tile_size if tile_size > 0 else 1
            self.dims[dim_id] = DimInfo(
                dim_id=dim_id, total_size=total_size, tile_size=tile_size, num_blocks=num_blocks, tiles_per_block=1
            )

    def compute_dtypes(self) -> None:
        """Compute dtype expressions for all tensors.

        Walks ops in order. Ops that produce float32 (matmul,
        tensor_reduce, activation_reduce reduce_res) get nl.float32.
        All other ops inherit from their primary data operand.
        """
        for inp in self.inputs:
            self.tensor_dtypes[inp] = f"{inp}.dtype"
        for traced_op in self.ops:
            op = traced_op.op
            output_keys = list(op.OUTPUT_AXES.keys())
            for out_idx, out_name in enumerate(traced_op.output_names):
                out_key = output_keys[out_idx]
                self.tensor_dtypes[out_name] = _dtype_for_op(op, out_key, traced_op, self.tensor_dtypes)


def _build_operand_shapes(traced_op_map: dict[str, str], tensors: dict[str, TensorInfo]) -> dict[str, tuple[int, ...]]:
    """Build operand shapes dict from operand map and tensors.

    Args:
        traced_op_map: Maps operand slot name to tensor name.
        tensors: All known tensors.

    Returns:
        Maps operand slot name to shape tuple.
    """
    operand_shapes: dict[str, tuple[int, ...]] = {}
    for slot_name, tensor_name in traced_op_map.items():
        tinfo = tensors.get(tensor_name)
        if tinfo is not None:
            operand_shapes[slot_name] = tinfo.shape_2d
    return operand_shapes
