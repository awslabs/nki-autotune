"""Dimension analysis: ID assignment, tile sizes, data-parallel classification.

Forward pass over all ops produces three results per dimension:
  1. A concrete dimension ID (d0, d1, ...)
  2. A tile size (max of hardware tile limits across ops)
  3. A data-parallel vs reduction classification

Usage::

    da = analyze_dims(matmul_nkigym, {"lhs_T": ((8192, 8192), "bfloat16"),
                                       "rhs": ((8192, 8192), "bfloat16")})
    print(da)
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.dim_analysis.parse import find_ops
from nkigym.ops.base import NKIOp


@dataclass
class DimInfo:
    """Per-dimension analysis result.

    Attributes:
        dim_size: Total number of elements along this dimension.
        tile_size: max(all op tile limits) on this dimension,
            clamped to dim_size. Referred to as d{i}_tile_size.
        min_tile_size: min(all op tile limits) on this dimension,
            clamped to dim_size. Referred to as d{i}_min_tile_size.
        is_data_parallel: True if this dimension appears in the
            kernel's return tensor.
    """

    dim_size: int
    tile_size: int
    min_tile_size: int
    is_data_parallel: bool


@dataclass
class TensorInfo:
    """Per-tensor analysis result.

    Attributes:
        dim_ids: Concrete dimension IDs (e.g. ``("d0", "d1")``).
        shape: Full shape (e.g. ``(2048, 128)``).
        dtype: Dtype string (e.g. ``"bfloat16"``).
        isa_loc: Where the tensor lives or is produced:
            ``"hbm"`` for kernel inputs,
            ``"psum"`` for nc_matmul/nc_transpose outputs,
            ``"sbuf"`` for vector-engine op outputs.
    """

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    isa_loc: str


@dataclass
class DimAnalysis:
    """Complete dimension analysis result.

    Attributes:
        func_name: Name of the math function.
        param_names: Input parameter names.
        return_name: Name of the returned tensor.
        dims: ``{dim_id: DimInfo}``.
        tensors: ``{tensor_name: TensorInfo}``.
        per_op_axis_maps: Per-op mapping from abstract axis
            names to concrete dim IDs. Used by the renderer
            to resolve ``BLOCKING_AXES`` to concrete dims.
        op_tile_sizes: Per-op tile sizes mapped to concrete
            dim IDs. ``op_tile_sizes[op_idx][dim_id]`` gives
            the op's own tile limit on that dimension. Used
            by the renderer for ISA call operand sizing.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dims: dict[str, DimInfo]
    tensors: dict[str, TensorInfo]
    per_op_axis_maps: list[dict[str, str]]
    op_tile_sizes: list[dict[str, int]]

    def __repr__(self) -> str:
        """Show final dimension analysis result."""
        lines: list[str] = []
        lines.append(f"DimAnalysis({self.func_name})")
        lines.append(f"  params: {self.param_names} -> {self.return_name}")

        lines.append("")
        lines.append("  Dimensions:")
        for dim_id, di in self.dims.items():
            par = "data-parallel" if di.is_data_parallel else "reduction"
            lines.append(
                f"    {dim_id}: size={di.dim_size},"
                f" {dim_id}_tile_size={di.tile_size},"
                f" {dim_id}_min_tile_size={di.min_tile_size},"
                f" {par}"
            )

        lines.append("")
        lines.append("  Tensors:")
        for name, t in self.tensors.items():
            dims_str = ", ".join(t.dim_ids)
            shape_str = ", ".join(str(s) for s in t.shape)
            lines.append(f"    {name}: ({dims_str}) shape=({shape_str}) {t.dtype} {t.isa_loc}")

        return "\n".join(lines)


@dataclass
class _Tensor:
    """Internal mutable tensor used during analysis."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    isa_loc: str
    dim_ids: list[str]


def _unify_dim(
    tensors: dict[str, _Tensor], per_op_maps: list[dict[str, str]], dim_sizes: dict[str, int], old_id: str, new_id: str
) -> None:
    """Rename *old_id* to *new_id* in all tensors and axis maps."""
    old_size = dim_sizes.get(old_id)
    new_size = dim_sizes.get(new_id)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(f"Cannot unify {old_id} (size {old_size})" f" with {new_id} (size {new_size})")
    if old_id in dim_sizes:
        dim_sizes.setdefault(new_id, dim_sizes.pop(old_id))

    for tensor in tensors.values():
        tensor.dim_ids = [new_id if d == old_id else d for d in tensor.dim_ids]
    for axis_map in per_op_maps:
        for ax in axis_map:
            if axis_map[ax] == old_id:
                axis_map[ax] = new_id


def _map_existing_dims(
    axes: tuple[str, ...],
    tensor: _Tensor,
    local: dict[str, str],
    tensors: dict[str, _Tensor],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    dim_counter: list[int],
) -> None:
    """Map dims from a tensor that already carries dim_ids.

    Only maps axes that the tensor actually has. Axes beyond the
    tensor's rank are absent (e.g. a 1D reduced tensor fed to a
    2D op) — no spurious size-1 dims are allocated.
    """
    for abstract, concrete in zip(axes, tensor.dim_ids):
        if abstract in local and local[abstract] != concrete:
            _unify_dim(tensors, per_op_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
        else:
            local[abstract] = concrete


def _map_fresh_dims(
    axes: tuple[str, ...], tensor: _Tensor, local: dict[str, str], dim_sizes: dict[str, int], dim_counter: list[int]
) -> None:
    """Allocate dims for a tensor with no dim_ids yet.

    Only allocates dims for axes the tensor actually has.
    Axes beyond the tensor's rank are skipped.
    """
    for i, abstract in enumerate(axes[: len(tensor.shape)]):
        if abstract not in local:
            fresh = f"d{dim_counter[0]}"
            dim_counter[0] += 1
            dim_sizes[fresh] = tensor.shape[i]
            local[abstract] = fresh
    tensor.dim_ids = [local[a] for a in axes[: len(tensor.shape)]]


def _build_axis_map(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    tensors: dict[str, _Tensor],
    dim_counter: list[int],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
) -> dict[str, str]:
    """Build local abstract->concrete axis map for one op."""
    local: dict[str, str] = {}

    for slot, axes in op_cls.OPERAND_AXES.items():
        if slot not in operand_map:
            continue
        tname = operand_map[slot]
        if tname not in tensors:
            continue
        tensor = tensors[tname]

        if tensor.dim_ids:
            _map_existing_dims(axes, tensor, local, tensors, per_op_maps, dim_sizes, dim_counter)
        else:
            _map_fresh_dims(axes, tensor, local, dim_sizes, dim_counter)

    return local


def _create_outputs(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    output_names: list[str],
    local: dict[str, str],
    tensors: dict[str, _Tensor],
    dim_sizes: dict[str, int],
) -> None:
    """Create output tensors for one op."""
    first_slot = next(iter(op_cls.OPERAND_AXES))
    has_first = first_slot in operand_map and operand_map[first_slot] in tensors
    dtype = (
        tensors[operand_map[first_slot]].dtype
        if has_first
        else next(t.dtype for t in tensors.values() if t.dtype != "")
    )

    for oname, (_, output_axes) in zip(output_names, op_cls.OUTPUT_AXES.items()):
        dim_ids = [local[a] for a in output_axes if a in local]
        shape = tuple(dim_sizes[d] for d in dim_ids)
        tensors[oname] = _Tensor(oname, shape, dtype, op_cls.ISA_LOC, dim_ids)


def _compute_tile_sizes(
    ops: list[tuple[type[NKIOp], dict[str, str], list[str]]],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    data_parallel_dims: set[str],
) -> tuple[dict[str, DimInfo], list[dict[str, int]]]:
    """Compute per-dimension tile sizes, classification, and per-op tile sizes."""
    max_tile: dict[str, int] = {}
    min_tile: dict[str, int] = {}
    op_tile_sizes: list[dict[str, int]] = []

    for (op_cls, _, _), local in zip(ops, per_op_maps):
        per_op: dict[str, int] = {}
        for abstract_axis, limit in op_cls.TILE_LIMITS.items():
            if abstract_axis not in local:
                continue
            dim_id = local[abstract_axis]
            clamped = min(limit, dim_sizes[dim_id])
            per_op[dim_id] = clamped
            max_tile[dim_id] = max(max_tile.get(dim_id, clamped), clamped)
            min_tile[dim_id] = min(min_tile.get(dim_id, clamped), clamped)
        op_tile_sizes.append(per_op)

    dims = {d: DimInfo(dim_sizes[d], max_tile[d], min_tile[d], d in data_parallel_dims) for d in max_tile}
    return dims, op_tile_sizes


def analyze_dims(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> DimAnalysis:
    """Run dimension analysis.

    Produces three results per dimension:
      1. Concrete dimension ID (d0, d1, ...)
      2. Tile size (max of hardware tile limits across ops)
      3. Data-parallel vs reduction classification

    Args:
        func: Math function using NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}``.

    Returns:
        DimAnalysis with dims and tensors.
    """
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    ops, return_name = find_ops(func)

    tensors: dict[str, _Tensor] = {}
    for name in param_names:
        shape, dtype = input_specs[name]
        tensors[name] = _Tensor(name, shape, dtype, "hbm", dim_ids=[])

    dim_counter = [0]
    per_op_maps: list[dict[str, str]] = []
    dim_sizes: dict[str, int] = {}

    for op_cls, name_kwargs, output_names in ops:
        operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
        local = _build_axis_map(op_cls, operand_map, tensors, dim_counter, per_op_maps, dim_sizes)
        per_op_maps.append(local)
        _create_outputs(op_cls, operand_map, output_names, local, tensors, dim_sizes)

    """Step 3: data-parallel classification from return tensor."""
    if return_name not in tensors:
        raise ValueError(f"Return tensor {return_name!r} not found")
    data_parallel_dims = set(tensors[return_name].dim_ids)

    dims, op_tile_sizes = _compute_tile_sizes(ops, per_op_maps, dim_sizes, data_parallel_dims)

    tensor_infos: dict[str, TensorInfo] = {}
    for name, t in tensors.items():
        tensor_infos[name] = TensorInfo(tuple(t.dim_ids), t.shape, t.dtype, t.isa_loc)

    return DimAnalysis(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dims=dims,
        tensors=tensor_infos,
        per_op_axis_maps=per_op_maps,
        op_tile_sizes=op_tile_sizes,
    )
