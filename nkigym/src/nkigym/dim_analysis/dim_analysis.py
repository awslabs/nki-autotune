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
class OpDimInfo:
    """Per-op per-dimension tiling info, derived from hardware limits.

    Attributes:
        op_tile_size: Hardware tile size for this op on this dim.
        num_ig: d{i}_tile_size / op_tile_size.
        tiles_per_ig: op_tile_size / d{i}_min_tile_size.
    """

    op_tile_size: int
    num_ig: int
    tiles_per_ig: int


@dataclass
class OpInfo:
    """Node in the computation DAG.

    Attributes:
        op_type: ISA call name (e.g. ``"nc_matmul"``).
        op_cls: The NKIOp subclass.
        operands: Tensor-valued kwargs ``{role: tensor_name}``.
        outputs: Output tensor names.
        dim_map: ``{abstract_axis: dim_id}`` for this op.
        per_dim: ``{dim_id: OpDimInfo}`` tiling info.
        predecessors: Indices of ops producing this op's inputs.
        blocking_axes: Abstract axes that are blocking.
    """

    op_type: str
    op_cls: type
    operands: dict[str, str]
    outputs: list[str]
    dim_map: dict[str, str]
    per_dim: dict[str, OpDimInfo]
    predecessors: list[int]
    blocking_axes: frozenset[str]


@dataclass
class OpGraph:
    """Computation DAG -- ops in topological order.

    Attributes:
        nodes: Op nodes indexed by op_idx.
        tensor_producers: ``{tensor_name: op_idx}``.
    """

    nodes: list[OpInfo]
    tensor_producers: dict[str, int]


@dataclass
class DimAnalysis:
    """Complete dimension analysis result.

    Attributes:
        func_name: Name of the math function.
        param_names: Input parameter names.
        return_name: Name of the returned tensor.
        dims: ``{dim_id: DimInfo}``.
        tensors: ``{tensor_name: TensorInfo}``.
        op_graph: The computation DAG.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dims: dict[str, DimInfo]
    tensors: dict[str, TensorInfo]
    op_graph: OpGraph

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

        lines.append("")
        lines.append("  Ops:")
        for i, node in enumerate(self.op_graph.nodes):
            ops_str = ", ".join(f"{k}={v}" for k, v in node.operands.items())
            outs = ", ".join(node.outputs)
            dm_str = ", ".join(f"{k}->{v}" for k, v in node.dim_map.items())
            lines.append(f"    [{i}] {node.op_type}({ops_str}) -> {outs}")
            lines.append(f"        dim_map: {dm_str}")
            for dim_id, odi in node.per_dim.items():
                lines.append(
                    f"        {dim_id}: op_tile={odi.op_tile_size},"
                    f" num_ig={odi.num_ig},"
                    f" tiles_per_ig={odi.tiles_per_ig}"
                )

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
) -> dict[str, DimInfo]:
    """Compute per-dimension tile sizes and classification."""
    max_tile: dict[str, int] = {}
    min_tile: dict[str, int] = {}

    for (op_cls, _, _), local in zip(ops, per_op_maps):
        for abstract_axis, limit in op_cls.TILE_LIMITS.items():
            if abstract_axis not in local:
                continue
            dim_id = local[abstract_axis]
            max_tile[dim_id] = max(max_tile.get(dim_id, limit), limit)
            min_tile[dim_id] = min(min_tile.get(dim_id, limit), limit)

    for dim_id in max_tile:
        max_tile[dim_id] = min(max_tile[dim_id], dim_sizes[dim_id])
    for dim_id in min_tile:
        min_tile[dim_id] = min(min_tile[dim_id], dim_sizes[dim_id])

    return {d: DimInfo(dim_sizes[d], max_tile[d], min_tile[d], d in data_parallel_dims) for d in max_tile}


def _build_op_node(
    i: int,
    op_cls: type[NKIOp],
    name_kwargs: dict[str, str],
    output_names: list[str],
    per_op_maps: list[dict[str, str]],
    tensors: dict[str, _Tensor],
    dims: dict[str, DimInfo],
    tensor_producers: dict[str, int],
) -> OpInfo:
    """Build a single OpInfo node."""
    operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
    preds = sorted({tensor_producers[v] for v in operand_map.values() if v in tensor_producers})
    per_dim: dict[str, OpDimInfo] = {}
    for abstract_axis, limit in op_cls.TILE_LIMITS.items():
        if abstract_axis not in per_op_maps[i]:
            continue
        dim_id = per_op_maps[i][abstract_axis]
        di = dims[dim_id]
        ot = min(limit, di.tile_size)
        per_dim[dim_id] = OpDimInfo(ot, di.tile_size // ot, ot // di.min_tile_size)

    return OpInfo(
        op_type=op_cls.NAME,
        op_cls=op_cls,
        operands=operand_map,
        outputs=output_names,
        dim_map=dict(per_op_maps[i]),
        per_dim=per_dim,
        predecessors=preds,
        blocking_axes=op_cls.BLOCKING_AXES,
    )


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
        DimAnalysis with dims, tensors, and op_graph.
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

    dims = _compute_tile_sizes(ops, per_op_maps, dim_sizes, data_parallel_dims)

    tensor_producers: dict[str, int] = {}
    op_nodes: list[OpInfo] = []
    for i, (op_cls, name_kwargs, output_names) in enumerate(ops):
        for oname in output_names:
            tensor_producers[oname] = i
        node = _build_op_node(i, op_cls, name_kwargs, output_names, per_op_maps, tensors, dims, tensor_producers)
        op_nodes.append(node)

    tensor_infos: dict[str, TensorInfo] = {}
    for name, t in tensors.items():
        tensor_infos[name] = TensorInfo(tuple(t.dim_ids), t.shape, t.dtype, t.isa_loc)

    return DimAnalysis(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dims=dims,
        tensors=tensor_infos,
        op_graph=OpGraph(op_nodes, tensor_producers),
    )
