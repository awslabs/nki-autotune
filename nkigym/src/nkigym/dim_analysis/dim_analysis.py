"""Dimension analysis (design doc section 3).

Forward pass over all ops assigns concrete dimension IDs and
computes per-dimension unified/min tile sizes and per-op tiling
parameters.  No tensor starts with dimension IDs -- they are
discovered automatically from op structure.

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
    """Per-dimension global info, computed once by analyze_dims.

    Attributes:
        dim_size: Total number of elements along this dimension.
        unified_tile_size: max(all op tile limits) on this dimension,
            clamped to dim_size.
        min_tile_size: min(all op tile limits) on this dimension,
            clamped to dim_size.
    """

    dim_size: int
    unified_tile_size: int
    min_tile_size: int

    def __repr__(self) -> str:
        """Compact single-line summary."""
        return f"DimInfo(size={self.dim_size}," f" unified={self.unified_tile_size}," f" min={self.min_tile_size})"


@dataclass
class TensorInfo:
    """Per-tensor info, computed once by analyze_dims.

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

    def __repr__(self) -> str:
        """Compact single-line summary."""
        dims = ", ".join(self.dim_ids)
        shape = ", ".join(str(s) for s in self.shape)
        return f"TensorInfo(({dims}), ({shape})," f" {self.dtype}, {self.isa_loc})"


@dataclass
class OpDimInfo:
    """Per-op per-dimension tiling info, derived from hardware limits.

    Attributes:
        op_tile_size: Hardware tile size for this op on this dim.
        num_ig: ``unified_tile_size / op_tile_size``.
        tiles_per_ig: ``op_tile_size / min_tile_size``.
    """

    op_tile_size: int
    num_ig: int
    tiles_per_ig: int

    def __repr__(self) -> str:
        """Compact single-line summary."""
        return f"OpDimInfo(tile={self.op_tile_size}," f" ig={self.num_ig}," f" per_ig={self.tiles_per_ig})"


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

    def __repr__(self) -> str:
        """Multi-line summary with dim map and tiling."""
        operands = ", ".join(f"{k}={v}" for k, v in self.operands.items())
        outs = ", ".join(self.outputs)
        dm = ", ".join(f"{k}->{v}" for k, v in self.dim_map.items())
        pd = "\n".join(f"      {k}: {v}" for k, v in self.per_dim.items())
        blk = ", ".join(sorted(self.blocking_axes)) or "none"
        return (
            f"OpInfo({self.op_type}({operands}) -> {outs}\n"
            f"    dim_map: {dm}\n"
            f"    per_dim:\n{pd}\n"
            f"    blocking: {{{blk}}},"
            f" preds: {self.predecessors})"
        )


@dataclass
class OpGraph:
    """Computation DAG -- ops in topological order.

    Attributes:
        nodes: Op nodes indexed by op_idx.
        tensor_producers: ``{tensor_name: op_idx}``.
    """

    nodes: list[OpInfo]
    tensor_producers: dict[str, int]

    def __repr__(self) -> str:
        """Summary listing each op node."""
        lines = "\n".join(f"    [{i}] {n.op_type} -> {', '.join(n.outputs)}" for i, n in enumerate(self.nodes))
        return f"OpGraph({len(self.nodes)} ops\n{lines})"


@dataclass
class DimAnalysis:
    """Complete result of section 3 dimension analysis.

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
        """Multi-line summary of dims, tensors, and ops."""
        lines = [f"DimAnalysis({self.func_name})", f"  params: {self.param_names} -> {self.return_name}", "  dims:"]
        for d, info in self.dims.items():
            lines.append(f"    {d}: {info}")
        lines.append("  tensors:")
        for name, t in self.tensors.items():
            lines.append(f"    {name}: {t}")
        lines.append("  ops:")
        for i, node in enumerate(self.op_graph.nodes):
            ops_str = ", ".join(f"{k}={v}" for k, v in node.operands.items())
            outs = ", ".join(node.outputs)
            lines.append(f"    [{i}] {node.op_type}({ops_str}) -> {outs}")
            for dim_id, odi in node.per_dim.items():
                lines.append(f"        {dim_id}: {odi}")
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
    """Map dims from a tensor that already carries dim_ids."""
    for abstract, concrete in zip(axes, tensor.dim_ids):
        if abstract in local and local[abstract] != concrete:
            _unify_dim(tensors, per_op_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
        else:
            local[abstract] = concrete
    for abstract in axes[len(tensor.dim_ids) :]:
        if abstract not in local:
            fresh = f"d{dim_counter[0]}"
            dim_counter[0] += 1
            dim_sizes[fresh] = 1
            local[abstract] = fresh


def _map_fresh_dims(
    axes: tuple[str, ...], tensor: _Tensor, local: dict[str, str], dim_sizes: dict[str, int], dim_counter: list[int]
) -> None:
    """Allocate dims for a tensor with no dim_ids yet."""
    for i, abstract in enumerate(axes):
        if abstract not in local:
            fresh = f"d{dim_counter[0]}"
            dim_counter[0] += 1
            size = tensor.shape[i] if i < len(tensor.shape) else 1
            dim_sizes[fresh] = size
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
        dim_ids = [local[a] for a in output_axes]
        shape = tuple(dim_sizes[d] for d in dim_ids)
        tensors[oname] = _Tensor(oname, shape, dtype, op_cls.ISA_LOC, dim_ids)


def _compute_tile_sizes(
    ops: list[tuple[type[NKIOp], dict[str, str], list[str]]],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
) -> dict[str, DimInfo]:
    """Compute per-dimension unified and min tile sizes."""
    unified: dict[str, int] = {}
    minimum: dict[str, int] = {}

    for (op_cls, _, _), local in zip(ops, per_op_maps):
        for abstract_axis, limit in op_cls.TILE_LIMITS.items():
            dim_id = local[abstract_axis]
            unified[dim_id] = max(unified.get(dim_id, limit), limit)
            minimum[dim_id] = min(minimum.get(dim_id, limit), limit)

    for dim_id in unified:
        unified[dim_id] = min(unified[dim_id], dim_sizes[dim_id])
    for dim_id in minimum:
        minimum[dim_id] = min(minimum[dim_id], dim_sizes[dim_id])

    return {d: DimInfo(dim_sizes[d], unified[d], minimum[d]) for d in unified}


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
        dim_id = per_op_maps[i][abstract_axis]
        di = dims[dim_id]
        ot = min(limit, di.unified_tile_size)
        per_dim[dim_id] = OpDimInfo(ot, di.unified_tile_size // ot, ot // di.min_tile_size)

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
    """Run dimension analysis (design doc section 3).

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

    dims = _compute_tile_sizes(ops, per_op_maps, dim_sizes)

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
