"""Build the initial ``(KernelContext, KernelGraph)`` from a math function.

Pipeline:

1. AST-parse the math function → ordered list of
   ``(op_cls, name_kwargs, output_names, all_kwargs)`` call-site
   tuples.
2. Instantiate one ``NKIOp`` per call site — each gets a
   distinct instance used as a dict key in ``KernelContext``.
3. Forward pass resolves per-op axis maps, tile sizes, blocking
   dims, and the logical-tensor catalog.
4. Wrap each op in a singleton ``FusionGroup``; assemble the
   graph with group-level edges.
"""

import inspect
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from nkigym.kernel_ir.context.context import DimInfo, DimRole, KernelContext, TensorInfo
from nkigym.kernel_ir.context.parse import find_ops
from nkigym.kernel_ir.context.trace import trace_scalar_kwargs
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, insert_dma_nodes, rebuild_edges
from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.load_transpose_pattern import LoadTransposePattern
from nkigym.kernel_ir.rewrites.merge_composites import MergeComposites
from nkigym.kernel_ir.rewrites.online_fusion_pattern import OnlineFusionPattern
from nkigym.kernel_ir.rewrites.pattern_rewrite import PatternRewrite, enumerate_graph_variants
from nkigym.kernel_ir.sampler.sampler import sample_valid_ir
from nkigym.ops.base import NKIOp

"""Structural graph rewrites exhaustively enumerated at variant-build time.

These produce a finite, small set of structurally distinct
graphs (each rewrite has O(1) or O(N) matches and terminates in
a handful of applies). The outer layer of hierarchical sampling
picks uniformly among them.
"""
GRAPH_REWRITES: list[PatternRewrite] = cast(list[PatternRewrite], [OnlineFusionPattern(), LoadTransposePattern()])

"""Merge rewrite — applied stochastically per-draw in the inner sampler.

Unlike ``GRAPH_REWRITES`` whose match sets are small, merges
have O(N²) matches per graph and chained merges produce a
Bell-number-sized lattice of partitions. Exhaustive enumeration
is infeasible; the sampler randomly applies a subset per draw
under R1 convexity. Semantically still a ``PatternRewrite``;
just plugged into a different layer of the sampler.
"""
MERGE_REWRITES: list[PatternRewrite] = cast(list[PatternRewrite], [MergeComposites()])


@dataclass
class _Tensor:
    """Internal mutable tensor used during analysis."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dim_ids: list[str]


def _unify_dim(
    tensors: dict[str, _Tensor], per_op_maps: list[dict[str, str]], dim_sizes: dict[str, int], old_id: str, new_id: str
) -> None:
    """Rename *old_id* to *new_id* in all tensors and axis maps."""
    old_size = dim_sizes.get(old_id)
    new_size = dim_sizes.get(new_id)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(f"Cannot unify {old_id} (size {old_size}) with {new_id} (size {new_size})")
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
) -> None:
    """Map dims from a tensor that already carries dim_ids."""
    for abstract, concrete in zip(axes, tensor.dim_ids):
        if abstract in local and local[abstract] != concrete:
            _unify_dim(tensors, per_op_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
        else:
            local[abstract] = concrete


def _map_fresh_dims(
    axes: tuple[str, ...], tensor: _Tensor, local: dict[str, str], dim_sizes: dict[str, int], dim_counter: list[int]
) -> None:
    """Allocate dims for a tensor with no dim_ids yet."""
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
    """Build abstract->concrete axis map for one op."""
    local: dict[str, str] = {}
    for slot, axes in op_cls.OPERAND_AXES.items():
        if slot not in operand_map:
            continue
        tname = operand_map[slot]
        if tname not in tensors:
            continue
        tensor = tensors[tname]
        if tensor.dim_ids:
            _map_existing_dims(axes, tensor, local, tensors, per_op_maps, dim_sizes)
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
        tensors[oname] = _Tensor(oname, shape, dtype, dim_ids)


def _resolve_blocking_dims(op_cls: type[NKIOp], axis_map: dict[str, str]) -> set[str]:
    """Resolve class-level ``BLOCKING_AXES`` through an op's axis map."""
    return {axis_map[axis] for axis in op_cls.BLOCKING_AXES if axis in axis_map}


_ParsedOp = tuple[type[NKIOp], dict[str, str], list[str], dict[str, str]]


def _compute_tile_sizes(
    parsed_ops: list[_ParsedOp],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    blocking_dims: set[str],
    tensors: dict[str, _Tensor],
) -> tuple[dict[str, DimInfo], list[dict[str, int]]]:
    """Compute per-dim tile sizes + DimInfo map + per-op tile-size maps."""
    PARTITION_MAX = 128
    partition_dims: set[str] = set()
    for t in tensors.values():
        if t.dim_ids:
            partition_dims.add(t.dim_ids[0])
    max_tile: dict[str, int] = {}
    min_tile: dict[str, int] = {}
    per_op_tile_sizes: list[dict[str, int]] = []
    for (op_cls, _, _, _), local in zip(parsed_ops, per_op_maps):
        per_op: dict[str, int] = {}
        for abstract_axis, limit in op_cls.TILE_LIMITS.items():
            if abstract_axis not in local:
                continue
            dim_id = local[abstract_axis]
            clamped = min(limit, dim_sizes[dim_id])
            per_op[dim_id] = clamped
            max_tile[dim_id] = max(max_tile.get(dim_id, clamped), clamped)
            min_tile[dim_id] = min(min_tile.get(dim_id, clamped), clamped)
        per_op_tile_sizes.append(per_op)
    for dim_id in max_tile:
        max_tile[dim_id] = min(max_tile[dim_id], dim_sizes[dim_id])
        if dim_id in partition_dims:
            min_tile[dim_id] = min(min_tile[dim_id], PARTITION_MAX)
    dims = {
        d: DimInfo(dim_sizes[d], max_tile[d], min_tile[d], DimRole.SERIAL if d in blocking_dims else DimRole.PARALLEL)
        for d in max_tile
    }
    return dims, per_op_tile_sizes


def build_initial(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[KernelContext, KernelGraph]:
    """Build the initial ``(context, graph)`` pair for a math function."""
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    parsed, return_name = find_ops(func)
    traced = trace_scalar_kwargs(func, input_specs)
    if len(traced) != len(parsed):
        raise ValueError(f"Traced {len(traced)} op calls but AST found {len(parsed)}")

    tensors: dict[str, _Tensor] = {}
    for name in param_names:
        shape, dtype = input_specs[name]
        tensors[name] = _Tensor(name, shape, dtype, dim_ids=[])

    dim_counter = [0]
    per_op_maps: list[dict[str, str]] = []
    dim_sizes: dict[str, int] = {}
    for op_cls, name_kwargs, output_names, _all_kwargs in parsed:
        operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
        local = _build_axis_map(op_cls, operand_map, tensors, dim_counter, per_op_maps, dim_sizes)
        per_op_maps.append(local)
        _create_outputs(op_cls, operand_map, output_names, local, tensors, dim_sizes)

    if return_name not in tensors:
        raise ValueError(f"Return tensor {return_name!r} not found")

    per_op_blocking: list[set[str]] = [
        _resolve_blocking_dims(op_cls, axis_map) for (op_cls, _, _, _), axis_map in zip(parsed, per_op_maps)
    ]
    blocking_union: set[str] = set().union(*per_op_blocking) if per_op_blocking else set()
    dimensions, per_op_tile_sizes = _compute_tile_sizes(parsed, per_op_maps, dim_sizes, blocking_union, tensors)

    logical_tensors: dict[str, TensorInfo] = {
        name: TensorInfo(tuple(t.dim_ids), t.shape, t.dtype) for name, t in tensors.items()
    }

    op_instances: list[NKIOp] = [op_cls() for op_cls, _, _, _ in parsed]
    op_inputs: dict[NKIOp, dict[str, str]] = {}
    op_outputs: dict[NKIOp, list[str]] = {}
    op_kwargs: dict[NKIOp, dict[str, str]] = {}
    op_axis_map: dict[NKIOp, dict[str, str]] = {}
    op_tile_sizes: dict[NKIOp, dict[str, int]] = {}
    op_blocking_dims: dict[NKIOp, set[str]] = {}
    for i, (op, (_cls, name_kwargs, output_names, all_kwargs)) in enumerate(zip(op_instances, parsed)):
        inputs = {role: tname for role, tname in name_kwargs.items() if tname in logical_tensors}
        op_inputs[op] = inputs
        op_outputs[op] = list(output_names)
        merged_kwargs = dict(all_kwargs)
        merged_kwargs.update(traced[i])
        op_kwargs[op] = merged_kwargs
        op_axis_map[op] = per_op_maps[i]
        op_tile_sizes[op] = per_op_tile_sizes[i]
        op_blocking_dims[op] = per_op_blocking[i]

    ltiles_per_block: dict[str, int] = {dim_id: 1 for dim_id in dimensions}

    context = KernelContext(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        ltiles_per_block=ltiles_per_block,
        op_inputs=op_inputs,
        op_outputs=op_outputs,
        op_kwargs=op_kwargs,
        op_axis_map=op_axis_map,
        op_tile_sizes=op_tile_sizes,
        op_blocking_dims=op_blocking_dims,
    )

    groups = [FusionGroup(ops=[op]) for op in op_instances]
    graph = KernelGraph(groups=groups)
    rebuild_edges(graph, context)
    return context, graph


def build_context_and_variants(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[KernelContext, list[KernelGraph]]:
    """Return ``(context, variants)`` after enumerating every reachable graph.

    Each rewrite creates new composite op instances and drops the
    absorbed ones; per-op dicts on each variant's ``KernelContext``
    hold only the ops that variant's graph contains. Downstream
    codegen takes one ``(context, graph)`` pair at a time, so we
    merge every variant's per-op data into a single
    ``shared_ctx`` — op-instance keys never collide across
    variants, and this keeps the caller's signature simple.
    """
    context, graph = build_initial(func, input_specs)
    context, graph = insert_dma_nodes(context, graph)
    pairs = enumerate_graph_variants(context, graph, GRAPH_REWRITES)
    if pairs:
        shared_ctx = _merge_contexts([c for c, _g in pairs])
        variants = [g for _c, g in pairs]
    else:
        shared_ctx = context
        variants = [graph]
    return shared_ctx, variants


def _merge_contexts(contexts: list[KernelContext]) -> KernelContext:
    """Union every per-op dict across variants so downstream lookups always find the op."""
    if not contexts:
        raise ValueError("cannot merge empty context list")
    base = contexts[-1]
    op_inputs: dict[NKIOp, dict[str, str]] = {}
    op_outputs: dict[NKIOp, list[str]] = {}
    op_kwargs: dict[NKIOp, dict[str, str]] = {}
    op_axis_map: dict[NKIOp, dict[str, str]] = {}
    op_tile_sizes: dict[NKIOp, dict[str, int]] = {}
    op_blocking_dims: dict[NKIOp, set[str]] = {}
    logical_tensors: dict[str, TensorInfo] = {}
    for ctx in contexts:
        op_inputs.update(ctx.op_inputs)
        op_outputs.update(ctx.op_outputs)
        op_kwargs.update(ctx.op_kwargs)
        op_axis_map.update(ctx.op_axis_map)
        op_tile_sizes.update(ctx.op_tile_sizes)
        op_blocking_dims.update(ctx.op_blocking_dims)
        logical_tensors.update(ctx.logical_tensors)
    return KernelContext(
        func_name=base.func_name,
        param_names=base.param_names,
        return_name=base.return_name,
        dimensions=base.dimensions,
        logical_tensors=logical_tensors,
        ltiles_per_block=base.ltiles_per_block,
        required_merges=base.required_merges,
        op_inputs=op_inputs,
        op_outputs=op_outputs,
        op_kwargs=op_kwargs,
        op_axis_map=op_axis_map,
        op_tile_sizes=op_tile_sizes,
        op_blocking_dims=op_blocking_dims,
    )


def build_ir(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], seed: int | None = None
) -> KernelIR:
    """Convenience: build context + pick last variant + sample one valid state."""
    context, variants = build_context_and_variants(func, input_specs)
    rng = random.Random(seed)
    seed_ir = KernelIR(context=context, graph=variants[-1])
    return sample_valid_ir(seed_ir, rng, merge_rewrites=MERGE_REWRITES)
