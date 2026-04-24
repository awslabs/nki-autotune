"""Build the initial ``(KernelIR, KernelIR)`` from a math function.

Pipeline:

1. AST-parse the math function → ordered list of
   ``(op_cls, name_kwargs, output_names, all_kwargs)`` call-site
   tuples.
2. Instantiate one ``NKIOp`` per call site — each gets a
   distinct instance used as a dict key in ``KernelIR``.
3. Forward pass resolves per-op axis maps, tile sizes, blocking
   dims, and the logical-tensor catalog.
4. Wrap each op in a singleton ``FusionGroup``; assemble the
   ir with group-level edges.
"""

import inspect
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from nkigym.kernel_ir.compute_skip_prop import propagate_compute_skip
from nkigym.kernel_ir.fusion_group import FusionGroup
from nkigym.kernel_ir.ir import KernelIR, insert_dma_nodes, rebuild_edges
from nkigym.kernel_ir.parse import find_ops
from nkigym.kernel_ir.rewrites.load_transpose_pattern import LoadTransposePattern
from nkigym.kernel_ir.rewrites.loop_fusion import LoopFusion
from nkigym.kernel_ir.rewrites.online_fusion_pattern import OnlineFusionPattern
from nkigym.kernel_ir.rewrites.pattern_rewrite import PatternRewrite, apply_rewrites_until_fixpoint
from nkigym.kernel_ir.sampler.sampler import sample_valid_ir
from nkigym.kernel_ir.trace import trace_scalar_kwargs
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo
from nkigym.ops.base import NKIOp

"""Graph rewrites — split into mandatory pre-passes and sampled rewrites.

``LoopFusion`` is applied to fixpoint as a mandatory pre-pass
inside ``build_naive_ir`` because it is a strict no-op-or-better
transform: merging a producer→consumer pair whose shared dims are
all PARALLEL in the producer preserves semantics, and any kernel
the sampler could produce with an unfused partition is strictly
reachable via placement draws on the merged partition (choosing
``full`` tier on the shared tensor's dims reproduces the
pre-loaded outer-group pattern). Leaving it as a sampled choice
wasted rewrite-count budget on a redundant search axis.

The remaining ``REWRITES`` cover transforms with genuine
performance trade-offs: the sampler picks ``k ∈ [0, num_ops -
1]`` uniformly, then applies ``k`` rewrites (one match per step,
chosen uniformly across all currently-matching
``(pattern, instance)`` pairs).
"""
_MANDATORY_REWRITES: list[PatternRewrite] = cast(list[PatternRewrite], [LoopFusion()])
REWRITES: list[PatternRewrite] = cast(list[PatternRewrite], [OnlineFusionPattern(), LoadTransposePattern()])


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


def build_initial(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Build the initial ``KernelIR`` for a math function — one singleton group per op."""
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
    groups = [FusionGroup(ops=[op]) for op in op_instances]

    ir = KernelIR(
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
        groups=groups,
    )
    rebuild_edges(ir)
    return ir


def build_naive_ir(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[KernelIR, KernelIR]:
    """Return ``(naive_ir, seed_ir)`` — pre-rewrite and rewrite-ready KernelIR.

    Pipeline:

    1. ``build_initial`` + ``insert_dma_nodes`` → naive ``KernelIR``.
       Used to emit ``kernel_0`` — the raw operator ir with
       ``NKIAffineSelect`` still present, no compute-skip
       annotations, no rewrites, no merges.
    2. ``propagate_compute_skip`` → mandatory pre-pass that lifts
       every ``NKIAffineSelect`` into per-op ``SkipPredicate``
       annotations and removes the standalone op.
    3. ``_MANDATORY_REWRITES`` applied to fixpoint — currently
       just ``LoopFusion``, which is a strict no-op-or-better
       transform and is therefore lifted out of the sampled
       ``REWRITES`` set.
    4. The resulting ``KernelIR`` is the seed every
       ``sample_valid_ir`` call starts from; remaining
       ``REWRITES`` are sampled per-draw inside
       ``sample_valid_ir``.
    """
    ir = build_initial(func, input_specs)
    ir = insert_dma_nodes(ir)
    naive = ir
    ir = propagate_compute_skip(ir)
    ir = apply_rewrites_until_fixpoint(ir, _MANDATORY_REWRITES)
    return naive, ir


def build_ir(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], seed: int | None = None
) -> KernelIR:
    """Convenience: build seed IR and sample one valid state."""
    _naive, seed_ir = build_naive_ir(func, input_specs)
    rng = random.Random(seed)
    return sample_valid_ir(seed_ir, rng, rewrites=REWRITES)
