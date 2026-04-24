"""Build a baseline :class:`KernelIR` from an nkigym math function.

Pipeline:

1. AST-parse the math function → ordered list of
   ``(op_cls, name_kwargs, output_names)`` call-site tuples.
2. Trace the function once against dummy arrays to capture scalar
   kwarg values (``op='multiply'``, ``operand0=1e-6``, ...).
3. Forward pass over parsed ops unifies abstract axes (``P``, ``F``,
   ``K``, ``M``, ``N``) into concrete dim ids (``d0``, ``d1``, ...),
   producing ``dimensions`` and ``logical_tensors``.
4. Compute tile sizes (min of per-op ``TILE_LIMITS``) and dim roles
   (``ACCUMULATION`` if any reducing op blocks on it, ``SERIAL`` for
   non-reducing blocking, else ``PARALLEL``).
5. Synthesize ``physical_buffers``: one ``sbuf_<name>`` per tensor
   that's read downstream, plus ``<return>_hbm`` for the store dest.
6. Synthesize ``ops``: one ``NKILoad`` per kernel param, one ``Op``
   per parsed op (with SBUF-aliased inputs/outputs), one ``NKIStore``
   at the tail.
7. Derive ``edges`` from input/output producer-consumer links.
8. Leave all tunable knobs at canonical defaults — the agent mutates
   them to explore performance variants.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nkigym.kernel_ir.ir import KernelIR, Op, PhysicalBuffer
from nkigym.kernel_ir.parse import find_ops
from nkigym.kernel_ir.trace import trace_scalar_kwargs
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo
from nkigym.ops.base import NKIOp


@dataclass
class _Tensor:
    """Internal mutable tensor record used during dim inference."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dim_ids: list[str] = field(default_factory=list)


def build_ir(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Build a baseline ``KernelIR`` from a math function and input specs.

    Args:
        func: nkigym math function — a chain of ``NKIOp()(...)`` calls
            ending in ``return <variable>``.
        input_specs: ``{param_name: (shape, dtype)}`` for every
            parameter in ``func``'s signature.

    Returns:
        A ``KernelIR`` with DMA nodes inserted, ops filled out, edges
        derived, and all tunable knobs at naive defaults
        (``dim_order`` in insertion order, ``ltiles_per_block={d: 1}``,
        empty ``buffer_scopes``, ``num_buffers``, ``emission_depth``).
    """
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
        tensors[name] = _Tensor(name, shape, dtype)

    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for op_cls, name_kwargs, output_names in parsed:
        operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
        axis_map = _build_axis_map(op_cls, operand_map, tensors, dim_counter, per_op_axis_maps, dim_sizes)
        per_op_axis_maps.append(axis_map)
        _create_outputs(op_cls, operand_map, output_names, axis_map, tensors, dim_sizes)

    if return_name not in tensors:
        raise ValueError(f"Return tensor {return_name!r} not found among produced tensors")

    per_op_blocking: list[set[str]] = [
        {axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map}
        for (op_cls, _, _), axis_map in zip(parsed, per_op_axis_maps)
    ]
    dimensions = _resolve_dimensions(parsed, per_op_axis_maps, per_op_blocking, dim_sizes)
    _promote_float32_consumers(parsed, tensors)

    logical_tensors: dict[str, TensorInfo] = {
        name: TensorInfo(dim_ids=tuple(t.dim_ids), shape=t.shape, dtype=t.dtype) for name, t in tensors.items()
    }

    used = _collect_used_tensors(parsed, param_names, return_name, tensors)
    physical_buffers = _build_physical_buffers(tensors, used, dimensions, return_name)
    ops = _build_ops(parsed, per_op_axis_maps, per_op_blocking, traced, tensors, used, param_names, return_name)
    edges = _derive_edges(ops)
    dim_order = _default_dim_order(dimensions)
    ltiles_per_block = {d: 1 for d in dimensions}

    return KernelIR(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        physical_buffers=physical_buffers,
        ops=ops,
        edges=edges,
        dim_order=dim_order,
        ltiles_per_block=ltiles_per_block,
    )


def _default_dim_order(dimensions: dict[str, DimInfo]) -> list[str]:
    """Naive default: non-ACCUMULATION dims first, ACCUMULATION dims innermost.

    Accumulator buffers need the ACC loop to live *inside* every non-ACC
    loop so the accumulator state is local to each output block. Putting
    ACC dims at the tail of ``dim_order`` gives the renderer a valid
    starting point the agent can tune.
    """
    non_acc = [d for d, info in dimensions.items() if info.role is not DimRole.ACCUMULATION]
    acc = [d for d, info in dimensions.items() if info.role is DimRole.ACCUMULATION]
    return non_acc + acc


def _resolve_dimensions(
    parsed: list[tuple[type[NKIOp], dict[str, str], list[str]]],
    per_op_axis_maps: list[dict[str, str]],
    per_op_blocking: list[set[str]],
    dim_sizes: dict[str, int],
) -> dict[str, DimInfo]:
    """Compute ``DimInfo`` per dim: tile size = min of per-op limits; role from blocking + reducer."""
    per_dim_tile: dict[str, int] = {}
    for (op_cls, _, _), axis_map in zip(parsed, per_op_axis_maps):
        for abstract_axis, limit in op_cls.TILE_LIMITS.items():
            if abstract_axis not in axis_map:
                continue
            dim_id = axis_map[abstract_axis]
            tile = min(limit, dim_sizes[dim_id])
            per_dim_tile[dim_id] = min(per_dim_tile.get(dim_id, tile), tile)
    for d in dim_sizes:
        if d not in per_dim_tile:
            raise ValueError(f"Dim {d!r} has no op-declared tile size")

    dim_role: dict[str, DimRole] = {d: DimRole.PARALLEL for d in dim_sizes}
    for (op_cls, _, _), blocking in zip(parsed, per_op_blocking):
        for d in blocking:
            if op_cls.REDUCE_COMBINATOR:
                dim_role[d] = DimRole.ACCUMULATION
            elif dim_role[d] is DimRole.PARALLEL:
                dim_role[d] = DimRole.SERIAL

    return {
        d: DimInfo(
            dim_size=dim_sizes[d],
            logical_tile_size=per_dim_tile[d],
            physical_tile_size=per_dim_tile[d],
            role=dim_role[d],
        )
        for d in dim_sizes
    }


def _collect_used_tensors(
    parsed: list[tuple[type[NKIOp], dict[str, str], list[str]]],
    param_names: list[str],
    return_name: str,
    tensors: dict[str, _Tensor],
) -> set[str]:
    """Set of tensor names that are referenced as an input somewhere + params + return."""
    used: set[str] = {return_name, *param_names}
    for _op_cls, name_kwargs, _output_names in parsed:
        for v in name_kwargs.values():
            if v in tensors:
                used.add(v)
    return used


def _promote_float32_consumers(
    parsed: list[tuple[type[NKIOp], dict[str, str], list[str]]], tensors: dict[str, _Tensor]
) -> None:
    """Promote tensors to ``float32`` if they feed a FLOAT32_KWARGS operand.

    NKI HW mandates fp32 for ``tensor_scalar.operand0/operand1``,
    ``activation.scale``, ``activation_reduce.scale`` etc. If a tensor
    produced by op A is consumed as such a slot in op B, A's sbuf must
    be fp32. Promote A's output record and iterate to fixpoint so the
    promotion walks upstream through intermediate pass-through ops.
    """
    changed = True
    while changed:
        changed = False
        for op_cls, name_kwargs, _outputs in parsed:
            for slot, value in name_kwargs.items():
                if slot not in op_cls.FLOAT32_KWARGS:
                    continue
                if value in tensors and tensors[value].dtype != "float32":
                    tensors[value].dtype = "float32"
                    changed = True


def _build_physical_buffers(
    tensors: dict[str, _Tensor], used: set[str], dimensions: dict[str, DimInfo], return_name: str
) -> dict[str, PhysicalBuffer]:
    """Build ``sbuf_<name>`` entries for every used tensor plus ``<return>_hbm``."""
    result: dict[str, PhysicalBuffer] = {}
    for name, t in tensors.items():
        if name not in used or not t.dim_ids:
            continue
        p_axis = t.dim_ids[0]
        f_axis = t.dim_ids[1] if len(t.dim_ids) >= 2 else None
        p_tile = dimensions[p_axis].physical_tile_size
        f_tile = dimensions[f_axis].physical_tile_size if f_axis is not None else 1
        result[f"sbuf_{name}"] = PhysicalBuffer(
            tile=(p_tile, f_tile), dim_ids=tuple(t.dim_ids), dtype=t.dtype, p_axis=p_axis, f_axis=f_axis
        )
    ret_t = tensors[return_name]
    ret_p = ret_t.dim_ids[0]
    ret_f = ret_t.dim_ids[1] if len(ret_t.dim_ids) >= 2 else None
    hbm_p = dimensions[ret_p].dim_size
    hbm_f = dimensions[ret_f].dim_size if ret_f is not None else 1
    result[f"{return_name}_hbm"] = PhysicalBuffer(
        tile=(hbm_p, hbm_f), dim_ids=tuple(ret_t.dim_ids), dtype=ret_t.dtype, p_axis=ret_p, f_axis=ret_f
    )
    return result


def _build_ops(
    parsed: list[tuple[type[NKIOp], dict[str, str], list[str]]],
    per_op_axis_maps: list[dict[str, str]],
    per_op_blocking: list[set[str]],
    traced: list[dict[str, Any]],
    tensors: dict[str, _Tensor],
    used: set[str],
    param_names: list[str],
    return_name: str,
) -> list[Op]:
    """Assemble the final ops list: NKILoad header + math ops + NKIStore tail."""
    ops: list[Op] = []
    for p in param_names:
        ops.append(Op(kind="NKILoad", inputs={"data": p}, outputs=[f"sbuf_{p}"]))

    for (op_cls, name_kwargs, output_names), axis_map, blocking, scalars in zip(
        parsed, per_op_axis_maps, per_op_blocking, traced
    ):
        inputs, extra_tensor_kwargs = _split_tensor_inputs(op_cls, name_kwargs, tensors)
        kept_outputs = [o for o in output_names if o in used]
        sbuf_outputs = [f"sbuf_{o}" for o in kept_outputs]
        op_kwargs: dict[str, Any] = dict(scalars)
        op_kwargs.update(extra_tensor_kwargs)
        ops.append(
            Op(
                kind=op_cls.__name__,
                inputs=inputs,
                outputs=sbuf_outputs,
                kwargs=op_kwargs,
                axis_map=dict(axis_map),
                blocking_dims=set(blocking),
            )
        )

    ops.append(Op(kind="NKIStore", inputs={"data": f"sbuf_{return_name}"}, outputs=[f"{return_name}_hbm"]))
    return ops


def _split_tensor_inputs(
    op_cls: type[NKIOp], name_kwargs: dict[str, str], tensors: dict[str, _Tensor]
) -> tuple[dict[str, str], dict[str, str]]:
    """Split Name-valued kwargs into ``inputs`` vs extra tensor-kwargs for ``kwargs``.

    Inputs: every role in ``OPERAND_AXES`` whose value is a known tensor,
    plus any extra Name-valued kwarg whose value is a known tensor.
    The extras are also returned separately so the caller can stash them
    into ``Op.kwargs`` alongside scalar kwargs (codegen reads tensor-
    valued kwargs from ``kwargs``, not from ``inputs``).
    """
    inputs: dict[str, str] = {}
    extras: dict[str, str] = {}
    for role in op_cls.OPERAND_AXES:
        if role in name_kwargs and name_kwargs[role] in tensors:
            inputs[role] = f"sbuf_{name_kwargs[role]}"
    for role, tname in name_kwargs.items():
        if role in op_cls.OPERAND_AXES:
            continue
        if tname in tensors:
            alias = f"sbuf_{tname}"
            inputs[role] = alias
            extras[role] = alias
    return inputs, extras


def _derive_edges(ops: list[Op]) -> list[tuple[int, int, str, str]]:
    """Return ``[(producer_idx, consumer_idx, tensor_name, role), ...]``.

    Tracks the *last writer before the consumer*, so a non-SSA chain
    (``scaled = A(); scaled = B(scaled)``) edges to the previous writer
    instead of self-looping. Self-loops (an op reading a buffer it also
    writes, e.g. in-place ``activation(sbuf, sbuf)``) are dropped — they
    add no scheduling information.
    """
    producer_of: dict[str, int] = {}
    edges: list[tuple[int, int, str, str]] = []
    for i, op in enumerate(ops):
        for role, tname in op.inputs.items():
            src = producer_of.get(tname)
            if src is not None and src != i:
                edges.append((src, i, tname, role))
        for out in op.outputs:
            producer_of[out] = i
    return edges


def _unify_dim(
    tensors: dict[str, _Tensor],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    old_id: str,
    new_id: str,
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
    for axis_map in per_op_axis_maps:
        for ax in axis_map:
            if axis_map[ax] == old_id:
                axis_map[ax] = new_id


def _build_axis_map(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    tensors: dict[str, _Tensor],
    dim_counter: list[int],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
) -> dict[str, str]:
    """Build abstract->concrete axis map for one op by walking its tensor operands."""
    local: dict[str, str] = {}
    for slot, axes in op_cls.OPERAND_AXES.items():
        if slot not in operand_map:
            continue
        tname = operand_map[slot]
        if tname not in tensors:
            continue
        tensor = tensors[tname]
        if tensor.dim_ids:
            for abstract, concrete in zip(axes, tensor.dim_ids):
                if abstract in local and local[abstract] != concrete:
                    _unify_dim(tensors, per_op_axis_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
                else:
                    local[abstract] = concrete
        else:
            for i, abstract in enumerate(axes[: len(tensor.shape)]):
                if abstract not in local:
                    fresh = f"d{dim_counter[0]}"
                    dim_counter[0] += 1
                    dim_sizes[fresh] = tensor.shape[i]
                    local[abstract] = fresh
            tensor.dim_ids = [local[a] for a in axes[: len(tensor.shape)]]
    return local


def _create_outputs(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    output_names: list[str],
    local: dict[str, str],
    tensors: dict[str, _Tensor],
    dim_sizes: dict[str, int],
) -> None:
    """Create output tensor records from the op's abstract ``OUTPUT_AXES``."""
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
