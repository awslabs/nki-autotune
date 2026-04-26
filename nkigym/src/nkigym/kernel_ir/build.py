"""Build a baseline :class:`KernelIR` from an nkigym math function.

Pipeline:

1. AST-parse the math function → ordered list of
   ``(op_cls, name_kwargs, op_kwargs, output_names)`` call-site tuples.
2. Forward pass over parsed ops unifies abstract axes (``P``, ``F``,
   ``K``, ``M``, ``N``, ...) into concrete dim ids (``d0``, ``d1``, ...),
   producing ``dimensions`` and ``logical_tensors``.
3. Compute tile sizes (min of per-op ``TILE_LIMITS``) and dim roles.
4. Synthesize ``physical_buffers``: one ``sbuf_<name>`` per referenced
   tensor plus ``hbm_<return>`` for the store destination, plus a PSUM
   sibling for every matmul-style op.
5. Synthesize ``ops``: one ``NKILoad`` per kernel param, one ``Op`` per
   parsed op, one ``NKIStore`` at the tail.
6. Derive ``edges`` from input/output producer-consumer links.
7. Set canonical knobs: 2N-entry ``loop_order`` (all ``.block`` first,
   then all ``.tile``); ``ltiles_per_block[d] = 1``; per-dim
   ``buffer_scopes[buf][d] = PER_BLOCK``.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from nkigym.kernel_ir.ir import DimScope, KernelIR, Op, PhysicalBuffer
from nkigym.kernel_ir.parse import find_ops
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo
from nkigym.ops.base import NKIOp
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore

_MATMUL_KINDS = frozenset({"NKIMatmul"})


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
        input_specs: ``{param_name: (shape, dtype)}`` for every parameter.

    Returns:
        A ``KernelIR`` with ops filled out, edges derived, and tunable
        knobs at canonical defaults.
    """
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    parsed, return_name = find_ops(func)
    tensors: dict[str, _Tensor] = {}
    for name in param_names:
        shape, dtype = input_specs[name]
        tensors[name] = _Tensor(name, shape, dtype)

    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for op_cls, name_kwargs, _op_kwargs, output_names in parsed:
        operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
        axis_map = _build_axis_map(op_cls, operand_map, tensors, dim_counter, per_op_axis_maps, dim_sizes)
        per_op_axis_maps.append(axis_map)
        _create_outputs(op_cls, operand_map, output_names, axis_map, tensors, dim_sizes)

    if return_name not in tensors:
        raise ValueError(f"Return tensor {return_name!r} not found among produced tensors")

    per_op_blocking: list[set[str]] = [
        {axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map}
        for (op_cls, _, _, _), axis_map in zip(parsed, per_op_axis_maps)
    ]
    dimensions = _resolve_dimensions(parsed, per_op_axis_maps, per_op_blocking, dim_sizes)

    logical_tensors: dict[str, TensorInfo] = {
        name: TensorInfo(dim_ids=tuple(t.dim_ids), shape=t.shape, dtype=t.dtype) for name, t in tensors.items()
    }

    used = _collect_used_tensors(parsed, param_names, return_name, tensors)
    physical_buffers = _build_physical_buffers(tensors, used, dimensions, return_name)
    ops = _build_ops(parsed, per_op_axis_maps, per_op_blocking, tensors, used, param_names, return_name)
    _mint_psum_buffers(ops, physical_buffers, dimensions)
    edges = _derive_edges(ops)
    loop_order = _default_loop_order(dimensions)
    ltiles_per_block = {d: 1 for d in dimensions}
    buffer_scopes = _default_buffer_scopes(physical_buffers, ops)

    return KernelIR(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        physical_buffers=physical_buffers,
        ops=ops,
        edges=edges,
        loop_order=loop_order,
        ltiles_per_block=ltiles_per_block,
        buffer_scopes=buffer_scopes,
    )


def _default_loop_order(dimensions: dict[str, DimInfo]) -> list[str]:
    """Canonical 2N-entry loop order: all ``.block`` first (by dim id),
    then all ``.tile`` (by dim id). Respects the invariant that
    ``{d}.block`` precedes ``{d}.tile`` for every dim."""
    ordered = sorted(dimensions, key=lambda d: int(d[1:]))
    return [f"{d}.block" for d in ordered] + [f"{d}.tile" for d in ordered]


def _default_buffer_scopes(
    physical_buffers: dict[str, PhysicalBuffer], ops: list[Op]
) -> dict[str, dict[str, DimScope]]:
    """Per-dim default scopes for every non-HBM buffer.

    * Plain buffers: ``PER_BLOCK`` on every dim.
    * Matmul SBUF outputs: reducing dim omitted (codegen rule pins it
      to FULL). Non-reducing output dims default to ``FULL`` too —
      the simplest always-valid setting; a tuner can shrink to
      PER_BLOCK whenever the non-reducing dim sits before the reducing
      dim in loop_order.
    * Matmul PSUM siblings: every matmul op axis (K, M, N) listed.
      Reducing dim affects emission depth; non-reducing dims shape
      the storage.
    """
    reducing_by_sbuf: dict[str, set[str]] = {}
    reducing_by_psum: dict[str, set[str]] = {}
    for op in ops:
        if op.kind in _MATMUL_KINDS and op.outputs:
            sbuf_out = op.outputs[0]
            reducing_by_sbuf[sbuf_out] = set(op.blocking_dims)
            psum_name = "psum_" + sbuf_out[len("sbuf_") :]
            reducing_by_psum[psum_name] = set(op.blocking_dims)
    result: dict[str, dict[str, DimScope]] = {}
    for name, buf in physical_buffers.items():
        if buf.loc == "hbm":
            continue
        if name in reducing_by_sbuf:
            dims = [d for d in buf.dim_ids if d not in reducing_by_sbuf[name]]
            result[name] = {d: DimScope.FULL for d in dims}
        elif name in reducing_by_psum:
            dims = list(reducing_by_psum[name]) + list(buf.dim_ids)
            result[name] = {d: DimScope.PER_BLOCK for d in dims}
        else:
            dims = list(buf.dim_ids)
            result[name] = {d: DimScope.PER_BLOCK for d in dims}
    return result


def _resolve_dimensions(
    parsed: list[tuple[type[NKIOp], dict[str, str], dict, list[str]]],
    per_op_axis_maps: list[dict[str, str]],
    per_op_blocking: list[set[str]],
    dim_sizes: dict[str, int],
) -> dict[str, DimInfo]:
    """Compute ``DimInfo`` per dim: tile size = min of per-op limits; role from blocking."""
    per_dim_tile: dict[str, int] = {}
    for (op_cls, _, _, _), axis_map in zip(parsed, per_op_axis_maps):
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
    for _blocking_set in per_op_blocking:
        for d in _blocking_set:
            dim_role[d] = DimRole.ACCUMULATION

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
    parsed: list[tuple[type[NKIOp], dict[str, str], dict, list[str]]],
    param_names: list[str],
    return_name: str,
    tensors: dict[str, _Tensor],
) -> set[str]:
    """Names referenced as inputs anywhere + params + return."""
    used: set[str] = {return_name, *param_names}
    for _op_cls, name_kwargs, _op_kwargs, _output_names in parsed:
        for v in name_kwargs.values():
            if v in tensors:
                used.add(v)
    return used


def _build_physical_buffers(
    tensors: dict[str, _Tensor], used: set[str], dimensions: dict[str, DimInfo], return_name: str
) -> dict[str, PhysicalBuffer]:
    """Build ``sbuf_<name>`` per used tensor + ``hbm_<return>`` store dest."""
    result: dict[str, PhysicalBuffer] = {}
    for name, t in tensors.items():
        if name not in used or not t.dim_ids:
            continue
        p_axis = t.dim_ids[0]
        f_axis = t.dim_ids[1] if len(t.dim_ids) >= 2 else None
        p_tile = dimensions[p_axis].physical_tile_size
        f_tile = dimensions[f_axis].physical_tile_size if f_axis is not None else 1
        result[f"sbuf_{name}"] = PhysicalBuffer(
            tile=(p_tile, f_tile), dim_ids=tuple(t.dim_ids), dtype=t.dtype, loc="sbuf"
        )
    ret_t = tensors[return_name]
    ret_p = ret_t.dim_ids[0]
    ret_f = ret_t.dim_ids[1] if len(ret_t.dim_ids) >= 2 else None
    hbm_p = dimensions[ret_p].dim_size
    hbm_f = dimensions[ret_f].dim_size if ret_f is not None else 1
    result[f"hbm_{return_name}"] = PhysicalBuffer(
        tile=(hbm_p, hbm_f), dim_ids=tuple(ret_t.dim_ids), dtype=ret_t.dtype, loc="hbm"
    )
    return result


def _mint_psum_buffers(
    ops: list[Op], physical_buffers: dict[str, PhysicalBuffer], dimensions: dict[str, DimInfo]
) -> None:
    """For every matmul-style op, add ``psum_<stem>`` to ``physical_buffers``.

    The PSUM entry's storage ``dim_ids`` is ``(M, N)`` — the reducing
    K-axis is consumed in-place by ``nc_matmul``'s HW accumulator and
    does not contribute to the storage shape. The reducing axis is
    still tracked in ``buffer_scopes`` for placement purposes (emission
    depth is capped by K.tile).
    """
    for op in ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        sbuf_out = op.outputs[0]
        if not sbuf_out.startswith("sbuf_"):
            raise ValueError(f"matmul op {op.kind} expected sbuf_-prefixed output, got {sbuf_out!r}")
        psum_name = "psum_" + sbuf_out[len("sbuf_") :]
        if psum_name in physical_buffers:
            continue
        m_dim = op.axis_map["M"]
        n_dim = op.axis_map["N"]
        m_tile = dimensions[m_dim].physical_tile_size
        n_tile = dimensions[n_dim].physical_tile_size
        physical_buffers[psum_name] = PhysicalBuffer(
            tile=(m_tile, n_tile), dim_ids=(m_dim, n_dim), dtype="float32", loc="psum"
        )


def _build_ops(
    parsed: list[tuple[type[NKIOp], dict[str, str], dict, list[str]]],
    per_op_axis_maps: list[dict[str, str]],
    per_op_blocking: list[set[str]],
    tensors: dict[str, _Tensor],
    used: set[str],
    param_names: list[str],
    return_name: str,
) -> list[Op]:
    """Assemble the final ops list: NKILoad header + math ops + NKIStore tail."""
    ops: list[Op] = []
    for p in param_names:
        ops.append(Op(kind=NKILoad.__name__, inputs={"data": p}, outputs=[f"sbuf_{p}"]))

    for (op_cls, name_kwargs, op_kwargs, output_names), axis_map, blocking in zip(
        parsed, per_op_axis_maps, per_op_blocking
    ):
        inputs: dict[str, str] = {
            role: f"sbuf_{name_kwargs[role]}"
            for role in op_cls.OPERAND_AXES
            if role in name_kwargs and name_kwargs[role] in tensors
        }
        kept_outputs = [o for o in output_names if o in used]
        sbuf_outputs = [f"sbuf_{o}" for o in kept_outputs]
        ops.append(
            Op(
                kind=op_cls.__name__,
                inputs=inputs,
                outputs=sbuf_outputs,
                axis_map=dict(axis_map),
                blocking_dims=set(blocking),
                kwargs=dict(op_kwargs),
            )
        )

    ops.append(Op(kind=NKIStore.__name__, inputs={"data": f"sbuf_{return_name}"}, outputs=[f"hbm_{return_name}"]))
    return ops


def _derive_edges(ops: list[Op]) -> list[tuple[int, int]]:
    """Return ``[(producer_idx, consumer_idx), ...]``, one edge per producer→consumer link."""
    producer_of: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        for tname in op.inputs.values():
            src = producer_of.get(tname)
            if src is not None and src != i:
                edges.append((src, i))
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
    default_dtype = (
        tensors[operand_map[first_slot]].dtype
        if has_first
        else next(t.dtype for t in tensors.values() if t.dtype != "")
    )
    for oname, (out_slot, output_axes) in zip(output_names, op_cls.OUTPUT_AXES.items()):
        dim_ids = [local[a] for a in output_axes if a in local]
        shape = tuple(dim_sizes[d] for d in dim_ids)
        dtype = op_cls.OUTPUT_DTYPES.get(out_slot, default_dtype)
        tensors[oname] = _Tensor(oname, shape, dtype, dim_ids)
