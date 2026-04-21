"""Dimension analysis: ID assignment, tile sizes, role classification.

Forward pass over all ops produces three results per dimension:
  1. A concrete dimension ID (d0, d1, ...)
  2. A tile size (max of hardware tile limits across ops)
  3. A ``DimRole`` classifying the loop-iteration dependency
     structure, seeded from ``BLOCKING_AXES`` of ops touching the
     dim. The online-fusion rewrite may later promote a ``SERIAL``
     dim to ``ACCUMULATION``.

Usage::

    da = analyze_dims(matmul_nkigym, {"lhs_T": ((8192, 8192), "bfloat16"),
                                       "rhs": ((8192, 8192), "bfloat16")})
    print(da)
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from nkigym.kernel_ir.parse import find_ops
from nkigym.ops.base import NKIOp


class DimRole(Enum):
    """Loop-iteration dependency structure of a dimension.

    ``PARALLEL``: iterations are independent — they read/write
    disjoint buffer slices and can run in any order or
    concurrently. Multi-buffering and sharding are legal.

    ``SERIAL``: iterations contribute to the same buffer via a
    non-additive combinator (max, min, argmax, etc.). Associative
    but not linear, so consumers of the buffer must wait for the
    full loop to finish (blocking barrier). Multi-buffering and
    sharding are illegal.

    ``ACCUMULATION``: iterations contribute to the same buffer via
    addition (associative AND linear). Linearity lets online fusion
    inject a per-iteration scale coefficient so consumers can read
    the partial accumulator before the loop finishes — the blocking
    barrier is lifted. Still serial in dependency; multi-buffering
    and sharding remain illegal.

    State transitions: only ``SERIAL → ACCUMULATION`` is legal,
    triggered by the online-fusion rewrite after verifying the
    downstream consumer is add-accumulation-shaped. ``PARALLEL``
    never participates.
    """

    PARALLEL = "parallel"
    SERIAL = "serial"
    ACCUMULATION = "accumulation"


@dataclass
class DimInfo:
    """Per-dimension analysis result.

    Attributes:
        dim_size: Total number of elements along this dimension.
        logical_tile_size: Iteration granularity. max(all op tile
            limits) on this dimension, clamped to dim_size.
        physical_tile_size: Buffer allocation granularity. min(all
            op tile limits) on this dimension, with partition cap
            (128) included when the dimension appears as the first
            axis of any tensor.
        role: ``DimRole`` for this dimension's loop-iteration
            dependency structure. Seeded from ``BLOCKING_AXES``
            (``SERIAL`` when any op blocks on the dim, else
            ``PARALLEL``) and possibly promoted to ``ACCUMULATION``
            by the online-fusion rewrite.
    """

    dim_size: int
    logical_tile_size: int
    physical_tile_size: int
    role: DimRole

    @property
    def blocks_consumers(self) -> bool:
        """True iff downstream consumers must wait for this dim's loop to finish.

        Equivalent to ``role is SERIAL``. ``ACCUMULATION`` doesn't
        block consumers because σ-correction lets them read the
        running buffer mid-loop.
        """
        return self.role is DimRole.SERIAL

    @property
    def is_sequential(self) -> bool:
        """True iff iterations share buffer state (``SERIAL`` or ``ACCUMULATION``).

        Gates multi-buffering and sharding: both are legal only on
        ``PARALLEL`` dims.
        """
        return self.role is not DimRole.PARALLEL

    @property
    def num_ptiles(self) -> int:
        """Physical tiles per logical tile: ``logical_tile_size // physical_tile_size``."""
        return self.logical_tile_size // self.physical_tile_size


@dataclass
class TensorInfo:
    """Per-tensor analysis result.

    Attributes:
        dim_ids: Concrete dimension IDs (e.g. ``("d0", "d1")``).
        shape: Full shape (e.g. ``(2048, 128)``).
        dtype: Dtype string (e.g. ``"bfloat16"``).
    """

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str


@dataclass
class DimAnalysis:
    """Complete dimension analysis result.

    Authority on per-op blocking semantics. ``NKIOp.BLOCKING_AXES``
    is the per-class declaration (abstract axes); this class carries
    the concrete, per-op, per-kernel set after any IR-level
    transforms (e.g. online fusion) have had a chance to clear
    blocking dims the rewrite has broken. Downstream callers
    (partition sampler, emission placement, loop-fusion legality,
    renderer) must read from ``op_blocking_dims(op_idx)`` rather
    than the class attribute so there is a single source of truth.

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
        per_op_blocking_dims: Per-op concrete blocking-dim sets
            (by op index). Seeded from ``NKIOp.BLOCKING_AXES``
            resolved through ``per_op_axis_maps``; mutated by
            online-fusion rewrites that remove dims from the
            producer's blocking set.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dims: dict[str, DimInfo]
    tensors: dict[str, TensorInfo]
    per_op_axis_maps: list[dict[str, str]]
    op_tile_sizes: list[dict[str, int]]
    per_op_blocking_dims: list[set[str]]

    def op_blocking_dims(self, op_idx: int) -> set[str]:
        """Return the concrete dims ``op_idx`` blocks on in this kernel.

        This is the sole source of truth for per-op blocking in
        downstream code. Returns a fresh set copy so callers can
        freely intersect/mutate without affecting the record.
        """
        return set(self.per_op_blocking_dims[op_idx])

    def __repr__(self) -> str:
        """Show final dimension analysis result."""
        lines: list[str] = []
        lines.append(f"DimAnalysis({self.func_name})")
        lines.append(f"  params: {self.param_names} -> {self.return_name}")

        lines.append("")
        lines.append("  Dimensions:")
        headers = ["dim", "size", "logical", "physical", "phys/logical", "type"]
        rows: list[list[str]] = []
        for dim_id, di in self.dims.items():
            num_physical = di.num_ptiles
            rows.append(
                [
                    dim_id,
                    str(di.dim_size),
                    str(di.logical_tile_size),
                    str(di.physical_tile_size),
                    str(num_physical),
                    di.role.value,
                ]
            )
        col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        header_line = "  | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        lines.append(f"    {header_line}")
        lines.append(f"    {sep_line}")
        for row in rows:
            lines.append("    " + "  | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))

        lines.append("")
        lines.append("  Logical Tensors:")
        t_headers = ["name", "dims", "shape", "dtype"]
        t_rows: list[list[str]] = []
        for name, t in self.tensors.items():
            t_rows.append([name, f"({', '.join(t.dim_ids)})", f"({', '.join(str(s) for s in t.shape)})", t.dtype])
        t_widths = [max(len(h), *(len(r[i]) for r in t_rows)) for i, h in enumerate(t_headers)]
        lines.append("    " + "  | ".join(h.ljust(t_widths[i]) for i, h in enumerate(t_headers)))
        lines.append("    " + "-+-".join("-" * t_widths[i] for i in range(len(t_headers))))
        for row in t_rows:
            lines.append("    " + "  | ".join(row[i].ljust(t_widths[i]) for i in range(len(t_headers))))

        return "\n".join(lines)


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
        tensors[oname] = _Tensor(oname, shape, dtype, dim_ids)


def op_blocking_dims(op_cls: type[NKIOp], axis_map: dict[str, str]) -> set[str]:
    """Seed the concrete blocking-dim set for an op from its class declaration.

    Reads ``op_cls.BLOCKING_AXES`` and resolves each abstract axis
    through ``axis_map``; axes not in the map (tensor collapsed
    before the op) are silently skipped. This is the default
    per-class seed — the authoritative per-op set lives on
    ``DimAnalysis.per_op_blocking_dims`` and may diverge once
    online-fusion rewrites have cleared dims from specific ops.
    Downstream call sites must use ``DimAnalysis.op_blocking_dims``
    rather than this helper.
    """
    return {axis_map[axis] for axis in op_cls.BLOCKING_AXES if axis in axis_map}


def _compute_tile_sizes(
    ops: list[tuple[type[NKIOp], dict[str, str], list[str], dict[str, str]]],
    per_op_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    blocking_dims: set[str],
    tensors: dict[str, "_Tensor"],
) -> tuple[dict[str, DimInfo], list[dict[str, int]]]:
    """Compute per-dimension tile sizes, classification, and per-op tile sizes.

    Applies the partition axis constraint: if a dimension appears
    as the first (partition) axis of any tensor, its tile_size is
    capped at 128.
    """
    PARTITION_MAX = 128

    partition_dims: set[str] = set()
    for t in tensors.values():
        if t.dim_ids:
            partition_dims.add(t.dim_ids[0])

    max_tile: dict[str, int] = {}
    min_tile: dict[str, int] = {}
    op_tile_sizes: list[dict[str, int]] = []

    for (op_cls, _, _, _), local in zip(ops, per_op_maps):
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

    for dim_id in max_tile:
        max_tile[dim_id] = min(max_tile[dim_id], dim_sizes[dim_id])
        if dim_id in partition_dims:
            min_tile[dim_id] = min(min_tile[dim_id], PARTITION_MAX)

    dims = {
        d: DimInfo(dim_sizes[d], max_tile[d], min_tile[d], DimRole.SERIAL if d in blocking_dims else DimRole.PARALLEL)
        for d in max_tile
    }
    return dims, op_tile_sizes


def analyze_dims(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> DimAnalysis:
    """Run dimension analysis.

    Produces three results per dimension:
      1. Concrete dimension ID (d0, d1, ...)
      2. Tile size (max of hardware tile limits across ops)
      3. Blocking vs non-blocking classification

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
        tensors[name] = _Tensor(name, shape, dtype, dim_ids=[])

    dim_counter = [0]
    per_op_maps: list[dict[str, str]] = []
    dim_sizes: dict[str, int] = {}

    for op_cls, name_kwargs, output_names, _all_kwargs in ops:
        operand_map = {k: v for k, v in name_kwargs.items() if v in tensors}
        local = _build_axis_map(op_cls, operand_map, tensors, dim_counter, per_op_maps, dim_sizes)
        per_op_maps.append(local)
        _create_outputs(op_cls, operand_map, output_names, local, tensors, dim_sizes)

    if return_name not in tensors:
        raise ValueError(f"Return tensor {return_name!r} not found")
    per_op_blocking: list[set[str]] = [
        op_blocking_dims(op_cls, local) for (op_cls, _, _, _), local in zip(ops, per_op_maps)
    ]
    blocking_dims: set[str] = set().union(*per_op_blocking) if per_op_blocking else set()

    dims, op_tile_sizes = _compute_tile_sizes(ops, per_op_maps, dim_sizes, blocking_dims, tensors)

    tensor_infos: dict[str, TensorInfo] = {}
    for name, t in tensors.items():
        tensor_infos[name] = TensorInfo(tuple(t.dim_ids), t.shape, t.dtype)

    return DimAnalysis(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dims=dims,
        tensors=tensor_infos,
        per_op_axis_maps=per_op_maps,
        op_tile_sizes=op_tile_sizes,
        per_op_blocking_dims=per_op_blocking,
    )
