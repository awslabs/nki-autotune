"""Dim unification for an ``f_nkigym`` callable via symbolic tracing.

Entry point: :func:`analyze_dimensions`. Returns a private
:class:`_AnalysisResult` consumed by :func:`build_initial_ir`.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import prod
from threading import RLock
from typing import Any

from nkigym.ops.base import NKIOp, OperationSSAName, collect_operation_ssa_names

_DIMENSION_TRACE_LOCK = RLock()


@dataclass
class TensorDims:
    """Per-tensor dim-unification result.

    Attributes:
        name: Source-level variable name.
        shape: Per-dim extents, aligned with ``dim_ids``.
        dim_ids: Concrete dim names (``d0``, ``d1`` ...).
        location: ``"shared_hbm"`` / ``"sbuf"`` / ``"psum"``. Params resolve
            to ``"shared_hbm"`` (the role lattice forbids any other location);
            for intermediates it is synthesized from the producing op's
            ``OUTPUT_LOCATION``.
        dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"``. For
            intermediates this propagates from the first input's logical
            dtype through the trace; for params it is seeded directly from
            the ``input_specs`` ``(shape, dtype)`` entry.
        storage_dtype: Optional physical allocation dtype override supplied by
            the producing op. ``None`` means the logical ``dtype``.
    """

    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]
    location: str
    dtype: str
    storage_dtype: str | None


@dataclass
class _OpRecord:
    """Per-op tracer record consumed by canonical tree construction.

    Attributes:
        op_cls: The NKIOp subclass.
        operand_names: ``slot → tensor_name`` for every operand in the call.
        axis_map: ``abstract_axis → concrete_dim``.
        kwargs: Configuration kwargs and literal operand slots. Tensor operands
            live in ``operand_names``; a scalar ``operand0`` remains here.
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    axis_map: dict[str, str]
    kwargs: dict[str, Any]


@dataclass
class _AnalysisResult:
    """Private hand-off from ``analyze_dimensions`` to ``build_initial_ir``.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_names: Identifiers in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent``.
        tensors: All named tensors, keyed by name.
        ops: Compute ops in source order.
    """

    func_name: str
    param_names: list[str]
    return_names: tuple[str, ...]
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    ops: list[_OpRecord]


def analyze_dimensions(
    func: Callable[..., Any], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> _AnalysisResult:
    """Trace ``func`` against sentinel inputs and run cross-op dim unification.

    Args:
        func: An ``@nkigym_kernel``-decorated callable to analyse.
        input_specs: ``{param_name: (shape, dtype)}`` for every positional parameter.
    """
    unwrapped = inspect.unwrap(func)
    param_names = list(inspect.signature(unwrapped).parameters)
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    state = _TraceState(ssa_names=collect_operation_ssa_names(unwrapped))
    for name in param_names:
        shape, dtype = input_specs[name]
        sym = _Sym(tuple(shape), name)
        sym.location = "shared_hbm"
        sym.dtype = dtype
        state.sentinels[name] = sym

    _run_trace(unwrapped, [state.sentinels[n] for n in param_names], state)
    _canonicalize_dim_names(state)

    tensors: dict[str, TensorDims] = {}
    for sym in state.sentinels.values():
        if any(d is None for d in sym.dim_ids):
            raise ValueError(f"Tensor {sym.source_name!r} has un-unified dims: {sym.dim_ids}")
        if sym.location is None:
            raise ValueError(f"Tensor {sym.source_name!r} has no location")
        if sym.dtype is None:
            raise ValueError(f"Tensor {sym.source_name!r} has no dtype")
        tensors[sym.source_name] = TensorDims(
            name=sym.source_name,
            shape=sym.shape,
            dim_ids=tuple(d for d in sym.dim_ids if d is not None),
            location=sym.location,
            dtype=sym.dtype,
            storage_dtype=sym.storage_dtype,
        )
    return _AnalysisResult(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_names=_parse_return_names(unwrapped),
        dim_sizes=state.dim_sizes,
        tensors=tensors,
        ops=state.op_records,
    )


class _Sym:
    """Symbolic tensor: shape + mutable ``dim_ids`` + source name + alloc kwargs."""

    __slots__ = ("shape", "dim_ids", "factor_dim_ids", "source_name", "location", "dtype", "storage_dtype")

    def __init__(self, shape: tuple[int, ...], source_name: str) -> None:
        self.shape: tuple[int, ...] = shape
        self.dim_ids: list[str | None] = [None] * len(shape)
        self.factor_dim_ids: list[tuple[str, ...] | None] = [None] * len(shape)
        self.source_name: str = source_name
        self.location: str | None = None
        self.dtype: str | None = None
        self.storage_dtype: str | None = None


class _TraceState:
    """Mutable state threaded through the hook during tracing."""

    def __init__(self, ssa_names: Iterator[OperationSSAName]) -> None:
        self.sentinels: dict[str, _Sym] = {}
        self.dim_sizes: dict[str, int] = {}
        self.op_records: list[_OpRecord] = []
        self.ssa_names = ssa_names
        self.next_dim = 0

    def fresh_dim(self, size: int) -> str:
        """Allocate a fresh ``d<N>`` dim id of ``size``.

        Monotonic because ``_unify`` pops retired ids mid-trace;
        reusing ``len(dim_sizes)`` would collide with a live id.
        """
        name = f"d{self.next_dim}"
        self.next_dim += 1
        self.dim_sizes[name] = size
        return name


def _run_trace(func: Callable[..., Any], args: list[_Sym], state: _TraceState) -> None:
    """Invoke ``func(*args)`` with :meth:`NKIOp.__call__` hooked for analysis."""
    with _DIMENSION_TRACE_LOCK:
        original = NKIOp.__call__
        NKIOp.__call__ = _make_hook(state)
        try:
            func(*args)
        finally:
            NKIOp.__call__ = original


def _make_hook(state: _TraceState) -> Callable[..., Any]:
    """Build a replacement for :meth:`NKIOp.__call__` that records into ``state`` and synthesizes outputs."""

    def hook(op: NKIOp, **kwargs: Any) -> Any:
        merged = {**getattr(op, "_init_kwargs", {}), **kwargs}
        cls = type(op)
        input_syms, record = _trace_compute_op(cls, merged, state)
        name = next(state.ssa_names)
        returned = cls.RETURN_RMW_OPERAND
        if returned is not None:
            result = merged.get(returned)
            if not isinstance(result, _Sym) or returned not in cls.rmw_operands(record.kwargs):
                raise ValueError(f"{cls.__name__}.{returned} must bind one RMW tensor")
        else:
            result = _synthesize_outputs(cls, name, input_syms, record, state)
        return result

    return hook


def _trace_compute_op(cls: type[NKIOp], kwargs: dict[str, Any], state: _TraceState) -> tuple[list["_Sym"], "_OpRecord"]:
    """Unify a compute op's operands and record an :class:`_OpRecord` entry.

    Returns the ordered input operand syms (``OPERAND_AXES`` order filtered
    to ``INPUT_OPERANDS``) so the caller can synthesize the op's output(s),
    plus the freshly-appended :class:`_OpRecord` so the caller can write the
    synthesized output slot names back into ``record.operand_names``.
    """
    local = {
        abstract: state.fresh_dim(int(kwargs[size]) if isinstance(size, str) else size)
        for abstract, size in cls.FIXED_AXIS_SIZES.items()
    }
    operand_names: dict[str, str] = {}
    for slot in cls.OPERAND_AXES:
        sym = kwargs.get(slot)
        if not isinstance(sym, _Sym):
            continue
        required_dtype = cls.REQUIRED_INPUT_STORAGE_DTYPES.get(slot)
        if required_dtype is not None:
            if sym.storage_dtype is not None and sym.storage_dtype != required_dtype:
                raise ValueError(
                    f"{cls.__name__}.{slot} requires {required_dtype}, but "
                    f"{sym.source_name} already requires {sym.storage_dtype}"
                )
            sym.storage_dtype = required_dtype
        operand_names[slot] = sym.source_name
        groups = cls.operand_axis_groups(slot)
        if len(sym.shape) > len(groups):
            raise ValueError(f"{cls.__name__}.{slot} has shape {sym.shape}, but only {len(groups)} dimensions")
        for i, group in enumerate(groups[: len(sym.shape)]):
            if len(group) > 1:
                if sym.dim_ids[i] is None:
                    sym.dim_ids[i] = state.fresh_dim(sym.shape[i])
                sizes = {axis: state.dim_sizes[local[axis]] for axis in group if axis in local}
                for axis, size in cls.infer_axis_group(slot, i, sym.shape[i], sizes).items():
                    local[axis] = state.fresh_dim(size)
                factors = sym.factor_dim_ids[i]
                if factors is None:
                    sym.factor_dim_ids[i] = tuple(local[axis] for axis in group)
                else:
                    existing_intervals = _factor_intervals(factors, state)
                    current = tuple(local[axis] for axis in group)
                    current_intervals = _factor_intervals(current, state)
                    for interval in existing_intervals.keys() & current_intervals.keys():
                        if existing_intervals[interval] != current_intervals[interval]:
                            _unify(existing_intervals[interval], current_intervals[interval], state, local)
                    if len(current) > len(factors):
                        sym.factor_dim_ids[i] = current
                continue
            if not group:
                raise ValueError(f"{cls.__name__}.{slot} has an empty axis group")
            abstract = group[0]
            existing = sym.dim_ids[i]
            if existing is None:
                if abstract not in local:
                    local[abstract] = state.fresh_dim(sym.shape[i])
                sym.dim_ids[i] = local[abstract]
            elif abstract in local and local[abstract] != existing:
                _unify(existing, local[abstract], state, local)
            else:
                local[abstract] = existing
    op_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, _Sym)}
    record = _OpRecord(op_cls=cls, operand_names=operand_names, axis_map=local, kwargs=op_kwargs)
    state.op_records.append(record)
    input_syms = [
        kwargs[slot] for slot in cls.OPERAND_AXES if slot in cls.INPUT_OPERANDS and isinstance(kwargs.get(slot), _Sym)
    ]
    return input_syms, record


def _synthesize_outputs(
    cls: type[NKIOp], name: OperationSSAName, input_syms: list["_Sym"], record: "_OpRecord", state: _TraceState
) -> "_Sym | tuple[_Sym, ...]":
    """Create the output sentinel(s) for an op call; return the primary (assigned) one.

    Output slots = ``OPERAND_AXES`` keys not in ``INPUT_OPERANDS``. The
    primary slot (gets ``name``, returned to thread the SSA chain) is
    ``reduce_res`` if declared, else ``dst``. Any secondary output slot
    (e.g. activation_reduce's scratch ``dst``) gets ``f"{name}_scratch"``.

    Output dims come from the op's already-unified ``record.axis_map``,
    filtered to the declared output axes actually present in that map: an
    output axis the op's inputs never bound (e.g. ``F`` when ``data`` is a
    1D ``(P,)`` reduce result) is not part of this instance's output, so a
    ``(P,)`` input yields a ``(P,)`` output rather than ``(P, F)``. Dtype
    propagates from the first input's logical dtype; location is
    ``cls.OUTPUT_LOCATION``.

    Each synthesized slot's tensor name is also written back into
    ``record.operand_names`` so every op's record carries its output
    slot(s); ``canonical_build`` reads these to emit write regions and form
    the producer-consumer dependency chain.
    """
    output_slots = [slot for slot in cls.OPERAND_AXES if slot not in cls.INPUT_OPERANDS]
    primary_slot = "reduce_res" if "reduce_res" in cls.OPERAND_AXES else "dst"
    default_dtype = cls.OUTPUT_DTYPE or (input_syms[0].dtype if input_syms else None)
    multiple = isinstance(name, tuple)
    if isinstance(name, str):
        names = tuple(name if slot == primary_slot else f"{name}_scratch" for slot in output_slots)
    else:
        names = name
    if len(names) != len(output_slots):
        raise ValueError(f"{cls.__name__}: expected {len(output_slots)} output names, got {len(names)}")
    primary_sym: _Sym | None = None
    output_syms: list[_Sym] = []
    for slot, slot_name in zip(output_slots, names, strict=True):
        groups = tuple(group for group in cls.operand_axis_groups(slot) if all(a in record.axis_map for a in group))
        shape = tuple(prod(state.dim_sizes[record.axis_map[axis]] for axis in group) for group in groups)
        dim_ids = [
            record.axis_map[group[0]] if len(group) == 1 else state.fresh_dim(extent)
            for group, extent in zip(groups, shape, strict=True)
        ]
        sym = _Sym(shape, slot_name)
        sym.dim_ids = list(dim_ids)
        sym.factor_dim_ids = [
            tuple(record.axis_map[axis] for axis in group) if len(group) > 1 else None for group in groups
        ]
        sym.location = cls.OUTPUT_LOCATION
        sym.dtype = cls.OUTPUT_DTYPES.get(slot, default_dtype)
        sym.storage_dtype = cls.OUTPUT_STORAGE_DTYPES.get(slot, cls.OUTPUT_STORAGE_DTYPE)
        state.sentinels[slot_name] = sym
        record.operand_names[slot] = slot_name
        output_syms.append(sym)
        if slot == primary_slot:
            primary_sym = sym
    if primary_sym is None:
        raise ValueError(f"{cls.__name__}: no primary output slot {primary_slot!r} in OPERAND_AXES")
    return tuple(output_syms) if multiple else primary_sym


def _unify(old: str, new: str, state: _TraceState, local: dict[str, str]) -> None:
    """Rename ``old`` dim id to ``new`` across sentinels, op records, and ``local``."""
    old_size = state.dim_sizes.get(old)
    new_size = state.dim_sizes.get(new)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(f"Cannot unify {old} (size {old_size}) with {new} (size {new_size})")
    if old in state.dim_sizes:
        state.dim_sizes.setdefault(new, state.dim_sizes.pop(old))
    _apply_rename(state, {old: new})
    for abstract in local:
        if local[abstract] == old:
            local[abstract] = new


def _factor_intervals(dimensions: tuple[str, ...], state: _TraceState) -> dict[tuple[int, int], str]:
    """Map row-major factor intervals to dimension IDs."""
    result: dict[tuple[int, int], str] = {}
    offset = 1
    for dimension in dimensions:
        end = offset * state.dim_sizes[dimension]
        result[(offset, end)] = dimension
        offset = end
    return result


def _canonicalize_dim_names(state: _TraceState) -> None:
    """Relabel surviving dims to a contiguous ``d0..dN`` sequence.

    Unification can retire intermediate ids (e.g. ``d2`` merged into
    ``d1``), leaving gaps. Rename in discovery order of the sentinels so
    the public surface stays dense.
    """
    order: list[str] = []
    seen: set[str] = set()
    for sym in state.sentinels.values():
        for d in sym.dim_ids:
            if d is not None and d not in seen:
                seen.add(d)
                order.append(d)
    for rec in state.op_records:
        for dimension in rec.axis_map.values():
            if dimension not in seen:
                seen.add(dimension)
                order.append(dimension)
    remap = {old: f"d{i}" for i, old in enumerate(order)}
    if all(old == new for old, new in remap.items()):
        return
    state.dim_sizes = {remap[old]: size for old, size in state.dim_sizes.items() if old in remap}
    _apply_rename(state, remap)


def _apply_rename(state: _TraceState, remap: dict[str, str]) -> None:
    """Substitute every dim id in sentinels and op records via ``remap``."""
    for sym in state.sentinels.values():
        sym.dim_ids = [remap.get(d, d) if d is not None else None for d in sym.dim_ids]
        sym.factor_dim_ids = [
            None if factors is None else tuple(remap.get(dimension, dimension) for dimension in factors)
            for factors in sym.factor_dim_ids
        ]
    for rec in state.op_records:
        for abstract in rec.axis_map:
            rec.axis_map[abstract] = remap.get(rec.axis_map[abstract], rec.axis_map[abstract])


def _parse_return_names(func: Callable[..., Any]) -> tuple[str, ...]:
    """Return identifiers named in the kernel's single top-level ``return`` statement.

    Raises ``ValueError`` if the function has no ``return`` statement, has
    more than one, or returns something other than names or a tuple of names.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    returns = [stmt for stmt in func_def.body if isinstance(stmt, ast.Return)]
    if len(returns) == 0:
        raise ValueError(f"{func.__name__}: no top-level return statement")
    if len(returns) > 1:
        raise ValueError(f"{func.__name__}: expected a single top-level return, found {len(returns)}")
    value = returns[0].value
    if isinstance(value, ast.Name):
        names = (value.id,)
    elif isinstance(value, ast.Tuple) and value.elts and all(isinstance(item, ast.Name) for item in value.elts):
        names = tuple(item.id for item in value.elts if isinstance(item, ast.Name))
    else:
        raise ValueError(f"{func.__name__}: return value must be a Name or tuple of Names")
    return names
