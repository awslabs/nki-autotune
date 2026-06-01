"""Dim unification for an ``f_nkigym`` callable via symbolic tracing.

Entry point: :func:`analyze_dimensions`. Returns a private
:class:`_AnalysisResult` consumed by :func:`build_initial_ir`.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from nkigym.ops.base import NKIOp


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
    """

    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]
    location: str
    dtype: str


@dataclass
class _OpRecord:
    """Per-op tracer record. Private — consumed by ``build_initial_tree``.

    Attributes:
        op_cls: The NKIOp subclass.
        operand_names: ``slot → tensor_name`` for every operand in the call.
        axis_map: ``abstract_axis → concrete_dim``.
        kwargs: Non-operand call kwargs (e.g. ``{"value": 0.0}`` for the
            synthesized memset, ``{"op": "exp"}`` for activation).
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
        return_name: Identifier in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent``.
        tensors: All named tensors, keyed by name.
        ops: Compute ops in source order.
    """

    func_name: str
    param_names: list[str]
    return_name: str
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

    state = _TraceState(ssa_names=_collect_ssa_names(unwrapped))
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
        )
    return _AnalysisResult(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=_parse_return_name(unwrapped),
        dim_sizes=state.dim_sizes,
        tensors=tensors,
        ops=state.op_records,
    )


class _Sym:
    """Symbolic tensor: shape + mutable ``dim_ids`` + source name + alloc kwargs."""

    __slots__ = ("shape", "dim_ids", "source_name", "location", "dtype")

    def __init__(self, shape: tuple[int, ...], source_name: str) -> None:
        self.shape: tuple[int, ...] = shape
        self.dim_ids: list[str | None] = [None] * len(shape)
        self.source_name: str = source_name
        self.location: str | None = None
        self.dtype: str | None = None


class _TraceState:
    """Mutable state threaded through the hook during tracing."""

    def __init__(self, ssa_names: Iterator[str]) -> None:
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
        return _synthesize_outputs(cls, name, input_syms, record, state)

    return hook


def _trace_compute_op(cls: type[NKIOp], kwargs: dict[str, Any], state: _TraceState) -> tuple[list["_Sym"], "_OpRecord"]:
    """Unify a compute op's operands and record an :class:`_OpRecord` entry.

    Returns the ordered input operand syms (``OPERAND_AXES`` order filtered
    to ``INPUT_OPERANDS``) so the caller can synthesize the op's output(s),
    plus the freshly-appended :class:`_OpRecord` so the caller can write the
    synthesized output slot names back into ``record.operand_names``.
    """
    local: dict[str, str] = {}
    operand_names: dict[str, str] = {}
    for slot, axes in cls.OPERAND_AXES.items():
        sym = kwargs.get(slot)
        if not isinstance(sym, _Sym):
            continue
        operand_names[slot] = sym.source_name
        for i, abstract in enumerate(axes[: len(sym.shape)]):
            existing = sym.dim_ids[i]
            if existing is None:
                if abstract not in local:
                    local[abstract] = state.fresh_dim(sym.shape[i])
                sym.dim_ids[i] = local[abstract]
            elif abstract in local and local[abstract] != existing:
                _unify(existing, local[abstract], state, local)
            else:
                local[abstract] = existing
    op_kwargs = {k: v for k, v in kwargs.items() if k not in cls.OPERAND_AXES}
    record = _OpRecord(op_cls=cls, operand_names=operand_names, axis_map=local, kwargs=op_kwargs)
    state.op_records.append(record)
    input_syms = [
        kwargs[slot] for slot in cls.OPERAND_AXES if slot in cls.INPUT_OPERANDS and isinstance(kwargs.get(slot), _Sym)
    ]
    return input_syms, record


def _synthesize_outputs(
    cls: type[NKIOp], name: str, input_syms: list["_Sym"], record: "_OpRecord", state: _TraceState
) -> "_Sym":
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
    logical_dtype = input_syms[0].dtype if input_syms else None
    primary_sym: _Sym | None = None
    for slot in output_slots:
        slot_name = name if slot == primary_slot else f"{name}_scratch"
        axes = cls.OPERAND_AXES[slot]
        dim_ids = [record.axis_map[a] for a in axes if a in record.axis_map]
        shape = tuple(state.dim_sizes[d] for d in dim_ids)
        sym = _Sym(shape, slot_name)
        sym.dim_ids = list(dim_ids)
        sym.location = cls.OUTPUT_LOCATION
        sym.dtype = logical_dtype
        state.sentinels[slot_name] = sym
        record.operand_names[slot] = slot_name
        if slot == primary_slot:
            primary_sym = sym
    if primary_sym is None:
        raise ValueError(f"{cls.__name__}: no primary output slot {primary_slot!r} in OPERAND_AXES")
    return primary_sym


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
    remap = {old: f"d{i}" for i, old in enumerate(order)}
    if all(old == new for old, new in remap.items()):
        return
    state.dim_sizes = {remap[old]: size for old, size in state.dim_sizes.items() if old in remap}
    _apply_rename(state, remap)


def _apply_rename(state: _TraceState, remap: dict[str, str]) -> None:
    """Substitute every dim id in sentinels and op records via ``remap``."""
    for sym in state.sentinels.values():
        sym.dim_ids = [remap.get(d, d) if d is not None else None for d in sym.dim_ids]
    for rec in state.op_records:
        for abstract in rec.axis_map:
            rec.axis_map[abstract] = remap.get(rec.axis_map[abstract], rec.axis_map[abstract])


def _collect_ssa_names(func: Callable[..., Any]) -> Iterator[str]:
    """Yield the LHS of every ``var = NKIOp()(...)`` assignment in source order.

    Each straight-line op call is assigned to a name; the trace consumes
    these in execution (= source) order to name synthesized outputs.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    for stmt in func_def.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and _is_op_call(stmt.value):
            yield target.id


def _is_op_call(node: ast.expr) -> bool:
    """Return True if ``node`` is an ``NKIXxx(...)(...)`` double-call (op invocation)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and isinstance(node.func.func, ast.Name)
        and node.func.func.id.startswith("NKI")
    )


def _parse_return_name(func: Callable[..., Any]) -> str:
    """Return the identifier named in the kernel's single top-level ``return`` statement.

    Raises ``ValueError`` if the function has no ``return`` statement, has
    more than one, or returns an expression that is not a single ``Name``.
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
    if not isinstance(value, ast.Name):
        raise ValueError(f"{func.__name__}: return value must be a bare Name, got {type(value).__name__}")
    return value.id
