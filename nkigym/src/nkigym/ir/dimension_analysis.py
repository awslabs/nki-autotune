"""Dim unification for an ``f_nkigym`` callable via symbolic tracing.

Entry point: :func:`analyze_dimensions`.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp


@dataclass
class TensorDims:
    """Per-tensor dim-unification result.

    Attributes:
        name: Source-level variable name.
        shape: Per-dim extents, aligned with ``dim_ids``.
        dim_ids: Concrete dim names (``d0``, ``d1`` ...).
    """

    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]


@dataclass
class OpAxes:
    """Per-op dim-unification result.

    Attributes:
        op_cls: The NKIOp subclass.
        operand_names: ``slot → tensor_name`` for every operand in the call.
        axis_map: ``abstract_axis → concrete_dim`` (e.g. ``{"K": "d0"}``).
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    axis_map: dict[str, str]


@dataclass
class DimensionAnalysis:
    """Output of :func:`analyze_dimensions`.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        dim_sizes: ``dim_name → extent`` (e.g. ``{"d0": 2048}``).
        tensors: All named tensors, keyed by name.
        ops: Compute ops in source order.
    """

    func_name: str
    param_names: list[str]
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    ops: list[OpAxes]

    def __repr__(self) -> str:
        """Multi-line dump: signature + dims + tensors + op calls."""
        lines: list[str] = [f"DimensionAnalysis {self.func_name}({', '.join(self.param_names)})", "dims:"]
        for name, size in self.dim_sizes.items():
            lines.append(f"  {name} size={size}")
        lines.append("tensors:")
        for t in self.tensors.values():
            shape_parts = [f"{d}={sz}" for d, sz in zip(t.dim_ids, t.shape)]
            shape_str = f"[{', '.join(shape_parts)}]" if shape_parts else "[]"
            lines.append(f"  {t.name}: shape={shape_str}")
        lines.append("ops:")
        for op in self.ops:
            operands = ",".join(f"{slot}={name}" for slot, name in op.operand_names.items())
            axes = ",".join(f"{k}={v}" for k, v in op.axis_map.items())
            lines.append(f"  {op.op_cls.__name__}({operands}) axes={{{axes}}}")
        return "\n".join(lines)


def analyze_dimensions(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> DimensionAnalysis:
    """Trace ``func`` against sentinel inputs and run cross-op dim unification.

    Args:
        func: An ``@nkigym_kernel``-decorated callable to analyse.
        input_specs: ``{param_name: shape}`` for every positional parameter.
    """
    unwrapped = inspect.unwrap(func)
    param_names = list(inspect.signature(unwrapped).parameters)
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    state = _TraceState(alloc_names=_collect_alloc_names(unwrapped))
    for name in param_names:
        state.sentinels[name] = _Sym(tuple(input_specs[name]), name)

    _run_trace(unwrapped, [state.sentinels[n] for n in param_names], state)
    _canonicalize_dim_names(state)

    tensors: dict[str, TensorDims] = {}
    for sym in state.sentinels.values():
        if any(d is None for d in sym.dim_ids):
            raise ValueError(f"Tensor {sym.source_name!r} has un-unified dims: {sym.dim_ids}")
        tensors[sym.source_name] = TensorDims(
            name=sym.source_name, shape=sym.shape, dim_ids=tuple(d for d in sym.dim_ids if d is not None)
        )
    return DimensionAnalysis(
        func_name=unwrapped.__name__,
        param_names=param_names,
        dim_sizes=state.dim_sizes,
        tensors=tensors,
        ops=state.op_records,
    )


class _Sym:
    """Symbolic tensor: shape + mutable ``dim_ids`` + source name."""

    __slots__ = ("shape", "dim_ids", "source_name")

    def __init__(self, shape: tuple[int, ...], source_name: str) -> None:
        self.shape: tuple[int, ...] = shape
        self.dim_ids: list[str | None] = [None] * len(shape)
        self.source_name: str = source_name


class _TraceState:
    """Mutable state threaded through the hook during tracing."""

    def __init__(self, alloc_names: Iterator[str]) -> None:
        self.sentinels: dict[str, _Sym] = {}
        self.dim_sizes: dict[str, int] = {}
        self.op_records: list[OpAxes] = []
        self.alloc_names = alloc_names
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
    """Build a replacement for :meth:`NKIOp.__call__` that records into ``state``."""

    def hook(op: NKIOp, **kwargs: Any) -> Any:
        merged = {**getattr(op, "_init_kwargs", {}), **kwargs}
        cls = type(op)
        if cls is NKIAlloc:
            name = next(state.alloc_names)
            sym = _Sym(tuple(merged["shape"]), name)
            state.sentinels[name] = sym
            return sym
        _trace_compute_op(cls, merged, state)
        return merged.get("dst")

    return hook


def _trace_compute_op(cls: type[NKIOp], kwargs: dict[str, Any], state: _TraceState) -> None:
    """Unify a compute op's operands and record an :class:`OpAxes` entry."""
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
    state.op_records.append(OpAxes(op_cls=cls, operand_names=operand_names, axis_map=local))


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


def _collect_alloc_names(func: Callable[..., Any]) -> Iterator[str]:
    """Yield the LHS of every ``var = NKIAlloc(...)()`` assignment in source order."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    for stmt in func_def.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and _is_alloc_call(stmt.value):
            yield target.id


def _is_alloc_call(node: ast.expr) -> bool:
    """Return True if ``node`` is an ``NKIAlloc(...)()`` double-call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and isinstance(node.func.func, ast.Name)
        and node.func.func.id == "NKIAlloc"
    )
