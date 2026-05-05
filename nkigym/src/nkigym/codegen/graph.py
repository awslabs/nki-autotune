"""``parse_and_resolve``: turn an ``f_nkigym`` callable into an :class:`OpGraph`.

Pipeline:
    1. AST-parse the math function to an ordered list of parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size + tile size from ``input_specs`` and
       per-op ``TILE_LIMITS``.
    4. Tag each tensor's ``origin`` (``param`` / ``intermediate`` / ``return``).

The resulting :class:`OpGraph` is read-only: for any
``(func, input_specs)`` exactly one graph exists. There are no tunable
knobs; the renderer lowers this graph mechanically.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from nkigym.ops.base import NKIOp

TensorOrigin = Literal["param", "intermediate", "return"]


@dataclass
class Tensor:
    """Named tensor appearing in the ``f_nkigym`` body.

    Attributes:
        name: Source-level variable name (e.g. ``"lhs"`` or ``"rms_inv"``).
        dim_ids: Concrete dim ids in operand order (e.g. ``("d0", "d1")``).
        shape: Element sizes aligned with ``dim_ids``.
        dtype: Element dtype (e.g. ``"bfloat16"``, ``"float32"``).
        origin: Lineage role — ``"param"`` (HBM kernel input),
            ``"intermediate"`` (SBUF handoff), or ``"return"`` (final
            op output).
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin


@dataclass
class DimInfo:
    """Concrete dimension metadata derived from ops + input specs."""

    dim_id: str
    total_size: int
    tile_size: int
    num_tiles: int


@dataclass
class ParsedOp:
    """One ``NKIOp()(...)`` call captured from the ``f_nkigym`` body.

    Attributes:
        idx: 0-indexed position in the math function.
        op_cls: The NKIOp subclass (e.g. ``NKIMatmul``).
        operand_names: Maps operand slot name (``"data"``, ``"stationary"``
            etc.) to the local variable name the call references.
        op_kwargs: Literal keyword arguments merged from constructor
            and call site (e.g. ``{"op": "square", "scale": 1/2048}``).
        output_names: Names the assignment target binds.
        axis_map: Abstract axis label (``"K"``, ``"M"`` ...) to concrete
            dim id.
        touched_dims: Every dim id this op's operands or outputs mention,
            in canonical loop-nest order: partition-axis dim first, then
            free-axis dims, then any reducing dim.
    """

    idx: int
    op_cls: type
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]


@dataclass
class OpGraph:
    """Read-only resolved view of an ``f_nkigym`` function.

    Attributes:
        func_name: Function name (lands on the emitted kernel).
        param_names: Kernel parameters in signature order.
        return_name: Tensor name of the return value (the ``NKIStore``
            output).
        tensors: All named tensors, keyed by name.
        dims: All dims, keyed by dim id.
        ops: Parsed ops in source order.
        per_op_attrs: Per-op annotation side-table keyed by
            ``ParsedOp.idx``. Empty by default — reserved for future
            passes like ``propagate_compute_skip``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    ops: list[ParsedOp]
    per_op_attrs: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _ParsedOpRaw:
    """Raw AST-parsed op record — pre dim unification.

    Attributes:
        op_cls: Resolved NKIOp subclass.
        operand_names: Name-valued kwargs from the outer call
            (tensor operands).
        op_kwargs: Merged literal kwargs from inner ``OpClass(...)``
            constructor and outer call.
        output_names: Names bound by the assignment target.
    """

    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]


def _parse_ast(func: Callable[..., np.ndarray]) -> tuple[list[_ParsedOpRaw], str]:
    """Walk ``func``'s AST to extract ordered ``NKIOp`` calls + return name.

    The ``@nkigym_kernel`` decorator wraps the function; the wrapper's
    ``__globals__`` points to ``nkigym.ops.base``'s globals, not the
    caller module's. We unwrap via ``__wrapped__`` to recover the
    original function's ``__globals__`` which contains the op imports.

    Args:
        func: An ``f_nkigym`` math function decorated with
            ``@nkigym_kernel``. Body must be straight-line
            ``var = NKIOp()(...)`` assignments terminated by a
            ``return var`` statement.

    Returns:
        ``(raws, return_name)`` — one ``_ParsedOpRaw`` per NKIOp call,
        in source order, plus the name of the returned tensor.

    Raises:
        ValueError: The source lacks a ``return`` or binds a
            non-``Name`` target.
    """
    unwrapped = getattr(func, "__wrapped__", func)
    func_globals = unwrapped.__globals__
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    raws: list[_ParsedOpRaw] = []
    return_name: str | None = None
    for stmt in func_def.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id
            continue
        if isinstance(stmt, ast.Assign):
            raw = _parse_assignment(stmt, func_globals)
            if raw is not None:
                raws.append(raw)
    if return_name is None:
        raise ValueError("f_nkigym must end with `return <tensor>`")
    return raws, return_name


def _parse_assignment(stmt: ast.Assign, func_globals: dict[str, object]) -> _ParsedOpRaw | None:
    """Convert one ``var = OpClass(...)(kw=...)`` statement into a ``_ParsedOpRaw``.

    Returns ``None`` for assignments that are not NKIOp double-calls.
    """
    result: _ParsedOpRaw | None = None
    op_cls = _resolve_op_class(stmt.value, func_globals) if len(stmt.targets) == 1 else None
    if op_cls is not None:
        output_names = _extract_output_names(stmt.targets[0])
        if output_names:
            if len(output_names) != len(op_cls.OUTPUT_AXES):
                raise ValueError(
                    f"Op {op_cls.NAME}: {len(output_names)} outputs assigned but "
                    f"OUTPUT_AXES has {len(op_cls.OUTPUT_AXES)} entries"
                )
            outer_call = stmt.value
            assert isinstance(outer_call, ast.Call)
            operand_names = _extract_name_kwargs(outer_call)
            outer_kwargs = _extract_literal_kwargs(outer_call, func_globals)
            inner_call = outer_call.func
            assert isinstance(inner_call, ast.Call)
            merged = {**_extract_literal_kwargs(inner_call, func_globals), **outer_kwargs}
            result = _ParsedOpRaw(
                op_cls=op_cls, operand_names=operand_names, op_kwargs=merged, output_names=output_names
            )
    return result


def _resolve_op_class(node: ast.expr, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Return the NKIOp subclass a double-call expression references, or ``None``."""
    result: type[NKIOp] | None = None
    is_double = isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name)
    if is_double:
        inner_call = node.func
        assert isinstance(inner_call, ast.Call)
        name_node = inner_call.func
        assert isinstance(name_node, ast.Name)
        candidate = func_globals.get(name_node.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            result = candidate
    return result


def _extract_name_kwargs(call: ast.Call) -> dict[str, str]:
    """Return ``{kwarg_name: local_variable_name}`` for every Name-valued kwarg."""
    return {kw.arg: kw.value.id for kw in call.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)}


def _extract_output_names(target: ast.expr) -> list[str]:
    """Extract the bound names from an assignment target."""
    if isinstance(target, ast.Name):
        names = [target.id]
    elif isinstance(target, ast.Tuple):
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
    else:
        names = []
    return names


def _extract_literal_kwargs(call: ast.Call, func_globals: dict[str, object]) -> dict[str, Any]:
    """Return ``{kwarg_name: python_literal}`` for every literal-valued kwarg.

    Handles plain ``ast.Constant`` values, BinOp/UnaryOp trees over
    constants (``1/2048``, ``-eps``), and Name references that resolve
    to numeric constants in ``func_globals`` (e.g. module-level ``EPS``).
    """
    out: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        ok, value = _literal_value(kw.value, func_globals)
        if ok:
            out[kw.arg] = value
    return out


def _literal_value(node: ast.expr, func_globals: dict[str, object]) -> tuple[bool, Any]:
    """Try to evaluate ``node`` as a Python literal. Returns ``(ok, value)``."""
    result: tuple[bool, Any] = (False, None)
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name):
        value = func_globals.get(node.id)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = (True, value)
    if isinstance(node, ast.UnaryOp):
        ok, inner = _literal_value(node.operand, func_globals)
        if ok and isinstance(node.op, ast.USub):
            result = (True, -inner)
        elif ok and isinstance(node.op, ast.UAdd):
            result = (True, +inner)
    if isinstance(node, ast.BinOp):
        ok_l, lhs = _literal_value(node.left, func_globals)
        ok_r, rhs = _literal_value(node.right, func_globals)
        if ok_l and ok_r:
            if isinstance(node.op, ast.Add):
                result = (True, lhs + rhs)
            elif isinstance(node.op, ast.Sub):
                result = (True, lhs - rhs)
            elif isinstance(node.op, ast.Mult):
                result = (True, lhs * rhs)
            elif isinstance(node.op, ast.Div):
                result = (True, lhs / rhs)
    return result


def parse_and_resolve(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> OpGraph:
    """AST-parse ``func`` and resolve dims, tensors, tile sizes.

    Args:
        func: A math function decorated with ``@nkigym_kernel`` whose
            body is straight-line ``NKIOp()(...)`` assignments followed
            by ``return <tensor>``.
        input_specs: ``{param_name: (shape, dtype)}`` for every function
            parameter.

    Returns:
        A fully resolved :class:`OpGraph`.
    """
    raws, return_name = _parse_ast(func)
    unwrapped = getattr(func, "__wrapped__", func)
    param_names = list(inspect.signature(unwrapped).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    tensors_scratch: dict[str, _ScratchTensor] = {}
    for name in param_names:
        shape, dtype = input_specs[name]
        tensors_scratch[name] = _ScratchTensor(name=name, shape=shape, dtype=dtype)

    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for raw in raws:
        op_cls = raw.op_cls
        operand_map = {k: v for k, v in raw.operand_names.items() if v in tensors_scratch}
        axis_map = _build_axis_map(op_cls, operand_map, tensors_scratch, dim_counter, per_op_axis_maps, dim_sizes)
        per_op_axis_maps.append(axis_map)
        _create_outputs(op_cls, operand_map, raw.output_names, axis_map, tensors_scratch, dim_sizes)

    if return_name not in tensors_scratch:
        raise ValueError(f"Return tensor {return_name!r} not produced by any op")

    dims = _resolve_dimensions(raws, per_op_axis_maps, dim_sizes)
    tensors = _finalize_tensors(tensors_scratch, param_names, return_name, raws)
    ops = _build_parsed_ops(raws, per_op_axis_maps, tensors)

    return OpGraph(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        ops=ops,
        per_op_attrs={},
    )


@dataclass
class _ScratchTensor:
    """Mutable dim-inference record used during resolution."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dim_ids: list[str] = field(default_factory=list)


def _build_axis_map(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    tensors: dict[str, _ScratchTensor],
    dim_counter: list[int],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
) -> dict[str, str]:
    """Zip abstract axis labels against each operand's concrete dim ids."""
    local: dict[str, str] = {}
    for slot, axes in op_cls.OPERAND_AXES.items():
        if slot not in operand_map:
            continue
        tensor = tensors[operand_map[slot]]
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


def _unify_dim(
    tensors: dict[str, _ScratchTensor],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    old_id: str,
    new_id: str,
) -> None:
    """Rename ``old_id`` to ``new_id`` everywhere; raise on size conflict."""
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


def _create_outputs(
    op_cls: type[NKIOp],
    operand_map: dict[str, str],
    output_names: list[str],
    local: dict[str, str],
    tensors: dict[str, _ScratchTensor],
    dim_sizes: dict[str, int],
) -> None:
    """Register output tensors with concrete dim ids derived from ``OUTPUT_AXES``."""
    first_slot = next(iter(op_cls.OPERAND_AXES))
    has_first = first_slot in operand_map and operand_map[first_slot] in tensors
    default_dtype = (
        tensors[operand_map[first_slot]].dtype if has_first else next(t.dtype for t in tensors.values() if t.dtype)
    )
    for oname, (out_slot, output_axes) in zip(output_names, op_cls.OUTPUT_AXES.items()):
        dim_ids = [local[a] for a in output_axes if a in local]
        shape = tuple(dim_sizes[d] for d in dim_ids)
        dtype = op_cls.OUTPUT_DTYPES.get(out_slot, default_dtype)
        tensors[oname] = _ScratchTensor(name=oname, shape=shape, dtype=dtype, dim_ids=dim_ids)


def _resolve_dimensions(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], dim_sizes: dict[str, int]
) -> dict[str, DimInfo]:
    """Derive per-dim tile size = min of op TILE_LIMITS touching the dim."""
    per_dim_tile: dict[str, int] = {}
    for raw, axis_map in zip(raws, per_op_axis_maps):
        for abstract, limit in raw.op_cls.TILE_LIMITS.items():
            if abstract not in axis_map:
                continue
            dim_id = axis_map[abstract]
            tile = min(limit, dim_sizes[dim_id])
            per_dim_tile[dim_id] = min(per_dim_tile.get(dim_id, tile), tile)
    for d in dim_sizes:
        if d not in per_dim_tile:
            """No op declared a tile limit on ``d`` — only DMA ops touch
            it (NKILoad/NKIStore have unbounded free axis). Default to
            the whole extent so the dim lives in a single tile."""
            per_dim_tile[d] = dim_sizes[d]
    return {
        d: DimInfo(
            dim_id=d, total_size=dim_sizes[d], tile_size=per_dim_tile[d], num_tiles=dim_sizes[d] // per_dim_tile[d]
        )
        for d in dim_sizes
    }


def _finalize_tensors(
    scratch: dict[str, _ScratchTensor], param_names: list[str], return_name: str, raws: list[_ParsedOpRaw]
) -> dict[str, Tensor]:
    """Convert scratch tensors to read-only ``Tensor``s, tagging origin."""
    out: dict[str, Tensor] = {}
    for name, st in scratch.items():
        if name in param_names:
            origin: TensorOrigin = "param"
        elif name == return_name:
            origin = "return"
        else:
            origin = "intermediate"
        out[name] = Tensor(name=name, dim_ids=tuple(st.dim_ids), shape=tuple(st.shape), dtype=st.dtype, origin=origin)
    _ = raws
    return out


def _build_parsed_ops(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], tensors: dict[str, Tensor]
) -> list[ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims``."""
    ops: list[ParsedOp] = []
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        ops.append(
            ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
            )
        )
    return ops


def _touched_dims(raw: _ParsedOpRaw, axis_map: dict[str, str], tensors: dict[str, Tensor]) -> tuple[str, ...]:
    """Canonical order: partition-axis dim first, then free dims, then reducing dims."""
    op_cls = raw.op_cls
    first_out_slot = next(iter(op_cls.OUTPUT_AXES))
    out_axes = op_cls.OUTPUT_AXES[first_out_slot]
    ordered: list[str] = []
    for abstract in out_axes:
        if abstract in axis_map and axis_map[abstract] not in ordered:
            ordered.append(axis_map[abstract])
    blocking = [axis_map[a] for a in op_cls.BLOCKING_AXES if a in axis_map]
    for d in blocking:
        if d not in ordered:
            ordered.append(d)
    for slot, axes in op_cls.OPERAND_AXES.items():
        tname = raw.operand_names.get(slot)
        if tname is None or tname not in tensors:
            continue
        for abstract in axes:
            if abstract in axis_map and axis_map[abstract] not in ordered:
                ordered.append(axis_map[abstract])
    return tuple(ordered)
