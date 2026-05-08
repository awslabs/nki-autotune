"""Canonical module builder: ``f_nkigym`` callable → :class:`KernelModule`.

Pipeline:
    1. AST-parse the math function to an ordered list of raw parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size + tile size from ``input_specs`` and
       per-op ``TILE_LIMITS``.
    4. Tag each tensor's ``origin`` (``param`` / ``intermediate`` / ``return``).
    5. Build the canonical 2N-per-op forest with phase leaves at the
       deepest point; populate every :class:`BodyLeaf` with FULL metadata
       so leaves are self-describing (no back-reference to a sidecar).
    6. Assign canonical loop names ``i_<dim>_<ordinal>`` across each tree.

The resulting :class:`KernelModule` is the IR the renderer and transform
atoms consume. :class:`KernelModule.dep` is populated lazily by the
default :class:`DepCache` factory.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, LoopNode, OpLocalBuffer, Tensor, TensorOrigin, TreeIR
from nkigym.ops.base import AxisRole, NKIOp


def build_canonical_module(func: Callable[..., np.ndarray], input_specs: dict[str, dict]) -> KernelModule:
    """Build a :class:`KernelModule` from an ``f_nkigym`` callable.

    Args:
        func: A math function decorated with ``@nkigym_kernel`` whose
            body is straight-line ``NKIOp()(...)`` assignments followed
            by ``return <tensor>``.
        input_specs: ``{param_name: {"shape": (...), "dtype": str}}`` for
            every function parameter.

    Returns:
        A fully resolved :class:`KernelModule` with canonical 2N-per-op
        schedule tree and self-describing leaves.
    """
    raws, return_name = _parse_ast(func)
    unwrapped = getattr(func, "__wrapped__", func)
    param_names = list(inspect.signature(unwrapped).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    tensors_scratch, dim_sizes, per_op_axis_maps = _unify_axes(raws, param_names, input_specs)
    if return_name not in tensors_scratch:
        raise ValueError(f"Return tensor {return_name!r} not produced by any op")

    dims = _derive_dims(raws, per_op_axis_maps, dim_sizes)
    _register_op_local_derived_dims(raws, per_op_axis_maps, dims)
    tensors = _build_tensor_map(tensors_scratch, param_names, return_name)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)

    body: TreeIR = [_build_tree(op, dims) for op in parsed_ops]
    for tree in body:
        _assign_canonical_names(tree, same_dim_counts={})

    return KernelModule(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        body=body,
    )


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


@dataclass
class _ScratchTensor:
    """Mutable dim-inference record used during resolution."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dim_ids: list[str] = field(default_factory=list)


@dataclass
class _ParsedOp:
    """One ``NKIOp()(...)`` call with fully resolved metadata.

    Attributes:
        idx: 0-indexed position in the math function.
        op_cls: The NKIOp subclass.
        operand_names: Operand slot → local variable name.
        op_kwargs: Merged literal kwargs from constructor + call.
        output_names: Names bound by the assignment target.
        axis_map: Abstract axis label → concrete dim id.
        touched_dims: Dim ids this op touches, canonical loop-nest order.
        dim_role: Concrete dim id → :class:`AxisRole` (op-local).
        op_local_buffers: Resolved op-local buffers keyed by logical name.
    """

    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]
    dim_role: dict[str, AxisRole]
    op_local_buffers: dict[str, OpLocalBuffer] = field(default_factory=dict)


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


def _unify_axes(
    raws: list[_ParsedOpRaw], param_names: list[str], input_specs: dict[str, dict]
) -> tuple[dict[str, _ScratchTensor], dict[str, int], list[dict[str, str]]]:
    """Unify abstract axes across ops into concrete dim ids.

    Seeds scratch tensors from ``input_specs``, then walks every raw op
    in source order, building its ``axis_map`` against the operands it
    references. Output tensors are materialised as we go so later ops
    see resolved dim ids.

    Args:
        raws: Raw AST-parsed ops in source order.
        param_names: Kernel parameter names (signature order).
        input_specs: ``{param_name: {"shape": ..., "dtype": ...}}``.

    Returns:
        ``(tensors_scratch, dim_sizes, per_op_axis_maps)``.
    """
    tensors_scratch: dict[str, _ScratchTensor] = {}
    for name in param_names:
        spec = input_specs[name]
        shape = tuple(spec["shape"])
        dtype = spec["dtype"]
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
    return tensors_scratch, dim_sizes, per_op_axis_maps


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


def _derive_dims(
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


def _register_op_local_derived_dims(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], dims: dict[str, DimInfo]
) -> None:
    """Register derived dims referenced by op-local buffer axis_ids.

    For now, the only derived axis label recognized is ``F_slot``: one
    element per F-tile, inherited from the op's F dim's ``num_tiles``.
    For each op that declares an ``OP_LOCAL_BUFFERS`` entry mentioning
    ``F_slot``, we register a derived ``DimInfo`` keyed ``<f_dim>_f_slot``
    with ``tile_size=1``, ``num_tiles=dims[f_dim].num_tiles``.

    Args:
        raws: Parsed ops in source order.
        per_op_axis_maps: Concrete axis_maps aligned with ``raws``.
        dims: The resolved dim table (mutated in place to add derived dims).
    """
    for raw, axis_map in zip(raws, per_op_axis_maps):
        local_buffers = getattr(raw.op_cls, "OP_LOCAL_BUFFERS", {})
        for _, (_, _, axis_ids) in local_buffers.items():
            for axis_id in axis_ids:
                if axis_id == "F_slot":
                    if "F" not in axis_map:
                        raise ValueError(
                            f"Op {raw.op_cls.NAME}: OP_LOCAL_BUFFERS references F_slot "
                            f"but op has no F axis in axis_map"
                        )
                    f_dim_id = axis_map["F"]
                    derived_id = f"{f_dim_id}_f_slot"
                    if derived_id not in dims:
                        f_info = dims[f_dim_id]
                        dims[derived_id] = DimInfo(
                            dim_id=derived_id, total_size=f_info.num_tiles, tile_size=1, num_tiles=f_info.num_tiles
                        )


def _build_tensor_map(
    scratch: dict[str, _ScratchTensor], param_names: list[str], return_name: str
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
    return out


def _resolve_op_local_buffers(
    raw: _ParsedOpRaw, axis_map: dict[str, str], dims: dict[str, DimInfo], counter: list[int]
) -> dict[str, OpLocalBuffer]:
    """Materialize op-local buffer records from ``OP_LOCAL_BUFFERS``.

    Each entry gets a unique ``emitted_name`` via the shared ``counter``.
    Buffer shape is computed from the declared ``axis_ids`` using
    op-local sizing rules: partition dim → ``(tile_size, 1, ...)``; the
    free dim contributes ``num_tiles * tile_size`` (one element per F
    element for ``F``, one element per F-tile for ``F_slot``).

    Args:
        raw: Parsed op record.
        axis_map: Concrete axis_map for this op.
        dims: Resolved dim table (must already include any derived
            ``F_slot`` entries; see :func:`_register_op_local_derived_dims`).
        counter: Mutable single-element list used to assign monotonic
            buffer ids across the whole kernel.

    Returns:
        Dict keyed by logical name.
    """
    out: dict[str, OpLocalBuffer] = {}
    local_buffers = getattr(raw.op_cls, "OP_LOCAL_BUFFERS", {})
    for logical_name, (location, dtype, axis_ids) in local_buffers.items():
        if len(axis_ids) != 2:
            raise ValueError(
                f"Op {raw.op_cls.NAME}: OP_LOCAL_BUFFERS['{logical_name}'] must have 2 axis_ids "
                f"(P, F-like), got {axis_ids}"
            )
        p_axis, f_axis = axis_ids
        p_dim_id = axis_map[p_axis]
        p_info = dims[p_dim_id]
        if f_axis == "F_slot":
            f_dim_id = f"{axis_map['F']}_f_slot"
        else:
            f_dim_id = axis_map[f_axis]
        f_info = dims[f_dim_id]
        shape = (p_info.tile_size, 1, f_info.num_tiles * f_info.tile_size)
        emitted = f"{location}_local_{counter[0]}"
        counter[0] += 1
        out[logical_name] = OpLocalBuffer(
            logical_name=logical_name,
            emitted_name=emitted,
            location=location,
            dtype=dtype,
            axis_ids=tuple(axis_ids),
            shape=shape,
        )
    return out


def _build_parsed_ops(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    tensors: dict[str, Tensor],
    dims: dict[str, DimInfo],
) -> list[_ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims``."""
    ops: list[_ParsedOp] = []
    counter = [0]
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        dim_role = _resolve_dim_role(raw.op_cls, axis_map, touched)
        op_local_buffers = _resolve_op_local_buffers(raw, axis_map, dims, counter)
        ops.append(
            _ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
                dim_role=dim_role,
                op_local_buffers=op_local_buffers,
            )
        )
    return ops


def _resolve_dim_role(op_cls: type[NKIOp], axis_map: dict[str, str], touched: tuple[str, ...]) -> dict[str, AxisRole]:
    """Map every ``dim_id`` in ``touched`` to the op's role for that dim."""
    abstract_role = getattr(op_cls, "AXIS_ROLES", {})
    concrete: dict[str, AxisRole] = {}
    for abstract, dim_id in axis_map.items():
        if dim_id in touched and abstract in abstract_role:
            concrete[dim_id] = abstract_role[abstract]
    for dim_id in touched:
        concrete.setdefault(dim_id, AxisRole.PARALLEL)
    return concrete


def _touched_dims(raw: _ParsedOpRaw, axis_map: dict[str, str], tensors: dict[str, Tensor]) -> tuple[str, ...]:
    """Canonical order: partition-axis dim first, then free dims, then reducing dims."""
    op_cls = raw.op_cls
    first_out_slot = next(iter(op_cls.OUTPUT_AXES))
    out_axes = op_cls.OUTPUT_AXES[first_out_slot]
    ordered: list[str] = []
    for abstract in out_axes:
        if abstract in axis_map and axis_map[abstract] not in ordered:
            ordered.append(axis_map[abstract])
    non_parallel_axes = getattr(op_cls, "AXIS_ROLES", {}).keys()
    for abstract in non_parallel_axes:
        if abstract in axis_map and axis_map[abstract] not in ordered:
            ordered.append(axis_map[abstract])
    for slot, axes in op_cls.OPERAND_AXES.items():
        tname = raw.operand_names.get(slot)
        if tname is None or tname not in tensors:
            continue
        for abstract in axes:
            if abstract in axis_map and axis_map[abstract] not in ordered:
                ordered.append(axis_map[abstract])
    return tuple(ordered)


def _make_leaf(op: _ParsedOp, phase: str) -> BodyLeaf:
    """Build a self-describing :class:`BodyLeaf` for ``op``'s ``phase``.

    Copies operand/output/kwargs/axis_map/dim_role/op_local_buffers so
    the leaf does not share mutable state with the parsed op.
    """
    return BodyLeaf(
        op_cls=op.op_cls,
        phase=phase,
        reads=dict(op.operand_names),
        writes=tuple(op.output_names),
        kwargs=dict(op.op_kwargs),
        axis_map=dict(op.axis_map),
        dim_role=dict(op.dim_role),
        op_local_buffers=dict(op.op_local_buffers),
    )


def _build_tree(op: _ParsedOp, dims: dict[str, DimInfo]) -> LoopNode:
    """Build the 2N-per-dim chain for ``op`` with phase leaves at the tip."""
    deepest_children = _build_leaves(op, dims)
    wrap_dims = _dims_to_wrap(op)
    return _wrap_dims(wrap_dims, op, dims, deepest_children)


def _dims_to_wrap(op: _ParsedOp) -> tuple[str, ...]:
    """Return the dims the outer wrapper should build around the leaves.

    Multi-phase builders may handle some interior dims themselves (e.g.
    matmul builds K internally). For those builders, the dims they
    consume are dropped from the outer wrap.
    """
    interior_fn = _BUILDER_INTERIOR_DIMS.get(op.op_cls.__name__)
    skip = interior_fn(op) if interior_fn is not None else set()
    return tuple(d for d in op.touched_dims if d not in skip)


def _wrap_dims(
    wrap: tuple[str, ...], op: _ParsedOp, dims: dict[str, DimInfo], inner_children: list[LoopNode | BodyLeaf]
) -> LoopNode:
    """Wrap ``inner_children`` in a 1N-per-dim chain over ``wrap``.

    Each dim contributes one ``LoopNode`` with ``trip_count = num_tiles``.
    The outer wrapper matches the dim ordering in ``touched_dims``
    (outermost first). Arithmetic-intensity dials (tiles-per-block)
    live in ``Split``, not here.
    """
    if not wrap:
        raise ValueError(f"Op {op.idx}: cannot build tree — no touched_dims to wrap")
    node_children: list[LoopNode | BodyLeaf] = inner_children
    for d in reversed(wrap):
        role = op.dim_role[d]
        num_t = dims[d].num_tiles
        block_node = LoopNode(dim_id=d, trip_count=num_t, role=role, children=node_children)
        node_children = [block_node]
    head = node_children[0]
    assert isinstance(head, LoopNode)
    return head


def _build_leaves(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """Return the deepest-point children list for ``op``'s tree.

    Dispatch on op-class name — single-phase ops return a single
    ``[BodyLeaf(... phase="main")]``; multi-phase ops (matmul,
    activation_reduce) use custom builders registered in
    :data:`_LEAF_BUILDERS`.
    """
    builder = _LEAF_BUILDERS.get(op.op_cls.__name__, _build_leaves_default)
    return builder(op, dims)


def _build_leaves_default(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """Single-phase default: one ``BodyLeaf(... phase='main')``."""
    _ = dims
    return [_make_leaf(op, "main")]


def _build_leaves_matmul(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """Matmul: ``[psum_init leaf, <K loop with compute leaf>, drain leaf]``.

    The outer M and N dims are consumed by ``_wrap_dims``. The K dim is
    handled here so the body placement mirrors the physical kernel:
    PSUM init lives outside K, ``nc_matmul`` fires inside K, drain runs
    after K closes. The K loop carries ``reduce_op="add"`` because
    nc_matmul's PSUM accumulator is summation.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = dims[k_dim].num_tiles
    compute_leaf = _make_leaf(op, "compute")
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[compute_leaf], reduce_op="add")
    return [_make_leaf(op, "psum_init"), k_block, _make_leaf(op, "drain")]


def _build_leaves_activation_reduce(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce Pattern 2: ``[<F loop with reduce_step>, reduce_close]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: one F loop holds the per-tile ``reduce_step`` BodyLeaf that
    writes each tile's partial sum into a distinct slot of the op-local
    ``slot_vec``. After the F loop exits, ``reduce_close`` folds the
    slot vector via one ``nisa.tensor_reduce`` into the op's ``(P, 1)``
    output.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = _make_leaf(op, "reduce_step")
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    return [f_block, _make_leaf(op, "reduce_close")]


def _assign_canonical_names(node: "LoopNode | BodyLeaf", same_dim_counts: dict[str, int]) -> None:
    """Walk the tree in a root-outward DFS, naming each LoopNode.

    ``same_dim_counts[d]`` tracks how many same-dim ancestors of ``d``
    are already open on the current path; the newly visited node takes
    that as its ordinal, emits a name, then recurses with the counter
    incremented. Restoring the counter after recursion means siblings
    see the same counts the parent did (they are not each other's
    ancestors).
    """
    if isinstance(node, BodyLeaf):
        return
    k = same_dim_counts.get(node.dim_id, 0)
    node.name = f"i_{node.dim_id}_{k}"
    same_dim_counts[node.dim_id] = k + 1
    for child in node.children:
        _assign_canonical_names(child, same_dim_counts)
    same_dim_counts[node.dim_id] = k


_LEAF_BUILDERS: dict[str, Callable[[_ParsedOp, dict[str, DimInfo]], list[LoopNode | BodyLeaf]]] = {
    "NKIMatmul": _build_leaves_matmul,
    "NKIActivationReduce": _build_leaves_activation_reduce,
}
"""Dispatch table: op-class name → deepest-children builder."""

_BUILDER_INTERIOR_DIMS: dict[str, Callable[[_ParsedOp], set[str]]] = {
    "NKIMatmul": lambda op: {op.axis_map["K"]},
    "NKIActivationReduce": lambda op: {op.axis_map["F"]},
}
"""Maps op-class name to a callable ``(op) -> set[dim_id]`` of dims that
the custom leaf builder handles internally (so the outer wrap skips them)."""
