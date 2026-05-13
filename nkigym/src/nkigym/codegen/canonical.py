"""Canonical module builder: ``f_nkigym`` callable → :class:`KernelModule`.

Pipeline:
    1. AST-parse the math function to an ordered list of raw parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size from ``input_specs``; each op derives
       its own tile sizes from its ``MAX_TILE_SIZE`` (no cross-op coupling).
    4. Tag each tensor's ``origin`` (``param`` / ``intermediate`` / ``return``).
    5. Build the iter-var schedule tree: one :class:`SBlock` per op carrying
       block-local iter vars, wrapped in a 1N-per-dim :class:`ForNode`
       chain. Allocs produce root-level empty-iter-var SBlocks in source
       order; compute blocks follow in source order.
    6. Assign canonical loop names ``i_<dim>_<ordinal>`` across each tree.

The resulting :class:`KernelModule` is the IR the renderer and transform
atoms consume. :class:`KernelModule.dep` is populated lazily by the
default :class:`DepCache` factory.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nkigym.codegen.ir import (
    AccessRange,
    BufferAccess,
    ForNode,
    IterVar,
    KernelModule,
    NKIOpCall,
    SBlock,
    Tensor,
    TreeIR,
)
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole, NKIOp


def build_canonical_module(func: Callable[..., np.ndarray], input_specs: dict[str, dict]) -> KernelModule:
    """Build a :class:`KernelModule` from an ``f_nkigym`` callable.

    See module docstring for the pipeline.
    """
    raws, allocs, return_name = _parse_ast(func)
    unwrapped = getattr(func, "__wrapped__", func)
    param_names = list(inspect.signature(unwrapped).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    tensors = _build_tensor_map(param_names, input_specs, allocs, return_name)
    per_op_axis_maps, dim_sizes = _unify_axes(raws, tensors)

    module = KernelModule(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[],
    )

    name_to_axis_id = _derive_axes(module, dim_sizes)
    op_tiles = _derive_op_tiles(raws, per_op_axis_maps, dim_sizes, name_to_axis_id)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, op_tiles, name_to_axis_id)

    """Alloc blocks first (source order), then compute trees (source order).
    Each compute block allocates fresh iter vars via ``module.allocate_iter_var``;
    allocs get empty-iter-var root SBlocks."""
    for alloc in allocs:
        module.body.append(_make_alloc_sblock(alloc))
    for op in parsed_ops:
        module.body.append(_build_tree(op, module))

    for tree in module.body:
        _assign_canonical_names(tree, module, same_axis_counts={})

    return module


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
class _AllocRecord:
    """Canonical-build-time record for an NKIAlloc call in f_nkigym.

    Captures the tensor identity (name, location, shape, dtype) that
    populates ``module.tensors``. The resulting alloc SBlock carries
    only ``tensor_name`` — the declaration lives in ``module.tensors``.
    """

    name: str
    location: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class _ParsedOp:
    """One ``NKIOp()(...)`` call with fully resolved metadata.

    Attributes:
        idx: 0-indexed position in the math function.
        op_cls: The NKIOp subclass.
        operand_names: Operand slot → local variable name.
        op_kwargs: Merged literal kwargs from constructor + call.
        output_names: Names bound by the assignment target.
        axis_map: Abstract axis label → axis_id.
        touched_axes: Axis ids this op touches, canonical loop-nest order.
        axis_role: axis_id → :class:`AxisRole` (op-local).
        axis_tile: axis_id → tile size for this op.
    """

    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, int]
    touched_axes: tuple[int, ...]
    axis_role: dict[int, AxisRole]
    axis_tile: dict[int, int]


def _parse_ast(func: Callable[..., np.ndarray]) -> tuple[list[_ParsedOpRaw], list[_AllocRecord], str]:
    """Walk ``func``'s AST. Returns (op records, alloc records, return name)."""
    unwrapped = getattr(func, "__wrapped__", func)
    func_globals = unwrapped.__globals__
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
    raws: list[_ParsedOpRaw] = []
    allocs: list[_AllocRecord] = []
    return_name: str | None = None
    for stmt in func_def.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id
            continue
        if isinstance(stmt, ast.Assign):
            alloc = _try_parse_alloc(stmt, func_globals)
            if alloc is not None:
                allocs.append(alloc)
                continue
            raw = _parse_assignment(stmt, func_globals)
            if raw is not None:
                raws.append(raw)
        elif isinstance(stmt, ast.Expr):
            """Bare op calls like ``NKILoad()(src=..., dst=...)`` — no assignment."""
            raw = _parse_expr_call(stmt, func_globals)
            if raw is not None:
                raws.append(raw)
    if return_name is None:
        raise ValueError("f_nkigym must end with `return <tensor>`")
    return raws, allocs, return_name


def _parse_assignment(stmt: ast.Assign, func_globals: dict[str, object]) -> _ParsedOpRaw | None:
    """Convert one ``var = OpClass(...)(kw=...)`` statement into a ``_ParsedOpRaw``.

    Returns ``None`` for assignments that are not NKIOp double-calls.
    After the v2 refactor, ops no longer use OUTPUT_AXES — writes go through
    explicit operand slots (dst=...). The LHS of the assignment is ignored;
    ops must have their write operands in the RHS call."""
    result: _ParsedOpRaw | None = None
    if len(stmt.targets) != 1:
        return None
    op_cls = _resolve_op_class(stmt.value, func_globals)
    if op_cls is not None:
        outer_call = stmt.value
        assert isinstance(outer_call, ast.Call)
        operand_names = _extract_name_kwargs(outer_call)
        outer_kwargs = _extract_literal_kwargs(outer_call, func_globals)
        inner_call = outer_call.func
        assert isinstance(inner_call, ast.Call)
        merged = {**_extract_literal_kwargs(inner_call, func_globals), **outer_kwargs}
        result = _ParsedOpRaw(op_cls=op_cls, operand_names=operand_names, op_kwargs=merged, output_names=[])
    return result


def _parse_expr_call(stmt: ast.Expr, func_globals: dict[str, object]) -> _ParsedOpRaw | None:
    """Parse a bare expression statement like ``NKILoad()(src=..., dst=...)``.

    Returns None if the expression is not an NKIOp double-call."""
    if not isinstance(stmt.value, ast.Call):
        return None
    op_cls = _resolve_op_class(stmt.value, func_globals)
    if op_cls is None:
        return None
    outer_call = stmt.value
    operand_names = _extract_name_kwargs(outer_call)
    outer_kwargs = _extract_literal_kwargs(outer_call, func_globals)
    inner_call = outer_call.func
    assert isinstance(inner_call, ast.Call)
    merged = {**_extract_literal_kwargs(inner_call, func_globals), **outer_kwargs}
    return _ParsedOpRaw(op_cls=op_cls, operand_names=operand_names, op_kwargs=merged, output_names=[])


def _try_parse_alloc(stmt: ast.Assign, func_globals: dict[str, object]) -> _AllocRecord | None:
    """Extract an ``_AllocRecord`` from ``var = NKIAlloc(...)()``; return None otherwise."""
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    outer = stmt.value
    if not isinstance(outer, ast.Call) or not isinstance(outer.func, ast.Call):
        return None
    inner = outer.func
    if not isinstance(inner.func, ast.Name) or inner.func.id != "NKIAlloc":
        return None
    """Verify that the NKIAlloc name resolves to our op class."""
    candidate = func_globals.get(inner.func.id)
    if candidate is not NKIAlloc:
        return None
    kwargs = _extract_literal_kwargs(inner, func_globals)
    for req in ("location", "shape", "dtype"):
        if req not in kwargs:
            raise ValueError(f"NKIAlloc missing required kwarg {req!r} at '{stmt.targets[0].id}'")
    return _AllocRecord(
        name=stmt.targets[0].id, location=kwargs["location"], shape=tuple(kwargs["shape"]), dtype=kwargs["dtype"]
    )


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
    if isinstance(node, ast.Tuple):
        """Recursively resolve tuple elements."""
        values = []
        for elt in node.elts:
            ok, val = _literal_value(elt, func_globals)
            if not ok:
                return (False, None)
            values.append(val)
        result = (True, tuple(values))
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


def _build_tensor_map(
    param_names: list[str], input_specs: dict[str, dict], allocs: list[_AllocRecord], return_name: str
) -> dict[str, Tensor]:
    """Populate ``module.tensors`` from params + alloc records.

    Params are HBM-origin. Alloc records are intermediate-origin with
    their declared location. Dim ids are assigned lazily — each tensor's
    dim_ids get populated by ``_unify_axes`` as ops reference it.
    """
    out: dict[str, Tensor] = {}
    for name in param_names:
        spec = input_specs[name]
        shape = tuple(spec["shape"])
        dtype = spec["dtype"]
        out[name] = Tensor(name=name, dim_ids=(), shape=shape, dtype=dtype, origin="param", location="hbm")
    for alloc in allocs:
        origin = "return" if alloc.name == return_name else "intermediate"
        out[alloc.name] = Tensor(
            name=alloc.name, dim_ids=(), shape=alloc.shape, dtype=alloc.dtype, origin=origin, location=alloc.location
        )
    if return_name not in out:
        raise ValueError(f"Return tensor {return_name!r} not declared (missing NKIAlloc?)")
    return out


def _unify_axes(raws: list[_ParsedOpRaw], tensors: dict[str, Tensor]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Walk each op, unifying abstract axes against declared tensor shapes.

    Tensors come pre-declared from ``NKIAlloc`` records. We assign dim_ids
    and unify across operands.
    """
    dim_sizes: dict[str, int] = {}
    dim_counter = [0]
    per_op_axis_maps: list[dict[str, str]] = []
    for raw in raws:
        op_cls = raw.op_cls
        operand_map = {k: v for k, v in raw.operand_names.items() if v in tensors}
        local: dict[str, str] = {}
        for slot, axes in op_cls.OPERAND_AXES.items():
            if slot not in operand_map:
                continue
            tname = operand_map[slot]
            tensor = tensors[tname]
            if not tensor.dim_ids:
                """First op to touch this tensor seeds fresh dim_ids."""
                ids: list[str] = []
                for i, abstract in enumerate(axes[: len(tensor.shape)]):
                    if abstract not in local:
                        fresh = f"d{dim_counter[0]}"
                        dim_counter[0] += 1
                        dim_sizes[fresh] = tensor.shape[i]
                        local[abstract] = fresh
                    ids.append(local[abstract])
                tensors[tname] = replace(tensor, dim_ids=tuple(ids))
            else:
                for abstract, concrete in zip(axes, tensor.dim_ids):
                    if abstract in local and local[abstract] != concrete:
                        _unify_dim(tensors, per_op_axis_maps, dim_sizes, old_id=concrete, new_id=local[abstract])
                    else:
                        local[abstract] = concrete
        per_op_axis_maps.append(local)
    return per_op_axis_maps, dim_sizes


def _unify_dim(
    tensors: dict[str, Tensor],
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
    for tname, tensor in list(tensors.items()):
        new_dim_ids = tuple(new_id if d == old_id else d for d in tensor.dim_ids)
        if new_dim_ids != tensor.dim_ids:
            tensors[tname] = replace(tensor, dim_ids=new_dim_ids)
    for axis_map in per_op_axis_maps:
        for ax in axis_map:
            if axis_map[ax] == old_id:
                axis_map[ax] = new_id


def _derive_axes(module: KernelModule, dim_sizes: dict[str, int]) -> dict[str, int]:
    """Allocate one :class:`Axis` per concrete dim.

    Returns a ``name → axis_id`` map so subsequent helpers can translate
    abstract-axis strings / dim names → integer axis_ids.
    """
    name_to_axis_id: dict[str, int] = {}
    for name, size in dim_sizes.items():
        axis = module.allocate_axis(name=name, total_size=size)
        name_to_axis_id[name] = axis.axis_id
    return name_to_axis_id


def _derive_op_tiles(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    name_to_axis_id: dict[str, int],
) -> list[dict[int, int]]:
    """Per-op tile size map: ``op_tiles[i][axis_id] = tile_for_that_op_on_that_axis``.

    For each op, every axis it touches gets a tile size derived from that
    op's ``MAX_TILE_SIZE`` (the largest legal innermost-tile extent). An
    abstract axis with ``MAX_TILE_SIZE[axis] = None`` (or absent entry)
    defaults to the full extent.
    """
    out: list[dict[int, int]] = []
    for raw, axis_map in zip(raws, per_op_axis_maps):
        tiles: dict[int, int] = {}
        for abstract, dim_name in axis_map.items():
            axis_id = name_to_axis_id[dim_name]
            max_tile = raw.op_cls.MAX_TILE_SIZE.get(abstract)
            total = dim_sizes[dim_name]
            if max_tile is None:
                tiles[axis_id] = total
            else:
                if total % max_tile != 0:
                    raise ValueError(
                        f"{raw.op_cls.__name__}.{abstract}: extent {total} not divisible by MAX_TILE_SIZE {max_tile}"
                    )
                tiles[axis_id] = min(max_tile, total)
        out.append(tiles)
    return out


def _build_parsed_ops(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    tensors: dict[str, Tensor],
    op_tiles: list[dict[int, int]],
    name_to_axis_id: dict[str, int],
) -> list[_ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_axes`` and per-op tile map."""
    ops: list[_ParsedOp] = []
    for idx, (raw, axis_map, tiles) in enumerate(zip(raws, per_op_axis_maps, op_tiles)):
        axis_map_ids: dict[str, int] = {abstract: name_to_axis_id[dim_name] for abstract, dim_name in axis_map.items()}
        touched = _touched_axes(raw, axis_map_ids, tensors, name_to_axis_id)
        axis_role = _resolve_axis_role(raw.op_cls, axis_map_ids, touched)
        ops.append(
            _ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=axis_map_ids,
                touched_axes=touched,
                axis_role=axis_role,
                axis_tile=dict(tiles),
            )
        )
    return ops


def _resolve_axis_role(op_cls: type[NKIOp], axis_map: dict[str, int], touched: tuple[int, ...]) -> dict[int, AxisRole]:
    """Map every ``axis_id`` in ``touched`` to the op's role for that axis."""
    abstract_role = getattr(op_cls, "AXIS_ROLES", {})
    concrete: dict[int, AxisRole] = {}
    for abstract, axis_id in axis_map.items():
        if axis_id in touched and abstract in abstract_role:
            concrete[axis_id] = abstract_role[abstract]
    for axis_id in touched:
        concrete.setdefault(axis_id, AxisRole.PARALLEL)
    return concrete


def _touched_axes(
    raw: _ParsedOpRaw, axis_map: dict[str, int], tensors: dict[str, Tensor], name_to_axis_id: dict[str, int]
) -> tuple[int, ...]:
    """Canonical order: partition-axis id first, then free axes, then reducing axes."""
    _ = name_to_axis_id
    op_cls = raw.op_cls
    ordered: list[int] = []
    if op_cls.OUTPUT_AXES:
        first_out_slot = next(iter(op_cls.OUTPUT_AXES))
        out_axes = op_cls.OUTPUT_AXES[first_out_slot]
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


def _make_sblock(op: _ParsedOp, module: KernelModule) -> SBlock:
    """Build an iter-var :class:`SBlock` for ``op`` with per-op tiling.

    Allocates iter-vars per ``axis_id`` in ``op.touched_axes``:

    - **Bounded axis** (``tile < total``): TWO fresh :class:`IterVar`\\ s —
      an outer trip iter-var (``extent = total // tile``) followed by an
      inner tile iter-var (``extent = tile``). Per-axis addressing is
      ``outer * tile + inner``; the renderer elides the inner for-loop by
      spelling the inner iter-var as ``0`` in slice expressions.
    - **Unbounded axis** (``tile == total``): ONE fresh iter-var playing
      the role of the inner tile (``extent = total``, the full axis).
      There is no outer trip loop. The renderer still elides this single
      (innermost) loop into the ISA slice width.

    Operand slots split into three buckets based on the op's
    ``INPUT_OPERANDS`` / ``RMW_OPERANDS``.
    """
    axis_to_outer: dict[int, IterVar] = {}
    axis_to_inner: dict[int, IterVar] = {}
    iter_vars: list[IterVar] = []
    for axis_id in op.touched_axes:
        total = module.axes[axis_id].total_size
        tile = op.axis_tile[axis_id]
        role = op.axis_role[axis_id]
        if tile < total:
            """Bounded axis: outer trip + inner tile (two ForNodes)."""
            outer = module.allocate_iter_var(axis_id=axis_id, extent=total // tile, role=role)
            inner = module.allocate_iter_var(axis_id=axis_id, extent=tile, role=role)
            axis_to_outer[axis_id] = outer
            axis_to_inner[axis_id] = inner
            iter_vars.append(outer)
            iter_vars.append(inner)
        else:
            """Unbounded axis (tile == total): single tile ForNode for the whole axis.

            The single iter-var plays the role of 'inner tile' for BufferAccess
            coefficient purposes; no outer trip loop exists.
            """
            tile_iv = module.allocate_iter_var(axis_id=axis_id, extent=total, role=role)
            axis_to_inner[axis_id] = tile_iv
            iter_vars.append(tile_iv)

    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads: dict[str, BufferAccess] = {}
    writes: dict[str, BufferAccess] = {}
    reads_writes: dict[str, BufferAccess] = {}
    for slot, axes in op.op_cls.OPERAND_AXES.items():
        tname = op.operand_names.get(slot)
        if tname is None or tname not in module.tensors:
            continue
        access = _build_buffer_access(tname, axes, op, axis_to_outer, axis_to_inner, module)
        if slot in rmw:
            reads_writes[slot] = access
        elif slot in input_slots:
            reads[slot] = access
        else:
            writes[slot] = access

    call = NKIOpCall(
        op_cls=op.op_cls, kwargs=dict(op.op_kwargs), axis_map=dict(op.axis_map), dim_role=dict(op.axis_role)
    )
    return SBlock(
        iter_vars=iter_vars,
        reads=reads,
        writes=writes,
        reads_writes=reads_writes,
        body=[call],
        annotations={"axis_map": dict(op.axis_map)},
    )


def _build_buffer_access(
    tname: str,
    axes: tuple[str, ...],
    op: _ParsedOp,
    axis_to_outer: dict[int, IterVar],
    axis_to_inner: dict[int, IterVar],
    module: KernelModule,
) -> BufferAccess:
    """Produce a :class:`BufferAccess` for ``tname`` referenced by ``op`` via ``axes``.

    Per-dim :class:`AccessRange` shape depends on whether the axis is
    bounded or unbounded for this op:

    - **Bounded axis**: coefficients for BOTH the outer trip iter-var
      (coeff = ``tile``) and the inner tile iter-var (coeff = 1); full
      affine address along that dim is ``outer * tile + inner``. The
      range's ``extent`` equals ``tile`` — a per-iteration width for the
      innermost loop, which the renderer elides.
    - **Unbounded axis**: single coefficient on the sole tile iter-var
      (coeff = 1); ``extent`` equals the full axis size. No outer trip
      entry exists (no outer loop to index).
    """
    tensor = module.tensors[tname]
    iv_ids_seen: list[int] = []
    pattern_entries: list[AccessRange] = []
    for i, abstract in enumerate(axes):
        if i >= len(tensor.dim_ids):
            break
        axis_id = op.axis_map.get(abstract)
        if axis_id is None:
            """Fall back to tensor's declared dim_id for this slot via the
            module-level name→axis_id lookup. Canonical build should always
            have filled ``op.axis_map``, but this keeps the function total."""
            tensor_dim_name = tensor.dim_ids[i]
            try:
                axis_id = module.axis_id_by_name(tensor_dim_name)
            except KeyError:
                axis_id = -1
        inner = axis_to_inner.get(axis_id)
        if inner is None:
            """Untouched axis — constant-offset access along this dim.
            Canonical build never hits this branch but keeps the
            function total."""
            extent = op.axis_tile.get(axis_id, tensor.shape[i]) if axis_id in module.axes else tensor.shape[i]
            pattern_entries.append(AccessRange.make({}, 0, extent))
            continue
        outer = axis_to_outer.get(axis_id)
        if outer is not None:
            """Bounded axis: outer*tile + inner coefficient entries, extent = tile."""
            tile = op.axis_tile[axis_id]
            if outer.var_id not in iv_ids_seen:
                iv_ids_seen.append(outer.var_id)
            if inner.var_id not in iv_ids_seen:
                iv_ids_seen.append(inner.var_id)
            pattern_entries.append(AccessRange.make({outer.var_id: tile, inner.var_id: 1}, 0, tile))
        else:
            """Unbounded axis: single inner coefficient, extent = full axis."""
            total = module.axes[axis_id].total_size
            if inner.var_id not in iv_ids_seen:
                iv_ids_seen.append(inner.var_id)
            pattern_entries.append(AccessRange.make({inner.var_id: 1}, 0, total))
    return BufferAccess(tensor_name=tname, iter_var_ids=tuple(iv_ids_seen), pattern=tuple(pattern_entries))


def _make_alloc_sblock(alloc: _AllocRecord) -> SBlock:
    """Build the single :class:`SBlock` for an NKIAlloc declaration.

    Alloc blocks have no iter vars and no reads. The single
    :class:`NKIOpCall` carries the alloc metadata (``tensor_name``,
    ``location``, ``shape``, ``dtype``); the renderer looks up
    ``module.tensors[tensor_name]`` at emission time.
    """
    call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={"tensor_name": alloc.name, "location": alloc.location, "shape": alloc.shape, "dtype": alloc.dtype},
        axis_map={},
        dim_role={},
    )
    output_access = BufferAccess(tensor_name=alloc.name, iter_var_ids=(), pattern=())
    return SBlock(iter_vars=[], reads={}, writes={"output": output_access}, reads_writes={}, body=[call])


def _build_tree(op: _ParsedOp, module: KernelModule) -> ForNode | SBlock:
    """Build the iter-var schedule tree for ``op``.

    Bounded axes contribute TWO ForNodes per touched dim (outer trip +
    inner tile); unbounded axes contribute ONE ForNode (the tile).

    If ``op.touched_axes`` is empty, returns the :class:`SBlock` directly
    (no loop nest needed). Otherwise wraps in source-order per axis, so
    ``block.iter_vars[0]`` becomes the outermost :class:`ForNode` and the
    innermost :class:`ForNode` binds the last iter-var of the last
    touched axis.
    """
    block = _make_sblock(op, module)
    return _wrap_block_in_fornodes(block)


def _wrap_block_in_fornodes(block: SBlock) -> ForNode | SBlock:
    """Wrap ``block`` in one :class:`ForNode` per iter var, outermost first.

    The block's ``iter_vars`` list holds one entry per unbounded axis
    (the tile) and two entries per bounded axis (outer trip + inner
    tile), in axis source-order. Iterating in reverse nests them so
    that the first iter-var becomes the outermost :class:`ForNode`.

    Returns ``block`` unwrapped if it has no iter vars (alloc blocks).
    """
    if not block.iter_vars:
        return block
    node: ForNode | SBlock = block
    for iv in reversed(block.iter_vars):
        node = ForNode(iter_var=iv, children=[node])
    return node


def _assign_canonical_names(node: ForNode | SBlock, module: KernelModule, same_axis_counts: dict[int, int]) -> None:
    """Walk the tree in a root-outward DFS, naming each :class:`ForNode`.

    ``same_axis_counts[axis_id]`` tracks how many same-axis ancestors are
    already open on the current path; the newly visited node takes that
    as its ordinal, emits a name, then recurses with the counter
    incremented. Restoring the counter after recursion means siblings
    see the same counts the parent did (they are not each other's
    ancestors).
    """
    if isinstance(node, SBlock):
        return
    axis_id = node.iter_var.axis_id
    axis_name = module.axes[axis_id].name
    k = same_axis_counts.get(axis_id, 0)
    node.name = f"i_{axis_name}_{k}"
    same_axis_counts[axis_id] = k + 1
    for child in node.children:
        _assign_canonical_names(child, module, same_axis_counts)
    same_axis_counts[axis_id] = k


"""TreeIR re-exported only for callers that currently expect
``canonical.TreeIR`` — the type lives in :mod:`nkigym.codegen.ir`."""
_ = TreeIR
