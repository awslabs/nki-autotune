"""Canonical module builder: ``f_nkigym`` callable → :class:`KernelModule`.

Pipeline:
    1. AST-parse the math function to an ordered list of raw parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size + tile size from ``input_specs`` and
       per-op ``TILE_LIMITS``.
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
    DimInfo,
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
    dims = _derive_dims(raws, per_op_axis_maps, dim_sizes)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)

    module = KernelModule(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        iter_var_counter=0,
        body=[],
    )

    """Alloc blocks first (source order), then compute trees (source order).
    Each compute block allocates fresh iter vars via ``module.allocate_iter_var``;
    allocs get empty-iter-var root SBlocks."""
    for alloc in allocs:
        module.body.append(_make_alloc_sblock(alloc))
    for op in parsed_ops:
        module.body.append(_build_tree(op, module))

    for tree in module.body:
        _assign_canonical_names(tree, same_dim_counts={})

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
        axis_map: Abstract axis label → concrete dim id.
        touched_dims: Dim ids this op touches, canonical loop-nest order.
        dim_role: Concrete dim id → :class:`AxisRole` (op-local).
    """

    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]
    dim_role: dict[str, AxisRole]


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


def _build_parsed_ops(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    tensors: dict[str, Tensor],
    dims: dict[str, DimInfo],
) -> list[_ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims``."""
    _ = dims
    ops: list[_ParsedOp] = []
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        dim_role = _resolve_dim_role(raw.op_cls, axis_map, touched)
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
    ordered: list[str] = []
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
    """Build an iter-var :class:`SBlock` for ``op``.

    Allocates one fresh :class:`IterVar` per ``dim_id`` in ``op.touched_dims``
    via ``module.allocate_iter_var`` (role = op-local role, extent = dim's
    num_tiles). Operand slots split into three buckets based on the op's
    ``INPUT_OPERANDS`` / ``RMW_OPERANDS``:

    - ``RMW_OPERANDS``          → ``reads_writes``
    - ``INPUT_OPERANDS``        → ``reads``
    - Otherwise (``dst`` etc.)  → ``writes``

    Each bucket maps slot_name → :class:`BufferAccess` whose pattern carries
    one :class:`AccessRange` per tensor dim (coefficient 1, offset 0,
    extent = the dim's tile_size).
    """
    dim_to_iv: dict[str, IterVar] = {}
    iter_vars: list[IterVar] = []
    for dim_id in op.touched_dims:
        extent = module.dims[dim_id].num_tiles
        iv = module.allocate_iter_var(dim_id=dim_id, extent=extent, role=op.dim_role[dim_id])
        dim_to_iv[dim_id] = iv
        iter_vars.append(iv)

    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads: dict[str, BufferAccess] = {}
    writes: dict[str, BufferAccess] = {}
    reads_writes: dict[str, BufferAccess] = {}
    for slot, axes in op.op_cls.OPERAND_AXES.items():
        tname = op.operand_names.get(slot)
        if tname is None or tname not in module.tensors:
            continue
        access = _build_buffer_access(tname, axes, op, dim_to_iv, module)
        if slot in rmw:
            reads_writes[slot] = access
        elif slot in input_slots:
            reads[slot] = access
        else:
            writes[slot] = access

    call = NKIOpCall(
        op_cls=op.op_cls, kwargs=dict(op.op_kwargs), axis_map=dict(op.axis_map), dim_role=dict(op.dim_role)
    )
    return SBlock(iter_vars=iter_vars, reads=reads, writes=writes, reads_writes=reads_writes, body=[call])


def _build_buffer_access(
    tname: str, axes: tuple[str, ...], op: _ParsedOp, dim_to_iv: dict[str, IterVar], module: KernelModule
) -> BufferAccess:
    """Produce a :class:`BufferAccess` for ``tname`` referenced by ``op`` via ``axes``.

    Each abstract axis maps to a concrete dim id via ``op.axis_map``;
    every such dim carries an iter var in ``dim_to_iv``. The per-dim
    :class:`AccessRange` has coefficient 1 on its iter var and extent
    equal to the dim's tile_size.
    """
    tensor = module.tensors[tname]
    iv_ids_seen: list[int] = []
    pattern_entries: list[AccessRange] = []
    for i, abstract in enumerate(axes):
        if i >= len(tensor.dim_ids):
            break
        dim_id = op.axis_map.get(abstract, tensor.dim_ids[i])
        iv = dim_to_iv.get(dim_id)
        if iv is None:
            """Dim untouched by this op — constant-offset access along this dim.
            Canonical build never hits this branch (every referenced dim is in
            ``touched_dims``), but the safety net keeps the function total."""
            extent = module.dims[dim_id].tile_size if dim_id in module.dims else tensor.shape[i]
            pattern_entries.append(AccessRange.make({}, 0, extent))
            continue
        if iv.var_id not in iv_ids_seen:
            iv_ids_seen.append(iv.var_id)
        pattern_entries.append(AccessRange.make({iv.var_id: 1}, 0, module.dims[dim_id].tile_size))
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
    """Build the iter-var schedule tree for ``op``: 1N ForNode per touched dim.

    If ``op.touched_dims`` is empty, returns the :class:`SBlock` directly
    (no loop nest needed). Otherwise wraps outermost-first: ``iter_vars[0]``
    is the outermost :class:`ForNode`.
    """
    block = _make_sblock(op, module)
    return _wrap_block_in_fornodes(block)


def _wrap_block_in_fornodes(block: SBlock) -> ForNode | SBlock:
    """Wrap ``block`` in one :class:`ForNode` per iter var, outermost first.

    Returns ``block`` unwrapped if it has no iter vars (alloc blocks).
    """
    if not block.iter_vars:
        return block
    node: ForNode | SBlock = block
    for iv in reversed(block.iter_vars):
        node = ForNode(iter_var=iv, children=[node])
    return node


def _assign_canonical_names(node: ForNode | SBlock, same_dim_counts: dict[str, int]) -> None:
    """Walk the tree in a root-outward DFS, naming each :class:`ForNode`.

    ``same_dim_counts[d]`` tracks how many same-dim ancestors of ``d``
    are already open on the current path; the newly visited node takes
    that as its ordinal, emits a name, then recurses with the counter
    incremented. Restoring the counter after recursion means siblings
    see the same counts the parent did (they are not each other's
    ancestors).
    """
    if isinstance(node, SBlock):
        return
    dim_id = node.iter_var.dim_id
    k = same_dim_counts.get(dim_id, 0)
    node.name = f"i_{dim_id}_{k}"
    same_dim_counts[dim_id] = k + 1
    for child in node.children:
        _assign_canonical_names(child, same_dim_counts)
    same_dim_counts[dim_id] = k


"""TreeIR re-exported only for callers that currently expect
``canonical.TreeIR`` — the type lives in :mod:`nkigym.codegen.ir`."""
_ = TreeIR
