"""Canonical module builder: ``f_nkigym`` callable → :class:`KernelModule`.

Pipeline:
    1. AST-parse the math function to an ordered list of raw parsed ops.
    2. Unify abstract axes (``P``, ``F``, ``K``, ``M``, ``N`` ...) across
       ops into concrete dim ids (``d0``, ``d1`` ...).
    3. Derive per-dim total size + tile size from ``input_specs`` and
       per-op ``TILE_LIMITS``.
    4. Tag each tensor's ``origin`` (``param`` / ``intermediate`` / ``return``).
    5. Build the canonical 1N-per-dim forest with phase leaves at the
       deepest point; populate every :class:`BodyLeaf` with FULL metadata
       so leaves are self-describing (no back-reference to a sidecar).
       Tiles-per-block arithmetic-intensity dials live in ``Split``, not
       canonical form.
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

from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, LoopNode, Tensor, TreeIR
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

    body: TreeIR = []
    """Alloc leaves are emitted at the forest root in source order (the
    user's declaration order). Compute/copy leaves follow, each with their
    own schedule tree built from touched_dims."""
    for alloc in allocs:
        alloc_leaf = _make_alloc_leaf(alloc)
        body.append(alloc_leaf)
    for op in parsed_ops:
        body.append(_build_tree(op, dims))
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
class _AllocRecord:
    """Canonical-build-time record for an NKIAlloc call in f_nkigym.

    Captures the tensor identity (name, location, shape, dtype) that
    populates ``module.tensors``. The resulting ``BodyLeaf`` carries
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
    from nkigym.ops.alloc import NKIAlloc

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


def _make_leaf(op: _ParsedOp) -> BodyLeaf:
    """Build a self-describing :class:`BodyLeaf` for ``op``.

    ``RMW_OPERANDS`` and ``INPUT_OPERANDS`` on the op class split operand
    slots into three buckets:
    - ``INPUT_OPERANDS`` (and not RMW) → ``reads``.
    - ``RMW_OPERANDS`` → ``reads_writes``.
    - Otherwise (e.g. ``dst``, ``reduce_res``) → ``writes``.
    """
    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads: dict[str, str] = {}
    writes_list: list[str] = []
    reads_writes_list: list[str] = []
    for slot, tname in op.operand_names.items():
        if slot in rmw:
            reads_writes_list.append(tname)
        elif slot in input_slots:
            reads[slot] = tname
        else:
            writes_list.append(tname)
    return BodyLeaf(
        op_cls=op.op_cls,
        reads=reads,
        writes=tuple(writes_list),
        reads_writes=tuple(reads_writes_list),
        kwargs=dict(op.op_kwargs),
        axis_map=dict(op.axis_map),
        dim_role=dict(op.dim_role),
    )


def _make_alloc_leaf(alloc: _AllocRecord) -> BodyLeaf:
    """Build the single BodyLeaf for an NKIAlloc declaration.

    The leaf's only kwarg is ``tensor_name`` — shape/dtype/location
    live in ``module.tensors[name]``.
    """
    from nkigym.ops.alloc import NKIAlloc

    return BodyLeaf(op_cls=NKIAlloc, reads={}, writes=(alloc.name,), kwargs={"tensor_name": alloc.name})


def _build_tree(op: _ParsedOp, dims: dict[str, DimInfo]) -> LoopNode | BodyLeaf:
    """Build the 1N-per-dim chain for ``op`` with a leaf at the tip.

    If op has no touched_dims (empty abstract axes), returns the leaf directly."""
    deepest_children = _build_leaves(op, dims)
    if not op.touched_dims:
        """Ops with no operand axes (or scalar-only ops) have no loop nest."""
        assert len(deepest_children) == 1
        return deepest_children[0]
    return _wrap_dims(op.touched_dims, op, dims, deepest_children)


def _wrap_dims(
    wrap: tuple[str, ...], op: _ParsedOp, dims: dict[str, DimInfo], inner_children: list[LoopNode | BodyLeaf]
) -> LoopNode:
    """Wrap ``inner_children`` in a 1N-per-dim chain over ``wrap``.

    Each dim contributes one ``LoopNode`` with ``trip_count = num_tiles``.
    The outer wrapper matches the dim ordering in ``touched_dims``
    (outermost first). Arithmetic-intensity dials (tiles-per-block)
    live in ``Split``, not here.
    """
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
    """Single-phase default: every op emits one ``BodyLeaf``."""
    _ = dims
    return [_make_leaf(op)]


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
