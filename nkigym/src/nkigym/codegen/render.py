"""Kernel renderer: inspects a math function and emits NKI kernel source.

Handles single-op and multi-op pipelines with inter-op buffer
detection: outputs consumed by later ops get full-range SBUF
buffers; only the final output gets degree-1 SBUF + DMA store.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext, Tensor
from nkigym.ops.matmul import MATMUL_FREE_MAX
from nkigym.ops.transpose import TRANSPOSE_BLOCK

_INDENT = "    "

_UNASSIGNED = (ast.Expr, ast.Return)

_ISA_TILE_LIMITS: dict[str, dict[str, int]] = {
    "nc_matmul": {"K": 128, "M": 128, "N": MATMUL_FREE_MAX},
    "nc_transpose": {"P": TRANSPOSE_BLOCK, "F": TRANSPOSE_BLOCK},
}


def _extract_op_call(stmt: ast.stmt) -> tuple[ast.Call | None, str | None]:
    """Extract the OpClass()(kwargs) call and output name from a statement.

    Returns ``(call_node, output_name)``.  *output_name* is ``None`` for
    unassigned calls (bare expression / return).
    """
    call_node: ast.Call | None = None
    output_name: str | None = None

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            output_name = target.id
    elif isinstance(stmt, _UNASSIGNED) and isinstance(stmt.value, ast.Call):
        call_node = stmt.value

    return call_node, output_name


def _resolve_op_class(call_node: ast.Call, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Resolve the NKIOp subclass from an ``OpClass()(...)`` call node."""
    cls: type[NKIOp] | None = None
    if isinstance(call_node.func, ast.Call) and isinstance(call_node.func.func, ast.Name):
        candidate = func_globals.get(call_node.func.func.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            cls = candidate
    return cls


def _find_ops(func: Callable[..., np.ndarray]) -> tuple[list[tuple[NKIOp, dict[str, str], str]], str]:
    """Find NKIOp subclasses and the return variable via AST inspection.

    Args:
        func: Math function using nkigym ops.

    Returns:
        ``(ops, return_name)`` where *ops* is a list of
        ``(NKIOp instance, operand_map, output_name)`` in source order.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    func_body = func_def.body
    ops: list[tuple[NKIOp, dict[str, str], str]] = []
    return_name: str | None = None
    for stmt in func_body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id

        call_node, output_name = _extract_op_call(stmt)
        if call_node is None:
            continue
        cls = _resolve_op_class(call_node, func.__globals__)
        if cls is None:
            continue
        if output_name is None:
            raise ValueError(f"Op {cls.NAME!r} result must be assigned to a variable")
        operand_map = {
            kw.arg: kw.value.id for kw in call_node.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)
        }
        ops.append((cls(), operand_map, output_name))

    if return_name is None:
        raise ValueError("Math function must have a 'return <variable>' statement")
    return ops, return_name


def _emit_header(
    func_name: str, param_names: list[str], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> list[str]:
    """Emit imports, decorator, signature, and input checks."""
    lines = [
        "import nki",
        "import nki.language as nl",
        "import nki.isa as nisa",
        "from nki.backends.mlir_tracer.tensor import Tensor",
        "",
        "",
        "@nki.jit",
        f"def {func_name}_kernel({', '.join(f'{n}: Tensor' for n in param_names)}):",
    ]
    for name in param_names:
        shape, dtype_str = _parse_input_spec(input_specs[name])
        shape_str = ", ".join(str(s) for s in shape)
        lines.append(f"{_INDENT}assert {name}.shape == ({shape_str})")
        lines.append(f"{_INDENT}assert {name}.dtype == nl.{dtype_str}")
    return lines


def _resolve_output(
    op: NKIOp, operand_map: dict[str, str], ctx: RenderContext
) -> tuple[tuple[int, ...], str, tuple[str, ...]]:
    """Trace output shape, dtype, and dim_ids from an op's operand axes.

    Returns:
        ``(output_shape, dtype_str, dim_ids)``.
    """
    axis_sizes: dict[str, int] = {}
    first_dtype: str | None = None
    for slot, axes in op.OPERAND_AXES.items():
        tensor = ctx.tensors[operand_map[slot]]
        if first_dtype is None:
            first_dtype = tensor.dtype
        for axis_label, size in zip(axes, tensor.shape, strict=True):
            axis_sizes[axis_label] = size

    if first_dtype is None:
        raise ValueError(f"Op {op.NAME!r} has no input operands")
    if len(op.OUTPUT_AXES) != 1:
        raise ValueError(f"Op {op.NAME!r} has {len(op.OUTPUT_AXES)} outputs — multi-output not supported")

    output_axes = next(iter(op.OUTPUT_AXES.values()))
    output_shape = tuple(axis_sizes[a] for a in output_axes)
    return output_shape, first_dtype, output_axes


def _consumed_names(ops: list[tuple[NKIOp, dict[str, str], str]]) -> set[str]:
    """Collect all tensor names consumed as operands by any op."""
    names: set[str] = set()
    for _, operand_map, _ in ops:
        names.update(operand_map.values())
    return names


def _dim_size(ctx: RenderContext, dim_id: str) -> int | None:
    """Return the size of *dim_id* from the first tensor that carries it."""
    result: int | None = next(
        (s for t in ctx.tensors.values() if t.dim_ids for d, s in zip(t.dim_ids, t.shape, strict=True) if d == dim_id),
        None,
    )
    return result


def _unify_dim(
    op: NKIOp, ctx: RenderContext, per_op_maps: list[dict[str, str]], abstract: str, old_id: str, new_id: str
) -> None:
    """Validate sizes match, then rename *old_id* → *new_id* everywhere."""
    old_size = _dim_size(ctx, old_id)
    new_size = _dim_size(ctx, new_id)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(
            f"Op {op.NAME!r}: axis {abstract!r} unifies"
            f" {old_id!r} (size {old_size}) with"
            f" {new_id!r} (size {new_size}) — size mismatch"
        )
    for tensor in ctx.tensors.values():
        if tensor.dim_ids and old_id in tensor.dim_ids:
            tensor.dim_ids = tuple(new_id if d == old_id else d for d in tensor.dim_ids)
    for axis_map in per_op_maps:
        for ax, concrete in axis_map.items():
            if concrete == old_id:
                axis_map[ax] = new_id


def _local_axis_map(
    op: NKIOp,
    operand_map: dict[str, str],
    ctx: RenderContext,
    dim_counter: list[int],
    per_op_maps: list[dict[str, str]],
) -> dict[str, str]:
    """Build a per-op-invocation abstract→concrete axis mapping.

    Abstract axes (P, F, K, M, N) are scoped per op invocation.
    Operands that already have dim_ids provide concrete mappings;
    operands without dim_ids get fresh d0, d1, ... IDs.

    When two operands map the same abstract axis to different
    concrete dims, the later one is unified into the earlier one
    by renaming across all tensors in ctx and prior per_op_maps.
    """
    local: dict[str, str] = {}
    for slot, axes in op.OPERAND_AXES.items():
        tensor = ctx.tensors[operand_map[slot]]
        if tensor.dim_ids:
            for abstract, concrete in zip(axes, tensor.dim_ids, strict=True):
                if abstract in local and local[abstract] != concrete:
                    _unify_dim(op, ctx, per_op_maps, abstract, old_id=concrete, new_id=local[abstract])
                else:
                    local[abstract] = concrete
        else:
            for abstract in axes:
                if abstract not in local:
                    local[abstract] = f"d{dim_counter[0]}"
                    dim_counter[0] += 1
            tensor.dim_ids = tuple(local[a] for a in axes)
    return local


def _process_ops(
    ops: list[tuple[NKIOp, dict[str, str], str]], ctx: RenderContext, return_name: str
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int]]:
    """Single forward pass: assign dim IDs, create output tensors, unify tiles.

    Each op gets its own abstract→concrete axis mapping (scoped to
    that invocation), so two transposes on different tensors get
    different concrete dim IDs.

    Returns:
        ``(per_op_axis_maps, dim_tiles, dim_min_tiles)``
    """
    dim_counter = [0]
    consumed = _consumed_names(ops)
    per_op_maps: list[dict[str, str]] = []

    for op, operand_map, output_name in ops:
        local = _local_axis_map(op, operand_map, ctx, dim_counter, per_op_maps)
        per_op_maps.append(local)
        _create_output_tensor(op, operand_map, output_name, ctx, local, return_name, consumed)

    dim_tiles, dim_min_tiles = _unify_tile_sizes(ops, per_op_maps, ctx)
    return per_op_maps, dim_tiles, dim_min_tiles


def _create_output_tensor(
    op: NKIOp,
    operand_map: dict[str, str],
    output_name: str,
    ctx: RenderContext,
    local_axis_map: dict[str, str],
    return_name: str,
    consumed: set[str],
) -> None:
    """Create the output tensor for one op and add it to ctx."""
    output_shape, output_dtype, output_axes = _resolve_output(op, operand_map, ctx)
    dim_ids = tuple(local_axis_map[a] for a in output_axes)
    is_interop = output_name != return_name and output_name in consumed
    location = "sbuf" if is_interop else "hbm"
    ctx.tensors[output_name] = Tensor(
        name=output_name, shape=output_shape, dtype=output_dtype, location=location, dim_ids=dim_ids
    )


def _collect_dim_sizes(ctx: RenderContext) -> dict[str, int]:
    """Collect actual dimension sizes from all tensors in ctx."""
    sizes: dict[str, int] = {}
    for tensor in ctx.tensors.values():
        for dim_id, size in zip(tensor.dim_ids, tensor.shape, strict=True):
            sizes[dim_id] = size
    return sizes


def _unify_tile_sizes(
    ops: list[tuple[NKIOp, dict[str, str], str]], per_op_maps: list[dict[str, str]], ctx: RenderContext
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute unified and min tile sizes across all ops.

    For each concrete dimension, takes the max of all ISA limits
    (unified) and the min of all ISA limits (min), both capped
    at the actual dimension size.

    Returns:
        ``(dim_tiles, dim_min_tiles)``
    """
    dim_tiles: dict[str, int] = {}
    dim_min_tiles: dict[str, int] = {}
    for (op, _, _), local in zip(ops, per_op_maps, strict=True):
        limits = _ISA_TILE_LIMITS.get(op.NAME, {})
        for abstract_axis, limit in limits.items():
            dim_id = local[abstract_axis]
            dim_tiles[dim_id] = max(dim_tiles.get(dim_id, limit), limit)
            dim_min_tiles[dim_id] = min(dim_min_tiles.get(dim_id, limit), limit)

    dim_sizes = _collect_dim_sizes(ctx)
    for dim_id in dim_tiles:
        dim_tiles[dim_id] = min(dim_tiles[dim_id], dim_sizes[dim_id])
    for dim_id in dim_min_tiles:
        dim_min_tiles[dim_id] = min(dim_min_tiles[dim_id], dim_sizes[dim_id])
    return dim_tiles, dim_min_tiles


def _parse_input_spec(spec: tuple) -> tuple[tuple[int, ...], str]:
    """Parse an input spec into (shape, dtype)."""
    return spec[0], spec[1]


@dataclass
class KernelIR:
    """Intermediate representation of a kernel variant.

    Carries structured state (ops, axis maps, context) plus
    transform-specific configuration (fusion groups, etc.).
    ``build_ir()`` creates the base IR with each op in its own
    singleton fusion group; transforms produce modified copies.

    Attributes:
        ops: Op instances as (NKIOp, operand_map, output_name).
        per_op_maps: Per-op abstract-to-concrete axis mappings.
        ctx: Render context with tensors and tile sizes.
        return_name: Name of the final output tensor.
        func_name: Original math function name.
        param_names: Ordered parameter names for the kernel signature.
        input_specs: Maps parameter name to (shape, dtype_str).
        fusion_groups: Which ops share loop nests, each inner
            list contains op indices.
    """

    ops: list[tuple[NKIOp, dict[str, str], str]]
    per_op_maps: list[dict[str, str]]
    ctx: RenderContext
    return_name: str
    func_name: str
    param_names: list[str]
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    fusion_groups: list[list[int]]


def build_ir(func: Callable[..., np.ndarray], input_specs: dict[str, tuple]) -> KernelIR:
    """Lower a math function to a base KernelIR.

    Parses the function to discover ops, runs dimension analysis,
    computes tile sizes, and wraps everything in a KernelIR with
    each op in its own singleton fusion group.

    Args:
        func: Math function using nkigym ops.
        input_specs: Maps each parameter name to ``(shape, dtype_str)``.

    Returns:
        Base KernelIR ready for rendering or transforms.
    """
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    ctx = RenderContext()
    for name in param_names:
        shape, dtype_str = _parse_input_spec(input_specs[name])
        ctx.tensors[name] = Tensor(name=name, shape=shape, dtype=dtype_str, location="hbm")

    ops, return_name = _find_ops(func)
    per_op_maps, dim_tiles, dim_min_tiles = _process_ops(ops, ctx, return_name)
    ctx.dim_tiles = dim_tiles
    ctx.dim_min_tiles = dim_min_tiles

    fusion_groups = [[i] for i in range(len(ops))]

    return KernelIR(
        ops=ops,
        per_op_maps=per_op_maps,
        ctx=ctx,
        return_name=return_name,
        func_name=func.__name__,
        param_names=param_names,
        input_specs=input_specs,
        fusion_groups=fusion_groups,
    )


def render_ir(ir: KernelIR) -> str:
    """Render a KernelIR to NKI kernel source string.

    For the base IR each op sits in its own singleton fusion group,
    producing a self-contained code block per op. Transforms modify
    the fusion groups; render_ir handles both base and transformed IRs.

    Args:
        ir: Kernel IR from ``build_ir`` or after transforms.

    Returns:
        Complete NKI kernel source string.
    """
    return_tensor = ir.ctx.tensors[ir.return_name]
    shape_str = ", ".join(str(s) for s in return_tensor.shape)

    lines = _emit_header(ir.func_name, ir.param_names, ir.input_specs)
    lines.append(
        f"{_INDENT}hbm_{ir.return_name} = nl.ndarray(({shape_str}),"
        f" dtype=nl.{return_tensor.dtype}, buffer=nl.shared_hbm)"
    )

    for group in ir.fusion_groups:
        for op_idx in group:
            op, operand_map, output_name = ir.ops[op_idx]
            is_final = output_name == ir.return_name
            for line in op.render(ir.ctx, operand_map, output_name, is_final):
                lines.append(f"{_INDENT}{line}")

    lines.append(f"{_INDENT}return hbm_{ir.return_name}")
    return "\n".join(lines)


def render(func: Callable[..., np.ndarray], input_specs: dict[str, tuple]) -> str:
    """Inspect a math function and generate NKI kernel source.

    Convenience wrapper: calls ``build_ir`` then ``render_ir``.

    Args:
        func: Math function using nkigym ops.
        input_specs: Maps each parameter name to ``(shape, dtype_str)``.

    Returns:
        Complete NKI kernel source string.
    """
    ir = build_ir(func, input_specs)
    return render_ir(ir)
