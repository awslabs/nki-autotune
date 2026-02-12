"""Conversion utilities between callable, source, and IR representations.

Provides three conversion functions for the tuple-based IR:
- callable_to_ir: Parse a tiled callable into a program tuple.
- ir_to_source: Render a program tuple back to Python source.
- ir_to_callable: Compile a program tuple into a callable.

The program tuple format is:
    (name, params, stmts, return_var)

Each statement is a tuple: (op_instance, operands)
where operands is a tuple of (var_name, slices) pairs.
"""

import ast
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from nkigym.ops import (
    ALLOC_OPS,
    ELEMENTWISE_OP_NAMES,
    LOAD_OP,
    OP_REGISTRY,
    STORE_OP,
    AllocOp,
    ElementwiseOp,
    LoadOp,
    NKIOp,
    StoreOp,
)
from nkigym.utils.source import exec_source_to_func, get_source

Operand = tuple[str, tuple[tuple[int, int], ...]]
Statement = tuple[NKIOp, tuple[Operand, ...]]


def _fmt_stmt(stmt: Statement) -> str:
    """Format a statement as a short single-line string for repr."""
    op, operands = stmt
    vars_ = ", ".join(f"'{v}'" for v, _ in operands)
    return f"({op.op_name}, ({vars_}))"


class Program(NamedTuple):
    """Immutable program IR representation.

    Programs are hashable and used as dictionary keys for deduplication
    in the transform search graph.
    """

    name: str
    params: tuple[str, ...]
    stmts: tuple[Statement, ...]
    return_var: str
    preamble: str

    def __repr__(self) -> str:
        """Concise repr showing first/last stmts with elision."""
        stmts_lines = [_fmt_stmt(s) for s in self.stmts]
        show = 2
        if len(stmts_lines) > 2 * show + 1:
            head = stmts_lines[:show]
            tail = stmts_lines[-show:]
            middle = f"    ... {len(stmts_lines) - 2 * show} more ..."
            body = "\n".join([f"    {l}," for l in head] + [middle] + [f"    {l}," for l in tail])
        else:
            body = "\n".join(f"    {l}," for l in stmts_lines)
        return (
            f"Program({self.name!r}, params={self.params!r},\n"
            f"  stmts=[\n{body}\n  ],\n"
            f"  return_var={self.return_var!r})"
        )


def _parse_slices(node: ast.Subscript) -> tuple[tuple[int, int], ...]:
    """Extract slice bounds from a subscript expression.

    Args:
        node: AST Subscript node.

    Returns:
        Tuple of (start, stop) pairs for each dimension.

    Raises:
        ValueError: If slice bounds are not integer constants or if
            non-slice subscript types are encountered.
    """
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        for elt in slice_node.elts:
            if not isinstance(elt, ast.Slice):
                raise ValueError(f"Unsupported subscript type {type(elt).__name__}; only slices are supported")
        return tuple(_parse_one_slice(elt) for elt in slice_node.elts)
    if isinstance(slice_node, ast.Slice):
        return (_parse_one_slice(slice_node),)
    raise ValueError(f"Unexpected subscript slice type: {ast.dump(slice_node)}")


def _parse_one_slice(slice_node: ast.Slice) -> tuple[int, int]:
    """Extract (start, stop) from an AST Slice node.

    Args:
        slice_node: AST Slice node with constant bounds.

    Returns:
        Tuple of (start, stop) integers.

    Raises:
        ValueError: If slice bounds are not integer constants.
    """
    if not isinstance(slice_node.lower, ast.Constant) or not isinstance(slice_node.upper, ast.Constant):
        raise ValueError(f"Slice bounds must be integer constants, got {ast.dump(slice_node)}")
    lower = slice_node.lower.value
    upper = slice_node.upper.value
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise ValueError(f"Slice bounds must be integers, got {type(lower)} and {type(upper)}")
    return (lower, upper)


def _parse_operand(node: ast.expr) -> Operand:
    """Parse an AST expression into an Operand.

    Handles both Name (bare variable) and Subscript (sliced variable).

    Args:
        node: AST expression node.

    Returns:
        Operand tuple (var_name, slices).

    Raises:
        ValueError: If the expression type is not recognized.
    """
    if isinstance(node, ast.Name):
        return (node.id, ())
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        var_name = node.value.id
        slices = _parse_slices(node)
        return (var_name, slices)
    raise ValueError(f"Unsupported argument expression: {ast.dump(node)}")


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Create full-range slices from a shape tuple.

    Args:
        shape: Shape tuple.

    Returns:
        Tuple of (0, size) pairs for each dimension.
    """
    return tuple((0, s) for s in shape)


def _shape_from_slices(slices: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """Derive shape from slice bounds.

    Args:
        slices: Tuple of (start, stop) pairs.

    Returns:
        Shape tuple computed as (stop - start) for each dimension.
    """
    return tuple(stop - start for start, stop in slices)


def _extract_shape_from_tuple(node: ast.Tuple) -> tuple[int, ...]:
    """Extract shape tuple from an AST Tuple of constants.

    Args:
        node: AST Tuple node with constant integer elements.

    Returns:
        Shape tuple.

    Raises:
        ValueError: If any element is not a constant integer.
    """
    shape: list[int] = []
    for elt in node.elts:
        if not isinstance(elt, ast.Constant):
            raise ValueError(f"Shape element must be a constant, got {type(elt).__name__}")
        if not isinstance(elt.value, int):
            raise ValueError(f"Shape element must be an integer, got {type(elt.value).__name__}")
        shape.append(elt.value)
    return tuple(shape)


def _extract_dtype_name(call_node: ast.Call) -> str:
    """Extract the dtype name from an nkigym.ndarray call.

    Args:
        call_node: AST Call node for nkigym.ndarray(...).

    Returns:
        Dtype name string (e.g., "float64").
    """
    for kw in call_node.keywords:
        if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
            return kw.value.attr
    return "float32"


def _is_nkigym_call(call_node: ast.Call) -> bool:
    """Check if a call is to nkigym.<something>.

    Args:
        call_node: AST Call node.

    Returns:
        True if the call is to nkigym.<attr>.
    """
    return (
        isinstance(call_node.func, ast.Attribute)
        and isinstance(call_node.func.value, ast.Name)
        and call_node.func.value.id == "nkigym"
    )


def _get_nkigym_attr(call_node: ast.Call) -> str:
    """Get the attribute name from a nkigym.<attr>() call.

    Args:
        call_node: AST Call node for nkigym.<attr>().

    Returns:
        The attribute name string.
    """
    return call_node.func.attr


def _extract_preamble(source: str) -> str:
    """Extract the function def line and optional docstring from source.

    The preamble includes the ``def`` line (with type annotations if present)
    and the docstring if one exists, preserving original formatting.

    Args:
        source: Full Python source code of the function.

    Returns:
        Preamble string containing the def line and optional docstring.
    """
    lines = source.splitlines()
    preamble_lines: list[str] = []
    def_idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            preamble_lines.append(line)
            def_idx = idx
            break

    rest = lines[def_idx + 1 :]
    if rest:
        first_body = rest[0].strip()
        for quote in ('"""', "'''"):
            if first_body.startswith(quote):
                preamble_lines.append(rest[0])
                if first_body.endswith(quote) and len(first_body) > len(quote):
                    break
                for line in rest[1:]:
                    preamble_lines.append(line)
                    if line.strip().endswith(quote):
                        break
                break

    return "\n".join(preamble_lines)


def callable_to_ir(func: Callable) -> Program:
    """Parse a tiled callable into a program tuple.

    This is the only function that calls ast.parse(). All subsequent
    operations work on the tuple representation directly.

    Args:
        func: A tiled callable with __source__ attribute or inspectable source.

    Returns:
        Program tuple: (name, params, stmts, return_var, preamble).

    Raises:
        ValueError: If the source cannot be parsed or the function
            structure is not recognized.
    """
    source = get_source(func)
    tree = ast.parse(source)
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        raise ValueError("Expected a function definition")

    name = func_def.name
    params = tuple(arg.arg for arg in func_def.args.args)
    input_names = set(params)
    preamble = _extract_preamble(source)

    stmts: list[Statement] = []
    return_var: str = "output"
    var_shapes: dict[str, tuple[int, ...]] = {}

    for node in func_def.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue

        stmt = _parse_ast_stmt(node, input_names, var_shapes)
        if stmt is not None:
            stmts.append(stmt)
            _record_var_shapes(stmt, var_shapes)
            continue

        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            return_var = node.value.id
            continue

        if isinstance(node, ast.AugAssign):
            aug_stmt = _parse_aug_assign(node, var_shapes)
            if aug_stmt is not None:
                stmts.append(aug_stmt)
                continue

        raise ValueError(f"Unsupported statement: {ast.dump(node)}")

    return Program(name, params, tuple(stmts), return_var, preamble)


def _record_var_shapes(stmt: Statement, var_shapes: dict[str, tuple[int, ...]]) -> None:
    """Record variable shapes defined by a statement.

    Updates var_shapes with the destination variable's shape derived from
    its slices in the statement.

    Args:
        stmt: Parsed statement tuple.
        var_shapes: Mutable map from variable name to shape, updated in place.
    """
    op, operands = stmt
    if isinstance(op, AllocOp):
        var_name, slices = operands[0]
        var_shapes[var_name] = _shape_from_slices(slices)
    elif isinstance(op, LoadOp):
        dst_var, dst_slices = operands[1]
        var_shapes[dst_var] = _shape_from_slices(dst_slices)
    elif not isinstance(op, StoreOp):
        dst_var, dst_slices = operands[-1]
        var_shapes[dst_var] = _shape_from_slices(dst_slices)


def _parse_ast_stmt(node: ast.stmt, input_names: set[str], var_shapes: dict[str, tuple[int, ...]]) -> Statement | None:
    """Parse a single AST statement into a Statement tuple.

    Args:
        node: AST statement node.
        input_names: Set of input parameter names.
        var_shapes: Map from variable name to known shape.

    Returns:
        Statement tuple or None if this node type is not handled here.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None

    target = node.targets[0]
    value = node.value

    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        if isinstance(value, ast.Call) and _is_nkigym_call(value):
            func_name = _get_nkigym_attr(value)
            if func_name != "ndarray":
                return _parse_compute_store(target, func_name, value, var_shapes)
        return _parse_store(target, value)

    if not isinstance(target, ast.Name):
        return None

    dst_name = target.id

    if isinstance(value, ast.Call) and _is_nkigym_call(value):
        func_name = _get_nkigym_attr(value)
        if func_name == "ndarray":
            return _parse_alloc(dst_name, value)
        return _parse_call_assign(dst_name, func_name, value, var_shapes)

    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        src_name = value.value.id
        if src_name in input_names:
            return _parse_load(dst_name, value)

    return None


def _parse_alloc(dst_name: str, call_node: ast.Call) -> Statement:
    """Parse an nkigym.ndarray allocation into an alloc Statement.

    Args:
        dst_name: Destination variable name.
        call_node: AST Call node for nkigym.ndarray(...).

    Returns:
        Statement tuple with AllocOp.
    """
    shape: tuple[int, ...] = ()
    if call_node.args and isinstance(call_node.args[0], ast.Tuple):
        shape = _extract_shape_from_tuple(call_node.args[0])

    dtype_name = _extract_dtype_name(call_node)
    if dtype_name not in ALLOC_OPS:
        raise ValueError(f"Unsupported alloc dtype: {dtype_name}")
    alloc_op = ALLOC_OPS[dtype_name]
    slices = _full_slices(shape)
    operand: Operand = (dst_name, slices)
    return (alloc_op, (operand,))


def _parse_load(dst_name: str, subscript: ast.Subscript) -> Statement:
    """Parse a load (input subscript) into a load Statement.

    Args:
        dst_name: Destination variable name.
        subscript: AST Subscript node for input[slices].

    Returns:
        Statement tuple with LOAD_OP.
    """
    src_name = subscript.value.id
    src_slices = _parse_slices(subscript)
    dst_shape = _shape_from_slices(src_slices)
    dst_slices = _full_slices(dst_shape)
    src_operand: Operand = (src_name, src_slices)
    dst_operand: Operand = (dst_name, dst_slices)
    return (LOAD_OP, (src_operand, dst_operand))


def _resolve_operand_shape(operand: Operand, var_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    """Get the shape of an operand, resolving bare names via var_shapes.

    Args:
        operand: Operand tuple (var_name, slices).
        var_shapes: Map from variable name to known shape.

    Returns:
        Shape tuple for the operand.

    Raises:
        ValueError: If a bare-name operand has no known shape.
    """
    var_name, slices = operand
    if slices:
        return _shape_from_slices(slices)
    if var_name in var_shapes:
        return var_shapes[var_name]
    raise ValueError(f"Cannot determine shape for bare variable '{var_name}'")


def _resolve_op(op_name: str, call_node: ast.Call) -> NKIOp:
    """Resolve an operation name to an NKIOp instance.

    Checks OP_REGISTRY first, then creates an ElementwiseOp for known
    elementwise ops with keyword arguments.

    Args:
        op_name: Operation name (e.g., "nc_matmul", "tensor_tensor").
        call_node: AST Call node (used to extract kwargs for elementwise ops).

    Returns:
        NKIOp instance for the operation.

    Raises:
        ValueError: If op_name is not recognized.
    """
    if op_name in OP_REGISTRY:
        return OP_REGISTRY[op_name]

    if op_name in ELEMENTWISE_OP_NAMES:
        kwargs_repr = _extract_kwargs_repr(call_node)
        return ElementwiseOp(op_name, kwargs_repr)

    raise ValueError(f"Unknown operation '{op_name}' not in OP_REGISTRY")


def _extract_kwargs_repr(call_node: ast.Call) -> tuple[tuple[str, str], ...]:
    """Extract keyword arguments from an AST Call as repr strings.

    Args:
        call_node: AST Call node.

    Returns:
        Sorted tuple of (key, repr_string) pairs.
    """
    pairs: list[tuple[str, str]] = []
    for kw in call_node.keywords:
        if kw.arg is not None:
            pairs.append((kw.arg, _kwarg_value_to_repr(kw.value)))
    return tuple(sorted(pairs))


def _kwarg_value_to_repr(node: ast.expr) -> str:
    """Convert an AST expression to a source repr string.

    Handles common patterns: np.func, constants.

    Args:
        node: AST expression node.

    Returns:
        Source code string for the value.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return ast.unparse(node)


def _parse_call_assign(
    dst_name: str, op_name: str, call_node: ast.Call, var_shapes: dict[str, tuple[int, ...]]
) -> Statement:
    """Parse a compute op call into a compute Statement.

    Args:
        dst_name: Destination variable name.
        op_name: Operation name (e.g., "nc_matmul").
        call_node: AST Call node for nkigym.<op>(...).
        var_shapes: Map from variable name to known shape.

    Returns:
        Statement tuple with the corresponding NKIOp.

    Raises:
        ValueError: If op_name is not in OP_REGISTRY or elementwise ops.
    """
    op = _resolve_op(op_name, call_node)
    operands: list[Operand] = []

    for arg in call_node.args:
        operands.append(_parse_operand(arg))

    input_shapes = [_resolve_operand_shape(o, var_shapes) for o in operands]
    dst_shape = op.output_shape(input_shapes)
    dst_slices = _full_slices(dst_shape)
    operands.append((dst_name, dst_slices))

    return (op, tuple(operands))


def _parse_compute_store(
    target: ast.Subscript, op_name: str, call_node: ast.Call, var_shapes: dict[str, tuple[int, ...]]
) -> Statement:
    """Parse a combined compute+store: output[slices] = nkigym.op(args).

    The destination operand uses the store target's variable name and slices.

    Args:
        target: AST Subscript node for the store target (e.g., output[0:128, 0:128]).
        op_name: Operation name (e.g., "nc_matmul").
        call_node: AST Call node for nkigym.<op>(...).
        var_shapes: Map from variable name to known shape.

    Returns:
        Statement tuple with the corresponding NKIOp.

    Raises:
        ValueError: If op_name is not in OP_REGISTRY.
    """
    op = _resolve_op(op_name, call_node)
    operands: list[Operand] = []

    for arg in call_node.args:
        operands.append(_parse_operand(arg))

    dst_name = target.value.id
    dst_slices = _parse_slices(target)
    operands.append((dst_name, dst_slices))

    return (op, tuple(operands))


def _parse_store(target: ast.Subscript, value: ast.expr) -> Statement:
    """Parse a store (output subscript assignment) into a store Statement.

    Args:
        target: AST Subscript node for output[slices].
        value: AST expression for the source value.

    Returns:
        Statement tuple with STORE_OP.
    """
    dst_name = target.value.id
    dst_slices = _parse_slices(target)
    src_operand = _parse_operand(value)
    dst_operand: Operand = (dst_name, dst_slices)
    return (STORE_OP, (src_operand, dst_operand))


def _parse_aug_assign(node: ast.AugAssign, var_shapes: dict[str, tuple[int, ...]]) -> Statement | None:
    """Parse an augmented assignment (+=) into a compute Statement.

    For reduction tiling, accumulate is represented as:
        tensor_2[0:128, 0:128] += nkigym.nc_matmul(a, b)

    This becomes a compute Statement where the dst operand uses the
    accumulator's slices.

    Args:
        node: AST AugAssign node.
        var_shapes: Map from variable name to known shape.

    Returns:
        Statement tuple or None if not a recognized pattern.
    """
    if not isinstance(node.op, ast.Add):
        return None

    value = node.value
    if not isinstance(value, ast.Call) or not _is_nkigym_call(value):
        return None

    op_name = _get_nkigym_attr(value)
    if op_name not in OP_REGISTRY and op_name not in ELEMENTWISE_OP_NAMES:
        return None

    op = _resolve_op(op_name, value)
    operands: list[Operand] = []

    for arg in value.args:
        operands.append(_parse_operand(arg))

    if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
        dst_name = node.target.value.id
        dst_slices = _parse_slices(node.target)
    elif isinstance(node.target, ast.Name):
        dst_name = node.target.id
        input_shapes = [_resolve_operand_shape(o, var_shapes) for o in operands]
        dst_shape = op.output_shape(input_shapes)
        dst_slices = _full_slices(dst_shape)
    else:
        return None

    operands.append((dst_name, dst_slices))
    return (op, tuple(operands))


def ir_to_source(program: Program) -> str:
    """Render a program tuple back to Python source code.

    Args:
        program: Program tuple (name, params, stmts, return_var, preamble).

    Returns:
        Python source code string for the program.
    """
    name, params, stmts, return_var, preamble = program
    lines: list[str] = ["import numpy as np", "import nkigym"]
    if preamble:
        lines.extend(preamble.splitlines())
    else:
        lines.append(f"def {name}({', '.join(params)}):")

    allocated_vars: set[str] = set()
    defined_compute_vars: set[str] = set()

    for op, operands in stmts:
        line = _stmt_to_source(op, operands, allocated_vars, defined_compute_vars)
        lines.append(f"    {line}")

        if isinstance(op, AllocOp):
            allocated_vars.add(operands[0][0])
        elif not isinstance(op, (LoadOp, StoreOp)):
            dst_var = operands[-1][0]
            defined_compute_vars.add(dst_var)

    lines.append(f"    return {return_var}")
    return "\n".join(lines) + "\n"


def _stmt_to_source(
    op: NKIOp, operands: tuple[Operand, ...], allocated_vars: set[str], defined_compute_vars: set[str]
) -> str:
    """Render a single statement to source code.

    Args:
        op: The NKIOp instance for this statement.
        operands: Tuple of (var_name, slices) operand pairs.
        allocated_vars: Set of variable names defined by alloc statements.
        defined_compute_vars: Set of variable names already defined by
            previous compute statements. Used to detect accumulations.

    Returns:
        Source code string for the statement.
    """
    if isinstance(op, AllocOp):
        var_name, slices = operands[0]
        shape = _shape_from_slices(slices)
        dtype_name = np.dtype(op.dtype).name
        return f"{var_name} = nkigym.ndarray({shape}, dtype=np.{dtype_name})"

    if isinstance(op, LoadOp):
        src_var, src_slices = operands[0]
        dst_var, _ = operands[1]
        slice_str = _slices_to_str(src_slices)
        return f"{dst_var} = {src_var}[{slice_str}]"

    if isinstance(op, StoreOp):
        src_var, src_slices = operands[0]
        dst_var, dst_slices = operands[1]
        src_str = _operand_to_slice_str(src_var, src_slices)
        dst_slice_str = _slices_to_str(dst_slices)
        return f"{dst_var}[{dst_slice_str}] = {src_str}"

    dst_var, dst_slices = operands[-1]
    input_operands = operands[:-1]

    input_strs = [_operand_to_slice_str(var, sl) for var, sl in input_operands]
    kwargs_str = ""
    if isinstance(op, ElementwiseOp) and op.kwargs_repr:
        kwargs_parts = [f"{k}={v}" for k, v in op.kwargs_repr]
        kwargs_str = ", " + ", ".join(kwargs_parts)
    expr = f"nkigym.{op.op_name}({', '.join(input_strs)}{kwargs_str})"

    is_compute_store = dst_var in allocated_vars
    is_accumulate = dst_var in defined_compute_vars and not is_compute_store

    if is_accumulate:
        dst_str = _operand_to_slice_str(dst_var, dst_slices)
        return f"{dst_str} += {expr}"

    if is_compute_store:
        dst_str = _operand_to_slice_str(dst_var, dst_slices)
        return f"{dst_str} = {expr}"

    return f"{dst_var} = {expr}"


def _operand_to_slice_str(var: str, slices: tuple[tuple[int, int], ...]) -> str:
    """Render an operand to source code.

    Args:
        var: Variable name.
        slices: Slice bounds.

    Returns:
        Source string like "var[0:128, 0:128]" or just "var" if no slices.
    """
    if not slices:
        return var
    return f"{var}[{_slices_to_str(slices)}]"


def _slices_to_str(slices: tuple[tuple[int, int], ...]) -> str:
    """Render slice bounds to a comma-separated string.

    Args:
        slices: Tuple of (start, stop) pairs.

    Returns:
        String like "0:128, 0:128".
    """
    return ", ".join(f"{start}:{stop}" for start, stop in slices)


def ir_to_callable(program: Program) -> Callable[..., np.ndarray]:
    """Compile a program tuple into a callable function.

    Args:
        program: Program tuple (name, params, stmts, return_var, preamble).

    Returns:
        Callable function with __source__ attribute.
    """
    source = ir_to_source(program)
    return exec_source_to_func(source, program.name)
