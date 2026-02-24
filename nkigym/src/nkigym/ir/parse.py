"""GymProgram IR: source parsing and source rendering.

Provides bidirectional translation between Python source code using nkigym
ops and the GymProgram IR:

- ``source_to_program``: parse source AST into a specialized GymProgram
- ``program_to_source``: render a GymProgram back to Python source
"""

import ast
from typing import Any

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.ops.base import GymOp


def source_to_program(source: str, input_shapes: dict[str, tuple[int, ...]], output_dtype: type) -> GymProgram:
    """Parse nkigym source code into a specialized GymProgram.

    Handles both high-level source (``nkigym.<op>(a, b)`` calls with shape
    inference) and tiled source (explicit ``np.empty``, subscript slicing,
    ``np_store`` assignments produced by ``program_to_source``).

    Args:
        source: Python source code containing a function using nkigym ops.
        input_shapes: Mapping from parameter names to shape tuples.
        output_dtype: Numpy dtype type for output allocation.

    Returns:
        Specialized GymProgram with TensorRef on all tensor references.

    Raises:
        ValueError: If the function structure is not recognized.
    """
    func_def = _find_func_def(source)
    if _has_np_empty(func_def):
        return _parse_tiled_body(func_def, input_shapes, output_dtype)
    return _parse_highlevel_body(func_def, input_shapes, output_dtype)


def program_to_source(program: GymProgram) -> str:
    """Render a GymProgram as Python source code.

    Each statement is rendered directly from its TensorRef with no
    cross-statement shape tracking.

    Args:
        program: A GymProgram.

    Returns:
        Complete Python source code string with imports.
    """
    lines: list[str] = ["import numpy as np", "import nkigym"]

    params_str = ", ".join(program.params)
    lines.append(f"def {program.name}({params_str}):")

    for stmt in program.stmts:
        if stmt.op == "np_empty":
            lines.append(_render_np_empty(stmt))
        elif stmt.op == "np_slice":
            lines.append(_render_np_slice(stmt))
        elif stmt.op == "np_store":
            lines.append(_render_np_store(stmt))
        else:
            lines.append(_render_compute(stmt))

    lines.append(f"    return {program.return_var}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _find_func_def(source: str) -> ast.FunctionDef:
    """Find the first FunctionDef in parsed source.

    Args:
        source: Python source code string.

    Returns:
        The AST FunctionDef node.

    Raises:
        ValueError: If no function definition is found.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("Expected a function definition")


def _has_np_empty(func_def: ast.FunctionDef) -> bool:
    """Check whether a function body contains an ``np.empty`` call.

    Args:
        func_def: The parsed AST FunctionDef node.

    Returns:
        True if any statement assigns from ``np.empty(...)``.
    """
    for node in func_def.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "np"
                    and call.func.attr == "empty"
                ):
                    return True
    return False


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Per-axis (0, size) bounds.
    """
    return tuple((0, s) for s in shape)


def _parse_one_slice(node: ast.expr) -> tuple[int, int]:
    """Parse a single AST Slice node into a (start, stop) pair.

    Args:
        node: AST Slice node.

    Returns:
        Tuple of (start, stop) integers.

    Raises:
        ValueError: If node is not an ast.Slice.
    """
    if isinstance(node, ast.Slice):
        lower = node.lower.value if isinstance(node.lower, ast.Constant) else 0
        upper = node.upper.value if isinstance(node.upper, ast.Constant) else 0
        return (lower, upper)
    raise ValueError(f"Expected slice, got {ast.dump(node)}")


def _parse_subscript_slices(node: ast.expr) -> tuple[tuple[int, int], ...]:
    """Parse AST subscript slices into (start, stop) pairs.

    Args:
        node: AST Tuple of slices or single slice.

    Returns:
        Tuple of (start, stop) pairs for each dimension.
    """
    if isinstance(node, ast.Tuple):
        return tuple(_parse_one_slice(elt) for elt in node.elts)
    return (_parse_one_slice(node),)


def _parse_arg_ref(node: ast.expr, var_shapes: dict[str, tuple[int, ...]]) -> TensorRef:
    """Parse a compute arg (Name or Subscript) into a TensorRef.

    Args:
        node: AST expression (Name or Subscript).
        var_shapes: Variable shape tracking dict.

    Returns:
        TensorRef with appropriate shape and slices.

    Raises:
        ValueError: If the node type is not recognized.
    """
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        slices = _parse_subscript_slices(node.slice)
        tile_shape = tuple(e - s for s, e in slices)
        return TensorRef(node.value.id, tile_shape, slices)
    if isinstance(node, ast.Name):
        shape = var_shapes.get(node.id, ())
        return TensorRef(node.id, shape, _full_slices(shape))
    raise ValueError(f"Expected Name or Subscript, got {ast.dump(node)}")


def _specialize(
    stmts: list[GymStatement], params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]]
) -> list[GymStatement]:
    """Specialize statements with concrete shapes.

    Walks the statements, infers shapes through ``op.output_shape()``,
    and replaces tensor kwargs values and outputs with TensorRef carrying
    full-range slices.

    Args:
        stmts: Parsed statements with string kwargs and placeholder outputs.
        params: Function parameter names.
        input_shapes: Mapping from parameter names to shape tuples.

    Returns:
        Specialized statements with TensorRef on all tensor references.
    """
    var_shapes: dict[str, tuple[int, ...]] = {}
    for param_name in params:
        var_shapes[param_name] = input_shapes[param_name]

    result: list[GymStatement] = []
    for stmt in stmts:
        op_cls = GymOp.get(stmt.op)
        n_inputs = len(op_cls.inputs)

        tensor_kwargs = stmt.kwargs[:n_inputs]
        config_kwargs = stmt.kwargs[n_inputs:]

        new_kwargs: list[tuple[str, Any]] = []
        input_shapes: list[tuple[int, ...]] = []
        for operand_name, var_name in tensor_kwargs:
            shape = var_shapes[var_name]
            input_shapes.append(shape)
            new_kwargs.append((operand_name, TensorRef(var_name, shape, _full_slices(shape))))

        for key, value in config_kwargs:
            if isinstance(value, str) and value in var_shapes:
                shape = var_shapes[value]
                new_kwargs.append((key, TensorRef(value, shape, _full_slices(shape))))
            else:
                new_kwargs.append((key, value))

        out_shape = op_cls().output_shape(tuple(input_shapes))
        out_name = stmt.output.name
        var_shapes[out_name] = out_shape

        result.append(
            GymStatement(
                op=stmt.op, kwargs=tuple(new_kwargs), output=TensorRef(out_name, out_shape, _full_slices(out_shape))
            )
        )

    return result


def _flatten_call(call: ast.Call, output: str, stmts: list[GymStatement], counter: list[int]) -> None:
    """Flatten a potentially nested ``nkigym.<op>(...)`` call into GymStatements.

    Recursively processes nested nkigym calls, introducing intermediate
    variables for inner calls before emitting the outer call.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.
        stmts: Accumulator list for emitted statements.
        counter: Single-element list used as a mutable counter for
            generating unique intermediate variable names.

    Raises:
        ValueError: If the op is not registered or arg count mismatches.
    """
    op_name = call.func.attr
    op_cls = GymOp.get(op_name)
    operand_names = tuple(t.name for t in op_cls.inputs)

    resolved_args: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Call) and _is_nkigym_call(arg):
            tmp_name = f"_nested_{counter[0]}"
            counter[0] += 1
            _flatten_call(arg, tmp_name, stmts, counter)
            resolved_args.append(tmp_name)
        else:
            resolved_args.append(_arg_name(arg))

    if len(resolved_args) != len(operand_names):
        raise ValueError(
            f"Op {op_name!r} expects {len(operand_names)} positional args "
            f"({', '.join(operand_names)}), got {len(resolved_args)}"
        )

    kwargs: list[tuple[str, str]] = []
    for operand, var in zip(operand_names, resolved_args):
        kwargs.append((operand, var))

    for kw in call.keywords:
        if isinstance(kw.value, (ast.Name, ast.Subscript)):
            kwargs.append((kw.arg, _arg_name(kw.value)))
        else:
            kwargs.append((kw.arg, _expr_to_str(kw.value)))

    stmts.append(GymStatement(op=op_name, kwargs=tuple(kwargs), output=TensorRef(output, (), ())))


def _parse_call(call: ast.Call, output: str) -> list[GymStatement]:
    """Parse a ``nkigym.<op>(...)`` call into GymStatements.

    Handles nested nkigym calls by flattening them into a sequence of
    statements with intermediate variables.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.

    Returns:
        List of GymStatements (one for simple calls, multiple for nested).

    Raises:
        ValueError: If the op is not registered or arg count mismatches.
    """
    stmts: list[GymStatement] = []
    counter: list[int] = [0]
    _flatten_call(call, output, stmts, counter)
    return stmts


def _is_nkigym_call(call: ast.Call) -> bool:
    """Check if a call is to ``nkigym.<op>``.

    Args:
        call: AST Call node.

    Returns:
        True if the call is to ``nkigym.<attr>``.
    """
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "nkigym"
    )


def _arg_name(node: ast.expr) -> str:
    """Extract variable name from an AST expression.

    Handles both plain names (``a``) and subscripted names (``a[0:128, 0:128]``),
    returning just the base variable name in either case.

    Args:
        node: AST expression node.

    Returns:
        Variable name string.

    Raises:
        ValueError: If the expression is not a Name or Subscript of a Name.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id
    raise ValueError(f"Expected a variable name, got {ast.dump(node)}")


def _expr_to_str(node: ast.expr) -> str:
    """Convert an AST expression to its source string.

    Handles dotted names (e.g., ``np.tanh``), simple names, and
    numeric/string constants.

    Args:
        node: AST expression node.

    Returns:
        Source-level string representation.

    Raises:
        ValueError: If the expression type is not supported.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    raise ValueError(f"Unsupported kwarg expression: {ast.dump(node)}")


def _subscript(ref: TensorRef) -> str:
    """Render a TensorRef as a subscripted variable.

    Scalar tensors (empty slices) are rendered as plain variable names.

    Args:
        ref: Tensor reference with name and slices.

    Returns:
        String like ``tensor_0[0:128, 0:128]`` or ``scalar`` for scalars.
    """
    if not ref.slices:
        return ref.name
    slices = ", ".join(f"{s}:{e}" for s, e in ref.slices)
    return f"{ref.name}[{slices}]"


def _render_np_empty(stmt: GymStatement) -> str:
    """Render an np_empty statement.

    Args:
        stmt: The np_empty GymStatement.

    Returns:
        Source line like ``output = np.empty((128, 128), dtype=np.float32)``.
    """
    dtype = ""
    for key, value in stmt.kwargs:
        if key == "dtype":
            dtype = value
    if not dtype:
        raise ValueError(f"np_empty statement for '{stmt.output.name}' missing 'dtype' kwarg")
    shape_str = ", ".join(str(s) for s in stmt.output.shape)
    return f"    {stmt.output.name} = np.empty(({shape_str}), dtype={dtype})"


def _render_np_slice(stmt: GymStatement) -> str:
    """Render an np_slice statement.

    Args:
        stmt: The np_slice GymStatement.

    Returns:
        Source line like ``tensor_0 = a[0:128, 0:128]``.
    """
    src = None
    for key, value in stmt.kwargs:
        if key == "src":
            src = value
    if src is None:
        raise ValueError(f"np_slice statement for '{stmt.output.name}' missing 'src' kwarg")
    slices = ", ".join(f"{s}:{e}" for s, e in src.slices)
    return f"    {stmt.output.name} = {src.name}[{slices}]"


def _render_np_store(stmt: GymStatement) -> str:
    """Render an np_store statement.

    Args:
        stmt: The np_store GymStatement.

    Returns:
        Source line like ``output[0:128, 0:128] = tensor_2[0:128, 0:128]``.
    """
    src = None
    dst = None
    for key, value in stmt.kwargs:
        if key == "src":
            src = value
        elif key == "dst":
            dst = value
    if src is None:
        raise ValueError(f"np_store statement for '{stmt.output.name}' missing 'src' kwarg")
    if dst is None:
        raise ValueError(f"np_store statement for '{stmt.output.name}' missing 'dst' kwarg")
    return f"    {_subscript(dst)} = {_subscript(src)}"


def _render_compute(stmt: GymStatement) -> str:
    """Render a compute GymStatement.

    Args:
        stmt: The compute GymStatement.

    Returns:
        Source line like ``tensor_2 = nkigym.nc_matmul(...)``.
    """
    args: list[str] = []
    for key, value in stmt.kwargs:
        if key == "acc":
            args.append(f"acc={_subscript(value)}")
        elif isinstance(value, TensorRef):
            args.append(_subscript(value))
        else:
            args.append(f"{key}={value}")

    args_str = ", ".join(args)
    return f"    {stmt.output.name} = nkigym.{stmt.op}({args_str})"


def _parse_highlevel_body(
    func_def: ast.FunctionDef, input_shapes: dict[str, tuple[int, ...]], output_dtype: type
) -> GymProgram:
    """Parse a high-level function body into a GymProgram.

    High-level source contains ``nkigym.<op>(...)`` calls with plain variable
    args and return statements. Shapes are inferred via ``_specialize()``.

    Args:
        func_def: The parsed AST FunctionDef node.
        input_shapes: Mapping from parameter names to shape tuples.
        output_dtype: Numpy dtype type for output allocation.

    Returns:
        Specialized GymProgram with TensorRef on all tensor references.

    Raises:
        ValueError: If a statement cannot be parsed.
    """
    name = func_def.name
    params = tuple(arg.arg for arg in func_def.args.args)

    stmts: list[GymStatement] = []
    return_var = ""

    for node in func_def.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                call = node.value
                if _is_nkigym_call(call):
                    stmts.extend(_parse_call(call, target.id))
                    continue

        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return_var = node.value.id
                continue
            if isinstance(node.value, ast.Call) and _is_nkigym_call(node.value):
                stmts.extend(_parse_call(node.value, "_return"))
                return_var = "_return"
                continue

        raise ValueError(f"Unsupported statement: {ast.dump(node)}")

    if not return_var:
        raise ValueError("Function must have a return statement")

    specialized = _specialize(stmts, params, input_shapes)

    return GymProgram(
        name=name,
        params=params,
        input_shapes=tuple((p, input_shapes[p]) for p in params),
        stmts=tuple(specialized),
        return_var=return_var,
        output_dtype=output_dtype,
    )


def _parse_tiled_body(
    func_def: ast.FunctionDef, input_shapes: dict[str, tuple[int, ...]], output_dtype: type
) -> GymProgram:
    """Parse a tiled function body into a GymProgram.

    Tiled source contains explicit ``np.empty``, subscript slicing,
    ``nkigym`` compute ops with subscripted args (including ``acc=`` kwargs
    for accumulation), and ``np_store`` assignments. Shapes are read directly
    from subscript bounds rather than inferred.

    Args:
        func_def: The parsed AST FunctionDef node.
        input_shapes: Mapping from parameter names to shape tuples.
        output_dtype: Numpy dtype type for output allocation.

    Returns:
        GymProgram with TensorRef on all tensor references.

    Raises:
        ValueError: If a statement cannot be parsed.
    """
    name = func_def.name
    params = tuple(arg.arg for arg in func_def.args.args)

    stmts: list[GymStatement] = []
    return_var = ""
    var_shapes: dict[str, tuple[int, ...]] = {}
    for p in params:
        var_shapes[p] = input_shapes[p]

    for node in func_def.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue

        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return_var = node.value.id
            continue

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, ast.Name):
                if isinstance(node.value, ast.Call):
                    func_node = node.value.func
                    if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Name):
                        if func_node.value.id == "np" and func_node.attr == "empty":
                            _parse_tiled_np_empty(target.id, node.value, var_shapes, stmts)
                            continue
                        if func_node.value.id == "nkigym":
                            _parse_tiled_compute(target.id, node.value, var_shapes, stmts)
                            continue

                if isinstance(node.value, ast.Subscript):
                    _parse_tiled_slice(target.id, node.value, var_shapes, stmts)
                    continue

            if isinstance(target, ast.Subscript):
                _parse_tiled_store(target, node.value, var_shapes, stmts)
                continue

        raise ValueError(f"Unsupported tiled statement: {ast.dump(node)}")

    if not return_var:
        raise ValueError("Function must have a return statement")

    return GymProgram(
        name=name,
        params=params,
        input_shapes=tuple((p, input_shapes[p]) for p in params),
        stmts=tuple(stmts),
        return_var=return_var,
        output_dtype=output_dtype,
    )


def _parse_tiled_np_empty(
    target_name: str, call: ast.Call, var_shapes: dict[str, tuple[int, ...]], stmts: list[GymStatement]
) -> None:
    """Parse an ``np.empty(...)`` call into a GymStatement.

    Args:
        target_name: Variable name being assigned.
        call: AST Call node for ``np.empty(...)``.
        var_shapes: Mutable dict tracking variable shapes.
        stmts: Mutable list of statements to append to.

    Raises:
        ValueError: If the dtype kwarg is missing.
    """
    shape = tuple(elt.value for elt in call.args[0].elts)
    dtype_str = ""
    for kw in call.keywords:
        if kw.arg == "dtype":
            dtype_str = _expr_to_str(kw.value)
    if not dtype_str:
        raise ValueError(f"np.empty for '{target_name}' missing dtype kwarg")
    out_ref = TensorRef(target_name, shape, _full_slices(shape))
    stmts.append(GymStatement("np_empty", (("dtype", dtype_str),), out_ref))
    var_shapes[target_name] = shape


def _parse_tiled_slice(
    target_name: str, subscript: ast.Subscript, var_shapes: dict[str, tuple[int, ...]], stmts: list[GymStatement]
) -> None:
    """Parse a subscript assignment (np_slice) into a GymStatement.

    Args:
        target_name: Variable name being assigned.
        subscript: AST Subscript node (``src[slices]``).
        var_shapes: Mutable dict tracking variable shapes.
        stmts: Mutable list of statements to append to.
    """
    src_name = subscript.value.id
    slices = _parse_subscript_slices(subscript.slice)
    tile_shape = tuple(e - s for s, e in slices)
    src_shape = var_shapes.get(src_name, tuple(e for _, e in slices))
    src_ref = TensorRef(src_name, src_shape, slices)
    dst_ref = TensorRef(target_name, tile_shape, _full_slices(tile_shape))
    stmts.append(GymStatement("np_slice", (("src", src_ref),), dst_ref))
    var_shapes[target_name] = tile_shape


def _parse_tiled_compute(
    target_name: str, call: ast.Call, var_shapes: dict[str, tuple[int, ...]], stmts: list[GymStatement]
) -> None:
    """Parse a ``nkigym.<op>(...)`` call with explicit slices into a GymStatement.

    Args:
        target_name: Variable name being assigned.
        call: AST Call node for ``nkigym.<op>(...)``.
        var_shapes: Mutable dict tracking variable shapes.
        stmts: Mutable list of statements to append to.
    """
    op_name = call.func.attr
    op_cls = GymOp.get(op_name)
    operand_names = tuple(t.name for t in op_cls.inputs)

    kwargs_list: list[tuple[str, object]] = []
    input_shape_list: list[tuple[int, ...]] = []

    for arg_idx, arg in enumerate(call.args):
        op_key = operand_names[arg_idx] if arg_idx < len(operand_names) else f"arg{arg_idx}"
        ref = _parse_arg_ref(arg, var_shapes)
        kwargs_list.append((op_key, ref))
        input_shape_list.append(ref.shape)

    for kw in call.keywords:
        if kw.arg == "acc":
            acc_ref = _parse_arg_ref(kw.value, var_shapes)
            kwargs_list.append(("acc", acc_ref))
        else:
            kwargs_list.append((kw.arg, _expr_to_str(kw.value)))

    out_shape = op_cls().output_shape(tuple(input_shape_list))
    out_ref = TensorRef(target_name, out_shape, _full_slices(out_shape))
    stmts.append(GymStatement(op_name, tuple(kwargs_list), out_ref))
    var_shapes[target_name] = out_shape


def _parse_tiled_store(
    target: ast.Subscript, value: ast.expr, var_shapes: dict[str, tuple[int, ...]], stmts: list[GymStatement]
) -> None:
    """Parse a subscripted assignment (np_store) into a GymStatement.

    Args:
        target: AST Subscript node for the destination (``dst[slices]``).
        value: AST expression for the source (name or ``name[slices]``).
        var_shapes: Mutable dict tracking variable shapes.
        stmts: Mutable list of statements to append to.

    Raises:
        ValueError: If the value expression type is not recognized.
    """
    dst_name = target.value.id
    dst_slices = _parse_subscript_slices(target.slice)
    dst_shape = var_shapes.get(dst_name, tuple(e for _, e in dst_slices))

    if isinstance(value, ast.Name):
        src_shape = var_shapes.get(value.id, ())
        src_ref = TensorRef(value.id, src_shape, _full_slices(src_shape))
    elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        src_slices = _parse_subscript_slices(value.slice)
        src_shape = tuple(e - s for s, e in src_slices)
        src_ref = TensorRef(value.value.id, src_shape, src_slices)
    else:
        raise ValueError(f"Unsupported np_store source: {ast.dump(value)}")

    dst_ref = TensorRef(dst_name, dst_shape, dst_slices)
    stmts.append(GymStatement("np_store", (("src", src_ref), ("dst", dst_ref)), dst_ref))
