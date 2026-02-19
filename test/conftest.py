"""Shared test utilities and fixtures for pytest."""

import ast
from collections.abc import Callable

import numpy as np
import pytest

import nkigym
from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import GymOp
from nkigym.utils import callable_to_source


@pytest.fixture
def matmul_func() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Fixture providing a standard matmul function for testing.

    Returns a function that computes matrix multiplication using nkigym.nc_matmul.
    This fixture reduces code duplication across test files by providing
    a consistent matmul implementation.

    Returns:
        A callable that takes two numpy arrays and returns their matrix product.
    """

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute matrix multiplication.

        Args:
            a: First input matrix of shape (K, M).
            b: Second input matrix of shape (K, N).

        Returns:
            Matrix product of shape (M, N).
        """
        return nkigym.nc_matmul(a, b)

    return matmul


@pytest.fixture
def double_matmul_func() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Fixture providing a standard double matmul function for testing.

    Returns a function that computes double matrix multiplication using nkigym.nc_matmul.
    This fixture reduces code duplication across test files by providing
    a consistent double matmul implementation.

    Returns:
        A callable that takes three numpy arrays and returns their chained matrix product.
    """

    def double_matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute double matrix multiplication.

        Args:
            a: First input matrix of shape (K1, M).
            b: Second input matrix of shape (K1, K2).
            c: Third input matrix of shape (K2, N).

        Returns:
            Matrix product of shape (M, N).
        """
        return nkigym.nc_matmul(nkigym.nc_matmul(a, b), c)

    return double_matmul


def normalize_source(source: str) -> str:
    """Normalize source code for comparison.

    Strips leading/trailing whitespace from each line, removes blank lines,
    and joins with single newlines.

    Args:
        source: Source code string.

    Returns:
        Normalized source string.
    """
    lines = [line.strip() for line in source.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def make_random_array(shape: tuple[int, ...], seed: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Generate a deterministic random array for testing.

    Args:
        shape: Shape of the array to generate.
        seed: Random seed for reproducibility.
        dtype: Data type for the array.

    Returns:
        Random array with values in [-1, 1] range.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Tuple of (0, dim_size) pairs for each dimension.
    """
    return tuple((0, s) for s in shape)


def _ref(name: str, shape: tuple[int, ...]) -> TensorRef:
    """Build a TensorRef with full-range slices.

    Args:
        name: Variable name.
        shape: Tensor shape tuple.

    Returns:
        TensorRef with full-range slices.
    """
    return TensorRef(name, shape, _full_slices(shape))


def _slice_ref(name: str, shape: tuple[int, ...], slices: tuple[tuple[int, int], ...]) -> TensorRef:
    """Build a TensorRef with explicit slices.

    Args:
        name: Variable name.
        shape: Tensor shape tuple.
        slices: Per-axis (start, stop) bounds.

    Returns:
        TensorRef with explicit slices.
    """
    return TensorRef(name, shape, slices)


def _parse_one_slice(node: ast.expr) -> tuple[int, int]:
    """Parse a single AST slice node into a (start, stop) pair.

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


def _expr_to_str(node: ast.expr) -> str:
    """Convert an AST expression to its string representation.

    Args:
        node: AST expression node.

    Returns:
        String representation of the expression.

    Raises:
        ValueError: If node type is unsupported.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    raise ValueError(f"Unsupported: {ast.dump(node)}")


def _golden_to_program(
    golden_func: object, params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]], output_dtype: type
) -> GymProgram:
    """Convert a golden fixture function to a GymProgram by parsing its source.

    The golden functions are pre-tiled with explicit np_slice/np_store/np_empty,
    so we parse them by inspecting the source and building GymStatements.

    Args:
        golden_func: A golden fixture function.
        params: Parameter names.
        input_shapes: Parameter shapes.
        output_dtype: Output dtype.

    Returns:
        GymProgram representation.
    """
    source = callable_to_source(golden_func)
    tree = ast.parse(source)
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        raise ValueError("No function found")

    name = func_def.name
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
                            shape_node = node.value.args[0]
                            shape = tuple(elt.value for elt in shape_node.elts)
                            dtype_kw = None
                            for kw in node.value.keywords:
                                if kw.arg == "dtype":
                                    dtype_kw = f"{kw.value.value.id}.{kw.value.attr}"
                            out_ref = TensorRef(target.id, shape, _full_slices(shape))
                            stmts.append(GymStatement("np_empty", (("dtype", dtype_kw),), out_ref))
                            var_shapes[target.id] = shape
                            continue

                        if func_node.value.id == "nkigym":
                            op_name = func_node.attr
                            op_cls = GymOp.get(op_name)
                            n_inputs = len(op_cls.inputs)
                            operand_names = tuple(t.name for t in op_cls.inputs)

                            kwargs_list: list[tuple[str, object]] = []
                            input_shape_list: list[tuple[int, ...]] = []

                            for arg_idx, arg in enumerate(node.value.args):
                                op_key = operand_names[arg_idx] if arg_idx < len(operand_names) else f"arg{arg_idx}"
                                if isinstance(arg, ast.Subscript):
                                    var_name = arg.value.id
                                    slices = _parse_subscript_slices(arg.slice)
                                    tile_shape = tuple(e - s for s, e in slices)
                                    kwargs_list.append(
                                        (op_key, TensorRef(var_name, var_shapes.get(var_name, tile_shape), slices))
                                    )
                                    input_shape_list.append(tile_shape)
                                elif isinstance(arg, ast.Name):
                                    var_name = arg.id
                                    shape = var_shapes.get(var_name, ())
                                    kwargs_list.append((op_key, _ref(var_name, shape)))
                                    input_shape_list.append(shape)

                            for kw in node.value.keywords:
                                kwargs_list.append((kw.arg, _expr_to_str(kw.value)))

                            out_shape = op_cls().output_shape(tuple(input_shape_list))
                            out_ref = TensorRef(target.id, out_shape, _full_slices(out_shape))
                            stmts.append(GymStatement(op_name, tuple(kwargs_list), out_ref))
                            var_shapes[target.id] = out_shape
                            continue

                if isinstance(node.value, ast.Subscript):
                    var_name = node.value.value.id
                    slices = _parse_subscript_slices(node.value.slice)
                    tile_shape = tuple(e - s for s, e in slices)
                    src_shape = var_shapes.get(var_name, tuple(e for _, e in slices))
                    src_ref = TensorRef(var_name, src_shape, slices)
                    dst_ref = TensorRef(target.id, tile_shape, _full_slices(tile_shape))
                    stmts.append(GymStatement("np_slice", (("src", src_ref),), dst_ref))
                    var_shapes[target.id] = tile_shape
                    continue

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                dst_name = target.value.id
                dst_slices = _parse_subscript_slices(target.slice)
                dst_shape = var_shapes.get(dst_name, tuple(e for _, e in dst_slices))

                if isinstance(node.value, ast.Name):
                    src_name = node.value.id
                    src_shape = var_shapes.get(src_name, ())
                    src_ref = TensorRef(src_name, src_shape, _full_slices(src_shape))
                elif isinstance(node.value, ast.Subscript):
                    src_name = node.value.value.id
                    src_slices = _parse_subscript_slices(node.value.slice)
                    src_shape = var_shapes.get(src_name, tuple(e - s for s, e in src_slices))
                    src_ref = TensorRef(src_name, src_shape, src_slices)
                else:
                    continue

                dst_ref = TensorRef(dst_name, dst_shape, dst_slices)
                stmts.append(GymStatement("np_store", (("src", src_ref), ("dst", dst_ref)), dst_ref))
                continue

    return GymProgram(
        name=name,
        params=params,
        input_shapes=tuple((p, input_shapes[p]) for p in params),
        stmts=tuple(stmts),
        return_var=return_var,
        output_dtype=output_dtype,
    )
