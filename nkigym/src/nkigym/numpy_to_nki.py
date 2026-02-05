"""NumPy to NKI lowering module.

This module provides a lookup table for translating numpy operations
to NKI (Neuron Kernel Interface) kernel operations.

The lowering works by:
1. Parsing vanilla numpy source code via AST
2. Pattern matching AST nodes to operation types (e.g., subscript assignment -> "load")
3. Looking up the operation in NKI_OP_TABLE to generate equivalent NKI code
"""

import ast
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from nkigym.codegen import get_source


@dataclass
class NkiOpSemantics:
    """Defines how a numpy operation lowers to NKI code.

    Attributes:
        op_name: Name of the operation (e.g., "load", "store", "matmul").
        generate_nki: Function to generate NKI code string.
    """

    op_name: str
    generate_nki: Callable[..., str]


def _load_expr(dst_name: str, src_name: str, slices: list[tuple[int, int]]) -> str:
    """Generate NKI code for tensor load (HBM to SBUF).

    Translates numpy slice access to NKI dma_copy operation.
    Dtype is auto-detected from the source tensor.

    Args:
        dst_name: Name of the destination tensor variable.
        src_name: Name of the source tensor (HBM input).
        slices: List of (start, end) tuples for each dimension.

    Returns:
        NKI code string with ndarray allocation and dma_copy.

    Example:
        Input: tensor_0 = mat_a[0:128, 0:128]
        Output:
            tensor_0 = nl.ndarray(shape=(128, 128), dtype=mat_a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tensor_0, src=mat_a[0:128, 0:128])
    """
    shape = tuple(end - start for start, end in slices)
    slice_str = ", ".join(f"{start}:{end}" for start, end in slices)

    lines = [
        f"{dst_name} = nl.ndarray(shape={shape}, dtype={src_name}.dtype, buffer=nl.sbuf)",
        f"nisa.dma_copy(dst={dst_name}, src={src_name}[{slice_str}])",
    ]
    return "\n".join(lines)


def _alloc_output_expr(dst_name: str, shape: tuple[int, ...], dtype: str) -> str:
    """Generate NKI code for output tensor allocation in HBM.

    Translates np.empty to nl.ndarray with shared_hbm buffer.

    Args:
        dst_name: Name of the output tensor variable.
        shape: Shape tuple for the tensor.
        dtype: Data type string (e.g., "np.float32").

    Returns:
        NKI code string with ndarray allocation in shared HBM.

    Example:
        Input: output = np.empty((256, 512), dtype=np.float32)
        Output: output = nl.ndarray(shape=(256, 512), dtype=np.float32, buffer=nl.shared_hbm)
    """
    return f"{dst_name} = nl.ndarray(shape={shape}, dtype={dtype}, buffer=nl.shared_hbm)"


NKI_OP_TABLE: dict[str, NkiOpSemantics] = {
    "load": NkiOpSemantics(op_name="load", generate_nki=_load_expr),
    "alloc_output": NkiOpSemantics(op_name="alloc_output", generate_nki=_alloc_output_expr),
}


def _extract_slice_bounds(slice_node: ast.Slice) -> tuple[int, int]:
    """Extract (start, end) from an AST Slice node.

    Args:
        slice_node: AST Slice node with Constant lower/upper bounds.

    Returns:
        Tuple of (start, end) integers.

    Raises:
        ValueError: If slice bounds are not integer constants.
    """
    if not isinstance(slice_node.lower, ast.Constant) or not isinstance(slice_node.upper, ast.Constant):
        raise ValueError("Slice bounds must be integer constants")
    lower = slice_node.lower.value
    upper = slice_node.upper.value
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise ValueError("Slice bounds must be integers")
    return (lower, upper)


@dataclass
class LoadInfo:
    """Information extracted from a load operation AST node.

    Attributes:
        dst_name: Destination variable name.
        src_name: Source tensor name (HBM input).
        slices: List of (start, end) tuples for each dimension.
    """

    dst_name: str
    src_name: str
    slices: list[tuple[int, int]]


@dataclass
class AllocOutputInfo:
    """Information extracted from an output allocation AST node.

    Attributes:
        dst_name: Name of the output tensor variable.
        shape: Shape tuple for the tensor.
        dtype: Data type string.
    """

    dst_name: str
    shape: tuple[int, ...]
    dtype: str


def _is_load(node: ast.AST, input_names: set[str]) -> LoadInfo | None:
    """Check if an AST node is a load operation.

    A load operation is an assignment of a subscript on an input tensor:
        tensor_0 = mat_a[0:128, 0:128]

    Args:
        node: AST statement node.
        input_names: Set of input tensor variable names.

    Returns:
        LoadInfo if the node is a load operation, None otherwise.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None

    target = node.targets[0]
    value = node.value

    if not isinstance(target, ast.Name):
        return None
    if not isinstance(value, ast.Subscript):
        return None
    if not isinstance(value.value, ast.Name):
        return None
    if value.value.id not in input_names:
        return None

    slices: list[tuple[int, int]] = []
    if isinstance(value.slice, ast.Tuple):
        for elt in value.slice.elts:
            if isinstance(elt, ast.Slice):
                slices.append(_extract_slice_bounds(elt))
    elif isinstance(value.slice, ast.Slice):
        slices.append(_extract_slice_bounds(value.slice))

    return LoadInfo(dst_name=target.id, src_name=value.value.id, slices=slices)


def _is_alloc_output(node: ast.AST) -> AllocOutputInfo | None:
    """Check if an AST node is an output allocation operation.

    An output allocation is a call to np.empty:
        output = np.empty((256, 512), dtype=np.float32)

    Args:
        node: AST statement node.

    Returns:
        AllocOutputInfo if the node is an output allocation, None otherwise.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None

    target = node.targets[0]
    value = node.value

    if not isinstance(target, ast.Name):
        return None
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Attribute):
        return None
    if not isinstance(value.func.value, ast.Name):
        return None
    if value.func.value.id != "np" or value.func.attr != "empty":
        return None

    shape: list[int] = []
    if value.args and isinstance(value.args[0], ast.Tuple):
        for elt in value.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                shape.append(elt.value)

    dtype = "np.float32"
    for kw in value.keywords:
        if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
            if isinstance(kw.value.value, ast.Name):
                dtype = f"{kw.value.value.id}.{kw.value.attr}"

    return AllocOutputInfo(dst_name=target.id, shape=tuple(shape), dtype=dtype)


def lower_numpy_to_nki(func: Callable) -> str:
    """Lower a numpy function to NKI kernel code.

    Extracts source from the function, parses it, and translates operations
    to NKI equivalents:
    - Subscript assignments on input tensors become dma_copy loads

    Args:
        func: A numpy function to lower.

    Returns:
        NKI kernel code string.

    Example:
        >>> def my_func(mat_a):
        ...     tensor_0 = mat_a[0:128, 0:128]
        ...     return tensor_0
        >>> lower_numpy_to_nki(my_func)
        "tensor_0 = nl.ndarray(shape=(128, 128), dtype=mat_a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tensor_0, src=mat_a[0:128, 0:128])"
    """
    source = get_source(func)
    sig = inspect.signature(func)
    input_names = set(sig.parameters.keys())
    param_list = ", ".join(sig.parameters.keys())

    tree = ast.parse(source)
    body_lines: list[str] = []

    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")

    for node in func_def.body:
        load = _is_load(node, input_names)
        if load:
            load_op = NKI_OP_TABLE["load"]
            nki_code = load_op.generate_nki(load.dst_name, load.src_name, load.slices)
            body_lines.append(nki_code)
            continue

        alloc_output = _is_alloc_output(node)
        if alloc_output:
            alloc_op = NKI_OP_TABLE["alloc_output"]
            nki_code = alloc_op.generate_nki(alloc_output.dst_name, alloc_output.shape, alloc_output.dtype)
            body_lines.append(nki_code)
            continue

        warnings.warn(f"Unsupported operation: {ast.dump(node)}", stacklevel=2)
        break

    imports = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "import numpy as np"]
    header = ["", "", "@nki.jit", f"def nki_{func.__name__}({param_list}):"]
    indented_body = ["    " + line for line in "\n".join(body_lines).split("\n")]

    return "\n".join(imports + header + indented_body)
