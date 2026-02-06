"""NKIGym to NKI lowering module.

This module provides lowering from nkigym intermediate representation
to NKI (Neuron Kernel Interface) kernel code. The lowering is nearly
1:1, translating each nkigym operation to its NKI equivalent with
explicit buffer management (HBM, SBUF, PSUM).

The lowering works by:
1. Parsing nkigym source code via AST
2. Pattern matching AST nodes to operation types
3. Using OP_REGISTRY's generate_nki() methods for compute operations
4. Generating explicit buffer allocations and DMA copies for loads/stores

To support a new operator in lowering, ensure its NkiOp.generate_nki()
method is implemented and the operator is registered in OP_REGISTRY.
"""

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass

from nkigym.ops import OP_REGISTRY
from nkigym.utils.source import get_source


@dataclass
class OutputAllocInfo:
    """Information extracted from nkigym.ndarray output allocation.

    Attributes:
        dst_name: Name of the output tensor variable.
        shape: Shape tuple for the tensor.
        dtype: Data type string (e.g., "np.float32").
    """

    dst_name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class LoadInfo:
    """Information extracted from a load operation (input subscript).

    Attributes:
        dst_name: Destination variable name.
        src_name: Source tensor name (HBM input).
        slices: List of (start, end) tuples for each dimension.
    """

    dst_name: str
    src_name: str
    slices: list[tuple[int, int]]


@dataclass
class ComputeInfo:
    """Information extracted from a compute operation (nkigym.nc_matmul etc).

    Attributes:
        dst_name: Output variable name.
        op_name: Operation name (e.g., "nc_matmul").
        inputs: List of input variable names.
    """

    dst_name: str
    op_name: str
    inputs: list[str]


@dataclass
class AccumulateInfo:
    """Information extracted from an accumulation operation (+=).

    Attributes:
        target_name: Name of the accumulator variable.
        op_name: Operation name (e.g., "nc_matmul").
        inputs: List of input variable names.
    """

    target_name: str
    op_name: str
    inputs: list[str]


@dataclass
class StoreInfo:
    """Information extracted from a store operation (output subscript assignment).

    Attributes:
        dst_name: Output tensor name.
        src_name: Source variable name.
        slices: List of (start, end) tuples for each dimension.
    """

    dst_name: str
    src_name: str
    slices: list[tuple[int, int]]


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


def _extract_subscript_slices(slice_node: ast.expr) -> list[tuple[int, int]]:
    """Extract slice bounds from a subscript slice expression.

    Args:
        slice_node: AST slice expression (Tuple of Slices or single Slice).

    Returns:
        List of (start, end) tuples for each dimension.
    """
    slices: list[tuple[int, int]] = []
    if isinstance(slice_node, ast.Tuple):
        for elt in slice_node.elts:
            if isinstance(elt, ast.Slice):
                slices.append(_extract_slice_bounds(elt))
    elif isinstance(slice_node, ast.Slice):
        slices.append(_extract_slice_bounds(slice_node))
    return slices


def _is_output_alloc(node: ast.AST) -> OutputAllocInfo | None:
    """Check if an AST node is an nkigym.ndarray output allocation.

    Pattern: dst = nkigym.ndarray((M, N), dtype=np.float32)

    Args:
        node: AST statement node.

    Returns:
        OutputAllocInfo if the node is an output allocation, None otherwise.
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
    if value.func.value.id != "nkigym" or value.func.attr != "ndarray":
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

    return OutputAllocInfo(dst_name=target.id, shape=tuple(shape), dtype=dtype)


def _is_load(node: ast.AST, input_names: set[str]) -> LoadInfo | None:
    """Check if an AST node is a load operation (subscript on input tensor).

    Pattern: dst = input[start:end, start:end]

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

    slices = _extract_subscript_slices(value.slice)
    return LoadInfo(dst_name=target.id, src_name=value.value.id, slices=slices)


def _is_compute(node: ast.AST) -> ComputeInfo | None:
    """Check if an AST node is a compute operation (nkigym op call).

    Pattern: dst = nkigym.op_name(input1, input2, ...)

    Args:
        node: AST statement node.

    Returns:
        ComputeInfo if the node is a compute operation, None otherwise.
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
    if value.func.value.id != "nkigym":
        return None

    op_name = value.func.attr
    if op_name == "ndarray":
        return None

    inputs = []
    for arg in value.args:
        if isinstance(arg, ast.Name):
            inputs.append(arg.id)

    return ComputeInfo(dst_name=target.id, op_name=op_name, inputs=inputs)


def _is_accumulate(node: ast.AST) -> AccumulateInfo | None:
    """Check if an AST node is an accumulation operation (+=).

    Pattern: target += nkigym.op_name(input1, input2, ...)

    Args:
        node: AST statement node.

    Returns:
        AccumulateInfo if the node is an accumulation, None otherwise.
    """
    if not isinstance(node, ast.AugAssign):
        return None
    if not isinstance(node.op, ast.Add):
        return None
    if not isinstance(node.target, ast.Name):
        return None

    value = node.value
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Attribute):
        return None
    if not isinstance(value.func.value, ast.Name):
        return None
    if value.func.value.id != "nkigym":
        return None

    op_name = value.func.attr
    inputs = []
    for arg in value.args:
        if isinstance(arg, ast.Name):
            inputs.append(arg.id)

    return AccumulateInfo(target_name=node.target.id, op_name=op_name, inputs=inputs)


def _is_store(node: ast.AST, output_name: str) -> StoreInfo | None:
    """Check if an AST node is a store operation (subscript assignment to output).

    Pattern: output[start:end, start:end] = src

    Args:
        node: AST statement node.
        output_name: Name of the output tensor.

    Returns:
        StoreInfo if the node is a store operation, None otherwise.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None

    target = node.targets[0]
    value = node.value

    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name):
        return None
    if target.value.id != output_name:
        return None
    if not isinstance(value, ast.Name):
        return None

    slices = _extract_subscript_slices(target.slice)
    return StoreInfo(dst_name=target.value.id, src_name=value.id, slices=slices)


def _generate_output_alloc(info: OutputAllocInfo, first_input_name: str) -> str:
    """Generate NKI code for output tensor allocation in HBM.

    Args:
        info: Output allocation info.
        first_input_name: Name of the first input tensor for dtype inference.

    Returns:
        NKI code: dst = nl.ndarray(shape=..., dtype=..., buffer=nl.shared_hbm)
    """
    return f"{info.dst_name} = nl.ndarray(shape={info.shape}, dtype={first_input_name}.dtype, buffer=nl.shared_hbm)"


def _generate_load(info: LoadInfo) -> str:
    """Generate NKI code for tensor load (HBM to SBUF via DMA).

    Args:
        info: Load info with source, destination, and slices.

    Returns:
        NKI code with ndarray allocation and dma_copy.
    """
    shape = tuple(end - start for start, end in info.slices)
    slice_str = ", ".join(f"{start}:{end}" for start, end in info.slices)

    lines = [
        f"{info.dst_name} = nl.ndarray(shape={shape}, dtype={info.src_name}.dtype, buffer=nl.sbuf)",
        f"nisa.dma_copy(dst={info.dst_name}, src={info.src_name}[{slice_str}])",
    ]
    return "\n".join(lines)


def _generate_compute(info: ComputeInfo) -> str:
    """Generate NKI code for a compute operation using OP_REGISTRY.

    Args:
        info: Compute info with operation name and inputs.

    Returns:
        NKI code for the operation.

    Raises:
        NotImplementedError: If operation is not in OP_REGISTRY.
    """
    if info.op_name not in OP_REGISTRY:
        raise NotImplementedError(f"Operation '{info.op_name}' not in OP_REGISTRY")

    nki_op = OP_REGISTRY[info.op_name]
    return nki_op.generate_nki(info.inputs, info.dst_name)


def _generate_accumulate(info: AccumulateInfo) -> str:
    """Generate NKI code for an accumulation operation.

    For nc_matmul, uses the accumulator parameter.

    Args:
        info: Accumulate info with operation name and inputs.

    Returns:
        NKI code for the accumulation.

    Raises:
        NotImplementedError: If operation is not in OP_REGISTRY.
    """
    if info.op_name == "nc_matmul":
        return f"nisa.nc_matmul({info.target_name}, {info.inputs[0]}, {info.inputs[1]})"
    else:
        if info.op_name not in OP_REGISTRY:
            raise NotImplementedError(f"Operation '{info.op_name}' not in OP_REGISTRY")
        nki_op = OP_REGISTRY[info.op_name]
        temp_var = f"_temp_{info.target_name}"
        compute_code = nki_op.generate_nki(info.inputs, temp_var)
        return f"{compute_code}\n{info.target_name} = nisa.tensor_tensor({info.target_name}, {temp_var}, op=nl.add)"


def _generate_store(info: StoreInfo, psum_tensors: set[str], first_input_name: str) -> str:
    """Generate NKI code for tensor store (buffer to HBM via DMA).

    If the source is in PSUM, first copies to SBUF using tensor_copy,
    then DMA copies to HBM. DMA copy only supports SBUF/DRAM sources.

    Args:
        info: Store info with source and destination slices.
        psum_tensors: Set of tensor names that are in PSUM buffer.
        first_input_name: Name of the first input tensor for dtype inference.

    Returns:
        NKI code with dma_copy (and tensor_copy if source is PSUM).
    """
    slice_str = ", ".join(f"{start}:{end}" for start, end in info.slices)

    if info.src_name in psum_tensors:
        sbuf_name = f"{info.src_name}_sbuf"
        lines = [
            f"{sbuf_name} = nl.ndarray({info.src_name}.shape, dtype={first_input_name}.dtype, buffer=nl.sbuf)",
            f"nisa.tensor_copy(dst={sbuf_name}, src={info.src_name}, dtype={first_input_name}.dtype)",
            f"nisa.dma_copy(dst={info.dst_name}[{slice_str}], src={sbuf_name})",
        ]
        return "\n".join(lines)

    return f"nisa.dma_copy(dst={info.dst_name}[{slice_str}], src={info.src_name})"


def lower_gym_to_nki(func: Callable) -> str:
    """Lower an nkigym function to NKI kernel code.

    Parses the function source, identifies operation patterns, and generates
    equivalent NKI code with explicit buffer management.

    Args:
        func: An nkigym function to lower.

    Returns:
        NKI kernel code string.

    Raises:
        ValueError: If function definition is not found or invalid.
        NotImplementedError: If an unsupported operation is encountered.
    """
    source = get_source(func)
    sig = inspect.signature(func)
    input_names = set(sig.parameters.keys())
    param_list = ", ".join(sig.parameters.keys())
    first_input_name = list(sig.parameters.keys())[0]

    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")

    body_lines: list[str] = []
    output_name: str | None = None
    psum_tensors: set[str] = set()

    for node in func_def.body:
        output_alloc = _is_output_alloc(node)
        if output_alloc:
            output_name = output_alloc.dst_name
            body_lines.append(_generate_output_alloc(output_alloc, first_input_name))
            continue

        load = _is_load(node, input_names)
        if load:
            body_lines.append(_generate_load(load))
            continue

        compute = _is_compute(node)
        if compute:
            body_lines.append(_generate_compute(compute))
            if compute.op_name == "nc_matmul":
                psum_tensors.add(compute.dst_name)
            continue

        accumulate = _is_accumulate(node)
        if accumulate:
            body_lines.append(_generate_accumulate(accumulate))
            continue

        if output_name:
            store = _is_store(node, output_name)
            if store:
                body_lines.append(_generate_store(store, psum_tensors, first_input_name))
                continue

        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                body_lines.append(f"return {node.value.id}")
            continue

        raise NotImplementedError(f"Unsupported operation: {ast.dump(node)}")

    imports = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "import numpy as np"]
    header = ["", "", "@nki.jit", f"def nki_{func.__name__}({param_list}):"]
    indented_body = ["    " + line for line in "\n".join(body_lines).split("\n")]

    return "\n".join(imports + header + indented_body)
