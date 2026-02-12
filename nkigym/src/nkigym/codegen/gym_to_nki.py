"""NKIGym to NKI lowering module.

This module provides lowering from nkigym intermediate representation
to NKI (Neuron Kernel Interface) kernel code. The lowering is nearly
1:1, translating each nkigym operation to its NKI equivalent with
explicit buffer management (HBM, SBUF, PSUM).

The lowering works by iterating over IR program statements:
- AllocOp: HBM ndarray allocation
- LoadOp: SBUF ndarray + dma_copy from HBM
- StoreOp: dma_copy to HBM (with PSUM->SBUF tensor_copy if needed)
- Compute (NC_MATMUL_OP etc): generate_nki() or nisa.nc_matmul for accumulation

To support a new operator in lowering, ensure its NKIOp.generate_nki()
method is implemented and the operator is registered in OP_REGISTRY.
"""

from nkigym.ir import Program, _shape_from_slices, _slices_to_str
from nkigym.ops import AllocOp, ElementwiseOp, LoadOp, NKIMatmul, StoreOp


def _operand_ref(var: str, slices: tuple[tuple[int, int], ...]) -> str:
    """Format a tensor reference with explicit slices.

    Args:
        var: Variable name.
        slices: Tuple of (start, stop) pairs for each dimension.

    Returns:
        String like ``tensor_0[0:128, 0:128]``.
    """
    slice_str = _slices_to_str(slices)
    return f"{var}[{slice_str}]"


def lower_ir_to_nki(program: Program) -> str:
    """Lower an IR program to NKI kernel code.

    Iterates over IR statements and generates equivalent NKI code with
    explicit buffer management (HBM, SBUF, PSUM).

    Args:
        program: Program tuple (name, params, stmts, return_var, preamble).

    Returns:
        NKI kernel code string.

    Raises:
        NotImplementedError: If an ElementwiseOp is encountered (not yet supported).
    """
    name, params, stmts, return_var, _preamble = program
    first_input_name = params[0]
    param_list = ", ".join(params)

    body_lines: list[str] = []
    psum_tensors: set[str] = set()
    allocated_vars: set[str] = set()
    defined_compute_vars: set[str] = set()

    for op, operands in stmts:
        if isinstance(op, AllocOp):
            line = _generate_alloc(op, operands, first_input_name)
            body_lines.append(line)
            allocated_vars.add(operands[0][0])

        elif isinstance(op, LoadOp):
            line = _generate_load(operands)
            body_lines.append(line)

        elif isinstance(op, StoreOp):
            line = _generate_store(operands, psum_tensors, first_input_name)
            body_lines.append(line)

        elif isinstance(op, ElementwiseOp):
            raise NotImplementedError(f"ElementwiseOp '{op.op_name}' lowering to NKI is not yet supported")

        else:
            dst_var = operands[-1][0]
            is_accumulate = dst_var in allocated_vars or dst_var in defined_compute_vars
            line = _generate_compute(op, operands, is_accumulate)
            body_lines.append(line)
            if isinstance(op, NKIMatmul) and not is_accumulate:
                psum_tensors.add(dst_var)
            defined_compute_vars.add(dst_var)

    body_lines.append(f"return {return_var}")

    imports = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "import numpy as np"]
    header = ["", "", "@nki.jit", f"def nki_{name}({param_list}):"]
    indented_body = ["    " + line for line in "\n".join(body_lines).split("\n")]

    return "\n".join(imports + header + indented_body)


def _generate_alloc(
    op: AllocOp, operands: tuple[tuple[str, tuple[tuple[int, int], ...]], ...], first_input_name: str
) -> str:
    """Generate NKI code for output tensor allocation in HBM.

    Args:
        op: The AllocOp instance.
        operands: Tuple containing (dst_name, slices).
        first_input_name: Name of the first input tensor for dtype inference.

    Returns:
        NKI code for HBM ndarray allocation.
    """
    dst_name, slices = operands[0]
    shape = _shape_from_slices(slices)
    return f"{dst_name} = nl.ndarray(shape={shape}, dtype={first_input_name}.dtype, buffer=nl.shared_hbm)"


def _generate_load(operands: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]) -> str:
    """Generate NKI code for tensor load (HBM to SBUF via DMA).

    Args:
        operands: Tuple of (src_operand, dst_operand) where each is (name, slices).

    Returns:
        NKI code with ndarray allocation and dma_copy.
    """
    src_name, src_slices = operands[0]
    dst_name, dst_slices = operands[1]
    shape = _shape_from_slices(src_slices)
    src_ref = _operand_ref(src_name, src_slices)
    dst_ref = _operand_ref(dst_name, dst_slices)

    lines = [
        f"{dst_name} = nl.ndarray(shape={shape}, dtype={src_name}.dtype, buffer=nl.sbuf)",
        f"nisa.dma_copy(dst={dst_ref}, src={src_ref})",
    ]
    return "\n".join(lines)


def _generate_store(
    operands: tuple[tuple[str, tuple[tuple[int, int], ...]], ...], psum_tensors: set[str], first_input_name: str
) -> str:
    """Generate NKI code for tensor store (buffer to HBM via DMA).

    If the source is in PSUM, first copies to SBUF using tensor_copy,
    then DMA copies to HBM. DMA copy only supports SBUF/DRAM sources.

    Args:
        operands: Tuple of (src_operand, dst_operand) where each is (name, slices).
        psum_tensors: Set of tensor names that are in PSUM buffer.
        first_input_name: Name of the first input tensor for dtype inference.

    Returns:
        NKI code with dma_copy (and tensor_copy if source is PSUM).
    """
    src_name, src_slices = operands[0]
    dst_name, dst_slices = operands[1]
    src_ref = _operand_ref(src_name, src_slices)
    dst_ref = _operand_ref(dst_name, dst_slices)

    if src_name in psum_tensors:
        sbuf_name = f"{src_name}_sbuf"
        sbuf_shape = _shape_from_slices(src_slices)
        lines = [
            f"{sbuf_name} = nl.ndarray(shape={sbuf_shape}, dtype={first_input_name}.dtype, buffer=nl.sbuf)",
            f"nisa.tensor_copy(dst={sbuf_name}, src={src_ref}, dtype={first_input_name}.dtype)",
            f"nisa.dma_copy(dst={dst_ref}, src={sbuf_name})",
        ]
        return "\n".join(lines)

    return f"nisa.dma_copy(dst={dst_ref}, src={src_ref})"


def _generate_compute(
    op: object, operands: tuple[tuple[str, tuple[tuple[int, int], ...]], ...], is_accumulate: bool
) -> str:
    """Generate NKI code for a compute operation.

    For first-write nc_matmul, uses generate_nki() which allocates a PSUM buffer.
    For accumulations, uses nisa.nc_matmul directly into the existing buffer.

    Args:
        op: The NKIOp instance.
        operands: Tuple of operands where the last is the destination.
        is_accumulate: Whether this is an accumulation into an existing buffer.

    Returns:
        NKI code for the compute operation.

    Raises:
        NotImplementedError: If operation does not support generate_nki().
    """
    dst_var, dst_slices = operands[-1]
    input_operands = operands[:-1]
    input_refs = [_operand_ref(var, slices) for var, slices in input_operands]

    if is_accumulate and isinstance(op, NKIMatmul):
        dst_ref = _operand_ref(dst_var, dst_slices)
        return f"nisa.nc_matmul({dst_ref}, {input_refs[0]}, {input_refs[1]})"

    return op.generate_nki(input_refs, dst_var)
