"""Tiled IR and source code generation for NKI Gym.

Generates program tuples (IR) from dimension analysis, with explicit slice
offsets for each tile position. Source and callable outputs are wrappers
around the IR generation.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from nkigym.ir import Operand, Program, Statement, _full_slices
from nkigym.ops import ALLOC_OPS, LOAD_OP, OP_REGISTRY, STORE_OP, NKIOp
from nkigym.tiling.analysis import OUTPUT_TENSOR_NAME, TensorSliceInfo, analyze_dimension

logger = logging.getLogger(__name__)


class TensorNameGenerator:
    """Generates sequential tensor_N names for all computed variables.

    All computed tensors (input slices, intermediate results, accumulators)
    use the pattern tensor_0, tensor_1, etc.

    The counter can be reset at the start of each subgraph iteration to
    maintain consistent naming within each tile computation.
    """

    def __init__(self) -> None:
        """Initialize the generator with counter at 0."""
        self._counter: int = 0

    def next_name(self) -> str:
        """Return the next tensor_N name.

        Returns:
            A string in the format 'tensor_N' where N is the current counter value.
            The counter is incremented after each call.
        """
        name = f"tensor_{self._counter}"
        self._counter += 1
        return name

    def reset(self) -> None:
        """Reset counter to 0.

        Call this at the start of each subgraph iteration to restart
        the naming sequence.
        """
        self._counter = 0


def _slice_info_to_operand(var_name: str, slice_info: TensorSliceInfo) -> Operand:
    """Convert a TensorSliceInfo to an Operand tuple.

    Args:
        var_name: Variable name for the operand.
        slice_info: Slice parameters with offsets and sizes.

    Returns:
        Operand tuple (var_name, slices).
    """
    slices = tuple((offset, offset + size) for offset, size in zip(slice_info.offsets, slice_info.sizes))
    return (var_name, slices)


def _full_operand(var_name: str, shape: tuple[int, ...]) -> Operand:
    """Create a full-range operand from a shape.

    Args:
        var_name: Variable name.
        shape: Shape of the tensor.

    Returns:
        Operand tuple with (0, size) slices for each dimension.
    """
    return (var_name, _full_slices(shape))


def _load_op_inputs(
    op_input_names: list[str],
    input_shapes: dict[str, tuple[int, ...]],
    get_slice_info: Callable[[str], TensorSliceInfo],
    intermediate_map: dict[str, str],
    name_gen: TensorNameGenerator,
    var_shapes: dict[str, tuple[int, ...]],
    stmts: list[Statement],
) -> tuple[list[str], list[tuple[int, ...]]]:
    """Load inputs for one op, emitting LOAD statements as needed.

    For each input name, either loads from a global input tensor (emitting a
    LOAD_OP statement) or resolves from the intermediate map.

    Args:
        op_input_names: Symbolic input names from the traced op.
        input_shapes: Maps global parameter name to shape tuple.
        get_slice_info: Callback returning TensorSliceInfo for a given input name.
        intermediate_map: Maps symbolic tensor name to concrete variable name.
        name_gen: Name generator for new tile variables.
        var_shapes: Mutable map from variable name to shape, updated in place.
        stmts: Mutable list of statements, appended in place.

    Returns:
        Tuple of (op_input_vars, op_input_shapes) listing the concrete variable
        names and their shapes for this op's inputs.

    Raises:
        RuntimeError: If an input name is not in input_shapes or intermediate_map.
    """
    op_inputs: list[str] = []
    op_input_shapes: list[tuple[int, ...]] = []

    for inp_name in op_input_names:
        if inp_name in input_shapes:
            slice_info = get_slice_info(inp_name)
            tile_var = name_gen.next_name()
            tile_shape = tuple(slice_info.sizes)
            var_shapes[tile_var] = tile_shape

            src_operand = _slice_info_to_operand(inp_name, slice_info)
            dst_operand = _full_operand(tile_var, tile_shape)
            stmts.append((LOAD_OP, (src_operand, dst_operand)))

            op_inputs.append(tile_var)
            op_input_shapes.append(tile_shape)
        elif inp_name in intermediate_map:
            var_name = intermediate_map[inp_name]
            shape = var_shapes[var_name]
            op_inputs.append(var_name)
            op_input_shapes.append(shape)
        else:
            raise RuntimeError(f"Unknown input tensor '{inp_name}' - not in inputs or intermediates")

    return op_inputs, op_input_shapes


def _emit_compute_stmt(
    nki_op: NKIOp,
    op_inputs: list[str],
    op_input_shapes: list[tuple[int, ...]],
    name_gen: TensorNameGenerator,
    var_shapes: dict[str, tuple[int, ...]],
    acc_var: str | None = None,
) -> tuple[Statement, str]:
    """Build a compute statement and return it with the destination variable.

    When ``acc_var`` is provided the statement accumulates into the existing
    variable; otherwise a fresh destination variable is allocated.

    Args:
        nki_op: The NKIOp instance for the compute.
        op_inputs: Concrete variable names for the op inputs.
        op_input_shapes: Shapes of the op inputs.
        name_gen: Name generator for new result variables.
        var_shapes: Mutable map from variable name to shape, updated in place.
        acc_var: If set, accumulate into this variable instead of creating a new one.

    Returns:
        Tuple of (statement, dst_var) where dst_var is the destination variable name.
    """
    if acc_var is not None:
        input_operands = tuple(_full_operand(v, var_shapes[v]) for v in op_inputs)
        dst_operand = _full_operand(acc_var, var_shapes[acc_var])
        return (nki_op, (*input_operands, dst_operand)), acc_var

    result_var = name_gen.next_name()
    result_shape = nki_op.output_shape(op_input_shapes)
    var_shapes[result_var] = result_shape
    input_operands = tuple(_full_operand(v, var_shapes[v]) for v in op_inputs)
    dst_operand = _full_operand(result_var, result_shape)
    return (nki_op, (*input_operands, dst_operand)), result_var


def generate_tiled_ir(func: Callable[..., np.ndarray], kernel_kwargs: dict[str, Any], output_dtype: type) -> Program:
    """Generate a program tuple (IR) for a tiled version of the function.

    Produces IR statements where each subgraph is expanded inline with
    explicit slice offsets. Handles both parallel and reduction tiling.

    Args:
        func: NumPy function to tile.
        kernel_kwargs: Maps parameter name to ndarray (or shape tuple).
        output_dtype: NumPy dtype for the output array allocation.

    Returns:
        Program tuple (name, params, stmts, return_var, preamble).

    Raises:
        NotImplementedError: If an operator is not in OP_REGISTRY or
            has reduction dimensions but no reduce function.
    """
    input_shapes = {
        key: kernel_kwargs[key].shape if hasattr(kernel_kwargs[key], "shape") else kernel_kwargs[key]
        for key in kernel_kwargs
    }
    analysis = analyze_dimension(func, input_shapes)
    logger.debug(analysis)
    param_names = tuple(input_shapes.keys())

    stmts: list[Statement] = []
    output_name = OUTPUT_TENSOR_NAME

    output_shape = analysis.tensor_shapes[OUTPUT_TENSOR_NAME]
    dtype_name = np.dtype(output_dtype).name
    if dtype_name not in ALLOC_OPS:
        raise ValueError(f"Unsupported alloc dtype: {dtype_name}")
    alloc_op = ALLOC_OPS[dtype_name]
    stmts.append((alloc_op, (_full_operand(output_name, output_shape),)))

    reduction_positions = list(analysis.iter_reduction_tile_positions())
    has_reduction_tiling = len(reduction_positions) > 1 or (len(reduction_positions) == 1 and reduction_positions[0])

    name_gen = TensorNameGenerator()
    var_shapes: dict[str, tuple[int, ...]] = {}

    for subgraph_idx, parallel_pos in analysis.iter_tile_positions():
        if has_reduction_tiling:
            acc_var: str | None = None

            for red_idx, reduction_pos in enumerate(reduction_positions):
                intermediate_map: dict[str, str] = {}

                def _get_reduction_slice(
                    inp_name: str, pp: dict[str, int] = parallel_pos, rp: dict[str, int] = reduction_pos
                ) -> TensorSliceInfo:
                    return analysis.compute_reduction_slice_params(inp_name, pp, rp)

                for op in analysis.ops:
                    if op.op_name not in OP_REGISTRY:
                        raise NotImplementedError(f"Operator '{op.op_name}' not supported")
                    nki_op = OP_REGISTRY[op.op_name]

                    op_inputs, op_input_shapes = _load_op_inputs(
                        op.inputs, input_shapes, _get_reduction_slice, intermediate_map, name_gen, var_shapes, stmts
                    )

                    is_output_op = op.output == OUTPUT_TENSOR_NAME
                    use_acc = is_output_op and red_idx > 0
                    stmt, dst_var = _emit_compute_stmt(
                        nki_op, op_inputs, op_input_shapes, name_gen, var_shapes, acc_var=acc_var if use_acc else None
                    )
                    stmts.append(stmt)

                    if is_output_op:
                        acc_var = dst_var
                    intermediate_map[op.output] = dst_var

            output_slice = analysis.compute_reduction_slice_params(OUTPUT_TENSOR_NAME, parallel_pos, {})
            src_operand = _full_operand(acc_var, var_shapes[acc_var])
            dst_operand = _slice_info_to_operand(output_name, output_slice)
            stmts.append((STORE_OP, (src_operand, dst_operand)))
        else:
            intermediate_map = {}

            def _get_parallel_slice(inp_name: str, si: int = subgraph_idx) -> TensorSliceInfo:
                return analysis.slice_params[inp_name][si]

            for op in analysis.ops:
                if op.op_name not in OP_REGISTRY:
                    raise NotImplementedError(f"Operator '{op.op_name}' not supported")
                nki_op = OP_REGISTRY[op.op_name]

                op_inputs, op_input_shapes = _load_op_inputs(
                    op.inputs, input_shapes, _get_parallel_slice, intermediate_map, name_gen, var_shapes, stmts
                )

                stmt, dst_var = _emit_compute_stmt(nki_op, op_inputs, op_input_shapes, name_gen, var_shapes)
                stmts.append(stmt)
                intermediate_map[op.output] = dst_var

            output_slice = analysis.slice_params[OUTPUT_TENSOR_NAME][subgraph_idx]
            result_var = intermediate_map[OUTPUT_TENSOR_NAME]
            src_operand = _full_operand(result_var, var_shapes[result_var])
            dst_operand = _slice_info_to_operand(output_name, output_slice)
            stmts.append((STORE_OP, (src_operand, dst_operand)))

    func_name = func.__name__
    return Program(func_name, param_names, tuple(stmts), output_name, "")
