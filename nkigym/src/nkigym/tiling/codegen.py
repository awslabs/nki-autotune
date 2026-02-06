"""Tiled source code generation for NKI Gym.

Generates flattened/unrolled Python source code from dimension analysis,
with explicit slice offsets for each tile position.
"""

import logging
from collections.abc import Callable

import numpy as np

from nkigym.ops import OP_REGISTRY
from nkigym.tiling.analysis import OUTPUT_TENSOR_NAME, TensorSliceInfo, analyze_dimension
from nkigym.utils.source import exec_source_to_func

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


def _generate_slice_expr(tensor_name: str, slice_info: TensorSliceInfo) -> str:
    """Generate a slice expression like 'a[0:128, 0:128]'.

    Args:
        tensor_name: Name of the tensor to slice.
        slice_info: Slice parameters.

    Returns:
        Slice expression string.
    """
    slices = []
    for offset, size in zip(slice_info.offsets, slice_info.sizes):
        slices.append(f"{offset}:{offset + size}")
    return f"{tensor_name}[{', '.join(slices)}]"


def generate_tiled_source(func: Callable[..., np.ndarray], input_shapes: dict[str, tuple[int, ...]]) -> str:
    """Generate source code string for a tiled version of the function.

    Produces flattened/unrolled code where each subgraph is expanded inline
    with explicit slice offsets. Handles both parallel and reduction tiling.
    No loops are generated - all tiles are fully unrolled.

    Uses the OP_REGISTRY table for operator-agnostic code generation:
    - generate_expr(inputs): Creates the initial computation
    - reduce(result_var, inputs): Creates in-place accumulation

    Args:
        func: NumPy function to tile.
        input_shapes: Maps parameter name to shape tuple.

    Returns:
        Python source code string for the tiled function.

    Raises:
        NotImplementedError: If an operator is not in OP_REGISTRY or
            has reduction dimensions but no reduce function.
    """
    analysis = analyze_dimension(func, input_shapes)
    logger.debug(analysis)
    param_names = list(input_shapes.keys())

    lines: list[str] = []
    output_name = OUTPUT_TENSOR_NAME

    lines.append(f"def tiled_{func.__name__}({', '.join(param_names)}):")
    output_shape = analysis.tensor_shapes[OUTPUT_TENSOR_NAME]
    lines.append(f"    {output_name} = nkigym.ndarray({output_shape}, dtype=np.float32)")

    reduction_positions = list(analysis.iter_reduction_tile_positions())
    has_reduction_tiling = len(reduction_positions) > 1 or (len(reduction_positions) == 1 and reduction_positions[0])

    name_gen = TensorNameGenerator()

    for subgraph_idx, parallel_pos in analysis.iter_tile_positions():
        if has_reduction_tiling:
            acc_var: str | None = None

            for red_idx, reduction_pos in enumerate(reduction_positions):
                intermediate_map: dict[str, str] = {}

                for op in analysis.ops:
                    if op.op_name not in OP_REGISTRY:
                        raise NotImplementedError(f"Operator '{op.op_name}' not supported")

                    nki_op = OP_REGISTRY[op.op_name]
                    op_inputs: list[str] = []

                    for inp_name in op.inputs:
                        if inp_name in input_shapes:
                            slice_info = analysis.compute_reduction_slice_params(inp_name, parallel_pos, reduction_pos)
                            slice_expr = _generate_slice_expr(inp_name, slice_info)
                            tile_var = name_gen.next_name()
                            lines.append(f"    {tile_var} = {slice_expr}")
                            op_inputs.append(tile_var)
                        elif inp_name in intermediate_map:
                            op_inputs.append(intermediate_map[inp_name])
                        else:
                            raise RuntimeError(f"Unknown input tensor '{inp_name}' - not in inputs or intermediates")

                    is_first_tile = red_idx == 0

                    if op.output == OUTPUT_TENSOR_NAME:
                        if is_first_tile:
                            result_var = name_gen.next_name()
                            expr = nki_op.generate_expr(op_inputs)
                            lines.append(f"    {result_var} = {expr}")
                            acc_var = result_var
                        else:
                            combine_expr = nki_op.reduce(acc_var, op_inputs)
                            if combine_expr is None:
                                raise NotImplementedError(
                                    f"Operator '{nki_op.op_name}' has reduction dimensions " "but no reduce"
                                )
                            lines.append(f"    {combine_expr}")
                        intermediate_map[op.output] = acc_var
                    else:
                        result_var = name_gen.next_name()
                        expr = nki_op.generate_expr(op_inputs)
                        lines.append(f"    {result_var} = {expr}")
                        intermediate_map[op.output] = result_var

            output_slice = analysis.compute_reduction_slice_params(OUTPUT_TENSOR_NAME, parallel_pos, {})
            if len(output_slice.offsets) != 2:
                raise ValueError(f"Expected 2D output, got {len(output_slice.offsets)} dimensions")
            row_off, col_off = output_slice.offsets[0], output_slice.offsets[1]
            row_sz, col_sz = output_slice.sizes[0], output_slice.sizes[1]
            lines.append(f"    {output_name}[{row_off}:{row_off + row_sz}, {col_off}:{col_off + col_sz}] = {acc_var}\n")
        else:
            intermediate_map = {}

            for op in analysis.ops:
                if op.op_name not in OP_REGISTRY:
                    raise NotImplementedError(f"Operator '{op.op_name}' not supported")

                nki_op = OP_REGISTRY[op.op_name]
                op_inputs = []
                for inp_name in op.inputs:
                    if inp_name in input_shapes:
                        slice_info = analysis.slice_params[inp_name][subgraph_idx]
                        slice_expr = _generate_slice_expr(inp_name, slice_info)
                        tile_var = name_gen.next_name()
                        lines.append(f"    {tile_var} = {slice_expr}")
                        op_inputs.append(tile_var)
                    elif inp_name in intermediate_map:
                        op_inputs.append(intermediate_map[inp_name])
                    else:
                        raise RuntimeError(f"Unknown input tensor '{inp_name}' - not in inputs or intermediates")

                output_var = name_gen.next_name()
                intermediate_map[op.output] = output_var

                expr = nki_op.generate_expr(op_inputs)
                lines.append(f"    {output_var} = {expr}")

            output_slice = analysis.slice_params[OUTPUT_TENSOR_NAME][subgraph_idx]
            if len(output_slice.offsets) != 2:
                raise ValueError(f"Expected 2D output, got {len(output_slice.offsets)} dimensions")
            row_off, col_off = output_slice.offsets[0], output_slice.offsets[1]
            row_sz, col_sz = output_slice.sizes[0], output_slice.sizes[1]
            result_var = intermediate_map[OUTPUT_TENSOR_NAME]
            lines.append(
                f"    {output_name}[{row_off}:{row_off + row_sz}, {col_off}:{col_off + col_sz}] = {result_var}\n"
            )

    lines.append(f"    return {output_name}")

    return "\n".join(lines) + "\n"


def generate_tiled_function(
    func: Callable[..., np.ndarray], input_shapes: dict[str, tuple[int, ...]]
) -> Callable[..., np.ndarray]:
    """Generate a callable tiled version of the function.

    Generates source code via generate_tiled_source() and executes it
    to produce a callable function.

    Args:
        func: NumPy function to tile.
        input_shapes: Maps parameter name to shape tuple.

    Returns:
        Callable tiled function with same signature as original.
    """
    source = generate_tiled_source(func, input_shapes)
    tiled_func_name = f"tiled_{func.__name__}"
    return exec_source_to_func(source, tiled_func_name)
