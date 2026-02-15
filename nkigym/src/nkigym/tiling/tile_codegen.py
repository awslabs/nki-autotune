"""Tiled IR generation for NKI Gym.

Generates a tiled GymProgram from a specialized GymProgram.
The output program uses standard numpy ops (np_empty, np_slice, np_store)
for memory operations and nkigym GymOp calls for compute.
"""

from typing import Any

import numpy as np

from nkigym.ir.tensor import TensorRef, ref_name
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.ops.base import GymOp
from nkigym.tiling.analysis import OUTPUT_TENSOR_NAME, TilingAnalysis, analyze_tiling


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Per-axis (0, size) bounds.
    """
    return tuple((0, s) for s in shape)


class TensorNameGenerator:
    """Generates sequential tensor_N names for all computed variables.

    All computed tensors (input slices, intermediate results, accumulators)
    use the pattern tensor_0, tensor_1, etc.
    """

    def __init__(self) -> None:
        """Initialize the generator with counter at 0."""
        self._counter: int = 0

    def next_name(self) -> str:
        """Return the next tensor_N name.

        Returns:
            A string in the format ``tensor_N`` where N is the current
            counter value. The counter is incremented after each call.
        """
        name = f"tensor_{self._counter}"
        self._counter += 1
        return name


def _make_slice_stmt(
    src_var: str, src_shape: tuple[int, ...], slices: list[tuple[int, int]], name_gen: TensorNameGenerator
) -> tuple[GymStatement, str, tuple[int, ...]]:
    """Create an np_slice statement for loading a tile.

    Args:
        src_var: Source variable name (a program parameter).
        src_shape: Full shape of the source variable.
        slices: List of ``(start, stop)`` per axis.
        name_gen: Name generator for the destination variable.

    Returns:
        Tuple of ``(statement, dest_var_name, tile_shape)``.
    """
    dst_var = name_gen.next_name()
    tile_shape = tuple(stop - start for start, stop in slices)
    src_ref = TensorRef(src_var, src_shape, tuple(slices))
    dst_ref = TensorRef(dst_var, tile_shape, _full_slices(tile_shape))
    kwargs: tuple[tuple[str, Any], ...] = (("src", src_ref),)
    return GymStatement(op="np_slice", kwargs=kwargs, output=dst_ref), dst_var, tile_shape


def _make_store_stmt(
    src_var: str, src_shape: tuple[int, ...], dst_slices: list[tuple[int, int]], output_shape: tuple[int, ...]
) -> GymStatement:
    """Create an np_store statement for writing a tile to output.

    Args:
        src_var: Source variable (computed tile).
        src_shape: Shape of the source tile.
        dst_slices: Destination slice bounds in the output tensor.
        output_shape: Full shape of the output tensor.

    Returns:
        A GymStatement for the store operation.
    """
    src_ref = TensorRef(src_var, src_shape, _full_slices(src_shape))
    dst_ref = TensorRef(OUTPUT_TENSOR_NAME, output_shape, tuple(dst_slices))
    kwargs: tuple[tuple[str, Any], ...] = (("src", src_ref), ("dst", dst_ref))
    return GymStatement(op="np_store", kwargs=kwargs, output=dst_ref)


def _make_alloc_stmt(output_shape: tuple[int, ...], dtype_name: str) -> GymStatement:
    """Create an np_empty statement for allocating the output tensor.

    Args:
        output_shape: Shape of the output tensor.
        dtype_name: Numpy dtype string (e.g., ``"np.float32"``).

    Returns:
        A GymStatement for the allocation.
    """
    output_ref = TensorRef(OUTPUT_TENSOR_NAME, output_shape, _full_slices(output_shape))
    kwargs: tuple[tuple[str, Any], ...] = (("dtype", dtype_name),)
    return GymStatement(op="np_empty", kwargs=kwargs, output=output_ref)


def _emit_tile_ops(
    analysis: TilingAnalysis,
    program: GymProgram,
    position: dict[str, int],
    name_gen: TensorNameGenerator,
    acc_var: str | None,
    acc_shape: tuple[int, ...] | None,
    is_first_reduction: bool,
) -> tuple[list[GymStatement], str, tuple[int, ...]]:
    """Emit GymStatements for one tile iteration of the original program.

    For each original statement, loads input tiles (np_slice for program
    params, reuse from intermediate_map for intermediates) and emits the
    compute statement with TensorRef on all kwargs and output.

    Args:
        analysis: Tiling analysis result.
        program: Original untiled program.
        position: Combined parallel + reduction tile position.
        name_gen: Name generator for tile variables.
        acc_var: Accumulation variable for the return-var op, or None.
        acc_shape: Shape of the accumulation variable, or None.
        is_first_reduction: Whether this is the first reduction iteration.

    Returns:
        Tuple of ``(statements, last_result_var, last_result_shape)``.
    """
    stmts: list[GymStatement] = []
    intermediate_map: dict[str, str] = {}
    intermediate_shapes: dict[str, tuple[int, ...]] = {}
    input_params = set(program.params)
    last_result_var = ""
    last_result_shape: tuple[int, ...] = ()
    n_inputs_cache: dict[str, int] = {}

    for orig_stmt in program.stmts:
        op_name = orig_stmt.op
        if op_name not in n_inputs_cache:
            n_inputs_cache[op_name] = len(GymOp.get(op_name).inputs)
        n_inputs = n_inputs_cache[op_name]
        tensor_kwargs = orig_stmt.kwargs[:n_inputs]
        config_kwargs = orig_stmt.kwargs[n_inputs:]

        new_kwargs: list[tuple[str, Any]] = []
        input_shapes: list[tuple[int, ...]] = []
        for operand_name, var_ref in tensor_kwargs:
            var_name = ref_name(var_ref)
            if var_name in input_params:
                src_shape = analysis.var_shapes[var_name]
                var_slices = analysis.compute_slices(var_name, position)
                slice_stmt, tile_var, tile_shape = _make_slice_stmt(var_name, src_shape, var_slices, name_gen)
                stmts.append(slice_stmt)
                new_kwargs.append((operand_name, TensorRef(tile_var, tile_shape, _full_slices(tile_shape))))
                input_shapes.append(tile_shape)
            elif var_name in intermediate_map:
                mapped_var = intermediate_map[var_name]
                mapped_shape = intermediate_shapes[var_name]
                new_kwargs.append((operand_name, TensorRef(mapped_var, mapped_shape, _full_slices(mapped_shape))))
                input_shapes.append(mapped_shape)
            else:
                raise RuntimeError(
                    f"Unknown variable '{var_name}' in statement " f"'{orig_stmt.op}' — not a param or intermediate"
                )

        for key, value in config_kwargs:
            if key == "acc":
                continue
            new_kwargs.append((key, value))

        is_return_op = orig_stmt.output.name == program.return_var
        use_acc = is_return_op and not is_first_reduction and acc_var is not None

        op_cls = GymOp.get(op_name)
        out_shape = op_cls().output_shape(tuple(input_shapes))

        if use_acc:
            new_kwargs.append(("acc", TensorRef(acc_var, acc_shape, _full_slices(acc_shape))))
            dst_var = acc_var
        else:
            dst_var = name_gen.next_name()

        compute_stmt = GymStatement(
            op=op_name, kwargs=tuple(new_kwargs), output=TensorRef(dst_var, out_shape, _full_slices(out_shape))
        )
        stmts.append(compute_stmt)
        intermediate_map[orig_stmt.output.name] = dst_var
        intermediate_shapes[orig_stmt.output.name] = out_shape

        if is_return_op:
            last_result_var = dst_var
            last_result_shape = out_shape

    return stmts, last_result_var, last_result_shape


def tile_program(program: GymProgram) -> GymProgram:
    """Generate a tiled GymProgram from a specialized program.

    Produces a GymProgram with explicit tile slicing using np_slice,
    np_store, and np_empty operations alongside the original compute ops.
    Reads shapes from TensorRef on the input program and output_dtype
    from the program.

    Args:
        program: A specialized GymProgram with TensorRef on all tensor kwargs.

    Returns:
        A tiled GymProgram.
    """
    analysis = analyze_tiling(program)

    output_shape = analysis.var_shapes[program.return_var]

    dtype_name = f"np.{np.dtype(program.output_dtype).name}"
    stmts: list[GymStatement] = []
    stmts.append(_make_alloc_stmt(output_shape, dtype_name))

    name_gen = TensorNameGenerator()

    for _, parallel_pos in analysis.iter_tile_positions():
        if analysis.has_reduction_tiling():
            acc_var: str | None = None
            acc_shape: tuple[int, ...] | None = None

            for red_idx, reduction_pos in enumerate(analysis.iter_reduction_positions()):
                combined_pos = {**parallel_pos, **reduction_pos}
                is_first = red_idx == 0

                tile_stmts, result_var, result_shape = _emit_tile_ops(
                    analysis,
                    program,
                    combined_pos,
                    name_gen,
                    acc_var=acc_var,
                    acc_shape=acc_shape,
                    is_first_reduction=is_first,
                )
                stmts.extend(tile_stmts)

                if is_first:
                    acc_var = result_var
                    acc_shape = result_shape
        else:
            combined_pos = dict(parallel_pos)
            tile_stmts, result_var, result_shape = _emit_tile_ops(
                analysis, program, combined_pos, name_gen, acc_var=None, acc_shape=None, is_first_reduction=True
            )
            stmts.extend(tile_stmts)
            acc_var = result_var
            acc_shape = result_shape

        output_slices = analysis.compute_slices(program.return_var, parallel_pos)
        stmts.append(_make_store_stmt(acc_var, acc_shape, output_slices, output_shape))

    return GymProgram(
        name=program.name,
        params=program.params,
        input_shapes=program.input_shapes,
        stmts=tuple(stmts),
        return_var=OUTPUT_TENSOR_NAME,
        output_dtype=program.output_dtype,
    )
