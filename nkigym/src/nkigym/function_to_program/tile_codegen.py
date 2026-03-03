"""Tiled IR generation for NKI Gym.

Generates a tiled GymProgram from a specialized GymProgram.
The output program uses standard numpy ops (np_empty, np_slice, np_store)
for memory operations and nkigym GymOp calls for compute.
"""

from dataclasses import dataclass, field
from typing import Any

from nkigym.function_to_program.analysis import OUTPUT_TENSOR_NAME, TilingAnalysis, analyze_tiling
from nkigym.ir.tensor import TensorRef, full_slices, ref_name
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.ops.tiling_ops import AllocateOp, LoadOp, StoreOp


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
    dst_ref = TensorRef(dst_var, tile_shape, full_slices(tile_shape))
    kwargs: tuple[tuple[str, Any], ...] = (("src", src_ref),)
    return GymStatement(op=LoadOp, kwargs=kwargs, output=dst_ref), dst_var, tile_shape


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
    src_ref = TensorRef(src_var, src_shape, full_slices(src_shape))
    dst_ref = TensorRef(OUTPUT_TENSOR_NAME, output_shape, tuple(dst_slices))
    kwargs: tuple[tuple[str, Any], ...] = (("src", src_ref), ("dst", dst_ref))
    return GymStatement(op=StoreOp, kwargs=kwargs, output=dst_ref)


def _make_alloc_stmt(output_shape: tuple[int, ...], output_dtype: type) -> GymStatement:
    """Create an np_empty statement for allocating the output tensor.

    Args:
        output_shape: Shape of the output tensor.
        output_dtype: Numpy dtype type (e.g., ``np.float32``).

    Returns:
        A GymStatement for the allocation.
    """
    output_ref = TensorRef(OUTPUT_TENSOR_NAME, output_shape, full_slices(output_shape))
    kwargs: tuple[tuple[str, Any], ...] = (("dtype", output_dtype),)
    return GymStatement(op=AllocateOp, kwargs=kwargs, output=output_ref)


@dataclass
class _TileEmitState:
    """Mutable state for tile emission.

    Attributes:
        stmts: Accumulated GymStatements.
        intermediate_map: Maps original output name to tile variable name.
        intermediate_shapes: Maps original output name to tile shape.
        input_params: Set of program parameter names.
        last_result_var: Variable name of the last return-op result.
        last_result_shape: Shape of the last return-op result.
    """

    stmts: list[GymStatement] = field(default_factory=list)
    intermediate_map: dict[str, str] = field(default_factory=dict)
    intermediate_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    input_params: set[str] = field(default_factory=set)
    last_result_var: str = ""
    last_result_shape: tuple[int, ...] = ()


@dataclass
class _TileContext:
    """Immutable parameters for tile emission.

    Attributes:
        analysis: Tiling analysis result.
        position: Combined parallel + reduction tile position.
        name_gen: Name generator for tile variables.
        acc_var: Accumulation variable (empty string if none).
        acc_shape: Shape of accumulation variable (empty tuple if none).
        is_first_reduction: Whether this is the first reduction iteration.
        return_var: The program's return variable name.
    """

    analysis: TilingAnalysis
    position: dict[str, int]
    name_gen: TensorNameGenerator
    acc_var: str
    acc_shape: tuple[int, ...]
    is_first_reduction: bool
    return_var: str


def _resolve_operand(
    var_name: str, operand_name: str, state: _TileEmitState, ctx: _TileContext
) -> tuple[tuple[str, Any], tuple[int, ...]]:
    """Resolve a single tensor operand to a tile kwarg and shape.

    For program params, creates an np_slice statement. For intermediates,
    looks up from the intermediate map.

    Args:
        var_name: Variable name of the operand.
        operand_name: Keyword argument name for the operand.
        state: Mutable emit state.
        ctx: Tile emission context.

    Returns:
        Tuple of (kwarg_pair, tile_shape).

    Raises:
        RuntimeError: If var_name is not a param or intermediate.
    """
    kwarg: tuple[str, Any]
    tile_shape: tuple[int, ...]

    if var_name in state.input_params:
        src_shape = ctx.analysis.var_shapes[var_name]
        var_slices = ctx.analysis.compute_slices(var_name, ctx.position)
        slice_stmt, tile_var, tile_shape = _make_slice_stmt(var_name, src_shape, var_slices, ctx.name_gen)
        state.stmts.append(slice_stmt)
        kwarg = (operand_name, TensorRef(tile_var, tile_shape, full_slices(tile_shape)))
    elif var_name in state.intermediate_map:
        mapped_var = state.intermediate_map[var_name]
        tile_shape = state.intermediate_shapes[var_name]
        kwarg = (operand_name, TensorRef(mapped_var, tile_shape, full_slices(tile_shape)))
    else:
        raise RuntimeError(f"Unknown variable '{var_name}' — not a param or intermediate")

    return kwarg, tile_shape


def _build_compute_kwargs(
    orig_stmt: GymStatement, state: _TileEmitState, ctx: _TileContext
) -> tuple[list[tuple[str, Any]], list[tuple[int, ...]]]:
    """Build kwargs and input shapes for a compute statement.

    Args:
        orig_stmt: Original untiled statement.
        state: Mutable emit state.
        ctx: Tile emission context.

    Returns:
        Tuple of (new_kwargs, input_shapes).
    """
    op_cls = orig_stmt.op
    n_inputs = len(op_cls.inputs)
    tensor_kwargs = orig_stmt.kwargs[:n_inputs]
    config_kwargs = orig_stmt.kwargs[n_inputs:]

    new_kwargs: list[tuple[str, Any]] = []
    input_shapes: list[tuple[int, ...]] = []
    for operand_name, var_ref in tensor_kwargs:
        kwarg, tile_shape = _resolve_operand(ref_name(var_ref), operand_name, state, ctx)
        new_kwargs.append(kwarg)
        input_shapes.append(tile_shape)

    for key, value in config_kwargs:
        if key != "acc":
            new_kwargs.append((key, value))

    return new_kwargs, input_shapes


def _emit_one_stmt(orig_stmt: GymStatement, state: _TileEmitState, ctx: _TileContext) -> None:
    """Emit tiled statements for one original statement.

    Args:
        orig_stmt: Original untiled statement.
        state: Mutable emit state.
        ctx: Tile emission context.
    """
    new_kwargs, input_shapes = _build_compute_kwargs(orig_stmt, state, ctx)

    is_return_op = orig_stmt.output.name == ctx.return_var
    use_acc = is_return_op and not ctx.is_first_reduction and ctx.acc_var != ""

    out_shape = orig_stmt.op.output_shape(tuple(input_shapes))

    if use_acc:
        new_kwargs.append(("acc", TensorRef(ctx.acc_var, ctx.acc_shape, full_slices(ctx.acc_shape))))

    dst_var = ctx.name_gen.next_name()
    compute_stmt = GymStatement(
        op=orig_stmt.op, kwargs=tuple(new_kwargs), output=TensorRef(dst_var, out_shape, full_slices(out_shape))
    )
    state.stmts.append(compute_stmt)
    state.intermediate_map[orig_stmt.output.name] = dst_var
    state.intermediate_shapes[orig_stmt.output.name] = out_shape

    if is_return_op:
        state.last_result_var = dst_var
        state.last_result_shape = out_shape


def _emit_tile_ops(
    analysis: TilingAnalysis,
    program: GymProgram,
    position: dict[str, int],
    name_gen: TensorNameGenerator,
    acc_var: str,
    acc_shape: tuple[int, ...],
    is_first_reduction: bool,
) -> tuple[list[GymStatement], str, tuple[int, ...]]:
    """Emit GymStatements for one tile iteration of the original program.

    Args:
        analysis: Tiling analysis result.
        program: Original untiled program.
        position: Combined parallel + reduction tile position.
        name_gen: Name generator for tile variables.
        acc_var: Accumulation variable (empty string if none).
        acc_shape: Shape of accumulation variable (empty tuple if none).
        is_first_reduction: Whether this is the first reduction iteration.

    Returns:
        Tuple of ``(statements, last_result_var, last_result_shape)``.
    """
    state = _TileEmitState(input_params=set(program.params))
    ctx = _TileContext(
        analysis=analysis,
        position=position,
        name_gen=name_gen,
        acc_var=acc_var,
        acc_shape=acc_shape,
        is_first_reduction=is_first_reduction,
        return_var=program.return_var,
    )

    for orig_stmt in program.stmts:
        _emit_one_stmt(orig_stmt, state, ctx)

    return state.stmts, state.last_result_var, state.last_result_shape


def _tile_parallel_with_reduction(
    analysis: TilingAnalysis, program: GymProgram, parallel_pos: dict[str, int], name_gen: TensorNameGenerator
) -> tuple[list[GymStatement], str, tuple[int, ...]]:
    """Tile a single parallel position with reduction iterations.

    Args:
        analysis: Tiling analysis result.
        program: Original untiled program.
        parallel_pos: Parallel tile position.
        name_gen: Name generator for tile variables.

    Returns:
        Tuple of ``(statements, final_acc_var, final_acc_shape)``.
    """
    stmts: list[GymStatement] = []
    acc_var = ""
    acc_shape: tuple[int, ...] = ()

    for red_idx, reduction_pos in enumerate(analysis.iter_reduction_positions()):
        combined_pos = {**parallel_pos, **reduction_pos}
        tile_stmts, result_var, result_shape = _emit_tile_ops(
            analysis,
            program,
            combined_pos,
            name_gen,
            acc_var=acc_var,
            acc_shape=acc_shape,
            is_first_reduction=(red_idx == 0),
        )
        stmts.extend(tile_stmts)
        acc_var = result_var
        acc_shape = result_shape

    return stmts, acc_var, acc_shape


def tile_program(program: GymProgram) -> GymProgram:
    """Generate a tiled GymProgram from a specialized program.

    Produces a GymProgram with explicit tile slicing using np_slice,
    np_store, and np_empty operations alongside the original compute ops.

    Args:
        program: A specialized GymProgram with TensorRef on all tensor kwargs.

    Returns:
        A tiled GymProgram.
    """
    analysis = analyze_tiling(program)
    output_shape = analysis.var_shapes[program.return_var]

    stmts: list[GymStatement] = [_make_alloc_stmt(output_shape, program.output_dtype)]
    name_gen = TensorNameGenerator()

    for _, parallel_pos in analysis.iter_tile_positions():
        if analysis.has_reduction_tiling():
            tile_stmts, acc_var, acc_shape = _tile_parallel_with_reduction(analysis, program, parallel_pos, name_gen)
        else:
            combined_pos = dict(parallel_pos)
            tile_stmts, acc_var, acc_shape = _emit_tile_ops(
                analysis, program, combined_pos, name_gen, acc_var="", acc_shape=(), is_first_reduction=True
            )
        stmts.extend(tile_stmts)
        output_slices = analysis.compute_slices(program.return_var, parallel_pos)
        stmts.append(_make_store_stmt(acc_var, acc_shape, output_slices, output_shape))

    return GymProgram(
        name=program.name,
        kwargs=program.kwargs,
        stmts=tuple(stmts),
        return_var=OUTPUT_TENSOR_NAME,
        output_dtype=program.output_dtype,
    )
