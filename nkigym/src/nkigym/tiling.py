"""NKI Gym Tiling Pass

Analyzes NumPy functions to extract dimension information and compute
tile parameters for data-parallel subgraph generation.
"""

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from nkigym.codegen import exec_source_to_func
from nkigym.dim_tracker import TracedOp, _DimTracker
from nkigym.numpy_ops import OP_SEMANTICS
from nkigym.tensor import TracedTensor

logger = logging.getLogger(__name__)

TILE_SIZE = 128
OUTPUT_TENSOR_NAME = "output"
RESERVED_INTERMEDIATE_PATTERN = re.compile(r"^tensor_\d+$")


class TensorNameGenerator:
    """Generates sequential tensor_N names for all computed variables.

    This class provides a simple counter-based naming system for computed
    variables in code generation. All computed tensors (input slices,
    intermediate results, accumulators) use the pattern tensor_0, tensor_1, etc.

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


def _validate_input_names(input_shapes: dict[str, tuple[int, ...]]) -> None:
    """Validate that input names don't use reserved names.

    Args:
        input_shapes: Maps parameter name to shape tuple.

    Raises:
        ValueError: If any input name conflicts with reserved names.
    """
    for name in input_shapes:
        if name == OUTPUT_TENSOR_NAME:
            raise ValueError(f"Input name '{name}' is reserved for the output tensor")
        if RESERVED_INTERMEDIATE_PATTERN.match(name):
            raise ValueError(f"Input name '{name}' is reserved for intermediate tensors")


@dataclass
class DimInfo:
    """Information about a global dimension.

    Attributes:
        id: Unique dimension identifier (e.g., "d0", "d1").
        size: Size of the dimension.
        iter_type: Either "parallel" or "reduction". A dimension is "parallel"
            if it appears in the output tensor (can be tiled independently),
            or "reduction" if it's absent from the output (contracted over).
            For matmul C[m,n] = A[m,k] @ B[k,n]: m and n are parallel, k is reduction.
    """

    id: str
    size: int
    iter_type: str = "parallel"

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return f"{self.id}: {self.iter_type} (size={self.size})"


@dataclass
class TensorSliceInfo:
    """Pre-computed slice parameters for a tensor at a tile position.

    Attributes:
        offsets: Slice offsets for each dimension.
        sizes: Slice sizes for each dimension.
        strides: Slice strides for each dimension (always [1, 1, ...]).
    """

    offsets: list[int]
    sizes: list[int]
    strides: list[int]

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return f"offsets={self.offsets}, sizes={self.sizes}"


@dataclass
class DimensionAnalysis:
    """Result of dimension analysis for a NumPy function.

    Attributes:
        tensor_dims: Maps tensor name to list of dimension IDs (includes output).
        tensor_shapes: Maps tensor name to shape tuple (includes output).
        dim_info: Maps dimension ID to DimInfo.
        dim_order: List of dimension IDs in discovery order.
        tile_counts: Maps parallel dimension ID to number of tiles.
        num_subgraphs: Total number of independent tile subgraphs.
        slice_params: Maps input tensor name to {subgraph_idx: TensorSliceInfo}.
        ops: List of traced operations in execution order.
        output: Name of the output tensor in tensor_dims/tensor_shapes.
        reduction_tile_counts: Maps reduction dimension ID to number of 128-tiles.
    """

    tensor_dims: dict[str, list[str]] = field(default_factory=dict)
    tensor_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dim_info: dict[str, DimInfo] = field(default_factory=dict)
    dim_order: list[str] = field(default_factory=list)
    tile_counts: dict[str, int] = field(default_factory=dict)
    num_subgraphs: int = 1
    slice_params: dict[str, dict[int, TensorSliceInfo]] = field(default_factory=dict)
    ops: list[TracedOp] = field(default_factory=list)
    output: str = OUTPUT_TENSOR_NAME
    reduction_tile_counts: dict[str, int] = field(default_factory=dict)

    def get_parallel_dims(self) -> list[str]:
        """Return dimension IDs that are parallel (tileable)."""
        return [d for d in self.dim_order if self.dim_info[d].iter_type == "parallel"]

    def get_reduction_dims(self) -> list[str]:
        """Return dimension IDs that are reduction."""
        return [d for d in self.dim_order if self.dim_info[d].iter_type == "reduction"]

    def compute_tile_counts(self) -> None:
        """Compute tile counts for each parallel dimension."""
        for dim_id in self.get_parallel_dims():
            size = self.dim_info[dim_id].size
            self.tile_counts[dim_id] = (size + TILE_SIZE - 1) // TILE_SIZE

        self.num_subgraphs = 1
        for count in self.tile_counts.values():
            self.num_subgraphs *= count

    def compute_reduction_tile_counts(self) -> None:
        """Compute tile counts for each reduction dimension.

        For each reduction dimension, computes the number of 128-size tiles
        as size // TILE_SIZE. This is used for reduction dimension tiling
        where the reduction is split into multiple partial computations.
        """
        for dim_id in self.get_reduction_dims():
            size = self.dim_info[dim_id].size
            self.reduction_tile_counts[dim_id] = size // TILE_SIZE

    def iter_reduction_tile_positions(self) -> Iterator[dict[str, int]]:
        """Yield {dim_id: tile_index} for each reduction tile position.

        Iterates over all combinations of reduction tile positions using
        itertools.product. If there are no reduction dimensions, yields
        a single empty dict.

        Yields:
            Dictionary mapping reduction dimension ID to tile index.
        """
        reduction_dims = self.get_reduction_dims()
        if not reduction_dims:
            yield {}
            return
        ranges = [range(self.reduction_tile_counts[d]) for d in reduction_dims]
        for positions in product(*ranges):
            yield dict(zip(reduction_dims, positions))

    def get_num_reduction_tiles(self) -> int:
        """Return total number of reduction tile iterations.

        Returns the product of all reduction tile counts. If there are no
        reduction dimensions, returns 1.

        Returns:
            Total number of reduction tile combinations.
        """
        if not self.reduction_tile_counts:
            return 1
        result = 1
        for count in self.reduction_tile_counts.values():
            result *= count
        return result

    def iter_tile_positions(self) -> Iterator[tuple[int, dict[str, int]]]:
        """Yield (subgraph_idx, {dim_id: tile_index}) for each subgraph."""
        parallel_dims = self.get_parallel_dims()
        ranges = [range(self.tile_counts[d]) for d in parallel_dims]
        for idx, positions in enumerate(product(*ranges)):
            yield idx, dict(zip(parallel_dims, positions))

    def compute_reduction_slice_params(
        self, tensor_id: str, parallel_position: dict[str, int], reduction_position: dict[str, int]
    ) -> TensorSliceInfo:
        """Compute slice parameters for a tensor at given parallel and reduction positions.

        Computes offsets and sizes for all dimensions of a tensor based on the
        current parallel tile position and reduction tile position.

        For each dimension:
        - Parallel dims: offset = tile_index * TILE_SIZE, size = TILE_SIZE
        - Reduction dims: offset = tile_index * TILE_SIZE, size = TILE_SIZE
        - Other dims: offset = 0, size = original_size

        Args:
            tensor_id: Name of the tensor.
            parallel_position: {parallel_dim_id: tile_index} from iter_tile_positions().
            reduction_position: {reduction_dim_id: tile_index} from iter_reduction_tile_positions().

        Returns:
            TensorSliceInfo with offsets, sizes, and strides for all dimensions.
        """
        dims = self.tensor_dims[tensor_id]
        shape = self.tensor_shapes[tensor_id]

        offsets: list[int] = []
        sizes: list[int] = []
        strides: list[int] = []

        for i, dim_id in enumerate(dims):
            original_size = shape[i] if i < len(shape) else 0

            if dim_id in parallel_position:
                tile_idx = parallel_position[dim_id]
                offsets.append(tile_idx * TILE_SIZE)
                sizes.append(TILE_SIZE)
            elif dim_id in reduction_position:
                tile_idx = reduction_position[dim_id]
                offsets.append(tile_idx * TILE_SIZE)
                sizes.append(TILE_SIZE)
            else:
                offsets.append(0)
                sizes.append(original_size)

            strides.append(1)

        return TensorSliceInfo(offsets=offsets, sizes=sizes, strides=strides)

    def compute_all_slice_params(self) -> None:
        """Compute slice parameters for all tensors at all tile positions.

        Populates self.slice_params with {tensor_id: {subgraph_idx: TensorSliceInfo}}.
        Must be called after compute_tile_counts().

        For each dimension of each tensor:
        - If parallel dim: offset = tile_index * TILE_SIZE, size = TILE_SIZE
        - If reduction dim: offset = 0, size = original_size
        """
        for tensor_id, dims in self.tensor_dims.items():
            if tensor_id not in self.tensor_shapes:
                continue

            shape = self.tensor_shapes[tensor_id]
            self.slice_params[tensor_id] = {}

            for subgraph_idx, tile_position in self.iter_tile_positions():
                offsets = []
                sizes = []
                strides = []

                for i, dim_id in enumerate(dims):
                    original_size = shape[i] if i < len(shape) else 0

                    if dim_id in tile_position:
                        tile_idx = tile_position[dim_id]
                        offsets.append(tile_idx * TILE_SIZE)
                        sizes.append(TILE_SIZE)
                    else:
                        offsets.append(0)
                        sizes.append(original_size)

                    strides.append(1)

                self.slice_params[tensor_id][subgraph_idx] = TensorSliceInfo(
                    offsets=offsets, sizes=sizes, strides=strides
                )

    def assert_equal(self, expected: "DimensionAnalysis") -> None:
        """Assert this analysis equals another, with detailed error messages.

        Compares all fields and provides detailed error messages on mismatch.

        Args:
            expected: The expected golden DimensionAnalysis.

        Raises:
            AssertionError: If any field differs between self and expected.
        """
        assert (
            self.dim_order == expected.dim_order
        ), f"dim_order mismatch:\n  actual: {self.dim_order}\n  expected: {expected.dim_order}"
        assert (
            self.dim_info == expected.dim_info
        ), f"dim_info mismatch:\n  actual: {self.dim_info}\n  expected: {expected.dim_info}"
        assert (
            self.tensor_dims == expected.tensor_dims
        ), f"tensor_dims mismatch:\n  actual: {self.tensor_dims}\n  expected: {expected.tensor_dims}"
        assert (
            self.tensor_shapes == expected.tensor_shapes
        ), f"tensor_shapes mismatch:\n  actual: {self.tensor_shapes}\n  expected: {expected.tensor_shapes}"
        assert (
            self.tile_counts == expected.tile_counts
        ), f"tile_counts mismatch:\n  actual: {self.tile_counts}\n  expected: {expected.tile_counts}"
        assert (
            self.num_subgraphs == expected.num_subgraphs
        ), f"num_subgraphs mismatch:\n  actual: {self.num_subgraphs}\n  expected: {expected.num_subgraphs}"

        actual_positions = list(self.iter_tile_positions())
        expected_positions = list(expected.iter_tile_positions())
        assert (
            actual_positions == expected_positions
        ), f"iter_tile_positions mismatch:\n  actual: {actual_positions}\n  expected: {expected_positions}"

        assert (
            self.slice_params == expected.slice_params
        ), f"slice_params mismatch:\n  actual: {self.slice_params}\n  expected: {expected.slice_params}"

        assert (
            self.output == expected.output
        ), f"output mismatch:\n  actual: {self.output}\n  expected: {expected.output}"

        assert (
            self.reduction_tile_counts == expected.reduction_tile_counts
        ), f"reduction_tile_counts mismatch:\n  actual: {self.reduction_tile_counts}\n  expected: {expected.reduction_tile_counts}"

        actual_reduction_positions = list(self.iter_reduction_tile_positions())
        expected_reduction_positions = list(expected.iter_reduction_tile_positions())
        assert (
            actual_reduction_positions == expected_reduction_positions
        ), f"iter_reduction_tile_positions mismatch:\n  actual: {actual_reduction_positions}\n  expected: {expected_reduction_positions}"

    def __repr__(self) -> str:
        """Return formatted string representation of the analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("Dimension Analysis")
        lines.append("=" * 60)
        lines.append(f"Dimensions: {', '.join(self.dim_order)}")
        lines.append("")
        lines.append("Tensor mappings:")
        for tensor_id, dims in self.tensor_dims.items():
            lines.append(f"  {tensor_id}: ({', '.join(dims)})")
        lines.append("")
        lines.append(f"Output: {self.output}")
        lines.append("")
        lines.append("Iterator types:")
        for dim_id in self.dim_order:
            lines.append(f"  {self.dim_info[dim_id]}")
        lines.append("")
        lines.append(f"Parallel dims (tileable): {self.get_parallel_dims()}")
        lines.append(f"Reduction dims: {self.get_reduction_dims()}")
        lines.append("")
        lines.append(f"Tile counts: {self.tile_counts}")
        lines.append(f"Number of subgraphs: {self.num_subgraphs}")
        lines.append("")
        lines.append(f"Reduction tile counts: {self.reduction_tile_counts}")
        lines.append(f"Number of reduction tiles: {self.get_num_reduction_tiles()}")
        lines.append("")
        lines.append("Tile positions:")
        for idx, pos in self.iter_tile_positions():
            lines.append(f"  subgraph {idx}: {pos}")
        lines.append("")
        lines.append("Reduction tile positions:")
        for pos in self.iter_reduction_tile_positions():
            lines.append(f"  {pos}")
        lines.append("")
        lines.append("Slice parameters:")
        for tensor_id, positions in self.slice_params.items():
            lines.append(f"  {tensor_id}:")
            for subgraph_idx, info in positions.items():
                lines.append(f"    subgraph {subgraph_idx}: {info}")
        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_dimension(func: Callable[..., np.ndarray], input_shapes: dict[str, tuple[int, ...]]) -> DimensionAnalysis:
    """Analyze a NumPy function to extract dimension information.

    Traces the function with symbolic tensors to determine:
    - Which dimensions are shared across tensors
    - Which dimensions are parallel vs reduction
    - Tile counts and slice parameters for each subgraph

    Implementation: Uses a shared mutable tracker pattern where all TracedTensor
    instances reference a single _DimTracker. Operations mutate the tracker to
    record dimension equivalences (e.g., matmul unifies contraction dims). After
    tracing, the tracker's final state encodes all dimension relationships. This
    pattern is appropriate because tracing is single-threaded and one-shot.

    Args:
        func: NumPy function to analyze (e.g., lambda a, b: np.matmul(a, b)).
        input_shapes: Maps parameter name to shape tuple.

    Returns:
        DimensionAnalysis with dimension info, tile counts, and slice params.

    Example:
        >>> def matmul(a, b):
        ...     return np.matmul(a, b)
        >>> analysis = analyze_dimension(matmul, {"a": (256, 128), "b": (128, 256)})
        >>> analysis.dim_order
        ['d0', 'd1', 'd2']
        >>> analysis.get_parallel_dims()
        ['d0', 'd2']
    """
    _validate_input_names(input_shapes)
    tracker = _DimTracker()
    traced_tensors: dict[str, TracedTensor] = {}

    for name, shape in input_shapes.items():
        dims = [tracker.new_dim(size) for size in shape]
        traced_tensors[name] = TracedTensor(name, shape, dims, tracker)

    result = func(**traced_tensors)

    if not isinstance(result, TracedTensor):
        raise TypeError(f"Function must return TracedTensor, got {type(result)}")

    if result.tracker is not tracker:
        raise RuntimeError(
            "Result TracedTensor references a different tracker. "
            "This indicates a bug in operation handling - all TracedTensors "
            "should share the same tracker instance created at analysis start."
        )

    output_dims = set(tracker.get_canonical_dims(result.dims))
    canonical_order = tracker.get_canonical_order()

    dim_remap: dict[str, str] = {old_id: f"d{i}" for i, old_id in enumerate(canonical_order)}

    analysis = DimensionAnalysis()
    analysis.dim_order = [dim_remap[d] for d in canonical_order]

    for dim_id in canonical_order:
        new_id = dim_remap[dim_id]
        iter_type = "parallel" if dim_id in output_dims else "reduction"
        analysis.dim_info[new_id] = DimInfo(id=new_id, size=tracker.dim_sizes[dim_id], iter_type=iter_type)

    for name, tensor in traced_tensors.items():
        canonical_dims = tracker.get_canonical_dims(tensor.dims)
        analysis.tensor_dims[name] = [dim_remap[d] for d in canonical_dims]
        analysis.tensor_shapes[name] = tensor.shape

    analysis.ops = tracker.ops

    result_canonical_dims = tracker.get_canonical_dims(result.dims)
    analysis.tensor_dims[OUTPUT_TENSOR_NAME] = [dim_remap[d] for d in result_canonical_dims]
    analysis.tensor_shapes[OUTPUT_TENSOR_NAME] = result.shape

    if analysis.ops:
        final_op = analysis.ops[-1]
        analysis.ops[-1] = TracedOp(final_op.op_name, final_op.inputs, OUTPUT_TENSOR_NAME)

    analysis.compute_tile_counts()
    analysis.compute_reduction_tile_counts()
    analysis.compute_all_slice_params()

    return analysis


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

    Uses the OP_SEMANTICS table for operator-agnostic code generation:
    - generate_expr(inputs): Creates the initial computation
    - combine_partials(result_var, inputs): Creates in-place accumulation

    Args:
        func: NumPy function to tile.
        input_shapes: Maps parameter name to shape tuple.

    Returns:
        Python source code string for the tiled function.

    Raises:
        NotImplementedError: If an operator is not in OP_SEMANTICS or
            has reduction dimensions but no combine_partials function.
    """
    analysis = analyze_dimension(func, input_shapes)
    logger.debug(analysis)
    param_names = list(input_shapes.keys())

    lines: list[str] = []
    output_name = OUTPUT_TENSOR_NAME

    lines.append(f"def tiled_{func.__name__}({', '.join(param_names)}):")
    output_shape = analysis.tensor_shapes[OUTPUT_TENSOR_NAME]
    lines.append(f"    {output_name} = np.empty({output_shape}, dtype=np.float32)")

    reduction_positions = list(analysis.iter_reduction_tile_positions())
    has_reduction_tiling = len(reduction_positions) > 1 or (len(reduction_positions) == 1 and reduction_positions[0])

    name_gen = TensorNameGenerator()

    for subgraph_idx, parallel_pos in analysis.iter_tile_positions():
        if has_reduction_tiling:
            acc_var: str | None = None

            for red_idx, reduction_pos in enumerate(reduction_positions):
                intermediate_map: dict[str, str] = {}

                for op in analysis.ops:
                    if op.op_name not in OP_SEMANTICS:
                        raise NotImplementedError(f"Operator '{op.op_name}' not supported")

                    semantics = OP_SEMANTICS[op.op_name]
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
                            expr = semantics.generate_expr(op_inputs)
                            lines.append(f"    {result_var} = {expr}")
                            acc_var = result_var
                        else:
                            if semantics.combine_partials is None:
                                raise NotImplementedError(
                                    f"Operator '{semantics.op_name}' has reduction dimensions "
                                    "but no combine_partials"
                                )
                            acc_expr = semantics.combine_partials(acc_var, op_inputs)
                            lines.append(f"    {acc_expr}")
                        intermediate_map[op.output] = acc_var
                    else:
                        result_var = name_gen.next_name()
                        expr = semantics.generate_expr(op_inputs)
                        lines.append(f"    {result_var} = {expr}")
                        intermediate_map[op.output] = result_var

            output_slice = analysis.compute_reduction_slice_params(OUTPUT_TENSOR_NAME, parallel_pos, {})
            row_off, col_off = output_slice.offsets[0], output_slice.offsets[1]
            row_sz, col_sz = output_slice.sizes[0], output_slice.sizes[1]
            lines.append(f"    {output_name}[{row_off}:{row_off + row_sz}, {col_off}:{col_off + col_sz}] = {acc_var}\n")
        else:
            intermediate_map = {}

            for op in analysis.ops:
                if op.op_name not in OP_SEMANTICS:
                    raise NotImplementedError(f"Operator '{op.op_name}' not supported")

                semantics = OP_SEMANTICS[op.op_name]
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

                expr = semantics.generate_expr(op_inputs)
                lines.append(f"    {output_var} = {expr}")

            output_slice = analysis.slice_params[OUTPUT_TENSOR_NAME][subgraph_idx]
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
