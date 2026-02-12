"""Dimension analysis for NKI Gym tiling.

Analyzes NumPy functions to extract dimension information and compute
tile parameters for data-parallel subgraph generation.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np

from nkigym.ops import tracing_enabled
from nkigym.tiling.dim_tracker import TracedOp, _DimTracker
from nkigym.tiling.tensor import TracedTensor

TILE_SIZE = 128
OUTPUT_TENSOR_NAME = "output"
RESERVED_INTERMEDIATE_PATTERN = re.compile(r"^tensor_\d+$")


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
    iter_type: Literal["parallel", "reduction"] = "parallel"

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

    def _compute_slice_for_position(self, tensor_id: str, position: dict[str, int]) -> TensorSliceInfo:
        """Compute slice parameters for a tensor at a single merged tile position.

        For each dimension of the tensor:
        - If the dimension appears in ``position``: offset = tile_index * TILE_SIZE,
          size = TILE_SIZE.
        - Otherwise: offset = 0, size = original dimension size.

        Args:
            tensor_id: Name of the tensor.
            position: Merged mapping of {dim_id: tile_index} covering both
                parallel and reduction dimensions as needed.

        Returns:
            TensorSliceInfo with offsets, sizes, and strides for all dimensions.

        Raises:
            ValueError: If the tensor's dimension count does not match its shape.
        """
        dims = self.tensor_dims[tensor_id]
        shape = self.tensor_shapes[tensor_id]

        if len(dims) != len(shape):
            raise ValueError(f"Dimension mismatch for '{tensor_id}': {len(dims)} dims vs {len(shape)} shape")

        offsets: list[int] = []
        sizes: list[int] = []
        strides: list[int] = []

        for i, dim_id in enumerate(dims):
            if dim_id in position:
                offsets.append(position[dim_id] * TILE_SIZE)
                sizes.append(TILE_SIZE)
            else:
                offsets.append(0)
                sizes.append(shape[i])
            strides.append(1)

        return TensorSliceInfo(offsets=offsets, sizes=sizes, strides=strides)

    def compute_reduction_slice_params(
        self, tensor_id: str, parallel_position: dict[str, int], reduction_position: dict[str, int]
    ) -> TensorSliceInfo:
        """Compute slice parameters for a tensor at given parallel and reduction positions.

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
        return self._compute_slice_for_position(tensor_id, {**parallel_position, **reduction_position})

    def compute_all_slice_params(self) -> None:
        """Compute slice parameters for all tensors at all tile positions.

        Populates self.slice_params with {tensor_id: {subgraph_idx: TensorSliceInfo}}.
        Must be called after compute_tile_counts().

        For each dimension of each tensor:
        - If parallel dim: offset = tile_index * TILE_SIZE, size = TILE_SIZE
        - If reduction dim: offset = 0, size = original_size
        """
        for tensor_id in self.tensor_dims:
            if tensor_id not in self.tensor_shapes:
                continue
            self.slice_params[tensor_id] = {}
            for subgraph_idx, tile_position in self.iter_tile_positions():
                self.slice_params[tensor_id][subgraph_idx] = self._compute_slice_for_position(tensor_id, tile_position)

    def __repr__(self) -> str:
        """Return formatted string representation of the analysis."""
        return format_analysis(self)


def format_analysis(analysis: "DimensionAnalysis") -> str:
    """Format a DimensionAnalysis as a human-readable multi-line string.

    Args:
        analysis: The analysis to format.

    Returns:
        Formatted string with dimension info, tile counts, and slice params.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Dimension Analysis")
    lines.append("=" * 60)
    lines.append(f"Dimensions: {', '.join(analysis.dim_order)}")
    lines.append("")
    lines.append("Tensor mappings:")
    for tensor_id, dims in analysis.tensor_dims.items():
        lines.append(f"  {tensor_id}: ({', '.join(dims)})")
    lines.append("")
    lines.append(f"Output: {analysis.output}")
    lines.append("")
    lines.append("Iterator types:")
    for dim_id in analysis.dim_order:
        lines.append(f"  {analysis.dim_info[dim_id]}")
    lines.append("")
    lines.append(f"Parallel dims (tileable): {analysis.get_parallel_dims()}")
    lines.append(f"Reduction dims: {analysis.get_reduction_dims()}")
    lines.append("")
    lines.append(f"Tile counts: {analysis.tile_counts}")
    lines.append(f"Number of subgraphs: {analysis.num_subgraphs}")
    lines.append("")
    lines.append(f"Reduction tile counts: {analysis.reduction_tile_counts}")
    lines.append(f"Number of reduction tiles: {analysis.get_num_reduction_tiles()}")
    lines.append("")
    lines.append("Tile positions:")
    for idx, pos in analysis.iter_tile_positions():
        lines.append(f"  subgraph {idx}: {pos}")
    lines.append("")
    lines.append("Reduction tile positions:")
    for pos in analysis.iter_reduction_tile_positions():
        lines.append(f"  {pos}")
    lines.append("")
    lines.append("Slice parameters:")
    for tensor_id, positions in analysis.slice_params.items():
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

    with tracing_enabled():
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
