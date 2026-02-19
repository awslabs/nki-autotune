"""Static dimension analysis for tiling GymPrograms.

Derives parallel/reduction dimension classification, tile counts, and
slice parameters by walking GymProgram statements and unifying axis
labels from GymOp Tensor descriptors.
"""

from dataclasses import dataclass, field
from itertools import product

from nkigym.ir.tensor import TensorRef, ref_name
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.ops.base import GymOp

TILE_SIZE = 128
OUTPUT_TENSOR_NAME = "output"


@dataclass
class DimInfo:
    """Information about a single global dimension.

    Attributes:
        id: Unique dimension identifier (e.g., ``"d0"``).
        size: Size of the dimension.
        iter_type: ``"parallel"`` if in output, ``"reduction"`` otherwise.
    """

    id: str
    size: int
    iter_type: str = "parallel"


@dataclass
class TilingAnalysis:
    """Result of static dimension analysis for a GymProgram.

    Attributes:
        var_dims: Maps variable name to tuple of dim IDs
            (``None`` for fixed-size axes).
        var_shapes: Maps variable name to shape tuple.
        dim_info: Maps dim ID to ``DimInfo``.
        parallel_dims: Dim IDs that appear in the output (tiled independently).
        reduction_dims: Dim IDs that do not appear in the output (contracted).
        tile_counts: Parallel dim ID to number of tiles.
        reduction_tile_counts: Reduction dim ID to number of tiles.
        ops: Original program statements.
        return_var: Name of the return variable.
    """

    var_dims: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    var_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dim_info: dict[str, DimInfo] = field(default_factory=dict)
    parallel_dims: list[str] = field(default_factory=list)
    reduction_dims: list[str] = field(default_factory=list)
    tile_counts: dict[str, int] = field(default_factory=dict)
    reduction_tile_counts: dict[str, int] = field(default_factory=dict)
    ops: tuple[GymStatement, ...] = ()
    return_var: str = ""

    def iter_tile_positions(self):
        """Yield ``(subgraph_idx, {dim_id: tile_index})`` for parallel tiles.

        Yields:
            Tuples of (sequential index, position dict).
        """
        if not self.parallel_dims:
            yield 0, {}
            return
        ranges = [range(self.tile_counts[d]) for d in self.parallel_dims]
        for idx, positions in enumerate(product(*ranges)):
            yield idx, dict(zip(self.parallel_dims, positions))

    def iter_reduction_positions(self):
        """Yield ``{dim_id: tile_index}`` for each reduction tile combination.

        If there are no reduction dimensions, yields a single empty dict.

        Yields:
            Position dict mapping reduction dim ID to tile index.
        """
        if not self.reduction_dims:
            yield {}
            return
        ranges = [range(self.reduction_tile_counts[d]) for d in self.reduction_dims]
        for positions in product(*ranges):
            yield dict(zip(self.reduction_dims, positions))

    def compute_slices(self, var_name: str, position: dict[str, int]) -> list[tuple[int, int]]:
        """Compute slice bounds for a variable at a given tile position.

        For each axis of the variable: if the dim ID appears in
        ``position``, compute ``tile_idx * TILE_SIZE`` offset with size
        ``TILE_SIZE``. Otherwise use the full range ``(0, dim_size)``.
        For fixed-size axes (``None`` dim), always use full range.

        Args:
            var_name: Variable name.
            position: Combined parallel + reduction position dict.

        Returns:
            List of ``(start, stop)`` pairs, one per axis.
        """
        dims = self.var_dims[var_name]
        shape = self.var_shapes[var_name]
        slices: list[tuple[int, int]] = []
        for dim_id, size in zip(dims, shape):
            if dim_id is not None and dim_id in position:
                offset = position[dim_id] * TILE_SIZE
                slices.append((offset, offset + TILE_SIZE))
            else:
                slices.append((0, size))
        return slices

    def has_reduction_tiling(self) -> bool:
        """Check if any reduction dimension requires multiple tiles.

        Returns:
            True if there is at least one reduction dimension with more
            than one tile.
        """
        return any(count > 1 for count in self.reduction_tile_counts.values())


def _extract_param_shapes(program: GymProgram) -> dict[str, tuple[int, ...]]:
    """Extract parameter shapes from TensorRef kwargs in program statements.

    Scans all statements for TensorRef values matching parameter names
    and reads their shapes.

    Args:
        program: A specialized GymProgram.

    Returns:
        Mapping from parameter name to shape tuple.
    """
    param_set = set(program.params)
    shapes: dict[str, tuple[int, ...]] = {}
    for stmt in program.stmts:
        for _, value in stmt.kwargs:
            if isinstance(value, TensorRef) and value.name in param_set:
                if value.name not in shapes:
                    shapes[value.name] = value.shape
    return shapes


def analyze_tiling(program: GymProgram) -> TilingAnalysis:
    """Analyze a GymProgram to derive tiling parameters.

    Walks the program statements, unifies axis labels from GymOp Tensor
    descriptors, classifies dimensions as parallel or reduction, and
    computes tile counts. Reads param shapes from TensorRef on kwargs.

    Args:
        program: A specialized GymProgram with TensorRef on all tensor kwargs.

    Returns:
        ``TilingAnalysis`` with all dimension metadata.

    Raises:
        KeyError: If a statement references an unknown variable.
        ValueError: If dimension sizes conflict during unification.
    """
    analysis = TilingAnalysis(ops=program.stmts, return_var=program.return_var)

    input_shapes = _extract_param_shapes(program)

    dim_counter = 0
    rename_map: dict[str, str] = {}

    def _canonical(dim_id: str) -> str:
        """Follow rename chain to canonical dim ID."""
        while dim_id in rename_map:
            dim_id = rename_map[dim_id]
        return dim_id

    def _unify(dim_a: str, dim_b: str) -> None:
        """Unify two dim IDs, keeping the lower-numbered one."""
        ca = _canonical(dim_a)
        cb = _canonical(dim_b)
        if ca == cb:
            return
        info_a = analysis.dim_info[ca]
        info_b = analysis.dim_info[cb]
        if info_a.size != info_b.size:
            raise ValueError(f"Dimension size conflict: {ca}={info_a.size} vs {cb}={info_b.size}")
        rename_map[cb] = ca

    for param in program.params:
        shape = input_shapes[param]
        dims: list[str | None] = []
        for size in shape:
            dim_id = f"d{dim_counter}"
            dim_counter += 1
            analysis.dim_info[dim_id] = DimInfo(id=dim_id, size=size)
            dims.append(dim_id)
        analysis.var_dims[param] = tuple(dims)
        analysis.var_shapes[param] = shape

    for stmt in program.stmts:
        op_cls = GymOp.get(stmt.op)
        n_inputs = len(op_cls.inputs)
        tensor_kwargs = stmt.kwargs[:n_inputs]

        axis_to_dim: dict[str, str] = {}

        for tensor_desc, (_, var_ref) in zip(op_cls.inputs, tensor_kwargs):
            var_name = ref_name(var_ref)
            var_dim_ids = analysis.var_dims[var_name]
            for axis_idx, axis_label in enumerate(tensor_desc.axes):
                if isinstance(axis_label, int):
                    continue
                var_dim = _canonical(var_dim_ids[axis_idx])
                if axis_label in axis_to_dim:
                    existing = axis_to_dim[axis_label]
                    _unify(existing, var_dim)
                    axis_to_dim[axis_label] = _canonical(existing)
                else:
                    axis_to_dim[axis_label] = var_dim

        output_desc = op_cls.outputs[0]
        output_dims: list[str | None] = []
        output_shape: list[int] = []
        for axis_label in output_desc.axes:
            if isinstance(axis_label, int):
                output_dims.append(None)
                output_shape.append(axis_label)
            else:
                dim_id = _canonical(axis_to_dim[axis_label])
                output_dims.append(dim_id)
                output_shape.append(analysis.dim_info[dim_id].size)

        out_name = stmt.output.name
        analysis.var_dims[out_name] = tuple(output_dims)
        analysis.var_shapes[out_name] = tuple(output_shape)

    return_dims = set()
    for d in analysis.var_dims[program.return_var]:
        if d is not None:
            return_dims.add(_canonical(d))

    all_dims: dict[str, DimInfo] = {}
    for dim_id, info in analysis.dim_info.items():
        canon = _canonical(dim_id)
        if canon not in all_dims:
            all_dims[canon] = info

    analysis.dim_info = all_dims

    for dim_id, info in all_dims.items():
        if info.size % TILE_SIZE != 0:
            raise ValueError(
                f"Dimension {dim_id} has size {info.size} which is not " f"divisible by TILE_SIZE={TILE_SIZE}"
            )
        if dim_id in return_dims:
            info.iter_type = "parallel"
            analysis.parallel_dims.append(dim_id)
            analysis.tile_counts[dim_id] = info.size // TILE_SIZE
        else:
            info.iter_type = "reduction"
            analysis.reduction_dims.append(dim_id)
            analysis.reduction_tile_counts[dim_id] = info.size // TILE_SIZE

    canon_var_dims: dict[str, tuple[str | None, ...]] = {}
    for var_name, dims in analysis.var_dims.items():
        canon_var_dims[var_name] = tuple(_canonical(d) if d is not None else None for d in dims)
    analysis.var_dims = canon_var_dims

    return analysis
