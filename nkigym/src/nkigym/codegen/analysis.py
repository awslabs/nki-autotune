"""Dimension analysis for AST-based codegen.

Unifies axis labels across ops, classifies dimensions as parallel
or reduction, and computes tile counts and slice bounds.
"""

from typing import NamedTuple

from nkigym.ops.base import NKIOp, _get_output_axes_tuple

_TILE = 128


class _OpCall(NamedTuple):
    """Parsed operation from the user function AST.

    Attributes:
        stmt_type: NKIOp subclass for axis metadata lookup.
        input_vars: Positional input variable names.
        config_kwargs: Non-tensor keyword arguments as (key, value) pairs.
        output_var: Variable name assigned to this op's result.
    """

    stmt_type: type[NKIOp]
    input_vars: tuple[str, ...]
    config_kwargs: tuple[tuple[str, object], ...]
    output_var: str


class _DimInfo(NamedTuple):
    """Information about a single global dimension.

    Attributes:
        dim_id: Unique identifier (e.g. ``"d0"``).
        size: Dimension size.
    """

    dim_id: str
    size: int


class _Analysis(NamedTuple):
    """Result of dimension analysis.

    Attributes:
        var_dims: Maps variable name to tuple of dim IDs (None for fixed).
        var_shapes: Maps variable name to shape tuple.
        parallel_dims: Ordered list of parallel dim IDs.
        reduction_dims: Ordered list of reduction dim IDs.
        tile_counts: Parallel dim ID to tile count.
        reduction_tile_counts: Reduction dim ID to tile count.
        dim_tile_sizes: Dim ID to tile size (from op TILE_LIMITS).
        return_var: Name of the return variable.
    """

    var_dims: dict[str, tuple[str | None, ...]]
    var_shapes: dict[str, tuple[int, ...]]
    parallel_dims: list[str]
    reduction_dims: list[str]
    tile_counts: dict[str, int]
    reduction_tile_counts: dict[str, int]
    dim_tile_sizes: dict[str, int]
    return_var: str


def analyze_dims(
    op_calls: list[_OpCall], params: tuple[str, ...], input_shapes: tuple[tuple[int, ...], ...]
) -> _Analysis:
    """Perform dimension analysis on parsed op calls.

    Args:
        op_calls: Parsed operation calls from the user function.
        params: Parameter names.
        input_shapes: Shape of each input parameter.

    Returns:
        Dimension analysis result.
    """
    var_dims, var_shapes, dim_info, _dim_counter, rename_map = _init_params(params, input_shapes)
    dim_to_tile_size = _unify_ops(op_calls, var_dims, var_shapes, dim_info, rename_map)
    return_var = op_calls[-1].output_var
    return _classify_dims(var_dims, var_shapes, dim_info, rename_map, return_var, dim_to_tile_size)


def _init_params(
    params: tuple[str, ...], input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[int, ...]], dict[str, _DimInfo], list[int], dict[str, str]]:
    """Initialize dimension tracking from input parameters.

    Args:
        params: Parameter names.
        input_shapes: Shape of each input parameter.

    Returns:
        Tuple of (var_dims, var_shapes, dim_info, dim_counter, rename_map).
    """
    var_dims: dict[str, tuple[str, ...]] = {}
    var_shapes: dict[str, tuple[int, ...]] = {}
    dim_info: dict[str, _DimInfo] = {}
    dim_counter = [0]
    rename_map: dict[str, str] = {}
    for param, shape in zip(params, input_shapes):
        dims: list[str] = []
        for size in shape:
            dim_id = f"d{dim_counter[0]}"
            dim_counter[0] += 1
            dim_info[dim_id] = _DimInfo(dim_id=dim_id, size=size)
            dims.append(dim_id)
        var_dims[param] = tuple(dims)
        var_shapes[param] = shape
    return (var_dims, var_shapes, dim_info, dim_counter, rename_map)


def _canonical(dim_id: str, rename_map: dict[str, str]) -> str:
    """Follow rename chain to canonical dim ID.

    Args:
        dim_id: Starting dimension ID.
        rename_map: Rename chain mapping.

    Returns:
        Canonical dimension ID.
    """
    while dim_id in rename_map:
        dim_id = rename_map[dim_id]
    return dim_id


def _unify(dim_a: str, dim_b: str, dim_info: dict[str, _DimInfo], rename_map: dict[str, str]) -> None:
    """Unify two dimension IDs, keeping the lower-numbered one.

    Args:
        dim_a: First dimension ID.
        dim_b: Second dimension ID.
        dim_info: Dimension info lookup.
        rename_map: Mutable rename chain.
    """
    ca = _canonical(dim_a, rename_map)
    cb = _canonical(dim_b, rename_map)
    if ca != cb:
        if dim_info[ca].size != dim_info[cb].size:
            raise ValueError(f"Dimension size conflict: {ca}={dim_info[ca].size} vs {cb}={dim_info[cb].size}")
        rename_map[cb] = ca


def _unify_axis(
    axis_label: str,
    var_dim: str,
    axis_to_dim: dict[str, str],
    dim_info: dict[str, _DimInfo],
    rename_map: dict[str, str],
) -> None:
    """Unify a single axis label with its corresponding variable dim.

    Args:
        axis_label: The axis label (e.g. ``"K"``, ``"M"``).
        var_dim: The variable's dimension ID for this axis.
        axis_to_dim: Mutable axis-to-dim accumulator.
        dim_info: Dimension info lookup.
        rename_map: Mutable rename chain.
    """
    canonical_dim = _canonical(var_dim, rename_map)
    if axis_label in axis_to_dim:
        existing = axis_to_dim[axis_label]
        _unify(existing, canonical_dim, dim_info, rename_map)
        axis_to_dim[axis_label] = _canonical(existing, rename_map)
    else:
        axis_to_dim[axis_label] = canonical_dim


def _unify_one_op(
    op_call: _OpCall,
    var_dims: dict[str, tuple[str, ...]],
    var_shapes: dict[str, tuple[int, ...]],
    dim_info: dict[str, _DimInfo],
    rename_map: dict[str, str],
    dim_to_tile_size: dict[str, int],
) -> None:
    """Unify dimensions for a single op call and register its output.

    Args:
        op_call: The parsed op call.
        var_dims: Mutable variable-to-dims mapping.
        var_shapes: Mutable variable-to-shape mapping.
        dim_info: Mutable dimension info mapping.
        rename_map: Mutable rename chain.
        dim_to_tile_size: Mutable dim ID to tile size mapping.
    """
    operand_axes: dict[str, tuple[str, ...]] = getattr(op_call.stmt_type, "OPERAND_AXES", {})
    output_axes: tuple[str, ...] = _get_output_axes_tuple(op_call.stmt_type)
    operand_names = list(operand_axes.keys())
    axis_to_dim: dict[str, str] = {}
    for operand_name, var_name in zip(operand_names, op_call.input_vars):
        axes = operand_axes[operand_name]
        var_dim_ids = var_dims[var_name]
        for axis_idx, axis_label in enumerate(axes):
            if axis_idx < len(var_dim_ids):
                _unify_axis(axis_label, var_dim_ids[axis_idx], axis_to_dim, dim_info, rename_map)
    tile_limits: dict[str, int] = getattr(op_call.stmt_type, "MAX_TILE_SIZES", {})
    for axis, limit in tile_limits.items():
        if axis in axis_to_dim:
            dim_id = _canonical(axis_to_dim[axis], rename_map)
            dim_to_tile_size[dim_id] = max(dim_to_tile_size.get(dim_id, 0), limit)
    _register_output(op_call, output_axes, axis_to_dim, var_dims, var_shapes, dim_info, rename_map)


def _register_output(
    op_call: _OpCall,
    output_axes: tuple[str, ...],
    axis_to_dim: dict[str, str],
    var_dims: dict[str, tuple[str, ...]],
    var_shapes: dict[str, tuple[int, ...]],
    dim_info: dict[str, _DimInfo],
    rename_map: dict[str, str],
) -> None:
    """Register the output variable's dimensions and shape.

    Args:
        op_call: The parsed op call.
        output_axes: Output axis labels from the stmt type.
        axis_to_dim: Axis-to-dim mapping from operand unification.
        var_dims: Mutable variable-to-dims mapping.
        var_shapes: Mutable variable-to-shape mapping.
        dim_info: Dimension info lookup.
        rename_map: Rename chain.
    """
    out_dims: list[str] = []
    out_shape: list[int] = []
    for axis_label in output_axes:
        if axis_label not in axis_to_dim:
            continue
        dim_id = _canonical(axis_to_dim[axis_label], rename_map)
        out_dims.append(dim_id)
        out_shape.append(dim_info[dim_id].size)
    var_dims[op_call.output_var] = tuple(out_dims)
    var_shapes[op_call.output_var] = tuple(out_shape)


def _unify_ops(
    op_calls: list[_OpCall],
    var_dims: dict[str, tuple[str, ...]],
    var_shapes: dict[str, tuple[int, ...]],
    dim_info: dict[str, _DimInfo],
    rename_map: dict[str, str],
) -> dict[str, int]:
    """Unify dimensions across all op calls.

    Args:
        op_calls: All parsed op calls.
        var_dims: Mutable variable-to-dims mapping.
        var_shapes: Mutable variable-to-shape mapping.
        dim_info: Mutable dimension info mapping.
        rename_map: Mutable rename chain.

    Returns:
        Mapping from canonical dim ID to tile size from op TILE_LIMITS.
    """
    dim_to_tile_size: dict[str, int] = {}
    for op_call in op_calls:
        _unify_one_op(op_call, var_dims, var_shapes, dim_info, rename_map, dim_to_tile_size)
    return dim_to_tile_size


def _collect_unique_dims(dim_info: dict[str, _DimInfo], rename_map: dict[str, str]) -> dict[str, _DimInfo]:
    """Collect unique canonical dimensions.

    Args:
        dim_info: All dimension info entries.
        rename_map: Rename chain.

    Returns:
        Mapping from canonical dim ID to DimInfo.
    """
    all_dims: dict[str, _DimInfo] = {}
    for dim_id, info in dim_info.items():
        canon = _canonical(dim_id, rename_map)
        if canon not in all_dims:
            all_dims[canon] = info
    return all_dims


def _canonicalize_var_dims(
    var_dims: dict[str, tuple[str, ...]], rename_map: dict[str, str]
) -> dict[str, tuple[str | None, ...]]:
    """Canonicalize all variable dimension mappings.

    Args:
        var_dims: Variable-to-dims mapping.
        rename_map: Rename chain.

    Returns:
        Canonicalized mapping.
    """
    canon: dict[str, tuple[str | None, ...]] = {}
    for var_name, dims in var_dims.items():
        canon[var_name] = tuple(_canonical(d, rename_map) for d in dims)
    return canon


def _classify_dims(
    var_dims: dict[str, tuple[str, ...]],
    var_shapes: dict[str, tuple[int, ...]],
    dim_info: dict[str, _DimInfo],
    rename_map: dict[str, str],
    return_var: str,
    dim_to_tile_size: dict[str, int],
) -> _Analysis:
    """Classify dimensions as parallel or reduction and compute tile counts.

    Args:
        var_dims: Variable-to-dims mapping.
        var_shapes: Variable-to-shape mapping.
        dim_info: Dimension info lookup.
        rename_map: Rename chain.
        return_var: Return variable name.
        dim_to_tile_size: Dim ID to tile size from op TILE_LIMITS.

    Returns:
        Complete analysis result.
    """
    return_dims = {_canonical(d, rename_map) for d in var_dims[return_var]}
    all_dims = _collect_unique_dims(dim_info, rename_map)
    parallel_dims: list[str] = []
    reduction_dims: list[str] = []
    tile_counts: dict[str, int] = {}
    reduction_tile_counts: dict[str, int] = {}
    dim_tile_sizes: dict[str, int] = {}
    for dim_id, info in all_dims.items():
        preferred = min(dim_to_tile_size.get(dim_id, _TILE), info.size)
        tile_size = preferred if info.size % preferred == 0 else _TILE
        if info.size % tile_size != 0:
            raise ValueError(f"Dimension {dim_id} size {info.size} not divisible by {tile_size}")
        dim_tile_sizes[dim_id] = tile_size
        if dim_id in return_dims:
            parallel_dims.append(dim_id)
            tile_counts[dim_id] = info.size // tile_size
        else:
            reduction_dims.append(dim_id)
            reduction_tile_counts[dim_id] = info.size // tile_size
    return _Analysis(
        var_dims=_canonicalize_var_dims(var_dims, rename_map),
        var_shapes=var_shapes,
        parallel_dims=parallel_dims,
        reduction_dims=reduction_dims,
        tile_counts=tile_counts,
        reduction_tile_counts=reduction_tile_counts,
        dim_tile_sizes=dim_tile_sizes,
        return_var=return_var,
    )


def has_reduction(op_call: _OpCall) -> bool:
    """Check if an op has any reduction dimensions.

    An op has reduction if any operand axis label is not in OUTPUT_AXES.

    Args:
        op_call: Parsed op call to check.

    Returns:
        True if the op has at least one reduction dimension.
    """
    operand_axes: dict[str, tuple[str, ...]] = getattr(op_call.stmt_type, "OPERAND_AXES", {})
    output_axes: tuple[str, ...] = _get_output_axes_tuple(op_call.stmt_type)
    output_set = set(output_axes)
    all_input_axes: set[str] = set()
    for axes in operand_axes.values():
        all_input_axes.update(axes)
    return bool(all_input_axes - output_set)
