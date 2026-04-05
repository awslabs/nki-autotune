"""Eager mode tensor construction helpers.

Functions that build Tensor IR objects from traced metadata,
compute loop index expressions, and build RenderContext.
"""

from nkigym.codegen.eager_trace import EagerTracer
from nkigym.codegen.eager_types import SBUF_PMAX, DimInfo, TensorInfo, TracedOp
from nkigym.codegen.ir import RenderContext, Tensor

EMPTY_STR_INT: dict[str, int] = {}
EMPTY_STR_SET: frozenset[str] = frozenset()


def _build_tensor(
    name: str,
    tinfo: TensorInfo,
    dims: dict[str, DimInfo],
    location: str,
    override_num_blocks: dict[str, int],
    override_tiles_per_block: dict[str, int],
    active_dims: frozenset[str],
    cap_partition: bool,
) -> Tensor:
    """Build a Tensor from traced metadata.

    Args:
        name: Buffer variable name.
        tinfo: Traced tensor info.
        dims: Global dimension info.
        location: Memory space.
        override_num_blocks: Overrides for num_blocks per dim (empty for none).
        override_tiles_per_block: Overrides for tiles_per_block (empty for none).
        active_dims: Dims with active loop variables (empty for none).
        cap_partition: Whether to cap partition dim at SBUF_PMAX.

    Returns:
        Tensor IR object.
    """
    tile_size = _compute_tile_sizes(tinfo, dims, location, cap_partition)
    num_blocks = _compute_num_blocks(tinfo, dims, override_num_blocks, tile_size)
    tiles_per_block = _compute_tiles_per_block(tinfo, dims, override_tiles_per_block)
    default_nb, default_tpb = _compute_loop_defaults(tinfo, active_dims)
    return Tensor(
        name=name,
        axes=tinfo.dims,
        tile_size=tile_size,
        num_blocks=num_blocks,
        tiles_per_block=tiles_per_block,
        location=location,
        default_nb=default_nb,
        default_tpb=default_tpb,
    )


def _compute_tile_sizes(
    tinfo: TensorInfo, dims: dict[str, DimInfo], location: str, cap_partition: bool
) -> dict[str, int]:
    """Compute tile sizes, optionally capping partition dim for on-chip buffers.

    Args:
        tinfo: Traced tensor info.
        dims: Global dimension info.
        location: Memory space.
        cap_partition: Whether to cap partition dim at SBUF_PMAX.
            Set False for nc_transpose outputs where sub-loops
            handle 128-element chunking.

    Returns:
        Maps dim ID to tile size.
    """
    tile_size: dict[str, int] = {}
    for dim_id in tinfo.dims:
        tile_size[dim_id] = dims[dim_id].tile_size
    if cap_partition and location in ("sbuf", "psum") and tinfo.dims:
        par_dim = tinfo.dims[0]
        if tile_size[par_dim] > SBUF_PMAX:
            tile_size[par_dim] = SBUF_PMAX
    return tile_size


def _compute_num_blocks(
    tinfo: TensorInfo, dims: dict[str, DimInfo], override_num_blocks: dict[str, int], tile_size: dict[str, int]
) -> dict[str, int]:
    """Compute num_blocks with overflow from partition capping.

    Args:
        tinfo: Traced tensor info.
        dims: Global dimension info.
        override_num_blocks: Overrides for num_blocks (empty for none).
        tile_size: Already-computed tile sizes (may be capped).

    Returns:
        Maps dim ID to num_blocks.
    """
    num_blocks: dict[str, int] = {}
    for dim_id in tinfo.dims:
        if override_num_blocks and dim_id in override_num_blocks:
            base = override_num_blocks[dim_id]
        else:
            base = dims[dim_id].num_blocks
        global_ts = dims[dim_id].tile_size
        if tile_size[dim_id] < global_ts:
            overflow = global_ts // tile_size[dim_id]
            num_blocks[dim_id] = base * overflow
        else:
            num_blocks[dim_id] = base
    return num_blocks


def _compute_tiles_per_block(
    tinfo: TensorInfo, dims: dict[str, DimInfo], override_tiles_per_block: dict[str, int]
) -> dict[str, int]:
    """Compute tiles_per_block with optional overrides.

    Args:
        tinfo: Traced tensor info.
        dims: Global dimension info.
        override_tiles_per_block: Overrides (empty for none).

    Returns:
        Maps dim ID to tiles_per_block.
    """
    tiles_per_block: dict[str, int] = {}
    for dim_id in tinfo.dims:
        if override_tiles_per_block and dim_id in override_tiles_per_block:
            tiles_per_block[dim_id] = override_tiles_per_block[dim_id]
        else:
            tiles_per_block[dim_id] = dims[dim_id].tiles_per_block
    return tiles_per_block


def _compute_loop_defaults(tinfo: TensorInfo, active_dims: frozenset[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Compute default loop index expressions for active dims.

    Args:
        tinfo: Traced tensor info.
        active_dims: Dims with active loop variables.

    Returns:
        Tuple of (default_nb, default_tpb) dicts.
    """
    default_nb: dict[str, str] = {}
    default_tpb: dict[str, str] = {}
    for dim_id in tinfo.dims:
        if dim_id in active_dims:
            default_nb[dim_id] = f"i_block_{dim_id}"
            default_tpb[dim_id] = f"i_tile_{dim_id}"
    return default_nb, default_tpb


def _tile_tensor(
    name: str,
    tinfo: TensorInfo,
    dims: dict[str, DimInfo],
    location: str,
    active_dims: frozenset[str],
    cap_partition: bool,
) -> Tensor:
    """Build a single-tile Tensor (num_blocks=1 for all dims).

    Used for DMA staging buffers and PSUM accumulators.

    Args:
        name: Buffer variable name.
        tinfo: Traced tensor info.
        dims: Global dimension info.
        location: Memory space.
        active_dims: Dims with active loop variables.
        cap_partition: Whether to cap partition dim at SBUF_PMAX.

    Returns:
        Tile-sized Tensor IR object.
    """
    overrides = {d: 1 for d in tinfo.dims}
    return _build_tensor(name, tinfo, dims, location, overrides, EMPTY_STR_INT, active_dims, cap_partition)


def _input_hbm_slice(input_name: str, tinfo: TensorInfo, dims: dict[str, DimInfo]) -> str:
    """Build HBM slice expression for a DMA load.

    Args:
        input_name: Kernel parameter name.
        tinfo: Tensor info for the input.
        dims: Global dimension info.

    Returns:
        HBM slice expression string.
    """
    parts: list[str] = []
    for dim_id in tinfo.dims:
        dinfo = dims[dim_id]
        ts = dinfo.tile_size
        parts.append(f"i_block_{dim_id}*{ts}:i_block_{dim_id}*{ts}+{ts}")
    return f"{input_name}[{', '.join(parts)}]"


def _consumed_dims(traced_op: TracedOp, tracer: EagerTracer) -> list[str]:
    """Find dimensions consumed by reduction (in operands, not in output).

    Args:
        traced_op: The traced op.
        tracer: The tracer with tensor metadata.

    Returns:
        List of consumed dimension IDs.
    """
    output_dims: set[str] = set()
    for out_name in traced_op.output_names:
        tinfo = tracer.tensors.get(out_name)
        if tinfo is not None:
            output_dims.update(tinfo.dims)

    operand_dims: list[str] = []
    seen: set[str] = set()
    for tensor_name in traced_op.operand_names.values():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is not None:
            for dim_id in tinfo.dims:
                if dim_id not in seen:
                    seen.add(dim_id)
                    operand_dims.append(dim_id)

    return [d for d in operand_dims if d not in output_dims]


def _build_loop_exprs(dim_ids: tuple[str, ...], prefix: str, active_dims: frozenset[str]) -> dict[str, str]:
    """Build loop index expressions for specified dims.

    Args:
        dim_ids: All dimension IDs for a tensor.
        prefix: Loop variable prefix (e.g. ``"i_block"``).
        active_dims: Only these dims get loop variable indices.

    Returns:
        Dim ID to loop index expression.
    """
    result: dict[str, str] = {}
    for dim_id in dim_ids:
        if dim_id in active_dims:
            result[dim_id] = f"{prefix}_{dim_id}"
    return result


def _build_nb_exprs(dim_ids: tuple[str, ...], active_dims: frozenset[str]) -> dict[str, str]:
    """Build block-loop index expressions for specified dims."""
    return _build_loop_exprs(dim_ids, "i_block", active_dims)


def _build_tpb_exprs(dim_ids: tuple[str, ...], active_dims: frozenset[str]) -> dict[str, str]:
    """Build tile-loop index expressions for specified dims."""
    return _build_loop_exprs(dim_ids, "i_tile", active_dims)


def _make_ctx(
    traced_op: TracedOp, tracer: EagerTracer, output_tensors: dict[str, Tensor], operand_tensors: dict[str, Tensor]
) -> RenderContext:
    """Build RenderContext for an op.

    Args:
        traced_op: The traced op.
        tracer: The full tracer state.
        output_tensors: Maps output key to Tensor.
        operand_tensors: Maps operand key to Tensor.

    Returns:
        RenderContext.
    """
    tile_start: dict[str, str] = {}
    for dim_id, dinfo in tracer.dims.items():
        tile_start[dim_id] = f"(i_block_{dim_id} * {dinfo.tiles_per_block} + " f"i_tile_{dim_id}) * {dinfo.tile_size}"

    return RenderContext(
        outputs=output_tensors,
        operands=operand_tensors,
        config_kwargs=traced_op.config_kwargs,
        tile_idx={d: f"i_block_{d} * {tracer.dims[d].tiles_per_block} + i_tile_{d}" for d in tracer.dims},
        tile_start=tile_start,
        dim_global_tile_sizes={d: dinfo.tile_size for d, dinfo in tracer.dims.items()},
    )


def _build_operand_tensors(
    traced_op: TracedOp,
    tracer: EagerTracer,
    active_dims: set[str],
    staging: dict[str, str],
    uncap_staging_slots: frozenset[str],
) -> dict[str, Tensor]:
    """Build operand Tensor objects for render context.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        active_dims: Dimensions with active loop variables.
        staging: Maps operand slot to DMA staging buffer name.
        uncap_staging_slots: Staging slots whose buffers were loaded
            with uncapped partition (e.g. matmul K > 128).

    Returns:
        Maps operand slot name to Tensor.
    """
    frozen_active = frozenset(active_dims)
    operand_tensors: dict[str, Tensor] = {}
    for slot_name, tensor_name in traced_op.operand_names.items():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is None:
            continue
        if slot_name in staging:
            cap = slot_name not in uncap_staging_slots
            t = _tile_tensor(staging[slot_name], tinfo, tracer.dims, "sbuf", EMPTY_STR_SET, cap)
        else:
            sbuf_name = f"sbuf_{tensor_name}"
            nb_overrides = {d: 1 for d in tinfo.dims} if tinfo.is_input else EMPTY_STR_INT
            t = _build_tensor(sbuf_name, tinfo, tracer.dims, "sbuf", nb_overrides, EMPTY_STR_INT, frozen_active, True)
        operand_tensors[slot_name] = t
    return operand_tensors
