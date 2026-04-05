"""Eager mode emit helpers.

Functions that emit NKI code lines for DMA loads, stores,
buffer allocations, ISA calls, and tensor copies.
"""

from nkigym.codegen.eager_tensors import EMPTY_STR_INT, EMPTY_STR_SET, _build_tensor, _input_hbm_slice, _tile_tensor
from nkigym.codegen.eager_trace import EagerTracer
from nkigym.codegen.eager_types import SBUF_PMAX, DimInfo, TensorInfo, TracedOp
from nkigym.codegen.ir import RenderContext, Tensor
from nkigym.ops.base import NKIOp


def _emit_output_allocs(traced_op: TracedOp, tracer: EagerTracer, lines: list[str]) -> None:
    """Emit output buffer allocations.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
    """
    for out_name in traced_op.output_names:
        tinfo = tracer.tensors[out_name]
        sbuf_name = f"sbuf_{out_name}"
        tensor = _build_tensor(sbuf_name, tinfo, tracer.dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, EMPTY_STR_SET, True)
        dtype = tracer.tensor_dtypes[out_name]
        lines.append(f"{sbuf_name} = nl.ndarray(" f"{tensor.shape()}, dtype={dtype}, " f"buffer=nl.sbuf)")


def _emit_output_loops(tinfo: TensorInfo, dims: dict[str, DimInfo], lines: list[str]) -> str:
    """Emit parallel double loops for output dims.

    Args:
        tinfo: Output tensor info.
        dims: Dimension metadata.
        lines: Output lines list.

    Returns:
        Indent string after all loops.
    """
    indent = ""
    for dim_id in tinfo.dims:
        dinfo = dims[dim_id]
        lines.append(f"{indent}for i_block_{dim_id} in " f"nl.affine_range({dinfo.num_blocks}):")
        indent += "    "
        lines.append(f"{indent}for i_tile_{dim_id} in " f"nl.affine_range({dinfo.tiles_per_block}):")
        indent += "    "
    return indent


def _emit_dma_loads(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], indent: str, uncap_par_slots: frozenset[str]
) -> dict[str, str]:
    """Emit DMA load statements for kernel inputs.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        indent: Current indentation.
        uncap_par_slots: Operand slots that need uncapped partition
            (e.g. matmul inputs where K dim > 128 is on partition).

    Returns:
        Maps operand slot to staging buffer name.
    """
    staging: dict[str, str] = {}
    for slot_name, tensor_name in traced_op.operand_names.items():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is None or not tinfo.is_input:
            continue
        sbuf_name = f"sbuf_{tensor_name}"
        cap = slot_name not in uncap_par_slots
        tile_t = _tile_tensor(sbuf_name, tinfo, tracer.dims, "sbuf", EMPTY_STR_SET, cap)
        dtype = f"{tensor_name}.dtype"
        lines.append(f"{indent}{sbuf_name} = nl.ndarray(" f"{tile_t.shape()}, dtype={dtype}, " f"buffer=nl.sbuf)")
        global_par_ts = tracer.dims[tinfo.dims[0]].tile_size
        if global_par_ts > SBUF_PMAX and cap:
            _emit_dma_load_chunked(tensor_name, tinfo, tracer.dims, tile_t, lines, indent)
        elif global_par_ts > SBUF_PMAX and not cap:
            _emit_dma_load_uncapped(tensor_name, tinfo, tracer.dims, tile_t, lines, indent)
        else:
            hbm_slice = _input_hbm_slice(tensor_name, tinfo, tracer.dims)
            tile_slice = tile_t.default_indexed_slice()
            lines.append(f"{indent}nisa.dma_copy(dst={tile_slice}, " f"src={hbm_slice})")
        staging[slot_name] = sbuf_name
    return staging


def _emit_dma_load_chunked(
    input_name: str, tinfo: TensorInfo, dims: dict[str, DimInfo], tile_t: Tensor, lines: list[str], indent: str
) -> None:
    """Emit sub-loop DMA load for partition dim > 128.

    Args:
        input_name: Kernel parameter name.
        tinfo: Tensor info for the input.
        dims: Dimension metadata.
        tile_t: Tile-sized Tensor for the staging buffer.
        lines: Output lines list.
        indent: Current indentation.
    """
    par_dim = tinfo.dims[0]
    global_par_ts = dims[par_dim].tile_size
    n_chunks = global_par_ts // SBUF_PMAX
    free_dims = tinfo.dims[1:]
    lines.append(f"{indent}for i_dma_par in nl.affine_range({n_chunks}):")
    sub_indent = indent + "    "
    nb_exprs = {par_dim: "i_dma_par"}
    dst_slice = tile_t.indexed_slice(nb_exprs, {})
    src_parts = [
        f"i_block_{par_dim}*{global_par_ts}+i_dma_par*{SBUF_PMAX}"
        f":i_block_{par_dim}*{global_par_ts}+i_dma_par*{SBUF_PMAX}+{SBUF_PMAX}"
    ]
    for fdim in free_dims:
        fts = dims[fdim].tile_size
        src_parts.append(f"i_block_{fdim}*{fts}:i_block_{fdim}*{fts}+{fts}")
    src_expr = f"{input_name}[{', '.join(src_parts)}]"
    lines.append(f"{sub_indent}nisa.dma_copy(dst={dst_slice}, src={src_expr})")


def _emit_dma_load_uncapped(
    input_name: str, tinfo: TensorInfo, dims: dict[str, DimInfo], tile_t: Tensor, lines: list[str], indent: str
) -> None:
    """Emit sub-loop DMA load for uncapped partition buffers.

    Uses range-based string replacement on the full partition axis
    instead of nb-based indexing, since the tile tensor has no
    num_blocks overflow for the partition dim.

    Args:
        input_name: Kernel parameter name.
        tinfo: Tensor info for the input.
        dims: Dimension metadata.
        tile_t: Tile-sized Tensor for the staging buffer.
        lines: Output lines list.
        indent: Current indentation.
    """
    par_dim = tinfo.dims[0]
    global_par_ts = dims[par_dim].tile_size
    n_chunks = global_par_ts // SBUF_PMAX
    free_dims = tinfo.dims[1:]
    lines.append(f"{indent}for i_dma_par in nl.affine_range({n_chunks}):")
    sub_indent = indent + "    "
    par_range = f"0:{global_par_ts}"
    chunk_range = f"i_dma_par*{SBUF_PMAX}:(i_dma_par+1)*{SBUF_PMAX}"
    dst_slice = tile_t.default_indexed_slice().replace(par_range, chunk_range, 1)
    src_parts = [
        f"i_block_{par_dim}*{global_par_ts}+i_dma_par*{SBUF_PMAX}"
        f":i_block_{par_dim}*{global_par_ts}+i_dma_par*{SBUF_PMAX}+{SBUF_PMAX}"
    ]
    for fdim in free_dims:
        fts = dims[fdim].tile_size
        src_parts.append(f"i_block_{fdim}*{fts}:i_block_{fdim}*{fts}+{fts}")
    src_expr = f"{input_name}[{', '.join(src_parts)}]"
    lines.append(f"{sub_indent}nisa.dma_copy(dst={dst_slice}, src={src_expr})")


def _emit_tensor_copy_chunked(
    dst_t: Tensor, src_t: Tensor, global_par_ts: int, par_dim: str, lines: list[str], indent: str
) -> None:
    """Emit tensor_copy, chunking via nb if global par > 128.

    Args:
        dst_t: Destination Tensor.
        src_t: Source Tensor.
        global_par_ts: Global partition tile size (may exceed 128).
        par_dim: Partition dimension ID.
        lines: Output lines list.
        indent: Current indentation.
    """
    if global_par_ts <= SBUF_PMAX:
        lines.append(
            f"{indent}nisa.tensor_copy("
            f"dst={dst_t.default_indexed_slice()}, "
            f"src={src_t.default_indexed_slice()})"
        )
    else:
        _emit_tensor_copy_chunked_loop(dst_t, src_t, global_par_ts, par_dim, lines, indent)


def _emit_tensor_copy_chunked_loop(
    dst_t: Tensor, src_t: Tensor, global_par_ts: int, par_dim: str, lines: list[str], indent: str
) -> None:
    """Emit the sub-loop body for chunked tensor_copy.

    Args:
        dst_t: Destination Tensor.
        src_t: Source Tensor.
        global_par_ts: Global partition tile size.
        par_dim: Partition dimension ID.
        lines: Output lines list.
        indent: Current indentation.
    """
    n_chunks = global_par_ts // SBUF_PMAX
    lines.append(f"{indent}for i_copy_par in nl.affine_range({n_chunks}):")
    sub = indent + "    "
    dst_nb = dict(dst_t.default_nb)
    dst_nb[par_dim] = "i_copy_par"
    src_nb = dict(src_t.default_nb)
    src_nb[par_dim] = "i_copy_par"
    d_slice = dst_t.indexed_slice(dst_nb, dst_t.default_tpb)
    s_slice = src_t.indexed_slice(src_nb, src_t.default_tpb)
    lines.append(f"{sub}nisa.tensor_copy(dst={d_slice}, src={s_slice})")


def _emit_matmul_dtype_casts(
    traced_op: TracedOp,
    tracer: EagerTracer,
    staging: dict[str, str],
    lines: list[str],
    indent: str,
    uncap_par_slots: frozenset[str],
) -> dict[str, str]:
    """Cast DMA-loaded matmul operands to match non-input operand dtypes.

    nc_matmul requires both inputs to have the same dtype. When one
    operand is an intermediate float32 tensor and the other is a
    DMA-loaded float16 input, emit a tensor_copy cast to float32.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        staging: Maps operand slot to DMA staging buffer name.
        lines: Output lines list.
        indent: Current indentation.
        uncap_par_slots: Slots with uncapped partition buffers.

    Returns:
        Updated staging dict with cast buffer names.
    """
    if traced_op.op.NAME != "nc_matmul":
        result = staging
    else:
        target_dtype = _find_non_input_dtype(traced_op, tracer)
        if not target_dtype:
            result = staging
        else:
            result = _cast_staging_buffers(traced_op, tracer, staging, target_dtype, lines, indent, uncap_par_slots)
    return result


def _find_non_input_dtype(traced_op: TracedOp, tracer: EagerTracer) -> str:
    """Find dtype of non-input operands for matmul casting.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.

    Returns:
        Target dtype string, or empty string if no non-input operands.
    """
    non_input_dtypes: set[str] = set()
    for tensor_name in traced_op.operand_names.values():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is not None and not tinfo.is_input:
            dtype = tracer.tensor_dtypes.get(tensor_name, "")
            if dtype:
                non_input_dtypes.add(dtype)
    result = ""
    if non_input_dtypes:
        result = next(iter(non_input_dtypes))
    return result


def _cast_staging_buffers(
    traced_op: TracedOp,
    tracer: EagerTracer,
    staging: dict[str, str],
    target_dtype: str,
    lines: list[str],
    indent: str,
    uncap_par_slots: frozenset[str],
) -> dict[str, str]:
    """Emit cast copies for all staging buffers to target dtype.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        staging: Maps operand slot to DMA staging buffer name.
        target_dtype: Target dtype expression.
        lines: Output lines list.
        indent: Current indentation.
        uncap_par_slots: Slots with uncapped partition buffers.

    Returns:
        Updated staging dict with cast buffer names.
    """
    updated = dict(staging)
    for slot, sbuf_name in staging.items():
        tensor_name = traced_op.operand_names[slot]
        tinfo = tracer.tensors[tensor_name]
        cap = slot not in uncap_par_slots
        src_t = _tile_tensor(sbuf_name, tinfo, tracer.dims, "sbuf", EMPTY_STR_SET, cap)
        cast_name = f"{sbuf_name}_f32"
        cast_t = _tile_tensor(cast_name, tinfo, tracer.dims, "sbuf", EMPTY_STR_SET, cap)
        lines.append(
            f"{indent}{cast_name} = nl.ndarray(" f"{cast_t.shape()}, dtype={target_dtype}, " f"buffer=nl.sbuf)"
        )
        par_dim = tinfo.dims[0]
        global_par_ts = tracer.dims[par_dim].tile_size
        if cap:
            _emit_tensor_copy_chunked(cast_t, src_t, global_par_ts, par_dim, lines, indent)
        else:
            _emit_tensor_copy_uncapped(cast_t, src_t, global_par_ts, lines, indent)
        updated[slot] = cast_name
    return updated


def _emit_tensor_copy_uncapped(dst_t: Tensor, src_t: Tensor, global_par_ts: int, lines: list[str], indent: str) -> None:
    """Emit tensor_copy for uncapped partition buffers.

    Iterates 128-element partition chunks using range indexing
    on the full partition axis, rather than nb indexing.

    Args:
        dst_t: Destination Tensor (uncapped partition).
        src_t: Source Tensor (uncapped partition).
        global_par_ts: Full partition tile size (e.g. 256).
        lines: Output lines list.
        indent: Current indentation.
    """
    if global_par_ts <= SBUF_PMAX:
        lines.append(
            f"{indent}nisa.tensor_copy("
            f"dst={dst_t.default_indexed_slice()}, "
            f"src={src_t.default_indexed_slice()})"
        )
    else:
        n_chunks = global_par_ts // SBUF_PMAX
        lines.append(f"{indent}for i_copy_par in nl.affine_range({n_chunks}):")
        sub = indent + "    "
        par_range = f"0:{global_par_ts}"
        chunk_range = f"i_copy_par*{SBUF_PMAX}:(i_copy_par+1)*{SBUF_PMAX}"
        dst_slice = dst_t.default_indexed_slice().replace(par_range, chunk_range, 1)
        src_slice = src_t.default_indexed_slice().replace(par_range, chunk_range, 1)
        lines.append(f"{sub}nisa.tensor_copy(dst={dst_slice}, src={src_slice})")


def _emit_isa(op: NKIOp, ctx: RenderContext, lines: list[str], indent: str) -> None:
    """Emit ISA call at given indent level.

    Args:
        op: NKIOp instance.
        ctx: Render context.
        lines: Output lines list.
        indent: Current indentation.
    """
    isa_lines = op.render(ctx)
    for isa_line in isa_lines:
        lines.append(f"{indent}{isa_line}")


def _emit_dma_store(out_tinfo: TensorInfo, dims: dict[str, DimInfo], lines: list[str], indent: str) -> None:
    """Emit DMA store for the final output.

    Handles chunked stores when partition dim > SBUF_PMAX.

    Args:
        out_tinfo: Output tensor info.
        dims: Dimension metadata.
        lines: Output lines list.
        indent: Current indentation.
    """
    sbuf_name = f"sbuf_{out_tinfo.name}"
    tile_t = _tile_tensor(sbuf_name, out_tinfo, dims, "sbuf", EMPTY_STR_SET, True)
    par_dim = out_tinfo.dims[0]
    global_par_ts = dims[par_dim].tile_size
    if global_par_ts <= SBUF_PMAX:
        hbm_slice = _input_hbm_slice(out_tinfo.name, out_tinfo, dims)
        lines.append(f"{indent}nisa.dma_copy(dst={hbm_slice}, " f"src={tile_t.default_indexed_slice()})")
    else:
        _emit_dma_store_chunked(out_tinfo, dims, tile_t, lines, indent)


def _emit_dma_store_chunked(
    out_tinfo: TensorInfo, dims: dict[str, DimInfo], tile_t: Tensor, lines: list[str], indent: str
) -> None:
    """Emit sub-loop DMA store for partition dim > 128.

    Args:
        out_tinfo: Output tensor info.
        dims: Dimension metadata.
        tile_t: Tile-sized Tensor for the source buffer.
        lines: Output lines list.
        indent: Current indentation.
    """
    par_dim = out_tinfo.dims[0]
    global_par_ts = dims[par_dim].tile_size
    n_chunks = global_par_ts // SBUF_PMAX
    free_dims = out_tinfo.dims[1:]
    lines.append(f"{indent}for i_store_par in nl.affine_range({n_chunks}):")
    sub_indent = indent + "    "
    src_nb = {par_dim: "i_store_par"}
    src_slice = tile_t.indexed_slice(src_nb, {})
    dst_parts = [
        f"i_block_{par_dim}*{global_par_ts}+i_store_par*{SBUF_PMAX}"
        f":i_block_{par_dim}*{global_par_ts}+i_store_par*{SBUF_PMAX}+{SBUF_PMAX}"
    ]
    for fdim in free_dims:
        fts = dims[fdim].tile_size
        dst_parts.append(f"i_block_{fdim}*{fts}:i_block_{fdim}*{fts}+{fts}")
    dst_expr = f"{out_tinfo.name}[{', '.join(dst_parts)}]"
    lines.append(f"{sub_indent}nisa.dma_copy(dst={dst_expr}, src={src_slice})")
