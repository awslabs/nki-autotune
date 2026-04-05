"""Eager mode reduction renderers.

Renders loop nests for ops with consumed dimensions: matmul
reductions, tensor_reduce, and activation_reduce.
"""

from nkigym.codegen.eager_emit import (
    _emit_dma_loads,
    _emit_dma_store,
    _emit_isa,
    _emit_matmul_dtype_casts,
    _emit_output_allocs,
    _emit_output_loops,
    _emit_tensor_copy_chunked,
)
from nkigym.codegen.eager_tensors import (
    EMPTY_STR_INT,
    EMPTY_STR_SET,
    _build_nb_exprs,
    _build_operand_tensors,
    _build_tensor,
    _build_tpb_exprs,
    _make_ctx,
    _tile_tensor,
)
from nkigym.codegen.eager_trace import EagerTracer
from nkigym.codegen.eager_types import DimInfo, TensorInfo, TracedOp
from nkigym.codegen.ir import Tensor


def _matmul_uncap_par_slots(traced_op: TracedOp, tracer: EagerTracer, consumed: list[str]) -> frozenset[str]:
    """Find DMA operand slots needing uncapped partition for matmul K sub-loop.

    When nc_matmul's K dimension exceeds MAX_K=128 and an input operand
    has K on its partition axis, the DMA staging buffer must use uncapped
    partition so render()'s string replacement finds the matching slice.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        consumed: Consumed (K) dimension IDs.

    Returns:
        Frozenset of operand slot names needing uncapped partition.
    """
    is_matmul_with_large_k = traced_op.op.NAME == "nc_matmul" and bool(consumed)
    if is_matmul_with_large_k:
        k_dim = consumed[0]
        k_tile = tracer.dims[k_dim].tile_size
        k_max = traced_op.op.MAX_TILE_SIZES.get("K", 128)
        is_matmul_with_large_k = k_tile > k_max

    slots: set[str] = set()
    if is_matmul_with_large_k:
        k_dim = consumed[0]
        for slot_name, tensor_name in traced_op.operand_names.items():
            tinfo = tracer.tensors.get(tensor_name)
            if tinfo is not None and tinfo.is_input and tinfo.dims and tinfo.dims[0] == k_dim:
                slots.add(slot_name)
    return frozenset(slots)


def _render_reduction(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], consumed: list[str], is_final: bool
) -> None:
    """Render an op with reduction (e.g. matmul accumulating over K).

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        consumed: Consumed dimension IDs.
        is_final: Whether this is the final op.
    """
    op = traced_op.op
    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]
    dims = tracer.dims
    active = set(out_tinfo.dims) | set(consumed)

    _emit_output_allocs(traced_op, tracer, lines)
    indent = _emit_output_loops(out_tinfo, dims, lines)

    psum_name = f"psum_{out_name}"
    psum_tensor = _tile_tensor(psum_name, out_tinfo, dims, "psum", EMPTY_STR_SET, True)
    lines.append(f"{indent}{psum_name} = nl.ndarray(" f"{psum_tensor.shape()}, dtype=nl.float32, " f"buffer=nl.psum)")

    red_indent = _emit_reduction_loops(consumed, dims, lines, indent)

    uncap_slots = _matmul_uncap_par_slots(traced_op, tracer, consumed)
    staging = _emit_dma_loads(traced_op, tracer, lines, red_indent, uncap_slots)
    staging = _emit_matmul_dtype_casts(traced_op, tracer, staging, lines, red_indent, uncap_slots)
    operand_tensors = _build_operand_tensors(traced_op, tracer, active, staging, uncap_slots)

    output_tensors = {next(iter(op.OUTPUT_AXES)): psum_tensor}
    ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
    _emit_isa(op, ctx, lines, red_indent)

    sbuf_out = _build_tensor(
        f"sbuf_{out_name}", out_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, frozenset(out_tinfo.dims), True
    )
    par_dim = out_tinfo.dims[0]
    _emit_tensor_copy_chunked(sbuf_out, psum_tensor, dims[par_dim].tile_size, par_dim, lines, indent)

    if is_final:
        _emit_dma_store(out_tinfo, dims, lines, indent)


def _emit_reduction_loops(consumed: list[str], dims: dict[str, DimInfo], lines: list[str], indent: str) -> str:
    """Emit reduction dimension loops.

    Args:
        consumed: Consumed dimension IDs.
        dims: Dimension metadata.
        lines: Output lines list.
        indent: Starting indentation.

    Returns:
        Indentation after all reduction loops.
    """
    red_indent = indent
    for dim_id in consumed:
        dinfo = dims[dim_id]
        lines.append(f"{red_indent}for i_block_{dim_id} in " f"nl.affine_range({dinfo.num_blocks}):")
        red_indent += "    "
        lines.append(f"{red_indent}for i_tile_{dim_id} in " f"nl.affine_range({dinfo.tiles_per_block}):")
        red_indent += "    "
    return red_indent


def _render_tensor_reduce(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], consumed: list[str], is_final: bool
) -> None:
    """Render tensor_reduce with partial+final reduction pattern.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        consumed: Consumed dimension IDs.
        is_final: Whether this is the final op.
    """
    dims = tracer.dims
    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]
    consumed_dim = consumed[0]
    consumed_dinfo = dims[consumed_dim]
    negate = traced_op.config_kwargs.get("negate", False)
    reduce_op_name = traced_op.config_kwargs["op"]
    nki_reduce = {"max": "maximum", "add": "add"}[reduce_op_name]

    _emit_output_allocs(traced_op, tracer, lines)
    indent = _emit_output_loops(out_tinfo, dims, lines)

    _render_tr_partial(traced_op, tracer, dims, out_tinfo, consumed_dim, consumed_dinfo, nki_reduce, lines, indent)

    _render_tr_final(tracer, out_name, out_tinfo, consumed_dinfo, nki_reduce, negate, lines, indent)


def _render_tr_partial_psum(
    dims: dict[str, DimInfo],
    out_tinfo: TensorInfo,
    consumed_dim: str,
    consumed_dinfo: DimInfo,
    lines: list[str],
    indent: str,
) -> str:
    """Emit PSUM allocation and reduction loops for partial phase.

    Args:
        dims: Dimension metadata.
        out_tinfo: Output tensor info.
        consumed_dim: Consumed dimension ID.
        consumed_dinfo: Consumed dimension info.
        lines: Output lines list.
        indent: Current indentation.

    Returns:
        Indentation inside the reduction loops.
    """
    par_dim = out_tinfo.dims[0]
    par_ts = dims[par_dim].tile_size
    psum_name = "psum_partial_max"
    lines.append(
        f"{indent}{psum_name} = nl.ndarray("
        f"({par_ts}, {consumed_dinfo.num_blocks}), "
        f"dtype=nl.float32, buffer=nl.psum)"
    )
    red_indent = indent
    lines.append(f"{red_indent}for i_block_{consumed_dim} in " f"nl.affine_range({consumed_dinfo.num_blocks}):")
    red_indent += "    "
    lines.append(f"{red_indent}for i_tile_{consumed_dim} in " f"nl.affine_range({consumed_dinfo.tiles_per_block}):")
    red_indent += "    "
    return red_indent


def _render_tr_partial(
    traced_op: TracedOp,
    tracer: EagerTracer,
    dims: dict[str, DimInfo],
    out_tinfo: TensorInfo,
    consumed_dim: str,
    consumed_dinfo: DimInfo,
    nki_reduce: str,
    lines: list[str],
    indent: str,
) -> None:
    """Render phase 1: partial per-block reduces into PSUM.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        dims: Dimension metadata.
        out_tinfo: Output tensor info.
        consumed_dim: Consumed dimension ID.
        consumed_dinfo: Consumed dimension info.
        nki_reduce: NKI reduce function name.
        lines: Output lines list.
        indent: Current indentation.
    """
    red_indent = _render_tr_partial_psum(dims, out_tinfo, consumed_dim, consumed_dinfo, lines, indent)
    par_dim = out_tinfo.dims[0]
    par_ts = dims[par_dim].tile_size
    data_tensor_name = traced_op.operand_names["data"]
    data_tinfo = tracer.tensors[data_tensor_name]
    data_tensor = _build_tensor(
        f"sbuf_{data_tensor_name}", data_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, EMPTY_STR_SET, True
    )
    all_active = frozenset(out_tinfo.dims) | frozenset([consumed_dim])
    data_slice = data_tensor.indexed_slice(
        _build_nb_exprs(data_tinfo.dims, all_active), _build_tpb_exprs(data_tinfo.dims, all_active)
    )
    psum_dst = f"psum_partial_max[0:{par_ts}, " f"i_block_{consumed_dim}:i_block_{consumed_dim}+1]"
    lines.append(
        f"{red_indent}nisa.tensor_reduce(" f"dst={psum_dst}, " f"data={data_slice}, " f"op=nl.{nki_reduce}, axis=1)"
    )


def _render_tr_final(
    tracer: EagerTracer,
    out_name: str,
    out_tinfo: TensorInfo,
    consumed_dinfo: DimInfo,
    nki_reduce: str,
    negate: bool,
    lines: list[str],
    indent: str,
) -> None:
    """Render phase 2: final reduce across partial results.

    Args:
        tracer: Tracer state.
        out_name: Output tensor name.
        out_tinfo: Output tensor info.
        consumed_dinfo: Consumed dimension info.
        nki_reduce: NKI reduce function name.
        negate: Whether to negate the result.
        lines: Output lines list.
        indent: Current indentation.
    """
    dims = tracer.dims
    sbuf_out = _build_tensor(
        f"sbuf_{out_name}", out_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, EMPTY_STR_SET, True
    )
    out_active = frozenset(out_tinfo.dims)
    final_dst = sbuf_out.indexed_slice(
        _build_nb_exprs(out_tinfo.dims, out_active), _build_tpb_exprs(out_tinfo.dims, out_active)
    )
    par_dim = out_tinfo.dims[0]
    par_ts = dims[par_dim].tile_size
    psum_src = f"psum_partial_max[0:{par_ts}, 0:{consumed_dinfo.num_blocks}]"
    negate_str = ", negate=True" if negate else ""
    lines.append(
        f"{indent}nisa.tensor_reduce("
        f"dst={final_dst}, "
        f"data={psum_src}, "
        f"op=nl.{nki_reduce}, axis=1{negate_str})"
    )


def _render_activation_reduce(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], consumed: list[str], is_final: bool
) -> None:
    """Render activation_reduce with per-block PSUM slots + cross-block combine.

    Same unified reduction pattern as tensor_reduce: each consumed
    block writes to its own PSUM slot via activation_reduce (reset is
    harmless — each slot written once), then tensor_reduce(add) combines
    all slots after the loop.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        consumed: Consumed dimension IDs (may be empty for multi-output).
        is_final: Whether this is the final op.
    """
    dims = tracer.dims
    out_name = traced_op.output_names[0]
    red_name = traced_op.output_names[1]
    out_tinfo = tracer.tensors[out_name]
    red_tinfo = tracer.tensors[red_name]

    red_dim_set = set(red_tinfo.dims)
    ar_consumed = [d for d in out_tinfo.dims if d not in red_dim_set]
    if not ar_consumed:
        raise ValueError(
            f"activation_reduce '{out_name}' has no consumed dimensions "
            f"(output dims {out_tinfo.dims}, reduce dims {red_tinfo.dims})"
        )
    consumed_dim = ar_consumed[0]
    consumed_dinfo = dims[consumed_dim]

    _emit_output_allocs(traced_op, tracer, lines)
    indent = _emit_ar_outer_loops(red_tinfo, dims, lines)

    par_dim = red_tinfo.dims[0]
    par_ts = dims[par_dim].tile_size
    psum_name = "psum_partial_sum"
    lines.append(
        f"{indent}{psum_name} = nl.ndarray("
        f"({par_ts}, {consumed_dinfo.num_blocks}), "
        f"dtype=nl.float32, buffer=nl.psum)"
    )

    red_indent = _emit_ar_body(
        traced_op, tracer, out_tinfo, ar_consumed, consumed_dim, consumed_dinfo, par_ts, psum_name, lines, indent
    )
    _emit_ar_combine(tracer, red_name, red_tinfo, consumed_dinfo, par_ts, psum_name, lines, indent)


def _emit_ar_body(
    traced_op: TracedOp,
    tracer: EagerTracer,
    out_tinfo: TensorInfo,
    ar_consumed: list[str],
    consumed_dim: str,
    consumed_dinfo: DimInfo,
    par_ts: int,
    psum_name: str,
    lines: list[str],
    indent: str,
) -> str:
    """Emit consumed-dim loop with per-block activation_reduce ISA call."""
    out_name = traced_op.output_names[0]
    dims = tracer.dims
    red_indent = indent
    lines.append(f"{red_indent}for i_block_{consumed_dim} in " f"nl.affine_range({consumed_dinfo.num_blocks}):")
    red_indent += "    "
    lines.append(f"{red_indent}for i_tile_{consumed_dim} in " f"nl.affine_range({consumed_dinfo.tiles_per_block}):")
    red_indent += "    "

    active = set(out_tinfo.dims) | set(ar_consumed)
    operand_tensors = _build_operand_tensors(traced_op, tracer, active, {}, EMPTY_STR_SET)

    psum_slot_slice = f"{psum_name}[0:{par_ts}, " f"i_block_{consumed_dim}:i_block_{consumed_dim}+1]"
    psum_slot_tensor = Tensor(
        name=psum_slot_slice, axes=(), tile_size={}, num_blocks={}, tiles_per_block={}, location="psum"
    )
    output_tensors: dict[str, Tensor] = {
        "output": _build_tensor(
            f"sbuf_{out_name}", out_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, frozenset(active), True
        ),
        "reduce_res": psum_slot_tensor,
    }
    ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
    _emit_isa(traced_op.op, ctx, lines, red_indent)
    return red_indent


def _emit_ar_combine(
    tracer: EagerTracer,
    red_name: str,
    red_tinfo: TensorInfo,
    consumed_dinfo: DimInfo,
    par_ts: int,
    psum_name: str,
    lines: list[str],
    indent: str,
) -> None:
    """Emit cross-block tensor_reduce(add) to combine partial sums.

    Args:
        tracer: Tracer state.
        red_name: Reduction output tensor name.
        red_tinfo: Reduction output tensor info.
        consumed_dinfo: Consumed dimension info.
        par_ts: Partition tile size.
        psum_name: PSUM partial buffer name.
        lines: Output lines list.
        indent: Indentation for the combine call (same level as the loop).
    """
    dims = tracer.dims
    sbuf_red = _build_tensor(
        f"sbuf_{red_name}", red_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, frozenset(red_tinfo.dims), True
    )
    out_active = frozenset(red_tinfo.dims)
    final_dst = sbuf_red.indexed_slice(
        _build_nb_exprs(red_tinfo.dims, out_active), _build_tpb_exprs(red_tinfo.dims, out_active)
    )
    psum_src = f"{psum_name}[0:{par_ts}, 0:{consumed_dinfo.num_blocks}]"
    lines.append(f"{indent}nisa.tensor_reduce(" f"dst={final_dst}, " f"data={psum_src}, " f"op=nl.add, axis=1)")


def _emit_ar_outer_loops(red_tinfo: TensorInfo, dims: dict[str, DimInfo], lines: list[str]) -> str:
    """Emit outer loops for activation_reduce (reduction output dims).

    Args:
        red_tinfo: Reduction output tensor info.
        dims: Dimension metadata.
        lines: Output lines list.

    Returns:
        Indent string after loops.
    """
    indent = ""
    for dim_id in red_tinfo.dims:
        dinfo = dims[dim_id]
        lines.append(f"{indent}for i_block_{dim_id} in " f"nl.affine_range({dinfo.num_blocks}):")
        indent += "    "
        lines.append(f"{indent}for i_tile_{dim_id} in " f"nl.affine_range({dinfo.tiles_per_block}):")
        indent += "    "
    return indent
