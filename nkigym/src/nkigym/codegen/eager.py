"""Eager mode kernel generator.

Chains NKIOp.render() calls, one per op in math function order.
Each op gets its own independent loop nest. The result is a complete
NKI kernel string that can be simulated or compiled.

Design doc reference: nkigym_ir_guide.md sections 2-3.
"""

from nkigym.codegen.eager_emit import (
    _emit_dma_loads,
    _emit_dma_store,
    _emit_isa,
    _emit_output_allocs,
    _emit_output_loops,
    _emit_tensor_copy_chunked,
    _emit_tensor_copy_uncapped,
)
from nkigym.codegen.eager_render_reduce import _render_activation_reduce, _render_reduction, _render_tensor_reduce
from nkigym.codegen.eager_tensors import (
    EMPTY_STR_INT,
    EMPTY_STR_SET,
    _build_operand_tensors,
    _build_tensor,
    _consumed_dims,
    _make_ctx,
    _tile_tensor,
)
from nkigym.codegen.eager_trace import EagerTracer
from nkigym.codegen.eager_types import SBUF_PMAX, TracedOp
from nkigym.codegen.ir import OperandInfo, RenderContext, Tensor
from nkigym.ops.base import NKIOp

"""Ops whose output lands in PSUM (TensorEngine ops)."""
_PSUM_OPS: frozenset[str] = frozenset({"nc_matmul", "nc_transpose"})

__all__ = ["EagerTracer", "generate_eager_kernel"]


def _primary_output_key(op: NKIOp) -> str:
    """Get the first output key from an op's OUTPUT_AXES.

    Args:
        op: The NKIOp instance.

    Returns:
        First output key string.
    """
    return next(iter(op.OUTPUT_AXES))


def generate_eager_kernel(tracer: EagerTracer, func_name: str, scale_param: str) -> str:
    """Generate a complete NKI kernel from traced ops.

    Chains NKIOp.render() calls, one per op, with greedy
    DMA placement at the innermost loop level.

    Args:
        tracer: Populated tracer with ops and tensor metadata.
        func_name: Name for the @nki.jit kernel function.
        scale_param: Extra scalar parameter name (empty string for none).

    Returns:
        Complete NKI kernel source code string.
    """
    tracer.compute_dim_info()
    tracer.compute_dtypes()
    lines: list[str] = _render_header(tracer, func_name, scale_param)

    final_op = tracer.ops[-1]
    final_output_name = final_op.output_names[0]

    _render_hbm_output(tracer, final_output_name, lines)

    for traced_op in tracer.ops:
        op_lines = _render_op(traced_op, tracer, final_output_name)
        for line in op_lines:
            lines.append(f"    {line}")
        lines.append("")

    lines.append(f"    return {final_output_name}")
    return "\n".join(lines)


def _render_header(tracer: EagerTracer, func_name: str, scale_param: str) -> list[str]:
    """Render kernel imports, signature, and docstring.

    Args:
        tracer: Tracer state.
        func_name: Kernel function name.
        scale_param: Extra scalar parameter (empty string for none).

    Returns:
        List of header lines.
    """
    lines: list[str] = []
    lines.append("import numpy as np")
    lines.append("import nki")
    lines.append("import nki.language as nl")
    lines.append("import nki.isa as nisa")
    lines.append("from nki.backends.mlir_tracer.tensor import Tensor")
    lines.append("")
    lines.append("")

    annotated = [f"{inp}: Tensor" for inp in tracer.inputs]
    if scale_param:
        annotated.append(scale_param)
    lines.append("@nki.jit")
    lines.append(f"def {func_name}({', '.join(annotated)}):")

    for inp in tracer.inputs:
        shape = tracer.input_shapes[inp]
        shape_str = ", ".join(str(s) for s in shape)
        lines.append(f"    assert {inp}.shape == ({shape_str})")

    dim_parts = []
    for dim_id in sorted(tracer.dims.keys(), key=lambda d: int(d[1:])):
        dinfo = tracer.dims[dim_id]
        dim_parts.append(f"{dim_id}: {dinfo.tile_size}x{dinfo.num_blocks}")
    lines.append(f'    """ {" ".join(dim_parts)}' f'  (tile_size x num_blocks) """')
    return lines


def _render_hbm_output(tracer: EagerTracer, final_output_name: str, lines: list[str]) -> None:
    """Render HBM output allocation.

    Args:
        tracer: Tracer state.
        final_output_name: Name of the final output tensor.
        lines: Output lines list.
    """
    final_tinfo = tracer.tensors[final_output_name]
    hbm_shape_parts = []
    for dim_id in final_tinfo.dims:
        dinfo = tracer.dims[dim_id]
        hbm_shape_parts.append(str(dinfo.total_size))
    hbm_shape = ", ".join(hbm_shape_parts)
    lines.append(
        f"    {final_output_name} = nl.ndarray(({hbm_shape}), " f"dtype={tracer.inputs[0]}.dtype, buffer=nl.shared_hbm)"
    )
    lines.append("")


def _render_op(traced_op: TracedOp, tracer: EagerTracer, final_output_name: str) -> list[str]:
    """Render a single op's complete loop nest.

    Args:
        traced_op: The traced op to render.
        tracer: The full tracer state.
        final_output_name: Name of the kernel's final output.

    Returns:
        List of source lines (without base indent).
    """
    lines: list[str] = []
    consumed = _consumed_dims(traced_op, tracer)
    is_final = traced_op.output_names[0] == final_output_name

    _render_op_header(traced_op, tracer, consumed, lines)
    _dispatch_render(traced_op, tracer, lines, consumed, is_final)
    return lines


def _render_op_header(traced_op: TracedOp, tracer: EagerTracer, consumed: list[str], lines: list[str]) -> None:
    """Render the op comment header.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        consumed: Consumed dimension IDs.
        lines: Output lines list.
    """
    op = traced_op.op
    primary_out_name = traced_op.output_names[0]
    primary_tinfo = tracer.tensors[primary_out_name]
    operand_desc_parts: list[str] = []
    for tname in traced_op.operand_names.values():
        tinfo = tracer.tensors.get(tname)
        if tinfo is not None:
            dim_str = ", ".join(tinfo.dims)
            operand_desc_parts.append(f"{tname}({dim_str})")
    operand_desc = ", ".join(operand_desc_parts)
    out_dim_str = ", ".join(primary_tinfo.dims)
    consumed_desc = ""
    if consumed:
        consumed_desc = f", accumulate over {', '.join(consumed)}"
    lines.append(
        f'""" Op {traced_op.op_idx}: nisa.{op.NAME} -- '
        f"{operand_desc} -> "
        f'{primary_out_name}({out_dim_str}){consumed_desc} """'
    )


def _dispatch_render(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], consumed: list[str], is_final: bool
) -> None:
    """Dispatch to the appropriate renderer based on op type.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        consumed: Consumed dimension IDs.
        is_final: Whether this is the final op.
    """
    op = traced_op.op
    if op.NAME == "nc_matmul":
        _render_full_nest_op(traced_op, tracer, lines, consumed, is_final)
    elif op.NAME == "tensor_reduce" and consumed:
        _render_tensor_reduce(traced_op, tracer, lines, consumed, is_final)
    elif op.NAME == "activation_reduce":
        _render_activation_reduce(traced_op, tracer, lines, consumed, is_final)
    elif consumed:
        _render_reduction(traced_op, tracer, lines, consumed, is_final)
    else:
        _render_elementwise(traced_op, tracer, lines, is_final)


def _render_full_nest_op(
    traced_op: TracedOp, tracer: EagerTracer, lines: list[str], consumed: list[str], is_final: bool
) -> None:
    """Build RenderContext and delegate to op.render() for full loop nest.

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

    operand_info: dict[str, OperandInfo] = {}
    non_input_tensors: dict[str, Tensor] = {}
    active = frozenset(out_tinfo.dims) | frozenset(consumed)
    for slot, tensor_name in traced_op.operand_names.items():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is None:
            continue
        dtype_expr = tracer.tensor_dtypes.get(tensor_name, f"{tensor_name}.dtype")
        operand_info[slot] = OperandInfo(
            tensor_name=tensor_name, dims=tinfo.dims, is_input=tinfo.is_input, dtype_expr=dtype_expr
        )
        if not tinfo.is_input:
            non_input_tensors[slot] = _build_tensor(
                f"sbuf_{tensor_name}", tinfo, tracer.dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, active, True
            )

    ctx = RenderContext(
        dim_info=dict(tracer.dims),
        consumed_dims=consumed,
        operand_info=operand_info,
        operands=non_input_tensors,
        is_final=is_final,
        output_name=out_name,
        output_dims=out_tinfo.dims,
        output_dtype=tracer.tensor_dtypes[out_name],
        config_kwargs=traced_op.config_kwargs,
        op_idx=traced_op.op_idx,
    )

    render_lines = op.render(ctx)
    lines.extend(render_lines)


def _render_elementwise(traced_op: TracedOp, tracer: EagerTracer, lines: list[str], is_final: bool) -> None:
    """Render an elementwise op (no consumed dims).

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        is_final: Whether this is the final op.
    """
    op = traced_op.op
    if op.NAME == "nc_transpose":
        _render_transpose(traced_op, tracer, lines, is_final)
        return

    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]
    dims = tracer.dims
    active = set(out_tinfo.dims)
    needs_psum = op.NAME in _PSUM_OPS

    if not is_final:
        _emit_output_allocs(traced_op, tracer, lines)

    indent = _emit_output_loops(out_tinfo, dims, lines)
    staging = _emit_dma_loads(traced_op, tracer, lines, indent, EMPTY_STR_SET)
    operand_tensors = _build_operand_tensors(traced_op, tracer, active, staging, EMPTY_STR_SET)

    if is_final:
        _render_ew_final(traced_op, tracer, lines, indent, operand_tensors, needs_psum)
    else:
        _render_ew_intermediate(traced_op, tracer, lines, indent, operand_tensors, active, needs_psum)


def _transpose_uncap_par_slots(traced_op: TracedOp, tracer: EagerTracer) -> frozenset[str]:
    """Find DMA slots needing uncapped partition for transpose sub-loops.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.

    Returns:
        Frozenset of operand slot names needing uncapped partition.
    """
    slots: set[str] = set()
    for slot_name, tensor_name in traced_op.operand_names.items():
        tinfo = tracer.tensors.get(tensor_name)
        if tinfo is not None and tinfo.is_input and tinfo.dims:
            par_dim = tinfo.dims[0]
            if tracer.dims[par_dim].tile_size > SBUF_PMAX:
                slots.add(slot_name)
    return frozenset(slots)


def _render_transpose(traced_op: TracedOp, tracer: EagerTracer, lines: list[str], is_final: bool) -> None:
    """Render nc_transpose via PSUM then tensor_copy to SBUF.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        is_final: Whether this is the final op.
    """
    op = traced_op.op
    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]
    dims = tracer.dims
    active = set(out_tinfo.dims)
    out_key = _primary_output_key(op)
    sbuf_name = f"sbuf_{out_name}"
    dtype = tracer.tensor_dtypes[out_name]

    out_t = _build_tensor(sbuf_name, out_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, frozenset(active), False)
    lines.append(f"{sbuf_name} = nl.ndarray({out_t.shape()}, dtype={dtype}, buffer=nl.sbuf)")

    uncap_slots = _transpose_uncap_par_slots(traced_op, tracer)
    indent = _emit_output_loops(out_tinfo, dims, lines)
    staging = _emit_dma_loads(traced_op, tracer, lines, indent, uncap_slots)
    operand_tensors = _build_operand_tensors(traced_op, tracer, active, staging, uncap_slots)

    psum_name = f"psum_{out_name}"
    psum_t = _tile_tensor(psum_name, out_tinfo, dims, "psum", EMPTY_STR_SET, False)
    lines.append(f"{indent}{psum_name} = nl.ndarray({psum_t.shape()}, dtype={dtype}, buffer=nl.psum)")
    output_tensors = {out_key: psum_t}
    ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
    _emit_isa(op, ctx, lines, indent)
    _emit_transpose_copy(out_t, psum_t, out_tinfo, dims, lines, indent)

    if is_final:
        _emit_dma_store(out_tinfo, dims, lines, indent)


def _emit_transpose_copy(
    dst_t: Tensor, src_t: Tensor, out_tinfo: "TensorInfo", dims: dict, lines: list[str], indent: str
) -> None:
    """Emit tensor_copy from PSUM to SBUF after transpose.

    Args:
        dst_t: SBUF destination tensor.
        src_t: PSUM source tensor.
        out_tinfo: Output tensor info.
        dims: Dimension metadata.
        lines: Output lines list.
        indent: Current indentation.
    """
    par_dim = out_tinfo.dims[0]
    par_ts = dims[par_dim].tile_size
    if par_ts <= SBUF_PMAX:
        lines.append(
            f"{indent}nisa.tensor_copy(dst={dst_t.default_indexed_slice()}, src={src_t.default_indexed_slice()})"
        )
    else:
        _emit_tensor_copy_uncapped(dst_t, src_t, par_ts, lines, indent)


def _render_ew_final(
    traced_op: TracedOp,
    tracer: EagerTracer,
    lines: list[str],
    indent: str,
    operand_tensors: dict[str, Tensor],
    needs_psum: bool,
) -> None:
    """Render final elementwise op with DMA store.

    When needs_psum is True, writes to PSUM first then copies to SBUF
    (TensorEngine ops accumulate in PSUM). Otherwise writes directly
    to SBUF.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        indent: Current indentation.
        operand_tensors: Operand tensors for render context.
        needs_psum: Whether op must write to PSUM first.
    """
    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]
    dims = tracer.dims
    out_key = _primary_output_key(traced_op.op)
    dtype = tracer.tensor_dtypes[out_name]

    if needs_psum:
        psum_name = f"psum_{out_name}"
        psum_t = _tile_tensor(psum_name, out_tinfo, dims, "psum", EMPTY_STR_SET, True)
        lines.append(f"{indent}{psum_name} = nl.ndarray({psum_t.shape()}, dtype={dtype}, buffer=nl.psum)")
        output_tensors = {out_key: psum_t}
        ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
        _emit_isa(traced_op.op, ctx, lines, indent)
        sbuf_name = f"sbuf_{out_name}"
        tile_t = _tile_tensor(sbuf_name, out_tinfo, dims, "sbuf", EMPTY_STR_SET, True)
        lines.append(f"{indent}{sbuf_name} = nl.ndarray({tile_t.shape()}, dtype={dtype}, buffer=nl.sbuf)")
        par_dim = out_tinfo.dims[0]
        _emit_tensor_copy_chunked(tile_t, psum_t, dims[par_dim].tile_size, par_dim, lines, indent)
    else:
        sbuf_name = f"sbuf_{out_name}"
        tile_t = _tile_tensor(sbuf_name, out_tinfo, dims, "sbuf", EMPTY_STR_SET, True)
        lines.append(f"{indent}{sbuf_name} = nl.ndarray({tile_t.shape()}, dtype={dtype}, buffer=nl.sbuf)")
        output_tensors = {out_key: tile_t}
        ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
        _emit_isa(traced_op.op, ctx, lines, indent)

    _emit_dma_store(out_tinfo, dims, lines, indent)


def _build_sbuf_output_tensors(traced_op: TracedOp, tracer: EagerTracer, active: frozenset[str]) -> dict[str, Tensor]:
    """Build SBUF output tensors for all outputs of an op.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        active: Active dimension set.

    Returns:
        Maps output key to SBUF Tensor.
    """
    result: dict[str, Tensor] = {}
    for out_idx, out_key in enumerate(traced_op.op.OUTPUT_AXES.keys()):
        if out_idx >= len(traced_op.output_names):
            break
        oname = traced_op.output_names[out_idx]
        otinfo = tracer.tensors[oname]
        result[out_key] = _build_tensor(
            f"sbuf_{oname}", otinfo, tracer.dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, active, True
        )
    return result


def _render_ew_intermediate(
    traced_op: TracedOp,
    tracer: EagerTracer,
    lines: list[str],
    indent: str,
    operand_tensors: dict[str, Tensor],
    active: set[str],
    needs_psum: bool,
) -> None:
    """Render non-final elementwise op to full-range buffer.

    When needs_psum is True, writes to PSUM then copies to SBUF.
    Otherwise builds SBUF output tensors directly for all outputs.

    Args:
        traced_op: The traced op.
        tracer: Tracer state.
        lines: Output lines list.
        indent: Current indentation.
        operand_tensors: Operand tensors for render context.
        active: Active dimension set.
        needs_psum: Whether op must write to PSUM first.
    """
    op = traced_op.op
    dims = tracer.dims
    out_name = traced_op.output_names[0]
    out_tinfo = tracer.tensors[out_name]

    if needs_psum:
        psum_name = f"psum_{out_name}"
        psum_t = _tile_tensor(psum_name, out_tinfo, dims, "psum", EMPTY_STR_SET, True)
        dtype = tracer.tensor_dtypes[out_name]
        lines.append(f"{indent}{psum_name} = nl.ndarray({psum_t.shape()}, dtype={dtype}, buffer=nl.psum)")
        output_tensors = {_primary_output_key(op): psum_t}
        ctx = _make_ctx(traced_op, tracer, output_tensors, operand_tensors)
        _emit_isa(op, ctx, lines, indent)
        sbuf_t = _build_tensor(
            f"sbuf_{out_name}", out_tinfo, dims, "sbuf", EMPTY_STR_INT, EMPTY_STR_INT, frozenset(active), True
        )
        par_dim = out_tinfo.dims[0]
        _emit_tensor_copy_chunked(sbuf_t, psum_t, dims[par_dim].tile_size, par_dim, lines, indent)
    else:
        output_tensors_sbuf = _build_sbuf_output_tensors(traced_op, tracer, frozenset(active))
        ctx = _make_ctx(traced_op, tracer, output_tensors_sbuf, operand_tensors)
        _emit_isa(op, ctx, lines, indent)
