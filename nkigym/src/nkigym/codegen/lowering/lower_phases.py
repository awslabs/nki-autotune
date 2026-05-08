"""``LowerPhases`` pass: per-``(op_cls, phase)`` ISA call-site emission.

Holds the body-emitter registry and the individual emitter implementations.
Each emitter receives a fully-resolved :class:`BodyLeaf`, the enclosing
:class:`KernelModule`, and the current forest path's same-dim ancestor
state (``path_names`` / ``path_trips``). It emits the ISA call-site
snippet (e.g. ``nisa.dma_copy(...)``, ``nisa.nc_matmul(...)``) for that
leaf; the walker opens and closes the wrapping ``for`` loops.

Multi-phase ops (``NKIMatmul``, ``NKIActivationReduce``) register one
emitter per ``(op_kind, phase)`` pair. Single-phase ops register under
phase ``"main"``. The walker in :mod:`nkigym.codegen.lowering.emit_source`
and the pipeline machinery in
:mod:`nkigym.codegen.lowering.inject_software_pipeline` both dispatch
through :data:`_BODY_EMITTERS`.

Slot-index expressions for multi-buffered tiles come from
:mod:`nkigym.codegen.lowering.inject_multi_buffer`; the ``pipeline_dim``
and ``stage_offset`` kwargs thread software-pipelining's stage skew
through to those expressions.
"""

from collections.abc import Callable

from nkigym.codegen.lowering._emit_utils import _hbm_name, _sbuf_name
from nkigym.codegen.lowering.inject_multi_buffer import (
    hbm_tile_slice,
    sbuf_tile_slice,
    slot_expr,
    swapped_dst_tile_slice,
)
from nkigym.codegen.lowering.place_buffers import tensor_total_slots

_REDUCE_IDENTITY: dict[str, float] = {"add": 0.0, "max": float("-inf")}
_REDUCE_MERGE_OP: dict[str, str] = {"add": "nl.add", "max": "nl.maximum"}


_BODY_EMITTERS: dict[tuple[str, str], Callable] = {}
"""Per-``(op_kind, phase)`` body emitter registry.

A body emitter receives ``(writer, module, leaf, path_names,
path_trips)`` and emits that phase's source lines without any
loop headers — the walker is responsible for opening and closing the
loops that frame the body. Emitters read every op detail from the
self-describing :class:`BodyLeaf` and consult ``module`` for dim/tensor
declarations. Single-phase ops register under phase ``"main"``.
"""


def _register_body(op_kind: str, phase: str = "main"):
    """Decorator: register a body emitter for ``(op_kind, phase)``."""

    def wrap(fn: Callable) -> Callable:
        """Attach ``fn`` to the ``(op_kind, phase)`` slot in the registry."""
        _BODY_EMITTERS[(op_kind, phase)] = fn
        return fn

    return wrap


@_register_body("NKILoad", "main")
def _body_load(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` at the innermost open-loop point."""
    src_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
    dst_p_slots = tensor_total_slots(dst_tensor, dst_tensor.dim_ids[0], module)
    dst_f_slots = tensor_total_slots(dst_tensor, dst_tensor.dim_ids[1], module) if len(dst_tensor.dim_ids) > 1 else 1
    src_p_slots = module.dims[src_tensor.dim_ids[0]].num_tiles
    src_f_slots = module.dims[src_tensor.dim_ids[1]].num_tiles if len(src_tensor.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = hbm_tile_slice(
        src_name, src_tensor.dim_ids, p_tile, f_tile, path_names, path_trips, src_p_slots, src_f_slots, off_p, off_f
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIStore", "main")
def _body_store(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` SBUF->HBM at the innermost open-loop point."""
    src_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
    dst_p_slots = module.dims[dst_tensor.dim_ids[0]].num_tiles
    dst_f_slots = module.dims[dst_tensor.dim_ids[1]].num_tiles if len(dst_tensor.dim_ids) > 1 else 1
    src_p_slots = tensor_total_slots(src_tensor, src_tensor.dim_ids[0], module)
    src_f_slots = tensor_total_slots(src_tensor, src_tensor.dim_ids[1], module) if len(src_tensor.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = hbm_tile_slice(
        _hbm_name(dst_name),
        dst_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = sbuf_tile_slice(
        _sbuf_name(src_name),
        src_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        off_p,
        off_f,
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIActivation", "main")
def _body_activation(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.activation`` at the innermost open-loop point."""
    src_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    src = module.tensors[src_name]
    dst = module.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1] if len(src.dim_ids) > 1 else None
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
    act = leaf.kwargs["op"]
    scale = leaf.kwargs.get("scale", 1.0)
    bias = leaf.kwargs.get("bias", 0.0)
    dst_p_slots = tensor_total_slots(dst, dst.dim_ids[0], module)
    dst_f_slots = tensor_total_slots(dst, dst.dim_ids[1], module) if len(dst.dim_ids) > 1 else 1
    src_p_slots = tensor_total_slots(src, src.dim_ids[0], module)
    src_f_slots = tensor_total_slots(src, src.dim_ids[1], module) if len(src.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        off_p,
        off_f,
    )
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")


@_register_body("NKITensorScalar", "main")
def _body_tensor_scalar(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.tensor_scalar`` at the innermost open-loop point."""
    data_name = leaf.reads["data"]
    op0_name = leaf.reads["operand0"]
    dst_name = leaf.writes[0]
    data = module.tensors[data_name]
    op0 = module.tensors[op0_name]
    dst = module.tensors[dst_name]
    p_axis = data.dim_ids[0]
    f_axis = data.dim_ids[1]
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size
    op_name = leaf.kwargs["op"]
    dst_p_slots = tensor_total_slots(dst, dst.dim_ids[0], module)
    dst_f_slots = tensor_total_slots(dst, dst.dim_ids[1], module) if len(dst.dim_ids) > 1 else 1
    data_p_slots = tensor_total_slots(data, data.dim_ids[0], module)
    data_f_slots = tensor_total_slots(data, data.dim_ids[1], module) if len(data.dim_ids) > 1 else 1
    op0_p_slots = tensor_total_slots(op0, op0.dim_ids[0], module)
    op0_f_slots = tensor_total_slots(op0, op0.dim_ids[1], module) if len(op0.dim_ids) > 1 else 1
    """Offset per-operand: stage_offset applies to the ancestors of the
    pipelined dim. Each operand's axes compare independently (op0 may
    be 1D and thus lack an f-axis)."""
    dst_off_p = stage_offset if dst.dim_ids[0] == pipeline_dim else 0
    dst_off_f = stage_offset if len(dst.dim_ids) > 1 and dst.dim_ids[1] == pipeline_dim else 0
    data_off_p = stage_offset if data.dim_ids[0] == pipeline_dim else 0
    data_off_f = stage_offset if len(data.dim_ids) > 1 and data.dim_ids[1] == pipeline_dim else 0
    op0_off_p = stage_offset if op0.dim_ids[0] == pipeline_dim else 0
    op0_off_f = stage_offset if len(op0.dim_ids) > 1 and op0.dim_ids[1] == pipeline_dim else 0
    dst_expr = sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        dst_off_p,
        dst_off_f,
    )
    data_expr = sbuf_tile_slice(
        _sbuf_name(data_name),
        data.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        data_p_slots,
        data_f_slots,
        data_off_p,
        data_off_f,
    )
    op0_expr = sbuf_tile_slice(
        _sbuf_name(op0_name),
        op0.dim_ids,
        p_tile,
        1,
        path_names,
        path_trips,
        op0_p_slots,
        op0_f_slots,
        op0_off_p,
        op0_off_f,
    )
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")


@_register_body("NKITranspose", "main")
def _body_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit PSUM alloc + ``nc_transpose`` + ``tensor_copy`` at the innermost open-loop point."""
    src_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    src = module.tensors[src_name]
    dst = module.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = module.dims[src_p_axis].tile_size
    f_tile = module.dims[src_f_axis].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_p_slots = tensor_total_slots(src, src.dim_ids[0], module)
    src_f_slots = tensor_total_slots(src, src.dim_ids[1], module)
    src_off_p = stage_offset if src_p_axis == pipeline_dim else 0
    src_off_f = stage_offset if src_f_axis == pipeline_dim else 0
    src_expr = sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    dst_p_slots = tensor_total_slots(dst, dst.dim_ids[0], module)
    dst_f_slots = tensor_total_slots(dst, dst.dim_ids[1], module)
    """dst's P slot uses src_f_axis ancestors; dst's F slot uses src_p_axis ancestors."""
    dst_off_p = stage_offset if src_f_axis == pipeline_dim else 0
    dst_off_f = stage_offset if src_p_axis == pipeline_dim else 0
    dst_expr = swapped_dst_tile_slice(
        dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips, dst_p_slots, dst_f_slots, dst_off_p, dst_off_f
    )
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")


@_register_body("NKIDMATranspose", "main")
def _body_dma_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_transpose`` at the innermost open-loop point."""
    src_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    src = module.tensors[src_name]
    dst = module.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = module.dims[src_p_axis].tile_size
    f_tile = module.dims[src_f_axis].tile_size
    src_p_slots = tensor_total_slots(src, src.dim_ids[0], module)
    src_f_slots = tensor_total_slots(src, src.dim_ids[1], module)
    src_off_p = stage_offset if src_p_axis == pipeline_dim else 0
    src_off_f = stage_offset if src_f_axis == pipeline_dim else 0
    src_expr = sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    dst_p_slots = tensor_total_slots(dst, dst.dim_ids[0], module)
    dst_f_slots = tensor_total_slots(dst, dst.dim_ids[1], module)
    dst_off_p = stage_offset if src_f_axis == pipeline_dim else 0
    dst_off_f = stage_offset if src_p_axis == pipeline_dim else 0
    dst_expr = swapped_dst_tile_slice(
        dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips, dst_p_slots, dst_f_slots, dst_off_p, dst_off_f
    )
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")


@_register_body("NKIMatmul", "psum_init")
def _body_matmul_psum_init(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Allocate + memset the PSUM accumulator once per (M, N) tile.

    PSUM lifetime spans the entire K loop. ``path_names`` / ``path_trips``
    are unused — the alloc uses constant ``(p_tile_M, f_tile_N)`` shapes
    derived from the leaf's axis map. ``pipeline_dim`` / ``stage_offset``
    are unused because the alloc is index-free.
    """
    _ = path_names, path_trips, pipeline_dim, stage_offset
    m_dim = leaf.axis_map["M"]
    n_dim = leaf.axis_map["N"]
    p_tile_M = module.dims[m_dim].tile_size
    f_tile_N = module.dims[n_dim].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")


@_register_body("NKIMatmul", "compute")
def _body_matmul_compute(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.nc_matmul`` per K tile inside the K loop."""
    stat_name = leaf.reads["stationary"]
    mov_name = leaf.reads["moving"]
    stat = module.tensors[stat_name]
    mov = module.tensors[mov_name]
    m_dim = leaf.axis_map["M"]
    n_dim = leaf.axis_map["N"]
    k_dim = leaf.axis_map["K"]
    p_tile_M = module.dims[m_dim].tile_size
    f_tile_N = module.dims[n_dim].tile_size
    p_tile_K = module.dims[k_dim].tile_size
    stat_p_slots = tensor_total_slots(stat, stat.dim_ids[0], module)
    stat_f_slots = tensor_total_slots(stat, stat.dim_ids[1], module) if len(stat.dim_ids) > 1 else 1
    mov_p_slots = tensor_total_slots(mov, mov.dim_ids[0], module)
    mov_f_slots = tensor_total_slots(mov, mov.dim_ids[1], module) if len(mov.dim_ids) > 1 else 1
    stat_off_p = stage_offset if stat.dim_ids[0] == pipeline_dim else 0
    stat_off_f = stage_offset if len(stat.dim_ids) > 1 and stat.dim_ids[1] == pipeline_dim else 0
    mov_off_p = stage_offset if mov.dim_ids[0] == pipeline_dim else 0
    mov_off_f = stage_offset if len(mov.dim_ids) > 1 and mov.dim_ids[1] == pipeline_dim else 0
    stat_expr = sbuf_tile_slice(
        _sbuf_name(stat_name),
        stat.dim_ids,
        p_tile_K,
        p_tile_M,
        path_names,
        path_trips,
        stat_p_slots,
        stat_f_slots,
        stat_off_p,
        stat_off_f,
    )
    mov_expr = sbuf_tile_slice(
        _sbuf_name(mov_name),
        mov.dim_ids,
        p_tile_K,
        f_tile_N,
        path_names,
        path_trips,
        mov_p_slots,
        mov_f_slots,
        mov_off_p,
        mov_off_f,
    )
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKIMatmul", "drain")
def _body_matmul_drain(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Drain the PSUM accumulator into the output SBUF once the K loop closes."""
    out_name = leaf.writes[0]
    out = module.tensors[out_name]
    m_dim = leaf.axis_map["M"]
    n_dim = leaf.axis_map["N"]
    p_tile_M = module.dims[m_dim].tile_size
    f_tile_N = module.dims[n_dim].tile_size
    out_p_slots = tensor_total_slots(out, out.dim_ids[0], module)
    out_f_slots = tensor_total_slots(out, out.dim_ids[1], module) if len(out.dim_ids) > 1 else 1
    out_off_p = stage_offset if out.dim_ids[0] == pipeline_dim else 0
    out_off_f = stage_offset if len(out.dim_ids) > 1 and out.dim_ids[1] == pipeline_dim else 0
    out_expr = sbuf_tile_slice(
        _sbuf_name(out_name),
        out.dim_ids,
        p_tile_M,
        f_tile_N,
        path_names,
        path_trips,
        out_p_slots,
        out_f_slots,
        out_off_p,
        out_off_f,
    )
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")


@_register_body("NKIActivationReduce", "reduce_close")
def _body_ar_reduce_close(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Fold ``slot_vec`` into the op's ``(P, 1)`` output via ``nisa.tensor_reduce``.

    Runs after the F loop exits; the slot vector holds ``num_f_tiles``
    partial sums, one per F-tile. ``axis=2`` reduces the free axis of
    the 3D ``(p_tile, 1, num_f_tiles)`` slot_vec.
    """
    dst_name = leaf.writes[0]
    dst = module.tensors[dst_name]
    p_axis = leaf.axis_map["P"]
    p_tile = module.dims[p_axis].tile_size
    f_axis = leaf.axis_map["F"]
    num_f = module.dims[f_axis].num_tiles
    reduce_op = leaf.kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    dst_p_slots = tensor_total_slots(dst, dst.dim_ids[0], module)
    off_p = stage_offset if p_axis == pipeline_dim else 0
    p_slot = slot_expr(path_names, path_trips, p_axis, dst_p_slots, off_p)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    slot_name = leaf.op_local_buffers["slot_vec"].emitted_name
    src_slot = f"{slot_name}[0:{p_tile}, 0:1, 0:{num_f}]"
    w.line(f"nisa.tensor_reduce({dst_slot}, {merge}, {src_slot}, axis=2)")


@_register_body("NKIActivationReduce", "reduce_step")
def _body_ar_reduce_step(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Per-F-tile activation_reduce writing into a distinct slot of ``slot_vec``.

    The dst operand goes to the op-local scratch buffer (discarded);
    the reduce_res lands in ``slot_vec[0:p_tile, 0, f_slot:f_slot+1]``
    where ``f_slot`` is the current F-tile ordinal on the path. No
    prologue memset, no cross-tile merge — each call owns its slot.
    """
    src_name = leaf.reads["data"]
    src = module.tensors[src_name]
    p_axis = leaf.axis_map["P"]
    f_axis = leaf.axis_map["F"]
    p_tile = module.dims[p_axis].tile_size
    f_tile = module.dims[f_axis].tile_size
    num_f = module.dims[f_axis].num_tiles
    act_op = leaf.kwargs.get("op", "copy")
    reduce_op = leaf.kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    """slot_vec's F extent is num_f — the slot ordinal indexes directly into it."""
    off_f = stage_offset if f_axis == pipeline_dim else 0
    f_slot = slot_expr(path_names, path_trips, f_axis, num_f, off_f)
    scratch_name = leaf.op_local_buffers["scratch"].emitted_name
    slot_name = leaf.op_local_buffers["slot_vec"].emitted_name
    scratch_slot = f"{scratch_name}[0:{p_tile}, 0, 0:{f_tile}]"
    reduce_res_slot = f"{slot_name}[0:{p_tile}, 0, {f_slot} : {f_slot} + 1]"
    src_p_slots = tensor_total_slots(src, src.dim_ids[0], module)
    src_f_slots = tensor_total_slots(src, src.dim_ids[1], module) if len(src.dim_ids) > 1 else 1
    src_off_p = stage_offset if src.dim_ids[0] == pipeline_dim else 0
    src_off_f = stage_offset if len(src.dim_ids) > 1 and src.dim_ids[1] == pipeline_dim else 0
    src_expr = sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst={scratch_slot},")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res={reduce_res_slot},")
    w.dedent()
    w.line(")")
