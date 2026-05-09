"""Per-op-class body emitters: ``NKIAlloc``, ``NKILoad``, ``NKIMatmul``, etc.

Each emitter receives a fully-resolved :class:`BodyLeaf`, the enclosing
:class:`KernelModule`, and the current forest path's same-dim ancestor
state (``path_names`` / ``path_trips``). It emits the ISA call-site
snippet (e.g. ``nisa.dma_copy(...)``, ``nisa.nc_matmul(...)``) for that
leaf; the walker opens and closes the wrapping ``for`` loops.

Allocation is itself an op (``NKIAlloc``) whose emitter synthesizes the
``nl.ndarray(...)`` declaration. SBUF/PSUM tensors get 3D shapes
``(P_tile, num_p_slots, num_f_tiles*f_tile)`` per the NKI convention;
shape computation delegates to :func:`place_buffers.sbuf_shape`. HBM
tensors keep their declared 2D shape.

Slot-index expressions for multi-buffered tiles come from
:mod:`nkigym.codegen.lowering.inject_multi_buffer`; the ``pipeline_dim``
and ``stage_offset`` kwargs thread software-pipelining's stage skew
through to those expressions.
"""

from collections.abc import Callable

from nkigym.codegen.lowering.inject_multi_buffer import hbm_tile_slice, sbuf_tile_slice
from nkigym.codegen.lowering.place_buffers import sbuf_shape

_BODY_EMITTERS: dict[str, Callable] = {}
"""Per-op-class body emitter registry.

A body emitter receives ``(writer, module, leaf, path_names, path_trips,
pipeline_dim=None, stage_offset=0)`` and emits that op's source lines
without any loop headers — the walker is responsible for opening and
closing the loops that frame the body. Emitters read every op detail
from the self-describing :class:`BodyLeaf` and consult ``module`` for
dim/tensor declarations.
"""

_LOCATION_BUFFER_EXPR = {"hbm": "nl.shared_hbm", "sbuf": "nl.sbuf", "psum": "nl.psum"}


def _register_body(op_kind: str):
    """Decorator: register a body emitter for ``op_kind``."""

    def wrap(fn: Callable) -> Callable:
        """Attach ``fn`` to the ``op_kind`` slot in the registry."""
        _BODY_EMITTERS[op_kind] = fn
        return fn

    return wrap


def _build_slice(
    tensor_name: str,
    tensor,
    module,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    pipeline_dim: str | None,
    stage_offset: int,
) -> str:
    """Dispatch on tensor.location to build HBM or SBUF tile slice expression."""
    from nkigym.codegen.lowering.place_buffers import tensor_total_slots

    if tensor.location == "hbm":
        """HBM tensor: tile size from module.dims, slots from num_tiles."""
        p_axis = tensor.dim_ids[0]
        f_axis = tensor.dim_ids[1] if len(tensor.dim_ids) > 1 else None
        p_tile = module.dims[p_axis].tile_size
        f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
        p_slots = module.dims[p_axis].num_tiles
        f_slots = module.dims[f_axis].num_tiles if f_axis is not None else 1
        off_p = stage_offset if p_axis == pipeline_dim else 0
        off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
        return hbm_tile_slice(
            tensor_name, tensor.dim_ids, p_tile, f_tile, path_names, path_trips, p_slots, f_slots, off_p, off_f
        )
    else:
        """SBUF/PSUM tensor: tile size from module.dims, slots from tensor_total_slots."""
        p_axis = tensor.dim_ids[0]
        f_axis = tensor.dim_ids[1] if len(tensor.dim_ids) > 1 else None
        p_tile = module.dims[p_axis].tile_size
        f_tile = module.dims[f_axis].tile_size if f_axis is not None else 1
        p_slots = tensor_total_slots(tensor, p_axis, module)
        f_slots = tensor_total_slots(tensor, f_axis, module) if f_axis is not None else 1
        off_p = stage_offset if p_axis == pipeline_dim else 0
        off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
        return sbuf_tile_slice(
            tensor_name, tensor.dim_ids, p_tile, f_tile, path_names, path_trips, p_slots, f_slots, off_p, off_f
        )


@_register_body("NKIAlloc")
def _body_alloc(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=<buffer_expr>)``.

    SBUF/PSUM tensors get the 3D ``(P_tile, num_p_slots, num_f_tiles*f_tile)``
    layout per the NKI ``nl.ndarray`` convention — shape comes from
    :func:`place_buffers.sbuf_shape`, which folds ``buffer_degree`` and
    LCA-coverage into the P-slot count. HBM tensors keep their declared
    2D shape (no tile folding — HBM is a flat global buffer).

    ``pipeline_dim`` and ``stage_offset`` are unused; allocation is
    index-free.
    """
    _ = path_names, path_trips, pipeline_dim, stage_offset
    name = leaf.writes[0]
    tensor = module.tensors[name]
    buffer_expr = _LOCATION_BUFFER_EXPR[tensor.location]
    if tensor.location == "hbm":
        shape_str = ", ".join(str(x) for x in tensor.shape)
        shape_tuple = f"({shape_str},)" if len(tensor.shape) == 1 else f"({shape_str})"
    else:
        shape = sbuf_shape(tensor, module)
        shape_tuple = f"({shape[0]}, {shape[1]}, {shape[2]})"
    w.line(f"{name} = nl.ndarray({shape_tuple}, dtype=nl.{tensor.dtype}, buffer={buffer_expr})")


@_register_body("NKIMemset")
def _body_memset(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.memset(<dst_slice>, value=<value>)``."""
    dst_name = leaf.writes[0]
    tensor = module.tensors[dst_name]
    value = leaf.kwargs["value"]
    dst_slice = _build_slice(dst_name, tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.memset({dst_slice}, value={value})")


@_register_body("NKILoad")
def _body_load(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.dma_copy(dst=<dst>, src=<src>)``."""
    src_name = leaf.reads["src"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    src_slice = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_copy(dst={dst_slice}, src={src_slice})")


@_register_body("NKIStore")
def _body_store(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.dma_copy(dst=<dst>, src=<src>)`` SBUF->HBM."""
    src_name = leaf.reads["src"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    src_slice = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_copy(dst={dst_slice}, src={src_slice})")


@_register_body("NKITensorCopy")
def _body_tensor_copy(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.tensor_copy(<dst>, <src>)``."""
    src_name = leaf.reads["src"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    src_slice = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_copy({dst_slice}, {src_slice})")


@_register_body("NKITensorReduce")
def _body_tensor_reduce(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.tensor_reduce(<dst>, <op_expr>, <data>, axis=<axis>)``."""
    data_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    op = leaf.kwargs["op"]
    axis = leaf.kwargs["axis"]
    op_expr = "nl.add" if op == "add" else "nl.maximum"
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    data_slice = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_reduce({dst_slice}, {op_expr}, {data_slice}, axis={axis})")


@_register_body("NKIMatmul")
def _body_matmul(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.nc_matmul(dst=<dst>, stationary=..., moving=...)``."""
    stat_name = leaf.reads["stationary"]
    mov_name = leaf.reads["moving"]
    dst_name = leaf.reads_writes[0]
    stat_tensor = module.tensors[stat_name]
    mov_tensor = module.tensors[mov_name]
    dst_tensor = module.tensors[dst_name]
    stat_slice = _build_slice(stat_name, stat_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    mov_slice = _build_slice(mov_name, mov_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst={dst_slice},")
    w.line(f"stationary={stat_slice},")
    w.line(f"moving={mov_slice},")
    w.dedent()
    w.line(")")


@_register_body("NKIActivation")
def _body_activation(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.activation(dst=..., op=nl.<act>, data=..., scale=..., bias=...)``."""
    data_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    act = leaf.kwargs["op"]
    scale = leaf.kwargs.get("scale", 1.0)
    bias = leaf.kwargs.get("bias", 0.0)
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    data_slice = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.activation(dst={dst_slice}, op=nl.{act}, data={data_slice}, scale={scale}, bias={bias})")


@_register_body("NKIActivationReduce")
def _body_activation_reduce(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit multi-line ``nisa.activation_reduce(dst=..., op=nl.<act>, data=..., reduce_op=..., reduce_res=...)``."""
    data_name = leaf.reads["data"]
    dst_name = leaf.writes[0]
    reduce_res_name = leaf.writes[1]
    data_tensor = module.tensors[data_name]
    dst_tensor = module.tensors[dst_name]
    reduce_res_tensor = module.tensors[reduce_res_name]
    act = leaf.kwargs.get("op", "copy")
    reduce_op = leaf.kwargs.get("reduce_op", "add")
    reduce_op_expr = "nl.add" if reduce_op == "add" else "nl.maximum"
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    data_slice = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    reduce_res_slice = _build_slice(
        reduce_res_name, reduce_res_tensor, module, path_names, path_trips, pipeline_dim, stage_offset
    )
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst={dst_slice},")
    w.line(f"op=nl.{act},")
    w.line(f"data={data_slice},")
    w.line(f"reduce_op={reduce_op_expr},")
    w.line(f"reduce_res={reduce_res_slice},")
    w.dedent()
    w.line(")")


@_register_body("NKITensorScalar")
def _body_tensor_scalar(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.tensor_scalar(dst=..., data=..., op0=nl.<op>, operand0=...)``."""
    data_name = leaf.reads["data"]
    op0_name = leaf.reads["operand0"]
    dst_name = leaf.writes[0]
    data_tensor = module.tensors[data_name]
    op0_tensor = module.tensors[op0_name]
    dst_tensor = module.tensors[dst_name]
    op_name = leaf.kwargs["op"]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    data_slice = _build_slice(data_name, data_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    op0_slice = _build_slice(op0_name, op0_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.tensor_scalar(dst={dst_slice}, data={data_slice}, op0=nl.{op_name}, operand0={op0_slice})")


@_register_body("NKITranspose")
def _body_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.nc_transpose(<dst>, <src>)``."""
    src_name = leaf.reads["src"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    src_slice = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.nc_transpose({dst_slice}, {src_slice})")


@_register_body("NKIDMATranspose")
def _body_dma_transpose(w, module, leaf, path_names, path_trips, pipeline_dim=None, stage_offset=0) -> None:
    """Emit ``nisa.dma_transpose(<dst>, <src>)``."""
    src_name = leaf.reads["src"]
    dst_name = leaf.writes[0]
    src_tensor = module.tensors[src_name]
    dst_tensor = module.tensors[dst_name]
    dst_slice = _build_slice(dst_name, dst_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    src_slice = _build_slice(src_name, src_tensor, module, path_names, path_trips, pipeline_dim, stage_offset)
    w.line(f"nisa.dma_transpose({dst_slice}, {src_slice})")
