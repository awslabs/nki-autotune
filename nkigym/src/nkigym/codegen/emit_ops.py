"""Per-op ISA call emitters for the iter-var IR.

Each emitter receives an :class:`NKIOpCall` (op-level kwargs), the
enclosing :class:`SBlock` (operand :class:`BufferAccess` maps), and an
:class:`EmitCtx` (iter-var id → rendered name + module tensors). It
returns the exact ``nisa.*`` / ``nl.ndarray`` source fragment for that
block. Loop headers are opened by the walker in :mod:`emit_source`;
emitters only produce the body fragment.

The registry is keyed by the NKIOp subclass's ``__name__`` so the
dispatch is robust against re-exports.
"""

from collections.abc import Callable

from nkigym.codegen._emit_utils import EmitCtx, emit_slice
from nkigym.ir.ir import NKIOpCall, SBlock

_DTYPE_MAP: dict[str, str] = {"float32": "nl.float32", "float16": "nl.float16", "bfloat16": "nl.bfloat16"}

_LOCATION_BUFFER_EXPR: dict[str, str] = {"hbm": "nl.shared_hbm", "sbuf": "nl.sbuf", "psum": "nl.psum"}


_EMITTERS: dict[str, Callable[[NKIOpCall, SBlock, EmitCtx], list[str]]] = {}
"""Registry: NKIOp subclass name → emitter function.

Each emitter returns a list of source lines (most return a single line;
multi-line emitters — matmul, activation_reduce — return several).
"""


def emit_op_call(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Dispatch to the per-op emitter by class name. Returns source lines."""
    emitter = _EMITTERS.get(call.op_cls.__name__)
    if emitter is None:
        raise NotImplementedError(f"No emitter for op class {call.op_cls.__name__!r}")
    return emitter(call, block, ctx)


def _register(op_name: str) -> Callable[[Callable], Callable]:
    """Decorator: register ``fn`` as the emitter for ``op_name``."""

    def wrap(fn: Callable) -> Callable:
        """Attach ``fn`` to the ``op_name`` slot in the registry."""
        _EMITTERS[op_name] = fn
        return fn

    return wrap


def _slice(slot: str, access_map: dict, ctx: EmitCtx) -> str:
    """Emit the affine slice expression for one operand slot.

    Looks up the operand's fully-lowered :class:`Tensor` in
    ``ctx.tensors`` so :func:`emit_slice` can dispatch on the physical
    shape (HBM 2D vs SBUF/PSUM 3+D).
    """
    access = access_map[slot]
    tensor = ctx.tensors[access.tensor_name]
    return emit_slice(tensor, access, ctx)


@_register("NKIAlloc")
def _emit_alloc(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=<loc>)``.

    Shape comes from the post-``place_buffers`` tensor record; dtype
    and location come from the alloc kwargs. ``place_buffers`` leaves
    HBM tensors with their declared shape and promotes SBUF/PSUM
    tensors to N-D; we just render what we find.
    """
    _ = block
    tname = call.kwargs["tensor_name"]
    tensor = ctx.tensors[tname]
    dtype = call.kwargs["dtype"]
    location = call.kwargs["location"]
    dt_expr = _DTYPE_MAP.get(dtype, f"nl.{dtype}")
    loc_expr = _LOCATION_BUFFER_EXPR[location]
    shape = tensor.shape
    if len(shape) == 1:
        shape_tuple = f"({shape[0]},)"
    else:
        shape_tuple = "(" + ", ".join(str(x) for x in shape) + ")"
    return [f"{tname} = nl.ndarray({shape_tuple}, dtype={dt_expr}, buffer={loc_expr})"]


@_register("NKIMemset")
def _emit_memset(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.memset(<dst>, value=<value>)``."""
    dst = _slice("dst", block.writes, ctx)
    value = call.kwargs["value"]
    return [f"nisa.memset({dst}, value={value})"]


@_register("NKILoad")
def _emit_load(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.dma_copy(dst=<dst>, src=<src>)`` HBM -> SBUF."""
    _ = call
    src = _slice("src", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    return [f"nisa.dma_copy(dst={dst}, src={src})"]


@_register("NKIStore")
def _emit_store(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.dma_copy(dst=<dst>, src=<src>)`` SBUF -> HBM."""
    _ = call
    src = _slice("src", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    return [f"nisa.dma_copy(dst={dst}, src={src})"]


@_register("NKITensorCopy")
def _emit_tensor_copy(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.tensor_copy(<dst>, <src>)``."""
    _ = call
    src = _slice("src", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    return [f"nisa.tensor_copy({dst}, {src})"]


@_register("NKITensorReduce")
def _emit_tensor_reduce(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.tensor_reduce(<dst>, <op_expr>, <data>, axis=<axis>)``."""
    data = _slice("data", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    op = call.kwargs["op"]
    axis = call.kwargs["axis"]
    op_expr = "nl.add" if op == "add" else "nl.maximum"
    return [f"nisa.tensor_reduce({dst}, {op_expr}, {data}, axis={axis})"]


@_register("NKIMatmul")
def _emit_matmul(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit the multi-line ``nisa.nc_matmul(dst=..., stationary=..., moving=...)``."""
    _ = call
    stat = _slice("stationary", block.reads, ctx)
    mov = _slice("moving", block.reads, ctx)
    dst = _slice("dst", block.reads_writes, ctx)
    return ["nisa.nc_matmul(", f"    dst={dst},", f"    stationary={stat},", f"    moving={mov},", ")"]


@_register("NKIActivation")
def _emit_activation(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.activation(dst=..., op=nl.<act>, data=..., scale=..., bias=...)``."""
    data = _slice("data", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    act = call.kwargs["op"]
    scale = call.kwargs.get("scale", 1.0)
    bias = call.kwargs.get("bias", 0.0)
    return [f"nisa.activation(dst={dst}, op=nl.{act}, data={data}, scale={scale}, bias={bias})"]


@_register("NKIActivationReduce")
def _emit_activation_reduce(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit the multi-line ``nisa.activation_reduce(...)`` call."""
    data = _slice("data", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    reduce_res = _slice("reduce_res", block.writes, ctx)
    act = call.kwargs.get("op", "copy")
    reduce_op = call.kwargs.get("reduce_op", "add")
    reduce_op_expr = "nl.add" if reduce_op == "add" else "nl.maximum"
    return [
        "nisa.activation_reduce(",
        f"    dst={dst},",
        f"    op=nl.{act},",
        f"    data={data},",
        f"    reduce_op={reduce_op_expr},",
        f"    reduce_res={reduce_res},",
        ")",
    ]


@_register("NKITensorScalar")
def _emit_tensor_scalar(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.tensor_scalar(dst=..., data=..., op0=nl.<op>, operand0=...)``."""
    data = _slice("data", block.reads, ctx)
    op0 = _slice("operand0", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    op_name = call.kwargs["op"]
    return [f"nisa.tensor_scalar(dst={dst}, data={data}, op0=nl.{op_name}, operand0={op0})"]


@_register("NKITranspose")
def _emit_transpose(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.nc_transpose(<dst>, <src>)``."""
    _ = call
    src = _slice("src", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    return [f"nisa.nc_transpose({dst}, {src})"]


@_register("NKIDMATranspose")
def _emit_dma_transpose(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> list[str]:
    """Emit ``nisa.dma_transpose(<dst>, <src>)``."""
    _ = call
    src = _slice("src", block.reads, ctx)
    dst = _slice("dst", block.writes, ctx)
    return [f"nisa.dma_transpose({dst}, {src})"]
