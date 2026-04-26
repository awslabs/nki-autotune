"""Fused activation + free-axis reduce op: ``nisa.activation_reduce`` + ``activation_reduce_block`` gadget.

Mirrors ``nisa.activation_reduce`` one-to-one: applies ``op(data * scale + bias)``
element-wise and simultaneously reduces the activated result along the free
dimension into a per-row scalar. ``post_op`` (optional) applies a final
activation to the closed reduction result once all F tiles have been summed —
covers the ``post_op='rsqrt'`` case used by rmsnorm.

OPERAND_AXES / OUTPUT_AXES reflect that only the reduce result is the
op's *output* — the activated tile is internally consumed and discarded.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512

_ACT_FNS: dict[str, Any] = {
    "square": np.square,
    "exp": np.exp,
    "copy": lambda x: x,
    "reciprocal": lambda x: 1.0 / x,
    "tanh": np.tanh,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "sqrt": np.sqrt,
}
_RED_FNS: dict[str, Any] = {"add": np.sum}

_NL_OPS: dict[str, Any] = {
    "square": nl.square,
    "exp": nl.exp,
    "copy": nl.copy,
    "reciprocal": nl.reciprocal,
    "tanh": nl.tanh,
    "rsqrt": nl.rsqrt,
    "sqrt": nl.sqrt,
}
_NL_REDUCE_OPS: dict[str, Any] = {"add": nl.add}


class NKIActivationReduce(NKIOp):
    """Fused activation + free-axis reduce, with optional post-op on the closed reduction.

    Math: ``reduce_op(op(data * scale + bias), axis=F)`` with an optional
    ``post_op`` applied once after the F reduction closes — the typical
    rmsnorm shape is ``post_op(reduce_op(op(data)*scale+bias, axis=F))``.

    The op's output is the ``(P,)`` per-row reduction vector. The fully
    activated ``(P, F)`` tile is an internal byproduct the gadget
    discards — downstream consumers either use the reduction vector
    directly (rmsnorm → ``tensor_scalar(multiply, operand0=rms_inv)``)
    or fuse it into a subsequent elementwise op.
    """

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P",)}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"output": "float32"}
    """Reduction accumulator must stay fp32 — narrowing to bf16 across a
    2048-element sum loses the low-bit accumulation precision and
    propagates into the matmul accumulator. Matches the nisa.activation_reduce
    HW contract."""
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    """The F axis is a reduction axis — the op iterates over all F tiles
    before its output is complete. Render emits an F-loop memset prologue
    (on the reduce accumulator) and places downstream consumers outside
    this F-loop, symmetric to how matmul's K dim is handled."""
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: ``post_op(reduce_op(op(data), axis=F) * scale + bias)``.

        ``scale`` and ``bias`` apply to the **closed reduction result**
        before the optional ``post_op`` — matches the rmsnorm formula
        ``rsqrt(sum(lhs^2) * (1/K) + eps)``. The codegen emits the
        activation+reduce as one fused call (no scale/bias on the
        reducer), then a separate ``activation_block(post_op, scale, bias)``
        on the closed reduction, which is semantically equivalent.
        """
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs.get("op", "copy")
        reduce_op: str = kwargs.get("reduce_op", "add")
        post_op_name: str | None = kwargs.get("post_op")
        scale = kwargs.get("scale", 1.0)
        bias = kwargs.get("bias", 0.0)
        data_f32 = data.astype(np.float32)
        activated = _ACT_FNS[op_name](data_f32)
        reduced = _RED_FNS[reduce_op](activated, axis=1).astype(np.float32)
        scaled = reduced * scale + bias
        if post_op_name is not None:
            scaled = _ACT_FNS[post_op_name](scaled)
        return scaled


def activation_reduce_block(sbuf_red: Any, sbuf_data: Any, op: Any, reduce_op: Any) -> None:
    """Apply ``op`` + free-axis ``reduce_op`` per leaf, accumulating into ``sbuf_red``.

    ``sbuf_data`` leaves shape ``(p_tile, f_tile)``. ``sbuf_red`` leaves
    shape ``(p_tile, 1)``. Each reduce is added into the accumulator;
    caller must memset ``sbuf_red`` before the first call in an F-block.

    The scale+bias+post_op kwargs of :class:`NKIActivationReduce` are
    emitted by the renderer — this gadget handles only the raw
    activation-reduce step. For rmsnorm's ``post_op='rsqrt'``, the
    renderer emits a separate ``activation_block`` call on the closed
    reduction once the F loop exits.
    """
    p_tile, f_tile = sbuf_data[0].shape
    for pi in range(len(sbuf_data)):
        tmp = nl.ndarray((p_tile, f_tile), dtype=nl.float32, buffer=nl.sbuf)
        tmp_red = nl.ndarray((p_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=tmp[0:p_tile, 0:f_tile],
            op=op,
            data=sbuf_data[pi][0:p_tile, 0:f_tile],
            reduce_op=reduce_op,
            reduce_res=tmp_red[0:p_tile, 0:1],
        )
        nisa.tensor_tensor(sbuf_red[pi][0:p_tile, 0:1], sbuf_red[pi][0:p_tile, 0:1], tmp_red[0:p_tile, 0:1], op=nl.add)


def activation_block(sbuf_dst: Any, sbuf_src: Any, op: Any, scale: float = 1.0, bias: float = 0.0) -> None:
    """Apply a standalone activation ``op(data * scale + bias)`` per leaf.

    Used by the renderer to emit the ``post_op`` of :class:`NKIActivationReduce`
    after the F reduction has closed — e.g. ``activation_block(rms_inv,
    m_sum, op=nl.rsqrt, scale=1/K, bias=eps)`` to turn ``sum(lhs^2)``
    into ``1/sqrt(mean + eps)``.

    ``sbuf_dst`` and ``sbuf_src`` must have matching shapes. For the
    rmsnorm closure they are both ``(p_tile, 1)`` 1D-style leaves.
    """
    p_tile, f_tile = sbuf_dst[0].shape
    for i in range(len(sbuf_dst)):
        nisa.activation(
            dst=sbuf_dst[i][0:p_tile, 0:f_tile], op=op, data=sbuf_src[i][0:p_tile, 0:f_tile], scale=scale, bias=bias
        )


def nl_op(name: str) -> Any:
    """Resolve an op-name string to its ``nl.*`` callable."""
    return _NL_OPS[name]


def nl_reduce_op(name: str) -> Any:
    """Resolve a reduce-op-name string to its ``nl.*`` callable."""
    return _NL_REDUCE_OPS[name]
