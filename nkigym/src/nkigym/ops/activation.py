"""Standalone activation op: ``nisa.activation`` + ``activation_block`` gadget.

Applies ``op(data * scale + bias)`` elementwise. Unlike
:class:`NKIActivationReduce` this op does not reduce the free axis —
its output matches the input shape. Used for 1D per-row math such as
``rsqrt(m_state/K + eps)`` and ``reciprocal(rms_old)`` in the
online-fused rmsnorm+matmul kernel.

The underlying gadget ``activation_block`` is shared with
:mod:`nkigym.ops.activation_reduce` — it's the same call the reducer's
``post_op`` phase emits after the F loop closes.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role

_ACT_FNS: dict[str, Any] = {
    "square": np.square,
    "exp": np.exp,
    "copy": lambda x: x,
    "reciprocal": lambda x: 1.0 / x,
    "tanh": np.tanh,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "sqrt": np.sqrt,
}


class NKIActivation(NKIOp):
    """Standalone activation: ``output = op(data * scale + bias)``.

    Declares both P and F axes so the op accepts 1D ``(P,)`` and 2D
    ``(P, F)`` inputs. The build pipeline's axis-unification layer
    zips ``OPERAND_AXES`` with the operand's concrete ``dim_ids``
    positionally — 1D operands simply skip the F axis slot.

    Output carries the input's logical dtype (propagated through the
    trace). A precision-sensitive use (e.g. the online rmsnorm kernel's
    rsqrt / reciprocal buffers) that needs fp32 is expressed by a
    separate fp32-producing op rather than a per-output dtype override.
    """

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be SBUF-resident."""
        role = _operand_role(kwargs["data"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIActivation(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``op(data * scale + bias)``."""
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        scale = kwargs.get("scale", 1.0)
        bias = kwargs.get("bias", 0.0)
        return _ACT_FNS[op_name](data.astype(np.float32) * scale + bias)
