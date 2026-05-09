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

from nkigym.ops.base import NKIOp

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

    Output matches the input dtype by default; pin to fp32 via
    ``OUTPUT_DTYPES`` at the use site if precision matters (the online
    rmsnorm kernel hand-builds rsqrt / reciprocal buffers as fp32).
    """

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128, "F": 512}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: ``op(data * scale + bias)``."""
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        scale = kwargs.get("scale", 1.0)
        bias = kwargs.get("bias", 0.0)
        return _ACT_FNS[op_name](data.astype(np.float32) * scale + bias).astype(np.float32)
