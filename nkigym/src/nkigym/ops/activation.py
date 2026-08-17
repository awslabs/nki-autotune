"""Standalone ``nisa.activation`` operation.

Applies ``op(data * scale + bias)`` elementwise. Unlike
:class:`NKIActivationReduce` this op does not reduce the free axis —
its output matches the input shape. Used for 1D per-row math such as
``rsqrt(m_state/K + eps)`` and ``reciprocal(rms_old)`` in the
online-fused rmsnorm+matmul kernel.
"""

from collections.abc import Mapping
from math import erf, sqrt
from numbers import Real
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

_ACT_FNS: dict[str, Any] = {
    "abs": np.abs,
    "square": np.square,
    "exp": np.exp,
    "copy": lambda x: x,
    "reciprocal": lambda x: 1.0 / x,
    "tanh": np.tanh,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "sqrt": np.sqrt,
    "erf": lambda x: np.frompyfunc(erf, 1, 1)(x).astype(np.float32),
    "gelu": lambda x: 0.5 * x * (1.0 + np.frompyfunc(erf, 1, 1)(x / sqrt(2.0)).astype(np.float32)),
    "log": np.log,
    "gelu_apprx_tanh": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))),
    "gelu_apprx_sigmoid": lambda x: x / (1.0 + np.exp(-1.702 * x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "sign": np.sign,
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
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "bias"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data": frozenset({"sbuf", "psum"}),
        "bias": frozenset({"sbuf"}),
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured unary pointwise operation."""
        return PointwiseContract(
            operator=str(kwargs["op"]),
            input_operands=("data",),
            output_operand="dst",
            scale=float(kwargs.get("scale", 1.0)),
            bias=float(kwargs["bias"]) if isinstance(kwargs.get("bias"), Real) else 0.0,
            bias_operand="bias",
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be on-chip and a tensor bias must reside in SBUF."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIActivation(data=<role={role}>) expects sbuf or psum")
        bias_role = _operand_role(kwargs.get("bias"))
        if bias_role is not None and bias_role != "sbuf":
            raise TypeError(f"NKIActivation(bias=<role={bias_role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``op(data * scale + bias)``."""
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        scale = kwargs.get("scale", 1.0)
        bias = kwargs.get("bias", 0.0)
        if isinstance(bias, np.ndarray) and data.ndim == bias.ndim + 1:
            bias = bias[..., np.newaxis]
        return _ACT_FNS[op_name](data.astype(np.float32) * scale + bias)
