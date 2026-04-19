"""Activation op: nisa.activation.

output = op(data * scale + bias).
Applies unary activation element-wise.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_ACT_FNS: dict[str, Any] = {
    "exp": np.exp,
    "tanh": np.tanh,
    "square": np.square,
    "reciprocal": lambda x: 1.0 / x,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIActivation(NKIOp):
    """Element-wise unary activation.

    Computes ``op(data * scale + bias)`` where ``op`` is one of
    exp, tanh, square, reciprocal, or rsqrt.

    Attributes:
        NAME: ``"activation"``.
        OPERAND_AXES: data is ``(P, F)``, bias is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``.
    """

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf", "bias": "sbuf"}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset({"scale"})

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: op(data * scale + bias).

        Kwargs:
            data: Array of shape (P, F) or (P,).
            op: Activation name.
            bias: Optional (P,) bias vector.
            scale: Scalar or (P,) scale vector.

        Returns:
            Activated array, same shape as data.
        """
        data: np.ndarray = kwargs["data"]
        op: str = kwargs["op"]
        bias: np.ndarray | None = kwargs.get("bias")
        scale = kwargs.get("scale", 1.0)
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        return _ACT_FNS[op](data * s + b)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.activation(dst, op, data, ...)."""
        sk = scalar_kwargs or {}
        op_arg = cls._to_nl(sk.get("op", "nl.copy"))
        bias_part = f", bias={operand_exprs['bias']}" if "bias" in operand_exprs else ""
        extra = cls._format_scalar_kwargs(sk, set(cls.OPERAND_AXES) | {"op"})
        return f"nisa.activation({dst_expr}, {op_arg}, {operand_exprs['data']}{bias_part}{extra})"
