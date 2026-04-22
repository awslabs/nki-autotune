"""Activation-reduce op: nisa.activation_reduce.

op(data * scale + bias) -> output(P, F), and simultaneously
reduce_op(output) along free axis -> reduce_res(P,).
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.activation import apply_unary_mask, parse_scale_literal
from nkigym.ops.base import NKIOp

_ACT_FNS: dict[str, Any] = {"exp": np.exp, "tanh": np.tanh, "square": np.square, "reciprocal": lambda x: 1.0 / x}
_RED_FNS: dict[str, Any] = {"add": np.sum}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIActivationReduce(NKIOp):
    """Fused activation + reduction.

    Applies an element-wise activation and simultaneously reduces
    the activated output along the free axis.  Returns both the
    full activated tensor and the reduction result.

    Attributes:
        NAME: ``"activation_reduce"``.
        OPERAND_AXES: data is ``(P, F)``, bias is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``, reduce_res is ``(P,)``.
    """

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F"), "reduce_res": ("P",)}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf", "bias": "sbuf"}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset({"scale"})
    REDUCE_COMBINATOR: ClassVar[dict[str, str]] = {"reduce_res": "reduce_op"}

    def __call__(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """CPU simulation: activation then reduction.

        Kwargs:
            data: Array of shape (P, F).
            op: Activation function name.
            reduce_op: Reduction operation (``"add"``).
            bias: Optional (P,) bias vector.
            scale: Scalar or (P,) scale vector.

        Returns:
            Tuple of (activated output (P, F), reduction result (P,)).
        """
        data: np.ndarray = kwargs["data"]
        op: str = kwargs["op"]
        reduce_op: str = kwargs["reduce_op"]
        bias: np.ndarray | None = kwargs.get("bias")
        scale = kwargs.get("scale", 1.0)
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        elem = _ACT_FNS[op](data * s + b)
        red = _RED_FNS[reduce_op](elem, axis=1)
        return elem, red

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.activation_reduce(dst, op, data, reduce_op, reduce_res, ...)."""
        sk = scalar_kwargs or {}
        op_arg = cls._to_nl(sk.get("op", "nl.copy"))
        reduce_op = cls._to_nl(sk.get("reduce_op", "nl.add"))
        reduce_res = sk.get("__dst_reduce_res", "None")
        bias_part = f", bias={operand_exprs['bias']}" if "bias" in operand_exprs else ""
        extra = cls._format_scalar_kwargs(sk, set(cls.OPERAND_AXES) | {"op", "reduce_op"})
        return (
            f"nisa.activation_reduce({dst_expr}, {op_arg}, {operand_exprs['data']}, "
            f"{reduce_op}, {reduce_res}{bias_part}{extra})"
        )

    @classmethod
    def propagate_mask_value(cls, op_kwargs: dict[str, str], input_value: float) -> float | None:
        """Apply ``op(input * scale + bias)`` element-wise (reducer semantics checked separately).

        ``bias`` is typically a per-partition tensor of finite
        values. For saturating inputs like ``±inf``, adding a
        finite bias can't change the result — we can safely
        propagate. For finite inputs with unknown bias we return
        ``None`` (not analyzable).
        """
        result: float | None = None
        scale = parse_scale_literal(op_kwargs.get("scale", "1.0"))
        if scale is not None:
            scaled = input_value * scale
            if "bias" in op_kwargs and not (scaled == float("inf") or scaled == float("-inf")):
                result = None
            else:
                raw_op = op_kwargs.get("op", "'copy'")
                op_name = raw_op[1:-1] if raw_op.startswith("'") and raw_op.endswith("'") else raw_op
                result = apply_unary_mask(op_name, scaled)
        return result
