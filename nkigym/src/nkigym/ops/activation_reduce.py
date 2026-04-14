"""Activation-reduce op: nisa.activation_reduce.

op(data * scale + bias) -> output(P, F), and simultaneously
reduce_op(output) along free axis -> reduce_res(P,).
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

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

    def __call__(
        self,
        data: np.ndarray,
        op: str,
        reduce_op: str,
        bias: np.ndarray | None = None,
        scale: np.ndarray | float = 1.0,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU simulation: activation then reduction.

        Args:
            data: Array of shape (P, F).
            op: Activation function name.
            reduce_op: Reduction operation (``"add"``).
            bias: Optional (P,) bias vector.
            scale: Scalar or (P,) scale vector.

        Returns:
            Tuple of (activated output (P, F), reduction result (P,)).
        """
        fns = {"exp": np.exp, "tanh": np.tanh, "square": np.square, "reciprocal": lambda x: 1.0 / x}
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        elem = fns[op](data * s + b)
        red = {"add": np.sum}[reduce_op](elem, axis=1)
        return elem, red

    @classmethod
    def format_isa_call(cls, dst_expr: str, operand_exprs: dict[str, str]) -> str:
        """Format nisa.activation_reduce(dst, data, ...)."""
        return f"nisa.activation_reduce({dst_expr}," f" {operand_exprs['data']}, ...)"
