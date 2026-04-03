"""Element-wise unary activation op: nisa.activation.

output = op(data). Works on both 2D (P, F) and 1D (P,) tiles.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKIActivation(NKIOp):
    """Activation: element-wise unary function.

    Attributes:
        NAME: ``"activation"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(P, F)``.
        MAX_TILE_SIZES: P capped at 128.
        ENGINE: ScalarEngine.
    """

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    ENGINE: ClassVar[str] = "ScalarEngine"

    _ACTIVATION_FNS: ClassVar[dict[str, object]] = {
        "exp": np.exp,
        "tanh": np.tanh,
        "square": np.square,
        "reciprocal": lambda x: 1.0 / x,
        "rsqrt": lambda x: 1.0 / np.sqrt(x),
    }

    def __call__(self, data: np.ndarray, op: str, **_: object) -> np.ndarray:
        """CPU simulation: op(data).

        Args:
            data: Array of shape (P, F) or (P,).
            op: Activation function name.

        Returns:
            Activated array.
        """
        return self._ACTIVATION_FNS[op](data)

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.activation call.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for activation.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
        return (
            f"nisa.activation(dst={dst.default_indexed_slice()}, "
            f"data={data.default_indexed_slice()}, "
            f"op=nl.{op_name})"
        )
