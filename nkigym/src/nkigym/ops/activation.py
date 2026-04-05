"""Element-wise unary activation op: nisa.activation.

output = op(data * scale + bias). Works on both 2D (P, F) and 1D (P,) tiles.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKIActivation(NKIOp):
    """Activation: element-wise unary function.

    Attributes:
        NAME: ``"activation"``.
        OPERAND_AXES: data is ``(P, F)``, bias is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``.
        MAX_TILE_SIZES: P capped at 128.
    """

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    AXIS_ROLES: ClassVar[dict[str, str]] = {"P": "partition", "F": "free"}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}

    _ACTIVATION_FNS: ClassVar[dict[str, object]] = {
        "exp": np.exp,
        "tanh": np.tanh,
        "square": np.square,
        "reciprocal": lambda x: 1.0 / x,
        "rsqrt": lambda x: 1.0 / np.sqrt(x),
    }

    def __call__(
        self, op: str, data: np.ndarray, bias: np.ndarray | None = None, scale: float = 1.0, **_: object
    ) -> np.ndarray:
        """CPU simulation: op(data * scale + bias).

        Args:
            op: Activation function name.
            data: Array of shape (P, F) or (P,).
            bias: Optional column vector of shape (P,).
            scale: Optional scale factor (scalar or (P,) array).

        Returns:
            Activated array.
        """
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        return self._ACTIVATION_FNS[op](data * s + b)

    def render(self, ctx: RenderContext) -> list[str]:
        """Emit nisa.activation call.

        Args:
            ctx: Render context.

        Returns:
            NKI source lines for activation.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
        parts = [f"dst={dst.default_indexed_slice()}", f"data={data.default_indexed_slice()}", f"op=nl.{op_name}"]
        bias = ctx.operands.get("bias")
        if bias is not None:
            parts.append(f"bias={bias.default_indexed_slice()}")
        scale = ctx.config_kwargs.get("scale")
        if scale is not None and scale != 1.0:
            parts.append(f"scale={scale}")
        return [f"nisa.activation({', '.join(parts)})"]
