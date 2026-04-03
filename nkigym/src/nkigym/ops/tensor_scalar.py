"""Element-wise tile-scalar op: nisa.tensor_scalar.

data(P, F) <op0> operand0 -> output(P, F).
operand0 can be a scalar constant or a (P,) column vector;
it broadcasts across the free axis.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKITensorScalar(NKIOp):
    """Tensor-scalar: element-wise op with broadcast.

    Attributes:
        NAME: ``"tensor_scalar"``.
        OPERAND_AXES: data is ``(P, F)``, operand0 is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``.
        MAX_TILE_SIZES: P capped at 128.
        ENGINE: VectorEngine.
    """

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "operand0": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    ENGINE: ClassVar[str] = "VectorEngine"

    _BINARY_OPS: ClassVar[dict[str, object]] = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}

    def __call__(self, data: np.ndarray, operand0: object, op0: str, **_: object) -> np.ndarray:
        """CPU simulation: data <op0> operand0.

        Args:
            data: Array of shape (P, F).
            operand0: Scalar or column vector of shape (P,).
            op0: Binary operation name (multiply, subtract, add).

        Returns:
            Result array with same shape as data.
        """
        broadcast_op0 = operand0
        if isinstance(operand0, np.ndarray):
            broadcast_op0 = operand0[..., np.newaxis]
        return self._BINARY_OPS[op0](data, broadcast_op0)

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.tensor_scalar call.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for tensor_scalar.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op0"]
        op0 = ctx.operands.get("operand0")
        if op0 is not None:
            op0_expr = op0.default_indexed_slice()
        else:
            op0_expr = str(ctx.config_kwargs["operand0"])
        return (
            f"nisa.tensor_scalar(dst={dst.default_indexed_slice()}, "
            f"data={data.default_indexed_slice()}, "
            f"op0=nl.{op_name}, operand0={op0_expr})"
        )
