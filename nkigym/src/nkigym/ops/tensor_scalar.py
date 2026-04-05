"""Element-wise tile-scalar op: nisa.tensor_scalar.

data(P, F) <op0> operand0 -> output(P, F).
operand0 can be a scalar constant or a (P,) column vector;
it broadcasts across the free axis.
Supports compound operations: result <op1> operand1.
Supports reverse mode: operand <op> data instead of data <op> operand.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKITensorScalar(NKIOp):
    """Tensor-scalar: element-wise op with broadcast.

    Attributes:
        NAME: ``"tensor_scalar"``.
        OPERAND_AXES: data is ``(P, F)``, operand0 is ``(P,)``, operand1 is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``.
        MAX_TILE_SIZES: P capped at 128.
    """

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "operand0": ("P",), "operand1": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    AXIS_ROLES: ClassVar[dict[str, str]] = {"P": "partition", "F": "free"}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}

    _BINARY_OPS: ClassVar[dict[str, object]] = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}

    def __call__(
        self,
        data: np.ndarray,
        op0: str,
        operand0: object,
        reverse0: bool = False,
        op1: str | None = None,
        operand1: object = None,
        reverse1: bool = False,
        **_: object,
    ) -> np.ndarray:
        """CPU simulation: data <op0> operand0, with optional compound op1.

        Args:
            data: Array of shape (P, F).
            op0: Primary binary operation name (multiply, subtract, add).
            operand0: Scalar or column vector of shape (P,).
            reverse0: If True, compute operand0 <op0> data.
            op1: Optional second binary operation name.
            operand1: Optional second operand (scalar or column vector).
            reverse1: If True, compute operand1 <op1> result.

        Returns:
            Result array with same shape as data.
        """
        b = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        result = self._BINARY_OPS[op0](b, data) if reverse0 else self._BINARY_OPS[op0](data, b)
        if op1 is not None:
            c = operand1[..., np.newaxis] if isinstance(operand1, np.ndarray) else operand1
            result = self._BINARY_OPS[op1](c, result) if reverse1 else self._BINARY_OPS[op1](result, c)
        return result

    def render(self, ctx: RenderContext) -> list[str]:
        """Emit nisa.tensor_scalar call.

        Args:
            ctx: Render context.

        Returns:
            NKI source lines for tensor_scalar.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op0"]
        op0 = ctx.operands.get("operand0")
        if op0 is not None:
            op0_expr = op0.default_indexed_slice()
        else:
            op0_expr = str(ctx.config_kwargs["operand0"])
        parts = [
            f"dst={dst.default_indexed_slice()}",
            f"data={data.default_indexed_slice()}",
            f"op0=nl.{op_name}",
            f"operand0={op0_expr}",
        ]
        reverse0 = ctx.config_kwargs.get("reverse0", False)
        if reverse0:
            parts.append("reverse0=True")
        op1_name = ctx.config_kwargs.get("op1")
        if op1_name is not None:
            parts.append(f"op1=nl.{op1_name}")
            op1_operand = ctx.operands.get("operand1")
            if op1_operand is not None:
                parts.append(f"operand1={op1_operand.default_indexed_slice()}")
            else:
                parts.append(f"operand1={ctx.config_kwargs['operand1']}")
            reverse1 = ctx.config_kwargs.get("reverse1", False)
            if reverse1:
                parts.append("reverse1=True")
        return [f"nisa.tensor_scalar({', '.join(parts)})"]
