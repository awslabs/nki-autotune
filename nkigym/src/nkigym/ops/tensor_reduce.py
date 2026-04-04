"""Free-axis reduction op: nisa.tensor_reduce.

data(P, F) -> output(P,). Collapses free dimension with the
specified reduction op. Optional negate flag.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKITensorReduce(NKIOp):
    """Tensor reduce: collapse free axis.

    Attributes:
        NAME: ``"tensor_reduce"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(P,)``.
        MAX_TILE_SIZES: P capped at 128.
    """

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P",)}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}

    _NKI_REDUCE_OPS: ClassVar[dict[str, str]] = {"max": "maximum", "add": "add"}
    _REDUCE_FNS: ClassVar[dict[str, object]] = {"max": np.max, "add": np.sum}

    def __call__(
        self, op: str, data: np.ndarray, axis: int, negate: bool = False, keepdims: bool = False, **_: object
    ) -> np.ndarray:
        """CPU simulation: reduce along specified axis.

        Args:
            op: Reduction operation (max, add).
            data: Array of shape (P, F).
            axis: Axis to reduce along.
            negate: Whether to negate the result.
            keepdims: Whether to keep the reduced dimension.

        Returns:
            Reduced array.
        """
        result = self._REDUCE_FNS[op](data, axis=axis, keepdims=keepdims)
        if negate:
            result = -result
        return result

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.tensor_reduce call.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for tensor_reduce.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        op_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["op"]]
        negate = ctx.config_kwargs.get("negate", False)
        negate_str = ", negate=True" if negate else ""
        return (
            f"nisa.tensor_reduce(dst={dst.default_indexed_slice()}, "
            f"data={data.default_indexed_slice()}, "
            f"op=nl.{op_name}, axis=1{negate_str})"
        )
