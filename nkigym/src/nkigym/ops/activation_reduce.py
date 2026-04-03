"""Compound activation+reduce op: nisa.activation_reduce.

op(data + bias) -> output(P, F), simultaneously
reduce_op(output) along free axis -> reduce_res(P,).
Dual-output: produces both activation and reduction results.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKIActivationReduce(NKIOp):
    """Activation+reduce: dual-output activation with simultaneous reduction.

    Attributes:
        NAME: ``"activation_reduce"``.
        OPERAND_AXES: data is ``(P, F)``, bias is ``(P,)``.
        OUTPUT_AXES: output is ``(P, F)``, reduce_res is ``(P,)``.
        MAX_TILE_SIZES: P capped at 128.
        ENGINE: ScalarEngine.
    """

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F"), "reduce_res": ("P",)}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    ENGINE: ClassVar[str] = "ScalarEngine"
    SCHEDULE_OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)

    _NKI_REDUCE_OPS: ClassVar[dict[str, str]] = {"max": "maximum", "add": "add"}
    _ACTIVATION_FNS: ClassVar[dict[str, object]] = {
        "exp": np.exp,
        "tanh": np.tanh,
        "square": np.square,
        "reciprocal": lambda x: 1.0 / x,
    }
    _REDUCE_FNS: ClassVar[dict[str, object]] = {"max": np.max, "add": np.sum}

    def __call__(
        self, data: np.ndarray, bias: np.ndarray, op: str, reduce_op: str, **_: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU simulation: activation with simultaneous reduction.

        Args:
            data: Array of shape (P, F).
            bias: Column vector of shape (P,).
            op: Activation function name.
            reduce_op: Reduction operation name.

        Returns:
            Tuple of (activation output, reduction result).
        """
        elem_result = self._ACTIVATION_FNS[op](data + bias[..., np.newaxis])
        reduce_result = self._REDUCE_FNS[reduce_op](elem_result, axis=1)
        return elem_result, reduce_result

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.activation_reduce call.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for activation_reduce.
        """
        dst = ctx.outputs["output"]
        red = ctx.outputs["reduce_res"]
        data = ctx.operands["data"]
        bias = ctx.operands["bias"]
        op_name = ctx.config_kwargs["op"]
        reduce_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["reduce_op"]]
        return (
            f"nisa.activation_reduce("
            f"dst={dst.default_indexed_slice()}, "
            f"data={data.default_indexed_slice()}, "
            f"op=nl.{op_name}, "
            f"bias={bias.default_indexed_slice()}, "
            f"reduce_op=nl.{reduce_name}, "
            f"reduce_res={red.default_indexed_slice()})"
        )
