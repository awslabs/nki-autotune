"""Compound activation+reduce op: nisa.activation_reduce.

op(data * scale + bias) -> output(P, F), simultaneously
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
    """

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "bias": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F"), "reduce_res": ("P",)}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    SCHEDULE_OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)

    _NKI_REDUCE_OPS: ClassVar[dict[str, str]] = {"add": "add"}
    _ACTIVATION_FNS: ClassVar[dict[str, object]] = {
        "exp": np.exp,
        "tanh": np.tanh,
        "square": np.square,
        "reciprocal": lambda x: 1.0 / x,
    }
    _REDUCE_FNS: ClassVar[dict[str, object]] = {"add": np.sum}

    def __call__(
        self, op: str, data: np.ndarray, reduce_op: str, bias: np.ndarray | None = None, scale: float = 1.0, **_: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU simulation: activation with simultaneous reduction.

        Args:
            op: Activation function name.
            data: Array of shape (P, F).
            reduce_op: Reduction operation name.
            bias: Optional column vector of shape (P,).
            scale: Optional scale factor (scalar or (P,) array).

        Returns:
            Tuple of (activation output, reduction result).
        """
        b = 0.0 if bias is None else bias[..., np.newaxis]
        s = scale[..., np.newaxis] if isinstance(scale, np.ndarray) else scale
        elem_result = self._ACTIVATION_FNS[op](data * s + b)
        reduce_result = self._REDUCE_FNS[reduce_op](elem_result, axis=1)
        return elem_result, reduce_result

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.activation_reduce call.

        Each tile gets its own PSUM slot (allocated by the renderer), so
        the per-call reset is harmless — each slot is written exactly once.
        A cross-block ``nisa.tensor_reduce`` combine is emitted by the
        renderer after the loop, following the same pattern as all other
        reductions.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for activation_reduce.
        """
        dst = ctx.outputs["output"]
        red = ctx.outputs["reduce_res"]
        data = ctx.operands["data"]
        op_name = ctx.config_kwargs["op"]
        reduce_name = self._NKI_REDUCE_OPS[ctx.config_kwargs["reduce_op"]]
        parts = [
            f"dst={dst.default_indexed_slice()}",
            f"op=nl.{op_name}",
            f"data={data.default_indexed_slice()}",
            f"reduce_op=nl.{reduce_name}",
            f"reduce_res={red.default_indexed_slice()}",
        ]
        bias = ctx.operands.get("bias")
        if bias is not None:
            parts.append(f"bias={bias.default_indexed_slice()}")
        scale = ctx.config_kwargs.get("scale")
        if scale is not None and scale != 1.0:
            parts.append(f"scale={scale}")
        return f"nisa.activation_reduce({', '.join(parts)})"
