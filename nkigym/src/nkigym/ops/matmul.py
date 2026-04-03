"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.

Sub-loop: if K exceeds MAX_K=128, render_isa() emits a sub-loop
that splits the contraction axis into 128-element chunks.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKIMatmul(NKIOp):
    """Matrix multiply: stationary.T @ moving -> output.

    Attributes:
        NAME: ``"nc_matmul"``.
        OPERAND_AXES: stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: output is ``(M, N)``.
        MAX_TILE_SIZES: K and M capped at 128, N capped at 512.
        ENGINE: TensorEngine.
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}
    ENGINE: ClassVar[str] = "TensorEngine"

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.nc_matmul with optional K sub-loop.

        Args:
            ctx: Render context.

        Returns:
            NKI source line(s) for nc_matmul.
        """
        dst = ctx.outputs["output"]
        stat = ctx.operands["stationary"]
        mov = ctx.operands["moving"]
        k_axis = stat.axes[0]
        k_size = stat.tile_size[k_axis]
        k_max = self.MAX_TILE_SIZES["K"]
        dst_slice = dst.default_indexed_slice()
        if k_size <= k_max:
            result = (
                f"nisa.nc_matmul(dst={dst_slice}, "
                f"stationary={stat.default_indexed_slice()}, "
                f"moving={mov.default_indexed_slice()})"
            )
        else:
            n_chunks = k_size // k_max
            lines = [f"for i_k_sub in range({n_chunks}):"]
            stat_slice = stat.default_indexed_slice()
            stat_slice = stat_slice.replace(f"0:{k_size}", f"i_k_sub*{k_max}:(i_k_sub+1)*{k_max}", 1)
            mov_slice = mov.default_indexed_slice()
            mov_slice = mov_slice.replace(f"0:{k_size}", f"i_k_sub*{k_max}:(i_k_sub+1)*{k_max}", 1)
            lines.append(f"    nisa.nc_matmul(dst={dst_slice}, " f"stationary={stat_slice}, " f"moving={mov_slice})")
            result = "\n".join(lines)
        return result
