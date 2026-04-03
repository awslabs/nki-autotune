"""PE array transpose op: nisa.nc_transpose.

data(P, F) -> output(F, P). Swaps partition and free dims.

Sub-loop: if P or F exceeds 128, render_isa() emits nested
sub-loops that transpose (128, 128) chunks.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext, Tensor
from nkigym.ops.base import NKIOp


class NKITranspose(NKIOp):
    """Transpose: swap partition and free dims.

    Attributes:
        NAME: ``"nc_transpose"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(F, P)`` -- swapped.
        MAX_TILE_SIZES: P and F capped at 128.
        ENGINE: TensorEngine.
    """

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    ENGINE: ClassVar[str] = "TensorEngine"

    def __call__(self, data: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: data.T.

        Args:
            data: Array of shape (P, F).

        Returns:
            Transposed array.
        """
        return data.T

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.nc_transpose with optional sub-loops.

        Args:
            ctx: Render context.

        Returns:
            NKI source line(s) for nc_transpose.
        """
        dst = ctx.outputs["output"]
        src = ctx.operands["data"]
        p_axis = src.axes[0]
        f_axis = src.axes[1]
        p_size = src.tile_size[p_axis]
        f_size = src.tile_size[f_axis]
        p_max = self.MAX_TILE_SIZES["P"]
        f_max = self.MAX_TILE_SIZES["F"]

        if p_size <= p_max and f_size <= f_max:
            result = f"nisa.nc_transpose(" f"dst={dst.default_indexed_slice()}, " f"src={src.default_indexed_slice()})"
        else:
            result = self._render_chunked(dst, src, p_size, f_size, p_max, f_max)
        return result

    def _render_chunked(self, dst: Tensor, src: Tensor, p_size: int, f_size: int, p_max: int, f_max: int) -> str:
        """Emit sub-loop transpose for chunks exceeding MAX_TILE_SIZES.

        Args:
            dst: Destination tensor.
            src: Source tensor.
            p_size: Partition axis tile size.
            f_size: Free axis tile size.
            p_max: Max partition tile size.
            f_max: Max free tile size.

        Returns:
            NKI source lines with sub-loops.
        """
        p_subs = p_size // p_max
        f_subs = f_size // f_max
        lines: list[str] = []
        indent = ""
        if p_subs > 1:
            lines.append(f"for i_p_sub in range({p_subs}):")
            indent += "    "
        if f_subs > 1:
            lines.append(f"{indent}for i_f_sub in range({f_subs}):")
            indent += "    "
        src_slice = src.default_indexed_slice()
        dst_slice = dst.default_indexed_slice()
        if p_subs > 1:
            src_slice = src_slice.replace(f"0:{p_size}", f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}", 1)
            dst_slice = dst_slice.replace(f"0:{p_size}", f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}", 1)
        if f_subs > 1:
            src_slice = src_slice.replace(f"0:{f_size}", f"i_f_sub*{f_max}:(i_f_sub+1)*{f_max}", 1)
            dst_slice = dst_slice.replace(f"0:{f_size}", f"i_f_sub*{f_max}:(i_f_sub+1)*{f_max}", 1)
        lines.append(f"{indent}nisa.nc_transpose(dst={dst_slice}, src={src_slice})")
        return "\n".join(lines)
