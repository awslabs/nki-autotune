"""PE array transpose op: nisa.nc_transpose.

data(P, F) -> output(F, P). Swaps partition and free dims.

Sub-loop: if P or F exceeds 128, render() emits nested
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
    """

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    AXIS_ROLES: ClassVar[dict[str, str]] = {"P": "partition", "F": "free"}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128, "F": 128}

    def __call__(self, data: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: data.T."""
        return data.T

    def render(self, ctx: RenderContext) -> list[str]:
        """Emit nisa.nc_transpose with optional sub-loops.

        Partition overflow is computed from capping only (global tile
        size > 128), not from natural dimension blocking.

        Args:
            ctx: Render context.

        Returns:
            NKI source lines for nc_transpose.
        """
        dst = ctx.outputs["output"]
        src = ctx.operands["data"]
        p_axis = src.axes[0]
        f_axis = src.axes[1]
        p_size = src.tile_size[p_axis]
        f_size = src.tile_size[f_axis]
        p_max = self.MAX_TILE_SIZES["P"]
        f_max = self.MAX_TILE_SIZES["F"]
        p_global_ts = ctx.dim_global_tile_sizes.get(p_axis, p_size)
        p_overflow = p_global_ts // p_size if p_size < p_global_ts else 1

        needs_chunking = p_size > p_max or f_size > f_max or p_overflow > 1
        if needs_chunking:
            result = self._render_chunked(ctx, dst, src, p_axis, f_axis, p_size, f_size, p_max, f_max, p_overflow)
        else:
            result = [f"nisa.nc_transpose(dst={dst.default_indexed_slice()}, data={src.default_indexed_slice()})"]
        return result

    def _render_chunked(
        self,
        ctx: RenderContext,
        dst: Tensor,
        src: Tensor,
        p_axis: str,
        f_axis: str,
        p_size: int,
        f_size: int,
        p_max: int,
        f_max: int,
        p_overflow: int,
    ) -> list[str]:
        """Emit sub-loop transpose for chunks exceeding MAX_TILE_SIZES.

        Transpose swaps axes, so sub-loop indices map to swapped
        dest positions: P sub-blocks -> dest free chunks,
        F sub-chunks -> dest partition blocks.
        """
        p_subs = p_overflow if p_overflow > 1 else (p_size // p_max)
        f_subs = f_size // f_max
        use_nb_for_p = p_overflow > 1

        lines: list[str] = []
        indent = ""
        if p_subs > 1:
            lines.append(f"for i_p_sub in range({p_subs}):")
            indent += "    "
        if f_subs > 1:
            lines.append(f"{indent}for i_f_sub in range({f_subs}):")
            indent += "    "

        src_slice = self._src_slice(src, p_axis, p_size, f_size, p_max, f_max, p_subs, f_subs, use_nb_for_p)
        dst_slice = self._dst_slice(dst, f_axis, p_size, p_max, p_subs, f_subs, use_nb_for_p)
        lines.append(f"{indent}nisa.nc_transpose(dst={dst_slice}, data={src_slice})")
        return lines

    def _src_slice(
        self,
        src: Tensor,
        p_axis: str,
        p_size: int,
        f_size: int,
        p_max: int,
        f_max: int,
        p_subs: int,
        f_subs: int,
        use_nb_for_p: bool,
    ) -> str:
        """Build source slice expression for chunked transpose."""
        if use_nb_for_p and p_subs > 1:
            src_nb = dict(src.default_nb)
            src_nb[p_axis] = "i_p_sub"
            result = src.indexed_slice(src_nb, src.default_tpb)
        else:
            result = src.default_indexed_slice()

        if not use_nb_for_p and p_subs > 1:
            result = result.replace(f"0:{p_size}", f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}", 1)
        if f_subs > 1:
            result = result.replace(f"0:{f_size}", f"i_f_sub*{f_max}:(i_f_sub+1)*{f_max}", 1)
        return result

    def _dst_slice(
        self, dst: Tensor, f_axis: str, p_size: int, p_max: int, p_subs: int, f_subs: int, use_nb_for_p: bool
    ) -> str:
        """Build dest slice, accounting for P/F axis swap."""
        if f_subs > 1:
            result = dst.indexed_slice({f_axis: "i_f_sub"}, {})
        else:
            result = dst.default_indexed_slice()

        if p_subs > 1:
            p_logical = p_size * p_subs if use_nb_for_p else p_size
            result = result.replace(f"0:{p_logical}", f"i_p_sub*{p_max}:(i_p_sub+1)*{p_max}", 1)
        return result
