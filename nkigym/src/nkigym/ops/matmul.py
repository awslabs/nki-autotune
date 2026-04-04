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
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}
    ACC_FREE_DIM_LIMIT: ClassVar[int] = 512

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

        K sub-loop handles two cases:
        - Tile-size overflow: K tile_size > 128, chunk the partition dim.
        - Capping overflow: K global tile > capped tile, iterate num_blocks.

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
        k_global_ts = ctx.dim_global_tile_sizes.get(k_axis, k_size)
        k_overflow = k_global_ts // k_size if k_size < k_global_ts else 1
        dst_slice = dst.default_indexed_slice()
        if k_size <= k_max and k_overflow <= 1:
            result = (
                f"nisa.nc_matmul(dst={dst_slice}, "
                f"stationary={stat.default_indexed_slice()}, "
                f"moving={mov.default_indexed_slice()})"
            )
        elif k_overflow > 1:
            result = self._render_nb_sub_loop(dst_slice, stat, mov, k_axis, k_overflow)
        else:
            result = self._render_tile_sub_loop(dst_slice, stat, mov, k_size, k_max)
        return result

    def _render_nb_sub_loop(self, dst_slice: str, stat: "Tensor", mov: "Tensor", k_axis: str, k_overflow: int) -> str:
        """Emit K sub-loop iterating over num_blocks (capping overflow).

        Args:
            dst_slice: Destination slice expression.
            stat: Stationary operand tensor.
            mov: Moving operand tensor.
            k_axis: K axis dim ID.
            k_overflow: Number of capping overflow blocks.

        Returns:
            NKI source lines with num_blocks sub-loop.
        """
        lines = [f"for i_k_sub in range({k_overflow}):"]
        stat_nb = dict(stat.default_nb)
        stat_nb[k_axis] = "i_k_sub"
        stat_slice = stat.indexed_slice(stat_nb, stat.default_tpb)
        mov_nb = dict(mov.default_nb)
        mov_nb[k_axis] = "i_k_sub"
        mov_slice = mov.indexed_slice(mov_nb, mov.default_tpb)
        lines.append(f"    nisa.nc_matmul(dst={dst_slice}, stationary={stat_slice}, moving={mov_slice})")
        return "\n".join(lines)

    def _render_tile_sub_loop(self, dst_slice: str, stat: "Tensor", mov: "Tensor", k_size: int, k_max: int) -> str:
        """Emit K sub-loop chunking partition dim (tile overflow).

        Args:
            dst_slice: Destination slice expression.
            stat: Stationary operand tensor.
            mov: Moving operand tensor.
            k_size: K axis tile size.
            k_max: Max K tile size (128).

        Returns:
            NKI source lines with partition-chunk sub-loop.
        """
        n_chunks = k_size // k_max
        lines = [f"for i_k_sub in range({n_chunks}):"]
        stat_slice = stat.default_indexed_slice()
        stat_slice = stat_slice.replace(f"0:{k_size}", f"i_k_sub*{k_max}:(i_k_sub+1)*{k_max}", 1)
        mov_slice = mov.default_indexed_slice()
        mov_slice = mov_slice.replace(f"0:{k_size}", f"i_k_sub*{k_max}:(i_k_sub+1)*{k_max}", 1)
        lines.append(f"    nisa.nc_matmul(dst={dst_slice}, stationary={stat_slice}, moving={mov_slice})")
        return "\n".join(lines)
