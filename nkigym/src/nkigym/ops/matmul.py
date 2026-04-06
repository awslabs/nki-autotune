"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext


def _find_output_name(ctx: RenderContext, operand_map: dict[str, str], output_shape: tuple[int, ...]) -> str:
    """Find output tensor by elimination from operand names."""
    operand_names = set(operand_map.values())
    for name, t in ctx.tensors.items():
        if name not in operand_names and t.shape == output_shape:
            return name
    raise ValueError(f"No output tensor found with shape {output_shape}")


def _hbm_range(block_var: str, tile: int) -> str:
    """Format an HBM slice expression: ``var*tile:var*tile+tile``."""
    return f"{block_var}*{tile}:{block_var}*{tile}+{tile}"


class NKIMatmul(NKIOp):
    """Matrix multiply: stationary.T @ moving -> output.

    Attributes:
        NAME: ``"nc_matmul"``.
        OPERAND_AXES: stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: output is ``(M, N)``.
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    def _emit_k_reduction(
        self, ctx: RenderContext, operand_map: dict[str, str], output_name: str, indent: str
    ) -> list[str]:
        """Emit PSUM alloc, K loops, DMA loads, ISA, copy, and store."""
        stat = ctx.tensors[operand_map["stationary"]]
        mov = ctx.tensors[operand_map["moving"]]
        K, M = stat.shape
        N = mov.shape[1]
        tile_K, tile_M, tile_N = 128, 128, 512
        psum_name = f"psum_{output_name}"
        sbuf_stat = f"sbuf_{stat.name}"
        sbuf_mov = f"sbuf_{mov.name}"
        sbuf_out = f"sbuf_{output_name}"
        hbm_out = f"hbm_{output_name}"
        i1 = indent + "    "
        i2 = i1 + "    "
        ps = f"0:{tile_M}, 0, 0, 0, 0, 0:{tile_N}"
        si = f"0:{tile_M}, i_block_M, i_block_N, i_tile_M, i_tile_N, 0:{tile_N}"
        k_rng = _hbm_range("i_block_K", tile_K)
        m_rng = _hbm_range("i_block_M", tile_M)
        n_rng = _hbm_range("i_block_N", tile_N)
        stat_sl = f"0:{tile_K}, 0, 0, 0, 0, 0:{tile_M}"
        mov_sl = f"0:{tile_K}, 0, 0, 0, 0, 0:{tile_N}"
        return [
            f"{indent}{psum_name} = nl.ndarray(({tile_M}, 1, 1, 1, 1, {tile_N}), dtype=nl.float32, buffer=nl.psum)",
            f"{indent}for i_block_K in nl.affine_range({K // tile_K}):",
            f"{i1}for i_tile_K in nl.affine_range(1):",
            f"{i2}{sbuf_stat} = nl.ndarray(({tile_K}, 1, 1, 1, 1, {tile_M}), dtype=nl.{stat.dtype}, buffer=nl.sbuf)",
            f"{i2}nisa.dma_copy(dst={sbuf_stat}[{stat_sl}], src={stat.name}[{k_rng}, {m_rng}])",
            f"{i2}{sbuf_mov} = nl.ndarray(({tile_K}, 1, 1, 1, 1, {tile_N}), dtype=nl.{mov.dtype}, buffer=nl.sbuf)",
            f"{i2}nisa.dma_copy(dst={sbuf_mov}[{mov_sl}], src={mov.name}[{k_rng}, {n_rng}])",
            f"{i2}nisa.nc_matmul(dst={psum_name}[{ps}], stationary={sbuf_stat}[{stat_sl}], moving={sbuf_mov}[{mov_sl}])",
            f"{indent}nisa.tensor_copy(dst={sbuf_out}[{si}], src={psum_name}[{ps}])",
            f"{indent}nisa.dma_copy(dst={hbm_out}[{m_rng}, {n_rng}], src={sbuf_out}[{si}])",
        ]

    def render(self, ctx: RenderContext, operand_map: dict[str, str]) -> list[str]:
        """Emit NKI source lines for nc_matmul.

        Args:
            ctx: Running render context with tensors and kwargs.
            operand_map: Maps op slot name to tensor name in ctx.

        Returns:
            List of NKI source lines.
        """
        stat = ctx.tensors[operand_map["stationary"]]
        mov = ctx.tensors[operand_map["moving"]]
        K, M = stat.shape
        N = mov.shape[1]
        tile_M, tile_N = 128, 512
        num_blocks_M = M // tile_M
        num_blocks_N = N // tile_N
        output_name = _find_output_name(ctx, operand_map, (M, N))
        sbuf_out = f"sbuf_{output_name}"
        input_dtype = stat.dtype

        lines = [
            f'"""nisa.nc_matmul -- {stat.name}(K, M) x {mov.name}(K, N)'
            f' -> {output_name}(M, N), accumulate over K"""',
            f"{sbuf_out} = nl.ndarray(({tile_M}, {num_blocks_M}, {num_blocks_N},"
            f" 1, 1, {tile_N}), dtype=nl.{input_dtype}, buffer=nl.sbuf)",
            f"for i_block_M in nl.affine_range({num_blocks_M}):",
            f"    for i_tile_M in nl.affine_range(1):",
            f"        for i_block_N in nl.affine_range({num_blocks_N}):",
            f"            for i_tile_N in nl.affine_range(1):",
        ]
        lines.extend(self._emit_k_reduction(ctx, operand_map, output_name, " " * 16))
        return lines
