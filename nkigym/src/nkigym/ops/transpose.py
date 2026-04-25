"""Tensor-Engine transpose op: ``nisa.nc_transpose`` + ``transpose_block`` gadget.

Swaps the partition and free axes of a tensor. Takes a ``(P, F)``
operand and produces an ``(F, P)`` output. Executes on Tensor Engine
by default (Vector Engine is a 32×32 fallback); the caller pays TE
cycles — contrast with :class:`NKIDMATranspose` which runs on the
DMA engine and leaves TE free for matmul.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.ops.base import NKIOp


class NKITranspose(NKIOp):
    """Transpose ``data(P, F) -> output(F, P)`` on Tensor Engine."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    """Tensor Engine caps the input at 128×128; Vector Engine at 32×32.
    We target Tensor Engine, so both axes are capped at 128."""
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128, "F": 128}

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation: ``data.T``."""
        data: np.ndarray = kwargs["data"]
        return data.T


def transpose_block(sbuf_dst: Any, sbuf_src: Any) -> None:
    """SBUF→SBUF transpose via ``nisa.nc_transpose`` (Tensor Engine).

    ``sbuf_src`` is ``num_m_tiles`` leaves of shape
    ``(p_tile, num_k_tiles * p_tile)`` with the packed free axis
    holding K-tiles. ``sbuf_dst`` is ``num_k_tiles`` leaves of shape
    ``(p_tile, num_m_tiles * p_tile)`` with the packed free axis
    holding M-tiles.

    Tensor Engine ``nc_transpose`` writes to PSUM, so each ``(ki, mi)``
    pair stages through a ``(p_tile, p_tile)`` PSUM tile, then copies
    to its slot in ``sbuf_dst``.
    """
    p_tile = sbuf_src[0].shape[0]
    num_m_tiles = len(sbuf_src)
    num_k_tiles = len(sbuf_dst)
    dtype = sbuf_dst[0].dtype
    for ki in range(num_k_tiles):
        for mi in range(num_m_tiles):
            psum_tile = nl.ndarray((p_tile, p_tile), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(psum_tile[0:p_tile, 0:p_tile], sbuf_src[mi][0:p_tile, ki * p_tile : (ki + 1) * p_tile])
            nisa.tensor_copy(sbuf_dst[ki][0:p_tile, mi * p_tile : (mi + 1) * p_tile], psum_tile[0:p_tile, 0:p_tile])
