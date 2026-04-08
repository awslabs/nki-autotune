"""Tile-by-tile DMA transfer helpers for 4D on-chip buffers.

Buffer layout: ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``.
"""

import nki.isa as nisa
import nki.language as nl
from nki.language import NkiTensor


def load_tensor_block(dst: NkiTensor, src: NkiTensor, par_ofs: int, free_ofs: int) -> None:
    """Load tiles from a 2D HBM tensor into a 4D SBUF buffer.

    Iterates over all tile slots in ``dst`` and copies each
    from ``src`` via DMA, starting at the given element offsets.

    Args:
        dst: 4D SBUF buffer (tile_size_P, num_tiles_P, num_tiles_F, tile_size_F).
        src: 2D HBM tensor (total_P, total_F).
        par_ofs: Element offset along the partition axis.
        free_ofs: Element offset along the free axis.
    """
    tile_size_p = dst.shape[0]
    num_tiles_p = dst.shape[1]
    num_tiles_f = dst.shape[2]
    tile_size_f = dst.shape[3]
    for par_tid in range(num_tiles_p):
        par_start = par_ofs + par_tid * tile_size_p
        for free_tid in range(num_tiles_f):
            free_start = free_ofs + free_tid * tile_size_f
            nisa.dma_copy(
                dst=dst[0:tile_size_p, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:tile_size_f],
                src=src[par_start : par_start + tile_size_p, free_start : free_start + tile_size_f],
            )


def save_tensor_block(dst: NkiTensor, src: NkiTensor, par_ofs: int, free_ofs: int) -> None:
    """Save tiles from a 4D on-chip buffer to a 2D HBM tensor.

    Supports both SBUF and PSUM sources. For PSUM sources, each tile is
    staged through a temporary SBUF buffer via ``tensor_copy`` before DMA.

    Args:
        dst: 2D HBM tensor (total_P, total_F).
        src: 4D on-chip buffer (tile_size_P, num_tiles_P, num_tiles_F, tile_size_F).
        par_ofs: Element offset along the partition axis.
        free_ofs: Element offset along the free axis.
    """
    tile_size_p = src.shape[0]
    num_tiles_p = src.shape[1]
    num_tiles_f = src.shape[2]
    tile_size_f = src.shape[3]

    from_psum = src.buffer == nl.psum

    for par_tid in range(num_tiles_p):
        par_start = par_ofs + par_tid * tile_size_p
        for free_tid in range(num_tiles_f):
            free_start = free_ofs + free_tid * tile_size_f
            src_tile = src[0:tile_size_p, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:tile_size_f]
            if from_psum:
                sbuf_staging = nl.ndarray((tile_size_p, 1, 1, tile_size_f), dtype=dst.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sbuf_staging[0:tile_size_p, 0:1, 0:1, 0:tile_size_f], src=src_tile)
                src_tile = sbuf_staging[0:tile_size_p, 0:1, 0:1, 0:tile_size_f]
            nisa.dma_copy(
                dst=dst[par_start : par_start + tile_size_p, free_start : free_start + tile_size_f], src=src_tile
            )
