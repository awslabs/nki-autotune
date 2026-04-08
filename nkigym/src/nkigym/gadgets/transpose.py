"""Tile-level transpose helpers for 4D SBUF buffers.

Each tile is transposed in-place via a PSUM intermediate:
``nc_transpose`` reads the tile into PSUM, then ``tensor_copy``
writes it back to the same SBUF slot.

Buffer layout: ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``.
"""

from typing import Any

import nki.isa as nisa
import nki.language as nl


def transpose_tile(sbuf_tensor: Any, par_tid: int, free_tid: int, dtype: Any) -> None:
    """Transpose a single tile in a 4D SBUF buffer in-place.

    Uses ``nisa.nc_transpose`` into a PSUM intermediate, then
    ``nisa.tensor_copy`` back to the same SBUF slot.

    Args:
        sbuf_tensor: 4D SBUF tensor.
        par_tid: Partition tile index.
        free_tid: Free tile index.
        dtype: Data type for the intermediate PSUM buffer.
    """
    pmax = nl.tile_size.pmax
    tile_t = nl.ndarray((pmax, pmax), dtype=dtype, buffer=nl.psum)
    nisa.nc_transpose(
        dst=tile_t[0:pmax, 0:pmax], data=sbuf_tensor[0:pmax, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:pmax]
    )
    nisa.tensor_copy(
        dst=sbuf_tensor[0:pmax, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:pmax], src=tile_t[0:pmax, 0:pmax]
    )


def transpose_all_tiles(sbuf_tensor: Any, dtype: Any) -> None:
    """Transpose every tile in a 4D SBUF buffer in-place.

    Iterates over all ``(par_tid, free_tid)`` positions and
    calls ``transpose_tile`` on each.

    Args:
        sbuf_tensor: 4D SBUF tensor.
        dtype: Data type for the intermediate PSUM buffer.
    """
    num_tiles_p = sbuf_tensor.shape[1]
    num_tiles_f = sbuf_tensor.shape[2]
    for par_tid in range(num_tiles_p):
        for free_tid in range(num_tiles_f):
            transpose_tile(sbuf_tensor, par_tid, free_tid, dtype)
