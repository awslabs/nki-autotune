"""PSUM-to-SBUF accumulation helper for 4D buffers.

Adds a 2D PSUM tile into a slot of a 4D SBUF result buffer
using temporary SBUF intermediates for the cross-buffer add.

Buffer layout: ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``.
"""

from typing import Any

import nki.isa as nisa
import nki.language as nl


def accumulate_to_sbuf(result: Any, psum_tile: Any, par_tid: int, free_tid: int) -> None:
    """Add a PSUM tile into a 4D SBUF result buffer slot.

    Copies both the PSUM tile and the existing SBUF slot into
    temporary SBUF buffers, adds them, and writes the sum back.

    Args:
        result: 4D SBUF result buffer.
        psum_tile: 2D PSUM tile to accumulate.
        par_tid: Partition tile index in result.
        free_tid: Free tile index in result.
    """
    tile_size_p = result.shape[0]
    tile_size_f = result.shape[3]

    new_contrib = nl.ndarray((tile_size_p, tile_size_f), dtype=result.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=new_contrib[0:tile_size_p, 0:tile_size_f], src=psum_tile[0:tile_size_p, 0:tile_size_f])

    existing = nl.ndarray((tile_size_p, tile_size_f), dtype=result.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=existing[0:tile_size_p, 0:tile_size_f],
        src=result[0:tile_size_p, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:tile_size_f],
    )

    nisa.tensor_tensor(
        dst=result[0:tile_size_p, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:tile_size_f],
        data1=existing[0:tile_size_p, 0:tile_size_f],
        data2=new_contrib[0:tile_size_p, 0:tile_size_f],
        op=nl.add,
    )
