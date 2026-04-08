"""Small loop gadgets for tile-level DMA and buffer manipulation.

These helpers encapsulate tile-by-tile iteration patterns so that
the renderer and generated kernels never need inline loop nests
for multi-tile transfers.  All SBUF buffers use the uniform 4D
layout: ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``.
"""

from nkigym.gadgets.accumulate import accumulate_to_sbuf
from nkigym.gadgets.dma import load_tensor_block, save_tensor_block
from nkigym.gadgets.transpose import transpose_all_tiles, transpose_tile

__all__ = ["accumulate_to_sbuf", "load_tensor_block", "save_tensor_block", "transpose_all_tiles", "transpose_tile"]
