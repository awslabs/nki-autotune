"""Small loop gadgets for tile-level DMA transfers.

These helpers encapsulate tile-by-tile iteration patterns so that
generated kernels never need inline loop nests for multi-tile
transfers. All SBUF buffers use the uniform 4D layout:
``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``
or the 2D layout for 1D tensors: ``(tile_size_P, num_tiles_P)``.
"""

from nkigym.gadgets.dma import load_tensor_block, stage_tensor_block, store_tensor_block

__all__ = ["load_tensor_block", "stage_tensor_block", "store_tensor_block"]
