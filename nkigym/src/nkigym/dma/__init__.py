"""DMA: load, store, and staging transfers between HBM, SBUF, and PSUM."""

from nkigym.dma.codegen import render_loads_for_group, render_store
from nkigym.dma.gadgets import load_tensor_block, stage_tensor_block, store_tensor_block

__all__ = ["load_tensor_block", "render_loads_for_group", "render_store", "stage_tensor_block", "store_tensor_block"]
