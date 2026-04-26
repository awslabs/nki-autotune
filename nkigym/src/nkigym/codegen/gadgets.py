"""Legacy gadget surface — retained only to keep ``nkigym.search.api.inline_gadgets``
importable after the 2N-refactor dropped every block-level helper.

Rendered kernels now emit bare ``nisa.dma_copy`` / ``nisa.nc_matmul`` /
``nisa.tensor_tensor`` / ``nisa.memset`` calls at the op fire depth —
no helper functions in between.
"""

__all__: list[str] = []
