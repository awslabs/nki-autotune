"""Shared block-level allocation + memset gadgets.

Per-op gadgets (``load_block``, ``store_block``, ``matmul_block``,
``transpose_block``, ``dma_transpose_block``) live alongside their
``NKIOp`` subclasses under ``nkigym/ops/*.py`` and are re-exported
here so the renderer can refer to a single import surface.

SBUF buffers are Python lists of 2D ``nl.ndarray(p_tile, f_tile)``
leaves, one leaf per P-tile slot. The F-axis is packed into the
leaf's free-axis width via ``num_f_tiles``. NKI forbids a single DMA
that spans multiple partition slots, so the per-op gadgets iterate
per leaf and emit one ISA call per tile.
"""

from typing import Any

import nki.isa as nisa
import nki.language as nl

from nkigym.ops.activation_reduce import activation_block, activation_reduce_block
from nkigym.ops.dma_transpose import dma_transpose_block
from nkigym.ops.load import load_block
from nkigym.ops.matmul import matmul_block, matmul_drain_block
from nkigym.ops.online_matmul import online_matmul_block
from nkigym.ops.store import store_block
from nkigym.ops.tensor_scalar import tensor_scalar_block
from nkigym.ops.transpose import transpose_block

__all__ = [
    "activation_block",
    "activation_reduce_block",
    "allocate_buffers",
    "dma_transpose_block",
    "load_block",
    "matmul_block",
    "matmul_drain_block",
    "memset_buffers",
    "online_matmul_block",
    "store_block",
    "tensor_scalar_block",
    "transpose_block",
]


def memset_buffers(sbuf: Any, value: float) -> None:
    """Zero every leaf of a buffer list with ``value`` via ``nisa.memset``."""
    if len(sbuf) == 0:
        raise ValueError("memset_buffers got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    for leaf in sbuf:
        nisa.memset(leaf[0:p_tile, 0:f_tile], value)


def allocate_buffers(
    p_tile_size: int,
    num_p_tiles: int,
    f_tile_size: int,
    num_f_tiles: int,
    loc,
    dtype,
    num_p_buffers: int | None,
    num_f_buffers: int | None,
) -> list:
    """Allocate tile buffers for SBUF with per-dim multi-buffering.

    Each leaf is an ``nl.ndarray`` of shape
    ``(p_tile_size, f_tile_size * num_f_tiles)``. ``num_f_tiles`` is packed
    *into* the leaf's free-axis width.

    ``num_p_buffers`` / ``num_f_buffers`` describe multi-buffering along
    the partition and free dims:

      * ``None`` → no rotation along that dim; the dim collapses in the
        return shape.
      * ``N`` (int ≥ 1) → ``N`` independent copies at that dim; caller
        indexes ``bufs[idx % N]`` at the use site.

    Return shape:

      * ``(None, None)`` → flat leaf list ``[leaf, ...]`` of length
        ``num_p_tiles``.
      * ``(P, None)`` → ``bufs[p_buf_idx]`` → leaf list.
      * ``(None, F)`` → ``bufs[f_buf_idx]`` → leaf list.
      * ``(P, F)`` → ``bufs[p_buf_idx][f_buf_idx]`` → leaf list.
    """
    leaf_shape = (p_tile_size, f_tile_size * num_f_tiles)
    p_count = 1 if num_p_buffers is None else num_p_buffers
    f_count = 1 if num_f_buffers is None else num_f_buffers
    nested = [
        [[nl.ndarray(leaf_shape, dtype=dtype, buffer=loc) for _ in range(num_p_tiles)] for _ in range(f_count)]
        for _ in range(p_count)
    ]
    result: Any = nested
    if num_f_buffers is None:
        result = [row[0] for row in result]
    if num_p_buffers is None:
        result = result[0]
    return result
