"""Block-level DMA and matmul gadgets.

SBUF buffers are Python lists of 2D ``nl.ndarray(p_tile, f_tile)`` leaves,
one leaf per P-tile slot. The F-axis is packed into the leaf's free-axis
width via ``num_f_tiles``. NKI forbids a single DMA that spans multiple
partition slots, so the gadgets Python-iterate per leaf and emit one ISA
call per tile.
"""

from typing import Any

import nki.isa as nisa
import nki.language as nl


def load_block(sbuf: Any, mem_slice: Any, transpose: bool) -> None:
    """HBM → SBUF: copy ``mem_slice`` into every leaf of ``sbuf``.

    ``sbuf`` is a list of ``num_p_tiles`` leaves of shape
    ``(p_tile, f_tile)``. The caller passes the HBM region as a
    Python slice; the gadget splits it across the P-slots.

    Required: ``mem_slice.shape == (num_p_tiles * p_tile, f_tile)``.
    When ``transpose=True``, required shape is
    ``(f_tile, num_p_tiles * p_tile)``.
    """
    num_p_tiles = len(sbuf)
    if num_p_tiles == 0:
        raise ValueError("load_block got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    expected = (f_tile, num_p_tiles * p_tile) if transpose else (num_p_tiles * p_tile, f_tile)
    if tuple(mem_slice.shape) != expected:
        raise ValueError(
            f"load_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), expected mem_slice {expected}, "
            f"got {tuple(mem_slice.shape)}"
        )
    for pt in range(num_p_tiles):
        dst = sbuf[pt][0:p_tile, 0:f_tile]
        if transpose:
            nisa.dma_transpose(dst, mem_slice[0:f_tile, pt * p_tile : (pt + 1) * p_tile])
        else:
            nisa.dma_copy(dst, mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile])


def store_block(mem_slice: Any, sbuf: Any) -> None:
    """SBUF → HBM: write every leaf of ``sbuf`` into ``mem_slice``.

    Required: ``mem_slice.shape == (num_p_tiles * p_tile, f_tile)``.
    """
    num_p_tiles = len(sbuf)
    if num_p_tiles == 0:
        raise ValueError("store_block got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    expected = (num_p_tiles * p_tile, f_tile)
    if tuple(mem_slice.shape) != expected:
        raise ValueError(
            f"store_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), expected mem_slice {expected}, "
            f"got {tuple(mem_slice.shape)}"
        )
    for pt in range(num_p_tiles):
        nisa.dma_copy(mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile], sbuf[pt][0:p_tile, 0:f_tile])


def matmul_block(sbuf_out: Any, sbuf_lhs_T: Any, sbuf_rhs: Any) -> None:
    """Two-level matmul block over flat K/M tile lists.

    Inputs are lists of 2D leaves:

      * ``sbuf_lhs_T[k]``: leaf ``(tile_k, num_m_tiles * tile_m)`` — stationary;
        ``num_m_tiles = len(sbuf_out)``.
      * ``sbuf_rhs[k]``: leaf ``(tile_k, num_n_tiles * tile_n)`` — moving.
      * ``sbuf_out[m]``: leaf ``(tile_m, num_n_tiles * tile_n)`` — accumulator.

    ``tile_m`` is the P-axis of the output leaf (HW-capped at 128).
    N-axis packing: if the leaf's free-axis width is ``<= 512`` it is
    one N-tile of that width; otherwise it splits into ``width // 512``
    tiles of ``tile_n = 512`` (the HW PSUM-free-axis cap).

    Caller must pre-memset every ``sbuf_out[m]`` leaf before the first
    outer-K invocation.
    """
    _TILE_M_MAX = 128
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    if tile_m > _TILE_M_MAX:
        raise ValueError(f"matmul_block: tile_m={tile_m} exceeds HW cap {_TILE_M_MAX}")
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            acc_tile = nl.ndarray((tile_m, tile_n), dtype=sbuf_out[0].dtype, buffer=nl.sbuf)
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            nisa.tensor_copy(acc_tile[0:tile_m, 0:tile_n], psum_tile[0:tile_m, 0:tile_n])
            nisa.tensor_tensor(
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                acc_tile[0:tile_m, 0:tile_n],
                op=nl.add,
            )


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
