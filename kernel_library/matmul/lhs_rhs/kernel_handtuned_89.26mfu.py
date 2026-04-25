from typing import Any

import nki.isa as nisa
import nki.language as nl


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


def dma_transpose_block(sbuf_dst: Any, src: Any) -> None:
    """DMA-engine transpose into an SBUF buffer.

    ``src`` may be either:

    * a **2D region** (HBM slice or a single SBUF leaf) of shape
      ``(f_tile, num_k_tiles * p_tile)`` — the HBM → SBUF case. Emits
      one ``nisa.dma_transpose`` per destination P-slot.
    * a **list of SBUF leaves** of shape ``(p_tile, num_k_tiles * p_tile)``
      with length ``num_m_tiles`` — the SBUF → SBUF case. Emits one
      ``nisa.dma_transpose`` per ``(ki, mi)`` pair.

    ``sbuf_dst`` is always ``num_k_tiles`` leaves of shape
    ``(p_tile, num_m_tiles * p_tile)`` with the packed free axis
    holding M-tiles.
    """
    num_k_tiles = len(sbuf_dst)
    p_tile, packed_free = sbuf_dst[0].shape
    if isinstance(src, list):
        num_m_tiles = len(src)
        for ki in range(num_k_tiles):
            for mi in range(num_m_tiles):
                nisa.dma_transpose(
                    sbuf_dst[ki][0:p_tile, mi * p_tile : (mi + 1) * p_tile],
                    src[mi][0:p_tile, ki * p_tile : (ki + 1) * p_tile],
                )
        return
    f_tile = src.shape[0]
    expected = (f_tile, num_k_tiles * p_tile)
    if tuple(src.shape) != expected:
        raise ValueError(
            f"dma_transpose_block extent mismatch: sbuf_dst covers "
            f"({num_k_tiles * p_tile}, {f_tile}) via {num_k_tiles} P-slots of "
            f"({p_tile}, {f_tile}), expected src {expected}, got {tuple(src.shape)}"
        )
    _ = packed_free
    for ki in range(num_k_tiles):
        nisa.dma_transpose(sbuf_dst[ki][0:p_tile, 0:f_tile], src[0:f_tile, ki * p_tile : (ki + 1) * p_tile])


def load_block(sbuf: Any, mem_slice: Any) -> None:
    """HBM → SBUF: copy ``mem_slice`` into every leaf of ``sbuf`` via ``nisa.dma_copy``.

    ``sbuf`` is a list of ``num_p_tiles`` leaves of shape
    ``(p_tile, f_tile)``. The caller passes the HBM region as a
    Python slice; the gadget splits it across the P-slots.

    Required: ``mem_slice.shape == (num_p_tiles * p_tile, f_tile)``.

    For HBM→SBUF transposes, use :func:`dma_transpose_block` instead.
    """
    num_p_tiles = len(sbuf)
    if num_p_tiles == 0:
        raise ValueError("load_block got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    expected = (num_p_tiles * p_tile, f_tile)
    if tuple(mem_slice.shape) != expected:
        raise ValueError(
            f"load_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), expected mem_slice {expected}, "
            f"got {tuple(mem_slice.shape)}"
        )
    for pt in range(num_p_tiles):
        nisa.dma_copy(sbuf[pt][0:p_tile, 0:f_tile], mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile])


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


import nki
import nki.language as nl


@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(
        p_tile_size=128,
        num_p_tiles=8,
        f_tile_size=128,
        num_f_tiles=8,
        loc=nl.sbuf,
        dtype=nl.bfloat16,
        num_p_buffers=4,
        num_f_buffers=4,
    )

    for i_block_d0 in range(2):
        sbuf_output = allocate_buffers(
            p_tile_size=128,
            num_p_tiles=8,
            f_tile_size=512,
            num_f_tiles=4,
            loc=nl.sbuf,
            dtype=nl.bfloat16,
            num_p_buffers=None,
            num_f_buffers=None,
        )
        cur_sbuf_output = sbuf_output
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d2 in range(4):
            sbuf_rhs = allocate_buffers(
                p_tile_size=128,
                num_p_tiles=8,
                f_tile_size=512,
                num_f_tiles=1,
                loc=nl.sbuf,
                dtype=nl.bfloat16,
                num_p_buffers=8,
                num_f_buffers=None,
            )
            for i_block_d1 in range(2):
                cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d1 % 4][i_block_d0 % 4]
                dma_transpose_block(
                    cur_sbuf_lhs_T,
                    lhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 1024 : i_block_d1 * 1024 + 1024],
                )
                cur_sbuf_rhs = sbuf_rhs[i_block_d1 % 8]
                load_block(
                    cur_sbuf_rhs,
                    rhs[i_block_d1 * 1024 : i_block_d1 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512],
                )
                matmul_block(
                    [leaf[:, i_block_d2 * 512 : i_block_d2 * 512 + 512] for leaf in cur_sbuf_output],
                    cur_sbuf_lhs_T,
                    cur_sbuf_rhs,
                )
            store_block(
                hbm_output[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512],
                [leaf[:, i_block_d2 * 512 : i_block_d2 * 512 + 512] for leaf in cur_sbuf_output],
            )
    return hbm_output
