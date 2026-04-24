"""Block-level DMA and matmul gadgets over the ``Buffers`` SBUF model.

SBUF buffers are ``Buffers`` grids indexable as
``buf[p_buffer][p_tile][f_buffer]``, with each leaf a 2D
``nl.ndarray(p_tile, f_tile)``. NKI forbids a single DMA that spans
multiple partition slots, so the gadgets Python-iterate per leaf and
emit one ISA call per tile.
"""

from typing import Any

import nki.isa as nisa
import nki.language as nl


def load_block(sbuf: Any, mem: Any, p_start: int, p_size: int, f_start: int, f_size: int, transpose: bool) -> None:
    """HBM → SBUF: copy ``mem[p_start:p_start+p_size, f_start:f_start+f_size]`` into every leaf of ``sbuf``.

    ``sbuf`` is a ``Buffers`` grid indexable as
    ``sbuf[p_buffer][p_tile][f_buffer]``. The P-axis splits across
    ``num_p_tiles`` list slots of ``p_tile`` rows each; the F-axis is
    packed into the leaf's free-axis width ``f_tile``. Multi-buffering
    dims ``num_p_buffers`` / ``num_f_buffers`` get the same HBM data
    replicated into every slot.

    Required: ``p_size == num_p_tiles * p_tile`` and ``f_size == f_tile``.

    When ``transpose=True``, the HBM region is read as
    ``mem[f_start:f_start+f_size, p_start:p_start+p_size]`` and
    each leaf is filled via ``nisa.dma_transpose``.
    """
    num_p_buffers = len(sbuf)
    num_p_tiles = len(sbuf[0]) if num_p_buffers else 0
    num_f_buffers = len(sbuf[0][0]) if num_p_tiles else 0
    if num_p_buffers == 0 or num_p_tiles == 0 or num_f_buffers == 0:
        raise ValueError(f"load_block got empty sbuf with shape ({num_p_buffers}, {num_p_tiles}, {num_f_buffers})")
    p_tile, f_tile = sbuf[0][0][0].shape
    if p_size != num_p_tiles * p_tile or f_size != f_tile:
        raise ValueError(
            f"load_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), got (p_size, f_size)=({p_size}, {f_size})"
        )
    for pb in range(num_p_buffers):
        for pt in range(num_p_tiles):
            for fb in range(num_f_buffers):
                dst = sbuf[pb][pt][fb][0:p_tile, 0:f_tile]
                p0 = p_start + pt * p_tile
                if transpose:
                    nisa.dma_transpose(dst, mem[f_start : f_start + f_tile, p0 : p0 + p_tile])
                else:
                    nisa.dma_copy(dst, mem[p0 : p0 + p_tile, f_start : f_start + f_tile])


def store_block(mem: Any, sbuf: Any, p_start: int, p_count: int, f_start: int, f_count: int) -> None:
    """SBUF → HBM: write the ``[p_start : p_start + p_count][f_start : f_start + f_count]`` sub-block of ``sbuf`` into ``mem``."""
    p, f = sbuf[0][0].shape
    op, of = mem.shape
    if op != p_count * p or of != f_count * f:
        raise ValueError(
            f"store_block shape mismatch: sbuf sub-block ({p_count}, {f_count})x({p}, {f}) "
            f"covers ({p_count * p}, {f_count * f}), mem {mem.shape}"
        )
    for pi in range(p_count):
        for fi in range(f_count):
            nisa.dma_copy(mem[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f], sbuf[p_start + pi][f_start + fi][0:p, 0:f])


def matmul_block(sbuf_out: Any, sbuf_lhs_T: Any, sbuf_rhs: Any) -> None:
    """Two-level matmul block over pre-sliced K/M lists.

    Inputs are flat lists whose leaves pack the free-axis ltiles:

      * ``sbuf_lhs_T[k]``: leaf ``(tile_k, num_m_tiles * tile_m)`` — stationary;
        ``num_m_tiles = len(sbuf_out)``.
      * ``sbuf_rhs[k]``: leaf ``(tile_k, num_n_tiles * tile_n)`` — moving.
      * ``sbuf_out[m]``: leaf ``(tile_m, num_n_tiles * tile_n)`` — accumulator.

    ``tile_m`` is the P-axis of the output leaf (HW-capped at 128).
    N-axis packing: if the leaf's free-axis width is ``<= 512`` it is
    one N-tile of that width; otherwise it splits into ``width // 512``
    tiles of ``tile_n = 512`` (the HW PSUM-free-axis cap).

    For each ``(m_idx, n_idx)`` the gadget zeroes a PSUM tile, reduces
    all K with ``nc_matmul``, then adds the PSUM result into
    ``sbuf_out[m_idx]``'s ``[:, n_idx*tile_n : (n_idx+1)*tile_n]`` strip.
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
    psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
    acc_tile = nl.ndarray((tile_m, tile_n), dtype=sbuf_out[0].dtype, buffer=nl.sbuf)
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
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


class Buffers:
    """Indexable 3D grid of 2D SBUF / PSUM leaves.

    Indexing forms (``buf`` wraps a full ``[p_buffer][p_tile][f_buffer]`` grid):
      * ``buf[pb][pt][fb]``        — one leaf (``nl.ndarray``).
      * ``buf[pb][pt]``            — list of ``fb`` leaves at that row.
      * ``buf[pb][pt_slice][fb]``  — list of leaves at ``fb`` across the slice.

    After a slice, integer indexing projects across the slice range
    instead of selecting a single row — this is the only divergence
    from plain-list semantics.
    """

    def __init__(self, data: list, _sliced: bool = False) -> None:
        self._data = data
        self._sliced = _sliced

    def __getitem__(self, idx):
        if self._sliced:
            return [row[idx] for row in self._data]
        if isinstance(idx, slice):
            return Buffers(self._data[idx], _sliced=True)
        inner = self._data[idx]
        if isinstance(inner, list) and inner and isinstance(inner[0], list):
            return Buffers(inner)
        return inner

    def __len__(self) -> int:
        return len(self._data)


def allocate_buffers(
    p_tile_size: int,
    num_p_tiles: int,
    num_p_buffers: int,
    f_tile_size: int,
    num_f_tiles: int,
    num_f_buffers: int,
    loc,
    dtype,
    initial_value: float | None = None,
) -> Buffers:
    """Allocate a ``Buffers`` grid of 2D SBUF / PSUM leaves.

    Returns ``Buffers`` indexable as ``buf[p_buffer][p_tile][f_buffer]``,
    with each leaf an ``nl.ndarray`` of shape
    ``(p_tile_size, f_tile_size * num_f_tiles)``. ``num_f_tiles`` is
    packed *into* the leaf; ``num_p_buffers`` / ``num_f_buffers`` are
    list-level multi-buffering factors.

    If ``initial_value`` is given, every leaf is ``nisa.memset`` to it.
    """
    leaf_shape = (p_tile_size, f_tile_size * num_f_tiles)
    grid = []
    for _ in range(num_p_buffers):
        p_slots = []
        for _ in range(num_p_tiles):
            f_slots = []
            for _ in range(num_f_buffers):
                leaf = nl.ndarray(leaf_shape, dtype=dtype, buffer=loc)
                if initial_value is not None:
                    nisa.memset(leaf[0 : leaf_shape[0], 0 : leaf_shape[1]], initial_value)
                f_slots.append(leaf)
            p_slots.append(f_slots)
        grid.append(p_slots)
    return Buffers(grid)
