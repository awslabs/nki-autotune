"""Matrix multiplication op: ``nisa.nc_matmul`` + ``matmul_block`` / ``matmul_drain_block``.

``stationary(K, M).T @ moving(K, N) -> output(M, N)``. Accumulates into
caller-provided PSUM at fp32 regardless of input dtype; a separate
drain gadget copies PSUM→SBUF once the K loop closes.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import numpy as np

from nkigym.ops.base import NKIOp

MATMUL_FREE_MAX = 512


class NKIMatmul(NKIOp):
    """Matrix multiply: ``stationary.T @ moving -> output``."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": MATMUL_FREE_MAX}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: ``stationary.T @ moving``."""
        stationary: np.ndarray = kwargs["stationary"]
        moving: np.ndarray = kwargs["moving"]
        return stationary.T @ moving


def matmul_block(psum_out: Any, sbuf_lhs_T: Any, sbuf_rhs: Any) -> None:
    """Per-K-block ``nc_matmul`` accumulating into caller-provided PSUM.

    Inputs are lists of 2D leaves:

      * ``psum_out[m]``: leaf ``(tile_m, num_n_tiles * tile_n)`` —
        K-accumulator. Caller must allocate in ``nl.psum`` and
        ``memset`` before the first outer-K invocation.
      * ``sbuf_lhs_T[k]``: leaf ``(tile_k, num_m_tiles * tile_m)`` —
        stationary. ``num_m_tiles = len(psum_out)``.
      * ``sbuf_rhs[k]``: leaf ``(tile_k, num_n_tiles * tile_n)`` — moving.

    HW automatically adds successive ``nc_matmul`` results into the
    same PSUM partition, so one call per K block contributes an
    additive partial-product — no explicit ``+=``. The drain gadget
    :func:`matmul_drain_block` copies PSUM→SBUF once the K loop closes.
    """
    _TILE_M_MAX = 128
    _TILE_N_MAX = 512
    num_m_tiles = len(psum_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = psum_out[0].shape[0]
    if tile_m > _TILE_M_MAX:
        raise ValueError(f"matmul_block: tile_m={tile_m} exceeds HW cap {_TILE_M_MAX}")
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )


def matmul_drain_block(sbuf_out: Any, psum_out: Any) -> None:
    """Copy a K-closed PSUM accumulator to SBUF, per (m, n) tile.

    Runs once the matmul's K loop closes — one ``tensor_copy`` per
    (m, n) output tile. No add-accumulate: the SBUF output slot is
    written exactly once, replacing any prior value.
    """
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    tile_m = sbuf_out[0].shape[0]
    free_width = sbuf_out[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            nisa.tensor_copy(
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                psum_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
            )
