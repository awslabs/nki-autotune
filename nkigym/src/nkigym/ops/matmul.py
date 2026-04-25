"""Matrix multiplication op: ``nisa.nc_matmul`` + ``matmul_block`` gadget.

``stationary(K, M).T @ moving(K, N) -> output(M, N)``. Accumulates into
PSUM at fp32 regardless of input dtype.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
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

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation: ``stationary.T @ moving``."""
        stationary: np.ndarray = kwargs["stationary"]
        moving: np.ndarray = kwargs["moving"]
        return stationary.T @ moving


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
