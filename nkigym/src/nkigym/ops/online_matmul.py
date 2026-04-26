"""Online-fused matmul op: ``nc_matmul`` K-reduce + ``scalar_tensor_tensor`` drain.

Implements ``output = output * scale + stationary.T @ moving`` per K
block — the per-K-iteration drain pattern used by online fusion
(Algorithm 4). Fresh PSUM accumulates the K-sub-reduce; the drain
folds the K-block result into a persistent SBUF accumulator scaled by
a caller-provided ``(M,)`` per-row ``scale`` vector.

Contrast with :class:`NKIMatmul` which drains via
``tensor_copy + tensor_tensor(add)`` — the online form replaces both
with a single ``scalar_tensor_tensor`` instruction so the ``s_k``
rescale fires in the same pass as the accumulate-add.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.ops.base import NKIOp

MATMUL_FREE_MAX = 512


class NKIOnlineMatmul(NKIOp):
    """Online-fused matmul: ``output = output * scale + stationary.T @ moving``.

    ``scale`` is a per-M-row ``(P=M,)`` vector broadcast across the N
    axis of ``output`` — it implements the Algorithm 4 ``s_k`` rescale
    of the persistent accumulator prior to folding the fresh K-block.

    The op declares ``K`` as its only blocking dim — same as
    :class:`NKIMatmul`. The renderer's reducer machinery (memset
    prologue, accumulator-coverage validator) treats both kinds the
    same way.
    """

    NAME: ClassVar[str] = "online_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "M"),
        "moving": ("K", "N"),
        "scale": ("M",),
    }
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("M", "N")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": MATMUL_FREE_MAX}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: ``output * scale[:, None] + stationary.T @ moving``.

        Takes the prior accumulator value through the ``output`` kwarg
        because the op is stateful — the caller threads the same
        accumulator tensor across iterations. On the first K block the
        caller passes a zero tensor.
        """
        stationary: np.ndarray = kwargs["stationary"]
        moving: np.ndarray = kwargs["moving"]
        scale: np.ndarray = kwargs["scale"]
        output: np.ndarray = kwargs["output"]
        return output * scale[..., np.newaxis] + stationary.T @ moving


def online_matmul_block(sbuf_out: Any, sbuf_lhs_T: Any, sbuf_rhs: Any, sbuf_scale: Any) -> None:
    """Online-fused drain: ``O_new = s_k * O_old + lhs_T @ rhs``.

    Inputs mirror :func:`nkigym.ops.matmul.matmul_block` plus a
    ``(p_tile, 1)`` per-M-tile scale leaf list. Per (m, n) output tile:

    1. Allocate a fresh PSUM tile, memset to 0.
    2. Loop K-tiles: ``nc_matmul(psum_tile, stationary, moving)``.
    3. ``scalar_tensor_tensor(dst=sbuf_out, data=sbuf_out,
       op0=multiply, operand0=sbuf_scale, op1=add, operand1=psum_tile)``
       — single fused instruction; no ``tensor_copy`` round-trip.

    Unlike :func:`nkigym.ops.matmul.matmul_block`, this gadget keeps
    its PSUM scratch internal: the online-fusion recurrence
    ``O_k = s_k · O_{k-1} + A_k`` is not add-associative across K
    iterations, so the drain must fire per K block, not once at the
    end of the K loop.
    """
    _TILE_M_MAX = 128
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    if tile_m > _TILE_M_MAX:
        raise ValueError(f"online_matmul_block: tile_m={tile_m} exceeds HW cap {_TILE_M_MAX}")
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            nisa.scalar_tensor_tensor(
                dst=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                data=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                op0=nl.multiply,
                operand0=sbuf_scale[m_idx][0:tile_m, 0:1],
                op1=nl.add,
                operand1=psum_tile[0:tile_m, 0:tile_n],
            )
