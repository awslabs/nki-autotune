"""DMA-engine transpose op: ``nisa.dma_transpose`` + ``dma_transpose_block`` gadget.

Same math as :class:`NKITranspose` (which uses Tensor Engine
``nisa.nc_transpose``), but runs on the DMA engine so it doesn't
contend with matmul for TE cycles. Useful when the matmul is TE-bound
and an explicit DMA transpose is cheaper than a round-trip through
PSUM. Also the target of the ``TransposeThroughLoad`` rewrite, where the
``src`` input is an HBM parameter instead of an SBUF buffer.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import nki.isa as nisa
import numpy as np

from nkigym.ops.base import BatchedPermutationContract, NKIOp, PermutationContract, _operand_role


class NKIDMATranspose(NKIOp):
    """DMA-engine transpose ``src(P, F) -> dst(F, P)``."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    """Both abstract axes become a partition axis on one side of the
    transpose, so one instruction covers at most 128 elements of each.
    Larger free dimensions are represented by outer loops."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Return the two-axis transpose contract."""
        _ = kwargs
        return PermutationContract(
            input_operand="src",
            output_operand="dst",
            permutation=(1, 0),
            batching=BatchedPermutationContract(permutation=(3, 1, 2, 0), input_axes=(0, 3), batch_axis=2),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` may be HBM param (``TransposeThroughLoad`` rewrite) or SBUF."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "sbuf"}:
            raise TypeError(f"NKIDMATranspose(src=<role={role}>) expects param or sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``src.T``."""
        return np.array(kwargs["src"]).T


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
