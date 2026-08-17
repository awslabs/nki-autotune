"""Indirect HBM row gather through ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIGather(NKIOp):
    """Gather valid HBM rows selected by one SBUF index per partition."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "indices": ("P", "I"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128, "I": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "I": 1}
    OUTPUT_ROLE: ClassVar[str] = "sbuf"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source and SBUF indices."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIGather(src=<role={role}>) expects an HBM parameter")
        if (role := _operand_role(kwargs["indices"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGather(indices=<role={role}>) expects SBUF indices")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather rows after rejecting indices outside the HBM source."""
        source = np.asarray(kwargs["src"])
        indices = np.asarray(kwargs["indices"]).reshape(-1).astype(np.int64)
        valid = (indices >= 0) & (indices < source.shape[0])
        if not np.all(valid):
            raise ValueError("NKIGather indices must select valid source rows")
        result = source[indices]
        return result


__all__ = ["NKIGather"]
