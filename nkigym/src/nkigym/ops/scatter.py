"""Indirect SBUF row scatter through ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIScatter(NKIOp):
    """Scatter SBUF rows into a newly allocated HBM output."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "scatter"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "indices": ("P", "I"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128, "I": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "I": 1}
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF data and indices."""
        for slot in ("src", "indices"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIScatter({slot}=<role={role}>) expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Scatter a full unique row permutation."""
        source = np.asarray(kwargs["src"])
        indices = np.asarray(kwargs["indices"]).reshape(-1).astype(np.int64)
        expected = np.arange(source.shape[0])
        if indices.size != source.shape[0] or not np.array_equal(np.sort(indices), expected):
            raise ValueError("NKIScatter requires a permutation of every output row")
        result = np.empty_like(source)
        result[indices] = source
        return result


__all__ = ["NKIScatter"]
