"""SBUF→SBUF transpose op: ``nisa.dma_transpose``.

Swaps the partition and free axes of a tensor. Takes a ``(P, F)``
operand and produces an ``(F, P)`` output.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKITranspose(NKIOp):
    """Transpose ``data(P, F) -> output(F, P)``."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128, "F": 128}

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation: ``data.T``."""
        data: np.ndarray = kwargs["data"]
        return data.T
