"""Flatten one SBUF tile into a single-row HBM buffer with ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIFlattenStore(NKIOp):
    """Store a two-dimensional SBUF tile as one contiguous HBM row."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("R", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"R": 1, "O": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"P", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "R": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "R": 1, "O": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"width"})
    OUTPUT_ROLE: ClassVar[str] = "shared_hbm"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF input with exactly the configured element count."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "sbuf":
            raise TypeError(f"NKIFlattenStore(src=<role={role}>) expects SBUF")
        if np.asarray(kwargs["src"]).size != int(kwargs["width"]):
            raise ValueError("NKIFlattenStore width must equal the source element count")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source flattened into one HBM row."""
        return np.asarray(kwargs["src"]).reshape(1, int(kwargs["width"])).copy()


__all__ = ["NKIFlattenStore"]
