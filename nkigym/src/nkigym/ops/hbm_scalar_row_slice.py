"""Load one dynamically selected HBM matrix with ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIHBMScalarRowSlice(NKIOp):
    """Load one matrix selected by a configured scalar SBUF index."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "scalar_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("E", "L"), "indices": ("I", "J"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "rows", "F": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"E", "L", "I", "J", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"E": 1, "L": 1, "I": 1, "J": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"E": None, "L": None, "I": 1, "J": None, "P": 128, "F": None}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"index", "rows", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one flattened HBM expert matrix and one scalar SBUF index."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIHBMScalarRowSlice(src=<role={role}>) expects an HBM parameter")
        if (role := _operand_role(kwargs["indices"])) is not None and role != "sbuf":
            raise TypeError(f"NKIHBMScalarRowSlice(indices=<role={role}>) expects SBUF indices")
        source, indices = np.asarray(kwargs["src"]), np.asarray(kwargs["indices"])
        index = int(kwargs.get("index", 0))
        rows, width = int(kwargs["rows"]), int(kwargs["width"])
        if (
            source.ndim != 2
            or source.shape[1] != rows * width
            or indices.ndim != 2
            or indices.shape[0] != 1
            or index < 0
            or index >= indices.shape[1]
        ):
            raise ValueError("scalar HBM row slice requires flattened expert matrices and one index")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the dynamically selected expert matrix."""
        source = np.asarray(kwargs["src"])
        selected = int(np.asarray(kwargs["indices"])[0, int(kwargs.get("index", 0))])
        if selected < 0 or selected >= source.shape[0]:
            raise ValueError("scalar HBM row slice index exceeds the expert extent")
        return source[selected].reshape(int(kwargs["rows"]), int(kwargs["width"])).copy()


__all__ = ["NKIHBMScalarRowSlice"]
