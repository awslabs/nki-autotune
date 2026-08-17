"""Load one contiguous HBM row slice with ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIHBMRowSlice(NKIOp):
    """Load ``src[start:start + rows, :]`` into one compact SBUF tensor."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("S", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "rows"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"S": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"S": None, "P": 128, "F": None}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str, str], ...]]] = {"src": ((0, "start", "rows", "dst"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"start", "rows"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source and a valid row interval."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIHBMRowSlice(src=<role={role}>) expects an HBM parameter")
        start, rows = int(kwargs["start"]), int(kwargs["rows"])
        if start < 0 or rows < 1 or start + rows > np.asarray(kwargs["src"]).shape[0]:
            raise ValueError(f"invalid HBM row slice start={start}, rows={rows}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one copied HBM row interval."""
        start, rows = int(kwargs["start"]), int(kwargs["rows"])
        return np.asarray(kwargs["src"])[start : start + rows, :].copy()


__all__ = ["NKIHBMRowSlice"]
