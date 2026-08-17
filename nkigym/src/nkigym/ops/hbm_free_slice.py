"""Load one contiguous HBM free-axis slice with ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIHBMFreeSlice(NKIOp):
    """Load ``src[:, start:start + width]`` into one compact SBUF tensor."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"O": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "O": None}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"src": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"start", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source and a valid free-axis interval."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIHBMFreeSlice(src=<role={role}>) expects an HBM parameter")
        start, width = int(kwargs["start"]), int(kwargs["width"])
        if start < 0 or width < 1 or start + width > np.asarray(kwargs["src"]).shape[1]:
            raise ValueError(f"invalid HBM free-axis slice start={start}, width={width}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one copied HBM free-axis interval."""
        start, width = int(kwargs["start"]), int(kwargs["width"])
        return np.asarray(kwargs["src"])[:, start : start + width].copy()


__all__ = ["NKIHBMFreeSlice"]
