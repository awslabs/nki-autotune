"""Copy a fixed free-axis slice with ``nisa.tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKITensorSlice(NKIOp):
    """Copy one contiguous prefix or interior slice into a compact tensor."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"O": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "O": None}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"src": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"start", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source and a valid contiguous slice."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKITensorSlice(src=<role={role}>) expects sbuf")
        start, width = int(kwargs["start"]), int(kwargs["width"])
        if start < 0 or width < 1 or start + width > np.asarray(kwargs["src"]).shape[-1]:
            raise ValueError(f"invalid tensor slice start={start}, width={width}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the requested free-axis interval for CPU validation."""
        start, width = int(kwargs["start"]), int(kwargs["width"])
        return np.asarray(kwargs["src"])[:, start : start + width].copy()


__all__ = ["NKITensorSlice"]
