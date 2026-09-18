"""Numeric conversion to uint16 through native ``tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIUInt16Cast(NKIOp):
    """Convert an on-chip tile numerically to uint16."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint16"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Use the Vector Engine for the native conversion."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF input."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("NKIUInt16Cast.src requires SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve uint16 values or round through the native FP32 representation."""
        source = np.asarray(kwargs["src"])
        return source.copy() if source.dtype == np.uint16 else np.rint(source.astype(np.float32)).astype(np.uint16)
