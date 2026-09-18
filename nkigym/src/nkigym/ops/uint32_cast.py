"""Numeric conversion to uint32 through native ``tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIUInt32Cast(NKIOp):
    """Convert an on-chip tile numerically to uint32."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    def __init__(self) -> None:
        """Use the Vector Engine for the native conversion."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF input."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("NKIUInt32Cast.src requires SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve uint32 bits or round through FP32 using native nearest-even conversion."""
        source = np.asarray(kwargs["src"])
        return source.copy() if source.dtype == np.uint32 else np.rint(source.astype(np.float32)).astype(np.uint32)
