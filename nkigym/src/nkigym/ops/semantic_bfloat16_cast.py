"""Semantic bfloat16 cast through ``nisa.activation``."""

from typing import Any, ClassVar

import ml_dtypes
import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKISemanticBF16Cast(NKIOp):
    """Round one on-chip tensor to bfloat16 values."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Configure the native activation as a semantic bfloat16 copy."""
        super().__init__(op="copy")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tile."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKISemanticBF16Cast(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return values rounded through bfloat16 for CPU validation."""
        return np.asarray(kwargs["data"]).astype(ml_dtypes.bfloat16)


__all__ = ["NKISemanticBF16Cast"]
