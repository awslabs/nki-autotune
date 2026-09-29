"""Conditional floating-point selection through native ``select_reduce``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKISelectReduce(NKIOp):
    """Select source elements or one scalar fallback without accumulation."""

    NAME: ClassVar[str] = "select_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "on_true": ("P", "F"),
        "predicate": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"on_true", "predicate"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {operand: frozenset({"sbuf"}) for operand in INPUT_OPERANDS}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        "on_true": frozenset({"float16", "bfloat16", "float32"}),
        "predicate": frozenset({"uint16"}),
    }
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"predicate": "uint16"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, on_false: float) -> None:
        """Select one FP32 scalar fallback and disable accumulated reduction."""
        super().__init__(on_false=float(np.float32(on_false)))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require matching SBUF source and unsigned predicate tiles."""
        for operand in self.INPUT_OPERANDS:
            if _operand_role(kwargs[operand]) not in {None, "sbuf"}:
                raise TypeError(f"NKISelectReduce.{operand} requires SBUF")
        source, predicate = (np.asarray(kwargs[name]) for name in ("on_true", "predicate"))
        if source.shape != predicate.shape or source.ndim != 2:
            raise ValueError("NKISelectReduce requires matching rank-two operands")
        if predicate.dtype != np.uint16:
            raise TypeError("NKISelectReduce predicate requires uint16")
        if str(source.dtype) not in self.INPUT_STORAGE_DTYPES["on_true"]:
            raise TypeError("NKISelectReduce requires a floating-point source")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy selected values without arithmetic on unselected elements."""
        return np.where(
            np.asarray(kwargs["predicate"]) != 0,
            np.asarray(kwargs["on_true"], dtype=np.float32),
            np.float32(kwargs["on_false"]),
        )


__all__ = ["NKISelectReduce"]
