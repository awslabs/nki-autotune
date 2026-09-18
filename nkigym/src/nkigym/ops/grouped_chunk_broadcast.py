"""Grouped chunk-vector broadcast."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.grouped_vector_broadcast import _OPERATIONS


class NKIGroupedChunkBroadcast(NKIOp):
    """Apply one scalar per row, group, and chunk."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "G", "T", "F"),
        "operand0": ("P", "G", "T"),
        "dst": ("P", "G", "T", "F"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G", "T", "F")),
        "operand0": (("P",), ("G", "T")),
        "dst": (("P",), ("G", "T", "F")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "T": 1, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks", "width"})
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, width: int, op0: str) -> None:
        """Configure one chunk-vector operation."""
        if op0 not in _OPERATIONS:
            raise ValueError(f"unsupported grouped chunk operation {op0!r}")
        super().__init__(groups=groups, partitions=partitions, chunks=chunks, width=width, op0=op0)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in ("data", "operand0")):
            raise TypeError("NKIGroupedChunkBroadcast expects on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the configured per-chunk broadcast."""
        g, p, t, f = (int(kwargs[name]) for name in ("groups", "partitions", "chunks", "width"))
        data = np.asarray(kwargs["data"]).reshape(p, g, t, f)
        vector = np.asarray(kwargs["operand0"]).reshape(p, g, t, 1)
        return _OPERATIONS[str(kwargs["op0"])](data, vector).reshape(p, g * t * f)


__all__ = ["NKIGroupedChunkBroadcast"]
