"""Load row groups and uniform free-axis chunks from HBM."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIGroupedChunkLoad(NKIOp):
    """Load independent ``(P, F)`` chunks while preserving a group axis."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "T", "F"), "dst": ("P", "G", "T", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G", "T", "F")) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "T": 1, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, width: int) -> None:
        """Configure grouped row and free-axis chunk extents."""
        super().__init__(groups=groups, partitions=partitions, chunks=chunks, width=width)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving grouped load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIGroupedChunkLoad(src=<role={role}>) expects an HBM parameter")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one value-preserving copy of the packed HBM matrix."""
        return np.array(kwargs["src"], copy=True)


__all__ = ["NKIGroupedChunkLoad"]
