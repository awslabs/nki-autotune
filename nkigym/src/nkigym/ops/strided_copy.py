"""Regular within-row selection through one native DMA copy."""

from math import prod
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIStridedCopy(NKIOp):
    """Copy a fixed strided free-axis view independently in each partition."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    STRIDED_COPY_INPUTS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"N": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F", "N"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"pattern", "offset", "width"})

    def __init__(self, pattern: tuple[tuple[int, int], ...], offset: int) -> None:
        """Configure a fixed DMA view with positive extents and nonnegative strides."""
        if not pattern or len(pattern) > 4 or offset < 0 or any(s < 0 or n < 1 for s, n in pattern):
            raise ValueError("strided copy requires one to four valid free-axis dimensions")
        super().__init__(pattern=pattern, offset=offset, width=prod(n for _, n in pattern))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a rank-two SBUF source and a view within its free axis."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("NKIStridedCopy requires a rank-two SBUF source")
        last = kwargs["offset"] + sum(stride * (extent - 1) for stride, extent in kwargs["pattern"])
        if last >= source.shape[1]:
            raise ValueError("strided copy exceeds the source free-axis extent")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the selected values without arithmetic or dtype conversion."""
        indices = np.array([kwargs["offset"]], dtype=np.int64)
        for stride, extent in kwargs["pattern"]:
            indices = (indices[:, None] + stride * np.arange(extent)).reshape(-1)
        return np.asarray(kwargs["src"])[:, indices].copy()
