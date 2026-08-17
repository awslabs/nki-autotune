"""Stable nonzero compaction with ``nisa.nonzero_with_count``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKINonzeroWithCount(NKIOp):
    """Return stable nonzero indices, padding, and the final count."""

    NAME: ClassVar[str] = "nonzero_with_count"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"F": "input_width", "O": "output_width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "O": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"input_width", "output_width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, input_width: int, output_width: int) -> None:
        """Configure the input/output extents and native padding values."""
        super().__init__(input_width=input_width, output_width=output_width, index_offset=0, padding_val=-1)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip row and exactly one extra output element."""
        source = np.asarray(kwargs["src"])
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKINonzeroWithCount(src=<role={role}>) expects sbuf")
        if source.shape != (1, int(kwargs["input_width"])) or int(kwargs["output_width"]) != source.shape[1] + 1:
            raise ValueError("NKINonzeroWithCount requires one partition and output_width=input_width+1")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Compact nonzero indices and append their count."""
        source = np.asarray(kwargs["src"])
        output = np.full((1, source.shape[1] + 1), -1, dtype=np.int32)
        indices = np.flatnonzero(source[0]).astype(np.int32)
        output[0, : indices.size] = indices
        output[0, -1] = indices.size
        return output


__all__ = ["NKINonzeroWithCount"]
