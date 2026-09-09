"""Stable nonzero compaction with ``nisa.nonzero_with_count``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKINonzeroWithCount(NKIOp):
    """Return stable nonzero indices, padding, and the final count."""

    NAME: ClassVar[str] = "nonzero_with_count"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "F"), "dst": ("P", "G", "O")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("G", "F")),
        "dst": (("P",), ("G", "O")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 1, "F": "input_width", "O": "output_width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"G"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "G": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 1, "G": 1, "F": None, "O": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"columns", "input_width", "output_width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, columns: int, input_width: int, output_width: int) -> None:
        """Configure the input/output extents and native padding values."""
        super().__init__(
            columns=columns, input_width=input_width, output_width=output_width, index_offset=0, padding_val=-1
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require packed on-chip rows and exactly one count per row."""
        source = np.asarray(kwargs["src"])
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKINonzeroWithCount(src=<role={role}>) expects sbuf")
        columns = int(kwargs["columns"])
        input_width = int(kwargs["input_width"])
        if columns < 1 or source.shape != (1, columns * input_width) or int(kwargs["output_width"]) != input_width + 1:
            raise ValueError("NKINonzeroWithCount requires packed rows and output_width=input_width+1")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Compact each packed row and append its count."""
        columns = int(kwargs["columns"])
        input_width = int(kwargs["input_width"])
        output_width = int(kwargs["output_width"])
        source = np.asarray(kwargs["src"]).reshape(columns, input_width)
        output = np.full((columns, output_width), -1, dtype=np.int32)
        for column, row in enumerate(source):
            indices = np.flatnonzero(row).astype(np.int32)
            output[column, : indices.size] = indices
            output[column, -1] = indices.size
        return output.reshape(1, columns * output_width)


__all__ = ["NKINonzeroWithCount"]
