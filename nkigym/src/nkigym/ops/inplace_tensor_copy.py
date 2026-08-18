"""In-place SBUF insertion through ``nisa.tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIInplaceTensorCopy(NKIOp):
    """Copy one packed source tile into an existing destination interval."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "P", "O"), "dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "P"), ("O",)),
        "dst": (("G", "P"), ("F",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": "width"}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"dst": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "start", "width"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "O": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "O": None, "F": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source and an SBUF destination."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf", "psum"}:
            raise TypeError("NKIInplaceTensorCopy.src expects SBUF or PSUM")
        if _operand_role(kwargs["dst"]) not in {None, "sbuf"}:
            raise TypeError("NKIInplaceTensorCopy.dst expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Insert the source into the selected free-axis interval."""
        start, width = int(kwargs["start"]), int(kwargs["width"])
        result = kwargs["dst"]
        result[:, start : start + width] = np.asarray(kwargs["src"])
        return result


__all__ = ["NKIInplaceTensorCopy"]
