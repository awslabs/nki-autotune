"""In-place top-eight destination slices through ``nisa.max8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIInplaceMax8(NKIOp):
    """Write one top-eight result into an existing packed destination."""

    NAME: ClassVar[str] = "max8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "P", "F"), "dst": ("G", "P", "K")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "P"), ("F",)),
        "dst": (("G", "P"), ("K",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {
        "src": ((1, "source_start", "source_width"),),
        "dst": ((1, "output_start", "output_width"),),
    }
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"groups", "partitions", "source_start", "source_width", "output_start", "output_width"}
    )
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "K": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("src", "dst")):
            raise TypeError("NKIInplaceMax8 expects SBUF operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Write descending top-eight values into the selected output interval."""
        source = np.asarray(kwargs["src"])[:, : int(kwargs["source_width"])]
        start, width = int(kwargs["output_start"]), int(kwargs["output_width"])
        result = kwargs["dst"]
        result[:, start : start + width] = np.sort(source, axis=1)[:, -width:][:, ::-1]
        return result


__all__ = ["NKIInplaceMax8"]
