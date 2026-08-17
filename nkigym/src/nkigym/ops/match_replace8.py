"""Replace eight matched values with ``nisa.nc_match_replace8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIMatchReplace8(NKIOp):
    """Replace the first occurrence of eight values in every partition."""

    NAME: ClassVar[str] = "nc_match_replace8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "vals": ("P", "K"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "vals"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both inputs to reside on chip."""
        for slot in ("data", "vals"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIMatchReplace8({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Replace each requested value once for CPU validation."""
        result = np.asarray(kwargs["data"]).copy()
        for row, values in enumerate(np.asarray(kwargs["vals"])):
            for value in values:
                matches = np.flatnonzero(result[row] == value)
                if matches.size:
                    result[row, matches[0]] = kwargs["imm"]
        return result


__all__ = ["NKIMatchReplace8"]
