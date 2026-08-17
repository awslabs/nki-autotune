"""Find top-value indices with ``nisa.nc_find_index8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIFindIndex8(NKIOp):
    """Return first-occurrence indices for eight values per partition."""

    NAME: ClassVar[str] = "nc_find_index8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "vals": ("P", "K"), "dst": ("P", "K")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "vals"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both inputs to reside on chip."""
        for slot in ("data", "vals"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIFindIndex8({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Find the first matching index of each requested value."""
        data = np.asarray(kwargs["data"])
        values = np.asarray(kwargs["vals"])
        result = np.empty(values.shape, dtype=np.uint32)
        for row in range(data.shape[0]):
            used = np.zeros(data.shape[1], dtype=np.bool_)
            for column in range(values.shape[1] - 1, -1, -1):
                matches = np.flatnonzero((data[row] == values[row, column]) & ~used)
                if not matches.size:
                    raise ValueError("NKIFindIndex8 value is absent from its data row")
                result[row, column] = matches[0]
                used[matches[0]] = True
        return result


__all__ = ["NKIFindIndex8"]
