"""Find the eight largest values with ``nisa.max8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIMax8(NKIOp):
    """Return the eight largest values in each source partition."""

    NAME: ClassVar[str] = "max8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "K")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tensor."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIMax8(src=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return descending top-eight values for CPU validation."""
        source = np.asarray(kwargs["src"])
        return np.sort(source, axis=1)[:, -8:][:, ::-1].copy()


__all__ = ["NKIMax8"]
