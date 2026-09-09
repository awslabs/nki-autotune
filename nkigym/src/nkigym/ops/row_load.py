"""Load one HBM vector as a single-partition SBUF row."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIRowLoad(NKIOp):
    """Copy ``src(N)`` into an SBUF ``dst(1, N)`` row."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("N",), "dst": ("K", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 1, "N": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving row-load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM source vector."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKIRowLoad(src=<role={role}>) expects HBM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source vector with a leading unit dimension."""
        return np.asarray(kwargs["src"]).reshape(1, -1).copy()


__all__ = ["NKIRowLoad"]
