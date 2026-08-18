"""Broadcasting transpose through ``nisa.nc_transpose``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


class NKITransposeBroadcast(NKIOp):
    """Broadcast a vector before transposing it into partition-major form."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("F",), "dst": ("P", "F")}
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"data": (("F",), ("P",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"P", "F"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions"})
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the copy-and-broadcast value contract."""
        _ = kwargs
        return PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")

    def __init__(self, partitions: int) -> None:
        """Configure the broadcasted partition extent."""
        if not 1 <= partitions <= 128:
            raise ValueError("transpose broadcast partitions must be between 1 and 128")
        super().__init__(partitions=partitions)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF vector."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKITransposeBroadcast(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the broadcasted transpose for CPU validation."""
        vector = np.asarray(kwargs["data"]).reshape(-1, 1)
        return np.tile(vector, (1, int(kwargs["partitions"]))).T


__all__ = ["NKITransposeBroadcast"]
