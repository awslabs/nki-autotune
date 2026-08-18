"""Grouped int32 conversion through ``nisa.activation``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


class NKIGroupedInt32Cast(NKIOp):
    """Copy packed groups into an int32 destination."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("G", "P", "F"), "dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("G", "P"), ("F",)) for slot in ("data", "dst")
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure packed groups and native copy conversion."""
        super().__init__(groups=groups, partitions=partitions, op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the value-preserving cast contract."""
        _ = kwargs
        return PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tile."""
        if _operand_role(kwargs["data"]) not in {None, "sbuf", "psum"}:
            raise TypeError("NKIGroupedInt32Cast.data expects SBUF or PSUM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return packed values represented as int32."""
        return np.asarray(kwargs["data"], dtype=np.int32)


__all__ = ["NKIGroupedInt32Cast"]
